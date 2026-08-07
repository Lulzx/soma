//! Kernel-owned bounded resident synchronization bridge.
//!
//! This deliberately narrow G2 slice supports local futures, exact FIFO local
//! process mailboxes, exact live host-backed Object `Ref64` values, and governed
//! fixed 8-byte object range reads/in-place writes. Installed pointer-free
//! bytecode executes once on Metal; after final readback, validated journals are
//! replayed through ordinary governed Kernel operations on a clone and
//! published atomically.
//!
//! The admitted subset is intentionally strict: local unsupervised processes,
//! RunClassBins + RunPartial, no foreign payloads, no pre-existing waiter
//! queues, and no object growth/allocation. Canonical final future-await and
//! mailbox parks are supported. Competing mutable continuations use the exact
//! longest-waiting/identity admission rule and ordinary canonical deferral
//! replay; unsupported shapes refuse before submission.
//!
//! Exact invocation/applied-disposition, wake, and per-epoch records drive
//! normal Phase-G effects, trace causality, Phase-H accounting, and admission
//! history. CPU reference execution is test-only unless the explicit
//! `resident-sync-measurement` feature exposes the separate measurement API.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::abi::continuations::ContinuationState;
use crate::abi::{EventKind, FutureState, Kind, Ref64, Rights};
use crate::executives::resident_sync::{
    InitialFuture, ResidentCapability, ResidentEffect, ResidentHandlerProgram, ResidentInstruction,
    ResidentObject, ResidentObjectCapability, ResidentSyncConfig, ResidentSyncResult,
    HANDLER_ADD_FRAME_IMMEDIATE_U64, HANDLER_COMPLETE_IF_FRAME_U64_EQ, HANDLER_EFFECT_FUTURE_AWAIT,
    HANDLER_EFFECT_FUTURE_OBSERVE, HANDLER_EFFECT_FUTURE_RESOLVE, HANDLER_EFFECT_MAILBOX_RECEIVE,
    HANDLER_EFFECT_MAILBOX_SEND, HANDLER_EFFECT_OBJECT_READ, HANDLER_EFFECT_OBJECT_WRITE,
    RESOURCE_FUTURE, RESOURCE_MAILBOX, RESOURCE_OBJECT, RIGHT_READ, RIGHT_WRITE,
};
use crate::kernel::Kernel;
use crate::scheduler::device::reference_lane_conflicts;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelResidentSyncError {
    UnsupportedShape,
    InvalidProgram,
    StalePlan,
    BackendUnavailable,
    BackendFailed,
    InvalidDeviceResult,
    NotQuiescent,
    InvariantViolation,
}

/// Installed bytecode instruction. Effect targets are exact live `Ref64`s;
/// `argument` remains the frame offset/skip/run-class operand for non-effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelResidentInstruction {
    pub opcode: u32,
    pub argument: u32,
    pub target: Ref64,
    pub value: u64,
}

impl KernelResidentInstruction {
    pub fn effect(opcode: u32, target: Ref64, value: Ref64) -> Self {
        Self {
            opcode,
            argument: 0,
            target,
            value: value.to_u64(),
        }
    }
    pub fn object_read(target: Ref64, offset: u32) -> Self {
        Self {
            opcode: HANDLER_EFFECT_OBJECT_READ,
            argument: offset,
            target,
            value: 0,
        }
    }
    pub fn object_write(target: Ref64, offset: u32, value: u64) -> Self {
        Self {
            opcode: HANDLER_EFFECT_OBJECT_WRITE,
            argument: offset,
            target,
            value,
        }
    }
    pub fn add_frame_immediate(offset: u32, value: u64) -> Self {
        Self::plain(HANDLER_ADD_FRAME_IMMEDIATE_U64, offset, value)
    }
    pub fn complete_if_frame_eq(offset: u32, value: u64) -> Self {
        Self::plain(HANDLER_COMPLETE_IF_FRAME_U64_EQ, offset, value)
    }
    pub fn plain(opcode: u32, argument: u32, value: u64) -> Self {
        Self {
            opcode,
            argument,
            target: Ref64::NULL,
            value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelResidentProgram {
    pub run_class: u32,
    pub instructions: Vec<KernelResidentInstruction>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct KernelResidentSyncPlan {
    config: ResidentSyncConfig,
    continuations: Vec<crate::executives::resident_sync::ResidentContinuation>,
    programs: BTreeMap<u32, ResidentHandlerProgram>,
    continuations_by_id: BTreeMap<u64, Ref64>,
    actors_by_id: BTreeMap<u64, Ref64>,
    objects: Vec<Ref64>,
    futures: Vec<Ref64>,
    mailboxes: Vec<Ref64>,
    initial_epoch: u32,
    initial_pending: usize,
    fingerprint: [u8; 32],
}

impl Kernel {
    /// Install bounded resident bytecode. Installation does not schedule it.
    pub fn install_resident_sync_program(
        &mut self,
        program: KernelResidentProgram,
    ) -> Result<(), KernelResidentSyncError> {
        if program.run_class < 1024
            || program.instructions.is_empty()
            || program.instructions.len() > 256
        {
            return Err(KernelResidentSyncError::InvalidProgram);
        }
        self.resident_sync_programs
            .insert(program.run_class, program);
        Ok(())
    }

    /// Snapshot all currently runnable, resident-supported work into one fixed
    /// plan. This is read-only: every rejection leaves the kernel unchanged.
    pub fn plan_resident_sync(
        &self,
        max_epochs: u32,
        max_effects_per_step: u32,
        max_frame_bytes: u32,
        cohort_width: u32,
    ) -> Result<KernelResidentSyncPlan, KernelResidentSyncError> {
        KernelResidentSyncPlan::build(
            self,
            max_epochs,
            max_effects_per_step,
            max_frame_bytes,
            cohort_width,
        )
    }

    /// Submit the whole plan once and publish only after device quiescence and
    /// complete result validation. No CPU oracle participates in production.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn run_resident_sync_metal(
        &mut self,
        plan: KernelResidentSyncPlan,
    ) -> Result<u32, KernelResidentSyncError> {
        KernelResidentMetalExecutor::new()?.execute(self, plan)
    }

    /// Test-only independent CPU reference; it is intentionally impossible to
    /// select this path in production scheduling.
    #[cfg(test)]
    pub(crate) fn run_resident_sync_cpu_reference(
        &mut self,
        plan: KernelResidentSyncPlan,
    ) -> Result<u32, KernelResidentSyncError> {
        run_cpu_reference(self, plan)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn execute_metal(
    metal: &crate::executives::metal_resident_sync::MetalResidentSync,
    kernel: &mut Kernel,
    plan: KernelResidentSyncPlan,
) -> Result<u32, KernelResidentSyncError> {
    if !plan.matches(kernel) {
        return Err(KernelResidentSyncError::StalePlan);
    }
    let result = metal
        .run(&plan.config, plan.continuations.clone(), &plan.programs)
        .map_err(|_| KernelResidentSyncError::BackendFailed)?;
    plan.validate_and_import(kernel, result)
}

/// Reusable physical-Metal executor for canonical owned-kernel resident runs.
///
/// Device, queue, shader library, and pipeline setup happen once in [`Self::new`].
/// Each [`Self::execute`] performs the exact stale-plan check, physical backend
/// run, result validation, and atomic import used by [`Kernel::run_resident_sync_metal`].
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct KernelResidentMetalExecutor {
    metal: crate::executives::metal_resident_sync::MetalResidentSync,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl KernelResidentMetalExecutor {
    pub fn new() -> Result<Self, KernelResidentSyncError> {
        let metal = crate::executives::metal_resident_sync::MetalResidentSync::new()
            .map_err(|_| KernelResidentSyncError::BackendUnavailable)?;
        Ok(Self { metal })
    }

    pub fn execute(
        &self,
        kernel: &mut Kernel,
        plan: KernelResidentSyncPlan,
    ) -> Result<u32, KernelResidentSyncError> {
        execute_metal(&self.metal, kernel, plan)
    }
}

#[cfg(any(test, feature = "resident-sync-measurement"))]
fn run_cpu_reference(
    kernel: &mut Kernel,
    plan: KernelResidentSyncPlan,
) -> Result<u32, KernelResidentSyncError> {
    if !plan.matches(kernel) {
        return Err(KernelResidentSyncError::StalePlan);
    }
    let result = crate::executives::resident_sync::run_resident_sync(
        &plan.config,
        plan.continuations.clone(),
        &plan.programs,
    )
    .ok_or(KernelResidentSyncError::BackendFailed)?;
    plan.validate_and_import(kernel, result)
}

/// Explicitly opt-in CPU reference execution for measurement and comparison.
#[cfg(feature = "resident-sync-measurement")]
pub mod measurement {
    use super::{run_cpu_reference, KernelResidentSyncError, KernelResidentSyncPlan};
    use crate::kernel::Kernel;

    /// Execute an owned resident plan with the CPU reference and canonically import it.
    pub fn execute_cpu_reference(
        kernel: &mut Kernel,
        plan: KernelResidentSyncPlan,
    ) -> Result<u32, KernelResidentSyncError> {
        run_cpu_reference(kernel, plan)
    }

    /// Stable hash of the complete canonical kernel position.
    pub fn deterministic_state_hash(kernel: &Kernel) -> [u8; 32] {
        KernelResidentSyncPlan::fingerprint(kernel)
    }
}

#[allow(dead_code)]
impl KernelResidentSyncPlan {
    fn fingerprint(k: &Kernel) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        macro_rules! n {
            ($v:expr) => {
                h.update($v.to_le_bytes())
            };
        }
        macro_rules! r {
            ($v:expr) => {
                n!($v.to_u64())
            };
        }
        macro_rules! tag {
            ($v:literal) => {
                h.update($v);
            };
        }
        tag!(b"kernel-position-v1");
        n!(k.epoch);
        n!(k.logical_time);
        n!(k.current_lane);
        n!(k.lane_sequence);
        n!(k.host_sequence);
        n!(k.lane_effect_sequence);
        n!(k.host_effect_sequence);
        n!(k.applying_effects as u8);
        n!(k.lane_trace.len() as u64);
        n!(k.lane_effects.len() as u64);
        n!(k.retention as u8);
        n!(k.local_node);
        h.update(k.scheduler.canonical_fingerprint_bytes());
        let mut lost: Vec<_> = k.lost_nodes.iter().copied().collect();
        lost.sort_unstable();
        n!(lost.len() as u64);
        for node in lost {
            n!(node)
        }
        tag!(b"domain-table");
        n!(k.domains.len() as u64);
        for (reference, d) in k.domains.iter() {
            r!(reference);
            n!(d.header.abi_version);
            n!(d.header.structure_kind);
            n!(d.header.byte_length);
            r!(d.id);
            r!(d.parent);
            n!(d.max_processes);
            n!(d.processes_created);
        }
        tag!(b"contract-table");
        n!(k.contracts.len() as u64);
        for (reference, c) in k.contracts.iter() {
            r!(reference);
            n!(c.header.abi_version);
            n!(c.header.structure_kind);
            n!(c.header.byte_length);
            r!(c.id);
            n!(c.shape as u8);
            n!(c.placement_policy as u8);
            n!(c.precision_policy as u8);
            n!(c.determinism_policy as u8);
            n!(c.minimum_parallelism);
            n!(c.preferred_parallelism);
            n!(c.maximum_steps);
            n!(c.local_memory_bytes);
            n!(c.deadline_ns);
            n!(c.expected_read_bytes);
            n!(c.expected_write_bytes);
            n!(c.objective_flags);
            n!(c.contract_flags);
        }
        tag!(b"process-table");
        n!(k.processes.len() as u64);
        for (reference, p) in k.processes.iter() {
            r!(reference);
            n!(p.header.abi_version);
            n!(p.header.structure_kind);
            n!(p.header.byte_length);
            r!(p.id);
            n!(p.node_id);
            r!(p.domain);
            r!(p.supervisor);
            n!(p.supervision_policy as u8);
            r!(p.restart_of);
            n!(p.restart_attempt);
            n!(p.restart_limit);
            r!(p.state);
            r!(p.inbox);
            r!(p.urgent_inbox);
            r!(p.active_continuation);
            r!(p.waiting_on);
            n!(p.status);
            n!(p.process_mode as u8);
            n!(p.base_priority);
            n!(p.compute_quota);
            n!(p.memory_quota);
            n!(p.deadline_ns);
            n!(p.last_committed_epoch);
            n!(p.failure_count);
            n!(p.live_continuations);
        }
        tag!(b"continuation-table");
        n!(k.continuations.len() as u64);
        for (reference, c) in k.continuations.iter() {
            r!(reference);
            n!(c.header.abi_version);
            n!(c.header.structure_kind);
            n!(c.header.byte_length);
            r!(c.id);
            r!(c.process);
            n!(c.run_class);
            n!(c.resume_point);
            r!(c.execution_contract);
            r!(c.frame);
            r!(c.dependency);
            n!(c.deadline_ns);
            n!(c.remaining_steps);
            n!(c.priority);
            n!(c.status as u8);
            n!(c.state_access as u8);
            n!(c.created_epoch);
            n!(c.last_run_epoch);
        }
        tag!(b"object-table");
        n!(k.objects.len() as u64);
        for (reference, o) in k.objects.iter() {
            r!(reference);
            n!(o.header.abi_version);
            n!(o.header.structure_kind);
            n!(o.header.byte_length);
            r!(o.id);
            r!(o.owner_domain);
            n!(o.byte_length);
            n!(o.physical_mapping_token);
            n!(o.version);
            n!(o.object_kind as u8);
            n!(o.flags);
            let payload = k.object_payloads.get(&reference.key());
            n!(payload.map_or(u64::MAX, |p| p.len() as u64));
            if let Some(payload) = payload {
                n!(payload.provenance().len() as u64);
                h.update(payload.provenance().as_bytes());
                h.update(payload.as_slice());
            }
        }
        tag!(b"future-table");
        n!(k.futures.len() as u64);
        for (reference, f) in k.futures.iter() {
            r!(reference);
            n!(f.header.abi_version);
            n!(f.header.structure_kind);
            n!(f.header.byte_length);
            r!(f.id);
            r!(f.owner_process);
            n!(f.state as u8);
            n!(f.waiter_count);
            r!(f.value);
            r!(f.failure);
            r!(f.waiter_list);
            n!(f.resolved_epoch);
            n!(f.flags);
        }
        let mut actors: Vec<_> = k.capability_spaces.iter().collect();
        actors.sort_by_key(|(actor, _)| **actor);
        n!(actors.len() as u64);
        for (actor, space) in actors {
            n!(*actor);
            n!(space.len() as u64);
            for (reference, c) in space.iter() {
                r!(reference);
                n!(c.header.abi_version);
                n!(c.header.structure_kind);
                n!(c.header.byte_length);
                r!(c.target);
                n!(c.offset);
                n!(c.length);
                n!(c.rights);
                n!(c.transfer_policy);
                n!(c.object_version);
                n!(c.valid_until_epoch);
                r!(c.parent_capability);
            }
        }
        let mut mailboxes: Vec<_> = k.mailboxes.iter().collect();
        mailboxes.sort_by_key(|(key, _)| **key);
        n!(mailboxes.len() as u64);
        for (key, m) in mailboxes {
            n!(*key);
            n!(m.capacity as u64);
            n!(m.entries.len() as u64);
            n!(m.recv_waiters.len() as u64);
            n!(m.full_waiters.len() as u64);
            for msg in &m.entries {
                r!(msg.sender);
                r!(msg.receiver);
                r!(msg.payload);
                r!(msg.transferred_capability);
                n!(msg.sender_sequence);
            }
            for waiter in &m.recv_waiters {
                r!(*waiter)
            }
            for waiter in &m.full_waiters {
                r!(*waiter)
            }
        }
        let mut sequences: Vec<_> = k.send_sequences.iter().collect();
        sequences.sort_by_key(|(key, _)| **key);
        n!(sequences.len() as u64);
        for ((sender, receiver), value) in sequences {
            n!(*sender);
            n!(*receiver);
            n!(*value)
        }
        tag!(b"resident-programs");
        n!(k.resident_sync_programs.len() as u64);
        for (rc, p) in &k.resident_sync_programs {
            tag!(b"program");
            n!(*rc);
            n!(p.instructions.len() as u64);
            for i in &p.instructions {
                n!(i.opcode);
                n!(i.argument);
                r!(i.target);
                n!(i.value)
            }
        }
        tag!(b"trace-log");
        n!(k.trace.len() as u64);
        for e in &k.trace {
            n!(e.logical_time);
            n!(e.epoch);
            n!(e.event_kind as u16);
            n!(e.engine);
            n!(e.lane);
            n!(e.lane_sequence);
            r!(e.process);
            r!(e.continuation);
            n!(e.run_class);
            n!(e.auxiliary);
            r!(e.subject);
            r!(e.causal);
        }
        tag!(b"trace-counters");
        n!(k.trace_counters.emitted);
        n!(k.trace_counters.taken);
        n!(k.trace_counters.dropped);
        tag!(b"effect-log");
        n!(k.effect_log.len() as u64);
        for e in &k.effect_log {
            n!(e.epoch);
            n!(e.lane);
            n!(e.sequence);
            n!(e.applied);
            n!(e.kind as u8);
            r!(e.continuation);
            n!(e.run_class);
        }
        n!(k.effect_counters.emitted);
        n!(k.effect_counters.taken);
        n!(k.effect_counters.dropped);
        tag!(b"admission-log");
        n!(k.admission_log.len() as u64);
        for record in &k.admission_log {
            n!(record.candidates.len() as u64);
            for c in &record.candidates {
                n!(c.bin);
                r!(c.continuation);
                r!(c.process);
                n!(c.run_class);
                n!(c.state_access as u8);
                n!(c.waiting_since);
            }
            let bins = record.decision.bins();
            let deferred = record.decision.deferred();
            n!(bins.len() as u64);
            for (bin, entries) in bins {
                n!(bin);
                n!(entries.len() as u64);
                for (reference, rc) in entries {
                    r!(reference);
                    n!(rc)
                }
            }
            n!(deferred.len() as u64);
            for (rc, reference) in deferred {
                r!(*reference);
                n!(*rc)
            }
        }
        n!(k.admission_counters.emitted);
        n!(k.admission_counters.taken);
        n!(k.admission_counters.dropped);
        tag!(b"phase-h");
        n!(k.epoch_runnable.len() as u64);
        for value in &k.epoch_runnable {
            n!(*value as u64)
        }
        n!(k.accounting.epochs);
        n!(k.accounting.steps);
        n!(k.accounting.cohorts);
        n!(k.accounting.full_cohorts);
        n!(k.accounting.lane_slots);
        n!(k.accounting.useful_lane_slots);
        n!(k.accounting.idle_lane_slots);
        n!(k.accounting.deferred_lanes);
        n!(k.accounting.serial_deferrals);
        n!(k.accounting.stalled_epochs);
        h.finalize().into()
    }

    fn build(
        kernel: &Kernel,
        max_epochs: u32,
        max_effects: u32,
        max_frame: u32,
        width: u32,
    ) -> Result<Self, KernelResidentSyncError> {
        if kernel.current_lane != crate::abi::traces::HOST_LANE
            || !kernel.lane_trace.is_empty()
            || !kernel.lane_effects.is_empty()
            || kernel.applying_effects
            || kernel.speculation_journal.is_some()
        {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
        if max_epochs == 0 || max_effects == 0 || max_frame == 0 || width == 0 {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
        let horizon = kernel
            .epoch
            .checked_add(max_epochs)
            .ok_or(KernelResidentSyncError::UnsupportedShape)?;
        // This slice begins at a clean resident boundary. Existing FIFO
        // mailbox entries are snapshotted exactly; pre-existing waiter queues
        // remain unsupported because their pending handler state is not part of
        // this admission shape.
        if kernel.future_waiters.values().any(|w| !w.is_empty())
            || kernel
                .mailboxes
                .values()
                .any(|m| !m.recv_waiters.is_empty() || !m.full_waiters.is_empty())
            || !kernel.remote_waiter_dependencies.is_empty()
        {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }

        let mut futures: Vec<Ref64> = kernel.futures.iter().map(|(r, _)| r).collect();
        futures.sort_by_key(Ref64::to_u64);
        let mut mailboxes: Vec<Ref64> = kernel
            .processes
            .iter()
            .filter_map(|(r, p)| {
                (p.node_id == kernel.local_node && kernel.mailboxes.contains_key(&r.key()))
                    .then_some(r)
            })
            .collect();
        mailboxes.sort_by_key(Ref64::to_u64);
        if kernel
            .processes
            .iter()
            .any(|(_, p)| p.node_id != kernel.local_node)
            || mailboxes.iter().any(|process| {
                kernel.mailboxes[&process.key()]
                    .entries
                    .iter()
                    .any(|message| {
                        message.receiver != *process
                            || kernel.processes.get(message.sender).is_err()
                            || kernel.objects.get(message.payload).is_err()
                            || kernel
                                .object_payloads
                                .get(&message.payload.key())
                                .is_none_or(|payload| payload.provenance() != "host")
                    })
            })
        {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
        let mut objects: Vec<Ref64> = kernel
            .resident_sync_programs
            .values()
            .flat_map(|program| program.instructions.iter())
            .filter(|instruction| {
                matches!(
                    instruction.opcode,
                    HANDLER_EFFECT_OBJECT_READ | HANDLER_EFFECT_OBJECT_WRITE
                )
            })
            .map(|instruction| instruction.target)
            .collect();
        objects.sort_by_key(Ref64::to_u64);
        objects.dedup();
        if objects.len() > crate::executives::resident_sync::MAX_RESIDENT_OBJECTS
            || objects
                .len()
                .checked_mul(max_frame as usize)
                .is_none_or(|bytes| {
                    bytes > crate::executives::resident_sync::MAX_RESIDENT_OBJECT_ARENA_BYTES
                })
        {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
        if objects.iter().any(|object| {
            object.kind != Kind::Object
                || kernel.objects.get(*object).is_err()
                || kernel.object_payloads.get(&object.key()).is_none_or(|p| {
                    let descriptor = kernel.objects.get(*object).ok();
                    p.provenance() != "host"
                        || p.len() > max_frame as usize
                        || descriptor.is_none_or(|d| p.len() as u64 != d.byte_length)
                })
        }) {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
        let object_index: HashMap<Ref64, u32> = objects
            .iter()
            .enumerate()
            .map(|(i, r)| (*r, i as u32))
            .collect();

        let future_index: HashMap<Ref64, u32> = futures
            .iter()
            .enumerate()
            .map(|(i, r)| (*r, i as u32))
            .collect();
        let mailbox_index: HashMap<Ref64, u32> = mailboxes
            .iter()
            .enumerate()
            .map(|(i, r)| (*r, i as u32))
            .collect();

        let mut packed_programs = BTreeMap::new();
        for (&rc, program) in &kernel.resident_sync_programs {
            let mut packed = Vec::with_capacity(program.instructions.len());
            for instruction in &program.instructions {
                let effect = matches!(
                    instruction.opcode,
                    HANDLER_EFFECT_OBJECT_READ
                        | HANDLER_EFFECT_OBJECT_WRITE
                        | HANDLER_EFFECT_FUTURE_OBSERVE
                        | HANDLER_EFFECT_FUTURE_AWAIT
                        | HANDLER_EFFECT_FUTURE_RESOLVE
                        | HANDLER_EFFECT_MAILBOX_SEND
                        | HANDLER_EFFECT_MAILBOX_RECEIVE
                );
                let argument = match instruction.opcode {
                    HANDLER_EFFECT_OBJECT_READ | HANDLER_EFFECT_OBJECT_WRITE => {
                        instruction.argument
                    }
                    HANDLER_EFFECT_FUTURE_OBSERVE
                    | HANDLER_EFFECT_FUTURE_AWAIT
                    | HANDLER_EFFECT_FUTURE_RESOLVE => *future_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    HANDLER_EFFECT_MAILBOX_SEND | HANDLER_EFFECT_MAILBOX_RECEIVE => *mailbox_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    _ => instruction.argument,
                };
                let packed_target = match instruction.opcode {
                    HANDLER_EFFECT_OBJECT_READ | HANDLER_EFFECT_OBJECT_WRITE => *object_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    HANDLER_EFFECT_FUTURE_OBSERVE
                    | HANDLER_EFFECT_FUTURE_AWAIT
                    | HANDLER_EFFECT_FUTURE_RESOLVE => *future_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    HANDLER_EFFECT_MAILBOX_SEND | HANDLER_EFFECT_MAILBOX_RECEIVE => *mailbox_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    _ => 0,
                };
                if effect && instruction.target.is_null() {
                    return Err(KernelResidentSyncError::InvalidProgram);
                }
                // The bounded value form is an exact, already-live local object.
                // No device allocation and no foreign payload may enter the plan.
                if matches!(
                    instruction.opcode,
                    HANDLER_EFFECT_FUTURE_RESOLVE | HANDLER_EFFECT_MAILBOX_SEND
                ) {
                    let value = Ref64::from_u64(instruction.value);
                    if value.kind != Kind::Object
                        || kernel.objects.get(value).is_err()
                        || kernel
                            .object_payloads
                            .get(&value.key())
                            .is_none_or(|p| p.provenance() != "host")
                    {
                        return Err(KernelResidentSyncError::UnsupportedShape);
                    }
                }
                packed.push(ResidentInstruction {
                    opcode: instruction.opcode,
                    argument,
                    target: packed_target,
                    reserved: 0,
                    value: if matches!(
                        instruction.opcode,
                        HANDLER_EFFECT_OBJECT_READ
                            | HANDLER_EFFECT_FUTURE_OBSERVE
                            | HANDLER_EFFECT_FUTURE_AWAIT
                            | HANDLER_EFFECT_MAILBOX_RECEIVE
                    ) {
                        0
                    } else {
                        instruction.value
                    },
                });
            }
            packed_programs.insert(
                rc,
                ResidentHandlerProgram {
                    run_class: rc,
                    instructions: packed,
                },
            );
        }

        let mut continuations = Vec::new();
        let mut continuations_by_id = BTreeMap::new();
        let mut actors_by_id = BTreeMap::new();
        for (r, c) in kernel.continuations.iter() {
            if c.status != ContinuationState::Runnable {
                continue;
            }
            if !packed_programs.contains_key(&c.run_class) || c.process.kind != Kind::Process {
                return Err(KernelResidentSyncError::UnsupportedShape);
            }
            let process = kernel
                .processes
                .get(c.process)
                .map_err(|_| KernelResidentSyncError::UnsupportedShape)?;
            if process.node_id != kernel.local_node
                || !process.supervisor.is_null()
                || !process.restart_of.is_null()
                || c.frame.kind != Kind::Object
            {
                return Err(KernelResidentSyncError::UnsupportedShape);
            }
            let frame = kernel
                .object_payloads
                .get(&c.frame.key())
                .ok_or(KernelResidentSyncError::UnsupportedShape)?
                .as_slice()
                .to_vec();
            if frame.len() > max_frame as usize {
                return Err(KernelResidentSyncError::UnsupportedShape);
            }
            let id = r.to_u64();
            continuations.push(crate::executives::resident_sync::ResidentContinuation {
                id,
                actor: c.process.to_u64(),
                run_class: c.run_class,
                mutable_access: c.state_access == crate::abi::StateAccess::Mutable,
                waiting_since: c.last_run_epoch.max(c.created_epoch),
                frame,
            });
            continuations_by_id.insert(id, r);
            actors_by_id.insert(id, c.process);
        }
        if kernel.scheduler.mode() != crate::scheduler::runnable_bins::SchedulingMode::RunClassBins
            || kernel.partial_policy != crate::abi::PartialCohortPolicy::RunPartial
        {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
        if continuations.is_empty() || kernel.scheduler.total_pending() != continuations.len() {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }

        let mut capabilities = Vec::new();
        for actor in actors_by_id.values().copied() {
            let Some(space) = kernel.capability_spaces.get(&actor.key()) else {
                return Err(KernelResidentSyncError::UnsupportedShape);
            };
            for (_, cap) in space.iter().filter(|(reference, c)| {
                c.valid_until_epoch >= horizon
                    && Kernel::capability_chain_is_live(space, *reference)
            }) {
                if let Some(&target) = future_index.get(&cap.target) {
                    let mut rights = 0;
                    if cap.rights & Rights::AWAIT != 0 {
                        rights |= RIGHT_READ
                    };
                    if cap.rights & Rights::RESOLVE != 0 {
                        rights |= RIGHT_WRITE
                    };
                    if rights & RIGHT_READ != 0
                        && kernel
                            .find_authorized_capability(actor, Rights::AWAIT, cap.target)
                            .is_none()
                    {
                        rights &= !RIGHT_READ;
                    }
                    if rights & RIGHT_WRITE != 0
                        && kernel
                            .find_authorized_capability(actor, Rights::RESOLVE, cap.target)
                            .is_none()
                    {
                        rights &= !RIGHT_WRITE;
                    }
                    if rights != 0 {
                        capabilities.push(ResidentCapability {
                            actor: actor.to_u64(),
                            resource_kind: RESOURCE_FUTURE,
                            target,
                            rights,
                        });
                    }
                }
                if let Some(&target) = mailbox_index.get(&cap.target) {
                    let mut rights = 0;
                    if cap.rights & Rights::RECEIVE != 0 {
                        rights |= RIGHT_READ
                    };
                    if cap.rights & Rights::SEND != 0 {
                        rights |= RIGHT_WRITE
                    };
                    if rights & RIGHT_READ != 0
                        && kernel
                            .find_authorized_capability(actor, Rights::RECEIVE, cap.target)
                            .is_none()
                    {
                        rights &= !RIGHT_READ;
                    }
                    if rights & RIGHT_WRITE != 0
                        && kernel
                            .find_authorized_capability(actor, Rights::SEND, cap.target)
                            .is_none()
                    {
                        rights &= !RIGHT_WRITE;
                    }
                    if rights != 0 {
                        capabilities.push(ResidentCapability {
                            actor: actor.to_u64(),
                            resource_kind: RESOURCE_MAILBOX,
                            target,
                            rights,
                        });
                    }
                }
            }
        }
        let mut object_capabilities = Vec::new();
        for actor in actors_by_id.values() {
            let space = kernel
                .capability_spaces
                .get(&actor.key())
                .ok_or(KernelResidentSyncError::UnsupportedShape)?;
            for (reference, cap) in space.iter() {
                let Some(&target) = object_index.get(&cap.target) else {
                    continue;
                };
                let object = kernel
                    .objects
                    .get(cap.target)
                    .map_err(|_| KernelResidentSyncError::UnsupportedShape)?;
                let rights = cap.rights & (Rights::READ | Rights::WRITE);
                if rights != 0
                    && cap.valid_until_epoch >= horizon
                    && cap.object_version == object.version
                    && Kernel::capability_chain_is_live(space, reference)
                {
                    if object_capabilities.len()
                        >= crate::executives::resident_sync::MAX_RESIDENT_OBJECT_CAPABILITIES
                    {
                        return Err(KernelResidentSyncError::UnsupportedShape);
                    }
                    object_capabilities.push(ResidentObjectCapability {
                        actor: actor.to_u64(),
                        target,
                        offset: cap.offset,
                        length: cap.length,
                        rights: (((rights & Rights::READ != 0) as u32) * RIGHT_READ)
                            | (((rights & Rights::WRITE != 0) as u32) * RIGHT_WRITE),
                        object_version: cap.object_version,
                        valid_until_epoch: cap.valid_until_epoch,
                    });
                }
            }
        }
        // Reject statically unauthorized effect shapes before submit.
        for c in &continuations {
            let p = &packed_programs[&c.run_class];
            for i in &p.instructions {
                let (kind, right) = match i.opcode {
                    HANDLER_EFFECT_OBJECT_READ => (RESOURCE_OBJECT, RIGHT_READ),
                    HANDLER_EFFECT_OBJECT_WRITE => (RESOURCE_OBJECT, RIGHT_WRITE),
                    HANDLER_EFFECT_FUTURE_OBSERVE | HANDLER_EFFECT_FUTURE_AWAIT => {
                        (RESOURCE_FUTURE, RIGHT_READ)
                    }
                    HANDLER_EFFECT_FUTURE_RESOLVE => (RESOURCE_FUTURE, RIGHT_WRITE),
                    HANDLER_EFFECT_MAILBOX_RECEIVE => (RESOURCE_MAILBOX, RIGHT_READ),
                    HANDLER_EFFECT_MAILBOX_SEND => (RESOURCE_MAILBOX, RIGHT_WRITE),
                    _ => continue,
                };
                let authorized = if kind == RESOURCE_OBJECT {
                    let object = kernel
                        .objects
                        .get(objects[i.target as usize])
                        .map_err(|_| KernelResidentSyncError::UnsupportedShape)?;
                    let end = u64::from(i.argument)
                        .checked_add(8)
                        .ok_or(KernelResidentSyncError::UnsupportedShape)?;
                    end <= object.byte_length
                        && object_capabilities.iter().any(|x| {
                            x.actor == c.actor && x.target == i.target && (x.rights & right) != 0
                            && u64::from(i.argument) >= x.offset
                            && end <= x.offset.saturating_add(x.length)
                            // Ordinary Kernel object methods currently govern full-object access.
                            && x.offset == 0 && x.length >= object.byte_length
                        })
                } else {
                    capabilities.iter().any(|x| {
                        x.actor == c.actor
                            && x.resource_kind == kind
                            && x.target == i.target
                            && (x.rights & right) != 0
                    })
                };
                if !authorized {
                    return Err(KernelResidentSyncError::UnsupportedShape);
                }
                if i.opcode == HANDLER_EFFECT_MAILBOX_SEND {
                    let value = Ref64::from_u64(i.value);
                    let actor = Ref64::from_u64(c.actor);
                    let object = kernel
                        .objects
                        .get(value)
                        .map_err(|_| KernelResidentSyncError::UnsupportedShape)?;
                    let space = kernel
                        .capability_spaces
                        .get(&actor.key())
                        .ok_or(KernelResidentSyncError::UnsupportedShape)?;
                    let transfer_is_stable =
                        space.for_target(value).into_iter().any(|(reference, cap)| {
                            cap.target == value
                                && cap.rights & Rights::TRANSFER != 0
                                && cap.valid_until_epoch >= horizon
                                && cap.object_version == object.version
                                && cap.offset == 0
                                && cap.length >= object.byte_length
                                && Kernel::capability_chain_is_live(space, reference)
                        });
                    if !transfer_is_stable {
                        return Err(KernelResidentSyncError::UnsupportedShape);
                    }
                }
            }
        }
        if packed_programs.values().any(|program| {
            !crate::executives::resident_sync::validate_handler_program(
                program,
                max_effects,
                max_frame,
            )
        }) {
            return Err(KernelResidentSyncError::InvalidProgram);
        }
        let initial_futures = futures
            .iter()
            .map(|r| {
                let f = kernel.futures.get(*r).unwrap();
                match f.state {
                    FutureState::Pending => Ok(InitialFuture::Pending),
                    FutureState::Resolved
                        if f.value.kind == Kind::Object
                            && kernel.objects.get(f.value).is_ok()
                            && kernel
                                .object_payloads
                                .get(&f.value.key())
                                .is_some_and(|p| p.provenance() == "host") =>
                    {
                        Ok(InitialFuture::Resolved(f.value.to_u64()))
                    }
                    _ => Err(KernelResidentSyncError::UnsupportedShape),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let capacities: Vec<u32> = mailboxes
            .iter()
            .map(|process| {
                u32::try_from(kernel.mailboxes[&process.key()].capacity)
                    .map_err(|_| KernelResidentSyncError::UnsupportedShape)
            })
            .collect::<Result<_, _>>()?;
        let mailbox_messages: Vec<Vec<(u64, u64)>> = mailboxes
            .iter()
            .map(|process| {
                kernel.mailboxes[&process.key()]
                    .entries
                    .iter()
                    .map(|message| (message.sender.to_u64(), message.payload.to_u64()))
                    .collect()
            })
            .collect();
        Ok(Self {
            config: ResidentSyncConfig {
                max_epochs,
                max_effects_per_step: max_effects,
                max_frame_bytes: max_frame,
                max_continuations: (continuations.len() as u32)
                    .max(capacities.iter().copied().max().unwrap_or(0)),
                cohort_width: width,
                initial_epoch: kernel.epoch,
                objects: objects
                    .iter()
                    .map(|reference| {
                        let descriptor = kernel.objects.get(*reference).expect("validated object");
                        ResidentObject {
                            version: descriptor.version,
                            bytes: kernel.object_payloads[&reference.key()].as_slice().to_vec(),
                        }
                    })
                    .collect(),
                object_capabilities,
                futures: initial_futures,
                mailbox_capacities: capacities,
                mailbox_messages,
                capabilities,
            },
            continuations,
            programs: packed_programs,
            continuations_by_id,
            actors_by_id,
            objects,
            futures,
            mailboxes,
            initial_epoch: kernel.epoch,
            initial_pending: kernel.scheduler.total_pending(),
            fingerprint: Self::fingerprint(kernel),
        })
    }

    fn matches(&self, k: &Kernel) -> bool {
        k.epoch == self.initial_epoch
            && k.scheduler.total_pending() == self.initial_pending
            && Self::fingerprint(k) == self.fingerprint
            && self
                .continuations_by_id
                .iter()
                .all(|(_, r)| k.continuations.get(*r).is_ok())
    }

    fn translate_target(&self, e: ResidentEffect) -> Option<Ref64> {
        match e {
            ResidentEffect::ObjectRead { target, .. }
            | ResidentEffect::ObjectWrite { target, .. } => {
                self.objects.get(target as usize).copied()
            }
            ResidentEffect::FutureObserve { target }
            | ResidentEffect::FutureAwait { target }
            | ResidentEffect::FutureResolve { target, .. } => {
                self.futures.get(target as usize).copied()
            }
            ResidentEffect::MailboxSend { target, .. }
            | ResidentEffect::MailboxReceive { target } => {
                self.mailboxes.get(target as usize).copied()
            }
        }
    }

    fn validate_and_import(
        self,
        kernel: &mut Kernel,
        result: ResidentSyncResult,
    ) -> Result<u32, KernelResidentSyncError> {
        if !self.matches(kernel) {
            return Err(KernelResidentSyncError::StalePlan);
        }
        if !result.quiescent {
            return Err(KernelResidentSyncError::NotQuiescent);
        }
        if result.frames.len() != self.continuations.len()
            || result.final_continuations.len() != self.continuations.len()
            || result.object_values.len() != self.objects.len()
            || result.future_values.len() != self.futures.len()
            || result.mailboxes.len() != self.mailboxes.len()
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        // A quiescent resident graph may contain a strictly canonical parked
        // future await. Mailbox parking remains outside this publication slice:
        // unlike a future waiter, its FIFO identity also depends on mailbox
        // capacity and sender/receiver queue ordering.
        let all_ids: BTreeSet<_> = self.continuations_by_id.keys().copied().collect();
        let final_ids: BTreeSet<_> = result
            .final_continuations
            .iter()
            .map(|continuation| continuation.id)
            .collect();
        if final_ids.len() != result.final_continuations.len() || final_ids != all_ids {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        let completed_ids: BTreeSet<_> = result.completed.iter().copied().collect();
        let final_completed_ids: BTreeSet<_> = result
            .final_continuations
            .iter()
            .filter(|continuation| continuation.completed)
            .map(|continuation| continuation.id)
            .collect();
        if completed_ids.len() != result.completed.len()
            || completed_ids != final_completed_ids
            || result.final_continuations.iter().any(|continuation| {
                continuation.completed != continuation.pending.is_none()
                    || (!continuation.completed
                        && !matches!(
                            continuation.pending,
                            Some(
                                ResidentEffect::FutureAwait { .. }
                                    | ResidentEffect::MailboxSend { .. }
                                    | ResidentEffect::MailboxReceive { .. }
                            )
                        ))
                    || (continuation.completed && continuation.waiter_order != 0)
            })
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }

        // Registration tickets are assigned to every blocking operation in
        // canonical execution order, including waiters subsequently woken.
        // Matching the ticket prevents a forged final waiter from being
        // substituted for a different invocation of the same continuation.
        let mut blocking = result
            .effects
            .iter()
            .filter(|effect| {
                matches!(
                    effect.outcome,
                    crate::executives::resident_sync::ResidentOutcome::Registered
                        | crate::executives::resident_sync::ResidentOutcome::Full
                        | crate::executives::resident_sync::ResidentOutcome::Empty
                )
            })
            .collect::<Vec<_>>();
        blocking.sort_by_key(|effect| (effect.epoch, effect.lane, effect.ordinal));
        for final_continuation in result
            .final_continuations
            .iter()
            .filter(|continuation| !continuation.completed)
        {
            let pending = final_continuation
                .pending
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            let invocation = result
                .invocations
                .iter()
                .filter(|invocation| invocation.continuation == final_continuation.id)
                .max_by_key(|invocation| (invocation.epoch, invocation.lane))
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            let expected_outcome = match pending {
                ResidentEffect::FutureAwait { .. } => {
                    crate::executives::resident_sync::ResidentOutcome::Registered
                }
                ResidentEffect::MailboxSend { .. } => {
                    crate::executives::resident_sync::ResidentOutcome::Full
                }
                ResidentEffect::MailboxReceive { .. } => {
                    crate::executives::resident_sync::ResidentOutcome::Empty
                }
                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
            };
            let matching = result
                .effects
                .iter()
                .filter(|effect| {
                    effect.epoch == invocation.epoch
                        && effect.lane == invocation.lane
                        && effect.continuation == final_continuation.id
                        && effect.effect == pending
                        && effect.outcome == expected_outcome
                })
                .collect::<Vec<_>>();
            let last_ordinal = result
                .effects
                .iter()
                .filter(|effect| effect.epoch == invocation.epoch && effect.lane == invocation.lane)
                .map(|effect| effect.ordinal)
                .max();
            if invocation.disposition != 3
                || invocation.next_run_class != final_continuation.run_class
                || matching.len() != 1
                || last_ordinal != Some(matching[0].ordinal)
                || final_continuation.waiter_order == 0
                || blocking
                    .iter()
                    .position(|effect| std::ptr::eq(*effect, matching[0]))
                    .map(|position| position as u32 + 1)
                    != Some(final_continuation.waiter_order)
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        let mut invocation_positions = BTreeSet::new();
        let mut invocation_entities = BTreeSet::new();
        let mut current_run_classes: BTreeMap<_, _> = self
            .continuations
            .iter()
            .map(|continuation| (continuation.id, continuation.run_class))
            .collect();
        let mut completed = BTreeSet::new();
        let mut invocation_offset = 0usize;
        for epoch in 0..result.epochs {
            let epoch_start = invocation_offset;
            while invocation_offset < result.invocations.len()
                && result.invocations[invocation_offset].epoch == epoch
            {
                invocation_offset += 1;
            }
            let epoch_invocations = &result.invocations[epoch_start..invocation_offset];
            if epoch_invocations.is_empty() {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            let mut previous_key = None;
            for (lane, invocation) in epoch_invocations.iter().enumerate() {
                let key = (invocation.run_class, invocation.continuation);
                if invocation.lane as usize != lane
                    || previous_key.is_some_and(|previous| previous >= key)
                    || current_run_classes.get(&invocation.continuation)
                        != Some(&invocation.run_class)
                    || completed.contains(&invocation.continuation)
                    || invocation.disposition == 0
                    || invocation.disposition > 3
                    || !self.programs.contains_key(&invocation.run_class)
                    || !invocation_positions.insert((invocation.epoch, invocation.lane))
                    || !invocation_entities.insert((invocation.epoch, invocation.continuation))
                {
                    return Err(KernelResidentSyncError::InvalidDeviceResult);
                }
                match invocation.disposition {
                    1 => {
                        if !self.programs.contains_key(&invocation.next_run_class) {
                            return Err(KernelResidentSyncError::InvalidDeviceResult);
                        }
                        current_run_classes
                            .insert(invocation.continuation, invocation.next_run_class);
                    }
                    2 => {
                        if invocation.next_run_class != 0 {
                            return Err(KernelResidentSyncError::InvalidDeviceResult);
                        }
                        completed.insert(invocation.continuation);
                    }
                    3 => {
                        if invocation.next_run_class != invocation.run_class {
                            return Err(KernelResidentSyncError::InvalidDeviceResult);
                        }
                    }
                    _ => unreachable!(),
                }
                previous_key = Some(key);
            }
        }
        if invocation_offset != result.invocations.len() {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        let mut effect_ordinals: BTreeMap<(u32, u32, u64), Vec<u32>> = BTreeMap::new();
        for effect in &result.effects {
            if !invocation_positions.contains(&(effect.epoch, effect.lane))
                || !invocation_entities.contains(&(effect.epoch, effect.continuation))
                || !result.invocations.iter().any(|invocation| {
                    invocation.epoch == effect.epoch
                        && invocation.lane == effect.lane
                        && invocation.continuation == effect.continuation
                })
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            effect_ordinals
                .entry((effect.epoch, effect.lane, effect.continuation))
                .or_default()
                .push(effect.ordinal);
        }
        if effect_ordinals.values_mut().any(|ordinals| {
            ordinals.sort_unstable();
            ordinals
                .iter()
                .enumerate()
                .any(|(expected, ordinal)| *ordinal as usize != expected)
        }) {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        let mut invocations: BTreeMap<u64, u32> = BTreeMap::new();
        for invocation in &result.invocations {
            let count = invocations.entry(invocation.continuation).or_default();
            *count = count.saturating_add(1);
        }
        for (&id, &reference) in &self.continuations_by_id {
            let used = invocations.get(&id).copied().unwrap_or(0);
            let remaining = kernel
                .continuations
                .get(reference)
                .map_err(|_| KernelResidentSyncError::StalePlan)?
                .remaining_steps;
            if used == 0 || used > remaining {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        if result.epoch_records.len() != result.epochs as usize
            || result
                .epoch_records
                .iter()
                .enumerate()
                .any(|(epoch, record)| {
                    record.epoch as usize != epoch
                        || record.invocations as usize
                            != result
                                .invocations
                                .iter()
                                .filter(|invocation| invocation.epoch as usize == epoch)
                                .count()
                })
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        let mut wake_positions = BTreeSet::new();
        for wake in &result.wakes {
            if wake.reserved != 0
                || wake.epoch >= result.epochs
                || !wake_positions.insert((wake.epoch, wake.lane, wake.ordinal, wake.continuation))
                || !self.continuations_by_id.contains_key(&wake.continuation)
                || !result.effects.iter().any(|effect| {
                    effect.epoch == wake.epoch
                        && effect.lane == wake.lane
                        && effect.ordinal == wake.ordinal
                        && effect.continuation == wake.cause_continuation
                        && match effect.effect {
                            ResidentEffect::FutureResolve { target, .. } => {
                                wake.cause_opcode == crate::scheduler::device_ops::OP_RESOLVE_FUTURE
                                    && target == wake.target
                            }
                            ResidentEffect::MailboxSend { target, .. } => {
                                wake.cause_opcode
                                    == crate::scheduler::device_ops::OP_ENQUEUE_MESSAGE
                                    && target == wake.target
                            }
                            ResidentEffect::MailboxReceive { target } => {
                                wake.cause_opcode
                                    == crate::scheduler::device_ops::OP_RECEIVE_MESSAGE
                                    && target == wake.target
                            }
                            _ => false,
                        }
                })
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        for effect in &result.effects {
            let cont = *self
                .continuations_by_id
                .get(&effect.continuation)
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            let actor = self.actors_by_id[&effect.continuation];
            let target = self
                .translate_target(effect.effect)
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            let _ = (cont, actor, target);
            if matches!(
                effect.outcome,
                crate::executives::resident_sync::ResidentOutcome::CapabilityDenied
                    | crate::executives::resident_sync::ResidentOutcome::InvalidTarget
            ) {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            match effect.effect {
                ResidentEffect::FutureResolve { value, .. }
                | ResidentEffect::MailboxSend { value, .. } => {
                    let value = Ref64::from_u64(value);
                    if value.kind != Kind::Object
                        || kernel.objects.get(value).is_err()
                        || kernel
                            .object_payloads
                            .get(&value.key())
                            .is_none_or(|p| p.provenance() != "host")
                    {
                        return Err(KernelResidentSyncError::InvalidDeviceResult);
                    }
                }
                _ => {}
            }
        }
        let effect_count = result.effects.len();
        if result.accesses.len() != effect_count
            || result
                .operations
                .iter()
                .map(|j| j.operations.len())
                .sum::<usize>()
                != effect_count
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        let operations: Vec<_> = result
            .operations
            .iter()
            .flat_map(|journal| {
                journal
                    .operations
                    .iter()
                    .map(|operation| (operation, journal.payload(*operation)))
            })
            .collect();
        for ((effect, access), (operation, payload)) in
            result.effects.iter().zip(&result.accesses).zip(operations)
        {
            let (kind, target, opcode, value, auxiliary) = match effect.effect {
                ResidentEffect::ObjectRead { target, offset } => (
                    RESOURCE_OBJECT,
                    target,
                    crate::scheduler::device_ops::OP_READ_OBJECT,
                    0,
                    u64::from(offset),
                ),
                ResidentEffect::ObjectWrite {
                    target,
                    offset,
                    value,
                } => (
                    RESOURCE_OBJECT,
                    target,
                    crate::scheduler::device_ops::OP_WRITE_OBJECT,
                    value,
                    u64::from(offset),
                ),
                ResidentEffect::FutureObserve { target } => (
                    RESOURCE_FUTURE,
                    target,
                    crate::scheduler::device_ops::OP_OBSERVE_FUTURE,
                    0,
                    0,
                ),
                ResidentEffect::FutureAwait { target } => (
                    RESOURCE_FUTURE,
                    target,
                    crate::scheduler::device_ops::OP_AWAIT_FUTURE,
                    0,
                    0,
                ),
                ResidentEffect::FutureResolve { target, value } => (
                    RESOURCE_FUTURE,
                    target,
                    crate::scheduler::device_ops::OP_RESOLVE_FUTURE,
                    value,
                    0,
                ),
                ResidentEffect::MailboxSend { target, value } => (
                    RESOURCE_MAILBOX,
                    target,
                    crate::scheduler::device_ops::OP_ENQUEUE_MESSAGE,
                    value,
                    0,
                ),
                ResidentEffect::MailboxReceive { target } => (
                    RESOURCE_MAILBOX,
                    target,
                    crate::scheduler::device_ops::OP_RECEIVE_MESSAGE,
                    0,
                    0,
                ),
            };
            let (expected_code, expected_ref) = match effect.outcome {
                crate::executives::resident_sync::ResidentOutcome::ObjectRead(read) => (2, read),
                crate::executives::resident_sync::ResidentOutcome::ObjectWritten => (0, 0),
                crate::executives::resident_sync::ResidentOutcome::Resolved(reference) => {
                    (2, reference)
                }
                crate::executives::resident_sync::ResidentOutcome::Pending
                | crate::executives::resident_sync::ResidentOutcome::Empty => (1, 0),
                crate::executives::resident_sync::ResidentOutcome::Registered => (3, 0),
                crate::executives::resident_sync::ResidentOutcome::Sent => (0, 0),
                crate::executives::resident_sync::ResidentOutcome::Received { value, .. } => {
                    (2, value)
                }
                crate::executives::resident_sync::ResidentOutcome::CapabilityDenied => (0x104, 0),
                crate::executives::resident_sync::ResidentOutcome::InvalidTarget => (0x101, 0),
                crate::executives::resident_sync::ResidentOutcome::Full => (0x10c, 0),
                crate::executives::resident_sync::ResidentOutcome::DoubleResolve => (0x111, 0),
            };
            let expected_mode = if matches!(
                effect.effect,
                ResidentEffect::ObjectRead { .. } | ResidentEffect::FutureObserve { .. }
            ) {
                crate::scheduler::device::DEVICE_ACCESS_READ
            } else {
                crate::scheduler::device::DEVICE_ACCESS_WRITE
            };
            if access.lane != effect.lane
                || access.ordinal != effect.ordinal
                || access.resource_kind != kind
                || access.resource != u64::from(target)
                || access.mode != expected_mode
                || operation.lane != effect.lane
                || operation.ordinal != effect.ordinal
                || operation.opcode != opcode
                || operation.actor != self.actors_by_id[&effect.continuation].to_u64()
                || operation.target != u64::from(target)
                || operation.value != value
                || operation.auxiliary != auxiliary
                || operation.flags != 0
                || operation.result_aux != 0
                || operation.result_code != expected_code
                || operation.result_ref != expected_ref
                || (kind == RESOURCE_OBJECT && operation.payload_len != 8)
                || match effect.outcome {
                    crate::executives::resident_sync::ResidentOutcome::ObjectRead(read) => {
                        payload != Some(read.to_le_bytes().as_slice())
                    }
                    crate::executives::resident_sync::ResidentOutcome::ObjectWritten => {
                        payload != Some(value.to_le_bytes().as_slice())
                    }
                    _ => operation.payload_len != 0 || payload != Some(&[]),
                }
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        // Translate every journal resource before conflict validation. Conflicts
        // are expected synchronization edges here, not speculative writes to discard.
        let translated: Vec<_> = result
            .accesses
            .iter()
            .map(|a| {
                let mut x = *a;
                x.resource = match a.resource_kind {
                    RESOURCE_OBJECT => self.objects.get(a.resource as usize),
                    RESOURCE_FUTURE => self.futures.get(a.resource as usize),
                    RESOURCE_MAILBOX => self.mailboxes.get(a.resource as usize),
                    _ => None,
                }
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?
                .to_u64();
                Ok(x)
            })
            .collect::<Result<_, _>>()?;
        let lane_count = translated.iter().map(|a| a.lane).max().map_or(0, |x| x + 1);
        let _conflicts = reference_lane_conflicts(&translated, lane_count);
        let object_accesses: Vec<_> = translated
            .iter()
            .copied()
            .filter(|access| access.resource_kind == RESOURCE_OBJECT)
            .collect();
        if reference_lane_conflicts(&object_accesses, lane_count)
            .iter()
            .any(|conflict| conflict.conflicts != 0)
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        for journal in &result.operations {
            for op in &journal.operations {
                let actor = Ref64::from_u64(op.actor);
                if !self.actors_by_id.values().any(|x| *x == actor) {
                    return Err(KernelResidentSyncError::InvalidDeviceResult);
                }
                let target = match op.opcode {
                    crate::scheduler::device_ops::OP_READ_OBJECT
                    | crate::scheduler::device_ops::OP_WRITE_OBJECT => {
                        self.objects.get(op.target as usize)
                    }
                    crate::scheduler::device_ops::OP_OBSERVE_FUTURE
                    | crate::scheduler::device_ops::OP_AWAIT_FUTURE
                    | crate::scheduler::device_ops::OP_RESOLVE_FUTURE => {
                        self.futures.get(op.target as usize)
                    }
                    crate::scheduler::device_ops::OP_ENQUEUE_MESSAGE
                    | crate::scheduler::device_ops::OP_RECEIVE_MESSAGE => {
                        self.mailboxes.get(op.target as usize)
                    }
                    _ => None,
                }
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                if target.is_null() {
                    return Err(KernelResidentSyncError::InvalidDeviceResult);
                }
            }
        }
        let mut staged = kernel.clone();
        self.import_resident_state_transaction(&mut staged, &result)?;
        let violations = crate::semantics::invariants::check(&staged);
        if !violations.is_empty() {
            #[cfg(test)]
            eprintln!("resident invariant violations: {violations:#?}");
            return Err(KernelResidentSyncError::InvariantViolation);
        }
        *kernel = staged;
        Ok(result.epochs)
    }

    /// Transactionally replay the validated bounded subset through ordinary
    /// Kernel operations, canonical Phase G, and Phase H on a clone.
    fn import_resident_state_transaction(
        &self,
        k: &mut Kernel,
        r: &ResidentSyncResult,
    ) -> Result<(), KernelResidentSyncError> {
        let base_epoch = self.initial_epoch;
        for epoch_record in &r.epoch_records {
            k.epoch = base_epoch.wrapping_add(epoch_record.epoch);
            if epoch_record.epoch != 0 {
                k.open_epoch_positions();
            }
            let mut invocations: Vec<_> = r
                .invocations
                .iter()
                .filter(|inv| inv.epoch == epoch_record.epoch)
                .collect();
            invocations.sort_by_key(|inv| inv.lane);
            let mut pending = k.scheduler.pending_entries();
            pending.sort_by_key(|(bin, continuation)| (*bin, continuation.to_u64()));
            let candidates: Vec<_> = pending
                .iter()
                .map(|(bin, reference)| {
                    let descriptor = k
                        .continuations
                        .get(*reference)
                        .map_err(|_| KernelResidentSyncError::StalePlan)?;
                    if descriptor.status != ContinuationState::Runnable {
                        return Err(KernelResidentSyncError::InvalidDeviceResult);
                    }
                    Ok(crate::scheduler::admission::Candidate {
                        bin: *bin,
                        continuation: *reference,
                        process: descriptor.process,
                        run_class: descriptor.run_class,
                        state_access: descriptor.state_access,
                        waiting_since: descriptor.last_run_epoch.max(descriptor.created_epoch),
                    })
                })
                .collect::<Result<_, _>>()?;
            let decision = crate::scheduler::admission::admit(&candidates);
            let admitted: Vec<_> = decision
                .bins()
                .iter()
                .flat_map(|(_, lanes)| lanes.iter().map(|(continuation, _)| *continuation))
                .collect();
            let invoked: Vec<_> = invocations
                .iter()
                .map(|invocation| self.continuations_by_id[&invocation.continuation])
                .collect();
            if admitted != invoked || epoch_record.invocations as usize != invoked.len() {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            for candidate in &candidates {
                k.scheduler.remove(candidate.continuation);
            }
            for (run_class, continuation) in decision.deferred() {
                k.emit(crate::kernel::effects::Effect::Requeue {
                    continuation: *continuation,
                    run_class: *run_class,
                });
                k.accounting.serial_deferrals += 1;
            }
            k.admission_counters.emit();
            k.admission_log
                .push(crate::scheduler::admission::AdmissionRecord {
                    candidates,
                    decision,
                });
            k.current_lane = crate::abi::traces::HOST_LANE;
            let width = self.config.cohort_width as usize;
            let mut cohort_start = 0usize;
            while cohort_start < invocations.len() {
                let run_class = invocations[cohort_start].run_class;
                let class_end = invocations[cohort_start..]
                    .iter()
                    .position(|inv| inv.run_class != run_class)
                    .map_or(invocations.len(), |offset| cohort_start + offset);
                let cohort_end = cohort_start.saturating_add(width).min(class_end);
                let active_lanes = cohort_end - cohort_start;
                let first = self.continuations_by_id[&invocations[cohort_start].continuation];
                k.trace(
                    EventKind::CohortCreated,
                    Ref64::NULL,
                    first,
                    run_class,
                    active_lanes as u32,
                );
                k.accounting.cohorts += 1;
                k.accounting.lane_slots += u64::from(self.config.cohort_width);
                k.accounting.useful_lane_slots += active_lanes as u64;
                k.accounting.idle_lane_slots +=
                    u64::from(self.config.cohort_width) - active_lanes as u64;
                if active_lanes == width {
                    k.accounting.full_cohorts += 1;
                }
                cohort_start = cohort_end;
            }
            for inv in invocations {
                let cont = self.continuations_by_id[&inv.continuation];
                let actor = self.actors_by_id[&inv.continuation];
                k.scheduler.remove(cont);
                k.current_lane = inv.lane + 1;
                k.lane_sequence = 0;
                k.lane_effect_sequence = 0;
                k.trace(
                    EventKind::ContinuationStarted,
                    actor,
                    cont,
                    inv.run_class,
                    0,
                );
                if let Ok(p) = k.processes.get_mut(actor) {
                    p.active_continuation = cont;
                }
                let descriptor = k
                    .continuations
                    .get_mut(cont)
                    .map_err(|_| KernelResidentSyncError::StalePlan)?;
                descriptor.remaining_steps = descriptor
                    .remaining_steps
                    .checked_sub(1)
                    .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                descriptor.status = ContinuationState::Running;
                descriptor.last_run_epoch = k.epoch;
                let mut effects: Vec<_> = r
                    .effects
                    .iter()
                    .filter(|effect| effect.epoch == inv.epoch && effect.lane == inv.lane)
                    .collect();
                effects.sort_by_key(|effect| effect.ordinal);
                let mut parked = None;
                for effect in effects {
                    let target = self
                        .translate_target(effect.effect)
                        .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                    match effect.effect {
                        ResidentEffect::ObjectRead { offset, .. } => {
                            let bytes = k
                                .object_bytes(actor, target)
                                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
                            let start = offset as usize;
                            let actual = bytes
                                .get(
                                    start
                                        ..start
                                            .checked_add(8)
                                            .ok_or(KernelResidentSyncError::InvalidDeviceResult)?,
                                )
                                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                            match effect.outcome {
                                crate::executives::resident_sync::ResidentOutcome::ObjectRead(
                                    expected,
                                ) if actual == expected.to_le_bytes() => {}
                                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
                            }
                        }
                        ResidentEffect::ObjectWrite { offset, value, .. } => {
                            let bytes = k
                                .object_bytes_mut(actor, target)
                                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
                            let start = offset as usize;
                            let actual = bytes
                                .get_mut(
                                    start
                                        ..start
                                            .checked_add(8)
                                            .ok_or(KernelResidentSyncError::InvalidDeviceResult)?,
                                )
                                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                            if effect.outcome
                                != crate::executives::resident_sync::ResidentOutcome::ObjectWritten
                            {
                                return Err(KernelResidentSyncError::InvalidDeviceResult);
                            }
                            actual.copy_from_slice(&value.to_le_bytes());
                        }
                        ResidentEffect::FutureObserve { .. } => {
                            let observed = k
                                .observe_future(actor, target)
                                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
                            match (observed, effect.outcome) {
                                (
                                    None,
                                    crate::executives::resident_sync::ResidentOutcome::Pending,
                                ) => {}
                                (
                                    Some(value),
                                    crate::executives::resident_sync::ResidentOutcome::Resolved(
                                        expected,
                                    ),
                                ) if value.to_u64() == expected => {}
                                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
                            }
                        }
                        ResidentEffect::FutureAwait { .. } => {
                            // `next_run_class` belongs to a registered park. An
                            // already-settled await preserves the executing class
                            // until the disposition performs yield/complete.
                            let await_run_class = if matches!(
                                effect.outcome,
                                crate::executives::resident_sync::ResidentOutcome::Registered
                            ) {
                                inv.next_run_class
                            } else {
                                inv.run_class
                            };
                            let outcome = k
                                .await_future(actor, cont, target, await_run_class)
                                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
                            match (outcome, effect.outcome) {
                                (
                                    crate::kernel::AwaitOutcome::Registered,
                                    crate::executives::resident_sync::ResidentOutcome::Registered,
                                ) => parked = Some(target),
                                (
                                    crate::kernel::AwaitOutcome::AlreadySettled(_),
                                    crate::executives::resident_sync::ResidentOutcome::Resolved(
                                        value,
                                    ),
                                ) if k.future_value(target).map(|reference| reference.to_u64())
                                    == Some(value) => {}
                                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
                            }
                        }
                        ResidentEffect::FutureResolve { value, .. } => {
                            let applied = k.resolve_future(actor, target, Ref64::from_u64(value));
                            match (applied,effect.outcome){(Ok(()),crate::executives::resident_sync::ResidentOutcome::Resolved(v)) if v==value=>{},(Err(crate::kernel::RuntimeError::AlreadyResolved),crate::executives::resident_sync::ResidentOutcome::DoubleResolve)=>{},_=>return Err(KernelResidentSyncError::InvalidDeviceResult)}
                        }
                        ResidentEffect::MailboxSend { value, .. } => {
                            let applied =
                                k.enqueue_message(actor, target, Ref64::from_u64(value), cont);
                            match (applied, effect.outcome) {
                                (
                                    Ok(()),
                                    crate::executives::resident_sync::ResidentOutcome::Sent,
                                ) => {}
                                (
                                    Err(crate::kernel::RuntimeError::MailboxFull),
                                    crate::executives::resident_sync::ResidentOutcome::Full,
                                ) => parked = Some(target),
                                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
                            }
                        }
                        ResidentEffect::MailboxReceive { .. } => {
                            let applied = k
                                .receive_message(actor, cont)
                                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
                            match (applied, effect.outcome) {
                                (
                                    None,
                                    crate::executives::resident_sync::ResidentOutcome::Empty,
                                ) => parked = Some(target),
                                (
                                    Some(message),
                                    crate::executives::resident_sync::ResidentOutcome::Received {
                                        value,
                                        sender,
                                    },
                                ) if message.payload.to_u64() == value
                                    && message.sender.to_u64() == sender => {}
                                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
                            }
                        }
                    }
                }
                let step = if let Some(target) = parked {
                    if inv.disposition != 3 {
                        return Err(KernelResidentSyncError::InvalidDeviceResult);
                    }
                    crate::abi::StepResult::await_on(target, inv.next_run_class)
                } else if inv.disposition == 1 {
                    crate::abi::StepResult::yield_next(inv.next_run_class)
                } else if inv.disposition == 2 {
                    crate::abi::StepResult::complete()
                } else {
                    return Err(KernelResidentSyncError::InvalidDeviceResult);
                };
                crate::kernel::commit::apply_step_result(k, cont, actor, step);
                k.drain_lane_trace();
            }
            k.apply_epoch_effects();
            k.current_lane = crate::abi::traces::HOST_LANE;
            let runnable = k.scheduler.total_pending();
            let completed = self
                .continuations_by_id
                .values()
                .filter(|continuation| {
                    k.continuations
                        .get(**continuation)
                        .is_ok_and(|descriptor| descriptor.status == ContinuationState::Completed)
                })
                .count();
            if runnable != epoch_record.runnable_after as usize
                || completed != epoch_record.completed_after as usize
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            k.epoch_runnable.push(runnable);
            k.accounting.epochs += 1;
            k.accounting.steps += u64::from(epoch_record.invocations);
        }
        k.epoch = base_epoch.wrapping_add(r.epochs);
        k.open_epoch_positions();
        for (&id, frame) in &r.frames {
            let cont = self.continuations_by_id[&id];
            let frame_ref = k
                .continuations
                .get(cont)
                .map_err(|_| KernelResidentSyncError::StalePlan)?
                .frame;
            let payload = k
                .object_payloads
                .get_mut(&frame_ref.key())
                .ok_or(KernelResidentSyncError::StalePlan)?;
            if payload.len() != frame.len() {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            payload.as_mut_slice().copy_from_slice(frame);
        }
        for (index, object) in self.objects.iter().enumerate() {
            let actual = k
                .object_payloads
                .get(&object.key())
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            if actual.as_slice() != r.object_values[index] {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        for (index, future) in self.futures.iter().enumerate() {
            if k.future_value(*future).map(|reference| reference.to_u64()) != r.future_values[index]
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        for (index, process) in self.mailboxes.iter().enumerate() {
            let actual = k
                .mailbox_entries(*process)
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            let expected = &r.mailboxes[index];
            if actual.len() != expected.len()
                || actual
                    .iter()
                    .zip(expected)
                    .any(|(message, (sender, value))| {
                        message.sender.to_u64() != *sender || message.payload.to_u64() != *value
                    })
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
        }
        let mut expected_future_waiters: BTreeMap<u64, Vec<(u32, Ref64)>> = BTreeMap::new();
        let mut expected_full_waiters: BTreeMap<u64, Vec<(u32, Ref64)>> = BTreeMap::new();
        let mut expected_recv_waiters: BTreeMap<u64, Vec<(u32, Ref64)>> = BTreeMap::new();
        for final_c in &r.final_continuations {
            let continuation = self.continuations_by_id[&final_c.id];
            let descriptor = k
                .continuations
                .get(continuation)
                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
            if descriptor.run_class != final_c.run_class {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            if final_c.completed {
                if descriptor.status != ContinuationState::Completed {
                    return Err(KernelResidentSyncError::InvalidDeviceResult);
                }
                continue;
            }
            let pending = final_c
                .pending
                .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
            let (target, dependency, expected) = match pending {
                ResidentEffect::FutureAwait { target } => {
                    let future = self
                        .futures
                        .get(target as usize)
                        .copied()
                        .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                    if k.future_value(future).is_some() {
                        return Err(KernelResidentSyncError::InvalidDeviceResult);
                    }
                    (future, future, &mut expected_future_waiters)
                }
                ResidentEffect::MailboxSend { target, .. } => {
                    let mailbox = self
                        .mailboxes
                        .get(target as usize)
                        .copied()
                        .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                    (mailbox, Ref64::NULL, &mut expected_full_waiters)
                }
                ResidentEffect::MailboxReceive { target } => {
                    let mailbox = self
                        .mailboxes
                        .get(target as usize)
                        .copied()
                        .ok_or(KernelResidentSyncError::InvalidDeviceResult)?;
                    (mailbox, Ref64::NULL, &mut expected_recv_waiters)
                }
                _ => return Err(KernelResidentSyncError::InvalidDeviceResult),
            };
            if descriptor.status != ContinuationState::Waiting
                || descriptor.dependency != dependency
                || k.scheduler
                    .pending_entries()
                    .iter()
                    .any(|(_, queued)| *queued == continuation)
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            expected
                .entry(target.key())
                .or_default()
                .push((final_c.waiter_order, continuation));
        }
        fn ordered_waiters(waiters: BTreeMap<u64, Vec<(u32, Ref64)>>) -> BTreeMap<u64, Vec<Ref64>> {
            waiters
                .into_iter()
                .map(|(target, mut waiters)| {
                    waiters.sort_by_key(|(order, _)| *order);
                    (
                        target,
                        waiters
                            .into_iter()
                            .map(|(_, continuation)| continuation)
                            .collect(),
                    )
                })
                .collect()
        }
        let actual_future_waiters: BTreeMap<u64, Vec<Ref64>> = k
            .future_waiters
            .iter()
            .filter(|(_, waiters)| !waiters.is_empty())
            .map(|(future, waiters)| (*future, waiters.clone()))
            .collect();
        let actual_full_waiters: BTreeMap<u64, Vec<Ref64>> = k
            .mailboxes
            .iter()
            .filter(|(_, mailbox)| !mailbox.full_waiters.is_empty())
            .map(|(mailbox, state)| (*mailbox, state.full_waiters.iter().copied().collect()))
            .collect();
        let actual_recv_waiters: BTreeMap<u64, Vec<Ref64>> = k
            .mailboxes
            .iter()
            .filter(|(_, mailbox)| !mailbox.recv_waiters.is_empty())
            .map(|(mailbox, state)| (*mailbox, state.recv_waiters.iter().copied().collect()))
            .collect();
        if actual_future_waiters != ordered_waiters(expected_future_waiters)
            || actual_full_waiters != ordered_waiters(expected_full_waiters)
            || actual_recv_waiters != ordered_waiters(expected_recv_waiters)
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{ObjectKind, ProcessMode, StateAccess};
    use crate::executives::resident_sync::{
        HANDLER_COMPLETE, HANDLER_IF_PREVIOUS_VALUE_NE_SKIP, HANDLER_YIELD,
    };
    use crate::kernel::ContinuationSpec;

    fn setup(width: u32) -> (Kernel, KernelResidentSyncPlan, Ref64, Ref64, Ref64, Ref64) {
        let mut k = Kernel::new();
        let waiter = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let resolver = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let receiver = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let sender = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let future = k.create_future(resolver);
        let payload = k.create_object(sender, ObjectKind::MessagePayload, vec![1, 2, 3, 4]);
        k.grant_capability(resolver, waiter, future, Rights::AWAIT, 0, 0)
            .unwrap();
        k.grant_capability(receiver, sender, receiver, Rights::SEND, 0, 0)
            .unwrap();
        let await_rc = 1100;
        let resolve_rc = 1101;
        let recv_rc = 1102;
        let send_rc = 1103;
        let resolve_effect_rc = 1105;
        let send_effect_rc = 1106;
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: await_rc,
            instructions: vec![
                KernelResidentInstruction::plain(
                    HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                    1,
                    payload.to_u64(),
                ),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
                KernelResidentInstruction::effect(HANDLER_EFFECT_FUTURE_AWAIT, future, Ref64::NULL),
                KernelResidentInstruction::plain(HANDLER_YIELD, await_rc, 0),
            ],
        })
        .unwrap();
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: resolve_rc,
            instructions: vec![KernelResidentInstruction::plain(
                HANDLER_YIELD,
                resolve_effect_rc,
                0,
            )],
        })
        .unwrap();
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: resolve_effect_rc,
            instructions: vec![
                KernelResidentInstruction::effect(HANDLER_EFFECT_FUTURE_RESOLVE, future, payload),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
            ],
        })
        .unwrap();
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: recv_rc,
            instructions: vec![
                KernelResidentInstruction::plain(
                    HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                    1,
                    payload.to_u64(),
                ),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
                KernelResidentInstruction::effect(
                    HANDLER_EFFECT_MAILBOX_RECEIVE,
                    receiver,
                    Ref64::NULL,
                ),
                KernelResidentInstruction::plain(HANDLER_YIELD, recv_rc, 0),
            ],
        })
        .unwrap();
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: send_rc,
            instructions: vec![KernelResidentInstruction::plain(
                HANDLER_YIELD,
                send_effect_rc,
                0,
            )],
        })
        .unwrap();
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: send_effect_rc,
            instructions: vec![
                KernelResidentInstruction::effect(HANDLER_EFFECT_MAILBOX_SEND, receiver, payload),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
            ],
        })
        .unwrap();
        let mk = |rc| ContinuationSpec::new(StateAccess::ReadOnly, rc, 0, vec![0; 8], 16);
        let a = k.create_continuation(waiter, waiter, mk(await_rc)).unwrap();
        let b = k
            .create_continuation(resolver, resolver, mk(resolve_rc))
            .unwrap();
        let c = k
            .create_continuation(receiver, receiver, mk(recv_rc))
            .unwrap();
        let d = k.create_continuation(sender, sender, mk(send_rc)).unwrap();
        let plan = k.plan_resident_sync(16, 1, 8, width).unwrap();
        (k, plan, a, b, c, d)
    }

    fn shared_class_setup(width: u32) -> (Kernel, KernelResidentSyncPlan, Vec<Ref64>) {
        let mut kernel = Kernel::new();
        let run_classes = [1400, 1401];
        for run_class in run_classes {
            kernel
                .install_resident_sync_program(KernelResidentProgram {
                    run_class,
                    instructions: vec![KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0)],
                })
                .unwrap();
        }
        let mut continuations = Vec::new();
        for index in 0..70 {
            let process = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
            continuations.push(
                kernel
                    .create_continuation(
                        process,
                        process,
                        ContinuationSpec::new(
                            StateAccess::ReadOnly,
                            run_classes[index / 35],
                            0,
                            vec![0; 8],
                            1,
                        ),
                    )
                    .unwrap(),
            );
        }
        let plan = kernel.plan_resident_sync(2, 1, 8, width).unwrap();
        (kernel, plan, continuations)
    }

    fn assert_shared_class_result(kernel: &Kernel, continuations: &[Ref64], width: u32) {
        assert_eq!(kernel.scheduler.total_pending(), 0);
        assert!(continuations.iter().all(|continuation| {
            kernel.continuation_state(*continuation).unwrap() == ContinuationState::Completed
        }));
        assert_eq!(kernel.admission_log.len(), 1);
        assert_eq!(kernel.admission_log[0].candidates.len(), 70);
        let cohorts: Vec<_> = kernel
            .trace_events()
            .iter()
            .filter(|event| event.event_kind == EventKind::CohortCreated)
            .map(|event| (event.continuation, event.run_class, event.auxiliary))
            .collect();
        let expected = if width == 1 {
            continuations
                .iter()
                .enumerate()
                .map(|(index, continuation)| (*continuation, 1400 + (index / 35) as u32, 1))
                .collect::<Vec<_>>()
        } else {
            vec![
                (continuations[0], 1400, 32),
                (continuations[32], 1400, 3),
                (continuations[35], 1401, 32),
                (continuations[67], 1401, 3),
            ]
        };
        assert_eq!(cohorts, expected);
        let cohort_count = if width == 1 { 70 } else { 4 };
        let full_count = if width == 1 { 70 } else { 2 };
        assert_eq!(kernel.accounting.cohorts, cohort_count);
        assert_eq!(
            kernel.accounting.lane_slots,
            u64::from(width) * cohort_count
        );
        assert_eq!(kernel.accounting.useful_lane_slots, 70);
        assert_eq!(
            kernel.accounting.idle_lane_slots,
            u64::from(width) * cohort_count - 70
        );
        assert_eq!(kernel.accounting.full_cohorts, full_count);
        assert!(crate::semantics::invariants::check(kernel).is_empty());
    }

    #[test]
    fn shared_run_classes_cpu_width_1_32_have_exact_canonical_cohorts() {
        let mut runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuations) = shared_class_setup(width);
            assert_eq!(kernel.run_resident_sync_cpu_reference(plan), Ok(1));
            assert_shared_class_result(&kernel, &continuations, width);
            runs.push(kernel);
        }
        assert!(crate::semantics::order::placement_neutral(&[&runs[0], &runs[1]]).is_empty());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn shared_run_classes_actual_metal_width_1_32_match_cpu_i19() {
        let (mut cpu, cpu_plan, cpu_continuations) = shared_class_setup(1);
        cpu.run_resident_sync_cpu_reference(cpu_plan).unwrap();
        assert_shared_class_result(&cpu, &cpu_continuations, 1);
        let mut metal_runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuations) = shared_class_setup(width);
            assert_eq!(kernel.run_resident_sync_metal(plan), Ok(1));
            assert_shared_class_result(&kernel, &continuations, width);
            metal_runs.push(kernel);
        }
        assert!(crate::semantics::order::placement_neutral(&[
            &cpu,
            &metal_runs[0],
            &metal_runs[1],
        ])
        .is_empty());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn reusable_metal_executor_runs_multiple_kernels_and_rejects_stale_plan() {
        let executor = KernelResidentMetalExecutor::new().unwrap();
        let mut metal_runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuations) = shared_class_setup(width);
            assert_eq!(executor.execute(&mut kernel, plan), Ok(1));
            assert_shared_class_result(&kernel, &continuations, width);
            metal_runs.push(kernel);
        }
        assert!(
            crate::semantics::order::placement_neutral(&[&metal_runs[0], &metal_runs[1],])
                .is_empty()
        );

        let (mut stale, plan, _) = shared_class_setup(32);
        stale.epoch = stale.epoch.wrapping_add(1);
        let before = KernelResidentSyncPlan::fingerprint(&stale);
        assert_eq!(
            executor.execute(&mut stale, plan),
            Err(KernelResidentSyncError::StalePlan)
        );
        assert_eq!(KernelResidentSyncPlan::fingerprint(&stale), before);
    }

    #[test]
    fn shared_run_class_device_order_and_run_class_tamper_refuse_atomically() {
        let (kernel, plan, _) = shared_class_setup(32);
        let result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        for case in 0..2 {
            let mut kernel = kernel.clone();
            let mut malformed = result.clone();
            if case == 0 {
                malformed.invocations.swap(31, 32);
            } else {
                malformed.invocations[34].run_class = 1401;
            }
            let fingerprint = KernelResidentSyncPlan::fingerprint(&kernel);
            let trace_len = kernel.trace_events().len();
            let accounting = kernel.accounting;
            let admission_len = kernel.admission_log.len();
            let effect_len = kernel.effect_log.len();
            assert_eq!(
                plan.clone().validate_and_import(&mut kernel, malformed),
                Err(KernelResidentSyncError::InvalidDeviceResult)
            );
            assert_refusal_preserves_kernel(
                &kernel,
                fingerprint,
                trace_len,
                &accounting,
                admission_len,
                effect_len,
            );
        }
    }

    fn observe_setup(
        width: u32,
        resolved: bool,
    ) -> (Kernel, KernelResidentSyncPlan, Ref64, Ref64, Option<Ref64>) {
        let mut k = Kernel::new();
        let process = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let future = k.create_future(process);
        let value = resolved.then(|| {
            let value = k.create_object(process, ObjectKind::RawBytes, vec![44]);
            k.resolve_future(process, future, value).unwrap();
            value
        });
        let run_class = 1300;
        k.install_resident_sync_program(KernelResidentProgram {
            run_class,
            instructions: vec![
                KernelResidentInstruction::effect(
                    HANDLER_EFFECT_FUTURE_OBSERVE,
                    future,
                    Ref64::NULL,
                ),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
            ],
        })
        .unwrap();
        let continuation = k
            .create_continuation(
                process,
                process,
                ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![0; 8], 2),
            )
            .unwrap();
        let plan = k.plan_resident_sync(2, 1, 8, width).unwrap();
        (k, plan, continuation, future, value)
    }

    #[test]
    fn governed_observe_future_pending_and_resolved_match_normal_kernel_reference() {
        for resolved in [false, true] {
            let (mut bridged, plan, continuation, future, expected) = observe_setup(1, resolved);
            let mut normal = bridged.clone();
            let actor = normal.continuations.get(continuation).unwrap().process;
            normal.current_lane = 1;
            assert_eq!(normal.observe_future(actor, future).unwrap(), expected);
            normal.drain_lane_trace();
            let normal_observation = normal
                .trace_events()
                .iter()
                .find(|event| event.event_kind == EventKind::FutureStateObserved)
                .copied()
                .unwrap();

            assert_eq!(bridged.run_resident_sync_cpu_reference(plan).unwrap(), 1);
            assert_eq!(
                bridged.continuation_state(continuation).unwrap(),
                ContinuationState::Completed
            );
            let observed: Vec<_> = bridged
                .trace_events()
                .iter()
                .filter(|event| event.event_kind == EventKind::FutureStateObserved)
                .collect();
            assert_eq!(observed.len(), 1);
            assert_eq!(
                (
                    observed[0].process,
                    observed[0].subject,
                    observed[0].causal,
                    observed[0].auxiliary
                ),
                (
                    normal_observation.process,
                    normal_observation.subject,
                    normal_observation.causal,
                    normal_observation.auxiliary
                )
            );
            assert_eq!(bridged.accounting.epochs, 1);
            assert_eq!(bridged.accounting.steps, 1);
            assert_eq!(bridged.admission_log.len(), 1);
            assert!(crate::semantics::invariants::check(&bridged).is_empty());
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_observe_future_pending_resolved_width_1_32_match_cpu_reference() {
        for resolved in [false, true] {
            let mut width_results = Vec::new();
            for width in [1, 32] {
                let (mut cpu, cpu_plan, cpu_continuation, _, _) = observe_setup(width, resolved);
                cpu.run_resident_sync_cpu_reference(cpu_plan).unwrap();
                let (mut metal, metal_plan, metal_continuation, _, _) =
                    observe_setup(width, resolved);
                metal.run_resident_sync_metal(metal_plan).unwrap();
                assert_eq!(
                    cpu.continuation_state(cpu_continuation),
                    metal.continuation_state(metal_continuation)
                );
                assert_eq!(cpu.effect_log, metal.effect_log);
                assert_eq!(cpu.admission_log, metal.admission_log);
                assert_eq!(cpu.accounting, metal.accounting);
                assert_eq!(cpu.log_accounting(), metal.log_accounting());
                assert!(crate::semantics::order::conforms(&cpu, &metal).is_empty());
                assert!(crate::semantics::order::conforms(&metal, &cpu).is_empty());
                width_results.push((cpu, metal));
            }
            assert_eq!(width_results[0].0.effect_log, width_results[1].0.effect_log);
            assert_eq!(width_results[0].1.effect_log, width_results[1].1.effect_log);
        }
    }

    #[test]
    fn kernel_cpu_reference_future_and_message_wake_width_1_32() {
        let (k1, p1, a1, b1, c1, d1) = setup(1);
        let mut x = k1.clone();
        x.run_resident_sync_cpu_reference(p1).unwrap();
        let (mut y, p32, a2, b2, c2, d2) = setup(32);
        y.run_resident_sync_cpu_reference(p32).unwrap();
        for c in [a1, b1, c1, d1] {
            assert_eq!(
                x.continuation_state(c).unwrap(),
                ContinuationState::Completed
            );
        }
        for c in [a2, b2, c2, d2] {
            assert_eq!(
                y.continuation_state(c).unwrap(),
                ContinuationState::Completed
            );
        }
        assert!(x
            .mailbox_entries(x.continuations.get(c1).unwrap().process)
            .unwrap()
            .is_empty());
        assert!(y
            .mailbox_entries(y.continuations.get(c2).unwrap().process)
            .unwrap()
            .is_empty());
        assert!(crate::semantics::invariants::check(&x).is_empty());
        assert!(crate::semantics::invariants::check(&y).is_empty());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn kernel_metal_future_and_message_wake_width_1_32() {
        type TraceProjection = (
            u64,
            u32,
            u32,
            u32,
            EventKind,
            Ref64,
            Ref64,
            u32,
            u32,
            Ref64,
            Ref64,
        );
        fn projection(k: &Kernel) -> Vec<TraceProjection> {
            k.trace_events()
                .iter()
                .map(|e| {
                    (
                        e.logical_time,
                        e.epoch,
                        e.lane,
                        e.lane_sequence,
                        e.event_kind,
                        e.process,
                        e.continuation,
                        e.run_class,
                        e.auxiliary,
                        e.subject,
                        e.causal,
                    )
                })
                .collect()
        }
        let (mut cpu, cpu_plan, _, _, _, _) = setup(1);
        cpu.run_resident_sync_cpu_reference(cpu_plan).unwrap();
        let mut metal_runs = Vec::new();
        for width in [1, 32] {
            let (mut k, plan, a, b, c, d) = setup(width);
            k.run_resident_sync_metal(plan).unwrap();
            for continuation in [a, b, c, d] {
                assert_eq!(
                    k.continuation_state(continuation).unwrap(),
                    ContinuationState::Completed
                );
            }
            assert!(crate::semantics::invariants::check(&k).is_empty());
            metal_runs.push(k);
        }
        for k in &metal_runs {
            assert_eq!(projection(k), projection(&cpu));
            assert_eq!(k.epoch_runnable, cpu.epoch_runnable);
            assert_eq!(k.accounting.epochs, cpu.accounting.epochs);
            assert_eq!(k.accounting.steps, cpu.accounting.steps);
            assert!(crate::semantics::order::conforms(&cpu, k).is_empty());
        }
        assert_eq!(projection(&metal_runs[0]), projection(&metal_runs[1]));
        let events = metal_runs[0].trace_events();
        let first_wait = events
            .iter()
            .position(|e| e.event_kind == EventKind::ContinuationWaiting)
            .unwrap();
        let ready = events
            .iter()
            .position(|e| {
                e.event_kind == EventKind::ContinuationReady
                    && e.logical_time > events[first_wait].logical_time
            })
            .unwrap();
        let resolved = events
            .iter()
            .position(|e| e.event_kind == EventKind::FutureResolved)
            .unwrap();
        assert!(first_wait < ready && ready < resolved);
        let sent = events
            .iter()
            .position(|e| e.event_kind == EventKind::MessageSent)
            .unwrap();
        let delivery = events
            .iter()
            .position(|e| e.event_kind == EventKind::MessageReceived)
            .unwrap();
        assert!(sent < delivery);
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event_kind == EventKind::ContinuationWaiting)
                .count(),
            2
        );
    }

    #[test]
    fn parked_future_quiescence_publishes_canonical_waiter() {
        let mut k = Kernel::new();
        let p = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let f = k.create_future(p);
        let payload = k.create_object(p, ObjectKind::MessagePayload, vec![1]);
        let rc = 1200;
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: rc,
            instructions: vec![
                KernelResidentInstruction::effect(HANDLER_EFFECT_FUTURE_AWAIT, f, Ref64::NULL),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
            ],
        })
        .unwrap();
        let c = k
            .create_continuation(
                p,
                p,
                ContinuationSpec::new(StateAccess::ReadOnly, rc, 0, vec![0; 8], 8),
            )
            .unwrap();
        let plan = k.plan_resident_sync(4, 1, 8, 1).unwrap();
        let before = KernelResidentSyncPlan::fingerprint(&k);
        let trace: Vec<_> = k
            .trace_events()
            .iter()
            .map(|e| {
                (
                    e.logical_time,
                    e.epoch,
                    e.lane,
                    e.lane_sequence,
                    e.event_kind,
                    e.engine,
                    e.process,
                    e.continuation,
                    e.run_class,
                    e.auxiliary,
                    e.subject,
                    e.causal,
                )
            })
            .collect();
        assert_eq!(k.run_resident_sync_cpu_reference(plan), Ok(1));
        assert_ne!(KernelResidentSyncPlan::fingerprint(&k), before);
        let after: Vec<_> = k
            .trace_events()
            .iter()
            .map(|e| {
                (
                    e.logical_time,
                    e.epoch,
                    e.lane,
                    e.lane_sequence,
                    e.event_kind,
                    e.engine,
                    e.process,
                    e.continuation,
                    e.run_class,
                    e.auxiliary,
                    e.subject,
                    e.causal,
                )
            })
            .collect();
        assert_ne!(after, trace);
        assert_eq!(k.continuation_state(c).unwrap(), ContinuationState::Waiting);
        assert_eq!(k.continuations.get(c).unwrap().dependency, f);
        assert_eq!(k.future_waiters.get(&f.key()), Some(&vec![c]));

        // The imported waiter is live canonical state: an ordinary governed
        // mutation drains it, normal Phase G makes it runnable, and the second
        // phase observes settlement and completes without resident import.
        k.resolve_future(p, f, payload).unwrap();
        k.apply_epoch_effects();
        assert!(k
            .scheduler
            .pending_entries()
            .iter()
            .any(|(_, queued)| *queued == c));
        assert_eq!(
            k.continuation_state(c).unwrap(),
            ContinuationState::Runnable
        );
        // Re-plan the live woken continuation. The same bounded resident
        // handler now observes settlement and completes canonically.
        let resume_plan = k.plan_resident_sync(4, 1, 8, 1).unwrap();
        assert_eq!(k.run_resident_sync_cpu_reference(resume_plan), Ok(1));
        assert_eq!(
            k.continuation_state(c).unwrap(),
            ContinuationState::Completed
        );
        assert!(k.future_waiters.get(&f.key()).is_none_or(Vec::is_empty));
        assert!(crate::semantics::invariants::check(&k).is_empty());
    }

    #[test]
    fn parked_future_metadata_tamper_refuses_atomically() {
        for case in 0..4 {
            let mut kernel = Kernel::new();
            let process = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
            let future = kernel.create_future(process);
            let run_class = 1201;
            kernel
                .install_resident_sync_program(KernelResidentProgram {
                    run_class,
                    instructions: vec![
                        KernelResidentInstruction::effect(
                            HANDLER_EFFECT_FUTURE_AWAIT,
                            future,
                            Ref64::NULL,
                        ),
                        KernelResidentInstruction::plain(HANDLER_YIELD, run_class, 0),
                    ],
                })
                .unwrap();
            kernel
                .create_continuation(
                    process,
                    process,
                    ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![0; 8], 8),
                )
                .unwrap();
            let plan = kernel.plan_resident_sync(1, 1, 8, 1).unwrap();
            let mut result = crate::executives::resident_sync::run_resident_sync(
                &plan.config,
                plan.continuations.clone(),
                &plan.programs,
            )
            .unwrap();
            match case {
                0 => result.final_continuations[0].waiter_order = 2,
                1 => {
                    result.final_continuations[0].pending =
                        Some(ResidentEffect::FutureAwait { target: u32::MAX })
                }
                2 => result.invocations[0].disposition = 1,
                _ => {
                    result.effects[0].outcome =
                        crate::executives::resident_sync::ResidentOutcome::Pending
                }
            }
            let fingerprint = KernelResidentSyncPlan::fingerprint(&kernel);
            let trace_len = kernel.trace_events().len();
            let accounting = kernel.accounting;
            let admission_len = kernel.admission_log.len();
            let effect_len = kernel.effect_log.len();
            assert_eq!(
                plan.validate_and_import(&mut kernel, result),
                Err(KernelResidentSyncError::InvalidDeviceResult)
            );
            assert_refusal_preserves_kernel(
                &kernel,
                fingerprint,
                trace_len,
                &accounting,
                admission_len,
                effect_len,
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_parked_future_width_1_32_matches_cpu_canonical_state() {
        fn parked(width: u32) -> (Kernel, KernelResidentSyncPlan) {
            let mut kernel = Kernel::new();
            let process = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
            let future = kernel.create_future(process);
            let run_class = 1202;
            kernel
                .install_resident_sync_program(KernelResidentProgram {
                    run_class,
                    instructions: vec![
                        KernelResidentInstruction::effect(
                            HANDLER_EFFECT_FUTURE_AWAIT,
                            future,
                            Ref64::NULL,
                        ),
                        KernelResidentInstruction::plain(HANDLER_YIELD, run_class, 0),
                    ],
                })
                .unwrap();
            kernel
                .create_continuation(
                    process,
                    process,
                    ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![0; 8], 8),
                )
                .unwrap();
            let plan = kernel.plan_resident_sync(1, 1, 8, width).unwrap();
            (kernel, plan)
        }

        for width in [1, 32] {
            let (base, plan) = parked(width);
            let mut cpu = base.clone();
            cpu.run_resident_sync_cpu_reference(plan.clone()).unwrap();
            let mut metal = base;
            metal.run_resident_sync_metal(plan).unwrap();
            assert_eq!(
                KernelResidentSyncPlan::fingerprint(&metal),
                KernelResidentSyncPlan::fingerprint(&cpu)
            );
            assert_eq!(metal.effect_log, cpu.effect_log);
            assert_eq!(metal.admission_log, cpu.admission_log);
            assert_eq!(metal.accounting, cpu.accounting);
        }
    }

    #[test]
    fn stale_plan_rejects_object_process_and_capability_mutations() {
        for mutation in 0..3 {
            let (mut k, plan, _, _, _, _) = setup(1);
            let payload = plan
                .programs
                .values()
                .flat_map(|p| &p.instructions)
                .find(|i| i.opcode == HANDLER_EFFECT_MAILBOX_SEND)
                .map(|i| Ref64::from_u64(i.value))
                .unwrap();
            match mutation {
                0 => k.objects.get_mut(payload).unwrap().version = 7,
                1 => {
                    let actor = Ref64::from_u64(plan.continuations[0].actor);
                    k.processes.get_mut(actor).unwrap().node_id = 9;
                }
                _ => {
                    let actor = plan
                        .actors_by_id
                        .values()
                        .copied()
                        .find(|actor| {
                            k.find_capability(*actor, payload, Rights::TRANSFER)
                                .is_some()
                        })
                        .unwrap();
                    let cap = k.find_capability(actor, payload, Rights::TRANSFER).unwrap();
                    k.capability_spaces
                        .get_mut(&actor.key())
                        .unwrap()
                        .get_mut(cap)
                        .unwrap()
                        .valid_until_epoch = 0;
                }
            }
            let before = KernelResidentSyncPlan::fingerprint(&k);
            let trace_len = k.trace_events().len();
            assert_eq!(
                k.run_resident_sync_cpu_reference(plan),
                Err(KernelResidentSyncError::StalePlan)
            );
            assert_eq!(KernelResidentSyncPlan::fingerprint(&k), before);
            assert_eq!(k.trace_events().len(), trace_len);
        }
    }

    #[test]
    fn fingerprint_domains_program_boundaries_and_trace_positions() {
        let (mut k, _plan, _a, _, _, _) = setup(1);
        let original = KernelResidentSyncPlan::fingerprint(&k);
        let keys: Vec<_> = k.resident_sync_programs.keys().copied().collect();
        let moved = k
            .resident_sync_programs
            .get_mut(&keys[0])
            .unwrap()
            .instructions
            .pop()
            .unwrap();
        k.resident_sync_programs
            .get_mut(&keys[1])
            .unwrap()
            .instructions
            .insert(0, moved);
        assert_ne!(KernelResidentSyncPlan::fingerprint(&k), original);
        let (mut k, plan, a, _, _, _) = setup(1);
        k.trace(
            EventKind::FutureStateObserved,
            k.continuations.get(a).unwrap().process,
            a,
            0,
            0,
        );
        let changed = KernelResidentSyncPlan::fingerprint(&k);
        let trace_len = k.trace_events().len();
        assert_eq!(
            k.run_resident_sync_cpu_reference(plan),
            Err(KernelResidentSyncError::StalePlan)
        );
        assert_eq!(KernelResidentSyncPlan::fingerprint(&k), changed);
        assert_eq!(k.trace_events().len(), trace_len);
    }

    fn assert_refusal_preserves_kernel(
        kernel: &Kernel,
        fingerprint: [u8; 32],
        trace_len: usize,
        accounting: &crate::kernel::accounting::Accounting,
        admission_len: usize,
        effect_len: usize,
    ) {
        assert_eq!(KernelResidentSyncPlan::fingerprint(kernel), fingerprint);
        assert_eq!(kernel.trace_events().len(), trace_len);
        assert_eq!(&kernel.accounting, accounting);
        assert_eq!(kernel.admission_log.len(), admission_len);
        assert_eq!(kernel.effect_log.len(), effect_len);
    }

    #[test]
    fn observe_denied_expired_and_invalid_target_refuse_atomically() {
        for case in 0..3 {
            let mut kernel = Kernel::new();
            let owner = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
            let observer = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
            let future = kernel.create_future(owner);
            if case == 1 {
                let capability = kernel
                    .grant_capability(owner, observer, future, Rights::AWAIT, 0, 0)
                    .unwrap();
                kernel
                    .capability_spaces
                    .get_mut(&observer.key())
                    .unwrap()
                    .get_mut(capability)
                    .unwrap()
                    .valid_until_epoch = 0;
            }
            let run_class = 1400;
            kernel
                .install_resident_sync_program(KernelResidentProgram {
                    run_class,
                    instructions: vec![
                        KernelResidentInstruction::effect(
                            HANDLER_EFFECT_FUTURE_OBSERVE,
                            if case == 2 {
                                Ref64::from_u64(u64::MAX)
                            } else {
                                future
                            },
                            Ref64::NULL,
                        ),
                        KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
                    ],
                })
                .unwrap();
            kernel
                .create_continuation(
                    observer,
                    observer,
                    ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![], 1),
                )
                .unwrap();
            let fingerprint = KernelResidentSyncPlan::fingerprint(&kernel);
            let trace_len = kernel.trace_events().len();
            let accounting = kernel.accounting;
            let admission_len = kernel.admission_log.len();
            let effect_len = kernel.effect_log.len();
            assert!(matches!(
                kernel.plan_resident_sync(1, 1, 8, 1),
                Err(KernelResidentSyncError::UnsupportedShape)
                    | Err(KernelResidentSyncError::InvalidProgram)
            ));
            assert_refusal_preserves_kernel(
                &kernel,
                fingerprint,
                trace_len,
                &accounting,
                admission_len,
                effect_len,
            );
        }
    }

    #[test]
    fn invalid_observe_device_target_refuses_import_atomically() {
        let (kernel, plan, ..) = observe_setup(1, false);
        let mut kernel = kernel;
        let mut result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        result.effects[0].effect = ResidentEffect::FutureObserve { target: u32::MAX };
        let fingerprint = KernelResidentSyncPlan::fingerprint(&kernel);
        let trace_len = kernel.trace_events().len();
        let accounting = kernel.accounting;
        let admission_len = kernel.admission_log.len();
        let effect_len = kernel.effect_log.len();
        assert_eq!(
            plan.validate_and_import(&mut kernel, result),
            Err(KernelResidentSyncError::InvalidDeviceResult)
        );
        assert_refusal_preserves_kernel(
            &kernel,
            fingerprint,
            trace_len,
            &accounting,
            admission_len,
            effect_len,
        );
    }

    fn mutable_admission_setup(width: u32) -> (Kernel, KernelResidentSyncPlan, Vec<Ref64>) {
        let mut kernel = Kernel::new();
        let process = kernel.create_process(Ref64::NULL, ProcessMode::Serial);
        let future = kernel.create_future(process);
        let value = kernel.create_object(process, ObjectKind::RawBytes, vec![1]);
        kernel.resolve_future(process, future, value).unwrap();
        let mut continuations = Vec::new();
        for run_class in [1300, 1301] {
            kernel
                .install_resident_sync_program(KernelResidentProgram {
                    run_class,
                    instructions: vec![
                        KernelResidentInstruction::plain(
                            HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                            1,
                            value.to_u64(),
                        ),
                        KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
                        KernelResidentInstruction::effect(
                            HANDLER_EFFECT_FUTURE_OBSERVE,
                            future,
                            Ref64::NULL,
                        ),
                        KernelResidentInstruction::plain(HANDLER_YIELD, run_class, 0),
                    ],
                })
                .unwrap();
            continuations.push(
                kernel
                    .create_continuation(
                        process,
                        process,
                        ContinuationSpec::new(StateAccess::Mutable, run_class, 0, vec![0; 8], 3),
                    )
                    .unwrap(),
            );
        }
        let plan = kernel.plan_resident_sync(4, 1, 8, width).unwrap();
        (kernel, plan, continuations)
    }

    #[test]
    fn multiple_mutable_continuations_defer_fairly_and_replay_exact_admission() {
        let mut runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuations) = mutable_admission_setup(width);
            assert_eq!(kernel.run_resident_sync_cpu_reference(plan), Ok(4));
            for continuation in continuations {
                assert_eq!(
                    kernel.continuation_state(continuation),
                    Ok(ContinuationState::Completed)
                );
            }
            assert_eq!(kernel.admission_log.len(), 4);
            assert_eq!(
                kernel
                    .admission_log
                    .iter()
                    .map(|record| record.candidates.len())
                    .collect::<Vec<_>>(),
                vec![2, 2, 1, 1]
            );
            assert_eq!(
                kernel
                    .admission_log
                    .iter()
                    .map(|record| record.decision.deferred().len())
                    .collect::<Vec<_>>(),
                vec![1, 1, 0, 0]
            );
            assert_eq!(kernel.accounting.serial_deferrals, 2);
            assert!(crate::semantics::invariants::check(&kernel).is_empty());
            runs.push(kernel);
        }
        assert!(crate::semantics::order::placement_neutral(&[&runs[0], &runs[1]]).is_empty());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn multiple_mutable_continuations_actual_metal_width_1_32_match_cpu() {
        let mut runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuations) = mutable_admission_setup(width);
            assert_eq!(kernel.run_resident_sync_metal(plan), Ok(4));
            for continuation in continuations {
                assert_eq!(
                    kernel.continuation_state(continuation),
                    Ok(ContinuationState::Completed)
                );
            }
            assert_eq!(kernel.accounting.serial_deferrals, 2);
            assert!(crate::semantics::invariants::check(&kernel).is_empty());
            runs.push(kernel);
        }
        assert!(crate::semantics::order::placement_neutral(&[&runs[0], &runs[1]]).is_empty());
    }

    #[test]
    fn mutable_admission_winner_tamper_refuses_atomically() {
        let (kernel, plan, continuations) = mutable_admission_setup(1);
        let mut result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        let first = result
            .invocations
            .iter_mut()
            .find(|invocation| invocation.epoch == 0)
            .unwrap();
        first.continuation = continuations[1].to_u64();
        first.run_class = 1301;
        let mut candidate = kernel.clone();
        let fingerprint = KernelResidentSyncPlan::fingerprint(&candidate);
        let trace_len = candidate.trace_events().len();
        let accounting = candidate.accounting;
        let admission_len = candidate.admission_log.len();
        let effect_len = candidate.effect_log.len();
        assert_eq!(
            plan.validate_and_import(&mut candidate, result),
            Err(KernelResidentSyncError::InvalidDeviceResult)
        );
        assert_refusal_preserves_kernel(
            &candidate,
            fingerprint,
            trace_len,
            &accounting,
            admission_len,
            effect_len,
        );
    }

    #[test]
    fn malformed_epoch_journal_refuses_atomically() {
        let (mut kernel, plan, ..) = setup(1);
        let mut result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        result.epoch_records[0].completed_after ^= 1;
        let before = KernelResidentSyncPlan::fingerprint(&kernel);
        assert_eq!(
            plan.validate_and_import(&mut kernel, result),
            Err(KernelResidentSyncError::InvalidDeviceResult)
        );
        assert_eq!(KernelResidentSyncPlan::fingerprint(&kernel), before);
    }

    #[test]
    fn governed_kernel_reference_matches_canonical_bridge() {
        let (initial, plan, a, b, c, d) = setup(1);
        let mut bridged = initial.clone();
        bridged
            .run_resident_sync_cpu_reference(plan.clone())
            .unwrap();
        let future = plan.futures[0];
        let receiver = plan
            .mailboxes
            .iter()
            .copied()
            .find(|p| *p == plan.actors_by_id[&c.to_u64()])
            .unwrap();
        let payload = plan
            .programs
            .values()
            .flat_map(|p| &p.instructions)
            .find(|i| i.opcode == HANDLER_EFFECT_MAILBOX_SEND)
            .map(|i| Ref64::from_u64(i.value))
            .unwrap();
        let mut reference = initial;
        let base = reference.epoch;
        let schedules = [vec![a, b, c, d], vec![b, d], vec![a, c], vec![c]];
        for (offset, schedule) in schedules.iter().enumerate() {
            reference.epoch = base + offset as u32;
            if offset != 0 {
                reference.open_epoch_positions();
            }
            let candidates: Vec<_> = schedule
                .iter()
                .map(|cont| {
                    let cd = reference.continuations.get(*cont).unwrap();
                    crate::scheduler::admission::Candidate {
                        bin: reference.scheduler.bin_of(cd.run_class),
                        continuation: *cont,
                        process: cd.process,
                        run_class: cd.run_class,
                        state_access: cd.state_access,
                        waiting_since: cd.last_run_epoch.max(cd.created_epoch),
                    }
                })
                .collect();
            let decision = crate::scheduler::admission::admit(&candidates);
            reference.admission_counters.emit();
            reference
                .admission_log
                .push(crate::scheduler::admission::AdmissionRecord {
                    candidates,
                    decision,
                });
            for cont in schedule {
                let rc = reference.continuations.get(*cont).unwrap().run_class;
                reference.trace(EventKind::CohortCreated, Ref64::NULL, *cont, rc, 1);
            }
            for (lane, cont) in schedule.iter().enumerate() {
                reference.scheduler.remove(*cont);
                let (actor, rc) = reference
                    .continuations
                    .get(*cont)
                    .map(|cd| (cd.process, cd.run_class))
                    .unwrap();
                reference.current_lane = lane as u32 + 1;
                reference.lane_sequence = 0;
                reference.lane_effect_sequence = 0;
                reference.trace(EventKind::ContinuationStarted, actor, *cont, rc, 0);
                reference
                    .continuations
                    .get_mut(*cont)
                    .unwrap()
                    .remaining_steps -= 1;
                let step = match (offset, *cont) {
                    (0, x) if x == a => {
                        assert_eq!(
                            reference.await_future(actor, a, future, rc).unwrap(),
                            crate::kernel::AwaitOutcome::Registered
                        );
                        crate::abi::StepResult::await_on(future, rc)
                    }
                    (0, x) if x == b => crate::abi::StepResult::yield_next(1105),
                    (0, x) if x == c => {
                        assert!(reference.receive_message(actor, c).unwrap().is_none());
                        crate::abi::StepResult::await_on(receiver, rc)
                    }
                    (0, x) if x == d => crate::abi::StepResult::yield_next(1106),
                    (1, x) if x == b => {
                        reference.resolve_future(actor, future, payload).unwrap();
                        crate::abi::StepResult::complete()
                    }
                    (1, x) if x == d => {
                        reference
                            .enqueue_message(actor, receiver, payload, d)
                            .unwrap();
                        crate::abi::StepResult::complete()
                    }
                    (2, x) if x == c => {
                        assert!(reference.receive_message(actor, c).unwrap().is_some());
                        crate::abi::StepResult::yield_next(rc)
                    }
                    _ => crate::abi::StepResult::complete(),
                };
                crate::kernel::commit::apply_step_result(&mut reference, *cont, actor, step);
                reference.drain_lane_trace();
            }
            reference.apply_epoch_effects();
            reference.current_lane = crate::abi::traces::HOST_LANE;
            let runnable = reference.scheduler.total_pending();
            reference.epoch_runnable.push(runnable);
            reference.accounting.epochs += 1;
            reference.accounting.steps += schedule.len() as u64;
            reference.accounting.cohorts += schedule.len() as u64;
            reference.accounting.full_cohorts += schedule.len() as u64;
            reference.accounting.lane_slots += schedule.len() as u64;
            reference.accounting.useful_lane_slots += schedule.len() as u64;
        }
        reference.epoch = base + 4;
        reference.open_epoch_positions();
        assert!(crate::semantics::invariants::check(&reference).is_empty());
        assert!(crate::semantics::order::conforms(&reference, &bridged).is_empty());
        assert!(crate::semantics::order::conforms(&bridged, &reference).is_empty());
        assert_eq!(reference.accounting, bridged.accounting);
        assert_eq!(reference.epoch_runnable, bridged.epoch_runnable);
        assert_eq!(reference.effect_log, bridged.effect_log);
        assert_eq!(reference.admission_log, bridged.admission_log);
        assert_eq!(
            reference.scheduler.canonical_fingerprint_bytes(),
            bridged.scheduler.canonical_fingerprint_bytes()
        );
        assert_eq!(reference.log_accounting(), bridged.log_accounting());
        assert_eq!(reference.capability_count(), bridged.capability_count());
        for cont in [a, b, c, d] {
            let x = reference.continuations.get(cont).unwrap();
            let y = bridged.continuations.get(cont).unwrap();
            assert_eq!(
                (x.status, x.run_class, x.remaining_steps, x.last_run_epoch),
                (y.status, y.run_class, y.remaining_steps, y.last_run_epoch)
            );
        }
        assert_eq!(
            reference.future_value(future).unwrap(),
            bridged.future_value(future).unwrap()
        );
        assert_eq!(
            reference.mailbox_len(receiver),
            bridged.mailbox_len(receiver)
        );
    }

    fn final_mailbox_receive_setup(
        width: u32,
    ) -> (Kernel, KernelResidentSyncPlan, Ref64, Ref64, Ref64) {
        let mut kernel = Kernel::new();
        let receiver = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
        let sender = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
        let payload = kernel.create_object(sender, ObjectKind::MessagePayload, vec![9]);
        kernel
            .grant_capability(receiver, sender, receiver, Rights::SEND, 0, 0)
            .unwrap();
        let run_class = 1600;
        kernel
            .install_resident_sync_program(KernelResidentProgram {
                run_class,
                instructions: vec![
                    KernelResidentInstruction::plain(
                        HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        1,
                        payload.to_u64(),
                    ),
                    KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
                    KernelResidentInstruction::effect(
                        HANDLER_EFFECT_MAILBOX_RECEIVE,
                        receiver,
                        Ref64::NULL,
                    ),
                    KernelResidentInstruction::plain(HANDLER_YIELD, run_class, 0),
                ],
            })
            .unwrap();
        let continuation = kernel
            .create_continuation(
                receiver,
                receiver,
                ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![0; 8], 4),
            )
            .unwrap();
        let plan = kernel.plan_resident_sync(1, 1, 8, width).unwrap();
        (kernel, plan, continuation, sender, payload)
    }

    fn final_mailbox_send_setup(width: u32) -> (Kernel, KernelResidentSyncPlan, Ref64, Ref64) {
        let mut kernel = Kernel::new();
        let receiver = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
        let sender = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
        let payload = kernel.create_object(sender, ObjectKind::MessagePayload, vec![7]);
        kernel
            .grant_capability(receiver, sender, receiver, Rights::SEND, 0, 0)
            .unwrap();
        let run_class = 1610;
        kernel
            .install_resident_sync_program(KernelResidentProgram {
                run_class,
                instructions: vec![
                    KernelResidentInstruction::effect(
                        HANDLER_EFFECT_MAILBOX_SEND,
                        receiver,
                        payload,
                    ),
                    KernelResidentInstruction::plain(HANDLER_YIELD, run_class, 0),
                ],
            })
            .unwrap();
        let continuation = kernel
            .create_continuation(
                sender,
                sender,
                ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![], 10),
            )
            .unwrap();
        let plan = kernel.plan_resident_sync(9, 1, 8, width).unwrap();
        (kernel, plan, continuation, receiver)
    }

    #[test]
    fn final_mailbox_parks_publish_exact_waiters_and_receive_can_resume() {
        let mut receive_runs = Vec::new();
        let mut send_runs = Vec::new();
        for width in [1, 32] {
            let (mut receive, plan, continuation, sender, payload) =
                final_mailbox_receive_setup(width);
            assert_eq!(receive.run_resident_sync_cpu_reference(plan), Ok(1));
            assert_eq!(
                receive.continuation_state(continuation),
                Ok(ContinuationState::Waiting)
            );
            let mailbox = receive.continuations.get(continuation).unwrap().process;
            assert_eq!(receive.mailbox_recv_waiter_count(mailbox), 1);
            assert_eq!(receive.mailbox_full_waiter_count(mailbox), 0);
            receive
                .enqueue_message(sender, mailbox, payload, Ref64::NULL)
                .unwrap();
            let retained = receive.create_object(sender, ObjectKind::MessagePayload, vec![10]);
            receive
                .enqueue_message(sender, mailbox, retained, Ref64::NULL)
                .unwrap();
            assert_eq!(
                receive.continuation_state(continuation),
                Ok(ContinuationState::Runnable)
            );
            assert_eq!(receive.mailbox_recv_waiter_count(mailbox), 0);
            let retry = receive.plan_resident_sync(8, 1, 8, width).unwrap();
            assert_eq!(receive.run_resident_sync_cpu_reference(retry), Ok(2));
            assert_eq!(
                receive.continuation_state(continuation),
                Ok(ContinuationState::Completed)
            );
            assert_eq!(
                receive.mailbox_entries(mailbox).unwrap()[0].payload,
                retained
            );
            assert!(crate::semantics::invariants::check(&receive).is_empty());
            receive_runs.push(receive);

            let (mut send, plan, parked, mailbox) = final_mailbox_send_setup(width);
            assert_eq!(send.run_resident_sync_cpu_reference(plan), Ok(9));
            assert_eq!(
                send.continuation_state(parked),
                Ok(ContinuationState::Waiting)
            );
            assert_eq!(send.mailbox_full_waiter_count(mailbox), 1);
            assert_eq!(send.mailbox_first_full_waiter(mailbox), Some(parked));
            assert_eq!(send.mailbox_recv_waiter_count(mailbox), 0);
            let receiver_continuation = send
                .create_continuation(
                    mailbox,
                    mailbox,
                    ContinuationSpec::new(StateAccess::ReadOnly, 1610, 0, vec![], 1),
                )
                .unwrap();
            assert!(send
                .receive_message(mailbox, receiver_continuation)
                .unwrap()
                .is_some());
            assert_eq!(send.mailbox_full_waiter_count(mailbox), 0);
            assert_eq!(
                send.continuation_state(parked),
                Ok(ContinuationState::Runnable)
            );
            assert!(crate::semantics::invariants::check(&send).is_empty());
            send_runs.push(send);
        }
        assert!(
            crate::semantics::order::placement_neutral(&[&receive_runs[0], &receive_runs[1],])
                .is_empty()
        );
        assert!(
            crate::semantics::order::placement_neutral(&[&send_runs[0], &send_runs[1],]).is_empty()
        );
    }

    #[test]
    fn final_mailbox_park_metadata_tamper_refuses_atomically() {
        let (kernel, plan, continuation, _, _) = final_mailbox_receive_setup(1);
        let result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        for case in 0..4 {
            let mut malformed = result.clone();
            match case {
                0 => {
                    malformed
                        .final_continuations
                        .iter_mut()
                        .find(|final_c| final_c.id == continuation.to_u64())
                        .unwrap()
                        .waiter_order += 1
                }
                1 => {
                    malformed
                        .final_continuations
                        .iter_mut()
                        .find(|final_c| final_c.id == continuation.to_u64())
                        .unwrap()
                        .pending = Some(ResidentEffect::MailboxReceive { target: u32::MAX })
                }
                2 => {
                    malformed.effects[0].outcome =
                        crate::executives::resident_sync::ResidentOutcome::Sent
                }
                3 => malformed.invocations[0].disposition = 2,
                _ => unreachable!(),
            }
            let mut candidate = kernel.clone();
            let fingerprint = KernelResidentSyncPlan::fingerprint(&candidate);
            let trace_len = candidate.trace_events().len();
            let accounting = candidate.accounting;
            let admission_len = candidate.admission_log.len();
            let effect_len = candidate.effect_log.len();
            assert_eq!(
                plan.clone().validate_and_import(&mut candidate, malformed),
                Err(KernelResidentSyncError::InvalidDeviceResult)
            );
            assert_refusal_preserves_kernel(
                &candidate,
                fingerprint,
                trace_len,
                &accounting,
                admission_len,
                effect_len,
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn final_mailbox_parks_actual_metal_width_1_32_match_cpu() {
        for receive in [true, false] {
            let mut runs = Vec::new();
            for width in [1, 32] {
                if receive {
                    let (mut kernel, plan, continuation, sender, payload) =
                        final_mailbox_receive_setup(width);
                    assert_eq!(kernel.run_resident_sync_metal(plan), Ok(1));
                    assert_eq!(
                        kernel.continuation_state(continuation),
                        Ok(ContinuationState::Waiting)
                    );
                    let mailbox = kernel.continuations.get(continuation).unwrap().process;
                    kernel
                        .enqueue_message(sender, mailbox, payload, Ref64::NULL)
                        .unwrap();
                    let retained =
                        kernel.create_object(sender, ObjectKind::MessagePayload, vec![10]);
                    kernel
                        .enqueue_message(sender, mailbox, retained, Ref64::NULL)
                        .unwrap();
                    let retry = kernel.plan_resident_sync(8, 1, 8, width).unwrap();
                    assert_eq!(kernel.run_resident_sync_metal(retry), Ok(2));
                    assert_eq!(
                        kernel.continuation_state(continuation),
                        Ok(ContinuationState::Completed)
                    );
                    assert_eq!(
                        kernel.mailbox_entries(mailbox).unwrap()[0].payload,
                        retained
                    );
                    runs.push(kernel);
                } else {
                    let (mut kernel, plan, continuation, mailbox) = final_mailbox_send_setup(width);
                    assert_eq!(kernel.run_resident_sync_metal(plan), Ok(9));
                    assert_eq!(
                        kernel.continuation_state(continuation),
                        Ok(ContinuationState::Waiting)
                    );
                    assert_eq!(
                        kernel.mailbox_first_full_waiter(mailbox),
                        Some(continuation)
                    );
                    runs.push(kernel);
                }
            }
            assert!(crate::semantics::order::placement_neutral(&[&runs[0], &runs[1]]).is_empty());
        }
    }

    fn arithmetic_frame_setup(width: u32) -> (Kernel, KernelResidentSyncPlan, Ref64) {
        let mut kernel = Kernel::new();
        let process = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
        let run_class = 1750;
        kernel
            .install_resident_sync_program(KernelResidentProgram {
                run_class,
                instructions: vec![
                    KernelResidentInstruction::add_frame_immediate(0, 1),
                    KernelResidentInstruction::complete_if_frame_eq(0, 0),
                    KernelResidentInstruction::plain(HANDLER_YIELD, run_class, 0),
                ],
            })
            .unwrap();
        let continuation = kernel
            .create_continuation(
                process,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    run_class,
                    0,
                    (u64::MAX - 4).to_le_bytes().to_vec(),
                    6,
                ),
            )
            .unwrap();
        let plan = kernel.plan_resident_sync(5, 1, 8, width).unwrap();
        (kernel, plan, continuation)
    }

    #[test]
    fn bounded_frame_arithmetic_drives_cpu_resident_completion_width_1_32() {
        let mut runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuation) = arithmetic_frame_setup(width);
            assert_eq!(kernel.run_resident_sync_cpu_reference(plan), Ok(5));
            assert_eq!(
                kernel.continuation_state(continuation),
                Ok(ContinuationState::Completed)
            );
            let frame = kernel.continuation_frame(continuation).unwrap();
            assert_eq!(
                u64::from_le_bytes(
                    kernel.object_payloads[&frame.key()].as_slice()[..8]
                        .try_into()
                        .unwrap()
                ),
                0
            );
            runs.push(kernel);
        }
        assert!(crate::semantics::order::placement_neutral(&[&runs[0], &runs[1]]).is_empty());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn bounded_frame_arithmetic_actual_metal_width_1_32_matches_cpu() {
        let mut runs = Vec::new();
        for width in [1, 32] {
            let (mut kernel, plan, continuation) = arithmetic_frame_setup(width);
            assert_eq!(kernel.run_resident_sync_metal(plan), Ok(5));
            assert_eq!(
                kernel.continuation_state(continuation),
                Ok(ContinuationState::Completed)
            );
            let frame = kernel.continuation_frame(continuation).unwrap();
            assert_eq!(
                u64::from_le_bytes(
                    kernel.object_payloads[&frame.key()].as_slice()[..8]
                        .try_into()
                        .unwrap()
                ),
                0
            );
            runs.push(kernel);
        }
        assert!(crate::semantics::order::placement_neutral(&[&runs[0], &runs[1]]).is_empty());
    }

    fn object_setup(width: u32) -> (Kernel, KernelResidentSyncPlan, Ref64, Ref64) {
        let mut k = Kernel::new();
        let process = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let object = k.create_object(process, ObjectKind::RawBytes, (0u8..16).collect());
        let run_class = 1700;
        k.install_resident_sync_program(KernelResidentProgram {
            run_class,
            instructions: vec![
                KernelResidentInstruction::object_read(object, 4),
                KernelResidentInstruction::object_write(object, 8, 0x8877665544332211),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
            ],
        })
        .unwrap();
        let continuation = k
            .create_continuation(
                process,
                process,
                ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![0; 8], 4),
            )
            .unwrap();
        let plan = k.plan_resident_sync(1, 2, 16, width).unwrap();
        let _ = continuation;
        (k, plan, object, process)
    }

    #[test]
    fn governed_object_read_write_replays_through_kernel_and_widths_match() {
        let (base1, plan1, object, process) = object_setup(1);
        let mut bridged1 = base1.clone();
        bridged1.run_resident_sync_cpu_reference(plan1).unwrap();
        let (base32, plan32, object32, _) = object_setup(32);
        let mut bridged32 = base32.clone();
        bridged32.run_resident_sync_cpu_reference(plan32).unwrap();
        assert_eq!(
            bridged1.object_payloads[&object.key()].as_slice(),
            bridged32.object_payloads[&object32.key()].as_slice()
        );
        assert_eq!(
            &bridged1.object_payloads[&object.key()].as_slice()[8..16],
            &0x8877665544332211u64.to_le_bytes()
        );

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let (metal_base1, metal_plan1, metal_object1, _) = object_setup(1);
            let mut metal1 = metal_base1;
            metal1.run_resident_sync_metal(metal_plan1).unwrap();
            assert_eq!(
                KernelResidentSyncPlan::fingerprint(&metal1),
                KernelResidentSyncPlan::fingerprint(&bridged1)
            );
            let (metal_base32, metal_plan32, metal_object32, _) = object_setup(32);
            let mut metal32 = metal_base32;
            metal32.run_resident_sync_metal(metal_plan32).unwrap();
            assert_eq!(
                KernelResidentSyncPlan::fingerprint(&metal32),
                KernelResidentSyncPlan::fingerprint(&bridged32)
            );
            assert_eq!(
                metal1.object_payloads[&metal_object1.key()].as_slice(),
                metal32.object_payloads[&metal_object32.key()].as_slice()
            );
        }

        // Independent normal governed Kernel methods reach the same object state
        // and emit the same ordered object authority decisions/effect.
        let mut ordinary = base1.clone();
        let trace_start = ordinary.trace.len();
        ordinary.current_lane = 1;
        ordinary.lane_sequence = 0;
        assert_eq!(
            &ordinary.object_bytes(process, object).unwrap()[4..12],
            &(4u8..12).collect::<Vec<_>>()
        );
        ordinary.object_bytes_mut(process, object).unwrap()[8..16]
            .copy_from_slice(&0x8877665544332211u64.to_le_bytes());
        assert_eq!(
            ordinary.object_payloads[&object.key()].as_slice(),
            bridged1.object_payloads[&object.key()].as_slice()
        );
        let authority = |kernel: &Kernel, start: usize| {
            kernel.trace[start..]
                .iter()
                .filter(|event| {
                    matches!(
                        event.event_kind,
                        EventKind::AuthorityGranted | EventKind::AuthorityEffect
                    ) && event.subject == object
                })
                .map(|event| {
                    (
                        event.event_kind,
                        event.process,
                        event.subject,
                        event.run_class,
                        event.auxiliary,
                        event.causal,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            authority(&ordinary, trace_start),
            authority(&bridged1, base1.trace.len())
        );
        assert_eq!(bridged1.admission_log.len(), base1.admission_log.len() + 1);
        assert_eq!(bridged1.accounting.steps, base1.accounting.steps + 1);
        assert_eq!(bridged1.accounting.epochs, base1.accounting.epochs + 1);
    }

    #[test]
    fn object_plan_rejects_stale_version_range_horizon_and_grow() {
        let (mut k, plan, object, _) = object_setup(1);
        k.objects.get_mut(object).unwrap().version =
            k.objects.get(object).unwrap().version.wrapping_add(1);
        assert_eq!(
            k.run_resident_sync_cpu_reference(plan),
            Err(KernelResidentSyncError::StalePlan)
        );

        let (mut payload_stale, payload_plan, object, _) = object_setup(1);
        payload_stale
            .object_payloads
            .get_mut(&object.key())
            .unwrap()
            .as_mut_slice()[0] ^= 1;
        let mutated = payload_stale.object_payloads[&object.key()]
            .as_slice()
            .to_vec();
        assert_eq!(
            payload_stale.run_resident_sync_cpu_reference(payload_plan),
            Err(KernelResidentSyncError::StalePlan)
        );
        assert_eq!(
            payload_stale.object_payloads[&object.key()].as_slice(),
            mutated
        );

        let (mut denied, _, object, actor) = object_setup(1);
        let cap = denied
            .find_capability(actor, object, Rights::READ | Rights::WRITE)
            .unwrap();
        denied
            .capability_spaces
            .get_mut(&actor.key())
            .unwrap()
            .get_mut(cap)
            .unwrap()
            .length = 7;
        assert!(matches!(
            denied.plan_resident_sync(1, 2, 16, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));

        let (mut rights_denied, _, object, actor) = object_setup(1);
        let cap = rights_denied
            .find_capability(actor, object, Rights::READ | Rights::WRITE)
            .unwrap();
        rights_denied
            .capability_spaces
            .get_mut(&actor.key())
            .unwrap()
            .get_mut(cap)
            .unwrap()
            .rights = Rights::READ;
        assert!(matches!(
            rights_denied.plan_resident_sync(1, 2, 16, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));

        let (mut invalid, _, _, _) = object_setup(1);
        invalid
            .resident_sync_programs
            .get_mut(&1700)
            .unwrap()
            .instructions[0]
            .target = Ref64::NULL;
        assert!(matches!(
            invalid.plan_resident_sync(1, 2, 16, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));

        let (mut overflow, _, _, _) = object_setup(1);
        overflow.epoch = u32::MAX;
        assert!(matches!(
            overflow.plan_resident_sync(1, 2, 16, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));

        let (mut expired, _, object, actor) = object_setup(1);
        let cap = expired
            .find_capability(actor, object, Rights::READ | Rights::WRITE)
            .unwrap();
        expired
            .capability_spaces
            .get_mut(&actor.key())
            .unwrap()
            .get_mut(cap)
            .unwrap()
            .valid_until_epoch = expired.epoch;
        assert!(matches!(
            expired.plan_resident_sync(1, 2, 16, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));

        let (mut grow, _, object, _) = object_setup(1);
        grow.object_payloads
            .get_mut(&object.key())
            .unwrap()
            .as_mut_vec()
            .unwrap()
            .push(99);
        assert!(matches!(
            grow.plan_resident_sync(1, 2, 16, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));
    }

    #[test]
    fn conflicting_object_writes_and_malformed_result_refuse_atomically() {
        let (mut k, _, object, process) = object_setup(1);
        let run_class = 1701;
        k.install_resident_sync_program(KernelResidentProgram {
            run_class,
            instructions: vec![
                KernelResidentInstruction::object_write(object, 0, 99),
                KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0),
            ],
        })
        .unwrap();
        k.create_continuation(
            process,
            process,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, vec![0; 8], 2),
        )
        .unwrap();
        let plan = k.plan_resident_sync(1, 2, 16, 1).unwrap();
        let result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        let before = k.object_payloads[&object.key()].as_slice().to_vec();
        assert_eq!(
            plan.validate_and_import(&mut k, result),
            Err(KernelResidentSyncError::InvalidDeviceResult)
        );
        assert_eq!(k.object_payloads[&object.key()].as_slice(), before);

        let (mut k, plan, object, _) = object_setup(1);
        let mut malformed = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .unwrap();
        malformed.operations[0].payload[0] ^= 1;
        let before = k.object_payloads[&object.key()].as_slice().to_vec();
        assert_eq!(
            plan.validate_and_import(&mut k, malformed),
            Err(KernelResidentSyncError::InvalidDeviceResult)
        );
        assert_eq!(k.object_payloads[&object.key()].as_slice(), before);
    }
}
