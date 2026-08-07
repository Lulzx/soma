//! Kernel-owned bounded resident synchronization bridge.
//!
//! This deliberately narrow G2 slice supports local futures, initially empty
//! local process mailboxes, exact live host-backed Object `Ref64` values, and
//! all-complete graphs. Installed pointer-free bytecode executes once on Metal;
//! after final readback, validated journals are replayed through the ordinary
//! governed Kernel operations on a clone and published atomically.
//!
//! The admitted subset is intentionally strict: local unsupervised processes,
//! RunClassBins + RunPartial, unique initial run classes, no competing mutable
//! continuation for one process, no foreign payloads, no initial waiters/mail,
//! and no parked final result. This makes the resident lane set equal the
//! canonical admission set; unsupported shapes refuse before submission.
//!
//! Exact invocation/applied-disposition, wake, and per-epoch records drive
//! normal Phase-G effects, trace causality, Phase-H accounting, and admission
//! history. CPU reference execution exists only in tests.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::abi::continuations::ContinuationState;
use crate::abi::{EventKind, FutureState, Kind, Ref64, Rights};
use crate::executives::resident_sync::{
    InitialFuture, ResidentCapability, ResidentEffect, ResidentHandlerProgram, ResidentInstruction,
    ResidentSyncConfig, ResidentSyncResult, HANDLER_EFFECT_FUTURE_AWAIT,
    HANDLER_EFFECT_FUTURE_RESOLVE, HANDLER_EFFECT_MAILBOX_RECEIVE, HANDLER_EFFECT_MAILBOX_SEND,
    RESOURCE_FUTURE, RESOURCE_MAILBOX, RIGHT_READ, RIGHT_WRITE,
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
        if !plan.matches(self) {
            return Err(KernelResidentSyncError::StalePlan);
        }
        let metal = crate::executives::metal_resident_sync::MetalResidentSync::new()
            .map_err(|_| KernelResidentSyncError::BackendUnavailable)?;
        let result = metal
            .run(&plan.config, plan.continuations.clone(), &plan.programs)
            .map_err(|_| KernelResidentSyncError::BackendFailed)?;
        plan.validate_and_import(self, result)
    }

    /// Test-only independent CPU reference; it is intentionally impossible to
    /// select this path in production scheduling.
    #[cfg(test)]
    pub(crate) fn run_resident_sync_cpu_reference(
        &mut self,
        plan: KernelResidentSyncPlan,
    ) -> Result<u32, KernelResidentSyncError> {
        if !plan.matches(self) {
            return Err(KernelResidentSyncError::StalePlan);
        }
        let result = crate::executives::resident_sync::run_resident_sync(
            &plan.config,
            plan.continuations.clone(),
            &plan.programs,
        )
        .ok_or(KernelResidentSyncError::BackendFailed)?;
        plan.validate_and_import(self, result)
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
        // This slice begins at a clean resident boundary. Initial waiter and
        // mailbox import is intentionally not claimed yet.
        if kernel.future_waiters.values().any(|w| !w.is_empty())
            || kernel.mailboxes.values().any(|m| {
                !m.entries.is_empty() || !m.recv_waiters.is_empty() || !m.full_waiters.is_empty()
            })
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
        {
            return Err(KernelResidentSyncError::UnsupportedShape);
        }
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
                    HANDLER_EFFECT_FUTURE_AWAIT
                        | HANDLER_EFFECT_FUTURE_RESOLVE
                        | HANDLER_EFFECT_MAILBOX_SEND
                        | HANDLER_EFFECT_MAILBOX_RECEIVE
                );
                let argument = match instruction.opcode {
                    HANDLER_EFFECT_FUTURE_AWAIT | HANDLER_EFFECT_FUTURE_RESOLVE => *future_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    HANDLER_EFFECT_MAILBOX_SEND | HANDLER_EFFECT_MAILBOX_RECEIVE => *mailbox_index
                        .get(&instruction.target)
                        .ok_or(KernelResidentSyncError::InvalidProgram)?,
                    _ => instruction.argument,
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
                    value: if matches!(
                        instruction.opcode,
                        HANDLER_EFFECT_FUTURE_AWAIT | HANDLER_EFFECT_MAILBOX_RECEIVE
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
                frame,
            });
            continuations_by_id.insert(id, r);
            actors_by_id.insert(id, c.process);
        }
        let mut classes = std::collections::HashSet::new();
        let mut mutable_processes = std::collections::HashSet::new();
        for continuation in continuations_by_id.values() {
            let descriptor = kernel
                .continuations
                .get(*continuation)
                .map_err(|_| KernelResidentSyncError::UnsupportedShape)?;
            if !classes.insert(descriptor.run_class) {
                return Err(KernelResidentSyncError::UnsupportedShape);
            }
            if descriptor.state_access == crate::abi::StateAccess::Mutable
                && !mutable_processes.insert(descriptor.process)
            {
                return Err(KernelResidentSyncError::UnsupportedShape);
            }
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
                c.valid_until_epoch >= kernel.epoch.saturating_add(max_epochs)
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
        // Reject statically unauthorized effect shapes before submit.
        for c in &continuations {
            let p = &packed_programs[&c.run_class];
            for i in &p.instructions {
                let (kind, right) = match i.opcode {
                    HANDLER_EFFECT_FUTURE_AWAIT => (RESOURCE_FUTURE, RIGHT_READ),
                    HANDLER_EFFECT_FUTURE_RESOLVE => (RESOURCE_FUTURE, RIGHT_WRITE),
                    HANDLER_EFFECT_MAILBOX_RECEIVE => (RESOURCE_MAILBOX, RIGHT_READ),
                    HANDLER_EFFECT_MAILBOX_SEND => (RESOURCE_MAILBOX, RIGHT_WRITE),
                    _ => continue,
                };
                if !capabilities.iter().any(|x| {
                    x.actor == c.actor
                        && x.resource_kind == kind
                        && x.target == i.argument
                        && (x.rights & right) != 0
                }) {
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
                                && cap.valid_until_epoch >= kernel.epoch.saturating_add(max_epochs)
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
            .map(|p| kernel.mailboxes[&p.key()].capacity as u32)
            .collect();
        Ok(Self {
            config: ResidentSyncConfig {
                max_epochs,
                max_effects_per_step: max_effects,
                max_frame_bytes: max_frame,
                max_continuations: (continuations.len() as u32)
                    .max(capacities.iter().copied().max().unwrap_or(0)),
                cohort_width: width,
                futures: initial_futures,
                mailbox_capacities: capacities,
                capabilities,
            },
            continuations,
            programs: packed_programs,
            continuations_by_id,
            actors_by_id,
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
            ResidentEffect::FutureAwait { target }
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
        if !result.quiescent {
            return Err(KernelResidentSyncError::NotQuiescent);
        }
        if result.frames.len() != self.continuations.len()
            || result.final_continuations.len() != self.continuations.len()
            || result.future_values.len() != self.futures.len()
            || result.mailboxes.len() != self.mailboxes.len()
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        // This first Kernel-owned slice publishes only completed graphs. A
        // parked-only "quiescent" result is a deadlock, not a publishable end
        // state, until pending retry import exists.
        if result
            .final_continuations
            .iter()
            .any(|c| !c.completed || c.pending.is_some())
        {
            return Err(KernelResidentSyncError::NotQuiescent);
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
            || completed_ids != self.continuations_by_id.keys().copied().collect()
        {
            return Err(KernelResidentSyncError::InvalidDeviceResult);
        }
        let mut invocation_positions = BTreeSet::new();
        let mut invocation_entities = BTreeSet::new();
        for invocation in &result.invocations {
            if invocation.epoch >= result.epochs
                || invocation.disposition == 0
                || invocation.disposition > 3
                || !self
                    .continuations_by_id
                    .contains_key(&invocation.continuation)
                || !invocation_positions.insert((invocation.epoch, invocation.lane))
                || !invocation_entities.insert((invocation.epoch, invocation.continuation))
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
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
            .flat_map(|journal| journal.operations.iter())
            .collect();
        for ((effect, access), operation) in
            result.effects.iter().zip(&result.accesses).zip(operations)
        {
            let (kind, target, opcode, value) = match effect.effect {
                ResidentEffect::FutureAwait { target } => (
                    RESOURCE_FUTURE,
                    target,
                    crate::scheduler::device_ops::OP_AWAIT_FUTURE,
                    0,
                ),
                ResidentEffect::FutureResolve { target, value } => (
                    RESOURCE_FUTURE,
                    target,
                    crate::scheduler::device_ops::OP_RESOLVE_FUTURE,
                    value,
                ),
                ResidentEffect::MailboxSend { target, value } => (
                    RESOURCE_MAILBOX,
                    target,
                    crate::scheduler::device_ops::OP_ENQUEUE_MESSAGE,
                    value,
                ),
                ResidentEffect::MailboxReceive { target } => (
                    RESOURCE_MAILBOX,
                    target,
                    crate::scheduler::device_ops::OP_RECEIVE_MESSAGE,
                    0,
                ),
            };
            if access.lane != effect.lane
                || access.ordinal != effect.ordinal
                || access.resource_kind != kind
                || access.resource != u64::from(target)
                || operation.lane != effect.lane
                || operation.ordinal != effect.ordinal
                || operation.opcode != opcode
                || operation.actor != self.actors_by_id[&effect.continuation].to_u64()
                || operation.target != u64::from(target)
                || operation.value != value
            {
                #[cfg(test)]
                eprintln!("journal mismatch effect={effect:?} access={access:?} operation={operation:?} expected={kind}/{target}/{opcode}/{value}");
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
        for journal in &result.operations {
            for op in &journal.operations {
                let actor = Ref64::from_u64(op.actor);
                if !self.actors_by_id.values().any(|x| *x == actor) {
                    return Err(KernelResidentSyncError::InvalidDeviceResult);
                }
                let target = match op.opcode {
                    crate::scheduler::device_ops::OP_AWAIT_FUTURE
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
            let candidates: Vec<_> = invocations
                .iter()
                .map(|inv| {
                    let reference = self.continuations_by_id[&inv.continuation];
                    let descriptor = k
                        .continuations
                        .get(reference)
                        .expect("validated continuation");
                    crate::scheduler::admission::Candidate {
                        bin: k.scheduler.bin_of(inv.run_class),
                        continuation: reference,
                        process: descriptor.process,
                        run_class: inv.run_class,
                        state_access: descriptor.state_access,
                        waiting_since: descriptor.last_run_epoch.max(descriptor.created_epoch),
                    }
                })
                .collect();
            let decision = crate::scheduler::admission::admit(&candidates);
            if !decision.deferred().is_empty() {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
            k.admission_counters.emit();
            k.admission_log
                .push(crate::scheduler::admission::AdmissionRecord {
                    candidates,
                    decision,
                });
            k.current_lane = crate::abi::traces::HOST_LANE;
            for inv in &invocations {
                let cont = self.continuations_by_id[&inv.continuation];
                k.trace(
                    EventKind::CohortCreated,
                    Ref64::NULL,
                    cont,
                    inv.run_class,
                    1,
                );
                k.accounting.cohorts += 1;
                k.accounting.lane_slots += u64::from(self.config.cohort_width);
                k.accounting.useful_lane_slots += 1;
                k.accounting.idle_lane_slots += u64::from(self.config.cohort_width - 1);
                if self.config.cohort_width == 1 {
                    k.accounting.full_cohorts += 1;
                }
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
                        ResidentEffect::FutureAwait { .. } => {
                            let outcome =
                                k.await_future(actor, cont, target, inv.next_run_class)
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
        for final_c in &r.final_continuations {
            let descriptor = k
                .continuations
                .get(self.continuations_by_id[&final_c.id])
                .map_err(|_| KernelResidentSyncError::InvalidDeviceResult)?;
            if descriptor.run_class != final_c.run_class
                || descriptor.status != ContinuationState::Completed
            {
                return Err(KernelResidentSyncError::InvalidDeviceResult);
            }
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
    fn refusal_is_atomic_for_parked_quiescence() {
        let mut k = Kernel::new();
        let p = k.create_process(Ref64::NULL, ProcessMode::Pure);
        let f = k.create_future(p);
        let rc = 1200;
        k.install_resident_sync_program(KernelResidentProgram {
            run_class: rc,
            instructions: vec![
                KernelResidentInstruction::effect(HANDLER_EFFECT_FUTURE_AWAIT, f, Ref64::NULL),
                KernelResidentInstruction::plain(HANDLER_YIELD, rc, 0),
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
        assert_eq!(
            k.run_resident_sync_cpu_reference(plan),
            Err(KernelResidentSyncError::NotQuiescent)
        );
        assert_eq!(KernelResidentSyncPlan::fingerprint(&k), before);
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
        assert_eq!(after, trace);
        assert_eq!(
            k.continuation_state(c).unwrap(),
            ContinuationState::Runnable
        );
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

    #[test]
    fn rejects_noncanonical_admission_shapes_before_submit() {
        let mut k = Kernel::new();
        let p = k.create_process(Ref64::NULL, ProcessMode::Serial);
        for rc in [1300, 1301] {
            k.install_resident_sync_program(KernelResidentProgram {
                run_class: rc,
                instructions: vec![KernelResidentInstruction::plain(HANDLER_COMPLETE, 0, 0)],
            })
            .unwrap();
            k.create_continuation(
                p,
                p,
                ContinuationSpec::new(StateAccess::Mutable, rc, 0, vec![0; 8], 2),
            )
            .unwrap();
        }
        let before = KernelResidentSyncPlan::fingerprint(&k);
        assert!(matches!(
            k.plan_resident_sync(2, 1, 8, 1),
            Err(KernelResidentSyncError::UnsupportedShape)
        ));
        assert_eq!(KernelResidentSyncPlan::fingerprint(&k), before);
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
}
