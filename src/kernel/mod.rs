//! Kernel: the tables and operations that make SOMA's processes, continuations,
//! messages, and futures real (§7, §8, §11, §12). Phase 1 is single-threaded and
//! deterministic; every table is generation-checked (§4).

pub mod accounting;
pub mod commit;
pub mod epochs;
pub mod ownership;
#[doc(hidden)]
pub mod raw;

use std::collections::{HashMap, VecDeque};

use crate::abi::capabilities::CapabilityEntry;
use crate::abi::{
    AbiError, EventKind, FutureDescriptor, FutureState, Kind, MessageDescriptor, ObjectDescriptor,
    ObjectKind, ProcessDescriptor, ProcessMode, ProcessState, Ref64, TraceEvent,
};
use crate::abi::cohorts::PartialCohortPolicy;
use crate::kernel::accounting::Accounting;
use crate::scheduler::runnable_bins::{Scheduler, SchedulingMode};
use crate::table::GenTable;

/// Runtime errors raised by kernel operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeError {
    Abi(AbiError),
    MissingPayload,
    MissingMailbox,
    MailboxFull,
    InvalidStateAccess,
    AlreadyResolved,
    NotResolved,
    MissingCapabilitySpace,
    InvalidCapabilityDerivation,
    AuthorityDenied,
}

/// Explicit principal used when the abstract machine itself performs an action.
pub const SYSTEM_PRINCIPAL: Ref64 = Ref64::NULL;

impl From<AbiError> for RuntimeError {
    fn from(e: AbiError) -> Self {
        RuntimeError::Abi(e)
    }
}

/// A bounded, ordered mailbox belonging to one process (§11).
#[derive(Debug, Default)]
pub struct Mailbox {
    /// In-flight messages, oldest first. Ordered per sender–receiver pair via
    /// `sender_sequence`.
    pub entries: VecDeque<MessageDescriptor>,
    pub capacity: usize,
    /// Continuations waiting for a message to arrive (FIFO).
    pub recv_waiters: VecDeque<Ref64>,
    /// Continuations waiting for capacity (a full mailbox), oldest first. One is
    /// woken per message received, since a receive frees exactly one slot.
    /// This must be a queue, not a single slot: with `Option`, a second blocked
    /// sender would never be registered and would park forever (§11).
    pub full_waiters: VecDeque<Ref64>,
}

impl Mailbox {
    pub fn new(capacity: usize) -> Mailbox {
        Mailbox {
            entries: VecDeque::new(),
            capacity,
            recv_waiters: VecDeque::new(),
            full_waiters: VecDeque::new(),
        }
    }
}

/// Outcome of registering a continuation on a future (§12).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AwaitOutcome {
    /// The future was pending; the continuation is now a waiter and will be
    /// woken by `resolve_future`. The caller should return `StepResult::await_on`.
    Registered,
    /// The future was already resolved, so there is nothing left to wait for.
    /// The caller should return `StepResult::yield_next` instead — registering
    /// would park the continuation forever, since `resolve_future` drains its
    /// waiter list exactly once.
    AlreadyResolved,
}

/// Inputs for creating one runnable continuation. Grouped so the access
/// declaration cannot be separated from the frame and dispatch metadata it
/// governs.
#[derive(Clone, Debug)]
pub struct ContinuationSpec {
    pub state_access: crate::abi::StateAccess,
    pub run_class: u32,
    pub resume_point: u32,
    pub frame_bytes: Vec<u8>,
    pub max_steps: u32,
}

impl ContinuationSpec {
    pub fn new(
        state_access: crate::abi::StateAccess,
        run_class: u32,
        resume_point: u32,
        frame_bytes: Vec<u8>,
        max_steps: u32,
    ) -> Self {
        Self {
            state_access,
            run_class,
            resume_point,
            frame_bytes,
            max_steps,
        }
    }
}

/// One comparable row of a trace snapshot (§21). Two runs are identical exactly
/// when their snapshot vectors are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceSnapshotRow {
    pub logical_time: u64,
    pub epoch: u32,
    pub event_kind: EventKind,
    pub engine: u16,
    pub process: u64,
    pub continuation: u64,
    pub run_class: u32,
    pub auxiliary: u32,
}

/// The whole kernel state: every table, the object payloads, mailboxes, the
/// scheduler, and the deterministic event trace (§21).
///
/// Kernel storage is intentionally private; safe callers can only change it
/// through the operations below.
///
/// ```compile_fail
/// let kernel = soma::kernel::Kernel::new();
/// let _ = kernel.objects;
/// ```
#[derive(Debug)]
pub struct Kernel {
    epoch: u32,
    logical_time: u64,
    trace: Vec<TraceEvent>,
    /// Total runnable continuations at the end of each epoch, for accounting.
    epoch_runnable: Vec<usize>,

    processes: GenTable<ProcessDescriptor>,
    objects: GenTable<ObjectDescriptor>,
    /// Capability references are relative to the acting process. Slot zero is
    /// the explicit system principal.
    capability_spaces: HashMap<u32, GenTable<CapabilityEntry>>,
    continuations: crate::table::GenTable<crate::abi::continuations::ContinuationDescriptor>,
    futures: GenTable<FutureDescriptor>,

    /// Object payload bytes, keyed by object slot. Kernel-private (§6: user
    /// programs cannot inspect or construct the physical mapping).
    object_payloads: HashMap<u32, Vec<u8>>,
    /// Mailboxes keyed by process slot.
    mailboxes: HashMap<u32, Mailbox>,
    /// Future waiters keyed by future slot.
    future_waiters: HashMap<u32, Vec<Ref64>>,
    /// Per (sender, receiver) pair, the next `sender_sequence` value (§11).
    send_sequences: HashMap<(u32, u32), u64>,

    scheduler: Scheduler,

    /// SIMD lanes per dispatch (§14). The default of 1 makes every cohort a
    /// single lane, which is exactly scalar execution.
    cohort_width: u16,
    /// What to do with a run class's final, incompletely filled cohort (§14).
    partial_policy: PartialCohortPolicy,
    /// Cumulative counters for the §27 measurements.
    accounting: Accounting,
}

impl Kernel {
    pub fn new() -> Kernel {
        Kernel::with_scheduler(Scheduler::default())
    }

    /// A kernel that bins runnable continuations by `mode` — the knob that
    /// selects between run-class cohorting and the persistent-FIFO baseline.
    pub fn with_mode(mode: SchedulingMode) -> Kernel {
        Kernel::with_scheduler(Scheduler::with_mode(mode))
    }

    fn with_scheduler(scheduler: Scheduler) -> Kernel {
        Kernel {
            epoch: 0,
            logical_time: 0,
            trace: Vec::new(),
            epoch_runnable: Vec::new(),
            processes: GenTable::new(Kind::Process),
            objects: GenTable::new(Kind::Object),
            capability_spaces: HashMap::from([(0, GenTable::new(Kind::Capability))]),
            continuations: GenTable::new(Kind::Continuation),
            futures: GenTable::new(Kind::Future),
            object_payloads: HashMap::new(),
            mailboxes: HashMap::new(),
            future_waiters: HashMap::new(),
            send_sequences: HashMap::new(),
            scheduler,
            cohort_width: 1,
            partial_policy: PartialCohortPolicy::default(),
            accounting: Accounting::default(),
        }
    }

    // ---- configuration and observation ---------------------------------

    /// Configure the lane width and partial-cohort policy used by dispatch.
    pub fn configure_cohorts(&mut self, width: u16, policy: PartialCohortPolicy) {
        self.cohort_width = width;
        self.partial_policy = policy;
    }

    pub fn cohort_width(&self) -> u16 {
        self.cohort_width
    }

    pub fn accounting(&self) -> &Accounting {
        &self.accounting
    }

    pub fn process_count(&self) -> usize {
        self.processes.len()
    }

    pub fn continuation_count(&self) -> usize {
        self.continuations.len()
    }

    pub fn capability_count(&self) -> usize {
        self.capability_spaces.values().map(GenTable::len).sum()
    }

    pub fn capability_table_kind(&self) -> Kind {
        Kind::Capability
    }

    pub fn mailbox_entries(&self, process: Ref64) -> Option<&VecDeque<MessageDescriptor>> {
        self.mailboxes.get(&process.slot).map(|mailbox| &mailbox.entries)
    }

    pub fn mailbox_full_waiter_count(&self, process: Ref64) -> usize {
        self.mailboxes
            .get(&process.slot)
            .map(|mailbox| mailbox.full_waiters.len())
            .unwrap_or(0)
    }

    pub fn mailbox_first_full_waiter(&self, process: Ref64) -> Option<Ref64> {
        self.mailboxes
            .get(&process.slot)
            .and_then(|mailbox| mailbox.full_waiters.front().copied())
    }

    pub fn trace_events(&self) -> &[TraceEvent] {
        &self.trace
    }

    pub(crate) fn epoch_number(&self) -> u32 {
        self.epoch
    }

    pub(crate) fn processes(&self) -> &GenTable<ProcessDescriptor> {
        &self.processes
    }

    pub(crate) fn objects(&self) -> &GenTable<ObjectDescriptor> {
        &self.objects
    }

    pub(crate) fn capability_spaces(&self) -> &HashMap<u32, GenTable<CapabilityEntry>> {
        &self.capability_spaces
    }

    pub(crate) fn continuations(
        &self,
    ) -> &GenTable<crate::abi::continuations::ContinuationDescriptor> {
        &self.continuations
    }

    pub(crate) fn futures(&self) -> &GenTable<FutureDescriptor> {
        &self.futures
    }

    pub(crate) fn mailboxes(&self) -> &HashMap<u32, Mailbox> {
        &self.mailboxes
    }

    pub(crate) fn future_waiters(&self) -> &HashMap<u32, Vec<Ref64>> {
        &self.future_waiters
    }

    pub(crate) fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    // ---- trace -----------------------------------------------------------

    /// Append a trace event, advancing the logical clock deterministically.
    fn trace(
        &mut self,
        event_kind: EventKind,
        process: Ref64,
        continuation: Ref64,
        run_class: u32,
        auxiliary: u32,
    ) {
        self.logical_time = self.logical_time.wrapping_add(1);
        self.trace.push(TraceEvent::new(
            self.logical_time,
            self.epoch,
            event_kind,
            process,
            continuation,
            run_class,
        ));
        if let Some(last) = self.trace.last_mut() {
            last.auxiliary = auxiliary;
        }
    }

    /// Compact, comparable snapshot of the trace for determinism tests.
    ///
    /// Every field that can diverge between two runs is included — notably
    /// `epoch` and `auxiliary`, so a run that produces the same events in the
    /// same order but on different epoch boundaries is still detected as
    /// divergent (§21).
    pub fn trace_snapshot(&self) -> Vec<TraceSnapshotRow> {
        self.trace
            .iter()
            .map(|t| TraceSnapshotRow {
                logical_time: t.logical_time,
                epoch: t.epoch,
                event_kind: t.event_kind,
                engine: t.engine,
                process: t.process.to_u64(),
                continuation: t.continuation.to_u64(),
                run_class: t.run_class,
                auxiliary: t.auxiliary,
            })
            .collect()
    }

    // ---- objects ---------------------------------------------------------

    pub fn create_object(&mut self, actor: Ref64, kind: ObjectKind, bytes: Vec<u8>) -> Ref64 {
        self.create_object_for(actor, kind, bytes)
    }

    fn create_object_for(&mut self, actor: Ref64, kind: ObjectKind, bytes: Vec<u8>) -> Ref64 {
        let r = self.objects.alloc(ObjectDescriptor::new(kind, bytes.len() as u64));
        let byte_length = bytes.len() as u64;
        self.object_payloads.insert(r.slot, bytes);
        {
            let o = self.objects.get_mut(r).expect("fresh object");
            o.id = r;
        }
        self.mint_genesis(actor, r, byte_length, 0);
        r
    }

    // ---- capabilities ----------------------------------------------------

    fn mint_genesis(
        &mut self,
        actor: Ref64,
        target: Ref64,
        length: u64,
        object_version: u32,
    ) -> Ref64 {
        let mut entry = CapabilityEntry::new(target, crate::abi::Rights::for_target(target.kind));
        entry.length = length;
        entry.object_version = object_version;
        self.capability_spaces
            .entry(actor.slot)
            .or_insert_with(|| GenTable::new(Kind::Capability))
            .alloc(entry)
    }

    /// Derive weaker authority in `actor`'s capability space.
    pub fn derive_capability(
        &mut self,
        actor: Ref64,
        parent: Ref64,
        rights: u32,
        offset: u64,
        length: u64,
    ) -> Result<Ref64, RuntimeError> {
        let space = self
            .capability_spaces
            .get_mut(&actor.slot)
            .ok_or(RuntimeError::MissingCapabilitySpace)?;
        let parent_entry = space.get(parent)?.clone();
        let child_end = offset
            .checked_add(length)
            .ok_or(RuntimeError::InvalidCapabilityDerivation)?;
        let parent_end = parent_entry
            .offset
            .checked_add(parent_entry.length)
            .ok_or(RuntimeError::InvalidCapabilityDerivation)?;
        if rights & !parent_entry.rights != 0
            || offset < parent_entry.offset
            || child_end > parent_end
        {
            return Err(RuntimeError::InvalidCapabilityDerivation);
        }

        let mut child = parent_entry;
        child.rights = rights;
        child.offset = offset;
        child.length = length;
        child.parent_capability = parent;
        Ok(space.alloc(child))
    }

    /// Find a capability in an actor-relative space by target and required rights.
    pub fn find_capability(&self, actor: Ref64, target: Ref64, rights: u32) -> Option<Ref64> {
        self.capability_spaces.get(&actor.slot)?.iter().find_map(|(r, cap)| {
            (cap.target == target && cap.rights & rights == rights).then_some(r)
        })
    }

    pub fn capability_entry(
        &self,
        actor: Ref64,
        capability: Ref64,
    ) -> Result<&CapabilityEntry, RuntimeError> {
        self.capability_spaces
            .get(&actor.slot)
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .get(capability)
            .map_err(RuntimeError::from)
    }

    /// Copy attenuated authority from `actor` into `receiver`'s space.
    /// Exported authority is a new root: grants survive the exporting holder,
    /// matching the failure policy chosen in the capability design note.
    pub fn grant_capability(
        &mut self,
        actor: Ref64,
        receiver: Ref64,
        target: Ref64,
        rights: u32,
        offset: u64,
        length: u64,
    ) -> Result<Ref64, RuntimeError> {
        let _ = self.processes.get(receiver)?;
        self.authorize(actor, crate::abi::Rights::TRANSFER, target)?;
        if rights & crate::abi::Rights::WRITE != 0 {
            // WRITE is linear ownership authority. Copying it would create a
            // second mutable holder; use `transfer_unique` to move it instead.
            return Err(RuntimeError::InvalidCapabilityDerivation);
        }
        let source_ref = self
            .find_authorized_capability(actor, crate::abi::Rights::TRANSFER, target)
            .ok_or(RuntimeError::AuthorityDenied)?;
        let source = self
            .capability_spaces
            .get(&actor.slot)
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .get(source_ref)?
            .clone();
        let end = offset
            .checked_add(length)
            .ok_or(RuntimeError::InvalidCapabilityDerivation)?;
        let source_end = source
            .offset
            .checked_add(source.length)
            .ok_or(RuntimeError::InvalidCapabilityDerivation)?;
        if rights & !source.rights != 0 || offset < source.offset || end > source_end {
            return Err(RuntimeError::InvalidCapabilityDerivation);
        }

        let mut exported = source;
        exported.rights = rights;
        exported.offset = offset;
        exported.length = length;
        exported.parent_capability = Ref64::NULL;
        self.authority_effect(actor, crate::abi::Rights::TRANSFER, target);
        Ok(self
            .capability_spaces
            .get_mut(&receiver.slot)
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .alloc(exported))
    }

    /// Central authority gate for every currently reachable operation right.
    fn authorize(&mut self, actor: Ref64, right: u32, target: Ref64) -> Result<(), RuntimeError> {
        // The explicit system principal is the implementation escape hatch
        // described in §8 of the capability design note.
        let granted = actor == SYSTEM_PRINCIPAL
            || self
                .find_authorized_capability(actor, right, target)
                .is_some();
        let decision = if granted {
            EventKind::AuthorityGranted
        } else {
            EventKind::AuthorityDenied
        };
        self.trace(decision, actor, target, right, 0);
        if granted {
            Ok(())
        } else {
            Err(RuntimeError::AuthorityDenied)
        }
    }

    /// Record the boundary at which a successful authorization takes effect.
    /// I10c requires this marker to be immediately preceded by the matching
    /// `AuthorityGranted` decision.
    pub(super) fn authority_effect(&mut self, actor: Ref64, right: u32, target: Ref64) {
        self.trace(EventKind::AuthorityEffect, actor, target, right, 0);
    }

    fn find_authorized_capability(
        &self,
        actor: Ref64,
        right: u32,
        target: Ref64,
    ) -> Option<Ref64> {
        let space = self.capability_spaces.get(&actor.slot)?;
        let object_metadata = if target.kind == Kind::Object {
            self.objects
                .get(target)
                .map(|object| (object.version, object.byte_length))
                .ok()
        } else {
            None
        };
        space.iter().find_map(|(capability_ref, capability)| {
            let full_object_access = object_metadata
                .map(|(_, byte_length)| {
                    capability.offset == 0 && capability.length >= byte_length
                })
                .unwrap_or(true);
            let object_access_rights = crate::abi::Rights::READ
                | crate::abi::Rights::WRITE
                | crate::abi::Rights::FREEZE;
            let range_is_sufficient = right & object_access_rights == 0 || full_object_access;
            (capability.target == target
                && capability.rights & right == right
                && capability.valid_until_epoch >= self.epoch
                && Self::capability_chain_is_live(space, capability_ref)
                && object_metadata
                    .map(|(version, _)| capability.object_version == version)
                    .unwrap_or(true)
                && range_is_sufficient)
                .then_some(capability_ref)
        })
    }

    fn capability_chain_is_live(
        space: &GenTable<CapabilityEntry>,
        capability: Ref64,
    ) -> bool {
        let mut current = capability;
        // More parent edges than live entries implies a cycle. The bound also
        // makes corrupted raw-test states terminate deterministically.
        for _ in 0..=space.len() {
            let Ok(entry) = space.get(current) else {
                return false;
            };
            if entry.parent_capability.is_null() {
                return true;
            }
            current = entry.parent_capability;
        }
        false
    }

    pub(crate) fn authority_holder_count(&self, target: Ref64, right: u32) -> usize {
        self.capability_spaces
            .keys()
            .filter(|holder| {
                self.find_authorized_capability(
                    Ref64 {
                        slot: **holder,
                        generation: 0,
                        kind: Kind::Process,
                        flags: 0,
                    },
                    right,
                    target,
                )
                .is_some()
            })
            .count()
    }

    pub(super) fn revoke_target_right(&mut self, target: Ref64, right: u32) {
        for space in self.capability_spaces.values_mut() {
            let roots: Vec<Ref64> = space
                .iter()
                .filter_map(|(capability, entry)| {
                    (entry.target == target && entry.rights & right == right)
                        .then_some(capability)
                })
                .collect();
            for root in roots {
                if space.get(root).is_ok() {
                    Self::revoke_capability_tree(space, root);
                }
            }
        }
    }

    pub(super) fn mint_object_read(
        &mut self,
        actor: Ref64,
        object: Ref64,
        byte_length: u64,
        version: u32,
    ) -> Ref64 {
        let mut capability = CapabilityEntry::new(object, crate::abi::Rights::READ);
        capability.length = byte_length;
        capability.object_version = version;
        self.capability_spaces
            .entry(actor.slot)
            .or_insert_with(|| GenTable::new(Kind::Capability))
            .alloc(capability)
    }

    pub(super) fn move_target_authority(
        &mut self,
        actor: Ref64,
        receiver: Ref64,
        target: Ref64,
    ) -> Result<(), RuntimeError> {
        let write_holders = self.authority_holder_count(target, crate::abi::Rights::WRITE);
        if write_holders != 1
            || self
                .find_authorized_capability(actor, crate::abi::Rights::WRITE, target)
                .is_none()
        {
            return Err(RuntimeError::AuthorityDenied);
        }
        let source_ref = self
            .find_authorized_capability(
                actor,
                crate::abi::Rights::WRITE | crate::abi::Rights::TRANSFER,
                target,
            )
            .ok_or(RuntimeError::AuthorityDenied)?;
        let mut exported = self
            .capability_spaces
            .get(&actor.slot)
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .get(source_ref)?
            .clone();
        exported.parent_capability = Ref64::NULL;

        if actor != receiver {
            self.authority_effect(actor, crate::abi::Rights::TRANSFER, target);
            self.capability_spaces
                .get_mut(&receiver.slot)
                .ok_or(RuntimeError::MissingCapabilitySpace)?
                .alloc(exported);
            let space = self
                .capability_spaces
                .get_mut(&actor.slot)
                .ok_or(RuntimeError::MissingCapabilitySpace)?;
            Self::revoke_capability_tree(space, source_ref);
        }
        Ok(())
    }

    fn revoke_capability_tree(space: &mut GenTable<CapabilityEntry>, root: Ref64) {
        let mut revoked = vec![root];
        let mut index = 0;
        while index < revoked.len() {
            let parent = revoked[index];
            let children: Vec<Ref64> = space
                .iter()
                .filter_map(|(r, capability)| {
                    (capability.parent_capability == parent && !revoked.contains(&r)).then_some(r)
                })
                .collect();
            revoked.extend(children);
            index += 1;
        }
        for capability in revoked.into_iter().rev() {
            let _ = space.delete(capability);
        }
    }

    pub fn object_bytes(&mut self, actor: Ref64, obj: Ref64) -> Result<&[u8], RuntimeError> {
        self.authorize(actor, crate::abi::Rights::READ, obj)?;
        let _ = self.objects.get(obj)?;
        self.object_payloads
            .get(&obj.slot)
            .map(|v| v.as_slice())
            .ok_or(RuntimeError::MissingPayload)
    }

    pub fn object_bytes_mut(
        &mut self,
        actor: Ref64,
        obj: Ref64,
    ) -> Result<&mut Vec<u8>, RuntimeError> {
        if self.objects.get(obj)?.object_kind == ObjectKind::ProcessState {
            return Err(RuntimeError::InvalidStateAccess);
        }
        self.authorize(actor, crate::abi::Rights::WRITE, obj)?;
        let _ = self.objects.get(obj)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, obj);
        self.object_payloads
            .get_mut(&obj.slot)
            .ok_or(RuntimeError::MissingPayload)
    }

    /// Read the first 8 bytes of an object's payload as a little-endian u64.
    pub fn read_u64_object(&mut self, actor: Ref64, obj: Ref64) -> Option<u64> {
        let b = self.object_bytes(actor, obj).ok()?;
        if b.len() < 8 {
            return None;
        }
        let arr: [u8; 8] = b[..8].try_into().ok()?;
        Some(u64::from_le_bytes(arr))
    }

    /// Mutate a process's canonical state under an explicitly mutable
    /// continuation. This is the only write entry point for `ProcessState`
    /// objects, so the I13 declaration cannot be bypassed through generic
    /// object access.
    pub fn process_state_bytes_mut(
        &mut self,
        actor: Ref64,
        continuation: Ref64,
        process: Ref64,
    ) -> Result<&mut Vec<u8>, RuntimeError> {
        let state = self.processes.get(process)?.state;
        if actor != SYSTEM_PRINCIPAL {
            let descriptor = self.continuations.get(continuation)?;
            let active = self.processes.get(process)?.active_continuation;
            if actor != process
                || descriptor.process != process
                || descriptor.state_access != crate::abi::StateAccess::Mutable
                || active != continuation
            {
                return Err(RuntimeError::InvalidStateAccess);
            }
        }
        self.authorize(actor, crate::abi::Rights::WRITE, state)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, state);
        self.object_payloads
            .get_mut(&state.slot)
            .ok_or(RuntimeError::MissingPayload)
    }

    // ---- processes -------------------------------------------------------

    pub fn create_process(&mut self, actor: Ref64, mode: ProcessMode) -> Ref64 {
        let r = self.processes.alloc(ProcessDescriptor::new(mode));
        self.capability_spaces.insert(r.slot, GenTable::new(Kind::Capability));
        self.mint_genesis(r, r, 0, 0);
        let state_obj = self.create_object_for(r, ObjectKind::ProcessState, Vec::new());
        if actor != r {
            self.mint_genesis(actor, r, 0, 0);
        }
        self.mailboxes.insert(r.slot, Mailbox::new(8));
        {
            let p = self.processes.get_mut(r).expect("fresh process");
            p.id = r;
            p.domain = r;
            p.state = state_obj;
            p.inbox = r;
            p.status = ProcessState::Created as u32;
        }
        self.trace(EventKind::ProcessCreated, r, Ref64::NULL, 0, mode as u32);
        r
    }

    pub fn process_state(&self, p: Ref64) -> Result<ProcessState, RuntimeError> {
        let pd = self.processes.get(p)?;
        Ok(match pd.status {
            1 => ProcessState::Created,
            2 => ProcessState::Runnable,
            3 => ProcessState::Running,
            4 => ProcessState::Waiting,
            5 => ProcessState::CancelPending,
            6 => ProcessState::Failed,
            7 => ProcessState::Terminated,
            _ => ProcessState::Created,
        })
    }

    // ---- continuations ---------------------------------------------------

    /// Allocate a continuation with a fresh frame object and make it runnable in
    /// its run-class bin. The child is scheduled for the next epoch.
    pub fn create_continuation(
        &mut self,
        actor: Ref64,
        process: Ref64,
        spec: ContinuationSpec,
    ) -> Result<Ref64, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::WRITE, process)?;
        if self.processes.get(process)?.process_mode == ProcessMode::Pure
            && spec.state_access == crate::abi::StateAccess::Mutable
        {
            return Err(RuntimeError::InvalidStateAccess);
        }
        self.authority_effect(actor, crate::abi::Rights::WRITE, process);
        let frame_obj = self.create_object_for(
            process,
            ObjectKind::ContinuationFrame,
            spec.frame_bytes,
        );
        let r = self.continuations.alloc(
            crate::abi::continuations::ContinuationDescriptor::new(
                process,
                spec.state_access,
                spec.run_class,
                spec.resume_point,
            ),
        );
        {
            let c = self.continuations.get_mut(r).expect("fresh continuation");
            c.id = r;
            c.frame = frame_obj;
            c.remaining_steps = spec.max_steps;
            c.status = crate::abi::continuations::ContinuationState::Runnable;
            c.created_epoch = self.epoch;
        }
        self.scheduler.enqueue(spec.run_class, r);
        self.mint_genesis(process, r, 0, 0);
        self.trace(EventKind::ContinuationReady, process, r, spec.run_class, 0);
        Ok(r)
    }

    pub fn continuation_state(
        &self,
        c: Ref64,
    ) -> Result<crate::abi::continuations::ContinuationState, RuntimeError> {
        let cd = self.continuations.get(c)?;
        Ok(cd.status)
    }

    // ---- futures ---------------------------------------------------------

    pub fn create_future(&mut self, actor: Ref64) -> Ref64 {
        let r = self.futures.alloc(FutureDescriptor::new());
        {
            let f = self.futures.get_mut(r).expect("fresh future");
            f.id = r;
            f.owner_domain = r;
        }
        self.mint_genesis(actor, r, 0, 0);
        r
    }

    /// Register `cont` as a waiter on `future`, moving it to run class
    /// `next_run_class` and the WAITING state. It is woken by `resolve_future`.
    ///
    /// If the future is *already* resolved, nothing is registered and
    /// `AlreadyResolved` is returned: the caller must yield rather than await,
    /// because `resolve_future` has already drained the waiter list and will
    /// never revisit it.
    pub fn await_future(
        &mut self,
        actor: Ref64,
        cont: Ref64,
        future: Ref64,
        next_run_class: u32,
    ) -> Result<AwaitOutcome, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::AWAIT, future)?;
        let resolved = self.futures.get(future)?.state != FutureState::Pending;
        let _ = self.continuations.get(cont)?;
        self.authority_effect(actor, crate::abi::Rights::AWAIT, future);
        {
            let c = self.continuations.get_mut(cont)?;
            c.run_class = next_run_class;
            c.dependency = future;
            if !resolved {
                c.status = crate::abi::continuations::ContinuationState::Waiting;
            }
        }
        if resolved {
            return Ok(AwaitOutcome::AlreadyResolved);
        }
        self.future_waiters.entry(future.slot).or_default().push(cont);
        // The `ContinuationWaiting` trace is emitted once, by the commit phase
        // (§18 Phase G), so every await path produces exactly one event.
        Ok(AwaitOutcome::Registered)
    }

    /// Single-assignment resolution of a future: publish the value, then wake
    /// every waiter into its (next) run-class bin (§12, §19).
    pub fn resolve_future(
        &mut self,
        actor: Ref64,
        future: Ref64,
        value: Ref64,
    ) -> Result<(), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::RESOLVE, future)?;
        if self.futures.get(future)?.state != FutureState::Pending {
            return Err(RuntimeError::AlreadyResolved);
        }
        self.authority_effect(actor, crate::abi::Rights::RESOLVE, future);
        {
            let f = self.futures.get_mut(future)?;
            f.state = FutureState::Resolved;
            f.value = value;
            f.resolved_epoch = self.epoch;
        }
        let waiters = self.future_waiters.remove(&future.slot).unwrap_or_default();
        let owner = self.futures.get(future)?.owner_domain;
        for w in waiters {
            let (process, rc) = {
                let c = self.continuations.get(w).ok();
                match c {
                    Some(c) => (c.process, c.run_class),
                    None => continue,
                }
            };
            {
                let c = self.continuations.get_mut(w).unwrap();
                c.status = crate::abi::continuations::ContinuationState::Runnable;
            }
            self.scheduler.enqueue(rc, w);
            self.trace(EventKind::ContinuationReady, process, w, rc, 0);
        }
        self.trace(EventKind::FutureResolved, owner, Ref64::NULL, 0, value.slot);
        Ok(())
    }

    /// Read a resolved future's value (an object ref), or `None` if unresolved.
    pub fn future_value(&self, future: Ref64) -> Option<Ref64> {
        match self.futures.get(future) {
            Ok(f) if f.state == FutureState::Resolved => Some(f.value),
            _ => None,
        }
    }

    // ---- messages --------------------------------------------------------

    /// Send a message from a continuation into `receiver`'s mailbox with
    /// ordered per-pair sequencing (§11), emitting a `MessageSent` trace on
    /// committed send. If the mailbox is full, registers `sender_cont` as the
    /// `full_waiter` and returns `MailboxFull` (the sender does not spin, §11).
    pub fn enqueue_message(
        &mut self,
        actor: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_cont: Ref64,
    ) -> Result<(), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::SEND, receiver)?;
        if self.mailbox_is_full(receiver)? {
            self.authority_effect(actor, crate::abi::Rights::SEND, receiver);
            return self.push_message(
                actor,
                receiver,
                payload,
                Ref64::NULL,
                sender_cont,
                true,
            );
        }
        let byte_length = self.objects.get(payload)?.byte_length;
        let transferred = self.grant_capability(
            actor,
            receiver,
            payload,
            crate::abi::Rights::READ,
            0,
            byte_length,
        )?;
        // Capability delegation emits its own TRANSFER decision/effect pair.
        // Recheck SEND so its effect marker remains immediately paired for I10c.
        self.authorize(actor, crate::abi::Rights::SEND, receiver)?;
        self.authority_effect(actor, crate::abi::Rights::SEND, receiver);
        self.push_message(actor, receiver, payload, transferred, sender_cont, true)
    }

    /// Deliver an externally-ingested message (§18 Phase A: external messages)
    /// without a `MessageSent` trace — it is an input, not a send.
    pub fn ingest_message(
        &mut self,
        actor: Ref64,
        sender: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_cont: Ref64,
    ) -> Result<(), RuntimeError> {
        if actor != SYSTEM_PRINCIPAL {
            return Err(RuntimeError::AuthorityDenied);
        }
        if self.mailbox_is_full(receiver)? {
            return self.push_message(
                sender,
                receiver,
                payload,
                Ref64::NULL,
                sender_cont,
                false,
            );
        }
        let object = self.objects.get(payload)?;
        let byte_length = object.byte_length;
        let version = object.version;
        let transferred = self.mint_object_read(receiver, payload, byte_length, version);
        self.push_message(sender, receiver, payload, transferred, sender_cont, false)
    }

    fn mailbox_is_full(&self, receiver: Ref64) -> Result<bool, RuntimeError> {
        let mailbox = self
            .mailboxes
            .get(&receiver.slot)
            .ok_or(RuntimeError::MissingMailbox)?;
        Ok(mailbox.entries.len() >= mailbox.capacity)
    }

    fn push_message(
        &mut self,
        sender: Ref64,
        receiver: Ref64,
        payload: Ref64,
        transferred_capability: Ref64,
        sender_cont: Ref64,
        trace_send: bool,
    ) -> Result<(), RuntimeError> {
        let seq = {
            let key = (sender.slot, receiver.slot);
            let s = self.send_sequences.entry(key).or_insert(0);
            let v = *s;
            *s = s.wrapping_add(1);
            v
        };
        let mut msg = MessageDescriptor::new(sender, receiver, payload);
        msg.transferred_capability = transferred_capability;
        msg.sender_sequence = seq;
        msg.logical_timestamp = self.logical_time;

        // Commit the send inside a scoped block so the mailbox borrow ends
        // before we touch other tables for tracing / waking.
        let waiter = {
            let mailbox = self
                .mailboxes
                .get_mut(&receiver.slot)
                .ok_or(RuntimeError::MissingMailbox)?;
            if mailbox.entries.len() >= mailbox.capacity {
                // Every blocked sender is registered, not just the first, so no
                // sender can park on a full mailbox and never be woken (§11).
                // A retrying sender is registered only once.
                if !sender_cont.is_null() && !mailbox.full_waiters.contains(&sender_cont) {
                    mailbox.full_waiters.push_back(sender_cont);
                }
                return Err(RuntimeError::MailboxFull);
            }
            mailbox.entries.push_back(msg);
            mailbox.recv_waiters.pop_front()
        };

        if trace_send {
            self.trace(EventKind::MessageSent, sender, sender_cont, 0, seq as u32);
        }

        if let Some(waiter) = waiter {
            let rc = self.continuations.get(waiter)?.run_class;
            self.scheduler.enqueue(rc, waiter);
            {
                let c = self.continuations.get_mut(waiter).unwrap();
                c.status = crate::abi::continuations::ContinuationState::Runnable;
            }
            self.trace(EventKind::MessageReceived, receiver, waiter, rc, 0);
        }
        Ok(())
    }

    /// Pop the next message for `cont`'s process, waking a `full_waiter` if
    /// capacity frees. Returns `Ok(None)` after registering `cont` as a receiver
    /// waiter (the caller should await).
    pub fn receive_message(
        &mut self,
        actor: Ref64,
        cont: Ref64,
    ) -> Result<Option<MessageDescriptor>, RuntimeError> {
        let process = self.continuations.get(cont)?.process;
        self.authorize(actor, crate::abi::Rights::RECEIVE, process)?;
        let _ = self
            .mailboxes
            .get(&process.slot)
            .ok_or(RuntimeError::MissingMailbox)?;
        self.authority_effect(actor, crate::abi::Rights::RECEIVE, process);
        let mailbox = self
            .mailboxes
            .get_mut(&process.slot)
            .ok_or(RuntimeError::MissingMailbox)?;
        if let Some(msg) = mailbox.entries.pop_front() {
            // One receive frees exactly one slot, so wake exactly one sender.
            if let Some(w) = mailbox.full_waiters.pop_front() {
                let rc = self.continuations.get(w)?.run_class;
                self.scheduler.enqueue(rc, w);
                {
                    let c = self.continuations.get_mut(w).unwrap();
                    c.status = crate::abi::continuations::ContinuationState::Runnable;
                }
                self.trace(EventKind::ContinuationReady, process, w, rc, 0);
            }
            self.trace(
                EventKind::MessageReceived,
                process,
                cont,
                0,
                msg.sender_sequence as u32,
            );
            return Ok(Some(msg));
        }
        mailbox.recv_waiters.push_back(cont);
        Ok(None)
    }

    /// Number of undelivered messages in a process's mailbox.
    pub fn mailbox_len(&self, p: Ref64) -> usize {
        self.mailboxes
            .get(&p.slot)
            .map(|m| m.entries.len())
            .unwrap_or(0)
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}
