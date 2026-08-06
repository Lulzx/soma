//! Kernel: the tables and operations that make SOMA's processes, continuations,
//! messages, futures, domains, modules, and collectives real. The reference
//! transition system is single-threaded and deterministic; every table is
//! generation-checked (§4 of the historical Phase-1 contract).

pub mod accounting;
pub mod capability_space;
pub mod commit;
pub mod effects;
pub mod epochs;
pub mod ownership;
pub mod payload;
#[doc(hidden)]
pub mod raw;
pub mod retention;

use std::collections::{HashMap, VecDeque};

use crate::abi::capabilities::CapabilityEntry;
use crate::abi::cohorts::PartialCohortPolicy;
use crate::abi::{
    AbiError, ChannelDescriptor, CollectiveDescriptor, CollectiveState, DomainDescriptor,
    EventKind, ExecutionContract, ExitReason, FutureDescriptor, FutureState, Kind,
    MessageDescriptor, ModuleDescriptor, ObjectDescriptor, ObjectKind, ProcessDescriptor,
    ProcessMode, ProcessState, Ref64, SupervisionNotice, SupervisionPolicy, TraceEvent,
};
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
    ProcessUnavailable,
    ChannelClosed,
    InvalidCollective,
    InvalidStateAccess,
    AlreadyResolved,
    NotResolved,
    MissingCapabilitySpace,
    InvalidCapabilityDerivation,
    InvalidSupervisionPolicy,
    InvalidMultiInput,
    DomainQuotaExceeded,
    InvalidContract,
    InvalidModule,
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

#[derive(Debug)]
struct EscrowMessage {
    descriptor: MessageDescriptor,
    payload_authority: CapabilityEntry,
}

#[derive(Debug)]
struct ChannelQueue {
    entries: VecDeque<EscrowMessage>,
    send_waiters: VecDeque<Ref64>,
    receive_waiters: VecDeque<Ref64>,
    next_sequence: u64,
}

/// Reliable kernel control queue for direct-child exit notifications.
#[derive(Debug, Default)]
pub struct SupervisionQueue {
    pub notices: VecDeque<SupervisionNotice>,
    pub waiters: VecDeque<Ref64>,
}

#[derive(Clone, Debug)]
struct RestartBlueprint {
    entry: ContinuationSpec,
}

pub(crate) struct ChannelQueueSnapshot {
    pub channel: Ref64,
    pub entries: Vec<(Ref64, u64, Ref64)>,
    pub send_waiters: Vec<Ref64>,
    pub receive_waiters: Vec<Ref64>,
}

impl ChannelQueue {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            send_waiters: VecDeque::new(),
            receive_waiters: VecDeque::new(),
            next_sequence: 0,
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
    AlreadySettled(FutureState),
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
    /// Emitting lane and its local sequence. Placement information: two runs
    /// that differ only in how work was grouped differ here, so the semantic
    /// projection I18 compares does not read these fields.
    pub lane: u32,
    pub lane_sequence: u32,
    pub event_kind: EventKind,
    pub engine: u16,
    pub process: u64,
    pub continuation: u64,
    pub run_class: u32,
    pub auxiliary: u32,
    pub subject: u64,
    pub causal: u64,
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
    /// The lane currently executing, or `HOST_LANE` between lanes.
    current_lane: u32,
    /// Emissions so far from `current_lane` this epoch. Reset when a lane is
    /// entered, which is safe because a lane is sequential; the host counter is
    /// separate so host events keep their order across an epoch's lanes.
    lane_sequence: u32,
    host_sequence: u32,
    /// Total runnable continuations at the end of each epoch, for accounting.
    epoch_runnable: Vec<usize>,
    /// Each epoch's admission candidates and the decision taken over them, kept
    /// so I22 can be checked over a whole run rather than over whichever epoch
    /// happened to be last.
    pub(crate) admission_log: Vec<crate::scheduler::admission::AdmissionRecord>,
    /// Scheduling effects the current lane has produced and the kernel has not
    /// applied yet, in production order (v0.3 §4.4). Empty outside a lane.
    pub(crate) lane_effects: Vec<((u32, u32, u32), crate::kernel::effects::Effect)>,
    /// Every effect applied over the run, with the position it was produced at
    /// and the index it was applied at. I24 asks whether the second is the
    /// first sorted.
    pub(crate) effect_log: Vec<crate::kernel::effects::EffectRecord>,
    /// Effects produced so far by `current_lane`, and by the host, this epoch.
    /// Separate from the trace counters: an effect's position says where in the
    /// plan it was produced, not how many events preceded it.
    lane_effect_sequence: u32,
    host_effect_sequence: u32,
    /// True while `apply_lane_effects` is running, so an effect produced by an
    /// application lands immediately instead of re-entering the journal it is
    /// draining.
    applying_effects: bool,

    processes: GenTable<ProcessDescriptor>,
    domains: GenTable<DomainDescriptor>,
    contracts: GenTable<ExecutionContract>,
    modules: GenTable<ModuleDescriptor>,
    root_domain: Ref64,
    objects: GenTable<ObjectDescriptor>,
    /// Capability references are relative to the acting process. Slot zero is
    /// the explicit system principal.
    capability_spaces: HashMap<u64, capability_space::CapabilitySpace>,
    continuations: crate::table::GenTable<crate::abi::continuations::ContinuationDescriptor>,
    futures: GenTable<FutureDescriptor>,
    channels: GenTable<ChannelDescriptor>,
    collectives: GenTable<CollectiveDescriptor>,

    /// Object payload bytes, keyed by object slot. Kernel-private (§6: user
    /// programs cannot inspect or construct the physical mapping).
    object_payloads: HashMap<u64, payload::Payload>,
    /// Mailboxes keyed by process slot.
    mailboxes: HashMap<u64, Mailbox>,
    /// Future waiters keyed by future slot.
    future_waiters: HashMap<u64, Vec<Ref64>>,
    channel_queues: HashMap<u64, ChannelQueue>,
    /// Per (sender, receiver) pair, the next `sender_sequence` value (§11).
    send_sequences: HashMap<(u64, u64), u64>,
    supervision_queues: HashMap<u64, SupervisionQueue>,
    restart_blueprints: HashMap<u64, RestartBlueprint>,
    module_evaluators: HashMap<u64, Vec<(u32, u32)>>,

    scheduler: Scheduler,

    /// How many allocator partitions an epoch's lanes are spread across
    /// (v0.3 §4.3). One means every allocation comes from partition 0, which is
    /// exactly what the table did before partitions existed.
    ///
    /// This is a placement knob, not a semantic one: changing it renames
    /// entities and changes nothing else, which is what I19 checks now that I18
    /// compares up to a correspondence between names (§2.6).
    allocation_partitions: u8,
    /// The partition every table is currently allocating from.
    active_partition: u8,
    /// SIMD lanes per dispatch (§14). The default of 1 makes every cohort a
    /// single lane, which is exactly scalar execution.
    cohort_width: u16,
    /// What to do with a run class's final, incompletely filled cohort (§14).
    partial_policy: PartialCohortPolicy,
    /// Consecutive epochs a runnable continuation may wait before I21 calls it
    /// starved. Generous by default: the clause exists to catch a policy that
    /// never runs a class at all, not to police scheduling latency.
    deferral_bound: u32,
    /// Cumulative counters for the §27 measurements.
    accounting: Accounting,

    /// How long the append-only logs are kept. `Retain` by default, which is
    /// what every whole-run invariant check needs; see `kernel::retention`.
    retention: retention::LogRetention,
    trace_counters: retention::LogCounters,
    effect_counters: retention::LogCounters,
    admission_counters: retention::LogCounters,
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
        let mut kernel = Kernel {
            epoch: 0,
            logical_time: 0,
            trace: Vec::new(),
            current_lane: crate::abi::traces::HOST_LANE,
            lane_sequence: 0,
            host_sequence: 0,
            epoch_runnable: Vec::new(),
            admission_log: Vec::new(),
            lane_effects: Vec::new(),
            effect_log: Vec::new(),
            lane_effect_sequence: 0,
            host_effect_sequence: 0,
            applying_effects: false,
            processes: GenTable::new(Kind::Process),
            domains: GenTable::new(Kind::Domain),
            contracts: GenTable::new(Kind::Contract),
            modules: GenTable::new(Kind::Module),
            root_domain: Ref64::NULL,
            objects: GenTable::new(Kind::Object),
            capability_spaces: HashMap::from([(
                0,
                capability_space::CapabilitySpace::new(Kind::Capability),
            )]),
            continuations: GenTable::new(Kind::Continuation),
            futures: GenTable::new(Kind::Future),
            channels: GenTable::new(Kind::Channel),
            collectives: GenTable::new(Kind::Collective),
            object_payloads: HashMap::new(),
            mailboxes: HashMap::new(),
            future_waiters: HashMap::new(),
            channel_queues: HashMap::new(),
            send_sequences: HashMap::new(),
            supervision_queues: HashMap::new(),
            restart_blueprints: HashMap::new(),
            module_evaluators: HashMap::new(),
            scheduler,
            allocation_partitions: 1,
            active_partition: 0,
            cohort_width: 1,
            partial_policy: PartialCohortPolicy::default(),
            deferral_bound: 64,
            accounting: Accounting::default(),
            retention: retention::LogRetention::default(),
            trace_counters: retention::LogCounters::default(),
            effect_counters: retention::LogCounters::default(),
            admission_counters: retention::LogCounters::default(),
        };
        let root = kernel.domains.alloc(DomainDescriptor::new(Ref64::NULL, 0));
        kernel.domains.get_mut(root).expect("fresh root domain").id = root;
        kernel.root_domain = root;
        kernel.mint_genesis(SYSTEM_PRINCIPAL, root, 0, 0);
        kernel
    }

    // ---- configuration and observation ---------------------------------

    /// Configure the lane width and partial-cohort policy used by dispatch.
    pub fn configure_cohorts(&mut self, width: u16, policy: PartialCohortPolicy) {
        self.cohort_width = width;
        self.partial_policy = policy;
    }

    /// Spread the epoch's lanes across `partitions` allocators.
    ///
    /// A lane's partition comes from its position in the plan, so it is decided
    /// before anything runs and does not depend on which worker picks the lane
    /// up. That is what makes partitioned allocation deterministic: within one
    /// partition, allocations still happen in lane order.
    pub fn set_allocation_partitions(&mut self, partitions: u8) {
        self.allocation_partitions = partitions.max(1);
    }

    pub fn allocation_partitions(&self) -> u8 {
        self.allocation_partitions
    }

    /// Point every table's allocator at `partition`.
    ///
    /// Capability spaces are included: they are actor-relative, and two
    /// read-only continuations of one process may run in the same epoch and
    /// both mint authority, so their space is as contended as any other table.
    fn set_active_partition(&mut self, partition: u8) {
        if self.allocation_partitions == 1 && self.active_partition == 0 {
            // Nothing to switch: one partition means everything already
            // allocates from partition 0. Worth the branch because the loop
            // over capability spaces below is proportional to the process
            // count and would otherwise run on every lane entry.
            return;
        }
        self.processes.set_active_partition(partition);
        self.domains.set_active_partition(partition);
        self.contracts.set_active_partition(partition);
        self.modules.set_active_partition(partition);
        self.objects.set_active_partition(partition);
        self.continuations.set_active_partition(partition);
        self.futures.set_active_partition(partition);
        self.channels.set_active_partition(partition);
        self.collectives.set_active_partition(partition);
        for space in self.capability_spaces.values_mut() {
            space.set_active_partition(partition);
        }
        self.active_partition = partition;
    }

    /// How many consecutive epochs a runnable continuation may sit in a bin
    /// before I21 reports starvation.
    ///
    /// `docs/SOMA-v0.2.md` §4 declined to guarantee fairness beyond I14 and
    /// explicitly permitted one run class to starve another. That is
    /// defensible for a sequential interpreter, where starvation is visible in
    /// a single trace. Under territory placement and class affinity it becomes
    /// a policy outcome nobody chose, so v0.3 makes it a bound rather than an
    /// emergent property.
    pub fn set_deferral_bound(&mut self, epochs: u32) {
        self.deferral_bound = epochs;
    }

    pub fn deferral_bound(&self) -> u32 {
        self.deferral_bound
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

    pub fn root_domain(&self) -> Ref64 {
        self.root_domain
    }

    pub fn continuation_count(&self) -> usize {
        self.continuations.len()
    }

    /// An object's declared byte length.
    ///
    /// A delegated `READ`/`WRITE` capability only authorises object access when
    /// it spans the whole object (`find_authorized_capability`), so a caller
    /// delegating authority needs this to ask for a grant that will actually
    /// authorise anything.
    /// Where an object's bytes live: `"host"` for a kernel-allocated `Vec`,
    /// or whatever a foreign payload calls itself.
    ///
    /// Not semantic — nothing in the abstract machine may branch on this. It
    /// exists so a test can tell a batch that was published in place from one
    /// that was copied out and back, which is otherwise invisible: both
    /// produce identical bytes, which is the whole point.
    pub fn object_provenance(&self, object: Ref64) -> Option<&'static str> {
        self.object_payloads
            .get(&object.key())
            .map(|payload| payload.provenance())
    }

    pub fn object_byte_length(&self, object: Ref64) -> Result<u64, RuntimeError> {
        Ok(self.objects.get(object)?.byte_length)
    }

    /// How many capabilities across all spaces name `target`.
    ///
    /// Reporting only: it exists because the cost of an authorization is the
    /// length of this bucket, so a benchmark that slows down wants to be able
    /// to say whether the bucket is why.
    pub fn capabilities_naming(&self, target: Ref64) -> usize {
        self.capability_spaces
            .values()
            .map(|space| space.for_target(target).len())
            .sum()
    }

    pub fn capability_count(&self) -> usize {
        self.capability_spaces
            .values()
            .map(capability_space::CapabilitySpace::len)
            .sum()
    }

    pub fn capability_table_kind(&self) -> Kind {
        Kind::Capability
    }

    pub fn mailbox_entries(&self, process: Ref64) -> Option<&VecDeque<MessageDescriptor>> {
        self.mailboxes
            .get(&process.key())
            .map(|mailbox| &mailbox.entries)
    }

    pub fn mailbox_full_waiter_count(&self, process: Ref64) -> usize {
        self.mailboxes
            .get(&process.key())
            .map(|mailbox| mailbox.full_waiters.len())
            .unwrap_or(0)
    }

    pub fn mailbox_first_full_waiter(&self, process: Ref64) -> Option<Ref64> {
        self.mailboxes
            .get(&process.key())
            .and_then(|mailbox| mailbox.full_waiters.front().copied())
    }

    pub fn trace_events(&self) -> &[TraceEvent] {
        &self.trace
    }

    // ---- log retention (see `kernel::retention`) -------------------------

    /// Choose how long the append-only logs are kept.
    ///
    /// `Retain` is the default and is what every whole-run check requires. Set
    /// `PerEpoch` only for a run whose logs are drained between epochs; the
    /// records it discards are counted, not forgotten.
    pub fn set_log_retention(&mut self, retention: retention::LogRetention) {
        self.retention = retention;
    }

    pub fn log_retention(&self) -> retention::LogRetention {
        self.retention
    }

    /// Drain the trace, transferring the records to the caller.
    ///
    /// Draining is what makes `PerEpoch` lossless: a taken record is the
    /// consumer's responsibility and is counted as taken rather than dropped.
    pub fn take_trace_events(&mut self) -> Vec<TraceEvent> {
        let taken = std::mem::take(&mut self.trace);
        self.trace_counters.take(taken.len());
        taken
    }

    pub fn take_effect_log(&mut self) -> Vec<crate::kernel::effects::EffectRecord> {
        let taken = std::mem::take(&mut self.effect_log);
        self.effect_counters.take(taken.len());
        taken
    }

    pub fn take_admission_log(&mut self) -> Vec<crate::scheduler::admission::AdmissionRecord> {
        let taken = std::mem::take(&mut self.admission_log);
        self.admission_counters.take(taken.len());
        taken
    }

    /// What became of every record each log produced.
    ///
    /// `emitted == retained + taken + dropped` holds for each log. A run that
    /// means to have seen everything should assert `is_complete()` rather than
    /// assume its draining kept up.
    pub fn log_accounting(&self) -> retention::LogAccounting {
        retention::LogAccounting {
            trace: self.trace_counters.census(self.trace.len()),
            effects: self.effect_counters.census(self.effect_log.len()),
            admissions: self.admission_counters.census(self.admission_log.len()),
        }
    }

    /// Discard whatever the logs still hold, counting it as dropped. Called at
    /// the epoch boundary under `PerEpoch`; a no-op under `Retain`.
    pub(crate) fn release_epoch_logs(&mut self) {
        if !self.retention.is_bounded() {
            return;
        }
        self.trace_counters.drop_all(self.trace.len());
        self.trace.clear();
        self.effect_counters.drop_all(self.effect_log.len());
        self.effect_log.clear();
        self.admission_counters.drop_all(self.admission_log.len());
        self.admission_log.clear();
    }

    /// Every epoch's admission, in epoch order: the candidates offered and the
    /// decision taken over them.
    ///
    /// The decision is meant to be a function of the candidate set and nothing
    /// else (v0.3 §4). Keeping both is what lets `semantics::schedule` check
    /// that over a whole run, and check that the run made the decision the rule
    /// specifies rather than one of its own.
    pub fn effect_log(&self) -> &[crate::kernel::effects::EffectRecord] {
        &self.effect_log
    }

    pub fn admission_log(&self) -> &[crate::scheduler::admission::AdmissionRecord] {
        &self.admission_log
    }

    pub(crate) fn epoch_number(&self) -> u32 {
        self.epoch
    }

    pub(crate) fn processes(&self) -> &GenTable<ProcessDescriptor> {
        &self.processes
    }

    pub(crate) fn domains(&self) -> &GenTable<DomainDescriptor> {
        &self.domains
    }

    pub(crate) fn contracts(&self) -> &GenTable<ExecutionContract> {
        &self.contracts
    }

    pub(crate) fn modules(&self) -> &GenTable<ModuleDescriptor> {
        &self.modules
    }

    pub(crate) fn objects(&self) -> &GenTable<ObjectDescriptor> {
        &self.objects
    }

    pub(crate) fn capability_spaces(&self) -> &HashMap<u64, capability_space::CapabilitySpace> {
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

    pub(crate) fn channels(&self) -> &GenTable<ChannelDescriptor> {
        &self.channels
    }

    pub(crate) fn collectives(&self) -> &GenTable<CollectiveDescriptor> {
        &self.collectives
    }

    pub(crate) fn channel_queue_snapshots(&self) -> Vec<ChannelQueueSnapshot> {
        self.channels
            .iter()
            .filter_map(|(channel, _)| {
                self.channel_queues
                    .get(&channel.key())
                    .map(|queue| ChannelQueueSnapshot {
                        channel,
                        entries: queue
                            .entries
                            .iter()
                            .map(|entry| {
                                (
                                    entry.descriptor.payload,
                                    entry.descriptor.sender_sequence,
                                    entry.payload_authority.target,
                                )
                            })
                            .collect(),
                        send_waiters: queue.send_waiters.iter().copied().collect(),
                        receive_waiters: queue.receive_waiters.iter().copied().collect(),
                    })
            })
            .collect()
    }

    pub(crate) fn mailboxes(&self) -> &HashMap<u64, Mailbox> {
        &self.mailboxes
    }

    pub(crate) fn future_waiters(&self) -> &HashMap<u64, Vec<Ref64>> {
        &self.future_waiters
    }

    pub(crate) fn supervision_queues(&self) -> &HashMap<u64, SupervisionQueue> {
        &self.supervision_queues
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
        self.trace_caused(
            event_kind,
            process,
            continuation,
            run_class,
            auxiliary,
            Ref64::NULL,
        );
    }

    /// Append a trace event that is causally related to another event through
    /// `causal` — the future that woke a continuation, the channel or receiver
    /// a message crossed, the child a notice reports.
    ///
    /// Only the events that participate in a cross-entity happens-before edge
    /// need this. Everything else is ordered by the continuation it belongs to
    /// and by its epoch, which `trace` already records.
    fn trace_caused(
        &mut self,
        event_kind: EventKind,
        process: Ref64,
        continuation: Ref64,
        run_class: u32,
        auxiliary: u32,
        causal: Ref64,
    ) {
        self.trace_full(
            event_kind,
            process,
            continuation,
            run_class,
            auxiliary,
            causal,
            Ref64::NULL,
        );
    }

    /// Append a trace event that is *about* a second entity — the future being
    /// awaited, the value a future took, the process a restart replaced.
    ///
    /// `subject` exists so that no entity is ever recorded as a bare slot
    /// number. A slot alone carries no kind and no generation, so a checker
    /// comparing two runs that name their entities differently cannot
    /// translate it, and cannot even tell it apart from a sequence number.
    pub(crate) fn trace_about(
        &mut self,
        event_kind: EventKind,
        process: Ref64,
        continuation: Ref64,
        run_class: u32,
        subject: Ref64,
    ) {
        self.trace_full(
            event_kind,
            process,
            continuation,
            run_class,
            0,
            Ref64::NULL,
            subject,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_full(
        &mut self,
        event_kind: EventKind,
        process: Ref64,
        continuation: Ref64,
        run_class: u32,
        auxiliary: u32,
        causal: Ref64,
        subject: Ref64,
    ) {
        self.logical_time = self.logical_time.wrapping_add(1);
        let (lane, lane_sequence) = self.next_position();
        self.trace_counters.emit();
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
            last.causal = causal;
            last.subject = subject;
            last.lane = lane;
            last.lane_sequence = lane_sequence;
        }
    }

    /// Take the next `(lane, lane_sequence)` for an event.
    ///
    /// Nothing shared is consulted beyond the counter belonging to whoever is
    /// emitting, which is the point: a concurrent implementation runs this per
    /// lane with no coordination, and the epoch's total order is recovered by
    /// sorting on `TraceEvent::position` afterwards (I23).
    fn next_position(&mut self) -> (u32, u32) {
        if self.current_lane == crate::abi::traces::HOST_LANE {
            let sequence = self.host_sequence;
            self.host_sequence = self.host_sequence.saturating_add(1);
            (crate::abi::traces::HOST_LANE, sequence)
        } else {
            let sequence = self.lane_sequence;
            self.lane_sequence = self.lane_sequence.saturating_add(1);
            (self.current_lane, sequence)
        }
    }

    /// Bind subsequent trace emissions to `lane` of the current epoch.
    ///
    /// Lanes are numbered from 1 in the epoch's admitted order. The number is a
    /// position in the plan, decided before anything runs, so a concurrent
    /// executive assigns it the same way a sequential one does.
    pub(crate) fn enter_lane(&mut self, lane: u32) {
        debug_assert_ne!(lane, crate::abi::traces::HOST_LANE);
        self.current_lane = lane;
        self.lane_sequence = 0;
        self.lane_effect_sequence = 0;
        let partition = ((lane - 1) % self.allocation_partitions as u32) as u8;
        self.set_active_partition(partition);
    }

    /// Return emission to the host: epoch bookkeeping and anything a caller
    /// does between epochs.
    pub(crate) fn leave_lane(&mut self) {
        self.current_lane = crate::abi::traces::HOST_LANE;
        self.set_active_partition(0);
    }

    /// Open a new epoch's position space, at the moment the epoch number
    /// advances rather than when the next epoch starts running.
    ///
    /// Positions are meaningful only within an epoch, and a caller that creates
    /// work *between* epochs already stamps its events with the new epoch
    /// number. Resetting later would restart the host counter underneath those
    /// events and sort the epoch's own bookkeeping ahead of work that preceded
    /// it.
    pub(crate) fn open_epoch_positions(&mut self) {
        self.current_lane = crate::abi::traces::HOST_LANE;
        self.lane_sequence = 0;
        self.host_sequence = 0;
        self.lane_effect_sequence = 0;
        self.host_effect_sequence = 0;
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
                lane: t.lane,
                lane_sequence: t.lane_sequence,
                event_kind: t.event_kind,
                engine: t.engine,
                process: t.process.to_u64(),
                continuation: t.continuation.to_u64(),
                run_class: t.run_class,
                auxiliary: t.auxiliary,
                subject: t.subject.to_u64(),
                causal: t.causal.to_u64(),
            })
            .collect()
    }

    // ---- objects ---------------------------------------------------------

    pub fn create_object(&mut self, actor: Ref64, kind: ObjectKind, bytes: Vec<u8>) -> Ref64 {
        self.create_object_for(actor, kind, bytes)
    }

    /// Create an object whose bytes the kernel did not allocate.
    ///
    /// The caller transfers ownership of the allocation; see
    /// `kernel::payload` for what that obliges it to stop doing.
    pub fn create_object_from_payload(
        &mut self,
        actor: Ref64,
        kind: ObjectKind,
        bytes: payload::Payload,
    ) -> Ref64 {
        self.create_object_for(actor, kind, bytes)
    }

    fn create_object_for(
        &mut self,
        actor: Ref64,
        kind: ObjectKind,
        bytes: impl Into<payload::Payload>,
    ) -> Ref64 {
        let bytes = bytes.into();
        let owner_domain = if actor == SYSTEM_PRINCIPAL {
            self.root_domain
        } else {
            self.processes
                .get(actor)
                .map(|process| process.domain)
                .unwrap_or(self.root_domain)
        };
        let r = self
            .objects
            .alloc(ObjectDescriptor::new(kind, bytes.len() as u64));
        let byte_length = bytes.len() as u64;
        self.object_payloads.insert(r.key(), bytes);
        {
            let o = self.objects.get_mut(r).expect("fresh object");
            o.id = r;
            o.owner_domain = owner_domain;
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
        let active = self.active_partition;
        let space = self
            .capability_spaces
            .entry(actor.key())
            .or_insert_with(|| capability_space::CapabilitySpace::new(Kind::Capability));
        space.set_active_partition(active);
        space.alloc(entry)
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
            .get_mut(&actor.key())
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
        self.capability_spaces
            .get(&actor.key())?
            .iter()
            .find_map(|(r, cap)| {
                (cap.target == target && cap.rights & rights == rights).then_some(r)
            })
    }

    pub fn capability_entry(
        &self,
        actor: Ref64,
        capability: Ref64,
    ) -> Result<&CapabilityEntry, RuntimeError> {
        self.capability_spaces
            .get(&actor.key())
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
            .get(&actor.key())
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
            .get_mut(&receiver.key())
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

    fn find_authorized_capability(&self, actor: Ref64, right: u32, target: Ref64) -> Option<Ref64> {
        let space = self.capability_spaces.get(&actor.key())?;
        let object_metadata = if target.kind == Kind::Object {
            self.objects
                .get(target)
                .map(|object| (object.version, object.byte_length))
                .ok()
        } else {
            None
        };
        space
            .for_target(target)
            .into_iter()
            .find_map(|(capability_ref, capability)| {
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
        space: &capability_space::CapabilitySpace,
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
                self.find_authorized_capability(Ref64::process_at(**holder), right, target)
                    .is_some()
            })
            .count()
    }

    pub(super) fn revoke_target_right(&mut self, target: Ref64, right: u32) {
        for space in self.capability_spaces.values_mut() {
            // By target rather than over the whole space: every freeze revokes
            // WRITE on the object it is freezing, so this runs once per
            // published batch and a full scan here made publication linear in
            // every capability the run had ever minted.
            let roots: Vec<Ref64> = space
                .for_target(target)
                .into_iter()
                .filter_map(|(capability, entry)| {
                    (entry.rights & right == right).then_some(capability)
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
            .entry(actor.key())
            .or_insert_with(|| capability_space::CapabilitySpace::new(Kind::Capability))
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
            .get(&actor.key())
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .get(source_ref)?
            .clone();
        exported.parent_capability = Ref64::NULL;

        if actor != receiver {
            self.authority_effect(actor, crate::abi::Rights::TRANSFER, target);
            self.capability_spaces
                .get_mut(&receiver.key())
                .ok_or(RuntimeError::MissingCapabilitySpace)?
                .alloc(exported);
            let space = self
                .capability_spaces
                .get_mut(&actor.key())
                .ok_or(RuntimeError::MissingCapabilitySpace)?;
            Self::revoke_capability_tree(space, source_ref);
        }
        Ok(())
    }

    fn revoke_capability_tree(space: &mut capability_space::CapabilitySpace, root: Ref64) {
        let mut revoked = vec![root];
        let mut index = 0;
        while index < revoked.len() {
            let parent = revoked[index];
            for child in space.children_of(parent) {
                if !revoked.contains(&child) {
                    revoked.push(child);
                }
            }
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
            .get(&obj.key())
            .map(|v| v.as_slice())
            .ok_or(RuntimeError::MissingPayload)
    }

    /// The bytes of several objects at once.
    ///
    /// `object_bytes` takes `&mut self` — authorization records an effect —
    /// so its result borrows the kernel mutably and only one such slice can
    /// be alive at a time. An epoch that wants to hand a backend every ready
    /// batch in one submission needs all of them alive together. Authorizing
    /// each in turn and only then reborrowing shared gives that without
    /// copying: the mutable phase finishes before the first slice exists.
    ///
    /// Fails as a whole if any object is unreadable, so a partially
    /// authorized epoch is not a state a caller can reach.
    pub fn object_bytes_many(
        &mut self,
        actor: Ref64,
        objects: &[Ref64],
    ) -> Result<Vec<&[u8]>, RuntimeError> {
        for object in objects {
            self.authorize(actor, crate::abi::Rights::READ, *object)?;
            let _ = self.objects.get(*object)?;
            if !self.object_payloads.contains_key(&object.key()) {
                return Err(RuntimeError::MissingPayload);
            }
        }
        let payloads = &self.object_payloads;
        objects
            .iter()
            .map(|object| {
                payloads
                    .get(&object.key())
                    .map(|payload| payload.as_slice())
                    .ok_or(RuntimeError::MissingPayload)
            })
            .collect()
    }

    pub fn object_bytes_mut(
        &mut self,
        actor: Ref64,
        obj: Ref64,
    ) -> Result<&mut [u8], RuntimeError> {
        if self.objects.get(obj)?.object_kind == ObjectKind::ProcessState {
            return Err(RuntimeError::InvalidStateAccess);
        }
        self.authorize(actor, crate::abi::Rights::WRITE, obj)?;
        let _ = self.objects.get(obj)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, obj);
        self.object_payloads
            .get_mut(&obj.key())
            .map(|v| v.as_mut_slice())
            .ok_or(RuntimeError::MissingPayload)
    }

    /// A host payload as the growable `Vec` it is, for the one caller that
    /// legitimately replaces a frame with a longer or shorter one.
    ///
    /// Growth cannot go through `object_bytes_mut`, which now hands out a
    /// slice because a payload is not always a `Vec`. It also cannot go
    /// through anything that updates `ObjectDescriptor::byte_length` to match:
    /// authorization at `find_authorized_capability` admits a write only when
    /// `capability.length >= byte_length`, and a capability carries the length
    /// the object had when it was minted. Growing the descriptor therefore
    /// revokes every capability over the object, which is what made the
    /// `Expand` machine stop replying when this was written the obvious way.
    ///
    /// So the payload may exceed the length authorization checks against, and
    /// this method preserves that rather than quietly changing what a
    /// capability covers. That is a real inconsistency in the object model and
    /// it predates the payload split; it wants deciding, not papering over.
    pub fn host_payload_mut(
        &mut self,
        actor: Ref64,
        obj: Ref64,
    ) -> Result<&mut Vec<u8>, RuntimeError> {
        if self.objects.get(obj)?.object_kind == ObjectKind::ProcessState {
            return Err(RuntimeError::InvalidStateAccess);
        }
        self.authorize(actor, crate::abi::Rights::WRITE, obj)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, obj);
        self.object_payloads
            .get_mut(&obj.key())
            .and_then(|v| v.as_mut_vec())
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
        // Process state is created here and never handed to a backend, so it
        // is always host-backed; a foreign payload would report as missing
        // rather than silently refusing to grow.
        self.object_payloads
            .get_mut(&state.key())
            .and_then(|v| v.as_mut_vec())
            .ok_or(RuntimeError::MissingPayload)
    }

    // ---- processes -------------------------------------------------------

    pub fn create_process(&mut self, actor: Ref64, mode: ProcessMode) -> Ref64 {
        let domain = if actor == SYSTEM_PRINCIPAL {
            self.root_domain
        } else {
            self.processes
                .get(actor)
                .map(|process| process.domain)
                .unwrap_or(self.root_domain)
        };
        self.allocate_process(actor, domain, mode)
            .expect("default domain is unbounded and resolves")
    }

    pub fn create_process_in_domain(
        &mut self,
        actor: Ref64,
        domain: Ref64,
        mode: ProcessMode,
    ) -> Result<Ref64, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::WRITE, domain)?;
        let descriptor = self.domains.get(domain)?;
        if descriptor.max_processes != 0 && descriptor.processes_created >= descriptor.max_processes
        {
            return Err(RuntimeError::DomainQuotaExceeded);
        }
        self.authorize(actor, crate::abi::Rights::WRITE, domain)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, domain);
        self.allocate_process(actor, domain, mode)
    }

    fn allocate_process(
        &mut self,
        actor: Ref64,
        domain: Ref64,
        mode: ProcessMode,
    ) -> Result<Ref64, RuntimeError> {
        let domain_descriptor = self.domains.get(domain)?;
        if domain_descriptor.max_processes != 0
            && domain_descriptor.processes_created >= domain_descriptor.max_processes
        {
            return Err(RuntimeError::DomainQuotaExceeded);
        }
        let r = self.processes.alloc(ProcessDescriptor::new(mode));
        {
            let process = self.processes.get_mut(r).expect("fresh process");
            process.id = r;
            process.domain = domain;
            process.inbox = r;
            process.status = ProcessState::Created as u32;
        }
        self.capability_spaces.insert(
            r.key(),
            capability_space::CapabilitySpace::new(Kind::Capability),
        );
        self.mint_genesis(r, r, 0, 0);
        self.mint_genesis(r, domain, 0, 0);
        let state_obj = self.create_object_for(r, ObjectKind::ProcessState, Vec::new());
        if actor != r {
            self.mint_genesis(actor, r, 0, 0);
        }
        self.mailboxes.insert(r.key(), Mailbox::new(8));
        self.supervision_queues
            .insert(r.key(), SupervisionQueue::default());
        {
            let p = self.processes.get_mut(r).expect("fresh process");
            p.state = state_obj;
        }
        let domain_descriptor = self.domains.get_mut(domain)?;
        domain_descriptor.processes_created = domain_descriptor.processes_created.saturating_add(1);
        self.trace(EventKind::ProcessCreated, r, Ref64::NULL, 0, mode as u32);
        Ok(r)
    }

    // ---- domains ---------------------------------------------------------

    pub fn create_domain(
        &mut self,
        actor: Ref64,
        parent: Ref64,
        max_processes: u32,
    ) -> Result<Ref64, RuntimeError> {
        let _ = self.domains.get(parent)?;
        self.authorize(actor, crate::abi::Rights::WRITE, parent)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, parent);
        let domain = self
            .domains
            .alloc(DomainDescriptor::new(parent, max_processes));
        self.domains.get_mut(domain)?.id = domain;
        self.mint_genesis(actor, domain, 0, 0);
        self.trace(EventKind::DomainCreated, actor, domain, 0, max_processes);
        Ok(domain)
    }

    pub fn process_domain(&self, process: Ref64) -> Result<Ref64, RuntimeError> {
        Ok(self.processes.get(process)?.domain)
    }

    pub fn domain_processes_created(&self, domain: Ref64) -> Result<u32, RuntimeError> {
        Ok(self.domains.get(domain)?.processes_created)
    }

    /// Create a direct child whose terminal outcome is reliably reported to
    /// `supervisor`. Only the supervisor itself (or the system principal) may
    /// establish the relationship.
    pub fn create_supervised_process(
        &mut self,
        actor: Ref64,
        supervisor: Ref64,
        mode: ProcessMode,
    ) -> Result<Ref64, RuntimeError> {
        self.create_supervised_process_with_policy(
            actor,
            supervisor,
            mode,
            SupervisionPolicy::Notify,
        )
    }

    pub fn create_supervised_process_with_policy(
        &mut self,
        actor: Ref64,
        supervisor: Ref64,
        mode: ProcessMode,
        policy: SupervisionPolicy,
    ) -> Result<Ref64, RuntimeError> {
        if policy == SupervisionPolicy::Restart {
            return Err(RuntimeError::InvalidSupervisionPolicy);
        }
        if actor != SYSTEM_PRINCIPAL && actor != supervisor {
            return Err(RuntimeError::AuthorityDenied);
        }
        if matches!(
            self.process_state(supervisor)?,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return Err(RuntimeError::ProcessUnavailable);
        }
        let domain = self.processes.get(supervisor)?.domain;
        let child = self.allocate_process(actor, domain, mode)?;
        let descriptor = self.processes.get_mut(child)?;
        descriptor.supervisor = supervisor;
        descriptor.supervision_policy = policy;
        Ok(child)
    }

    /// Create a supervised process that is replaced with a fresh generational
    /// identity and the same entry continuation when it fails. `restart_limit`
    /// counts replacements; exhausting it escalates failure to the supervisor.
    pub fn create_restartable_process(
        &mut self,
        actor: Ref64,
        supervisor: Ref64,
        mode: ProcessMode,
        restart_limit: u32,
        entry: ContinuationSpec,
    ) -> Result<Ref64, RuntimeError> {
        if restart_limit == 0
            || (mode == ProcessMode::Pure && entry.state_access == crate::abi::StateAccess::Mutable)
        {
            return Err(RuntimeError::InvalidSupervisionPolicy);
        }
        let child = self.create_supervised_process(actor, supervisor, mode)?;
        {
            let descriptor = self.processes.get_mut(child)?;
            descriptor.supervision_policy = SupervisionPolicy::Restart;
            descriptor.restart_limit = restart_limit;
        }
        self.restart_blueprints.insert(
            child.key(),
            RestartBlueprint {
                entry: entry.clone(),
            },
        );
        self.create_continuation(child, child, entry)?;
        Ok(child)
    }

    pub fn process_restart_lineage(
        &self,
        process: Ref64,
    ) -> Result<(Ref64, u32, u32), RuntimeError> {
        let descriptor = self.processes.get(process)?;
        Ok((
            descriptor.restart_of,
            descriptor.restart_attempt,
            descriptor.restart_limit,
        ))
    }

    pub(crate) fn has_restart_blueprint(&self, process: Ref64) -> bool {
        self.restart_blueprints.contains_key(&process.key())
    }

    pub fn process_supervisor(&self, process: Ref64) -> Result<Ref64, RuntimeError> {
        Ok(self.processes.get(process)?.supervisor)
    }

    /// Receive the oldest direct-child exit notice, or register `continuation`
    /// to be awakened by the next one. The supervisor's RECEIVE authority over
    /// its own process gates access to this kernel control queue.
    pub fn receive_supervision(
        &mut self,
        actor: Ref64,
        continuation: Ref64,
    ) -> Result<Option<SupervisionNotice>, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::RECEIVE, actor)?;
        if !continuation.is_null() && self.continuations.get(continuation)?.process != actor {
            return Err(RuntimeError::InvalidStateAccess);
        }
        self.authority_effect(actor, crate::abi::Rights::RECEIVE, actor);
        let queue = self
            .supervision_queues
            .get_mut(&actor.key())
            .ok_or(RuntimeError::MissingMailbox)?;
        if let Some(notice) = queue.notices.pop_front() {
            return Ok(Some(notice));
        }
        if !continuation.is_null() && !queue.waiters.contains(&continuation) {
            queue.waiters.push_back(continuation);
        }
        Ok(None)
    }

    pub fn pending_supervision_notices(&self, supervisor: Ref64) -> usize {
        self.supervision_queues
            .get(&supervisor.key())
            .map(|queue| queue.notices.len())
            .unwrap_or(0)
    }

    pub(super) fn notify_supervisor(&mut self, child: Ref64, reason: ExitReason) {
        let Some((supervisor, failure_count, policy)) =
            self.processes.get(child).ok().map(|process| {
                (
                    process.supervisor,
                    process.failure_count,
                    process.supervision_policy,
                )
            })
        else {
            return;
        };
        if supervisor.is_null() {
            return;
        }
        let replacement = if reason == ExitReason::Failed && policy == SupervisionPolicy::Restart {
            self.restart_process(child).unwrap_or(Ref64::NULL)
        } else {
            Ref64::NULL
        };
        let waiter = self
            .supervision_queues
            .get_mut(&supervisor.key())
            .and_then(|queue| {
                queue.notices.push_back(SupervisionNotice {
                    child,
                    replacement,
                    reason,
                    failure_count,
                });
                queue.waiters.pop_front()
            });
        if let Some(waiter) = waiter {
            self.wake_waiting_continuation(waiter);
        }
        self.trace(
            EventKind::SupervisionNotified,
            supervisor,
            child,
            0,
            reason as u32,
        );
        if reason == ExitReason::Failed && policy == SupervisionPolicy::Escalate {
            self.fail_process_from_supervision(supervisor, child);
        }
        if reason == ExitReason::Failed
            && policy == SupervisionPolicy::Restart
            && replacement.is_null()
        {
            self.fail_process_from_supervision(supervisor, child);
        }
    }

    fn restart_process(&mut self, failed: Ref64) -> Option<Ref64> {
        let failed_descriptor = self.processes.get(failed).ok()?.clone();
        if failed_descriptor.restart_attempt >= failed_descriptor.restart_limit {
            return None;
        }
        let blueprint = self.restart_blueprints.get(&failed.key())?.clone();
        let replacement = self
            .allocate_process(
                failed_descriptor.supervisor,
                failed_descriptor.domain,
                failed_descriptor.process_mode,
            )
            .ok()?;
        {
            let descriptor = self.processes.get_mut(replacement).ok()?;
            descriptor.supervisor = failed_descriptor.supervisor;
            descriptor.supervision_policy = SupervisionPolicy::Restart;
            descriptor.restart_of = failed;
            descriptor.restart_attempt = failed_descriptor.restart_attempt + 1;
            descriptor.restart_limit = failed_descriptor.restart_limit;
            descriptor.base_priority = failed_descriptor.base_priority;
            descriptor.compute_quota = failed_descriptor.compute_quota;
            descriptor.memory_quota = failed_descriptor.memory_quota;
            descriptor.deadline_ns = failed_descriptor.deadline_ns;
        }
        self.restart_blueprints
            .insert(replacement.key(), blueprint.clone());
        if self
            .create_continuation(replacement, replacement, blueprint.entry)
            .is_err()
        {
            return None;
        }
        self.trace_about(
            EventKind::ProcessRestarted,
            failed_descriptor.supervisor,
            replacement,
            0,
            failed,
        );
        Some(replacement)
    }

    fn fail_process_from_supervision(&mut self, process: Ref64, failed_child: Ref64) {
        let Ok(status) = self.process_state(process) else {
            return;
        };
        if matches!(
            status,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return;
        }
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.status = ProcessState::Failed as u32;
            descriptor.failure_count = descriptor.failure_count.wrapping_add(1);
        }
        self.contain_process_failure(process, Ref64::NULL);
        self.trace(EventKind::ProcessFailed, process, failed_child, 0, 1);
        self.notify_supervisor(process, ExitReason::Failed);
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
            8 => ProcessState::Cancelled,
            _ => ProcessState::Created,
        })
    }

    pub fn cancel_process(&mut self, actor: Ref64, process: Ref64) -> Result<(), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::WRITE, process)?;
        let status = self.process_state(process)?;
        if matches!(
            status,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return Err(RuntimeError::ProcessUnavailable);
        }
        self.authority_effect(actor, crate::abi::Rights::WRITE, process);
        let active = {
            let descriptor = self.processes.get_mut(process)?;
            descriptor.status = ProcessState::CancelPending as u32;
            descriptor.active_continuation
        };
        if active.is_null() {
            self.finalize_process_cancellation(process);
        }
        Ok(())
    }

    pub(super) fn contain_process_failure(&mut self, process: Ref64, faulted: Ref64) {
        self.cancel_process_continuations(process, Some(faulted));
        self.settle_owned_futures(process, FutureState::Failed);
        self.settle_owned_collectives(process, CollectiveState::Failed);
        self.drain_terminal_mailbox(process);
        self.capability_spaces.remove(&process.key());
    }

    pub(super) fn finalize_process_cancellation(&mut self, process: Ref64) {
        self.cancel_process_continuations(process, None);
        self.settle_owned_futures(process, FutureState::Cancelled);
        self.settle_owned_collectives(process, CollectiveState::Cancelled);
        self.drain_terminal_mailbox(process);
        self.capability_spaces.remove(&process.key());
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.status = ProcessState::Cancelled as u32;
            descriptor.active_continuation = Ref64::NULL;
        }
        self.trace(EventKind::ProcessCancelled, process, Ref64::NULL, 0, 0);
        self.notify_supervisor(process, ExitReason::Cancelled);
    }

    fn cancel_process_continuations(&mut self, process: Ref64, except: Option<Ref64>) {
        let cancelled: Vec<Ref64> = self
            .continuations
            .iter()
            .filter_map(|(continuation, descriptor)| {
                (descriptor.process == process
                    && Some(continuation) != except
                    && matches!(
                        descriptor.status,
                        crate::abi::continuations::ContinuationState::New
                            | crate::abi::continuations::ContinuationState::Runnable
                            | crate::abi::continuations::ContinuationState::Waiting
                            | crate::abi::continuations::ContinuationState::Running
                    ))
                .then_some(continuation)
            })
            .collect();

        // Cancellation empties the bins of everything it cancels, so it has to
        // see the whole lane, not the part of it that has landed. Applying the
        // journal first is what makes that true, and it leaves the state
        // exactly where performing the effects inline used to: the entry is
        // made, then removed. The alternative — withdrawing pending effects
        // instead — would leave the same state by a path no handler in this
        // executive can currently reach, and so by a path nothing tests.
        self.apply_lane_effects();

        for continuation in &cancelled {
            self.scheduler.remove(*continuation);
            self.set_continuation_status(
                *continuation,
                crate::abi::continuations::ContinuationState::Cancelled,
            );
            self.trace(
                EventKind::ContinuationCancelled,
                process,
                *continuation,
                0,
                0,
            );
        }

        for waiters in self.future_waiters.values_mut() {
            waiters.retain(|waiter| !cancelled.contains(waiter));
        }
        for mailbox in self.mailboxes.values_mut() {
            mailbox
                .recv_waiters
                .retain(|waiter| !cancelled.contains(waiter));
            mailbox
                .full_waiters
                .retain(|waiter| !cancelled.contains(waiter));
        }
        for channel in self.channel_queues.values_mut() {
            channel
                .send_waiters
                .retain(|waiter| !cancelled.contains(waiter));
            channel
                .receive_waiters
                .retain(|waiter| !cancelled.contains(waiter));
        }
        for queue in self.supervision_queues.values_mut() {
            queue.waiters.retain(|waiter| !cancelled.contains(waiter));
        }
    }

    fn settle_owned_futures(&mut self, process: Ref64, terminal: FutureState) {
        let futures: Vec<Ref64> = self
            .futures
            .iter()
            .filter_map(|(future, descriptor)| {
                (descriptor.owner_process == process && descriptor.state == FutureState::Pending)
                    .then_some(future)
            })
            .collect();

        for future in futures {
            if let Ok(descriptor) = self.futures.get_mut(future) {
                descriptor.state = terminal;
                descriptor.resolved_epoch = self.epoch;
                if terminal == FutureState::Failed {
                    descriptor.failure = process;
                }
            }
            let waiters = self
                .future_waiters
                .remove(&future.key())
                .unwrap_or_default();
            for waiter in waiters {
                let Some((waiter_process, run_class, status)) =
                    self.continuations.get(waiter).ok().map(|descriptor| {
                        (descriptor.process, descriptor.run_class, descriptor.status)
                    })
                else {
                    continue;
                };
                if waiter_process == process
                    || status != crate::abi::continuations::ContinuationState::Waiting
                {
                    continue;
                }
                self.emit(crate::kernel::effects::Effect::Wake {
                    continuation: waiter,
                    run_class,
                });
                self.trace(
                    EventKind::ContinuationReady,
                    waiter_process,
                    waiter,
                    run_class,
                    0,
                );
            }
            let event = if terminal == FutureState::Failed {
                EventKind::FutureFailed
            } else {
                EventKind::FutureCancelled
            };
            self.trace(event, process, Ref64::NULL, 0, future.slot);
        }
    }

    fn settle_owned_collectives(&mut self, process: Ref64, terminal: CollectiveState) {
        let collectives: Vec<Ref64> = self
            .collectives
            .iter()
            .filter_map(|(collective, descriptor)| {
                (descriptor.owner_process == process
                    && descriptor.state == CollectiveState::Pending)
                    .then_some(collective)
            })
            .collect();
        let event = if terminal == CollectiveState::Failed {
            EventKind::CollectiveFailed
        } else {
            EventKind::CollectiveCancelled
        };
        for collective in collectives {
            if let Ok(descriptor) = self.collectives.get_mut(collective) {
                descriptor.state = terminal;
            }
            self.trace(event, process, collective, 0, 0);
        }
    }

    fn drain_terminal_mailbox(&mut self, process: Ref64) {
        let full_waiters = if let Some(mailbox) = self.mailboxes.get_mut(&process.key()) {
            mailbox.entries.clear();
            mailbox.recv_waiters.clear();
            std::mem::take(&mut mailbox.full_waiters)
        } else {
            VecDeque::new()
        };

        for waiter in full_waiters {
            let Some((waiter_process, run_class, status)) =
                self.continuations.get(waiter).ok().map(|descriptor| {
                    (descriptor.process, descriptor.run_class, descriptor.status)
                })
            else {
                continue;
            };
            if waiter_process == process
                || status != crate::abi::continuations::ContinuationState::Waiting
            {
                continue;
            }
            self.emit(crate::kernel::effects::Effect::Wake {
                continuation: waiter,
                run_class,
            });
            self.trace(
                EventKind::ContinuationReady,
                waiter_process,
                waiter,
                run_class,
                0,
            );
        }
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
        if matches!(
            self.process_state(process)?,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return Err(RuntimeError::ProcessUnavailable);
        }
        if self.processes.get(process)?.process_mode == ProcessMode::Pure
            && spec.state_access == crate::abi::StateAccess::Mutable
        {
            return Err(RuntimeError::InvalidStateAccess);
        }
        self.authority_effect(actor, crate::abi::Rights::WRITE, process);
        let frame_obj =
            self.create_object_for(process, ObjectKind::ContinuationFrame, spec.frame_bytes);
        let r = self
            .continuations
            .alloc(crate::abi::continuations::ContinuationDescriptor::new(
                process,
                spec.state_access,
                spec.run_class,
                spec.resume_point,
            ));
        {
            let c = self.continuations.get_mut(r).expect("fresh continuation");
            c.id = r;
            c.frame = frame_obj;
            c.remaining_steps = spec.max_steps;
            c.status = crate::abi::continuations::ContinuationState::Runnable;
            c.created_epoch = self.epoch;
        }
        // `New` and `Runnable` are both live, so a fresh continuation is
        // counted exactly once, here.
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.live_continuations = descriptor.live_continuations.saturating_add(1);
        }
        self.emit(crate::kernel::effects::Effect::Bin {
            continuation: r,
            run_class: spec.run_class,
        });
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

    /// The single path by which an existing continuation's status changes.
    ///
    /// Routing every transition through one function is what lets
    /// `ProcessDescriptor::live_continuations` stay correct without the
    /// table scan `retire_process_if_idle` used to perform. A status write
    /// that bypasses this leaves the count stale, which I3 reports.
    pub(crate) fn set_continuation_status(
        &mut self,
        continuation: Ref64,
        status: crate::abi::continuations::ContinuationState,
    ) {
        let epoch = self.epoch;
        let Ok(descriptor) = self.continuations.get(continuation) else {
            return;
        };
        let process = descriptor.process;
        let previous = descriptor.status;
        if let Ok(descriptor) = self.continuations.get_mut(continuation) {
            descriptor.status = status;
            descriptor.last_run_epoch = epoch;
        }
        if previous.is_live() == status.is_live() {
            return;
        }
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.live_continuations = if status.is_live() {
                descriptor.live_continuations.saturating_add(1)
            } else {
                descriptor.live_continuations.saturating_sub(1)
            };
        }
    }

    /// Whether `process` still has a continuation that could run (I3).
    pub(crate) fn has_live_continuation(&self, process: Ref64) -> bool {
        self.processes
            .get(process)
            .map(|descriptor| descriptor.live_continuations > 0)
            .unwrap_or(false)
    }

    // ---- execution contracts -------------------------------------------

    pub(crate) fn contract_is_valid(contract: &ExecutionContract) -> bool {
        contract.shape == crate::abi::Shape::Scalar
            && contract.placement_policy == crate::abi::PlacementPolicy::Any
            && contract.determinism_policy == crate::abi::DeterminismPolicy::Deterministic
            && contract.minimum_parallelism == 1
            && contract.preferred_parallelism == 1
            && contract.maximum_steps > 0
            && contract.deadline_ns == 0
    }

    pub fn create_execution_contract(
        &mut self,
        actor: Ref64,
        mut contract: ExecutionContract,
    ) -> Result<Ref64, RuntimeError> {
        if !Self::contract_is_valid(&contract) {
            return Err(RuntimeError::InvalidContract);
        }
        let reference = self.contracts.alloc(contract.clone());
        contract.id = reference;
        *self.contracts.get_mut(reference)? = contract;
        self.mint_genesis(actor, reference, 0, 0);
        self.trace(
            EventKind::ContractCreated,
            actor,
            reference,
            0,
            self.contracts.get(reference)?.maximum_steps,
        );
        Ok(reference)
    }

    pub fn create_contracted_continuation(
        &mut self,
        actor: Ref64,
        process: Ref64,
        contract: Ref64,
        spec: ContinuationSpec,
    ) -> Result<Ref64, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::READ, contract)?;
        let descriptor = self.contracts.get(contract)?;
        if !Self::contract_is_valid(descriptor)
            || spec.max_steps > descriptor.maximum_steps
            || (descriptor.local_memory_bytes != 0
                && spec.frame_bytes.len() > descriptor.local_memory_bytes as usize)
        {
            return Err(RuntimeError::InvalidContract);
        }
        let continuation = self.create_continuation(actor, process, spec)?;
        self.continuations.get_mut(continuation)?.execution_contract = contract;
        self.trace_about(
            EventKind::ContractAttached,
            process,
            continuation,
            0,
            contract,
        );
        Ok(continuation)
    }

    pub fn continuation_contract(&self, continuation: Ref64) -> Result<Ref64, RuntimeError> {
        Ok(self.continuations.get(continuation)?.execution_contract)
    }

    // ---- modules ---------------------------------------------------------

    fn module_name_hash(name: &str) -> u64 {
        name.as_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
    }

    pub fn load_module(
        &mut self,
        actor: Ref64,
        name: &str,
        evaluators: &[(u32, u32)],
    ) -> Result<Ref64, RuntimeError> {
        if name.trim().is_empty() || evaluators.is_empty() {
            return Err(RuntimeError::InvalidModule);
        }
        let mut manifest = evaluators.to_vec();
        manifest.sort_unstable_by_key(|entry| entry.0);
        if manifest.iter().any(|(id, stride)| *id == 0 || *stride == 0)
            || manifest.windows(2).any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(RuntimeError::InvalidModule);
        }
        let module = self.modules.alloc(ModuleDescriptor::new(
            Self::module_name_hash(name.trim()),
            manifest.len() as u32,
        ));
        self.modules.get_mut(module)?.id = module;
        self.module_evaluators.insert(module.key(), manifest);
        self.mint_genesis(actor, module, 0, 0);
        self.trace(
            EventKind::ModuleLoaded,
            actor,
            module,
            0,
            evaluators.len() as u32,
        );
        Ok(module)
    }

    pub(crate) fn module_manifest(&self, module: Ref64) -> Option<&[(u32, u32)]> {
        self.module_evaluators.get(&module.key()).map(Vec::as_slice)
    }

    pub(crate) fn module_matches(
        &self,
        module: Ref64,
        name: &str,
        evaluators: &[(u32, u32)],
    ) -> bool {
        let Ok(descriptor) = self.modules.get(module) else {
            return false;
        };
        let mut expected = evaluators.to_vec();
        expected.sort_unstable_by_key(|entry| entry.0);
        descriptor.name_hash == Self::module_name_hash(name.trim())
            && self.module_manifest(module) == Some(expected.as_slice())
    }

    // ---- futures ---------------------------------------------------------

    pub fn create_future(&mut self, actor: Ref64) -> Ref64 {
        let r = self.futures.alloc(FutureDescriptor::new());
        {
            let f = self.futures.get_mut(r).expect("fresh future");
            f.id = r;
            f.owner_process = actor;
        }
        self.mint_genesis(actor, r, 0, 0);
        r
    }

    /// Register `cont` as a waiter on `future`, moving it to run class
    /// `next_run_class` and the WAITING state. It is woken by `resolve_future`.
    ///
    /// If the future is *already* resolved, nothing is registered and
    /// `AlreadySettled` is returned: the caller must yield rather than await,
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
        let future_state = self.futures.get(future)?.state;
        let settled = future_state != FutureState::Pending;
        let _ = self.continuations.get(cont)?;
        self.authority_effect(actor, crate::abi::Rights::AWAIT, future);
        {
            let c = self.continuations.get_mut(cont)?;
            c.run_class = next_run_class;
            c.dependency = future;
            if !settled {
                c.status = crate::abi::continuations::ContinuationState::Waiting;
            }
        }
        if settled {
            return Ok(AwaitOutcome::AlreadySettled(future_state));
        }
        self.future_waiters
            .entry(future.key())
            .or_default()
            .push(cont);
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
        let waiters = self
            .future_waiters
            .remove(&future.key())
            .unwrap_or_default();
        let owner = self.futures.get(future)?.owner_process;
        for w in waiters {
            let (process, rc) = {
                let c = self.continuations.get(w).ok();
                match c {
                    Some(c) => (c.process, c.run_class),
                    None => continue,
                }
            };
            self.emit(crate::kernel::effects::Effect::Wake {
                continuation: w,
                run_class: rc,
            });
            self.trace_caused(EventKind::ContinuationReady, process, w, rc, 0, future);
        }
        // §3.3 emits the wakes first and the resolution last, so the semantic
        // order runs wake ≺ resolution. See `semantics::order` — the edge is an
        // ordering constraint taken from the transition rule, not a claim about
        // which event physically caused the other.
        self.trace_full(
            EventKind::FutureResolved,
            owner,
            Ref64::NULL,
            0,
            0,
            future,
            value,
        );
        Ok(())
    }

    /// Read a resolved future's value (an object ref), or `None` if unresolved.
    pub fn future_value(&self, future: Ref64) -> Option<Ref64> {
        match self.futures.get(future) {
            Ok(f) if f.state == FutureState::Resolved => Some(f.value),
            _ => None,
        }
    }

    pub fn future_state(&self, future: Ref64) -> Result<FutureState, RuntimeError> {
        Ok(self.futures.get(future)?.state)
    }

    // ---- channels --------------------------------------------------------

    pub fn create_channel(&mut self, actor: Ref64, capacity: u32) -> Ref64 {
        let channel = self.channels.alloc(ChannelDescriptor::new(capacity.max(1)));
        self.channels.get_mut(channel).expect("fresh channel").id = channel;
        self.channel_queues
            .insert(channel.key(), ChannelQueue::new());
        self.mint_genesis(actor, channel, 0, 0);
        channel
    }

    pub fn send_channel(
        &mut self,
        actor: Ref64,
        channel: Ref64,
        payload: Ref64,
        sender_continuation: Ref64,
    ) -> Result<(), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::SEND, channel)?;
        let descriptor = self.channels.get(channel)?;
        if descriptor.closed != 0 {
            return Err(RuntimeError::ChannelClosed);
        }
        let full = self
            .channel_queues
            .get(&channel.key())
            .ok_or(RuntimeError::MissingMailbox)?
            .entries
            .len()
            >= descriptor.capacity as usize;
        if full {
            self.authority_effect(actor, crate::abi::Rights::SEND, channel);
            let queue = self
                .channel_queues
                .get_mut(&channel.key())
                .ok_or(RuntimeError::MissingMailbox)?;
            if !sender_continuation.is_null() && !queue.send_waiters.contains(&sender_continuation)
            {
                queue.send_waiters.push_back(sender_continuation);
            }
            return Err(RuntimeError::MailboxFull);
        }

        let payload_authority = self.escrow_payload_read(actor, payload)?;
        self.authorize(actor, crate::abi::Rights::SEND, channel)?;
        self.authority_effect(actor, crate::abi::Rights::SEND, channel);
        let (sequence, receiver_waiter) = {
            let queue = self
                .channel_queues
                .get_mut(&channel.key())
                .ok_or(RuntimeError::MissingMailbox)?;
            let sequence = queue.next_sequence;
            queue.next_sequence = queue.next_sequence.wrapping_add(1);
            let mut message = MessageDescriptor::new(actor, channel, payload);
            message.sender_sequence = sequence;
            message.logical_timestamp = self.logical_time;
            queue.entries.push_back(EscrowMessage {
                descriptor: message,
                payload_authority,
            });
            (sequence, queue.receive_waiters.pop_front())
        };
        if let Some(waiter) = receiver_waiter {
            self.wake_waiting_continuation(waiter);
        }
        self.trace(EventKind::ChannelSent, actor, channel, 0, sequence as u32);
        Ok(())
    }

    pub fn receive_channel(
        &mut self,
        actor: Ref64,
        channel: Ref64,
        continuation: Ref64,
    ) -> Result<Option<MessageDescriptor>, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::RECEIVE, channel)?;
        if actor != SYSTEM_PRINCIPAL
            && !continuation.is_null()
            && self.continuations.get(continuation)?.process != actor
        {
            return Err(RuntimeError::InvalidStateAccess);
        }
        let closed = self.channels.get(channel)?.closed != 0;
        let empty = self
            .channel_queues
            .get(&channel.key())
            .ok_or(RuntimeError::MissingMailbox)?
            .entries
            .is_empty();
        if closed && empty {
            return Err(RuntimeError::ChannelClosed);
        }
        self.authority_effect(actor, crate::abi::Rights::RECEIVE, channel);
        let (entry, sender_waiter) = {
            let queue = self
                .channel_queues
                .get_mut(&channel.key())
                .ok_or(RuntimeError::MissingMailbox)?;
            match queue.entries.pop_front() {
                Some(entry) => (Some(entry), queue.send_waiters.pop_front()),
                None => {
                    if !continuation.is_null() && !queue.receive_waiters.contains(&continuation) {
                        queue.receive_waiters.push_back(continuation);
                    }
                    (None, None)
                }
            }
        };
        if let Some(waiter) = sender_waiter {
            self.wake_waiting_continuation(waiter);
        }
        let Some(mut entry) = entry else {
            return Ok(None);
        };
        let transferred = self
            .capability_spaces
            .get_mut(&actor.key())
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .alloc(entry.payload_authority);
        entry.descriptor.transferred_capability = transferred;
        self.trace(
            EventKind::ChannelReceived,
            actor,
            channel,
            0,
            entry.descriptor.sender_sequence as u32,
        );
        Ok(Some(entry.descriptor))
    }

    /// Atomically receive one message from every channel in `channels`.
    /// Nothing is consumed until all inputs are ready. If any open input is
    /// empty, `continuation` is registered on those inputs and `None` is
    /// returned; retry removes stale registrations before re-evaluating the
    /// whole set.
    pub fn receive_channels_all(
        &mut self,
        actor: Ref64,
        channels: &[Ref64],
        continuation: Ref64,
    ) -> Result<Option<Vec<MessageDescriptor>>, RuntimeError> {
        if channels.is_empty() {
            return Err(RuntimeError::InvalidMultiInput);
        }
        let mut identities = channels
            .iter()
            .map(|channel| channel.to_u64())
            .collect::<Vec<_>>();
        identities.sort_unstable();
        identities.dedup();
        if identities.len() != channels.len() {
            return Err(RuntimeError::InvalidMultiInput);
        }
        if !continuation.is_null() && self.continuations.get(continuation)?.process != actor {
            return Err(RuntimeError::InvalidStateAccess);
        }
        for channel in channels {
            let _ = self.channels.get(*channel)?;
            let _ = self
                .channel_queues
                .get(&channel.key())
                .ok_or(RuntimeError::MissingMailbox)?;
            self.authorize(actor, crate::abi::Rights::RECEIVE, *channel)?;
        }
        for channel in channels {
            self.authorize(actor, crate::abi::Rights::RECEIVE, *channel)?;
            self.authority_effect(actor, crate::abi::Rights::RECEIVE, *channel);
        }

        if !continuation.is_null() {
            for queue in self.channel_queues.values_mut() {
                queue
                    .receive_waiters
                    .retain(|waiter| *waiter != continuation);
            }
        }
        let mut missing = Vec::new();
        for channel in channels {
            let descriptor = self.channels.get(*channel)?;
            let queue = self
                .channel_queues
                .get(&channel.key())
                .ok_or(RuntimeError::MissingMailbox)?;
            if queue.entries.is_empty() {
                if descriptor.closed != 0 {
                    return Err(RuntimeError::ChannelClosed);
                }
                missing.push(*channel);
            }
        }
        if !missing.is_empty() {
            if !continuation.is_null() {
                for channel in missing {
                    let queue = self
                        .channel_queues
                        .get_mut(&channel.key())
                        .ok_or(RuntimeError::MissingMailbox)?;
                    queue.receive_waiters.push_back(continuation);
                }
            }
            return Ok(None);
        }

        let mut messages = Vec::with_capacity(channels.len());
        for channel in channels {
            let (mut entry, sender_waiter) = {
                let queue = self
                    .channel_queues
                    .get_mut(&channel.key())
                    .ok_or(RuntimeError::MissingMailbox)?;
                (
                    queue.entries.pop_front().expect("all inputs prechecked"),
                    queue.send_waiters.pop_front(),
                )
            };
            if let Some(waiter) = sender_waiter {
                self.wake_waiting_continuation(waiter);
            }
            let transferred = self
                .capability_spaces
                .get_mut(&actor.key())
                .ok_or(RuntimeError::MissingCapabilitySpace)?
                .alloc(entry.payload_authority);
            entry.descriptor.transferred_capability = transferred;
            self.trace(
                EventKind::ChannelReceived,
                actor,
                *channel,
                0,
                entry.descriptor.sender_sequence as u32,
            );
            messages.push(entry.descriptor);
        }
        Ok(Some(messages))
    }

    pub fn close_channel(&mut self, actor: Ref64, channel: Ref64) -> Result<(), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::DESTROY, channel)?;
        if self.channels.get(channel)?.closed != 0 {
            return Ok(());
        }
        self.authority_effect(actor, crate::abi::Rights::DESTROY, channel);
        self.channels.get_mut(channel)?.closed = 1;
        let waiters = {
            let queue = self
                .channel_queues
                .get_mut(&channel.key())
                .ok_or(RuntimeError::MissingMailbox)?;
            queue
                .send_waiters
                .drain(..)
                .chain(queue.receive_waiters.drain(..))
                .collect::<Vec<_>>()
        };
        for waiter in waiters {
            self.wake_waiting_continuation(waiter);
        }
        self.trace(EventKind::ChannelClosed, actor, channel, 0, 0);
        Ok(())
    }

    pub fn channel_len(&self, channel: Ref64) -> Result<usize, RuntimeError> {
        let _ = self.channels.get(channel)?;
        Ok(self
            .channel_queues
            .get(&channel.key())
            .ok_or(RuntimeError::MissingMailbox)?
            .entries
            .len())
    }

    fn escrow_payload_read(
        &mut self,
        actor: Ref64,
        payload: Ref64,
    ) -> Result<CapabilityEntry, RuntimeError> {
        self.authorize(actor, crate::abi::Rights::TRANSFER, payload)?;
        let capability = self
            .find_authorized_capability(
                actor,
                crate::abi::Rights::READ | crate::abi::Rights::TRANSFER,
                payload,
            )
            .ok_or(RuntimeError::AuthorityDenied)?;
        let mut escrowed = self
            .capability_spaces
            .get(&actor.key())
            .ok_or(RuntimeError::MissingCapabilitySpace)?
            .get(capability)?
            .clone();
        escrowed.rights = crate::abi::Rights::READ;
        escrowed.parent_capability = Ref64::NULL;
        self.authority_effect(actor, crate::abi::Rights::TRANSFER, payload);
        Ok(escrowed)
    }

    fn wake_waiting_continuation(&mut self, continuation: Ref64) {
        let Some((process, run_class, status)) = self
            .continuations
            .get(continuation)
            .ok()
            .map(|entry| (entry.process, entry.run_class, entry.status))
        else {
            return;
        };
        if status != crate::abi::continuations::ContinuationState::Waiting {
            return;
        }
        self.emit(crate::kernel::effects::Effect::Wake {
            continuation,
            run_class,
        });
        self.trace(
            EventKind::ContinuationReady,
            process,
            continuation,
            run_class,
            0,
        );
    }

    // ---- collectives -----------------------------------------------------

    pub fn create_batch_evaluate(
        &mut self,
        actor: Ref64,
        inputs: Ref64,
        element_count: u32,
        element_stride: u32,
    ) -> Result<(Ref64, Ref64), RuntimeError> {
        self.create_batch_evaluate_for(actor, 0, inputs, element_count, element_stride)
    }

    pub fn create_batch_evaluate_for(
        &mut self,
        actor: Ref64,
        evaluator_id: u32,
        inputs: Ref64,
        element_count: u32,
        element_stride: u32,
    ) -> Result<(Ref64, Ref64), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::READ, inputs)?;
        self.validate_frozen_array(inputs, element_count, element_stride)?;
        let completion = self.create_future(actor);
        let collective = self.collectives.alloc(CollectiveDescriptor::batch_evaluate(
            actor,
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            completion,
        ));
        self.collectives.get_mut(collective)?.id = collective;
        self.mint_genesis(actor, collective, 0, 0);
        self.trace(
            EventKind::CollectiveCreated,
            actor,
            collective,
            0,
            element_count,
        );
        Ok((collective, completion))
    }

    pub fn create_batch_evaluate_in_module(
        &mut self,
        actor: Ref64,
        module: Ref64,
        evaluator_id: u32,
        inputs: Ref64,
        element_count: u32,
    ) -> Result<(Ref64, Ref64), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::READ, module)?;
        let stride = self
            .module_manifest(module)
            .and_then(|manifest| {
                manifest
                    .iter()
                    .find_map(|(id, stride)| (*id == evaluator_id).then_some(*stride))
            })
            .ok_or(RuntimeError::InvalidModule)?;
        let result =
            self.create_batch_evaluate_for(actor, evaluator_id, inputs, element_count, stride)?;
        self.collectives.get_mut(result.0)?.module = module;
        Ok(result)
    }

    pub fn complete_batch_evaluate(
        &mut self,
        actor: Ref64,
        collective: Ref64,
        outputs: Ref64,
    ) -> Result<(), RuntimeError> {
        self.authorize(actor, crate::abi::Rights::WRITE, collective)?;
        let (state, count, stride, completion) = {
            let descriptor = self.collectives.get(collective)?;
            (
                descriptor.state,
                descriptor.element_count,
                descriptor.element_stride,
                descriptor.completion_future,
            )
        };
        if state != CollectiveState::Pending
            || self.future_state(completion)? != FutureState::Pending
        {
            return Err(RuntimeError::InvalidCollective);
        }
        self.authorize(actor, crate::abi::Rights::RESOLVE, completion)?;
        self.authorize(actor, crate::abi::Rights::READ, outputs)?;
        self.validate_frozen_array(outputs, count, stride)?;
        self.authorize(actor, crate::abi::Rights::WRITE, collective)?;
        self.authority_effect(actor, crate::abi::Rights::WRITE, collective);
        {
            let descriptor = self.collectives.get_mut(collective)?;
            descriptor.outputs = outputs;
            descriptor.state = CollectiveState::Completed;
        }
        self.resolve_future(actor, completion, outputs)?;
        self.trace(EventKind::CollectiveCompleted, actor, collective, 0, count);
        Ok(())
    }

    pub fn collective_state(&self, collective: Ref64) -> Result<CollectiveState, RuntimeError> {
        Ok(self.collectives.get(collective)?.state)
    }

    pub fn collective_evaluator(&self, collective: Ref64) -> Result<u32, RuntimeError> {
        Ok(self.collectives.get(collective)?.evaluator_id)
    }

    pub fn collective_module(&self, collective: Ref64) -> Result<Ref64, RuntimeError> {
        Ok(self.collectives.get(collective)?.module)
    }

    pub fn batch_evaluate_request(
        &self,
        collective: Ref64,
    ) -> Result<(u32, Ref64, u32, u32), RuntimeError> {
        let descriptor = self.collectives.get(collective)?;
        if descriptor.collective_kind != crate::abi::CollectiveKind::BatchEvaluate
            || descriptor.state != CollectiveState::Pending
        {
            return Err(RuntimeError::InvalidCollective);
        }
        Ok((
            descriptor.evaluator_id,
            descriptor.inputs,
            descriptor.element_count,
            descriptor.element_stride,
        ))
    }

    fn validate_frozen_array(
        &self,
        object: Ref64,
        element_count: u32,
        element_stride: u32,
    ) -> Result<(), RuntimeError> {
        let descriptor = self.objects.get(object)?;
        let required = u64::from(element_count)
            .checked_mul(u64::from(element_stride))
            .ok_or(RuntimeError::InvalidCollective)?;
        if descriptor.object_kind != ObjectKind::FrozenArray
            || descriptor.byte_length < required
            || crate::kernel::ownership::ownership_state(self, object)?
                != crate::abi::OwnershipState::FrozenShared
        {
            return Err(RuntimeError::InvalidCollective);
        }
        Ok(())
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
        if matches!(
            self.process_state(receiver)?,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return Err(RuntimeError::ProcessUnavailable);
        }
        if self.mailbox_is_full(receiver)? {
            self.authority_effect(actor, crate::abi::Rights::SEND, receiver);
            return self.push_message(actor, receiver, payload, Ref64::NULL, sender_cont, true);
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
        if matches!(
            self.process_state(receiver)?,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return Err(RuntimeError::ProcessUnavailable);
        }
        if self.mailbox_is_full(receiver)? {
            return self.push_message(sender, receiver, payload, Ref64::NULL, sender_cont, false);
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
            .get(&receiver.key())
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
            let key = (sender.key(), receiver.key());
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
                .get_mut(&receiver.key())
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
            self.trace_caused(
                EventKind::MessageSent,
                sender,
                sender_cont,
                0,
                seq as u32,
                receiver,
            );
        }

        if let Some(waiter) = waiter {
            let rc = self.continuations.get(waiter)?.run_class;
            self.emit(crate::kernel::effects::Effect::Wake {
                continuation: waiter,
                run_class: rc,
            });
            self.trace_caused(
                EventKind::MessageReceived,
                receiver,
                waiter,
                rc,
                seq as u32,
                sender,
            );
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
        if matches!(
            self.process_state(process)?,
            ProcessState::Failed | ProcessState::Terminated | ProcessState::Cancelled
        ) {
            return Err(RuntimeError::ProcessUnavailable);
        }
        let _ = self
            .mailboxes
            .get(&process.key())
            .ok_or(RuntimeError::MissingMailbox)?;
        self.authority_effect(actor, crate::abi::Rights::RECEIVE, process);
        let mailbox = self
            .mailboxes
            .get_mut(&process.key())
            .ok_or(RuntimeError::MissingMailbox)?;
        if let Some(msg) = mailbox.entries.pop_front() {
            // One receive frees exactly one slot, so wake exactly one sender.
            if let Some(w) = mailbox.full_waiters.pop_front() {
                let rc = self.continuations.get(w)?.run_class;
                self.emit(crate::kernel::effects::Effect::Wake {
                    continuation: w,
                    run_class: rc,
                });
                self.trace(EventKind::ContinuationReady, process, w, rc, 0);
            }
            self.trace_caused(
                EventKind::MessageReceived,
                process,
                cont,
                0,
                msg.sender_sequence as u32,
                msg.sender,
            );
            return Ok(Some(msg));
        }
        mailbox.recv_waiters.push_back(cont);
        Ok(None)
    }

    /// Number of undelivered messages in a process's mailbox.
    pub fn mailbox_len(&self, p: Ref64) -> usize {
        self.mailboxes
            .get(&p.key())
            .map(|m| m.entries.len())
            .unwrap_or(0)
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}
