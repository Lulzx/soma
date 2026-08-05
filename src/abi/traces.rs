//! Trace ABI (§21). Every significant event emits a compact trace record for
//! deterministic replay.

/// Required trace event kinds (§21).
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventKind {
    ProcessCreated = 1,
    MessageSent = 2,
    MessageReceived = 3,
    ContinuationReady = 4,
    ContinuationPlaced = 5,
    CohortCreated = 6,
    ContinuationStarted = 7,
    ContinuationYielded = 8,
    ContinuationWaiting = 9,
    ContinuationCompleted = 10,
    FutureResolved = 11,
    ProcessFailed = 12,
    ProcessCancelled = 13,
    AuthorityGranted = 14,
    AuthorityDenied = 15,
    /// Boundary marker for a state-changing governed operation. Its actor,
    /// target, and right use the same fields as the immediately preceding
    /// `AuthorityGranted` event.
    AuthorityEffect = 16,
    ContinuationCancelled = 17,
    FutureFailed = 18,
    FutureCancelled = 19,
    ChannelSent = 20,
    ChannelReceived = 21,
    ChannelClosed = 22,
    CollectiveCreated = 23,
    CollectiveCompleted = 24,
    CollectiveFailed = 25,
    CollectiveCancelled = 26,
    SupervisionNotified = 27,
    ProcessRestarted = 28,
    DomainCreated = 29,
    ContractCreated = 30,
    ContractAttached = 31,
    ModuleLoaded = 32,
}

/// The lane an event was emitted from when it was not emitted from one: epoch
/// bookkeeping, cohort records, and anything a caller does between epochs.
///
/// Zero rather than a sentinel at the top of the range, so that ordering by
/// `(epoch, lane, lane_sequence)` puts an epoch's host events before the lanes
/// they set up. That is also the order they happen in.
pub const HOST_LANE: u32 = 0;

/// Compact trace record (§21).
#[derive(Clone, Copy, Debug)]
pub struct TraceEvent {
    /// A total order over the whole run, assigned by whoever appended the
    /// event. It is what I11 checks, and it is exactly what a device cannot
    /// produce: concurrent lanes have no shared clock to draw it from.
    ///
    /// It is kept because the sequential interpreter is the reference and
    /// because replay reads it, but it carries no information beyond
    /// `(epoch, lane, lane_sequence)` — which is what I23 checks, and what
    /// makes it reconstructible rather than required.
    pub logical_time: u64,

    pub epoch: u32,
    pub event_kind: EventKind,
    pub engine: u16,

    /// Which lane of its epoch emitted this event, or `HOST_LANE`.
    ///
    /// Lanes are numbered from 1 in the epoch's admitted order, so the number
    /// is a position in the epoch's plan rather than a fact about hardware. It
    /// is placement information and the semantic projection does not carry it.
    pub lane: u32,
    /// The event's position within its lane's own emissions this epoch.
    ///
    /// A lane is sequential, so this needs nothing shared to assign: a
    /// concurrent implementation counts locally and the total order falls out
    /// of the three fields together.
    pub lane_sequence: u32,

    pub process: Ref64,
    pub continuation: Ref64,

    pub run_class: u32,
    /// Purely numeric detail: a sequence number, a count, a right mask. Never
    /// an entity.
    ///
    /// It used to carry a bare slot for four event kinds, which made those
    /// events uncomparable across two runs that name their entities
    /// differently: a slot number alone has no kind and no generation, so
    /// nothing can tell it apart from a sequence number and nothing can
    /// translate it. Entities live in `subject` now.
    pub auxiliary: u32,

    /// The secondary entity an event is about: the future a continuation is
    /// waiting on, the value a future resolved to, the process a restart
    /// replaced, the contract a continuation was created under.
    ///
    /// Distinct from `causal`, which names the entity two events are ordered
    /// *through*. An event may have both — `FutureResolved` is caused through
    /// the future and is about the value.
    pub subject: Ref64,

    /// The entity through which this event is causally related to another
    /// event: the future a wake came from, the channel or receiver a message
    /// travelled through, the child a supervision notice reports, the
    /// collective a completion belongs to. `NULL` when the event has no
    /// cross-entity cause.
    ///
    /// The sequential interpreter does not need this — its total order over
    /// `logical_time` already encodes every dependency. A concurrent or
    /// distributed implementation does: it emits events with no shared clock,
    /// so causality has to be carried in the record rather than inferred from
    /// adjacency. This field is what makes I18 checkable against such an
    /// implementation.
    pub causal: Ref64,
}

use super::refs::Ref64;

impl TraceEvent {
    pub fn new(
        logical_time: u64,
        epoch: u32,
        event_kind: EventKind,
        process: Ref64,
        continuation: Ref64,
        run_class: u32,
    ) -> TraceEvent {
        TraceEvent {
            logical_time,
            epoch,
            event_kind,
            engine: 0,
            lane: HOST_LANE,
            lane_sequence: 0,
            process,
            continuation,
            run_class,
            auxiliary: 0,
            subject: Ref64::NULL,
            causal: Ref64::NULL,
        }
    }

    /// The event's position, independent of any clock: the key a concurrent
    /// implementation sorts on to recover the order this run emitted.
    pub fn position(&self) -> (u32, u32, u32) {
        (self.epoch, self.lane, self.lane_sequence)
    }
}
