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
    /// A process gave up authority it held. Not an `AuthorityEffect`: that
    /// records authority being *exercised* and I-checks demand an adjacent
    /// grant for one, while letting go needs no permission beyond having held
    /// it. It is traced because it changes what a process may do next, and
    /// because it is what makes something collectable.
    AuthorityReleased = 33,
    /// A domain refused a process creation because its quota was full.
    ///
    /// The refusal is a thing that happened and the trace had no way to say it:
    /// a step whose allocation is refused faults, and `ProcessFailed` alone does
    /// not distinguish a program that failed from one that was told no. It is
    /// also what makes I25's clause 2 precise rather than conservative — two
    /// lanes drawing on a bounded domain only decide anything by their order
    /// once the bound actually binds, and this is the event that says it did.
    ProcessCreationRefused = 34,
    /// A send found the receiver's mailbox full and parked the sender.
    ///
    /// Back-pressure is a designed outcome and not a failure — the sender is
    /// registered as a waiter and retries when a slot frees (§11) — but it is
    /// still something that happened to a message that did not arrive, and the
    /// trace could not say it. `causal` is the receiver, as it is for
    /// `MessageSent`, and `auxiliary` is the capacity that was reached.
    ///
    /// Like `ProcessCreationRefused`, it is also what makes I25 clause 2
    /// precise: two lanes sending to one mailbox decide nothing between them
    /// until the mailbox is actually full.
    MessageSendBlocked = 35,
    /// A receive found the mailbox empty and parked the receiver.
    ///
    /// `MessageSendBlocked`'s mirror, and it exists for the same two reasons.
    /// A receive that found nothing left no record at all: the trace showed a
    /// continuation that started and then waited, and could not say whether it
    /// was waiting on a future or on an empty mailbox. And I25 clause 2 cannot
    /// tell a mailbox several lanes drained between them from one that had a
    /// message for each of them without it.
    ///
    /// `process` is the mailbox's owner, as it is for `MessageReceived`, which
    /// is what lets the clause key the two on one resource.
    ///
    /// `auxiliary` is the occupancy that refused the receive, which is always
    /// zero — a receive is refused by emptiness and by nothing else. It is
    /// recorded as the constant rather than filled with the receiver-waiter
    /// queue's depth, which would be the more informative number and is the
    /// wrong one: the depth is the *parking order*, and parking order differs
    /// between two lane orders in runs whose epoch outcome does not, so putting
    /// it in the trace would make those runs disagree over something no epoch
    /// decided.
    MessageReceiveBlocked = 36,
    /// A resolve found the future already settled and refused the write.
    ///
    /// Single assignment (§12) is the property this enforces, and until now it
    /// enforced it silently: the loser's step faulted, and `ProcessFailed` alone
    /// does not say a program was told the value was already published rather
    /// than having gone wrong on its own.
    ///
    /// `causal` is the future, as it is for `FutureResolved`, which is what lets
    /// I25 clause 2 key the two on one resource. `subject` is the value the
    /// refused lane had built and did not publish — an entity, so it goes where
    /// entities go.
    FutureResolutionRefused = 37,
    /// An await found the future already settled and did not park.
    ///
    /// The first traced decision that is not a refusal. The other four exist
    /// because an operation said no and the trace could not say so; this one
    /// **succeeded**, and by either route — the awaiting continuation continues
    /// in its next run class whether it registered as a waiter or found the
    /// value already published. What differs is which of two states of the
    /// future it read, and a resolving lane of the same epoch decides that.
    ///
    /// That is why it is here. `ContinuationWaiting` records the other branch,
    /// so an await that registered leaves a mark and an await that did not left
    /// nothing to distinguish it from any other yield — and a trace with a hole
    /// exactly where the two branches differ cannot report the run in which the
    /// resolver went first. See v0.3 §4.15, which widens I25 clause 2's
    /// question from operations that can refuse to operations whose result
    /// another lane can decide.
    ///
    /// `causal` is the future, as it is for `FutureResolved`, and `subject` is
    /// the value the await found published.
    FutureAwaitSettled = 38,
    /// A lane looked at a future's state without awaiting it.
    ///
    /// `future_value` was the one read on `LaneView` that was neither
    /// authorized nor recorded, which made it the only way a lane could learn
    /// something about another lane's epoch and leave nothing behind. What it
    /// returns is decided by whether a resolver of the same epoch ran first,
    /// and a poll that acts on what it saw turns that into behaviour an epoch
    /// or two later — so the run that was decided by lane order is not the run
    /// that reports it (v0.3 §4.16).
    ///
    /// `causal` is the future and `subject` is the value, or `Ref64::NULL` when
    /// the poll found it still pending. `auxiliary` is 1 when it was resolved
    /// and 0 when it was not, so the clause can read the outcome without
    /// inspecting the subject — and so that *both* outcomes are recorded.
    /// Recording only the resolved one would repeat this section's own mistake:
    /// a poll that saw nothing was equally decided by the lane that had not yet
    /// run.
    FutureStateObserved = 39,
    /// The process's owning node was explicitly declared lost. Distinct from
    /// `ProcessFailed`: no continuation returned `Fault`.
    ProcessLost = 40,
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
