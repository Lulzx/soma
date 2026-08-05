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
}

/// Compact trace record (§21).
#[derive(Clone, Copy, Debug)]
pub struct TraceEvent {
    pub logical_time: u64,

    pub epoch: u32,
    pub event_kind: EventKind,
    pub engine: u16,

    pub process: Ref64,
    pub continuation: Ref64,

    pub run_class: u32,
    pub auxiliary: u32,
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
            process,
            continuation,
            run_class,
            auxiliary: 0,
        }
    }
}
