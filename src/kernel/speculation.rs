//! Conflict metadata for the speculative CPU epoch executive.
//!
//! A worker runs against an isolated kernel snapshot. `LaneView` records the
//! semantic resources touched by the handler; the epoch validator then accepts
//! only histories whose read/write sets commute. Everything else is replayed
//! by the reference executive in plan order.

use std::collections::HashSet;

use crate::abi::{MessageDescriptor, ObjectKind, ProcessMode, Ref64};
use crate::kernel::{AwaitOutcome, ContinuationSpec, RuntimeError};

/// How Phase F executes admitted lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EpochExecutive {
    /// The normative, plan-order interpreter.
    #[default]
    Reference,
    /// Execute at most `max_lanes` isolated snapshots concurrently, validate
    /// their recorded accesses, then commit in canonical lane order.
    Speculative { max_lanes: usize },
}

/// Cumulative measurements for speculative epochs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculationStats {
    pub attempted_epochs: u64,
    pub committed_epochs: u64,
    pub fallback_epochs: u64,
    pub speculative_lanes: u64,
    pub committed_lanes: u64,
    pub conflict_fallbacks: u64,
    pub unsupported_fallbacks: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Resource {
    Object(Ref64),
    Future(Ref64),
    Process(Ref64),
    Mailbox(Ref64),
    Domain(Ref64),
    Allocation(u8),
}

/// Concrete kernel calls made by one handler. Accepted speculative executions
/// replay this list against the real kernel; results are retained so a debug
/// build can detect an incomplete conflict declaration at the commit boundary.
#[derive(Clone, Debug)]
pub(crate) enum LaneOperation {
    ObserveFuture {
        actor: Ref64,
        future: Ref64,
        result: Result<Option<Ref64>, RuntimeError>,
    },
    ReadObject {
        actor: Ref64,
        object: Ref64,
    },
    CreateProcess {
        actor: Ref64,
        mode: ProcessMode,
        result: Result<Ref64, RuntimeError>,
    },
    CreateContinuation {
        actor: Ref64,
        process: Ref64,
        spec: ContinuationSpec,
        result: Result<Ref64, RuntimeError>,
    },
    CreateFuture {
        actor: Ref64,
        result: Ref64,
    },
    CreateObject {
        actor: Ref64,
        kind: ObjectKind,
        bytes: Vec<u8>,
        result: Ref64,
    },
    WriteObject {
        actor: Ref64,
        object: Ref64,
        growable: bool,
    },
    EnqueueMessage {
        actor: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_continuation: Ref64,
        result: Result<(), RuntimeError>,
    },
    ReceiveMessage {
        actor: Ref64,
        continuation: Ref64,
        result: Result<Option<MessageDescriptor>, RuntimeError>,
    },
    ResolveFuture {
        actor: Ref64,
        future: Ref64,
        value: Ref64,
        result: Result<(), RuntimeError>,
    },
    AwaitFuture {
        actor: Ref64,
        continuation: Ref64,
        future: Ref64,
        next_run_class: u32,
        result: Result<AwaitOutcome, RuntimeError>,
    },
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LaneJournal {
    pub(crate) reads: HashSet<Resource>,
    pub(crate) writes: HashSet<Resource>,
    pub(crate) mutated_objects: HashSet<Ref64>,
    pub(crate) operations: Vec<LaneOperation>,
    pub(crate) unsupported: bool,
}

impl LaneJournal {
    pub(crate) fn read(&mut self, resource: Resource) {
        self.reads.insert(resource);
    }

    pub(crate) fn write(&mut self, resource: Resource) {
        self.writes.insert(resource);
    }

    pub(crate) fn push(&mut self, operation: LaneOperation) {
        self.operations.push(operation);
    }

    pub(crate) fn conflicts_with(&self, other: &Self) -> bool {
        self.writes
            .iter()
            .any(|resource| other.writes.contains(resource) || other.reads.contains(resource))
            || other
                .writes
                .iter()
                .any(|resource| self.reads.contains(resource))
    }
}
