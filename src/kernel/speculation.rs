//! Conflict metadata for the speculative CPU epoch executive.
//!
//! A worker runs against an isolated kernel snapshot. `LaneView` records the
//! semantic resources touched by the handler; the epoch validator then accepts
//! only histories whose read/write sets commute. Everything else is replayed
//! by the reference executive in plan order.

use std::collections::HashSet;

use crate::abi::Ref64;

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
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LaneJournal {
    pub(crate) reads: HashSet<Resource>,
    pub(crate) writes: HashSet<Resource>,
    pub(crate) mutated_objects: HashSet<Ref64>,
    pub(crate) unsupported: bool,
}

impl LaneJournal {
    pub(crate) fn read(&mut self, resource: Resource) {
        self.reads.insert(resource);
    }

    pub(crate) fn write(&mut self, resource: Resource) {
        self.writes.insert(resource);
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
