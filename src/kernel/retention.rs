//! What the kernel does with its append-only logs.
//!
//! The trace, the effect log, and the admission log are all append-only and all
//! grow with the run rather than with the machine. That is the right default:
//! every whole-run check in `semantics` — I18's conformance, I22's admission
//! determinism, I24's effect accounting — reads a log from the first event to
//! the last, and a log that forgot part of the run cannot answer them.
//!
//! It is the wrong default for a long run. The trace alone costs nine events per
//! continuation step at 64 bytes each, so ten thousand processes stepping for a
//! thousand epochs produce something on the order of six gigabytes of trace, and
//! the admission log separately retains every epoch's whole candidate set. A
//! workload that wants to *stream* its trace out rather than accumulate it needs
//! somewhere to put the records and permission to forget them.
//!
//! [`LogRetention`] is that permission, and it is opt-in. Under `Retain` — the
//! default — nothing changes and no existing check is weakened. Under
//! `PerEpoch` the logs are cleared at the start of each epoch, so after
//! `run_epoch` returns they hold exactly that epoch's records and a consumer
//! that drains them between epochs sees every record exactly once.
//!
//! # Nothing is dropped silently
//!
//! Forgetting records is a way to make a log look complete when it is not, so
//! every record leaves an accounting trail whether it was kept, taken, or
//! dropped. [`LogCensus`] reports all three, and `emitted == retained + taken +
//! dropped` holds for each log over the whole run. A consumer that expected to
//! see everything can check that `dropped` is zero rather than trusting that it
//! drained often enough.

/// How long the kernel keeps its append-only logs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogRetention {
    /// Keep every record for the whole run.
    ///
    /// The default, and what the whole-run invariant checks require. A run under
    /// any other policy cannot be handed to `semantics::order` or
    /// `semantics::schedule` and get a meaningful answer.
    #[default]
    Retain,
    /// Clear the logs at the start of each epoch.
    ///
    /// After `run_epoch` returns, each log holds exactly the records that epoch
    /// produced. Records a consumer does not take before the next epoch begins
    /// are counted as dropped, not lost quietly.
    ///
    /// Records produced *before* the first epoch — process creation, capability
    /// grants, whatever the workload builds during setup — belong to no epoch
    /// and are cleared by the first `run_epoch` like any other. Drain them
    /// before running if they matter.
    PerEpoch,
}

impl LogRetention {
    /// Whether this policy discards records at the epoch boundary.
    pub fn is_bounded(&self) -> bool {
        matches!(self, LogRetention::PerEpoch)
    }
}

/// What became of one log's records over a run.
///
/// `emitted == retained + taken + dropped` is the point of the type: it makes
/// the arithmetic checkable rather than assumed, so a bounded run can state
/// exactly what it discarded instead of presenting a truncated log as a whole
/// one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogCensus {
    /// Records appended over the whole run.
    pub emitted: u64,
    /// Records a consumer drained.
    pub taken: u64,
    /// Records discarded at an epoch boundary without being drained.
    pub dropped: u64,
    /// Records still held in the log.
    pub retained: u64,
}

impl LogCensus {
    /// Whether every emitted record is accounted for. Always true for a kernel
    /// that maintains its own counters; false is a bug in this module.
    pub fn is_balanced(&self) -> bool {
        self.emitted == self.retained + self.taken + self.dropped
    }

    /// Whether the run kept or handed over every record it produced.
    pub fn is_complete(&self) -> bool {
        self.dropped == 0
    }
}

/// The census of all three append-only logs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogAccounting {
    pub trace: LogCensus,
    pub effects: LogCensus,
    pub admissions: LogCensus,
}

impl LogAccounting {
    pub fn is_balanced(&self) -> bool {
        self.trace.is_balanced() && self.effects.is_balanced() && self.admissions.is_balanced()
    }

    /// Whether no log dropped anything.
    pub fn is_complete(&self) -> bool {
        self.trace.is_complete() && self.effects.is_complete() && self.admissions.is_complete()
    }
}

/// Per-log counters. Kept next to the log rather than derived from it, because
/// the quantity being counted is precisely what the log no longer holds.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LogCounters {
    pub(crate) emitted: u64,
    pub(crate) taken: u64,
    pub(crate) dropped: u64,
}

impl LogCounters {
    pub(crate) fn emit(&mut self) {
        self.emitted = self.emitted.saturating_add(1);
    }

    pub(crate) fn take(&mut self, n: usize) {
        self.taken = self.taken.saturating_add(n as u64);
    }

    pub(crate) fn drop_all(&mut self, n: usize) {
        self.dropped = self.dropped.saturating_add(n as u64);
    }

    pub(crate) fn census(&self, retained: usize) -> LogCensus {
        LogCensus {
            emitted: self.emitted,
            taken: self.taken,
            dropped: self.dropped,
            retained: retained as u64,
        }
    }
}
