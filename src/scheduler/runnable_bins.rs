//! Double-buffered append-only runnable bins (§13).
//!
//! SOMA-P1 does not need a general lock-free work queue. Each run class has two
//! buffers — `current` and `next`. During an epoch, workers consume only from
//! `current`; new runnable continuations append only to `next`; the buffers swap
//! at the next scheduling boundary. This gives deterministic epoch boundaries,
//! cheap grouping by run class, and no concurrent pop / ABA problems.
//!
//! This grouping *is* the simplest form of continuation cohorting (§9): every
//! yielded continuation already knows the exact queue it belongs to via
//! `next_run_class`.

use crate::abi::Ref64;

/// One double-buffered bin for a single run class.
#[derive(Clone, Debug, Default)]
pub struct DoubleBin {
    /// Consumed this epoch.
    current: Vec<Ref64>,
    /// Appended this epoch; swapped to `current` at the boundary.
    next: Vec<Ref64>,
    pub capacity: u32,
}

impl DoubleBin {
    fn new(capacity: u32) -> DoubleBin {
        DoubleBin {
            current: Vec::new(),
            next: Vec::new(),
            capacity,
        }
    }

    /// Append a runnable continuation to the next-epoch buffer.
    pub fn enqueue(&mut self, cont: Ref64) {
        self.next.push(cont);
    }

    /// Drain the current-epoch buffer for execution.
    pub fn drain_current(&mut self) -> Vec<Ref64> {
        std::mem::take(&mut self.current)
    }

    pub fn current_len(&self) -> usize {
        self.current.len()
    }

    pub fn next_len(&self) -> usize {
        self.next.len()
    }

    /// Swap `next` into `current` at the epoch boundary.
    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        self.next.clear();
    }
}

/// How runnable continuations are assigned to bins — the single variable the
/// cohorting experiment turns (§26).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SchedulingMode {
    /// One bin per run class, so a bin's contents are uniform by construction.
    /// This is the mechanism under test.
    #[default]
    RunClassBins,
    /// A single persistent queue in arrival order, ignoring run class. The
    /// baseline that isolates launch elimination from cohorting: work is still
    /// resident and still dispatched in lane groups, but those groups mix run
    /// classes exactly as they arrive.
    PersistentFifo,
}

/// The bin every continuation lands in under `PersistentFifo`.
pub const FIFO_BIN: u32 = 0;

/// All runnable bins, keyed by run-class id. Run-class grouping is implicit:
/// each bin belongs to exactly one run class.
#[derive(Debug, Default)]
pub struct Scheduler {
    bins: std::collections::HashMap<u32, DoubleBin>,
    mode: SchedulingMode,
}

impl Scheduler {
    /// A scheduler that bins by the given mode.
    pub fn with_mode(mode: SchedulingMode) -> Scheduler {
        Scheduler {
            bins: std::collections::HashMap::new(),
            mode,
        }
    }

    pub fn mode(&self) -> SchedulingMode {
        self.mode
    }

    /// The bin a continuation of `run_class` belongs to under the current mode.
    pub fn bin_of(&self, run_class: u32) -> u32 {
        match self.mode {
            SchedulingMode::RunClassBins => run_class,
            SchedulingMode::PersistentFifo => FIFO_BIN,
        }
    }

    /// Register a run class with a bin capacity.
    pub fn register_run_class(&mut self, run_class: u32, capacity: u32) {
        self.bins
            .entry(run_class)
            .or_insert_with(|| DoubleBin::new(capacity));
    }

    /// Append a runnable continuation to the next-epoch bin its run class maps
    /// to. Under `PersistentFifo` every run class maps to the same bin, which is
    /// what makes lane groups divergent downstream.
    pub fn enqueue(&mut self, run_class: u32, cont: Ref64) {
        let bin = self.bin_of(run_class);
        self.bins
            .entry(bin)
            .or_insert_with(|| DoubleBin::new(u32::MAX))
            .enqueue(cont);
    }

    /// Total runnable continuations in current-epoch buffers (all classes).
    pub fn total_runnable(&self) -> usize {
        self.bins.values().map(|b| b.current_len()).sum()
    }

    /// Total continuations still outstanding anywhere: current + next buffers.
    /// Used to detect quiescence, since produced work lands in `next` until the
    /// next epoch-boundary swap (§13).
    pub fn total_pending(&self) -> usize {
        self.bins
            .values()
            .map(|b| b.current_len() + b.next_len())
            .sum()
    }

    /// Per-class counts of all pending (current + next) work, in class order.
    pub fn pending_counts(&self) -> Vec<(u32, usize)> {
        let mut v: Vec<(u32, usize)> = self
            .bins
            .iter()
            .map(|(k, b)| (*k, b.current_len() + b.next_len()))
            .filter(|(_, n)| *n > 0)
            .collect();
        v.sort();
        v
    }

    /// Snapshot of current-epoch runnable counts per bin, for tracing /
    /// cohorting statistics. Sorted, so iteration order is deterministic
    /// regardless of the underlying map.
    pub fn runnable_counts(&self) -> Vec<(u32, usize)> {
        let mut v: Vec<(u32, usize)> = self
            .bins
            .iter()
            .map(|(k, b)| (*k, b.current_len()))
            .filter(|(_, n)| *n > 0)
            .collect();
        v.sort();
        v
    }

    /// Drain one run class's current-epoch bin for execution.
    pub fn drain(&mut self, run_class: u32) -> Vec<Ref64> {
        match self.bins.get_mut(&run_class) {
            Some(b) => b.drain_current(),
            None => Vec::new(),
        }
    }

    /// Move every `next` buffer into `current` (epoch boundary). Deterministic:
    /// within a run class, order is preserved from insertion.
    pub fn swap_all(&mut self) {
        for b in self.bins.values_mut() {
            b.swap();
        }
    }
}
