//! Epoch accounting (§18 Phase H, §27).
//!
//! These are the counters the go/no-go criteria are computed from. The two that
//! matter most for §28.1 are the useful active SIMD-lane ratio and the cohort
//! fill ratio.

/// Cumulative execution counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Accounting {
    pub epochs: u64,
    /// Continuation steps executed.
    pub steps: u64,
    /// SIMD dispatches issued.
    pub cohorts: u64,
    /// Dispatches in which every lane carried work.
    pub full_cohorts: u64,
    /// Lane-slots issued: each dispatch costs its full width.
    pub lane_slots: u64,
    /// Lane-slots that carried a real continuation.
    pub useful_lane_slots: u64,
    /// Lane-slots masked off inside a dispatch.
    pub idle_lane_slots: u64,
    /// Continuations held back by the partial-cohort policy.
    pub deferred_lanes: u64,
    /// Continuations deferred by the serial-process invariant (§19).
    pub serial_deferrals: u64,
    /// Epochs that admitted work and then dispatched nothing.
    ///
    /// I21's first half is a statement about a transition, not a state, so it
    /// cannot be read off the tables afterwards. Counting the stalls as they
    /// happen turns it into one that can: a legal run has none.
    pub stalled_epochs: u64,
}

impl Accounting {
    /// Useful active SIMD-lane ratio (§27): the fraction of issued lane-slots
    /// that did real work. This is the quantity §28.1 compares between the
    /// persistent-FIFO baseline and run-class cohorting.
    ///
    /// Returns 0.0 when nothing has been dispatched.
    pub fn lane_occupancy(&self) -> f64 {
        if self.lane_slots == 0 {
            return 0.0;
        }
        self.useful_lane_slots as f64 / self.lane_slots as f64
    }

    /// Cohort fill ratio (§27): the fraction of dispatches that were completely
    /// full. Distinct from occupancy — a run of half-empty dispatches and a mix
    /// of full and near-empty ones can share an occupancy but not a fill ratio.
    pub fn cohort_fill_ratio(&self) -> f64 {
        if self.cohorts == 0 {
            return 0.0;
        }
        self.full_cohorts as f64 / self.cohorts as f64
    }

    /// Mean number of active lanes per dispatch.
    pub fn mean_active_lanes(&self) -> f64 {
        if self.cohorts == 0 {
            return 0.0;
        }
        self.useful_lane_slots as f64 / self.cohorts as f64
    }
}
