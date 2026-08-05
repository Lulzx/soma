//! The cohorting experiment (§26, §27, §28.1).
//!
//! Compares two runs of the same workload that differ in exactly one variable:
//! how runnable continuations are binned. The persistent-FIFO baseline keeps
//! every process resident and dispatches lane groups in arrival order; run-class
//! binning dispatches the same continuations grouped by run class. Everything
//! else — the workload, the executive, the commit path, the cohort construction
//! code — is shared.
//!
//! # What this measures, and what it does not
//!
//! Occupancy here is a **structural** quantity, derived from how continuations
//! group, not a measurement of silicon. A lane group holding `k` distinct run
//! classes is counted as `k` masked dispatches because a uniform-dispatch SIMD
//! executive cannot do better (§15); real hardware may do worse, never better.
//!
//! So this is an upper bound on what cohorting can buy, computed exactly, and it
//! is meaningful for the go/no-go's occupancy limb (§28.1). It says nothing
//! about throughput, which is the other limb, and nothing about scheduler
//! overhead (§28.2) — both need the GPU executive and wall-clock timing that
//! Phase 1 does not have. A favourable ratio here is a necessary condition for
//! the hypothesis, not evidence that it holds.

use crate::abi::cohorts::PartialCohortPolicy;
use crate::experiments::bulk_frontier::{self, BulkFrontierRun};
use crate::experiments::dynamic_search::{build_in, ControlKnobs};
use crate::kernel::accounting::Accounting;
use crate::kernel::Kernel;
use crate::scheduler::runnable_bins::SchedulingMode;

/// One configured run of the workload.
#[derive(Clone, Copy, Debug)]
pub struct StudyRun {
    pub mode: SchedulingMode,
    pub cohort_width: u16,
    pub epochs: u32,
    pub accounting: Accounting,
}

impl StudyRun {
    pub fn lane_occupancy(&self) -> f64 {
        self.accounting.lane_occupancy()
    }

    pub fn cohort_fill_ratio(&self) -> f64 {
        self.accounting.cohort_fill_ratio()
    }

    /// Total SIMD dispatches issued — the quantity cohorting reduces.
    pub fn dispatches(&self) -> u64 {
        self.accounting.cohorts
    }
}

/// Run the branching search under one scheduling mode.
pub fn run(
    knobs: &ControlKnobs,
    mode: SchedulingMode,
    cohort_width: u16,
    partial_policy: PartialCohortPolicy,
) -> StudyRun {
    let mut kernel = Kernel::with_mode(mode);
    kernel.configure_cohorts(cohort_width, partial_policy);
    let mut kernel = build_in(kernel, knobs);
    let epochs = kernel.run_to_quiescence(100_000);
    StudyRun {
        mode,
        cohort_width,
        epochs,
        accounting: *kernel.accounting(),
    }
}

/// The §28.1 comparison: run-class cohorting against the persistent-FIFO
/// baseline on identical work.
#[derive(Clone, Copy, Debug)]
pub struct Comparison {
    pub fifo: StudyRun,
    pub cohorted: StudyRun,
}

impl Comparison {
    /// Useful-lane-occupancy ratio. §28.1's occupancy limb asks for at least
    /// 1.5x in a meaningful divergent regime.
    pub fn occupancy_ratio(&self) -> f64 {
        let baseline = self.fifo.lane_occupancy();
        if baseline <= 0.0 {
            return 0.0;
        }
        self.cohorted.lane_occupancy() / baseline
    }

    /// How much of the baseline's dispatch count cohorting eliminates.
    pub fn dispatch_reduction(&self) -> f64 {
        let baseline = self.fifo.dispatches();
        if baseline == 0 {
            return 0.0;
        }
        1.0 - (self.cohorted.dispatches() as f64 / baseline as f64)
    }

    /// Whether the occupancy limb of §28.1 is met. The throughput limb needs
    /// wall-clock numbers from a real executive and is not evaluated here.
    pub fn meets_occupancy_criterion(&self) -> bool {
        self.occupancy_ratio() >= 1.5
    }

    /// Both runs must execute exactly the same work; only the binning differs.
    pub fn executed_identical_work(&self) -> bool {
        self.fifo.accounting.steps == self.cohorted.accounting.steps
            && self.fifo.accounting.useful_lane_slots == self.cohorted.accounting.useful_lane_slots
    }
}

/// Compare the two modes at a given SIMD width.
pub fn compare(knobs: &ControlKnobs, cohort_width: u16) -> Comparison {
    let policy = PartialCohortPolicy::RunPartial;
    Comparison {
        fifo: run(knobs, SchedulingMode::PersistentFifo, cohort_width, policy),
        cohorted: run(knobs, SchedulingMode::RunClassBins, cohort_width, policy),
    }
}

/// SOMA against both required baselines (§26).
#[derive(Clone, Copy, Debug)]
pub struct Baselines {
    pub fifo: StudyRun,
    pub cohorted: StudyRun,
    /// The naive manual batch: one kernel over the frontier, mixed classes.
    pub bulk_unsorted: BulkFrontierRun,
    /// The strong manual batch: the host partitions each frontier by run class
    /// before dispatching it.
    pub bulk_sorted: BulkFrontierRun,
}

impl Baselines {
    /// §28.3: on work already well suited to bulk execution, SOMA must stay
    /// within 15% of the manually batched implementation. Measured in
    /// dispatches, the only currency Phase 1 can compare without a real clock.
    pub fn within_bulk_tolerance(&self) -> bool {
        let bulk = self.bulk_sorted.dispatches();
        if bulk == 0 {
            return false;
        }
        self.cohorted.dispatches() as f64 <= bulk as f64 * 1.15
    }

    /// How SOMA's dispatch count compares to the strong manual batch. Above 1.0
    /// means SOMA issues more dispatches than a hand-written frontier kernel.
    pub fn dispatch_ratio_vs_bulk(&self) -> f64 {
        let bulk = self.bulk_sorted.dispatches();
        if bulk == 0 {
            return 0.0;
        }
        self.cohorted.dispatches() as f64 / bulk as f64
    }

    /// §28.4: host round-trips the baseline needs and SOMA does not. SOMA's
    /// epoch loop is device-resident by design, so the host submits nothing per
    /// operation; the bulk frontier needs one launch per level.
    pub fn host_launches_avoided(&self) -> u64 {
        self.bulk_sorted.host_launches
    }
}

/// Run SOMA both ways and both baselines on one workload.
pub fn baselines(knobs: &ControlKnobs, cohort_width: u16) -> Baselines {
    let c = compare(knobs, cohort_width);
    Baselines {
        fifo: c.fifo,
        cohorted: c.cohorted,
        bulk_unsorted: bulk_frontier::run(knobs, cohort_width, false),
        bulk_sorted: bulk_frontier::run(knobs, cohort_width, true),
    }
}

/// A human-readable report of the comparison, for `cargo run --example` style
/// inspection and for pasting into the measurement log.
pub fn report(knobs: &ControlKnobs, cohort_width: u16) -> String {
    let c = compare(knobs, cohort_width);
    let mut s = String::new();
    s.push_str(&format!(
        "branching={} depth={} roots={} classes={} width={}\n",
        knobs.branching_factor,
        knobs.depth,
        knobs.process_count,
        knobs.class_count,
        cohort_width
    ));
    for (label, run) in [("persistent-fifo", c.fifo), ("run-class", c.cohorted)] {
        s.push_str(&format!(
            "  {label:<16} dispatches={:<7} occupancy={:.3}  fill={:.3}  epochs={}\n",
            run.dispatches(),
            run.lane_occupancy(),
            run.cohort_fill_ratio(),
            run.epochs,
        ));
    }
    s.push_str(&format!(
        "  occupancy ratio {:.2}x, dispatch reduction {:.1}%  [{}]\n",
        c.occupancy_ratio(),
        c.dispatch_reduction() * 100.0,
        if c.meets_occupancy_criterion() {
            "meets 28.1 occupancy limb"
        } else {
            "below 28.1 occupancy limb"
        }
    ));
    s
}

/// The full §26 baseline table for one workload.
pub fn baseline_report(knobs: &ControlKnobs, cohort_width: u16) -> String {
    let b = baselines(knobs, cohort_width);
    let mut s = String::new();
    s.push_str(&format!(
        "branching={} depth={} roots={} classes={} width={}\n",
        knobs.branching_factor,
        knobs.depth,
        knobs.process_count,
        knobs.class_count,
        cohort_width
    ));
    s.push_str(&format!(
        "  {:<22} dispatches={:<7} occupancy={:.3}  host_launches={}\n",
        "soma/persistent-fifo",
        b.fifo.dispatches(),
        b.fifo.lane_occupancy(),
        0
    ));
    s.push_str(&format!(
        "  {:<22} dispatches={:<7} occupancy={:.3}  host_launches={}\n",
        "soma/run-class",
        b.cohorted.dispatches(),
        b.cohorted.lane_occupancy(),
        0
    ));
    s.push_str(&format!(
        "  {:<22} dispatches={:<7} occupancy={:.3}  host_launches={}\n",
        "bulk-frontier/naive",
        b.bulk_unsorted.dispatches(),
        b.bulk_unsorted.lane_occupancy(),
        b.bulk_unsorted.host_launches
    ));
    s.push_str(&format!(
        "  {:<22} dispatches={:<7} occupancy={:.3}  host_launches={}\n",
        "bulk-frontier/sorted",
        b.bulk_sorted.dispatches(),
        b.bulk_sorted.lane_occupancy(),
        b.bulk_sorted.host_launches
    ));
    s.push_str(&format!(
        "  soma/bulk-sorted dispatch ratio {:.2}x  [{}]\n",
        b.dispatch_ratio_vs_bulk(),
        if b.within_bulk_tolerance() {
            "within 28.3 15% tolerance"
        } else {
            "OUTSIDE 28.3 15% tolerance"
        }
    ));
    s
}
