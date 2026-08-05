//! Step 6 verification: run-class bins and continuation cohorting (§14, §26).
//!
//! The claim under test is narrow and structural: binning by run class makes
//! SIMD dispatches uniform, and a persistent FIFO does not. These tests pin the
//! cohort-construction arithmetic, the partial-cohort policies, and the
//! occupancy comparison — including its negative control, where a workload with
//! a single run class must show cohorting buying exactly nothing.

use soma::abi::cohorts::{PartialCohortPolicy, MAX_COHORT_WIDTH};
use soma::abi::{EventKind, Kind, Ref64};
use soma::compiler::run_classes::{search_class, SEARCH_BRANCH};
use soma::experiments::cohort_study::{compare, run};
use soma::experiments::dynamic_search::{build_in, ControlKnobs};
use soma::kernel::Kernel;
use soma::replay::trace_reader::{events_of, same_trace};
use soma::scheduler::cohorts::build_cohorts;
use soma::scheduler::runnable_bins::SchedulingMode;

fn lanes(spec: &[(u32, u32)]) -> Vec<(Ref64, u32)> {
    // (slot, run_class) pairs -> (continuation ref, run class)
    spec.iter()
        .map(|(slot, rc)| (Ref64::new(*slot, 0, Kind::Continuation), *rc))
        .collect()
}

// ---- §14: cohort construction --------------------------------------------

#[test]
fn uniform_bin_splits_into_full_cohorts_plus_a_remainder() {
    // §14: full = floor(n / W), remaining = n mod W.
    let bin = lanes(&(1..=10).map(|s| (s, SEARCH_BRANCH)).collect::<Vec<_>>());
    let plan = build_cohorts(&bin, 4, PartialCohortPolicy::RunPartial);

    assert_eq!(plan.cohorts.len(), 3, "floor(10/4) full + 1 partial");
    assert_eq!(plan.cohorts.iter().filter(|c| c.is_full()).count(), 2);
    assert_eq!(plan.cohorts[2].active_lanes, 2, "10 mod 4");
    assert_eq!(plan.cohorts[2].idle_lanes(), 2);
    assert!(plan.deferred.is_empty());

    // Every lane accounted for exactly once, in bin order.
    let ordered: Vec<u32> = plan
        .cohorts
        .iter()
        .flat_map(|c| c.lanes().iter().map(|r| r.slot))
        .collect();
    assert_eq!(ordered, (1..=10).collect::<Vec<u32>>());

    assert_eq!(plan.useful_lane_slots(), 10);
    assert_eq!(plan.lane_slots(), 12, "three dispatches of width 4");
}

#[test]
fn a_lane_group_spanning_k_run_classes_costs_k_dispatches() {
    // One width-4 lane group holding three distinct classes cannot be one
    // uniform dispatch (§15) — it is three, each with the others masked off.
    let bin = lanes(&[(1, 10), (2, 11), (3, 10), (4, 12)]);
    let plan = build_cohorts(&bin, 4, PartialCohortPolicy::RunPartial);

    assert_eq!(plan.cohorts.len(), 3);
    assert_eq!(plan.lane_slots(), 12, "three width-4 dispatches");
    assert_eq!(plan.useful_lane_slots(), 4, "only four real continuations");

    // Every cohort is uniform, and classes appear in first-appearance order.
    let classes: Vec<u32> = plan.cohorts.iter().map(|c| c.run_class).collect();
    assert_eq!(classes, vec![10, 11, 12]);
    assert_eq!(plan.cohorts[0].active_lanes, 2, "both class-10 lanes coalesce");
    assert!(plan.cohorts.iter().all(|c| !c.is_full()));
}

#[test]
fn cohort_width_is_clamped_to_the_abi_maximum() {
    let bin = lanes(&(1..=40).map(|s| (s, SEARCH_BRANCH)).collect::<Vec<_>>());
    let plan = build_cohorts(&bin, 64, PartialCohortPolicy::RunPartial);
    assert!(plan
        .cohorts
        .iter()
        .all(|c| c.width as usize <= MAX_COHORT_WIDTH));
    assert_eq!(plan.useful_lane_slots(), 40);
}

// ---- §14: partial-cohort policies ----------------------------------------

#[test]
fn partial_policies_differ_only_on_the_remainder() {
    let bin = lanes(&(1..=6).map(|s| (s, SEARCH_BRANCH)).collect::<Vec<_>>());

    let run_partial = build_cohorts(&bin, 4, PartialCohortPolicy::RunPartial);
    assert_eq!(run_partial.cohorts.len(), 2);
    assert_eq!(run_partial.lane_slots(), 8, "the remainder wastes two lanes");
    assert!(run_partial.deferred.is_empty());

    let deferred = build_cohorts(&bin, 4, PartialCohortPolicy::Defer);
    assert_eq!(deferred.cohorts.len(), 1, "only the full cohort dispatches");
    assert_eq!(deferred.deferred.len(), 2, "the remainder waits for a refill");
    assert_eq!(deferred.lane_slots(), 4);

    let spilled = build_cohorts(&bin, 4, PartialCohortPolicy::SendToCpu);
    assert_eq!(spilled.cohorts.len(), 3, "one full cohort + two scalar lanes");
    assert_eq!(spilled.lane_slots(), 6, "scalar lanes waste nothing");
    assert_eq!(spilled.useful_lane_slots(), 6);
    assert!(spilled.deferred.is_empty());

    // Phase 1 has no generic class, so the merge policy runs the partial.
    let merged = build_cohorts(&bin, 4, PartialCohortPolicy::MergeWithGenericClass);
    assert_eq!(merged.cohorts.len(), run_partial.cohorts.len());
    assert_eq!(merged.lane_slots(), run_partial.lane_slots());
}

#[test]
fn defer_policy_still_reaches_quiescence() {
    // Deferring every partial cohort could starve a run class that never fills.
    // The epoch guard must force progress rather than spin.
    let knobs = ControlKnobs {
        class_count: 4,
        depth: 3,
        branching_factor: 2,
        process_count: 1,
        ..ControlKnobs::default()
    };
    let deferred = run(
        &knobs,
        SchedulingMode::RunClassBins,
        32,
        PartialCohortPolicy::Defer,
    );
    let eager = run(
        &knobs,
        SchedulingMode::RunClassBins,
        32,
        PartialCohortPolicy::RunPartial,
    );

    assert!(deferred.epochs > 0);
    assert_eq!(
        deferred.accounting.steps, eager.accounting.steps,
        "deferring changes when work runs, never whether it runs"
    );
}

// ---- §26 / §28.1: the comparison -----------------------------------------

#[test]
fn a_single_run_class_gives_cohorting_no_advantage() {
    // The negative control. With one run class the FIFO's arrival order is
    // already uniform, so run-class binning cannot improve on it. If this ever
    // shows a gain, the measurement is measuring itself.
    let knobs = ControlKnobs {
        class_count: 1,
        depth: 4,
        branching_factor: 3,
        process_count: 2,
        ..ControlKnobs::default()
    };
    let c = compare(&knobs, 32);

    assert!(
        (c.occupancy_ratio() - 1.0).abs() < 1e-9,
        "expected parity on homogeneous work, got {:.4}x",
        c.occupancy_ratio()
    );
    assert_eq!(c.fifo.dispatches(), c.cohorted.dispatches());
    assert!(!c.meets_occupancy_criterion());
}

#[test]
fn divergent_work_meets_the_occupancy_criterion() {
    let knobs = ControlKnobs {
        class_count: 4,
        depth: 5,
        branching_factor: 3,
        process_count: 4,
        ..ControlKnobs::default()
    };
    let c = compare(&knobs, 32);

    assert!(
        c.meets_occupancy_criterion(),
        "expected >= 1.5x useful lane occupancy, got {:.2}x",
        c.occupancy_ratio()
    );
    assert!(
        c.cohorted.dispatches() < c.fifo.dispatches(),
        "cohorting must eliminate dispatches, not just relabel them"
    );
    assert!(c.dispatch_reduction() > 0.0);
}

#[test]
fn both_modes_execute_exactly_the_same_work() {
    // Occupancy is only meaningful if the two runs did identical work. The
    // scheduling mode must change how continuations are grouped, nothing else.
    let knobs = ControlKnobs {
        class_count: 4,
        depth: 4,
        branching_factor: 3,
        process_count: 2,
        ..ControlKnobs::default()
    };
    let c = compare(&knobs, 16);
    assert!(
        c.executed_identical_work(),
        "fifo steps={} lanes={}, cohorted steps={} lanes={}",
        c.fifo.accounting.steps,
        c.fifo.accounting.useful_lane_slots,
        c.cohorted.accounting.steps,
        c.cohorted.accounting.useful_lane_slots
    );

    // And the same node population, reached independently of binning.
    let expected = knobs.node_count();
    for mode in [SchedulingMode::PersistentFifo, SchedulingMode::RunClassBins] {
        let mut kernel = Kernel::with_mode(mode);
        kernel.configure_cohorts(16, Default::default());
        let mut kernel = build_in(kernel, &knobs);
        kernel.run_to_quiescence(100_000);
        assert_eq!(kernel.process_count() as u64, expected, "{mode:?}");
        assert_eq!(kernel.total_pending(), 0, "{mode:?}");
    }
}

// ---- determinism and tracing ---------------------------------------------

#[test]
fn cohorted_execution_is_deterministic_and_traced() {
    let knobs = ControlKnobs {
        class_count: 4,
        depth: 3,
        branching_factor: 3,
        process_count: 2,
        ..ControlKnobs::default()
    };

    let mut kernels = Vec::new();
    for _ in 0..2 {
        let mut kernel = Kernel::with_mode(SchedulingMode::RunClassBins);
        kernel.configure_cohorts(8, Default::default());
        let mut kernel = build_in(kernel, &knobs);
        kernel.run_to_quiescence(100_000);
        kernels.push(kernel);
    }
    assert!(
        same_trace(&kernels[0], &kernels[1]),
        "cohort construction must not depend on map iteration order"
    );

    let kernel = &kernels[0];
    let cohort_events: Vec<_> = events_of(kernel, EventKind::CohortCreated).collect();
    assert_eq!(
        cohort_events.len() as u64,
        kernel.accounting().cohorts,
        "every dispatch emits exactly one CohortCreated event (§21)"
    );
    assert!(
        cohort_events.iter().all(|e| e.auxiliary > 0),
        "no dispatch is issued with zero active lanes"
    );
    assert!(cohort_events
        .iter()
        .all(|e| search_class(0, 1) <= e.run_class));
}

#[test]
fn width_one_cohorting_reduces_to_scalar_execution() {
    // The default width must be a no-op: one lane per dispatch, nothing idle.
    let knobs = ControlKnobs::default();
    let mut kernel = build_in(Kernel::new(), &knobs);
    kernel.run_to_quiescence(100_000);

    assert_eq!(kernel.cohort_width(), 1);
    assert_eq!(kernel.accounting().idle_lane_slots, 0);
    assert_eq!(
        kernel.accounting().lane_slots,
        kernel.accounting().useful_lane_slots
    );
    assert!((kernel.accounting().lane_occupancy() - 1.0).abs() < 1e-9);
    assert!((kernel.accounting().cohort_fill_ratio() - 1.0).abs() < 1e-9);
}
