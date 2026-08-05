//! Step 7 verification: the bulk frontier baseline (§26).
//!
//! §26 is blunt about why this baseline exists — without it SOMA might appear
//! successful only because the baselines are weak. These tests therefore pin
//! the *unflattering* result as firmly as the flattering one: on
//! level-synchronous work a competent manual batch matches SOMA exactly, and if
//! that ever stops being true the reason should be a deliberate change rather
//! than an accident.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{Kind, Ref64};
use soma::compiler::run_classes::SEARCH_BRANCH;
use soma::experiments::bulk_frontier;
use soma::experiments::cohort_study::baselines;
use soma::experiments::dynamic_search::ControlKnobs;
use soma::scheduler::cohorts::{build_cohorts, dispatch_cost};

fn divergent() -> ControlKnobs {
    ControlKnobs {
        class_count: 4,
        depth: 5,
        branching_factor: 3,
        process_count: 4,
        ..ControlKnobs::default()
    }
}

fn homogeneous() -> ControlKnobs {
    ControlKnobs {
        class_count: 1,
        ..divergent()
    }
}

// ---- the two scoring paths must agree ------------------------------------

#[test]
fn dispatch_cost_matches_build_cohorts() {
    // SOMA is scored through `build_cohorts`, the bulk frontier through
    // `dispatch_cost`. A comparison between them is only valid if the two
    // agree lane for lane on identical input.
    let specs: Vec<Vec<u32>> = vec![
        vec![10; 10],
        vec![10, 11, 10, 12, 11, 11, 13, 10],
        vec![10, 10, 10, 11],
        (0..37).map(|i| SEARCH_BRANCH + (i % 5)).collect(),
        vec![],
    ];

    for classes in specs {
        for width in [1u16, 4, 8, 32] {
            let lanes: Vec<(Ref64, u32)> = classes
                .iter()
                .enumerate()
                .map(|(i, rc)| (Ref64::new(i as u32 + 1, 0, Kind::Continuation), *rc))
                .collect();
            let plan = build_cohorts(&lanes, width, PartialCohortPolicy::RunPartial);
            let cost = dispatch_cost(&classes, width);

            assert_eq!(
                plan.cohorts.len() as u64,
                cost.dispatches,
                "classes={classes:?} width={width}"
            );
            assert_eq!(plan.lane_slots(), cost.lane_slots, "width={width}");
            assert_eq!(
                plan.useful_lane_slots(),
                cost.useful_lane_slots,
                "width={width}"
            );
        }
    }
}

// ---- the baseline does the same work -------------------------------------

#[test]
fn bulk_frontier_expands_the_same_tree_as_soma() {
    for knobs in [homogeneous(), divergent()] {
        let bulk = bulk_frontier::run(&knobs, 32, true);
        assert_eq!(
            bulk.nodes_expanded,
            knobs.node_count(),
            "the baseline must search the same tree, not a smaller one"
        );
        assert_eq!(
            bulk.levels,
            knobs.depth + 1,
            "a level-synchronous search takes one level per tree level"
        );
        assert_eq!(bulk.host_launches, bulk.levels as u64);
        assert_eq!(bulk.global_barriers, bulk.levels as u64);
    }
}

#[test]
fn bulk_frontier_is_deterministic() {
    let knobs = divergent();
    for sorted in [false, true] {
        let a = bulk_frontier::run(&knobs, 32, sorted);
        let b = bulk_frontier::run(&knobs, 32, sorted);
        assert_eq!(a.cost, b.cost);
        assert_eq!(a.nodes_expanded, b.nodes_expanded);
    }
}

// ---- sorting is what makes the baseline strong ---------------------------

#[test]
fn sorting_the_frontier_is_what_makes_the_manual_batch_strong() {
    let knobs = divergent();
    let naive = bulk_frontier::run(&knobs, 32, false);
    let sorted = bulk_frontier::run(&knobs, 32, true);

    assert!(
        sorted.dispatches() < naive.dispatches(),
        "partitioning by run class must reduce dispatches: {} vs {}",
        sorted.dispatches(),
        naive.dispatches()
    );
    assert!(sorted.lane_occupancy() > naive.lane_occupancy());

    // On homogeneous work there is nothing to sort, so the variants coincide.
    let h = homogeneous();
    assert_eq!(
        bulk_frontier::run(&h, 32, false).cost,
        bulk_frontier::run(&h, 32, true).cost
    );
}

// ---- the honest headline -------------------------------------------------

#[test]
fn a_competent_manual_batch_ties_soma_on_level_synchronous_work() {
    // This is the finding the baseline exists to surface. SOMA's cohorting
    // gains nothing in lane efficiency over a hand-written frontier kernel that
    // partitions each level by run class — because on level-synchronous work
    // both end up forming exactly the same groups.
    //
    // If this test starts failing in SOMA's favour, check whether the workload
    // stopped being level-synchronous before believing the win.
    for knobs in [homogeneous(), divergent()] {
        for width in [8u16, 32] {
            let b = baselines(&knobs, width);
            assert_eq!(
                b.cohorted.dispatches(),
                b.bulk_sorted.dispatches(),
                "classes={} width={width}",
                knobs.class_count
            );
            assert!(
                (b.cohorted.lane_occupancy() - b.bulk_sorted.lane_occupancy()).abs() < 1e-9,
                "classes={} width={width}",
                knobs.class_count
            );
            assert!((b.dispatch_ratio_vs_bulk() - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn soma_stays_within_the_bulk_tolerance() {
    // §28.3: on work already well suited to bulk execution, SOMA must remain
    // within 15% of the manually batched implementation.
    for knobs in [homogeneous(), divergent()] {
        let b = baselines(&knobs, 32);
        assert!(
            b.within_bulk_tolerance(),
            "classes={} ratio={:.3}",
            knobs.class_count,
            b.dispatch_ratio_vs_bulk()
        );
    }
}

#[test]
fn soma_beats_only_the_weak_baseline() {
    // Cohorting's large win is over the persistent FIFO, not over a competent
    // manual batch. Stating both keeps the FIFO number in proportion.
    let b = baselines(&divergent(), 32);

    assert!(
        b.cohorted.lane_occupancy() > b.fifo.lane_occupancy() * 1.5,
        "cohorting should clear the FIFO baseline by a wide margin"
    );
    assert!(
        b.bulk_sorted.lane_occupancy() > b.fifo.lane_occupancy() * 1.5,
        "so should a sorted manual batch — the FIFO is simply weak"
    );
}

// ---- what SOMA does have over the baseline -------------------------------

#[test]
fn the_baseline_needs_the_host_and_soma_does_not() {
    // §28.4: the host must not submit or schedule individual search operations.
    // The bulk frontier needs a launch and a barrier per level; SOMA's epoch
    // loop is device-resident by design and submits nothing per operation.
    let knobs = divergent();
    let b = baselines(&knobs, 32);

    assert_eq!(b.host_launches_avoided(), (knobs.depth + 1) as u64);
    assert!(b.bulk_sorted.host_launches > 0);
    assert!(b.bulk_unsorted.global_barriers > 0);
}
