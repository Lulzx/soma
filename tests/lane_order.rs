//! §4.6: an epoch's lanes are reorderable, checked by reordering them.
//!
//! Canonical commit (§4.5) made an epoch's commit independent of the order its
//! lanes ran, and I25 is the obligation that pays for it. Neither was ever
//! exercised: the executive ran its lanes in a `for` loop, in plan order, every
//! time, so "the order does not matter" was a property of a machine that only
//! ever chose one order.
//!
//! These tests choose others. A workload runs under plan order, reversed, and
//! two seeded permutations, and the runs have to agree — on what happened
//! (I18), on what an epoch committed and in what order (I24's record), and on
//! the invariants. Every acceptance carries the null that the orders really did
//! differ, because a `LaneOrder` that silently did nothing would pass all of
//! this perfectly.
//!
//! This is deliberately not a threaded executive. A permutation exercises the
//! property threads need — no lane observes another within its epoch, and
//! commit does not care who finished first — while staying deterministic, so a
//! defect is a reproducible failure at a fixed place rather than an
//! intermittent corruption.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::Kernel;
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order};

const ORDERS: [LaneOrder; 4] = [
    LaneOrder::Plan,
    LaneOrder::Reverse,
    LaneOrder::Permuted(0x5EED),
    LaneOrder::Permuted(0xA11CE),
];

fn search(order: LaneOrder, width: u16) -> Kernel {
    let knobs = ControlKnobs {
        branching_factor: 3,
        depth: 3,
        process_count: 2,
        class_count: 3,
        ..ControlKnobs::default()
    };
    let mut kernel = build(&knobs);
    kernel.configure_cohorts(width, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(200);
    kernel
}

fn expand(order: LaneOrder, width: u16) -> Kernel {
    use soma::compiler::state_machine_lowering::create_expand;
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 7);
    kernel.configure_cohorts(width, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(200);
    kernel
}

// ---- the runs agree -------------------------------------------------------

#[test]
fn a_reordered_run_reproduces_the_plan_order_run() {
    // The claim §4.5 made and could not test. If any lane of an epoch observed
    // another, this is where it surfaces — as a difference in what happened,
    // not as a difference in when.
    for width in [2u16, 4, 16] {
        for order in ORDERS {
            let reference = search(LaneOrder::Plan, width);
            let candidate = search(order, width);
            // Sorted by position first, which is the obligation §4.2 states for
            // an executive that does not append in plan order. The reference is
            // unchanged by the sort — its emission order already is its
            // position order — so this weakens nothing about the comparison.
            let violations = conforms_traces(
                &in_position_order(&reference.trace_snapshot()),
                &in_position_order(&candidate.trace_snapshot()),
            );
            assert!(
                violations.is_empty(),
                "width {width}, {order:?}: {:?}",
                violations.first().map(|v| v.to_string())
            );
        }
    }
}

#[test]
fn reordering_is_a_placement_and_i19_treats_it_as_one() {
    // Where work runs and in what order are the same kind of decision, so a
    // reordering belongs in the same set a cohort width does. I19 is the clause
    // that says a placement cannot change what a program observes.
    for width in [1u16, 8] {
        let runs: Vec<Vec<_>> = ORDERS
            .iter()
            .map(|order| in_position_order(&expand(*order, width).trace_snapshot()))
            .collect();
        for (index, candidate) in runs.iter().enumerate().skip(1) {
            let violations = conforms_traces(&runs[0], candidate);
            assert!(
                violations.is_empty(),
                "width {width}, {:?}: {:?}",
                ORDERS[index],
                violations.first().map(|v| v.to_string())
            );
        }
    }
}

#[test]
fn every_order_leaves_a_legal_state() {
    for order in ORDERS {
        for kernel in [search(order, 4), expand(order, 8)] {
            let violations = check(&kernel);
            assert!(violations.is_empty(), "{order:?}: {:?}", violations.first());
        }
    }
}

#[test]
fn no_lane_observes_another_under_any_order() {
    // I25 is what makes reordering legal, so it is worth asking of the
    // reordered runs specifically and not only of the plan-order one.
    for order in ORDERS {
        for kernel in [search(order, 16), expand(order, 16)] {
            let violations: Vec<_> = check(&kernel)
                .into_iter()
                .filter(|v| v.invariant == Invariant::LaneIndependence)
                .collect();
            assert!(violations.is_empty(), "{order:?}: {:?}", violations.first());
        }
    }
}

// ---- an epoch commits the same thing --------------------------------------

#[test]
fn commit_applies_the_same_effects_in_the_same_order_whatever_order_lanes_ran() {
    // The most direct statement of what canonical commit bought. The effect
    // log's *application* sequence is compared by the position each effect was
    // produced at, which is a place in the plan and so is stable across
    // execution orders. Comparing the continuations instead would compare
    // allocator output, which reordering legitimately changes — I18 already
    // handles that up to a renaming (§2.6), and this clause is about ordering.
    let reference: Vec<(u32, u32, u32)> = search(LaneOrder::Plan, 8)
        .effect_log()
        .iter()
        .map(|record| record.position())
        .collect();

    assert!(
        !reference.is_empty(),
        "the workload committed nothing, so the comparison is vacuous"
    );

    for order in ORDERS {
        let actual: Vec<(u32, u32, u32)> = search(order, 8)
            .effect_log()
            .iter()
            .map(|record| record.position())
            .collect();
        assert_eq!(
            actual, reference,
            "{order:?} committed its epoch in a different order"
        );
    }
}

// ---- the nulls ------------------------------------------------------------

#[test]
fn the_orders_really_do_run_lanes_differently() {
    // Without this, every acceptance above passes for a `LaneOrder` that does
    // nothing. The raw trace is the evidence: it is in *emission* order, so a
    // run whose lanes ran in a different order emits its events in a different
    // order, even though sorting by position recovers the same run.
    let plan = search(LaneOrder::Plan, 16).trace_snapshot();
    let mut differed = 0;
    for order in [
        LaneOrder::Reverse,
        LaneOrder::Permuted(0x5EED),
        LaneOrder::Permuted(0xA11CE),
    ] {
        let other = search(order, 16).trace_snapshot();
        let plan_positions: Vec<_> = plan.iter().map(|row| (row.lane, row.lane_sequence)).collect();
        let other_positions: Vec<_> = other
            .iter()
            .map(|row| (row.lane, row.lane_sequence))
            .collect();
        if plan_positions != other_positions {
            differed += 1;
        }
    }
    assert_eq!(
        differed, 3,
        "some order emitted its events in plan order anyway, so it did not reorder"
    );
}

#[test]
fn sorting_a_reordered_trace_by_position_recovers_the_plan_order_trace() {
    // The other half of the null, and the thing I23's clause 2 stops asking of
    // a reordering executive. The clause is not weakened — it is asked here, of
    // the sorted trace, against the run it must reproduce.
    let plan = search(LaneOrder::Plan, 16).trace_snapshot();
    for order in ORDERS {
        let mut other = search(order, 16).trace_snapshot();
        other.sort_by_key(|row| (row.epoch, row.lane, row.lane_sequence));
        let mut expected = plan.clone();
        expected.sort_by_key(|row| (row.epoch, row.lane, row.lane_sequence));
        assert_eq!(
            other.len(),
            expected.len(),
            "{order:?} emitted a different number of events"
        );
        for (index, (a, b)) in other.iter().zip(&expected).enumerate() {
            assert_eq!(
                (a.epoch, a.lane, a.lane_sequence, a.event_kind),
                (b.epoch, b.lane, b.lane_sequence, b.event_kind),
                "{order:?} differs from the plan-order run at sorted index {index}"
            );
        }
    }
}

#[test]
fn i23s_second_clause_still_holds_of_the_plan_order_executive() {
    // The exemption is scoped to the executives that need it. A plan-order run
    // is still required to emit in position order, which is what stops the
    // exemption from quietly disabling the clause everywhere.
    let kernel = search(LaneOrder::Plan, 16);
    let trace = kernel.trace_snapshot();
    for window in trace.windows(2) {
        let a = (window[0].epoch, window[0].lane, window[0].lane_sequence);
        let b = (window[1].epoch, window[1].lane, window[1].lane_sequence);
        assert!(a < b, "the plan-order executive emitted out of position order");
    }
    assert!(check(&kernel)
        .iter()
        .all(|v| v.invariant != Invariant::PositionDerivedEmission));
}
