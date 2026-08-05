//! The irregular-arrival experiment (§25.1, §27, §29).
//!
//! This is the experiment the thesis rests on, so the tests are weighted toward
//! the things that would make its numbers meaningless rather than toward the
//! numbers themselves: that both policies dispatch exactly the arrivals they
//! were given, that neither is scored on a different trace than the other, and
//! that the level-synchronous control still shows no advantage.

use soma::experiments::irregular_arrival::{
    bulk_policy, frontiers, regime_map, regime_point, soma_policy, trace, IrregularKnobs,
    RegimePoint,
};

fn level_synchronous() -> IrregularKnobs {
    IrregularKnobs {
        arrival_span: 0,
        jitter: 0,
        ..IrregularKnobs::default()
    }
}

fn irregular() -> IrregularKnobs {
    IrregularKnobs::default()
}

fn tree_size(k: &IrregularKnobs) -> u64 {
    let b = k.branching_factor as u64;
    let mut total = 0u64;
    let mut level = 1u64;
    for _ in 0..=k.depth {
        total += level;
        level = level.saturating_mul(b);
    }
    k.roots as u64 * total
}

// ---- the trace itself -----------------------------------------------------

#[test]
fn the_trace_covers_the_whole_tree_exactly_once() {
    for knobs in [level_synchronous(), irregular()] {
        let t = trace(&knobs);
        assert_eq!(t.len() as u64, tree_size(&knobs));

        let mut ids: Vec<u64> = t.events.iter().map(|e| e.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), t.len(), "every arrival is a distinct node");

        assert!(
            t.events.windows(2).all(|w| w[0].tick <= w[1].tick),
            "the trace must be in tick order"
        );
    }
}

#[test]
fn irregularity_spreads_the_trace_out_in_time() {
    let level = trace(&level_synchronous());
    let irreg = trace(&irregular());

    assert_eq!(level.len(), irreg.len(), "same work, different timing");
    assert!(
        irreg.horizon() > level.horizon(),
        "staggered arrival and jitter must widen the horizon: {} vs {}",
        irreg.horizon(),
        level.horizon()
    );
    assert!(irreg.arrival_rate() < level.arrival_rate());
}

#[test]
fn trace_generation_is_deterministic() {
    let a = trace(&irregular());
    let b = trace(&irregular());
    assert_eq!(a.events, b.events);
}

// ---- conservation: the load-bearing correctness property ------------------

#[test]
fn both_policies_dispatch_every_arrival_exactly_once() {
    // If a policy silently drops or double-counts work, its occupancy is
    // meaningless and the comparison is worthless. This is the check that makes
    // every other number in this experiment trustworthy.
    for knobs in [level_synchronous(), irregular()] {
        let t = trace(&knobs);
        for width in [8u16, 32] {
            for knob in [0u32, 1, 4, 16] {
                let s = soma_policy(&t, width, knob);
                let b = bulk_policy(&t, width, knob.max(1));

                assert_eq!(s.items(), t.len(), "soma knob={knob} width={width}");
                assert_eq!(b.items(), t.len(), "bulk knob={knob} width={width}");
                assert_eq!(
                    s.cost.useful_lane_slots,
                    t.len() as u64,
                    "soma useful lanes must equal the arrivals dispatched"
                );
                assert_eq!(b.cost.useful_lane_slots, t.len() as u64);
            }
        }
    }
}

#[test]
fn the_waiting_knob_actually_bounds_the_waiting() {
    // Waits are `dispatch_tick - ready_tick` on unsigned ticks, so dispatching
    // anything before it was ready would underflow and panic in debug. Beyond
    // that, the knob must mean what it says: raising it must not lower the
    // waiting, and no wait may exceed the trace horizon.
    let t = trace(&irregular());
    let mut previous_max = 0u32;
    for knob in [0u32, 2, 8, 32] {
        let s = soma_policy(&t, 32, knob);
        let max_wait = s.waits.iter().max().copied().unwrap_or(0);
        assert!(max_wait <= t.horizon(), "knob={knob}");
        assert!(
            max_wait >= previous_max,
            "a larger defer budget must not shorten waits: knob={knob}"
        );
        previous_max = max_wait;
    }
    assert!(previous_max > 0, "some work must actually have waited");
}

#[test]
fn policies_are_deterministic() {
    let t = trace(&irregular());
    let a = soma_policy(&t, 32, 4);
    let b = soma_policy(&t, 32, 4);
    assert_eq!(a.cost, b.cost);
    assert_eq!(a.waits, b.waits);
}

// ---- the control ---------------------------------------------------------

#[test]
fn level_synchronous_work_shows_no_advantage() {
    // The same tie the bulk frontier baseline found, reproduced through an
    // entirely separate code path. If this ever shows an advantage, the
    // experiment is measuring its own machinery.
    let p = regime_point(&level_synchronous(), 32, 1.0);
    assert!(p.is_level_synchronous());
    assert!(
        (p.advantage - 1.0).abs() < 1e-9,
        "expected parity on level-synchronous work, got {:.3}x",
        p.advantage
    );
    assert!((p.soma_occupancy - p.bulk_occupancy).abs() < 1e-9);
}

#[test]
fn a_zero_wait_budget_removes_the_advantage_entirely() {
    // With no waiting permitted, neither policy can accumulate anything, so
    // both dispatch whatever is ready and reach identical occupancy. The
    // advantage is bought with waiting, and this proves it.
    let t = trace(&irregular());
    let s = soma_policy(&t, 32, 0);
    let b = bulk_policy(&t, 32, 1);

    assert_eq!(s.cost.dispatches, b.cost.dispatches);
    assert!((s.occupancy() - b.occupancy()).abs() < 1e-9);
    assert_eq!(s.p99_wait(), 0);
    assert_eq!(b.p99_wait(), 0);
}

// ---- the finding ---------------------------------------------------------

#[test]
fn irregular_arrival_separates_the_two_frontiers() {
    let p = regime_point(&irregular(), 32, 1.0);

    assert!(
        p.advantage >= 1.5,
        "expected >= 1.5x occupancy at a matched latency budget, got {:.2}x",
        p.advantage
    );
    assert!(
        p.soma_occupancy > p.bulk_occupancy,
        "{:.3} vs {:.3}",
        p.soma_occupancy,
        p.bulk_occupancy
    );
    assert!(
        p.wait_reduction() > 5.0,
        "SOMA should reach the same occupancy with far less waiting, got {:.1}x",
        p.wait_reduction()
    );
}

#[test]
fn the_advantage_needs_irregularity_and_grows_with_it() {
    let base = IrregularKnobs::default();
    let none = regime_point(
        &IrregularKnobs {
            arrival_span: 0,
            jitter: 0,
            ..base
        },
        32,
        1.0,
    );
    let some = regime_point(
        &IrregularKnobs {
            arrival_span: 0,
            jitter: 1,
            ..base
        },
        32,
        1.0,
    );
    let more = regime_point(
        &IrregularKnobs {
            arrival_span: 12,
            jitter: 1,
            ..base
        },
        32,
        1.0,
    );

    assert!((none.advantage - 1.0).abs() < 1e-9);
    assert!(some.advantage > none.advantage);
    assert!(more.advantage > some.advantage);
}

#[test]
fn the_regime_map_is_not_a_single_lucky_point() {
    // §29's Outcome B is that cohorting helps only in narrow synthetic corners.
    // Check the advantage across the sweep rather than at one setting.
    let map = regime_map(
        &IrregularKnobs::default(),
        &[0, 4, 12, 32],
        &[0, 1, 3, 8],
        &[1, 4],
        32,
        1.0,
    );

    let (level, irreg): (Vec<&RegimePoint>, Vec<&RegimePoint>) =
        map.iter().partition(|p| p.is_level_synchronous());

    assert!(
        level.iter().all(|p| (p.advantage - 1.0).abs() < 1e-9),
        "every level-synchronous cell must be a tie"
    );
    assert!(
        irreg.iter().all(|p| p.advantage >= 1.0),
        "cohorting must never be worse than the manual batch"
    );

    let helped = irreg.iter().filter(|p| p.advantage >= 1.25).count();
    assert!(
        helped * 2 >= irreg.len(),
        "the advantage should hold broadly, not in a corner: {helped}/{}",
        irreg.len()
    );
}

#[test]
fn sparse_arrivals_starve_both_policies() {
    // The honest limit. When arrivals per class per tick fall far below the
    // cohort width, nobody can fill a cohort and the advantage collapses.
    let starved = regime_point(
        &IrregularKnobs {
            arrival_span: 32,
            jitter: 8,
            class_count: 4,
            ..IrregularKnobs::default()
        },
        32,
        1.0,
    );
    assert!(
        starved.advantage < 1.25,
        "expected the advantage to collapse under starvation, got {:.2}x",
        starved.advantage
    );
    assert!(starved.soma_occupancy < 0.5);
}

// ---- host orchestration --------------------------------------------------

#[test]
fn only_the_bulk_policy_needs_the_host() {
    let t = trace(&irregular());
    let s = soma_policy(&t, 32, 4);
    let b = bulk_policy(&t, 32, 4);

    assert_eq!(s.host_launches, 0, "SOMA submits nothing per operation");
    assert!(b.host_launches > 0);

    // A wider window trades host launches for latency.
    let narrow = bulk_policy(&t, 32, 1);
    assert!(narrow.host_launches > b.host_launches);
    assert!(narrow.mean_wait() < b.mean_wait());
}

#[test]
fn waiting_longer_buys_occupancy_for_both() {
    let t = trace(&irregular());
    let f = frontiers(&t, 32, &[0, 1, 2, 4, 8, 16, 32]);

    for policy in [&f.soma, &f.bulk] {
        assert!(
            policy.windows(2).all(|w| w[1].occupancy >= w[0].occupancy - 1e-9),
            "occupancy must be non-decreasing in the waiting knob"
        );
        assert!(policy.windows(2).all(|w| w[1].mean_wait >= w[0].mean_wait - 1e-9));
    }
}
