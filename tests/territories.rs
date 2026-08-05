//! Distribution across execution territories.
//!
//! The question this settles is whether cohorting is an artefact of assuming a
//! single global pool of ready work. It partly is: locality-blind distribution
//! destroys it. What rescues it is placement, so these tests pin both the
//! failure and the fix.

use soma::experiments::irregular_arrival::{soma_policy, trace, IrregularKnobs};
use soma::experiments::territories::{
    evaluate, territory_policy, Routing, TerritoryConfig,
};

fn knobs() -> IrregularKnobs {
    IrregularKnobs {
        roots: 64,
        arrival_span: 12,
        depth: 4,
        class_count: 4,
        ..IrregularKnobs::default()
    }
}

fn cfg(territories: u32, routing: Routing) -> TerritoryConfig {
    TerritoryConfig {
        territories,
        width: 32,
        routing,
        max_defer: 8,
    }
}

const ALL: [Routing; 5] = [
    Routing::Local,
    Routing::RoundRobin,
    Routing::ClassAffinity,
    Routing::ClassAffinityBalanced { cap: 64 },
    Routing::ProportionalAffinity,
];

#[test]
fn a_single_territory_reduces_to_the_undistributed_scheduler() {
    // With one territory there is nothing to place, so every routing policy
    // must collapse onto the global scheduler measured in the previous slice.
    let t = trace(&knobs());
    let reference = soma_policy(&t, 32, 8);

    for routing in ALL {
        let o = territory_policy(&t, &cfg(1, routing));
        assert_eq!(o.cost, reference.cost, "{routing:?}");
        assert_eq!(o.imbalance(), 1.0, "{routing:?}");
    }
}

#[test]
fn every_arrival_is_dispatched_exactly_once() {
    let t = trace(&knobs());
    for territories in [1u32, 4, 16, 64] {
        for routing in ALL {
            let o = territory_policy(&t, &cfg(territories, routing));
            assert_eq!(o.items(), t.len(), "{territories} {routing:?}");
            assert_eq!(
                o.cost.useful_lane_slots,
                t.len() as u64,
                "{territories} {routing:?}"
            );
            assert_eq!(
                o.per_territory.iter().sum::<u64>(),
                o.dispatches(),
                "per-territory counts must account for every dispatch"
            );
        }
    }
}

#[test]
fn placement_is_deterministic() {
    let t = trace(&knobs());
    for routing in ALL {
        let a = territory_policy(&t, &cfg(16, routing));
        let b = territory_policy(&t, &cfg(16, routing));
        assert_eq!(a.cost, b.cost, "{routing:?}");
        assert_eq!(a.per_territory, b.per_territory, "{routing:?}");
    }
}

// ---- the failure ---------------------------------------------------------

#[test]
fn locality_blind_distribution_shreds_cohorts() {
    // The honest bad news: spreading work evenly across territories fragments
    // every run class until cohorts cannot fill. This is what the global
    // occupancy numbers elsewhere in this crate assumed away.
    let t = trace(&knobs());
    let one = evaluate(&t, &cfg(1, Routing::RoundRobin)).occupancy;
    let many = evaluate(&t, &cfg(64, Routing::RoundRobin)).occupancy;

    assert!(
        many < one * 0.5,
        "expected fragmentation to halve occupancy at least: {one:.3} -> {many:.3}"
    );

    // And it gets monotonically worse as the machine gets wider.
    let series: Vec<f64> = [1u32, 4, 16, 64]
        .iter()
        .map(|n| evaluate(&t, &cfg(*n, Routing::RoundRobin)).occupancy)
        .collect();
    assert!(
        series.windows(2).all(|w| w[1] <= w[0] + 1e-9),
        "occupancy must not improve with more territories under blind routing: {series:?}"
    );
}

#[test]
fn class_affinity_fills_cohorts_but_idles_the_machine() {
    // The opposite failure. Concentrating a class perfectly fills its cohorts,
    // but a class can only occupy one territory, so everything past the class
    // count sits idle. High occupancy on a machine that is mostly switched off
    // is not a win, which is what `effective_occupancy` exists to say.
    let t = trace(&knobs());
    let r = evaluate(&t, &cfg(64, Routing::ClassAffinity));

    assert!(r.occupancy > 0.95, "cohorts do fill: {:.3}", r.occupancy);
    assert_eq!(
        r.idle_territories,
        64 - knobs().class_count as usize,
        "only one territory per class does any work"
    );
    assert!(r.imbalance > 10.0, "imbalance={:.2}", r.imbalance);
    assert!(
        r.effective_occupancy < 0.1,
        "effective occupancy must expose the idle machine: {:.3}",
        r.effective_occupancy
    );
}

// ---- the fix -------------------------------------------------------------

#[test]
fn proportional_affinity_recovers_both_fill_and_balance() {
    // Sizing each class a block of territories by its share decouples the two
    // failures above: concentration fills cohorts, block width keeps the
    // machine busy.
    let t = trace(&knobs());
    let mut ratios = Vec::new();
    for territories in [16u32, 64] {
        let p = evaluate(&t, &cfg(territories, Routing::ProportionalAffinity));
        let blind = evaluate(&t, &cfg(territories, Routing::RoundRobin));
        let affinity = evaluate(&t, &cfg(territories, Routing::ClassAffinity));

        assert_eq!(p.idle_territories, 0, "T={territories}");
        assert!(p.imbalance < 1.5, "T={territories}: {:.2}", p.imbalance);
        assert!(
            p.occupancy > blind.occupancy,
            "T={territories}: {:.3} vs blind {:.3}",
            p.occupancy,
            blind.occupancy
        );
        assert!(
            p.effective_occupancy > affinity.effective_occupancy * 3.0,
            "T={territories}: {:.3} vs affinity {:.3}",
            p.effective_occupancy,
            affinity.effective_occupancy
        );
        ratios.push(p.occupancy / blind.occupancy);
    }

    // The gain over blind routing is small on a narrow machine and large on a
    // wide one, because fragmentation is what placement is defending against.
    assert!(
        ratios[1] > ratios[0],
        "the advantage must grow with machine width: {ratios:?}"
    );
    assert!(
        ratios[1] > 2.0,
        "at 64 territories placement should more than double occupancy: {:.2}x",
        ratios[1]
    );
}

#[test]
fn distribution_still_costs_something() {
    // Placement rescues cohorting but does not make distribution free. Saying
    // so keeps the result from being oversold.
    let t = trace(&knobs());
    let one = evaluate(&t, &cfg(1, Routing::ProportionalAffinity)).occupancy;
    let wide = evaluate(&t, &cfg(64, Routing::ProportionalAffinity)).occupancy;

    assert!(
        wide < one,
        "a wider machine should still lose some fill: {one:.3} -> {wide:.3}"
    );
    assert!(
        wide > 0.85,
        "but not much of it, or the mechanism does not survive: {wide:.3}"
    );
}
