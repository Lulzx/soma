//! The ant-colony workload (`experiments::ant_colony`).
//!
//! The workload exists to make cohorting visible on a population of independent
//! agents, so the tests are in two groups: the colony must actually be a colony
//! (persistent, deterministic, and forming trails), and the comparison it
//! supports must be a fair one.
//!
//! Per the repository's test discipline the controls are here too — the width-1
//! null, and a case proving the identical-world check can fail. A control that
//! never fails is not a control.

use soma::compiler::run_classes::{
    ANT_CARRY_FOOD, ANT_EXPLORE, ANT_FOLLOW_TRAIL, COLONY_AGGREGATE, WORLD_STEP,
};
use soma::experiments::ant_colony::{
    build, compare, field_totals, observe_ants, run_mode, ColonyKnobs,
};
use soma::kernel::Kernel;
use soma::scheduler::runnable_bins::SchedulingMode;
use soma::semantics::invariants::check;

/// Small enough to run under a debug build, large enough to forage.
fn knobs() -> ColonyKnobs {
    ColonyKnobs {
        width: 48,
        height: 48,
        colonies: 3,
        ants_per_colony: 24,
        food_sources: 4,
        epochs: 120,
        ..ColonyKnobs::default()
    }
}

fn run_epochs(kernel: &mut Kernel, epochs: u32) -> u32 {
    let mut ran = 0;
    while ran < epochs && kernel.total_pending() > 0 {
        kernel.run_epoch();
        ran += 1;
    }
    ran
}

/// Processes are *persistent*: nothing retires, nothing faults, and the
/// population is still there at the end. An ant whose step budget ran out would
/// show up here as a missing ant.
#[test]
fn the_population_persists() {
    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    let expected = colony.ant_count();

    let ran = run_epochs(&mut kernel, knobs.epochs);
    assert_eq!(ran, knobs.epochs, "the colony must never go quiet");

    let ants = observe_ants(&mut kernel, &colony);
    assert_eq!(ants.len(), expected, "every ant must still exist");
    assert!(
        ants.iter().all(|ant| ant.alive),
        "no ant may fault or complete during a normal run"
    );
    // One continuation per ant, per colony, and one for the world.
    assert_eq!(
        kernel.total_pending(),
        expected + knobs.colonies as usize + 1
    );
}

/// The machine's own rules, applied to this workload. The capability structure
/// is the part of the design most likely to be wrong — every shared object has
/// exactly one writer by construction, and `CapabilityIntegrity` is what says so.
#[test]
fn the_colony_satisfies_the_invariants() {
    let knobs = ColonyKnobs {
        epochs: 24,
        ..knobs()
    };
    let (mut kernel, _colony) = build(&knobs);
    run_epochs(&mut kernel, knobs.epochs);

    let violations = check(&kernel);
    assert!(
        violations.is_empty(),
        "the colony must not violate the specification: {violations:?}"
    );
}

/// Ants find food, carry it home, and lay trails others follow. Without this the
/// run-class histogram would be static and the workload would be a picture of
/// nothing.
#[test]
fn foraging_emerges() {
    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    let ran = run_epochs(&mut kernel, knobs.epochs);

    let (food_trail, home_trail) = field_totals(&mut kernel, &colony, ran);
    assert!(home_trail > 0, "outbound ants must lay a trail home");
    assert!(
        food_trail > 0,
        "some ant must have carried food and laid a trail to it"
    );

    let ants = observe_ants(&mut kernel, &colony);
    let delivered: u64 = ants.iter().map(|ant| ant.delivered as u64).sum();
    assert!(delivered > 0, "food must reach a nest");

    // The population spreads over behaviours rather than sitting in one.
    let occupied: Vec<u32> = [ANT_EXPLORE, ANT_FOLLOW_TRAIL, ANT_CARRY_FOOD]
        .into_iter()
        .filter(|rc| ants.iter().any(|ant| ant.run_class == *rc))
        .collect();
    assert!(
        occupied.len() >= 2,
        "ants must occupy several behaviours, saw {occupied:?}"
    );
}

/// Ants move. A workload whose agents all sat still would satisfy everything
/// above and still be worthless.
#[test]
fn ants_move_away_from_their_nests() {
    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    let before = observe_ants(&mut kernel, &colony);
    run_epochs(&mut kernel, knobs.epochs);
    let after = observe_ants(&mut kernel, &colony);

    let moved = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a.x != b.x || a.y != b.y)
        .count();
    assert!(
        moved > before.len() / 2,
        "most ants must have moved, only {moved} of {} did",
        before.len()
    );
}

/// The run is a pure function of the knobs.
#[test]
fn the_run_is_deterministic() {
    let knobs = knobs();
    let first = run_mode(&knobs, SchedulingMode::RunClassBins, 32);
    let second = run_mode(&knobs, SchedulingMode::RunClassBins, 32);

    assert_eq!(first.population, second.population);
    assert_eq!(first.delivered, second.delivered);
    assert_eq!(first.food_trail, second.food_trail);
    assert_eq!(first.home_trail, second.home_trail);
    assert_eq!(first.accounting.steps, second.accounting.steps);
    assert_eq!(first.dispatches(), second.dispatches());
}

/// The control the whole comparison rests on: changing how continuations are
/// binned must not change what the colony does. This is what the parity-slot
/// double buffering in the pipeline buys.
#[test]
fn both_schedules_simulate_the_same_world() {
    let comparison = compare(&knobs(), 32);
    assert!(
        comparison.simulated_identical_world(),
        "binning changed the simulation: fifo={:?} cohorted={:?}",
        comparison.fifo,
        comparison.cohorted
    );
}

/// The failing case for the control above. Two runs that genuinely differ must
/// be reported as differing — otherwise `simulated_identical_world` would be a
/// check that passes because it cannot tell anything apart.
#[test]
fn the_identical_world_check_can_fail() {
    let base = knobs();
    let other = ColonyKnobs {
        seed: base.seed ^ 0xFFFF_FFFF,
        ..base
    };
    let a = run_mode(&base, SchedulingMode::RunClassBins, 32);
    let b = run_mode(&other, SchedulingMode::RunClassBins, 32);

    assert_ne!(
        a.population, b.population,
        "a different seed must produce a different colony"
    );
    let mixed = soma::experiments::ant_colony::ColonyComparison {
        fifo: a,
        cohorted: b,
    };
    assert!(
        !mixed.simulated_identical_world(),
        "the control must reject two runs that simulated different worlds"
    );
}

/// The null control. At width 1 every dispatch is a single lane, so binning
/// cannot affect occupancy. A ratio other than 1.00 would mean the harness is
/// measuring itself rather than the mechanism.
#[test]
fn binning_has_no_effect_at_width_one() {
    let comparison = compare(&knobs(), 1);
    assert!(
        (comparison.occupancy_ratio() - 1.0).abs() < 1e-9,
        "expected no effect at width 1, got {:.4}x",
        comparison.occupancy_ratio()
    );
    assert_eq!(comparison.fifo.dispatches(), comparison.cohorted.dispatches());
}

/// The measurement itself, stated as a floor rather than a fixed number so it
/// documents the claim without pinning an incidental digit.
#[test]
fn run_class_binning_beats_a_persistent_fifo() {
    let comparison = compare(&knobs(), 32);
    assert!(
        comparison.simulated_identical_world(),
        "the control must hold before the ratio means anything"
    );
    assert!(
        comparison.occupancy_ratio() >= 1.5,
        "expected a meaningful occupancy gain, got {:.2}x",
        comparison.occupancy_ratio()
    );
    assert!(
        comparison.dispatch_reduction() > 0.0,
        "cohorting must issue fewer dispatches, not more"
    );
}

/// The pipeline above the ants runs too: colonies aggregate and the world folds.
/// If either stalled, the field would stay empty and `foraging_emerges` would be
/// the only thing to notice — from a symptom rather than the cause.
#[test]
fn the_aggregation_pipeline_keeps_running() {
    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    run_epochs(&mut kernel, knobs.epochs);

    let pending = kernel.pending_counts();
    let colonies = pending
        .iter()
        .find(|(rc, _)| *rc == COLONY_AGGREGATE)
        .map(|(_, n)| *n)
        .unwrap_or(0);
    let world = pending
        .iter()
        .find(|(rc, _)| *rc == WORLD_STEP)
        .map(|(_, n)| *n)
        .unwrap_or(0);

    assert_eq!(colonies, colony.colonies.len(), "every colony must still run");
    assert_eq!(world, 1, "the world must still run");
}

// ---- predation ------------------------------------------------------------

/// The containment claim. A predator takes ants out of one colony; those ants
/// fault through the ordinary step path, the failure stops inside that subtree,
/// and no ant of any other colony is affected.
#[test]
fn a_failure_stays_inside_the_colony_it_happened_in() {
    use soma::experiments::ant_colony::{inject_predator, predator_outcome, PredatorStrike};

    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    run_epochs(&mut kernel, 40);

    let strike = PredatorStrike {
        colony: 1,
        victims: 8,
    };
    let before = observe_ants(&mut kernel, &colony);
    let bystanders_before = before
        .iter()
        .filter(|ant| ant.colony != strike.colony && ant.alive)
        .count();

    let struck = inject_predator(&mut kernel, &colony, strike);
    assert_eq!(struck.len(), strike.victims as usize, "the strike must land");

    // The victims fault at their next resume, not when they were struck.
    run_epochs(&mut kernel, 5);
    let outcome = predator_outcome(&mut kernel, &colony, strike, struck);

    assert_eq!(
        outcome.survivors,
        knobs.ants_per_colony as usize - strike.victims as usize,
        "exactly the struck ants must be gone"
    );
    assert_eq!(
        outcome.bystanders, bystanders_before,
        "no ant of another colony may be affected"
    );
    assert!(
        outcome.failures >= strike.victims as usize,
        "each victim must fail through the real path, saw {} failures",
        outcome.failures
    );
    assert!(
        outcome.colony_alive,
        "Notify contains the failure at the child; the colony must survive"
    );
    assert!(outcome.world_alive, "the world must survive");
    assert!(
        outcome.notices >= strike.victims as usize,
        "the supervisor must be told about every child it lost, saw {}",
        outcome.notices
    );
}

/// The colony keeps working after the strike: the survivors still forage and the
/// pipeline above them still runs. Containment that left the subtree dead would
/// pass the test above and still be useless.
#[test]
fn the_colony_keeps_working_after_a_strike() {
    use soma::experiments::ant_colony::{inject_predator, PredatorStrike};

    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    run_epochs(&mut kernel, 40);

    let strike = PredatorStrike {
        colony: 1,
        victims: 8,
    };
    inject_predator(&mut kernel, &colony, strike);
    run_epochs(&mut kernel, 5);

    let (_, home_before) = field_totals(&mut kernel, &colony, 45);
    run_epochs(&mut kernel, 40);
    let (_, home_after) = field_totals(&mut kernel, &colony, 85);

    assert!(
        home_after > home_before,
        "the surviving ants must still be laying trail"
    );
    assert!(
        check(&kernel).is_empty(),
        "the machine must remain legal after a contained failure"
    );
}

/// The fault-free control. Without a strike nothing fails, so the numbers above
/// are attributable to the predator rather than to the workload.
#[test]
fn without_a_predator_nothing_fails() {
    use soma::experiments::ant_colony::{predator_outcome, PredatorStrike};

    let knobs = knobs();
    let (mut kernel, colony) = build(&knobs);
    run_epochs(&mut kernel, 45);

    let strike = PredatorStrike {
        colony: 1,
        victims: 0,
    };
    let outcome = predator_outcome(&mut kernel, &colony, strike, Vec::new());
    assert_eq!(outcome.failures, 0, "an unstruck colony must not fail");
    assert_eq!(outcome.notices, 0, "and must generate no terminal notices");
    assert_eq!(outcome.survivors, knobs.ants_per_colony as usize);
}
