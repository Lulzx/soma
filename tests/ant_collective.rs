//! End-to-end ant sensing executive: evaluator output is consumed by ant steps.

use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::ant_colony::{build, field_totals, observe_ants, ColonyKnobs};
use soma::experiments::ant_scoring::{prepare_colony_epoch, sensing_program, ColonySensing};

fn knobs() -> ColonyKnobs {
    ColonyKnobs {
        width: 36,
        height: 32,
        colonies: 2,
        ants_per_colony: 18,
        food_sources: 3,
        epochs: 64,
        ..ColonyKnobs::default()
    }
}

fn run_host(knobs: &ColonyKnobs) -> (Vec<soma::experiments::ant_colony::AntView>, (u64, u64), u64) {
    let (mut kernel, colony) = build(knobs);
    for epoch in 0..knobs.epochs {
        prepare_colony_epoch(&mut kernel, &colony, epoch, ColonySensing::HostReference).unwrap();
        kernel.run_epoch();
    }
    soma::semantics::invariants::assert_legal(&kernel);
    let field = field_totals(&mut kernel, &colony, knobs.epochs);
    let ants = observe_ants(&mut kernel, &colony);
    (ants, field, kernel.accounting().steps)
}

fn run_backend(
    knobs: &ColonyKnobs,
    backend: &mut dyn soma::executives::batch::BatchBackend,
) -> (Vec<soma::experiments::ant_colony::AntView>, (u64, u64), u64) {
    let (mut kernel, colony) = build(knobs);
    for epoch in 0..knobs.epochs {
        let scored = prepare_colony_epoch(
            &mut kernel,
            &colony,
            epoch,
            ColonySensing::Collective(backend),
        )
        .unwrap();
        assert_eq!(scored, colony.ant_count());
        kernel.run_epoch();
    }
    soma::semantics::invariants::assert_legal(&kernel);
    let field = field_totals(&mut kernel, &colony, knobs.epochs);
    let ants = observe_ants(&mut kernel, &colony);
    (ants, field, kernel.accounting().steps)
}

#[test]
fn collective_sensing_simulates_the_host_reference_world() {
    let knobs = knobs();
    let sensing = sensing_program();
    let mut cpu = CpuReferenceBackend::with(&[&sensing]);
    let host = run_host(&knobs);
    let collective = run_backend(&knobs, &mut cpu);
    assert_eq!(
        host, collective,
        "collective sensing changed the simulated world"
    );
}

#[test]
fn collective_sensing_reclaims_epoch_temporaries() {
    // Epoch count matches the long colony run.  Population size is deliberately
    // small: resource cardinality must be independent of both axes, while this
    // test is aimed at the epoch-over-epoch leak rather than backend throughput.
    let knobs = ColonyKnobs {
        width: 16,
        height: 14,
        colonies: 2,
        ants_per_colony: 4,
        food_sources: 2,
        epochs: 260,
        ..ColonyKnobs::default()
    };
    let (mut kernel, colony) = build(&knobs);
    let sensing = sensing_program();
    let mut cpu = CpuReferenceBackend::with(&[&sensing]);
    let baseline = (
        kernel.object_count(),
        kernel.collective_count(),
        kernel.future_count(),
        kernel.capability_count(),
    );

    for epoch in 0..knobs.epochs {
        prepare_colony_epoch(
            &mut kernel,
            &colony,
            epoch,
            ColonySensing::Collective(&mut cpu),
        )
        .unwrap();
        assert_eq!(
            (
                kernel.object_count(),
                kernel.collective_count(),
                kernel.future_count(),
                kernel.capability_count(),
            ),
            baseline,
            "epoch {epoch} retained sensing input/aux/output/collective/future state"
        );
        kernel.run_epoch();
    }
    soma::semantics::invariants::assert_legal(&kernel);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_collective_simulates_the_host_reference_world() {
    use soma::executives::metal::MetalBatchBackend;
    let knobs = ColonyKnobs {
        epochs: 24,
        ..knobs()
    };
    let sensing = sensing_program();
    let Ok(mut metal) = MetalBatchBackend::with(&[&sensing]) else {
        return;
    };
    assert_eq!(run_host(&knobs), run_backend(&knobs, &mut metal));
}
