use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::ant_colony::{build, field_totals, observe_ants, ColonyKnobs};
use soma::experiments::ant_scoring::{prepare_colony_epoch, sensing_program, ColonySensing};

fn run(
    knobs: &ColonyKnobs,
    mut sensing: impl FnMut(&mut soma::kernel::Kernel, &soma::experiments::ant_colony::AntColony, u32),
) -> (u64, u64, u64) {
    let (mut kernel, colony) = build(knobs);
    for epoch in 0..knobs.epochs {
        sensing(&mut kernel, &colony, epoch);
        kernel.run_epoch();
    }
    let (food, home) = field_totals(&mut kernel, &colony, knobs.epochs);
    let delivered = observe_ants(&mut kernel, &colony)
        .iter()
        .map(|a| a.delivered as u64)
        .sum();
    (delivered, food, home)
}

fn main() {
    let knobs = ColonyKnobs {
        width: 48,
        height: 48,
        colonies: 3,
        ants_per_colony: 32,
        epochs: 120,
        ..ColonyKnobs::default()
    };
    let host = run(&knobs, |kernel, colony, epoch| {
        prepare_colony_epoch(kernel, colony, epoch, ColonySensing::HostReference).unwrap();
    });
    let program = sensing_program();
    let mut cpu = CpuReferenceBackend::with(&[&program]);
    let collective = run(&knobs, |kernel, colony, epoch| {
        prepare_colony_epoch(kernel, colony, epoch, ColonySensing::Collective(&mut cpu)).unwrap();
    });
    println!(
        "ant sensing executive: {} ants, {} epochs",
        knobs.ant_count(),
        knobs.epochs
    );
    println!(
        "  host reference: delivered={} food_trail={} home_trail={}",
        host.0, host.1, host.2
    );
    println!(
        "  collective/cpu: delivered={} food_trail={} home_trail={}",
        collective.0, collective.1, collective.2
    );
    println!("  identical world: {}", host == collective);
}
