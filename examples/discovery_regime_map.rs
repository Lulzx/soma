use soma::experiments::discovery_search::{regime_map, DiscoveryKnobs};

fn main() {
    let base = DiscoveryKnobs {
        branching_factor: 2,
        depth: 3,
        elements_per_experiment: 16,
        ..Default::default()
    };
    println!(
        "duplicate,rejection,classes,elements,compute_compression,elimination,batch_compression"
    );
    for point in regime_map(base).expect("regime sweep should complete") {
        println!(
            "{:.2},{:.2},{},{},{:.4},{:.4},{:.4}",
            point.duplicate_rate,
            point.rejection_rate,
            point.evaluator_classes,
            point.elements_per_experiment,
            point.compute_compression,
            point.elimination_rate,
            point.batch_compression,
        );
    }
}
