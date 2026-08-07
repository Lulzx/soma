use soma::experiments::discovery_search::{run_cpu, DiscoveryKnobs};

#[test]
fn deterministic_duplicates_have_one_realization() {
    let knobs = DiscoveryKnobs {
        duplicate_rate: 1.0,
        shared_prefix_rate: 1.0,
        rejection_rate: 0.0,
        observation_rate: 0.0,
        ..Default::default()
    };
    let report = run_cpu(&knobs).unwrap();
    assert!(
        report.optimized.metrics.cache_hits + report.optimized.metrics.pending_request_joins > 0
    );
    assert!(
        report.optimized.metrics.physical_evaluator_executions
            < report.naive.metrics.physical_evaluator_executions
    );
}

#[test]
fn no_duplicate_control_does_not_invent_cache_hits() {
    let knobs = DiscoveryKnobs {
        duplicate_rate: 0.0,
        shared_prefix_rate: 0.0,
        rejection_rate: 0.0,
        observation_rate: 0.0,
        ..Default::default()
    };
    let report = run_cpu(&knobs).unwrap();
    assert_eq!(report.optimized.metrics.cache_hits, 0);
    assert_eq!(report.optimized.metrics.pending_request_joins, 0);
}
