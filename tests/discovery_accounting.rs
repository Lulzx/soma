use soma::discovery::{
    execute_optimized, DiscoveryError, DiscoveryEvent, DiscoveryNode, DiscoveryTrace,
    EvaluationSpec, FusionClass, ModuleDigest,
};
use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::discovery_search::{run_cpu, DiscoveryKnobs};

#[test]
fn every_logical_request_has_exactly_one_fate() {
    let report = run_cpu(&DiscoveryKnobs::default()).unwrap();
    for metrics in [&report.naive.metrics, &report.optimized.metrics] {
        assert_eq!(
            metrics.logical_requests,
            metrics.physical_evaluator_executions
                + metrics.cache_hits
                + metrics.pending_request_joins
                + metrics.cancelled_before_execution
        );
    }
}

#[test]
fn malformed_logical_input_is_rejected_before_batching() {
    let mut trace = DiscoveryTrace::default();
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 1,
        parent: None,
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 1,
        hypothesis: 1,
        node: DiscoveryNode::Derivation(EvaluationSpec::new(
            "bad-shape",
            ModuleDigest::of(b"m"),
            10_000,
            vec![1, 2, 3],
            1,
            8,
            FusionClass::Pointwise,
        )),
    });
    let error = execute_optimized(&trace, &mut CpuReferenceBackend::default()).unwrap_err();
    assert_eq!(
        error,
        DiscoveryError::InvalidTrace("discovery input shape mismatch")
    );
}
