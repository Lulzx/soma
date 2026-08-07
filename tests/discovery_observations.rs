use soma::discovery::{
    execute_optimized, DiscoveryEvent, DiscoveryNode, DiscoveryTrace, EvaluationSpec, FusionClass,
    ModuleDigest,
};
use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::discovery_search::evaluator_programs;

#[test]
fn identical_observations_execute_twice_but_may_share_a_dispatch() {
    let programs = evaluator_programs(1);
    let mut backend = CpuReferenceBackend::with(&[&programs[0]]);
    let spec = EvaluationSpec::new(
        "sample",
        ModuleDigest::of(b"m"),
        10_000,
        7u64.to_le_bytes().to_vec(),
        1,
        8,
        FusionClass::Pointwise,
    );
    let mut trace = DiscoveryTrace::default();
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 1,
        parent: None,
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 1,
        hypothesis: 1,
        node: DiscoveryNode::Observation {
            sample: 1,
            evaluation: spec.clone(),
        },
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 2,
        hypothesis: 1,
        node: DiscoveryNode::Observation {
            sample: 2,
            evaluation: spec,
        },
    });
    trace.push(DiscoveryEvent::EvidencePublished);
    let result = execute_optimized(&trace, &mut backend).unwrap();
    assert_eq!(result.metrics.physical_evaluator_executions, 2);
    assert_eq!(result.metrics.physical_dispatches, 1);
    assert_eq!(result.outputs.len(), 2);
}
