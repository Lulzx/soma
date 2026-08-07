use soma::discovery::{
    execute_naive, execute_optimized, DiscoveryEvent, DiscoveryNode, DiscoveryTrace,
    EvaluationSpec, FusionClass, ModuleDigest,
};
use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::discovery_search::evaluator_programs;
use soma::experiments::discovery_search::{run_cpu, DiscoveryKnobs};

#[test]
fn rejecting_hypotheses_withdraws_unshared_pending_work() {
    let knobs = DiscoveryKnobs {
        rejection_rate: 1.0,
        rejection_depth: 0,
        arrival_skew: 4,
        duplicate_rate: 0.0,
        shared_prefix_rate: 0.0,
        observation_rate: 0.0,
        ..Default::default()
    };
    let report = run_cpu(&knobs).unwrap();
    assert!(report.optimized.metrics.cancelled_before_execution > 0);
    assert!(
        report.optimized.metrics.physical_evaluator_executions
            < report.naive.metrics.physical_evaluator_executions
    );
    assert_eq!(report.naive.scientific, report.optimized.scientific);
}

#[test]
fn zero_rejection_control_cancels_nothing() {
    let report = run_cpu(&DiscoveryKnobs {
        rejection_rate: 0.0,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(report.optimized.metrics.cancelled_before_execution, 0);
}

#[test]
fn a_request_made_after_rejection_is_cancelled_even_if_its_key_is_ready() {
    let programs = evaluator_programs(1);
    let mut backend = CpuReferenceBackend::with(&[&programs[0]]);
    let node = derivation(9);
    let mut trace = DiscoveryTrace::default();
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 1,
        parent: None,
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 1,
        hypothesis: 1,
        node: node.clone(),
    });
    trace.push(DiscoveryEvent::EvidencePublished);
    trace.push(DiscoveryEvent::HypothesisRejected { id: 1 });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 2,
        hypothesis: 1,
        node,
    });
    let result = execute_optimized(&trace, &mut backend).unwrap();
    assert_eq!(result.metrics.cache_hits, 0);
    assert_eq!(result.metrics.cancelled_before_execution, 1);
    assert!(!result.outputs.contains_key(&2));
}

#[test]
fn a_cancelled_dependency_is_rescheduled_if_new_interest_reaches_it() {
    let programs = evaluator_programs(1);
    let mut naive_backend = CpuReferenceBackend::with(&[&programs[0]]);
    let mut optimized_backend = CpuReferenceBackend::with(&[&programs[0]]);
    let mut trace = DiscoveryTrace::default();
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 1,
        parent: None,
    });
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 2,
        parent: None,
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 1,
        hypothesis: 1,
        node: derivation(1),
    });
    trace.push(DiscoveryEvent::HypothesisRejected { id: 1 });
    trace.push(DiscoveryEvent::EvidencePublished);
    trace.push(DiscoveryEvent::NodeRequested {
        request: 2,
        hypothesis: 2,
        node: derivation(2),
    });
    trace.push(DiscoveryEvent::DependencyAdded {
        node: 2,
        depends_on: 1,
    });
    trace.push(DiscoveryEvent::EvidencePublished);
    let naive = execute_naive(&trace, &mut naive_backend).unwrap();
    let optimized = execute_optimized(&trace, &mut optimized_backend).unwrap();
    assert_eq!(naive.scientific, optimized.scientific);
    assert!(optimized.outputs.contains_key(&1));
}

#[test]
fn rejecting_a_parent_withdraws_its_entire_hypothesis_branch() {
    let programs = evaluator_programs(1);
    let mut backend = CpuReferenceBackend::with(&[&programs[0]]);
    let mut trace = DiscoveryTrace::default();
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 1,
        parent: None,
    });
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 2,
        parent: Some(1),
    });
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 3,
        parent: Some(2),
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 1,
        hypothesis: 2,
        node: derivation(1),
    });
    trace.push(DiscoveryEvent::NodeRequested {
        request: 2,
        hypothesis: 3,
        node: derivation(2),
    });
    trace.push(DiscoveryEvent::HypothesisRejected { id: 1 });
    trace.push(DiscoveryEvent::EvidencePublished);
    let result = execute_optimized(&trace, &mut backend).unwrap();
    assert_eq!(result.metrics.cancelled_before_execution, 2);
    assert!(result.outputs.is_empty());
    assert!(result
        .scientific
        .hypotheses
        .values()
        .all(|status| { *status == soma::discovery::graph::HypothesisStatus::Rejected }));
}

fn derivation(value: u64) -> DiscoveryNode {
    DiscoveryNode::Derivation(EvaluationSpec::new(
        "derive",
        ModuleDigest::of(b"module"),
        10_000,
        value.to_le_bytes().to_vec(),
        1,
        8,
        FusionClass::Pointwise,
    ))
}
