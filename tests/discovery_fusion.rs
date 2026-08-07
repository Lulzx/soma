use soma::compiler::body::{ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};
use soma::discovery::{
    execute_optimized, DiscoveryEvent, DiscoveryNode, DiscoveryTrace, EvaluationSpec, FusionClass,
    ModuleDigest,
};
use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::discovery_search::{run_cpu, DiscoveryKnobs};

#[test]
fn pointwise_work_is_partitioned_from_fused_outputs() {
    let knobs = DiscoveryKnobs {
        evaluator_classes: 1,
        duplicate_rate: 0.0,
        shared_prefix_rate: 0.0,
        rejection_rate: 0.0,
        observation_rate: 1.0,
        arrival_skew: 100,
        ..Default::default()
    };
    let report = run_cpu(&knobs).unwrap();
    assert_eq!(report.naive.scientific, report.optimized.scientific);
    assert_eq!(report.optimized.metrics.physical_dispatches, 1);
    assert!(report.optimized.metrics.batch_compression() > 1.0);
}

#[test]
fn evaluator_heterogeneity_is_a_fusion_control() {
    let one = run_cpu(&DiscoveryKnobs {
        evaluator_classes: 1,
        observation_rate: 1.0,
        arrival_skew: 100,
        ..Default::default()
    })
    .unwrap();
    let many = run_cpu(&DiscoveryKnobs {
        evaluator_classes: 32,
        observation_rate: 1.0,
        arrival_skew: 100,
        ..Default::default()
    })
    .unwrap();
    assert!(many.optimized.metrics.physical_dispatches > one.optimized.metrics.physical_dispatches);
}

#[test]
fn gathering_bodies_are_never_concatenated() {
    let program = EvaluatorProgram::new(
        999,
        "gather",
        ElementLayout::new(vec![FieldWidth::U64]),
        vec![Op::Index, Op::Gather(0, 0)],
        vec![Store { field: 0, value: 1 }],
    )
    .unwrap();
    assert_eq!(FusionClass::for_program(&program), FusionClass::LocalGather);
    let mut backend = CpuReferenceBackend::with(&[&program]);
    let mut trace = DiscoveryTrace::default();
    trace.push(DiscoveryEvent::HypothesisCreated {
        id: 1,
        parent: None,
    });
    for request in 1..=2 {
        let spec = EvaluationSpec::new(
            "gather",
            ModuleDigest::of(b"gather-module"),
            999,
            [1u64.to_le_bytes(), 2u64.to_le_bytes()].concat(),
            2,
            8,
            FusionClass::for_program(&program),
        );
        trace.push(DiscoveryEvent::NodeRequested {
            request,
            hypothesis: 1,
            node: DiscoveryNode::Observation {
                sample: request,
                evaluation: spec,
            },
        });
    }
    trace.push(DiscoveryEvent::EvidencePublished);
    let result = execute_optimized(&trace, &mut backend).unwrap();
    assert_eq!(result.metrics.physical_evaluator_executions, 2);
    assert_eq!(result.metrics.physical_dispatches, 2);
}
