use soma::discovery::{DiscoveryError, ObjectDigest};
use soma::experiments::self_tuning::{
    capture_with, cpu_configurations, replay, replay_with_preparation_sharing, run_cpu,
    CapturedStudy, PhysicalMeasurement, TimingObservation, TuningStudy, TuningWorkload,
};

fn tiny_study(trials: u32) -> TuningStudy {
    TuningStudy {
        workloads: vec![TuningWorkload {
            id: 7,
            alu_ops: 8,
            cohorts: 2,
            elements_per_cohort: 16,
        }],
        trials,
    }
}

#[test]
fn acquisition_is_counterbalanced_and_preserves_every_trial() {
    let study = tiny_study(3);
    let configs = cpu_configurations(&[1, 2]);
    let mut calls = Vec::new();
    let captured = capture_with(&study, &configs, |_, config| {
        calls.push(config.id);
        Ok(PhysicalMeasurement {
            elapsed_nanos: u64::from(config.id),
            output_digest: ObjectDigest::of(b"same output"),
        })
    })
    .unwrap();

    assert_eq!(&calls[..4], &[1, 2, 3, 4], "warmup order");
    assert_eq!(&calls[4..8], &[1, 2, 3, 4]);
    assert_eq!(&calls[8..12], &[2, 3, 4, 1]);
    assert_eq!(&calls[12..16], &[3, 4, 1, 2]);
    assert_eq!(captured.observations.len(), 12);
}

fn identical_observations() -> CapturedStudy {
    let study = tiny_study(3);
    let configs = cpu_configurations(&[1]);
    let observations = (0..3)
        .flat_map(|trial| {
            configs.iter().map(move |config| TimingObservation {
                workload_id: 7,
                config_id: config.id,
                trial,
                elapsed_nanos: 42,
                output_digest: ObjectDigest::of(b"same output"),
            })
        })
        .collect();
    CapturedStudy {
        study,
        configs,
        observations,
    }
}

#[test]
fn replay_shares_preparation_but_not_identical_observations() {
    let report = replay(identical_observations()).unwrap();

    assert!(report.invariants.all_hold());
    assert_eq!(report.naive.scientific, report.optimized.scientific);
    assert_eq!(report.optimized.metrics.logical_requests, 10);
    assert_eq!(
        report.optimized.metrics.deterministic_physical_executions,
        2
    );
    assert_eq!(report.optimized.metrics.pending_request_joins, 2);
    assert_eq!(
        report.optimized.metrics.physical_evaluator_executions
            - report.optimized.metrics.deterministic_physical_executions,
        6,
        "all six independent timing samples must execute"
    );
}

#[test]
fn disabling_preparation_sharing_is_a_true_negative_control() {
    let shared = replay(identical_observations()).unwrap();
    let isolated = replay_with_preparation_sharing(identical_observations(), false).unwrap();

    assert!(isolated.invariants.all_hold());
    assert_eq!(isolated.optimized.metrics.pending_request_joins, 0);
    assert_eq!(
        isolated.optimized.metrics.deterministic_physical_executions,
        4
    );
    assert_eq!(
        shared.optimized.metrics.deterministic_physical_executions,
        2
    );
}

#[test]
fn malformed_capture_is_rejected_before_replay() {
    let mut malformed = identical_observations();
    malformed.observations.pop();
    assert_eq!(
        replay(malformed).unwrap_err(),
        DiscoveryError::InvalidTrace("self-tuning observation count mismatch")
    );
}

#[test]
fn differing_configuration_outputs_invalidate_the_study() {
    let mut malformed = identical_observations();
    malformed.observations[0].output_digest = ObjectDigest::of(b"wrong output");
    assert_eq!(
        replay(malformed).unwrap_err(),
        DiscoveryError::InvalidTrace("self-tuning configuration output mismatch")
    );
}

#[test]
fn real_cpu_search_produces_equivalent_scientific_state() {
    let report = run_cpu(&tiny_study(3), &[1, 2]).unwrap();
    assert!(report.invariants.all_hold());
    assert_eq!(report.captured.observations.len(), 12);
    assert_eq!(report.rankings.len(), 1);
    assert_eq!(report.rankings[0].configs.len(), 4);
}

#[cfg(feature = "native")]
#[test]
fn real_native_search_is_compared_to_the_reference_oracle() {
    use soma::experiments::self_tuning::{run_native, Placement};

    let report = run_native(&tiny_study(2), &[1, 2]).unwrap();
    assert!(report.invariants.all_hold());
    assert!(report
        .captured
        .configs
        .iter()
        .any(|config| config.placement == Placement::NativeCpu));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn real_metal_search_shares_compilation_and_preserves_outputs() {
    use soma::experiments::self_tuning::{hardware_configurations, run_metal};

    let report = run_metal(&tiny_study(2), &[1, 2]).unwrap();
    assert!(report.invariants.all_hold());
    assert_eq!(report.captured.configs, hardware_configurations(&[1, 2]));
    assert_eq!(
        report.captured.observations.len(),
        report.captured.configs.len() * 2
    );
    assert!(report
        .captured
        .configs
        .iter()
        .any(|config| config.placement == soma::experiments::self_tuning::Placement::Metal));
}
