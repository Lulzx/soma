use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{Kind, Ref64, StateAccess};
use soma::scheduler::admission::Candidate;
use soma::scheduler::device::{
    reference_device_schedule, DEVICE_DEFERRED, DEVICE_POLICY_DEFERRED, DEVICE_RUN,
    DEVICE_SEND_TO_CPU,
};

fn candidate(
    id: u32,
    process: u32,
    bin: u32,
    run_class: u32,
    waiting_since: u32,
    state_access: StateAccess,
) -> Candidate {
    Candidate {
        bin,
        continuation: Ref64::new(id, 1, Kind::Continuation),
        process: Ref64::new(process, 1, Kind::Process),
        run_class,
        state_access,
        waiting_since,
    }
}

fn workload() -> Vec<Candidate> {
    vec![
        candidate(10, 1, 7, 7, 4, StateAccess::Mutable),
        candidate(11, 1, 7, 7, 2, StateAccess::Mutable),
        candidate(12, 2, 3, 3, 1, StateAccess::ReadOnly),
        candidate(13, 2, 3, 3, 1, StateAccess::ReadOnly),
        candidate(14, 3, 7, 7, 5, StateAccess::Mutable),
        candidate(15, 4, 7, 7, 5, StateAccess::ReadOnly),
        candidate(16, 5, 7, 7, 5, StateAccess::ReadOnly),
    ]
}

#[test]
fn reference_device_admission_is_the_scheduler_rule() {
    let schedule = reference_device_schedule(&workload(), 4, PartialCohortPolicy::RunPartial);
    assert_eq!(schedule.placements[0].disposition, DEVICE_DEFERRED);
    assert_eq!(schedule.placements[1].disposition, DEVICE_RUN);
    assert_eq!(schedule.placements[2].bin_rank, 0);
    assert_eq!(schedule.placements[3].bin_rank, 1);
    assert_eq!(schedule.placements[4].bin_rank, 1);
    assert_eq!(schedule.placements[6].bin_rank, 3);
}

#[test]
fn every_partial_policy_has_an_explicit_device_disposition() {
    let candidates = workload();
    let deferred = reference_device_schedule(&candidates, 3, PartialCohortPolicy::Defer);
    let spilled = reference_device_schedule(&candidates, 3, PartialCohortPolicy::SendToCpu);
    let partial = reference_device_schedule(&candidates, 3, PartialCohortPolicy::RunPartial);
    assert!(deferred
        .placements
        .iter()
        .any(|placement| placement.disposition == DEVICE_POLICY_DEFERRED));
    assert!(spilled
        .placements
        .iter()
        .any(|placement| placement.disposition == DEVICE_SEND_TO_CPU));
    assert!(!partial.placements.iter().any(|placement| {
        matches!(
            placement.disposition,
            DEVICE_POLICY_DEFERRED | DEVICE_SEND_TO_CPU
        )
    }));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn real_metal_scheduler_matches_the_reference_and_reuses_resident_buffers() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let candidates = workload();
    let mut metal = MetalDeviceScheduler::new().unwrap();
    for policy in [
        PartialCohortPolicy::Defer,
        PartialCohortPolicy::SendToCpu,
        PartialCohortPolicy::RunPartial,
        PartialCohortPolicy::MergeWithGenericClass,
    ] {
        let expected = reference_device_schedule(&candidates, 3, policy);
        assert_eq!(metal.schedule(&candidates, 3, policy).unwrap(), expected);
    }
    let capacity = metal.resident_capacity();
    let expected = reference_device_schedule(&candidates[..3], 2, PartialCohortPolicy::RunPartial);
    assert_eq!(
        metal
            .schedule(&candidates[..3], 2, PartialCohortPolicy::RunPartial)
            .unwrap(),
        expected
    );
    assert_eq!(metal.resident_capacity(), capacity);
}
