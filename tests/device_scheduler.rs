use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{Kind, Ref64, StateAccess};
use soma::scheduler::admission::Candidate;
use soma::scheduler::device::{
    reference_device_schedule, reference_lane_conflicts, reference_resident_search,
    DeviceLaneAccess, ResidentSearchConfig, DEVICE_DEFERRED, DEVICE_POLICY_DEFERRED, DEVICE_RUN,
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

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn dynamic_search_stays_on_device_across_epochs() {
    use soma::executives::metal_scheduler::MetalResidentSearch;

    let config = ResidentSearchConfig {
        roots: 7,
        branching: 3,
        depth: 4,
        class_count: 4,
        work_iters: 17,
        cohort_width: 32,
    };
    let expected = reference_resident_search(config);
    let actual = MetalResidentSearch::new().unwrap().run(config).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.nodes, config.node_count().unwrap());
    assert_eq!(actual.epochs, config.depth + 1);
    assert_eq!(actual.overflow, 0);
    assert_eq!(actual.useful_lane_slots, actual.nodes);
    assert!(actual.lane_slots >= actual.useful_lane_slots);

    let uniform = ResidentSearchConfig {
        class_count: 1,
        ..config
    };
    let uniform_actual = MetalResidentSearch::new().unwrap().run(uniform).unwrap();
    assert_eq!(uniform_actual, reference_resident_search(uniform));
    assert!(
        actual.cohorts > uniform_actual.cohorts,
        "class fragmentation must cost real device cohorts"
    );

    let scalar = ResidentSearchConfig {
        cohort_width: 1,
        ..config
    };
    let scalar_actual = MetalResidentSearch::new().unwrap().run(scalar).unwrap();
    assert_eq!(scalar_actual, reference_resident_search(scalar));
    assert_eq!(scalar_actual.cohorts, scalar_actual.nodes);
    assert_eq!(scalar_actual.lane_slots, scalar_actual.nodes);
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

fn journal_workload() -> Vec<DeviceLaneAccess> {
    let object = Ref64::new(30, 1, Kind::Object);
    let same_bits_other_namespace = object;
    let future = Ref64::new(31, 1, Kind::Future);
    vec![
        DeviceLaneAccess::read(0, 1, object, 0),
        DeviceLaneAccess::read(1, 1, object, 0),
        DeviceLaneAccess::write(2, 1, object, 0),
        DeviceLaneAccess::write(2, 1, object, 1),
        DeviceLaneAccess::write(3, 2, same_bits_other_namespace, 0),
        DeviceLaneAccess::read(4, 2, future, 0),
        DeviceLaneAccess::write(5, 2, future, 0),
    ]
}

#[test]
fn reference_lane_journal_validation_is_namespace_aware_and_order_independent() {
    let expected = reference_lane_conflicts(&journal_workload(), 7);
    assert_eq!(expected[0].first_other_lane, 2);
    assert_eq!(expected[1].first_other_lane, 2);
    assert_eq!(expected[2].first_other_lane, 0);
    assert_eq!(expected[3].conflicts, 0);
    assert_eq!(expected[4].first_other_lane, 5);
    assert_eq!(expected[5].first_other_lane, 4);
    assert_eq!(expected[6].conflicts, 0);

    let mut reversed = journal_workload();
    reversed.reverse();
    assert_eq!(reference_lane_conflicts(&reversed, 7), expected);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn real_metal_lane_journal_validation_matches_the_reference() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let accesses = journal_workload();
    let expected = reference_lane_conflicts(&accesses, 7);
    let mut metal = MetalDeviceScheduler::new().unwrap();
    assert_eq!(
        metal.validate_lane_journals(&accesses, 7).unwrap(),
        expected
    );
    let capacity = metal.journal_resident_capacity();
    assert_eq!(
        metal.validate_lane_journals(&accesses[..3], 3).unwrap(),
        reference_lane_conflicts(&accesses[..3], 3)
    );
    assert_eq!(metal.journal_resident_capacity(), capacity);
    assert_eq!(
        metal.validate_lane_journals(&[], 3).unwrap()[2].conflicts,
        0
    );
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

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn sorted_metal_placement_matches_irregular_non_power_of_two_epochs() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let mut metal = MetalDeviceScheduler::new().unwrap();
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    for count in [1usize, 2, 3, 7, 31, 32, 33, 127] {
        let mut candidates = Vec::with_capacity(count);
        for index in 0..count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            candidates.push(candidate(
                index as u32 + 1,
                ((state >> 8) % 11) as u32 + 1,
                ((state >> 24) % 19) as u32,
                ((state >> 40) % 19) as u32,
                ((state >> 52) % 9) as u32,
                if state & 3 == 0 {
                    StateAccess::Mutable
                } else {
                    StateAccess::ReadOnly
                },
            ));
        }
        for policy in [
            PartialCohortPolicy::Defer,
            PartialCohortPolicy::SendToCpu,
            PartialCohortPolicy::RunPartial,
            PartialCohortPolicy::MergeWithGenericClass,
        ] {
            assert_eq!(
                metal.schedule(&candidates, 7, policy).unwrap(),
                reference_device_schedule(&candidates, 7, policy),
                "count={count} policy={policy:?}"
            );
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn sorted_mutable_admission_matches_the_set_rule() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let candidates: Vec<_> = (0..257u32)
        .map(|index| {
            candidate(
                index + 1,
                index % 17 + 1,
                index.wrapping_mul(11) % 23,
                index.wrapping_mul(11) % 23,
                index.wrapping_mul(7) % 13,
                if index % 4 == 0 {
                    StateAccess::ReadOnly
                } else {
                    StateAccess::Mutable
                },
            )
        })
        .collect();
    let mut metal = MetalDeviceScheduler::new().unwrap();
    for policy in [
        PartialCohortPolicy::Defer,
        PartialCohortPolicy::SendToCpu,
        PartialCohortPolicy::RunPartial,
        PartialCohortPolicy::MergeWithGenericClass,
    ] {
        assert_eq!(
            metal.schedule(&candidates, 9, policy).unwrap(),
            reference_device_schedule(&candidates, 9, policy),
            "policy={policy:?}"
        );
    }
}
