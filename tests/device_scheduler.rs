use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{Kind, Ref64, StateAccess};
use soma::scheduler::admission::Candidate;
use soma::scheduler::device::{
    reference_device_schedule, reference_lane_conflicts, DeviceLaneAccess, DEVICE_DEFERRED,
    DEVICE_POLICY_DEFERRED, DEVICE_RUN, DEVICE_SEND_TO_CPU,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use soma::scheduler::device::{
    reference_resident_search, reference_resident_search_with_trace, ResidentSearchConfig,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use soma::scheduler::device_ops::{DeviceLaneOperation, DeviceOperationJournal};

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
    let mut metal = MetalResidentSearch::new().unwrap();
    let (traced_actual, trace) = metal.run_with_trace(config).unwrap();
    let (traced_expected, expected_trace) = reference_resident_search_with_trace(config);
    assert_eq!(traced_actual, traced_expected);
    assert_eq!(trace, expected_trace);
    assert_eq!(trace.len(), config.node_count().unwrap() as usize);
    assert!(trace.iter().all(|event| event.lane_sequence == 0));
    assert!(trace.windows(2).all(|pair| {
        pair[0].epoch < pair[1].epoch
            || (pair[0].epoch == pair[1].epoch && pair[0].lane < pair[1].lane)
    }));

    let actual = metal.run(config).unwrap();
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

fn quadratic_lane_conflicts(
    accesses: &[DeviceLaneAccess],
    lane_count: u32,
) -> Vec<soma::scheduler::device::DeviceLaneConflict> {
    (0..lane_count)
        .map(|lane| {
            let other = accesses
                .iter()
                .filter(|access| access.lane == lane)
                .flat_map(|access| {
                    accesses.iter().filter_map(move |candidate| {
                        (candidate.lane != lane
                            && candidate.resource_kind == access.resource_kind
                            && candidate.resource == access.resource
                            && (access.mode == 2 || candidate.mode == 2))
                            .then_some(candidate.lane)
                    })
                })
                .min();
            soma::scheduler::device::DeviceLaneConflict {
                lane,
                conflicts: u32::from(other.is_some()),
                first_other_lane: other.unwrap_or(u32::MAX),
                reserved: 0,
            }
        })
        .collect()
}

#[test]
fn sorted_lane_journal_validation_matches_the_pairwise_rule() {
    let mut state = 0xA076_1D64_78BD_642Fu64;
    for access_count in [0usize, 1, 2, 3, 17, 128, 1024] {
        let lane_count = 37;
        let mut accesses = Vec::with_capacity(access_count);
        for ordinal in 0..access_count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            accesses.push(DeviceLaneAccess::new(
                ((state >> 7) % lane_count as u64) as u32,
                ((state >> 19) % 5) as u32,
                state.rotate_left(23) % 29,
                if state & 3 == 0 { 2 } else { 1 },
                ordinal as u32,
            ));
            // Exercise duplicate accesses and read-then-write aggregation in
            // one lane without making physical record order significant.
            if ordinal % 31 == 0 {
                let mut duplicate = *accesses.last().unwrap();
                duplicate.mode = 2;
                duplicate.ordinal = duplicate.ordinal.wrapping_add(10_000);
                accesses.push(duplicate);
            }
        }
        let expected = quadratic_lane_conflicts(&accesses, lane_count);
        assert_eq!(reference_lane_conflicts(&accesses, lane_count), expected);
        accesses.reverse();
        assert_eq!(reference_lane_conflicts(&accesses, lane_count), expected);
    }
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

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn grouped_metal_journal_validation_matches_irregular_access_sets() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let lane_count = 73;
    let mut state = 0xE703_7ED1_A0B4_28DBu64;
    let mut accesses = Vec::new();
    for ordinal in 0..4096u32 {
        state = state
            .wrapping_mul(2_862_933_555_777_941_757)
            .wrapping_add(3_037_000_493);
        let access = DeviceLaneAccess::new(
            ((state >> 9) % lane_count as u64) as u32,
            ((state >> 21) % 7) as u32,
            state.rotate_left(17) % 113,
            if state & 7 < 3 { 2 } else { 1 },
            ordinal,
        );
        accesses.push(access);
        if ordinal % 43 == 0 {
            let mut duplicate = access;
            duplicate.mode = 2;
            accesses.push(duplicate);
        }
    }
    let expected = quadratic_lane_conflicts(&accesses, lane_count);
    let mut metal = MetalDeviceScheduler::new().unwrap();
    assert_eq!(
        metal.validate_lane_journals(&accesses, lane_count).unwrap(),
        expected
    );
    accesses.reverse();
    assert_eq!(
        metal.validate_lane_journals(&accesses, lane_count).unwrap(),
        expected
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn real_metal_validates_operation_records_and_arena_bounds() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let valid = DeviceOperationJournal {
        operations: vec![DeviceLaneOperation {
            lane: 7,
            ordinal: 0,
            opcode: 6,
            payload_offset: 1,
            payload_len: 3,
            ..DeviceLaneOperation::default()
        }],
        payload: vec![1, 2, 3, 4],
    };
    let mut metal = MetalDeviceScheduler::new().unwrap();
    metal.validate_operation_journals(&[&valid]).unwrap();

    let mut bad_bounds = valid.clone();
    bad_bounds.operations[0].payload_len = 4;
    assert!(metal.validate_operation_journals(&[&bad_bounds]).is_err());

    let mut bad_opcode = valid;
    bad_opcode.operations[0].opcode = 12;
    assert!(metal.validate_operation_journals(&[&bad_opcode]).is_err());
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

fn resident_frame_fixture() -> (
    soma::compiler::body::EvaluatorProgram,
    Vec<soma::scheduler::device::ResidentFrameBinding>,
    Vec<u8>,
) {
    let program = soma::compiler::surface::compile_evaluator(
        44_001,
        "resident-private-frame",
        "field u64\nlet x = load 0\nlet three = const 3\nlet product = mul x three\nlet one = const 1\nlet result = add product one\nstore 0 result\n",
    )
    .unwrap();
    let bindings = (0..9u64)
        .map(|lane| soma::scheduler::device::ResidentFrameBinding {
            continuation: 100 + lane,
            process: 200 + lane,
            frame: 300 + lane,
            actor: 400 + lane,
            target: 500 + lane,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let frames = (1..=9u64).flat_map(u64::to_le_bytes).collect();
    (program, bindings, frames)
}

#[test]
fn resident_frame_graph_cpu_oracle_emits_commit_abi_and_canonical_trace() {
    use soma::scheduler::device::{reference_resident_frame_graph, ResidentFrameGraphConfig};
    use soma::scheduler::device_ops::OP_WRITE_OBJECT;
    let (program, bindings, frames) = resident_frame_fixture();
    let result = reference_resident_frame_graph(
        &program,
        ResidentFrameGraphConfig {
            run_class: 2048,
            epochs: 4,
            cohort_width: 1,
        },
        &bindings,
        &frames,
    )
    .unwrap();
    let values = result
        .frames
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        (1..=9u64).map(|value| 81 * value + 40).collect::<Vec<_>>()
    );
    assert_eq!(result.trace.len(), bindings.len() * 4);
    assert_eq!(result.accesses.len(), bindings.len() * 2);
    assert!(result
        .operations
        .iter()
        .all(|journal| journal.operations.len() == 2
            && journal.operations[1].opcode == OP_WRITE_OBJECT));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn compiled_resident_frame_handler_matches_cpu_i19_and_trace() {
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::{reference_resident_frame_graph, ResidentFrameGraphConfig};
    let (program, bindings, frames) = resident_frame_fixture();
    let mut metal = MetalResidentSearch::new().unwrap();
    metal.install_frame_handler(2048, &program).unwrap();
    let narrow = ResidentFrameGraphConfig {
        run_class: 2048,
        epochs: 5,
        cohort_width: 1,
    };
    let wide = ResidentFrameGraphConfig {
        cohort_width: 32,
        ..narrow
    };
    let expected = reference_resident_frame_graph(&program, narrow, &bindings, &frames).unwrap();
    let actual_narrow = metal.run_frame_graph(narrow, &bindings, &frames).unwrap();
    let actual_wide = metal.run_frame_graph(wide, &bindings, &frames).unwrap();
    assert_eq!(
        actual_narrow, expected,
        "Metal/CPU and trace correspondence"
    );
    assert_eq!(
        actual_wide.frames, actual_narrow.frames,
        "I19 cohort-width neutrality"
    );
    assert_eq!(actual_wide.operations, actual_narrow.operations);
    assert_eq!(actual_wide.accesses, actual_narrow.accesses);
    assert_eq!(actual_wide.trace, actual_narrow.trace);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn resident_frame_handler_is_a_canonical_kernel_epoch_executive() {
    use soma::abi::{ProcessMode, StateAccess};
    use soma::compiler::run_classes::DEFAULT_MAX_STEPS;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::kernel::speculation::EpochExecutive;
    use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
    use soma::semantics::order::conforms_traces;

    const RUN_CLASS: u32 = 2048;
    let (program, _, _) = resident_frame_fixture();
    let build = || {
        let mut kernel = Kernel::new();
        kernel
            .install_frame_evaluator(RUN_CLASS, program.clone())
            .unwrap();
        let mut continuations = Vec::new();
        for value in 1..=9u64 {
            let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            let continuation = kernel
                .create_continuation(
                    SYSTEM_PRINCIPAL,
                    process,
                    ContinuationSpec::new(
                        StateAccess::ReadOnly,
                        RUN_CLASS,
                        0,
                        value.to_le_bytes().to_vec(),
                        DEFAULT_MAX_STEPS,
                    ),
                )
                .unwrap();
            continuations.push((process, continuation));
        }
        (kernel, continuations)
    };
    let (mut reference, continuations) = build();
    let (mut narrow, _) = build();
    let (mut wide, _) = build();
    narrow.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });
    wide.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });
    let mut narrow_backend = MetalResidentSearch::new().unwrap();
    narrow_backend
        .install_frame_handler(RUN_CLASS, &program)
        .unwrap();
    narrow_backend.set_frame_cohort_width(1).unwrap();
    let mut wide_backend = MetalResidentSearch::new().unwrap();
    wide_backend
        .install_frame_handler(RUN_CLASS, &program)
        .unwrap();
    wide_backend.set_frame_cohort_width(32).unwrap();

    reference.run_epoch();
    narrow.run_epoch_with_device_backend(&mut narrow_backend);
    wide.run_epoch_with_device_backend(&mut wide_backend);

    assert!(conforms_traces(&reference.trace_snapshot(), &narrow.trace_snapshot()).is_empty());
    assert!(conforms_traces(&reference.trace_snapshot(), &wide.trace_snapshot()).is_empty());
    assert_eq!(
        narrow.trace_snapshot(),
        wide.trace_snapshot(),
        "I19 kernel trace neutrality"
    );
    assert_eq!(
        narrow_backend.last_frame_trace(),
        wide_backend.last_frame_trace()
    );
    assert_eq!(narrow.speculation_stats().device_evaluated_lanes, 9);
    assert_eq!(wide.speculation_stats().device_evaluated_lanes, 9);
    for (process, continuation) in continuations {
        let reference_frame = reference.continuation_frame(continuation).unwrap();
        let narrow_frame = narrow.continuation_frame(continuation).unwrap();
        let wide_frame = wide.continuation_frame(continuation).unwrap();
        let expected = reference.object_bytes(process, reference_frame).unwrap();
        assert_eq!(
            narrow.object_bytes(process, narrow_frame).unwrap(),
            expected
        );
        assert_eq!(wide.object_bytes(process, wide_frame).unwrap(), expected);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn resident_future_resolve_commits_canonically_and_is_i18_i19_neutral() {
    use soma::abi::{ObjectKind, ProcessMode, StateAccess};
    use soma::compiler::run_classes::DEFAULT_MAX_STEPS;
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::kernel::speculation::EpochExecutive;
    use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
    use soma::semantics::order::conforms_traces;

    const RUN_CLASS: u32 = 2050;
    let program = compile_evaluator(
        44_050,
        "resident-resolve",
        "field u64\nfield u64\nfield u64\nlet n = load 2\nlet one = const 1\nlet next = add n one\nstore 2 next\n",
    ).unwrap();
    let build = || {
        let mut kernel = Kernel::new();
        kernel
            .install_frame_future_resolve(RUN_CLASS, program.clone(), 0, 8)
            .unwrap();
        let mut roots = Vec::new();
        for lane in 0..6u64 {
            let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            let future = kernel.create_future(process);
            let value =
                kernel.create_object(process, ObjectKind::RawBytes, lane.to_le_bytes().to_vec());
            let mut frame = future.to_u64().to_le_bytes().to_vec();
            frame.extend_from_slice(&value.to_u64().to_le_bytes());
            frame.extend_from_slice(&lane.to_le_bytes());
            let continuation = kernel
                .create_continuation(
                    SYSTEM_PRINCIPAL,
                    process,
                    ContinuationSpec::new(
                        StateAccess::ReadOnly,
                        RUN_CLASS,
                        0,
                        frame,
                        DEFAULT_MAX_STEPS,
                    ),
                )
                .unwrap();
            roots.push((process, continuation, future, value));
        }
        (kernel, roots)
    };
    let (mut reference, roots) = build();
    let (mut narrow, _) = build();
    let (mut wide, _) = build();
    narrow.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });
    wide.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });
    let mut nb = MetalResidentSearch::new().unwrap();
    nb.install_frame_future_resolve_handler(RUN_CLASS, &program, 0, 8)
        .unwrap();
    nb.set_frame_cohort_width(1).unwrap();
    let mut wb = MetalResidentSearch::new().unwrap();
    wb.install_frame_future_resolve_handler(RUN_CLASS, &program, 0, 8)
        .unwrap();
    wb.set_frame_cohort_width(32).unwrap();
    reference.run_epoch();
    narrow.run_epoch_with_device_backend(&mut nb);
    wide.run_epoch_with_device_backend(&mut wb);
    assert!(conforms_traces(&reference.trace_snapshot(), &narrow.trace_snapshot()).is_empty());
    assert!(conforms_traces(&reference.trace_snapshot(), &wide.trace_snapshot()).is_empty());
    assert_eq!(narrow.trace_snapshot(), wide.trace_snapshot());
    assert_eq!(nb.last_frame_trace(), wb.last_frame_trace());
    assert_eq!(narrow.speculation_stats().device_evaluated_lanes, 6);
    assert_eq!(narrow.speculation_stats().committed_epochs, 1);
    assert_eq!(wide.speculation_stats().committed_epochs, 1);
    for (process, continuation, future, value) in roots {
        assert_eq!(
            reference.observe_future(process, future).unwrap(),
            Some(value)
        );
        assert_eq!(narrow.observe_future(process, future).unwrap(), Some(value));
        assert_eq!(wide.observe_future(process, future).unwrap(), Some(value));
        let rf = reference.continuation_frame(continuation).unwrap();
        let nf = narrow.continuation_frame(continuation).unwrap();
        assert_eq!(
            reference.object_bytes(process, rf).unwrap(),
            narrow.object_bytes(process, nf).unwrap()
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn resident_future_resolve_conflict_falls_back_before_canonical_commit() {
    use soma::abi::{ObjectKind, ProcessMode, StateAccess};
    use soma::compiler::run_classes::DEFAULT_MAX_STEPS;
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::kernel::speculation::EpochExecutive;
    use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
    use soma::semantics::order::conforms_traces;
    const RUN_CLASS: u32 = 2051;
    let program = compile_evaluator(44_051, "resident-resolve-conflict",
        "field u64\nfield u64\nfield u64\nlet n = load 2\nlet one = const 1\nlet next = add n one\nstore 2 next\n").unwrap();
    let build = || {
        let mut kernel = Kernel::new();
        kernel
            .install_frame_future_resolve(RUN_CLASS, program.clone(), 0, 8)
            .unwrap();
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let future = kernel.create_future(process);
        let value = kernel.create_object(process, ObjectKind::RawBytes, vec![7]);
        for counter in 0..2u64 {
            let mut frame = future.to_u64().to_le_bytes().to_vec();
            frame.extend_from_slice(&value.to_u64().to_le_bytes());
            frame.extend_from_slice(&counter.to_le_bytes());
            kernel
                .create_continuation(
                    SYSTEM_PRINCIPAL,
                    process,
                    ContinuationSpec::new(
                        StateAccess::ReadOnly,
                        RUN_CLASS,
                        0,
                        frame,
                        DEFAULT_MAX_STEPS,
                    ),
                )
                .unwrap();
        }
        (kernel, process, future, value)
    };
    let (mut reference, _process, _future, _value) = build();
    let (mut device, _device_process, device_future, device_value) = build();
    device.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });
    let mut backend = MetalResidentSearch::new().unwrap();
    backend
        .install_frame_future_resolve_handler(RUN_CLASS, &program, 0, 8)
        .unwrap();
    reference.run_epoch();
    device.run_epoch_with_device_backend(&mut backend);
    assert!(conforms_traces(&reference.trace_snapshot(), &device.trace_snapshot()).is_empty());
    assert_eq!(
        device
            .observe_future(SYSTEM_PRINCIPAL, device_future)
            .unwrap(),
        Some(device_value)
    );
    assert_eq!(device.speculation_stats().device_evaluated_lanes, 2);
    assert_eq!(device.speculation_stats().conflict_fallbacks, 1);
    assert_eq!(device.speculation_stats().committed_epochs, 0);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn dynamic_two_handler_frontier_quiesces_without_intermediate_host_reads() {
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::{ResidentDynamicFrameConfig, ResidentDynamicTraceEvent};
    const A: u32 = 2100;
    const B: u32 = 2101;
    let source = |next: u32, multiplier: u32| {
        format!(
        "field u32\nfield u32\nfield u64\nlet remaining = load 0\nlet minus_one = const 0xffffffff\nlet decremented = add remaining minus_one\nstore 0 decremented\nlet next_class = const {next}\nstore 1 next_class\nlet value = load 2\nlet factor = const {multiplier}\nlet product = mul value factor\nlet one = const 1\nlet changed = add product one\nstore 2 changed\n")
    };
    let first = compile_evaluator(45_100, "dynamic-a", &source(B, 3)).unwrap();
    let second = compile_evaluator(45_101, "dynamic-b", &source(A, 5)).unwrap();
    let mut input = Vec::new();
    for (lane, remaining) in [1u32, 2, 3, 5, 7].into_iter().enumerate() {
        input.extend_from_slice(&remaining.to_le_bytes());
        input.extend_from_slice(&A.to_le_bytes());
        input.extend_from_slice(&(lane as u64 + 1).to_le_bytes());
    }
    let config = ResidentDynamicFrameConfig {
        first_run_class: A,
        second_run_class: B,
        max_steps: 8,
        cohort_width: 1,
        remaining_offset: 0,
        next_class_offset: 4,
    };
    let mut cpu_frames = input.clone();
    let mut classes = [A; 5];
    let mut active = [true; 5];
    let mut cpu_trace = Vec::new();
    for epoch in 0..config.max_steps {
        let frozen = cpu_frames.clone();
        for lane in 0..5usize {
            if active[lane] {
                let program = if classes[lane] == A { &first } else { &second };
                let range = lane * 16..lane * 16 + 16;
                program.evaluate_at(&frozen, 5, lane as u32, &mut cpu_frames[range.clone()]);
                let remaining = u32::from_le_bytes(
                    cpu_frames[range.start..range.start + 4].try_into().unwrap(),
                );
                let old_class = classes[lane];
                let step_kind = if remaining == 0 { 1 } else { 2 };
                if remaining == 0 {
                    active[lane] = false;
                } else {
                    classes[lane] = u32::from_le_bytes(
                        cpu_frames[range.start + 4..range.start + 8]
                            .try_into()
                            .unwrap(),
                    );
                }
                let word = cpu_frames[range]
                    .iter()
                    .fold(2_166_136_261u32, |hash, byte| {
                        (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
                    });
                cpu_trace.push(ResidentDynamicTraceEvent {
                    epoch,
                    lane: lane as u32 + 1,
                    run_class: old_class,
                    step_kind,
                    word,
                    ..Default::default()
                });
            }
        }
    }
    assert!(active.iter().all(|active| !active));
    let run = |width| {
        let mut metal = MetalResidentSearch::new().unwrap();
        metal.install_frame_handler(A, &first).unwrap();
        metal.install_frame_handler(B, &second).unwrap();
        metal
            .run_dynamic_frame_graph(
                ResidentDynamicFrameConfig {
                    cohort_width: width,
                    ..config
                },
                &input,
                5,
            )
            .unwrap()
    };
    let narrow = run(1);
    let narrow_repeat = run(1);
    let wide = run(32);
    let wide_repeat = run(32);
    assert_eq!(
        narrow_repeat, narrow,
        "atomic packed rank must remain non-semantic"
    );
    assert_eq!(
        wide_repeat, wide,
        "repeated physical packing must preserve I19"
    );
    assert!(narrow.quiescent);
    assert_eq!(narrow.steps, 18);
    assert_eq!(narrow.first_handler_invocations, 11);
    assert_eq!(narrow.second_handler_invocations, 7);
    assert_eq!(narrow.frames, cpu_frames);
    assert_eq!(narrow.trace, cpu_trace);
    assert_eq!(
        wide, narrow,
        "I19 width neutrality for dynamic rebin/frontier"
    );
    assert!(narrow
        .trace
        .iter()
        .any(|event| event.step_kind == 2 && event.run_class == A));
    assert!(narrow
        .trace
        .iter()
        .any(|event| event.step_kind == 2 && event.run_class == B));
    assert!(narrow.trace.iter().any(|event| event.step_kind == 1));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn dynamic_resident_frontier_rejects_unknown_device_next_class() {
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::batch::BackendError;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::ResidentDynamicFrameConfig;
    let source = |next| {
        format!("field u32\nfield u32\nlet n = load 0\nlet minus = const 0xffffffff\nlet d = add n minus\nstore 0 d\nlet c = const {next}\nstore 1 c\n")
    };
    let first = compile_evaluator(45_110, "invalid-next-a", &source(9_999)).unwrap();
    let second = compile_evaluator(45_111, "invalid-next-b", &source(2_110)).unwrap();
    let mut metal = MetalResidentSearch::new().unwrap();
    metal.install_frame_handler(2_110, &first).unwrap();
    metal.install_frame_handler(2_111, &second).unwrap();
    let mut frame = 2u32.to_le_bytes().to_vec();
    frame.extend_from_slice(&2_110u32.to_le_bytes());
    assert_eq!(
        metal.run_dynamic_frame_graph(
            ResidentDynamicFrameConfig {
                first_run_class: 2_110,
                second_run_class: 2_111,
                max_steps: 1,
                cohort_width: 8,
                remaining_offset: 0,
                next_class_offset: 4,
            },
            &frame,
            1
        ),
        Err(BackendError::InvalidInput)
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn ungrouped_generic_persistent_worker_matches_cohorted_handlers() {
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::ResidentDynamicFrameConfig;
    const A: u32 = 2_300;
    const B: u32 = 2_301;
    const G: u32 = 2_302;
    let prefix = "field u32\nfield u32\nfield u64\nfield u32\nfield u32\nlocal acc\nlet n = load 0\nlet minus = const 0xffffffff\nlet d = add n minus\nstore 0 d\nlet v = load 2\nset acc v\n";
    let loop_a = "repeat 32\nlet stop_a = load 3\nbreak_if stop_a\nlet old_a = get acc\nlet three = const 3\nlet product_a = mul old_a three\nlet one_a = const 1\nlet changed_a = add product_a one_a\nset acc changed_a\nend\n";
    let loop_b = "repeat 32\nlet stop_b = load 4\nbreak_if stop_b\nlet old_b = get acc\nlet five = const 5\nlet product_b = mul old_b five\nlet one_b = const 1\nlet changed_b = add product_b one_b\nset acc changed_b\nend\n";
    let suffix = |next, skip_a, skip_b| {
        format!("let out = get acc\nstore 2 out\nlet next = const {next}\nstore 1 next\nlet sa = const {skip_a}\nstore 3 sa\nlet sb = const {skip_b}\nstore 4 sb\n")
    };
    let first = compile_evaluator(
        46_300,
        "cohorted-a",
        &(prefix.to_owned() + loop_a + &suffix(B, 1, 0)),
    )
    .unwrap();
    let second = compile_evaluator(
        46_301,
        "cohorted-b",
        &(prefix.to_owned() + loop_b + &suffix(A, 0, 1)),
    )
    .unwrap();
    // The generic body can skip the irrelevant counted loop at its head; it
    // does not unfairly compute both class bodies branchlessly.
    let generic_suffix = format!("let out = get acc\nstore 2 out\nlet current_sa = load 3\nlet current_sb = load 4\nstore 3 current_sb\nstore 4 current_sa\nlet base = const {B}\nlet delta = const 0xffffffff\nlet shift = mul current_sa delta\nlet next = add base shift\nstore 1 next\n");
    let generic = compile_evaluator(
        46_302,
        "generic-divergent",
        &(prefix.to_owned() + loop_a + loop_b + &generic_suffix),
    )
    .unwrap();
    let mut input = Vec::new();
    for lane in 0..128u32 {
        input.extend_from_slice(&(1 + lane % 8).to_le_bytes());
        input.extend_from_slice(&A.to_le_bytes());
        input.extend_from_slice(&(lane as u64 + 1).to_le_bytes());
        input.extend_from_slice(&0u32.to_le_bytes());
        input.extend_from_slice(&1u32.to_le_bytes());
    }
    let config = ResidentDynamicFrameConfig {
        first_run_class: A,
        second_run_class: B,
        max_steps: 8,
        cohort_width: 32,
        remaining_offset: 0,
        next_class_offset: 4,
    };
    let mut grouped = MetalResidentSearch::new().unwrap();
    grouped.install_frame_handler(A, &first).unwrap();
    grouped.install_frame_handler(B, &second).unwrap();
    let expected = grouped
        .run_dynamic_frame_graph(config, &input, 128)
        .unwrap();
    let mut ungrouped = MetalResidentSearch::new().unwrap();
    ungrouped.install_frame_handler(G, &generic).unwrap();
    let actual = ungrouped
        .run_dynamic_ungrouped_frame_graph(config, G, &input, 128)
        .unwrap();
    assert_eq!(actual.frames, expected.frames);
    assert_eq!(actual.trace, expected.trace);
    assert_eq!(actual.steps, expected.steps);
    assert!(actual.quiescent);
}
