use soma::abi::{ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{
    DEFAULT_MAX_STEPS, EXPAND_RESUME_2, POLL_FUTURE, SEARCH_HEURISTIC,
};
use soma::compiler::state_machine_lowering::{
    create_expand, ExpandFrame, HeuristicFrame, JoinFrame, SearchFrame,
};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::speculation::EpochExecutive;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::device_ops::{ALL_OPERATION_KINDS, OP_CREATE_PROCESS, OP_ENQUEUE_MESSAGE};
use soma::semantics::invariants::assert_legal;
use soma::semantics::order::conforms_traces;

fn leaf_knobs() -> ControlKnobs {
    ControlKnobs {
        branching_factor: 0,
        depth: 0,
        process_count: 8,
        class_count: 3,
        arithmetic_ops: 2_000,
        ..ControlKnobs::default()
    }
}

#[test]
fn disjoint_lanes_commit_the_reference_run() {
    let knobs = leaf_knobs();
    let mut reference = build(&knobs);
    let mut speculative = build(&knobs);
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });

    assert_eq!(reference.run_epoch(), knobs.process_count as usize);
    assert_eq!(speculative.run_epoch(), knobs.process_count as usize);

    let stats = speculative.speculation_stats();
    assert_eq!(stats.attempted_epochs, 1);
    assert_eq!(stats.committed_epochs, 1);
    assert_eq!(stats.fallback_epochs, 0);
    assert_eq!(stats.committed_lanes, knobs.process_count as u64);
    assert!(
        conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot()).is_empty(),
        "canonical commit must reproduce the reference trace"
    );
    assert_legal(&speculative);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn actual_search_leaf_handlers_execute_and_validate_on_metal_before_commit() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let knobs = leaf_knobs();
    let mut reference = build(&knobs);
    let mut device_validated = build(&knobs);
    let continuations: Vec<_> = reference
        .trace_snapshot()
        .into_iter()
        .filter(|row| {
            row.event_kind == soma::abi::EventKind::ContinuationReady
                && row.run_class >= soma::compiler::run_classes::SEARCH_BRANCH
        })
        .map(|row| {
            (
                Ref64::from_u64(row.process),
                Ref64::from_u64(row.continuation),
            )
        })
        .collect();
    device_validated.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });
    let mut metal = MetalDeviceScheduler::new().unwrap();

    reference.run_epoch();
    device_validated.run_epoch_with_device_backend(&mut metal);

    let stats = device_validated.speculation_stats();
    assert_eq!(stats.device_evaluated_epochs, 1);
    assert_eq!(stats.device_evaluated_lanes, knobs.process_count as u64);
    assert_eq!(stats.committed_epochs, 1);
    assert!(conforms_traces(
        &reference.trace_snapshot(),
        &device_validated.trace_snapshot()
    )
    .is_empty());
    for (process, continuation) in continuations {
        let reference_frame = reference.continuation_frame(continuation).unwrap();
        let device_frame = device_validated.continuation_frame(continuation).unwrap();
        assert_eq!(
            reference.object_bytes(process, reference_frame).unwrap(),
            device_validated
                .object_bytes(process, device_frame)
                .unwrap(),
            "device handler must publish the exact reference frame bytes"
        );
    }
    assert_legal(&device_validated);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn arbitrary_evaluator_program_runs_as_a_metal_continuation_handler() {
    use soma::compiler::examples;
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    const RUN_CLASS: u32 = 1024;
    let module = examples::module();
    let program = module.program(examples::RUN_LENGTH).unwrap().clone();
    let build = || {
        let mut kernel = Kernel::new();
        kernel
            .install_frame_evaluator(RUN_CLASS, program.clone())
            .unwrap();
        let mut lanes = Vec::new();
        for (left, right) in [(1u32, 2u32), (17, 31), (u32::MAX, 9), (55, 55)] {
            let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            let mut frame = left.to_le_bytes().to_vec();
            frame.extend_from_slice(&right.to_le_bytes());
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
            lanes.push((process, continuation));
        }
        (kernel, lanes)
    };

    let (mut reference, lanes) = build();
    let (mut device, _) = build();
    device.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });
    let mut metal = MetalDeviceScheduler::new().unwrap();
    metal.install_frame_evaluator(RUN_CLASS, &program).unwrap();

    reference.run_epoch();
    device.run_epoch_with_device_backend(&mut metal);

    let stats = device.speculation_stats();
    assert_eq!(stats.device_evaluated_epochs, 1);
    assert_eq!(stats.device_evaluated_lanes, lanes.len() as u64);
    assert!(conforms_traces(&reference.trace_snapshot(), &device.trace_snapshot()).is_empty());
    for (process, continuation) in lanes {
        let expected = reference.continuation_frame(continuation).unwrap();
        let actual = device.continuation_frame(continuation).unwrap();
        assert_eq!(
            reference.object_bytes(process, expected).unwrap(),
            device.object_bytes(process, actual).unwrap()
        );
    }
    assert_legal(&device);
}

#[cfg(feature = "native")]
#[test]
fn arbitrary_evaluator_program_runs_as_native_compiled_continuation_handler() {
    use soma::compiler::examples;
    use soma::executives::native::NativeEpochBackend;

    const RUN_CLASS: u32 = 1025;
    let module = examples::module();
    let program = module.program(examples::RUN_LENGTH).unwrap().clone();
    let build = || {
        let mut kernel = Kernel::new();
        kernel
            .install_frame_evaluator(RUN_CLASS, program.clone())
            .unwrap();
        let mut lanes = Vec::new();
        for (left, right) in [(1u32, 2u32), (17, 17), (u32::MAX, 1), (0, 0)] {
            let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            let mut frame = left.to_le_bytes().to_vec();
            frame.extend_from_slice(&right.to_le_bytes());
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
            lanes.push((process, continuation));
        }
        (kernel, lanes)
    };

    let (mut reference, lanes) = build();
    let (mut native, _) = build();
    native.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });
    let mut backend = NativeEpochBackend::new().unwrap();
    backend
        .install_frame_evaluator(RUN_CLASS, &program)
        .unwrap();

    reference.run_epoch();
    native.run_epoch_with_device_backend(&mut backend);

    let stats = native.speculation_stats();
    assert_eq!(stats.device_evaluated_epochs, 1);
    assert_eq!(stats.device_evaluated_lanes, lanes.len() as u64);
    assert!(conforms_traces(&reference.trace_snapshot(), &native.trace_snapshot()).is_empty());
    for (process, continuation) in lanes {
        let expected = reference.continuation_frame(continuation).unwrap();
        let actual = native.continuation_frame(continuation).unwrap();
        assert_eq!(
            reference.object_bytes(process, expected).unwrap(),
            native.object_bytes(process, actual).unwrap()
        );
    }
    assert_legal(&native);
}

fn two_leaves_in_one_process() -> Kernel {
    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    for value in [11, 29] {
        let frame = SearchFrame {
            value,
            depth: 0,
            branching: 0,
            work_iters: 64,
            class_count: 1,
        };
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    frame.run_class(),
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel
}

#[test]
fn a_process_commit_conflict_replays_the_whole_epoch() {
    let mut reference = two_leaves_in_one_process();
    let mut speculative = two_leaves_in_one_process();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    let stats = speculative.speculation_stats();
    assert_eq!(stats.committed_epochs, 0);
    assert_eq!(stats.fallback_epochs, 1);
    assert_eq!(stats.conflict_fallbacks, 1);
    let disagreements = conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
    assert_legal(&speculative);
}

#[test]
fn contended_allocation_falls_back_before_any_snapshot_effect_can_escape() {
    let knobs = ControlKnobs {
        depth: 1,
        process_count: 3,
        branching_factor: 2,
        ..leaf_knobs()
    };
    let mut reference = build(&knobs);
    let mut speculative = build(&knobs);
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    let stats = speculative.speculation_stats();
    assert_eq!(stats.fallback_epochs, 1);
    assert_eq!(stats.conflict_fallbacks, 1);
    assert_ne!(
        stats.device_operation_kinds & (1 << (OP_CREATE_PROCESS - 1)),
        0
    );
    assert!(conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot()).is_empty());
    assert_legal(&speculative);
}

fn independent_expands() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(16);
    for value in 1..=4 {
        create_expand(&mut kernel, value);
    }
    kernel
}

#[test]
fn independent_mailboxes_futures_and_allocations_commit() {
    let mut reference = independent_expands();
    let mut speculative = independent_expands();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });

    reference.run_to_quiescence(64);
    speculative.run_to_quiescence(64);

    let stats = speculative.speculation_stats();
    assert!(stats.committed_epochs >= 2, "{stats:?}");
    assert!(stats.committed_lanes >= 8, "{stats:?}");
    assert_eq!(stats.device_operation_kinds, ALL_OPERATION_KINDS);
    let disagreements = conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
    assert_legal(&speculative);
}

fn contested_future() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(8);
    let future = kernel.create_future(SYSTEM_PRINCIPAL);
    for input in [7, 13] {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .grant_capability(SYSTEM_PRINCIPAL, process, future, Rights::RESOLVE, 0, 0)
            .unwrap();
        let frame = HeuristicFrame { future, input };
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    SEARCH_HEURISTIC,
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel
}

#[test]
fn two_future_writers_conflict_and_replay_in_plan_order() {
    let mut reference = contested_future();
    let mut speculative = contested_future();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    let stats = speculative.speculation_stats();
    assert_eq!(stats.committed_epochs, 0);
    assert_eq!(stats.conflict_fallbacks, 1);
    let disagreements = conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_conflict_decision_falls_back_before_future_writes_escape() {
    use soma::executives::metal_scheduler::MetalDeviceScheduler;

    let mut reference = contested_future();
    let mut device_validated = contested_future();
    device_validated.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });
    let mut metal = MetalDeviceScheduler::new().unwrap();

    reference.run_epoch();
    device_validated.run_epoch_with_lane_validator(&mut metal);

    assert_eq!(device_validated.speculation_stats().conflict_fallbacks, 1);
    assert!(conforms_traces(
        &reference.trace_snapshot(),
        &device_validated.trace_snapshot()
    )
    .is_empty());
}

fn future_poll_race() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(8);
    let future = kernel.create_future(SYSTEM_PRINCIPAL);

    let resolver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .grant_capability(SYSTEM_PRINCIPAL, resolver, future, Rights::RESOLVE, 0, 0)
        .unwrap();
    let mut resolver_frame = Vec::new();
    HeuristicFrame { future, input: 41 }.encode(&mut resolver_frame);
    kernel
        .create_continuation(
            SYSTEM_PRINCIPAL,
            resolver,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_HEURISTIC,
                0,
                resolver_frame,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();

    let poller = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .grant_capability(SYSTEM_PRINCIPAL, poller, future, Rights::AWAIT, 0, 0)
        .unwrap();
    let mut poller_frame = Vec::new();
    JoinFrame {
        future,
        observed: Ref64::NULL,
    }
    .encode(&mut poller_frame);
    kernel
        .create_continuation(
            SYSTEM_PRINCIPAL,
            poller,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                POLL_FUTURE,
                0,
                poller_frame,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();
    kernel
}

#[test]
fn a_future_poll_conflicts_with_resolution() {
    let mut reference = future_poll_race();
    let mut speculative = future_poll_race();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    assert_eq!(speculative.speculation_stats().conflict_fallbacks, 1);
    let disagreements = conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
}

fn contested_mailbox() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(8);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    for value in [5, 9] {
        let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .grant_capability(SYSTEM_PRINCIPAL, sender, receiver, Rights::SEND, 0, 0)
            .unwrap();
        let mut frame = ExpandFrame::initial(value, receiver);
        frame.heuristic_result = value;
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                sender,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    EXPAND_RESUME_2,
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel
}

#[test]
fn two_senders_to_one_mailbox_conflict() {
    let mut reference = contested_mailbox();
    let mut speculative = contested_mailbox();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    assert_eq!(speculative.speculation_stats().conflict_fallbacks, 1);
    assert_ne!(
        speculative.speculation_stats().device_operation_kinds & (1 << (OP_ENQUEUE_MESSAGE - 1)),
        0
    );
    let disagreements = conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
}
