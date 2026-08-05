use soma::abi::continuations::ContinuationState;
use soma::abi::{ObjectKind, ProcessMode, ProcessState, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::SearchFrame;
use soma::experiments::multi_input_graph::{run, MultiInputConfig};
use soma::kernel::raw;
use soma::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};

#[test]
fn all_input_receive_is_atomic() {
    let mut kernel = Kernel::new();
    let join = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let source = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let left = kernel.create_channel(join, 1);
    let right = kernel.create_channel(join, 1);
    for channel in [left, right] {
        kernel
            .grant_capability(join, source, channel, Rights::SEND, 0, 0)
            .unwrap();
    }
    let left_value = kernel.create_object(source, ObjectKind::MessagePayload, vec![1]);
    let right_value = kernel.create_object(source, ObjectKind::MessagePayload, vec![2]);
    kernel
        .send_channel(source, left, left_value, Ref64::NULL)
        .unwrap();

    assert!(kernel
        .receive_channels_all(join, &[left, right], Ref64::NULL)
        .unwrap()
        .is_none());
    assert_eq!(kernel.channel_len(left).unwrap(), 1);
    assert_eq!(kernel.channel_len(right).unwrap(), 0);

    kernel
        .send_channel(source, right, right_value, Ref64::NULL)
        .unwrap();
    let pair = kernel
        .receive_channels_all(join, &[left, right], Ref64::NULL)
        .unwrap()
        .unwrap();
    assert_eq!(pair[0].payload, left_value);
    assert_eq!(pair[1].payload, right_value);
}

#[test]
fn skewed_inputs_preserve_fifo_and_apply_backpressure() {
    let report = run(MultiInputConfig::default()).unwrap();
    assert!(report.legal);
    assert!(report.ordered);
    assert_eq!(report.joined.len(), 16);
    assert!(report.left_backpressure > 0);
    assert_eq!(report.left_state, ProcessState::Created);
    assert_eq!(report.right_state, ProcessState::Created);
}

#[test]
fn producer_failure_preserves_exactly_the_committed_prefix() {
    let report = run(MultiInputConfig {
        fail_left_after: Some(7),
        ..MultiInputConfig::default()
    })
    .unwrap();
    assert!(report.legal);
    assert!(report.committed_prefix_preserved);
    assert_eq!(report.joined.len(), 7);
    assert_eq!(report.left_state, ProcessState::Failed);
    assert_eq!(report.right_state, ProcessState::Created);
}

#[test]
fn multi_input_runs_are_deterministic() {
    let config = MultiInputConfig::default();
    assert_eq!(run(config), run(config));
}

#[test]
fn duplicate_inputs_are_rejected() {
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let channel = kernel.create_channel(actor, 1);
    assert!(matches!(
        kernel.receive_channels_all(actor, &[channel, channel], Ref64::NULL),
        Err(RuntimeError::InvalidMultiInput)
    ));
}

#[test]
fn retry_moves_the_waiter_to_the_inputs_that_are_still_missing() {
    let mut kernel = Kernel::new();
    let join = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let source = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let left = kernel.create_channel(join, 1);
    let right = kernel.create_channel(join, 1);
    for channel in [left, right] {
        kernel
            .grant_capability(join, source, channel, Rights::SEND, 0, 0)
            .unwrap();
    }
    let mut frame = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut frame);
    let waiter = kernel
        .create_continuation(
            join,
            join,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_BRANCH,
                0,
                frame,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();
    assert!(kernel
        .receive_channels_all(join, &[left, right], waiter)
        .unwrap()
        .is_none());
    {
        let state = unsafe { raw::state(&mut kernel) };
        state.scheduler.remove(waiter);
        state.continuations.get_mut(waiter).unwrap().status = ContinuationState::Waiting;
    }

    let left_value = kernel.create_object(source, ObjectKind::MessagePayload, vec![1]);
    kernel
        .send_channel(source, left, left_value, Ref64::NULL)
        .unwrap();
    assert_eq!(
        kernel.continuation_state(waiter).unwrap(),
        ContinuationState::Runnable
    );
    assert!(kernel
        .receive_channels_all(join, &[left, right], waiter)
        .unwrap()
        .is_none());
    assert_eq!(kernel.channel_len(left).unwrap(), 1);
    {
        let state = unsafe { raw::state(&mut kernel) };
        state.scheduler.remove(waiter);
        state.continuations.get_mut(waiter).unwrap().status = ContinuationState::Waiting;
    }

    let right_value = kernel.create_object(source, ObjectKind::MessagePayload, vec![2]);
    kernel
        .send_channel(source, right, right_value, Ref64::NULL)
        .unwrap();
    assert_eq!(
        kernel.continuation_state(waiter).unwrap(),
        ContinuationState::Runnable
    );
    assert!(kernel
        .receive_channels_all(join, &[left, right], waiter)
        .unwrap()
        .is_some());
}
