use soma::abi::{
    CollectiveState, ContinuationState, FutureState, ObjectKind, ProcessMode, Ref64, Rights,
    StateAccess,
};
use soma::kernel::ownership::freeze;
use soma::kernel::raw;
use soma::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::{assert_legal, check, Invariant};

fn continuation(kernel: &mut Kernel, process: Ref64) -> Ref64 {
    kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
        )
        .unwrap()
}

fn frozen_array(kernel: &mut Kernel, owner: Ref64, bytes: Vec<u8>) -> Ref64 {
    let object = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(kernel, owner, object).unwrap();
    object
}

#[test]
fn channel_transfers_read_authority_and_committed_payload_survives_sender_cancellation() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let channel = kernel.create_channel(owner, 2);
    kernel
        .grant_capability(owner, sender, channel, Rights::SEND, 0, 0)
        .unwrap();
    kernel
        .grant_capability(owner, receiver, channel, Rights::RECEIVE, 0, 0)
        .unwrap();

    let payload = kernel.create_object(sender, ObjectKind::MessagePayload, vec![4, 2]);
    kernel
        .send_channel(sender, channel, payload, Ref64::NULL)
        .unwrap();
    kernel.cancel_process(SYSTEM_PRINCIPAL, sender).unwrap();

    let message = kernel
        .receive_channel(receiver, channel, Ref64::NULL)
        .unwrap()
        .unwrap();
    assert_eq!(message.payload, payload);
    assert!(!message.transferred_capability.is_null());
    assert_eq!(kernel.object_bytes(receiver, payload).unwrap(), &[4, 2]);
    assert_legal(&kernel);
}

#[test]
fn bounded_channel_wakes_waiters_and_drains_after_close() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender_continuation = continuation(&mut kernel, sender);
    let receiver_continuation = continuation(&mut kernel, receiver);
    let channel = kernel.create_channel(owner, 1);
    kernel
        .grant_capability(owner, sender, channel, Rights::SEND, 0, 0)
        .unwrap();
    kernel
        .grant_capability(owner, receiver, channel, Rights::RECEIVE, 0, 0)
        .unwrap();
    let first = kernel.create_object(sender, ObjectKind::MessagePayload, vec![1]);
    let second = kernel.create_object(sender, ObjectKind::MessagePayload, vec![2]);
    kernel
        .send_channel(sender, channel, first, sender_continuation)
        .unwrap();

    assert_eq!(
        kernel.send_channel(sender, channel, second, sender_continuation),
        Err(RuntimeError::MailboxFull)
    );
    {
        let state = unsafe { raw::state(&mut kernel) };
        state.scheduler.remove(sender_continuation);
        state
            .continuations
            .get_mut(sender_continuation)
            .unwrap()
            .status = ContinuationState::Waiting;
    }
    kernel.close_channel(owner, channel).unwrap();
    assert_eq!(
        kernel.continuation_state(sender_continuation).unwrap(),
        ContinuationState::Runnable
    );
    assert_eq!(
        kernel.send_channel(sender, channel, second, sender_continuation),
        Err(RuntimeError::ChannelClosed)
    );
    assert_eq!(
        kernel
            .receive_channel(receiver, channel, receiver_continuation)
            .unwrap()
            .unwrap()
            .payload,
        first
    );
    assert!(matches!(
        kernel.receive_channel(receiver, channel, receiver_continuation),
        Err(RuntimeError::ChannelClosed)
    ));
    assert_legal(&kernel);
}

#[test]
fn channel_capacity_is_an_executable_invariant() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let channel = kernel.create_channel(owner, 1);
    let payload = kernel.create_object(owner, ObjectKind::MessagePayload, vec![1]);
    kernel
        .send_channel(owner, channel, payload, Ref64::NULL)
        .unwrap();
    unsafe { raw::state(&mut kernel) }
        .channels
        .get_mut(channel)
        .unwrap()
        .capacity = 0;
    assert!(check(&kernel)
        .iter()
        .any(|violation| violation.invariant == Invariant::MailboxBound));
}

#[test]
fn batch_evaluate_publishes_only_frozen_arrays_through_its_future() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mutable = kernel.create_object(owner, ObjectKind::FrozenArray, vec![0; 8]);
    assert_eq!(
        kernel.create_batch_evaluate(owner, mutable, 2, 4),
        Err(RuntimeError::InvalidCollective)
    );

    let inputs = frozen_array(&mut kernel, owner, vec![1; 8]);
    let (collective, completion) = kernel.create_batch_evaluate(owner, inputs, 2, 4).unwrap();
    let outputs = frozen_array(&mut kernel, owner, vec![9; 8]);
    kernel
        .complete_batch_evaluate(owner, collective, outputs)
        .unwrap();

    assert_eq!(
        kernel.collective_state(collective).unwrap(),
        CollectiveState::Completed
    );
    assert_eq!(
        kernel.future_state(completion).unwrap(),
        FutureState::Resolved
    );
    assert_eq!(kernel.future_value(completion), Some(outputs));
    assert_legal(&kernel);
}

#[test]
fn cancelling_collective_owner_settles_collective_and_completion_future() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = frozen_array(&mut kernel, owner, vec![1; 4]);
    let (collective, completion) = kernel.create_batch_evaluate(owner, inputs, 1, 4).unwrap();
    kernel.cancel_process(SYSTEM_PRINCIPAL, owner).unwrap();
    assert_eq!(
        kernel.collective_state(collective).unwrap(),
        CollectiveState::Cancelled
    );
    assert_eq!(
        kernel.future_state(completion).unwrap(),
        FutureState::Cancelled
    );
    assert_legal(&kernel);
}

#[test]
fn collective_and_completion_future_must_agree() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = frozen_array(&mut kernel, owner, vec![1; 4]);
    let (collective, _) = kernel.create_batch_evaluate(owner, inputs, 1, 4).unwrap();
    unsafe { raw::state(&mut kernel) }
        .collectives
        .get_mut(collective)
        .unwrap()
        .state = CollectiveState::Completed;
    assert!(check(&kernel)
        .iter()
        .any(|violation| violation.invariant == Invariant::FutureSingleAssignment));
}
