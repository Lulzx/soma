use soma::abi::continuations::ContinuationState;
use soma::abi::{EventKind, Kind, ProcessMode, Ref64, StateAccess};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn continuation(kernel: &mut Kernel) -> Ref64 {
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(StateAccess::ReadOnly, 17, 0, Vec::new(), 8),
        )
        .unwrap()
}

fn ready_count(kernel: &Kernel, continuation: Ref64) -> usize {
    kernel
        .trace_events()
        .iter()
        .filter(|event| {
            event.event_kind == EventKind::ContinuationReady && event.continuation == continuation
        })
        .count()
}

#[test]
fn stale_future_wake_from_colliding_node_does_not_wake_reparked_continuation() {
    let mut kernel = Kernel::new();
    let continuation = continuation(&mut kernel);
    let entity = Ref64::new(77, 4, Kind::Future);

    kernel
        .register_remote_future_waiter(continuation, 11, entity, 23)
        .unwrap();
    assert_legal(&kernel); // I1: the remote entity never entered the descriptor; I7: parked.
    kernel
        .register_remote_future_waiter(continuation, 12, entity, 29)
        .unwrap();
    assert_legal(&kernel);

    let before = ready_count(&kernel, continuation);
    kernel.wake_remote_future_waiter(continuation, 11, entity);
    assert_eq!(
        kernel.continuation_state(continuation),
        Ok(ContinuationState::Waiting)
    );
    assert_eq!(
        ready_count(&kernel, continuation),
        before,
        "stale cross-node receipt must not trace a wake"
    );
    assert_legal(&kernel); // I1/I7 must also hold after the rejected wake.

    kernel.wake_remote_future_waiter(continuation, 12, entity);
    assert_eq!(
        kernel.continuation_state(continuation),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(ready_count(&kernel, continuation), before + 1);
    assert_legal(&kernel);
}

#[test]
fn stale_channel_wake_from_colliding_node_does_not_wake_reparked_continuation() {
    let mut kernel = Kernel::new();
    let continuation = continuation(&mut kernel);
    let entity = Ref64::new(77, 4, Kind::Channel);

    kernel
        .register_remote_channel_waiter(continuation, 21, entity, 31)
        .unwrap();
    assert_legal(&kernel);
    kernel
        .register_remote_channel_waiter(continuation, 22, entity, 37)
        .unwrap();
    assert_legal(&kernel);

    let before = ready_count(&kernel, continuation);
    kernel.wake_remote_channel_waiter(continuation, 21, entity);
    assert_eq!(
        kernel.continuation_state(continuation),
        Ok(ContinuationState::Waiting)
    );
    assert_eq!(
        ready_count(&kernel, continuation),
        before,
        "stale cross-node receipt must not trace a wake"
    );
    assert_legal(&kernel);

    kernel.wake_remote_channel_waiter(continuation, 22, entity);
    assert_eq!(
        kernel.continuation_state(continuation),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(ready_count(&kernel, continuation), before + 1);
    assert_legal(&kernel);
}
