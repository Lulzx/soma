//! Negative-path coverage for the kernel/executive seam.
//!
//! The happy-path suites (`expand_state_machine`, `dynamic_search`) never fill a
//! mailbox, never exhaust a step budget, never put two continuations of one
//! process in the same epoch, and never await an already-resolved future. Those
//! are exactly the paths where the epoch lifecycle can lose or duplicate work,
//! so each one is pinned here.

use soma::abi::continuations::ContinuationState;
use soma::abi::objects::OwnershipState;
use soma::abi::{Kind, ObjectKind, ProcessMode, ProcessState, Ref64};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{
    DEFAULT_MAX_STEPS, EXPAND_RESUME_1, EXPAND_RESUME_2, SEARCH_BRANCH,
};
use soma::compiler::state_machine_lowering::{create_expand, ExpandFrame, SearchFrame};
use soma::kernel::ownership::{assert_live, freeze, ownership_state, transfer_unique};
use soma::kernel::{AwaitOutcome, Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::kernel::raw;

/// A leaf search frame: one node that does no work, spawns nothing, completes.
fn leaf_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    bytes
}

/// Attach a runnable leaf continuation to `process`.
fn spawn_leaf(kernel: &mut Kernel, process: Ref64) -> Ref64 {
    kernel.create_continuation(
        process,
        process,
        SEARCH_BRANCH,
        0,
        leaf_bytes(),
        DEFAULT_MAX_STEPS,
    )
    .unwrap()
}

// ---- §19: the serial-process invariant -----------------------------------

#[test]
fn serial_process_runs_at_most_one_continuation_per_epoch() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let first = spawn_leaf(&mut kernel, p);
    let second = spawn_leaf(&mut kernel, p);

    kernel.run_epoch();

    assert_eq!(
        kernel.continuation_state(first).unwrap(),
        ContinuationState::Completed
    );
    assert_eq!(
        kernel.continuation_state(second).unwrap(),
        ContinuationState::Runnable,
        "a second continuation of the same serial process must defer, not run"
    );
    assert_eq!(kernel.total_pending(), 1, "the deferred continuation is requeued");

    kernel.run_epoch();

    assert_eq!(
        kernel.continuation_state(second).unwrap(),
        ContinuationState::Completed,
        "the deferred continuation runs in the following epoch"
    );
    assert_eq!(kernel.total_pending(), 0);
}

// ---- §8: the step budget --------------------------------------------------

#[test]
fn exhausted_step_budget_faults_before_dispatch_and_leaves_no_bin_entry() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    let mut bytes = Vec::new();
    ExpandFrame::initial(5, p).encode(&mut bytes);
    // A budget of exactly one step: resume_1 runs once, yields into resume_2's
    // bin, and is left with nothing to spend.
    let cont = kernel
        .create_continuation(p, p, EXPAND_RESUME_1, EXPAND_RESUME_1, bytes, 1)
        .unwrap();

    kernel.run_epoch();
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Runnable
    );
    assert_eq!(kernel.total_pending(), 1);

    kernel.run_epoch();

    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Faulted,
        "a continuation over its budget faults instead of running again"
    );
    assert_eq!(kernel.process_state(p).unwrap(), ProcessState::Failed);
    assert_eq!(
        kernel.total_pending(),
        0,
        "the faulted continuation must not be left live in a runnable bin"
    );

    // Nothing more happens: quiescence, not an epoch that keeps re-running it.
    assert_eq!(kernel.run_to_quiescence(10), 0);
}

// ---- §8: re-entry must not repeat side effects ----------------------------

#[test]
fn blocked_reply_does_not_respawn_children_on_re_entry() {
    let mut kernel = Kernel::new();
    let (expand, requester) = create_expand(&mut kernel, 7);

    // Fill the requester's mailbox to capacity so resume_2's reply blocks.
    for _ in 0..8 {
        let filler = kernel.create_object(requester, ObjectKind::MessagePayload, vec![0u8; 8]);
        kernel
            .ingest_message(SYSTEM_PRINCIPAL, requester, requester, filler, Ref64::NULL)
            .unwrap();
    }

    kernel.run_to_quiescence(100);

    let cont = unsafe { raw::state(&mut kernel) }
        .continuations
        .iter()
        .find(|(_, c)| c.process == expand)
        .map(|(r, _)| r)
        .expect("the expand process has a continuation");
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Waiting,
        "resume_2 parks on the full reply mailbox"
    );
    assert_eq!(
        kernel.process_count(),
        5,
        "requester + expand + one child per move"
    );

    // Free one slot and wake the blocked sender, as receive_message would.
    unsafe { raw::state(&mut kernel) }
        .mailboxes
        .get_mut(&requester.slot)
        .unwrap()
        .entries
        .pop_front();
    let state = unsafe { raw::state(&mut kernel) };
    state.continuations.get_mut(cont).unwrap().status = ContinuationState::Runnable;
    state.scheduler.enqueue(EXPAND_RESUME_2, cont);

    kernel.run_to_quiescence(100);

    assert_eq!(
        kernel.process_count(),
        5,
        "re-entering resume_2 must not spawn the children a second time"
    );
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Completed
    );

    let reply = kernel.mailbox_entries(requester).unwrap().back().unwrap();
    assert_eq!(
        kernel.read_u64_object(requester, reply.payload),
        Some(15),
        "the reply carries the heuristic result, sent exactly once"
    );
}

// ---- §11: mailbox back-pressure ------------------------------------------

#[test]
fn every_blocked_sender_is_registered_and_woken_in_order() {
    let mut kernel = Kernel::new();
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender_a = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender_b = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    for sender in [sender_a, sender_b] {
        kernel
            .grant_capability(
                SYSTEM_PRINCIPAL,
                sender,
                receiver,
                soma::abi::Rights::SEND,
                0,
                0,
            )
            .unwrap();
    }

    let cont_a = spawn_leaf(&mut kernel, sender_a);
    let cont_b = spawn_leaf(&mut kernel, sender_b);
    let cont_r = spawn_leaf(&mut kernel, receiver);

    for _ in 0..8 {
        let filler = kernel.create_object(sender_a, ObjectKind::MessagePayload, vec![0u8; 8]);
        kernel
            .ingest_message(SYSTEM_PRINCIPAL, sender_a, receiver, filler, Ref64::NULL)
            .unwrap();
    }

    let payload_a = kernel.create_object(sender_a, ObjectKind::MessagePayload, vec![1u8; 8]);
    let payload_b = kernel.create_object(sender_b, ObjectKind::MessagePayload, vec![2u8; 8]);
    assert_eq!(
        kernel.enqueue_message(sender_a, receiver, payload_a, cont_a),
        Err(RuntimeError::MailboxFull)
    );
    assert_eq!(
        kernel.enqueue_message(sender_b, receiver, payload_b, cont_b),
        Err(RuntimeError::MailboxFull),
        "the second blocked sender must also be registered"
    );
    assert_eq!(
        kernel.mailbox_full_waiter_count(receiver),
        2
    );

    // Retrying does not register the same sender twice.
    assert_eq!(
        kernel.enqueue_message(sender_a, receiver, payload_a, cont_a),
        Err(RuntimeError::MailboxFull)
    );
    assert_eq!(
        kernel.mailbox_full_waiter_count(receiver),
        2
    );

    // One receive frees one slot, so exactly the oldest blocked sender wakes.
    kernel.receive_message(receiver, cont_r).unwrap().unwrap();

    assert_eq!(kernel.mailbox_full_waiter_count(receiver), 1);
    assert_eq!(
        kernel.mailbox_first_full_waiter(receiver),
        Some(cont_b),
        "senders wake in arrival order"
    );
    assert_eq!(
        kernel.continuation_state(cont_a).unwrap(),
        ContinuationState::Runnable
    );
}

// ---- §12: futures ---------------------------------------------------------

#[test]
fn awaiting_a_resolved_future_yields_instead_of_parking() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let cont = spawn_leaf(&mut kernel, p);

    let resolved = kernel.create_future(p);
    let value = kernel.create_object(p, ObjectKind::FutureValue, 42u64.to_le_bytes().to_vec());
    kernel.resolve_future(p, resolved, value).unwrap();

    // Registering here would park forever: resolve_future drains its waiter
    // list exactly once, and that already happened.
    assert_eq!(
        kernel.await_future(p, cont, resolved, EXPAND_RESUME_1).unwrap(),
        AwaitOutcome::AlreadyResolved
    );
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Runnable,
        "the continuation stays runnable rather than waiting on a settled future"
    );

    // A pending future still registers and wakes normally.
    let pending = kernel.create_future(p);
    assert_eq!(
        kernel.await_future(p, cont, pending, EXPAND_RESUME_1).unwrap(),
        AwaitOutcome::Registered
    );
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Waiting
    );

    let value2 = kernel.create_object(p, ObjectKind::FutureValue, 1u64.to_le_bytes().to_vec());
    kernel.resolve_future(p, pending, value2).unwrap();
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        ContinuationState::Runnable
    );
    assert_eq!(kernel.future_value(pending), Some(value2));
}

#[test]
fn futures_are_single_assignment() {
    let mut kernel = Kernel::new();
    let future = kernel.create_future(SYSTEM_PRINCIPAL);
    assert_eq!(kernel.future_value(future), None);

    let first = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::FutureValue, 1u64.to_le_bytes().to_vec());
    kernel.resolve_future(SYSTEM_PRINCIPAL, future, first).unwrap();

    let second = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::FutureValue, 2u64.to_le_bytes().to_vec());
    assert_eq!(
        kernel.resolve_future(SYSTEM_PRINCIPAL, future, second),
        Err(RuntimeError::AlreadyResolved)
    );
    assert_eq!(kernel.future_value(future), Some(first));
}

// ---- §6: ownership transitions -------------------------------------------

#[test]
fn freezing_is_idempotent_and_makes_an_object_untransferable() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let other = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let object = kernel.create_object(owner, ObjectKind::MessagePayload, vec![1, 2, 3]);

    assert_eq!(
        ownership_state(&kernel, object).unwrap(),
        OwnershipState::UniqueMutable
    );
    transfer_unique(&mut kernel, owner, object, other).unwrap();

    let version = freeze(&mut kernel, other, object).unwrap();
    assert_eq!(
        ownership_state(&kernel, object).unwrap(),
        OwnershipState::FrozenShared
    );
    assert_eq!(
        freeze(&mut kernel, other, object).unwrap(),
        version,
        "freezing an already-frozen object does not bump the version again"
    );
    assert!(
        transfer_unique(&mut kernel, other, object, owner).is_err(),
        "a frozen object has no unique authority to transfer"
    );
}

#[test]
fn assert_live_enforces_kind_and_liveness() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let object = kernel.create_object(p, ObjectKind::RawBytes, vec![9]);

    assert_live(&kernel, p, Kind::Process).unwrap();
    assert_live(&kernel, object, Kind::Object).unwrap();
    assert!(
        assert_live(&kernel, object, Kind::Process).is_err(),
        "an object reference is not a process reference"
    );

    let stale = Ref64::new(object.slot, object.generation.wrapping_add(1), Kind::Object);
    assert!(assert_live(&kernel, stale, Kind::Object).is_err());
}
