//! Step 2/3 verification: the `Expand` state machine (§22) runs end-to-end over
//! a real epoch lifecycle, demonstrating resumable continuations driven by a
//! message, a future, a cross-epoch yield, child spawning, and a reply.

use soma::abi::EventKind;
use soma::abi::ProcessState;
use soma::compiler::state_machine_lowering::create_expand;
use soma::kernel::Kernel;
use soma::replay::trace_reader::{event_count, events_of, same_trace, summarize};

fn run_expand(request_value: u64) -> (Kernel, soma::abi::Ref64, soma::abi::Ref64) {
    let mut kernel = Kernel::new();
    let (expand, requester) = create_expand(&mut kernel, request_value);
    // resume_0 is pending (in the next-epoch buffer) before the first epoch.
    assert_eq!(kernel.total_pending(), 1);
    let epochs = kernel.run_to_quiescence(100);
    assert!(epochs > 0, "Expand must make progress");
    (kernel, expand, requester)
}

#[test]
fn expand_process_terminates_and_replies() {
    let (kernel, expand, requester) = run_expand(7);

    // The search process completed.
    assert_eq!(kernel.process_state(expand).unwrap(), ProcessState::Terminated);

    // Exactly one reply arrived at the requester.
    assert_eq!(kernel.mailbox_len(requester), 1);

    // The reply payload is the heuristic result: input*2+1.
    let mb = kernel.mailboxes.get(&requester.slot).unwrap();
    let msg = mb.entries.front().unwrap();
    assert_eq!(kernel.read_u64_object(msg.payload), Some(15));

    // Three children (moves 7,14,21) plus requester, expand, all terminated.
    assert_eq!(kernel.processes.len(), 5);
}

#[test]
fn expand_visits_expected_events_in_order() {
    let (kernel, _expand, _requester) = run_expand(3);

    // Reconstruct the phase sequence from the trace.
    let kinds: Vec<EventKind> = kernel.trace.iter().map(|t| t.event_kind).collect();
    assert!(kinds.contains(&EventKind::ProcessCreated));
    assert!(kinds.contains(&EventKind::MessageReceived), "resume_0 receives the request");
    assert!(kinds.contains(&EventKind::ContinuationWaiting), "resume_0 awaits the heuristic future");
    assert!(kinds.contains(&EventKind::FutureResolved), "heuristic resolves its future");
    assert!(kinds.contains(&EventKind::ContinuationYielded), "resume_1 yields to resume_2");
    assert!(kinds.contains(&EventKind::MessageSent), "resume_2 sends the reply");
    assert!(kinds.contains(&EventKind::ContinuationCompleted));

    // The reply must be sent after the future is resolved.
    let pos_future = kernel
        .trace
        .iter()
        .position(|t| t.event_kind == EventKind::FutureResolved)
        .unwrap();
    let pos_send = kernel
        .trace
        .iter()
        .position(|t| t.event_kind == EventKind::MessageSent)
        .unwrap();
    assert!(pos_send > pos_future);
}

#[test]
fn expand_is_deterministic() {
    let (k1, _, _) = run_expand(11);
    let (k2, _, _) = run_expand(11);
    assert!(same_trace(&k1, &k2), "identical runs must produce identical traces");
}

#[test]
fn expand_reply_value_tracks_input() {
    let (kernel, _expand, requester) = run_expand(100);
    assert_eq!(kernel.mailbox_len(requester), 1);
    let mb = kernel.mailboxes.get(&requester.slot).unwrap();
    let msg = mb.entries.front().unwrap();
    assert_eq!(kernel.read_u64_object(msg.payload), Some(201));
    assert!(event_count(&kernel) > 0);
    assert!(!summarize(&kernel).is_empty());
    assert!(events_of(&kernel, EventKind::FutureResolved).count() >= 1);
}
