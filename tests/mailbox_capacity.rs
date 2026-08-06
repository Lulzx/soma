//! I25 clause 2, second resource: a mailbox's capacity (`docs/SOMA-v0.3.md`
//! §4.12).
//!
//! §4.12 found that a domain's process quota makes an epoch's lanes
//! order-dependent, and said the general shape is a lane reading a *decision*
//! off shared bounded state — naming a mailbox a second lane fills as the next
//! candidate. This is that candidate, checked rather than assumed.
//!
//! It behaves exactly as predicted. Several senders reply to one receiver in one
//! epoch; the mailbox has one free slot; the sender that gets it is the one
//! whose lane ran first, and the rest park. Under `Plan` the slot goes to the
//! first sender and under `Reverse` to the last, and the two runs are not
//! I18-equivalent. As with the quota, no ≺ edge joins the lanes.
//!
//! `the_orders_agree_when_the_mailbox_is_already_full` is the case that made the
//! clause sharper than §4.12 first stated it. Every sender blocked is not a
//! race: they all lose under every order. The condition is a winner *and* a
//! different loser.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ObjectKind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, EXPAND_RESUME_2};
use soma::compiler::state_machine_lowering::ExpandFrame;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order, SemanticOrder};

/// A mailbox holds eight messages (§11).
const MAILBOX_CAPACITY: usize = 8;

/// `senders` processes, each of which replies to one shared receiver in its
/// first step, with `prefill` of the receiver's eight slots already taken. The
/// senders run as lanes of one epoch; `8 - prefill` of them get in.
fn contended(order: LaneOrder, senders: u64, prefill: usize) -> Kernel {
    let mut kernel = Kernel::new();
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    for i in 0..prefill {
        let payload = kernel.create_object(
            SYSTEM_PRINCIPAL,
            ObjectKind::MessagePayload,
            (i as u64).to_le_bytes().to_vec(),
        );
        kernel
            .ingest_message(SYSTEM_PRINCIPAL, receiver, receiver, payload, Ref64::NULL)
            .expect("the mailbox has room for the prefill");
    }

    for value in 0..senders {
        let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        // A process reference carries no authority; SEND is delegated.
        kernel
            .grant_capability(SYSTEM_PRINCIPAL, sender, receiver, Rights::SEND, 0, 0)
            .expect("system created both");
        let frame = ExpandFrame::initial(value, receiver);
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                sender,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    EXPAND_RESUME_2,
                    EXPAND_RESUME_2,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create the initial continuation");
    }
    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(30);
    kernel
}

/// Four senders, one free slot.
fn one_slot(order: LaneOrder) -> Kernel {
    contended(order, 4, MAILBOX_CAPACITY - 1)
}

fn violations(kernel: &Kernel) -> Vec<String> {
    check(kernel)
        .into_iter()
        .filter(|v| v.invariant == Invariant::LaneIndependence)
        .map(|v| v.detail)
        .collect()
}

fn senders_that_got_in(kernel: &Kernel) -> Vec<u32> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::MessageSent)
        .map(|e| e.process.slot)
        .collect()
}

fn blocked(kernel: &Kernel) -> Vec<&soma::abi::TraceEvent> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::MessageSendBlocked)
        .collect()
}

// ---- the counterexample ---------------------------------------------------

#[test]
fn two_lane_orders_disagree_when_the_mailbox_has_one_slot() {
    let plan = one_slot(LaneOrder::Plan);
    let reverse = one_slot(LaneOrder::Reverse);

    assert_eq!(senders_that_got_in(&plan).len(), 1);
    assert_ne!(
        senders_that_got_in(&plan),
        senders_that_got_in(&reverse),
        "the same sender won under both orders, so nothing was raced for"
    );
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(
        !disagreements.is_empty(),
        "a contended mailbox must make the epoch's outcome depend on the lane order"
    );
}

/// The null: room for everyone, and the orders agree again.
#[test]
fn the_orders_agree_when_the_mailbox_has_room() {
    let plan = contended(LaneOrder::Plan, 4, 0);
    let reverse = contended(LaneOrder::Reverse, 4, 0);
    assert_eq!(senders_that_got_in(&plan).len(), 4);
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(
        disagreements.is_empty(),
        "an uncontended mailbox decides nothing: {disagreements:?}"
    );
}

/// The second null, and the one that sharpened the clause. Everyone is blocked,
/// so nobody won and no order decided anything — even though the resource is
/// bounded, several lanes drew on it, and it refused.
#[test]
fn the_orders_agree_when_the_mailbox_is_already_full() {
    let plan = contended(LaneOrder::Plan, 4, MAILBOX_CAPACITY);
    let reverse = contended(LaneOrder::Reverse, 4, MAILBOX_CAPACITY);
    assert!(senders_that_got_in(&plan).is_empty());
    assert_eq!(
        blocked(&plan).len(),
        4,
        "every sender must have been refused"
    );
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(disagreements.is_empty(), "{disagreements:?}");
    assert!(
        violations(&plan).is_empty(),
        "all losers is not a race: {:?}",
        violations(&plan)
    );
}

/// Clause 1 is blind to this for the quota's reason: the dependence is carried
/// by a mailbox's occupancy, and occupancy is not an event.
#[test]
fn clause_1_does_not_see_it() {
    let kernel = one_slot(LaneOrder::Plan);
    let edges = SemanticOrder::of(&kernel).cross_lane_edges();
    assert!(edges.is_empty(), "{edges:?}");
    assert!(
        !violations(&kernel).is_empty(),
        "and yet I25 must report it"
    );
}

// ---- the clause -----------------------------------------------------------

#[test]
fn i25_reports_a_contended_mailbox() {
    for order in [LaneOrder::Plan, LaneOrder::Reverse] {
        let reported = violations(&one_slot(order));
        assert_eq!(reported.len(), 1, "one report per run: {reported:?}");
        assert!(reported[0].contains("one mailbox"), "{}", reported[0]);
    }
}

#[test]
fn i25_is_silent_when_the_mailbox_has_room() {
    let kernel = contended(LaneOrder::Plan, 4, 0);
    assert!(blocked(&kernel).is_empty());
    assert!(violations(&kernel).is_empty());
}

// ---- the trace ------------------------------------------------------------

/// A lane that both won and lost is still raced by a lane that only won.
///
/// The clause used to take the lowest-numbered winner and look for a loser that
/// was not it, which is not the same question: a lane that sends twice fills the
/// last slot and is then refused itself, and if it is also the lowest winner
/// then asking only about it finds nothing, while a *higher* lane that got in
/// ahead of it plainly decided the outcome. No workload reaches the shape — a
/// handler sends once — so it is built by appending the three events a workload
/// that sent twice would have produced, in the fault-injection style the
/// repository uses for states it cannot reach by running.
#[test]
fn a_lane_that_won_and_lost_is_still_raced_by_a_lane_that_only_won() {
    let mut kernel = Kernel::new();
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    let event = |kind: EventKind, lane: u32, sequence: u32| {
        let mut row = soma::abi::TraceEvent::new(0, 0, kind, sender, Ref64::NULL, 0);
        row.lane = lane;
        row.lane_sequence = sequence;
        row.causal = receiver;
        row.auxiliary = MAILBOX_CAPACITY as u32;
        row
    };
    let rows = vec![
        // Lane 2 gets in, then fills the mailbox and is refused its second send.
        event(EventKind::MessageSent, 2, 0),
        event(EventKind::MessageSendBlocked, 2, 1),
        // Lane 5 got in too — and if it had run first, lane 2's first send is
        // the one that would have been refused.
        event(EventKind::MessageSent, 5, 0),
    ];
    unsafe { soma::kernel::raw::state(&mut kernel) }
        .trace
        .extend(rows);

    let reported = violations(&kernel);
    assert_eq!(reported.len(), 1, "{reported:?}");
    assert!(reported[0].contains("lanes 2 and 5"), "{}", reported[0]);
}

#[test]
fn a_blocked_send_is_in_the_trace() {
    // Back-pressure used to leave no record: the trace showed a sender that had
    // not sent, and could not say the mailbox was why.
    let kernel = one_slot(LaneOrder::Plan);
    let events = blocked(&kernel);
    assert_eq!(events.len(), 3, "three of the four senders lost");
    for event in events {
        assert_eq!(
            event.causal.kind,
            soma::abi::Kind::Process,
            "a blocked send names its receiver, as a sent one does"
        );
        assert_eq!(event.auxiliary, MAILBOX_CAPACITY as u32);
        assert_ne!(event.lane, soma::abi::traces::HOST_LANE);
    }
}

#[test]
fn a_blocked_sender_parks_rather_than_losing_its_message() {
    // The blocked senders are registered as waiters, so a freed slot wakes them
    // (§11). Without this the clause would be reporting a race for something
    // that was simply dropped.
    let kernel = one_slot(LaneOrder::Plan);
    let receiver = kernel
        .trace_events()
        .iter()
        .find(|e| e.event_kind == EventKind::MessageSent)
        .map(|e| e.causal)
        .expect("one sender got in");
    assert!(
        kernel.mailbox_first_full_waiter(receiver).is_some(),
        "a blocked sender must be parked on the mailbox, not discarded"
    );
    let other: Vec<_> = check(&kernel)
        .into_iter()
        .filter(|v| v.invariant != Invariant::LaneIndependence)
        .collect();
    assert!(other.is_empty(), "{other:?}");
}
