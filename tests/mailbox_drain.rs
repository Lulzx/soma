//! I25 clause 2, third resource: a mailbox's *occupancy* (`docs/SOMA-v0.3.md`
//! §4.13).
//!
//! §4.12 found two resources an epoch's lanes race for — a domain's quota and a
//! mailbox's capacity — and closed by saying the remaining candidates were
//! places an operation can say no, none of them reachable from a step today.
//! That last clause was wrong, and this file is the case it missed: the mailbox
//! §4.12 filled from one end is drained from the other, and a receive that finds
//! nothing says no as surely as a send that finds no room.
//!
//! Several continuations of one process receive in one epoch and one message is
//! waiting. The message goes to whichever lane ran first — lane 1 under `Plan`,
//! the last lane under `Reverse` — and the rest park. The two runs are not
//! I18-equivalent, and neither clause saw it: no ≺ edge joins the lanes, and
//! clause 2 knew about the mailbox only from the sending end.
//!
//! It is the same experiment as `mailbox_capacity.rs` with the arrow turned
//! around, deliberately so — and turning it around is what showed that clause 2
//! asked the wrong question. `a_message_for_everyone_is_still_a_race` is the
//! case: four receivers and four messages, nobody refused, and the two orders
//! still disagree, because a mailbox hands out *identified* messages where a
//! quota and a capacity hand out interchangeable units. "A winner and a
//! different loser" was a statement about the two resources that existed when it
//! was written.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ObjectKind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, EXPAND_RESUME_0};
use soma::compiler::state_machine_lowering::ExpandFrame;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order, SemanticOrder};

/// One receiver process with `receivers` continuations, all of them at
/// `EXPAND_RESUME_0` — which receives — and `messages` messages waiting in its
/// mailbox. The continuations declare read-only state access, so I13 admits all
/// of them in one epoch and each is a lane of it.
fn drained(order: LaneOrder, receivers: u64, messages: u64) -> Kernel {
    let mut kernel = Kernel::new();
    let requester = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    // The reply the handler sends when it does get a message. A process
    // reference carries no authority, so SEND is delegated.
    kernel
        .grant_capability(SYSTEM_PRINCIPAL, receiver, requester, Rights::SEND, 0, 0)
        .expect("system created both");

    for value in 0..receivers {
        let frame = ExpandFrame::initial(value, requester);
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                receiver,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    EXPAND_RESUME_0,
                    EXPAND_RESUME_0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create the initial continuation");
    }
    for value in 0..messages {
        let payload = kernel.create_object(
            SYSTEM_PRINCIPAL,
            ObjectKind::MessagePayload,
            value.to_le_bytes().to_vec(),
        );
        kernel
            .ingest_message(SYSTEM_PRINCIPAL, requester, receiver, payload, Ref64::NULL)
            .expect("the mailbox has room for the prefill");
    }
    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(30);
    kernel
}

/// Four receivers, one message.
fn one_message(order: LaneOrder) -> Kernel {
    drained(order, 4, 1)
}

/// Two receiver processes, one continuation and one message each. Two lanes of
/// one epoch, both receiving, and nothing shared between them.
fn separate(order: LaneOrder) -> Kernel {
    let mut kernel = Kernel::new();
    let requester = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    for value in 0..2u64 {
        let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .grant_capability(SYSTEM_PRINCIPAL, receiver, requester, Rights::SEND, 0, 0)
            .expect("system created both");
        let frame = ExpandFrame::initial(value, requester);
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                receiver,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    EXPAND_RESUME_0,
                    EXPAND_RESUME_0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create the initial continuation");
        let payload = kernel.create_object(
            SYSTEM_PRINCIPAL,
            ObjectKind::MessagePayload,
            value.to_le_bytes().to_vec(),
        );
        kernel
            .ingest_message(SYSTEM_PRINCIPAL, requester, receiver, payload, Ref64::NULL)
            .expect("an empty mailbox has room");
    }
    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(30);
    kernel
}

fn violations(kernel: &Kernel) -> Vec<String> {
    check(kernel)
        .into_iter()
        .filter(|v| v.invariant == Invariant::LaneIndependence)
        .map(|v| v.detail)
        .collect()
}

/// The lane of every receive that found a message. A lane number is a position
/// in the epoch's plan and does not move when the walk order does (§4.6), so
/// comparing it across two orders compares who won and not who ran.
fn lanes_that_received(kernel: &Kernel) -> Vec<u32> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::MessageReceived)
        .map(|e| e.lane)
        .collect()
}

fn blocked(kernel: &Kernel) -> Vec<&soma::abi::TraceEvent> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::MessageReceiveBlocked)
        .collect()
}

// ---- the counterexample ---------------------------------------------------

#[test]
fn two_lane_orders_disagree_when_the_mailbox_holds_one_message() {
    let plan = one_message(LaneOrder::Plan);
    let reverse = one_message(LaneOrder::Reverse);

    assert_eq!(lanes_that_received(&plan).len(), 1);
    assert_ne!(
        lanes_that_received(&plan),
        lanes_that_received(&reverse),
        "the same lane won under both orders, so nothing was raced for"
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

/// The case that corrected the clause.
///
/// Four receivers and four messages looks like the "room for everyone" null that
/// a quota and a capacity both have — every lane succeeds, nobody is refused —
/// and it is not one. A mailbox hands out *identified* items: each lane takes a
/// different message, from a different sender, with a different sequence number,
/// and which lane takes which is decided by the order. The runs disagree with no
/// refusal anywhere in either of them.
#[test]
fn a_message_for_everyone_is_still_a_race() {
    let plan = drained(LaneOrder::Plan, 4, 4);
    let reverse = drained(LaneOrder::Reverse, 4, 4);
    assert_eq!(lanes_that_received(&plan).len(), 4);
    assert!(
        blocked(&plan).is_empty(),
        "nobody was refused, which is the point"
    );
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(
        !disagreements.is_empty(),
        "a mailbox hands out identified messages, so succeeding is not agreeing"
    );
    assert_eq!(
        violations(&plan).len(),
        1,
        "and the clause has to report it without a refusal to key on: {:?}",
        violations(&plan)
    );
}

/// The null that says the clause is about *one* mailbox: two lanes, two
/// mailboxes, a message in each. Both lanes receive, and the orders agree.
#[test]
fn the_orders_agree_when_each_lane_has_its_own_mailbox() {
    let plan = separate(LaneOrder::Plan);
    let reverse = separate(LaneOrder::Reverse);
    assert_eq!(
        lanes_that_received(&plan).len(),
        2,
        "the null needs both lanes to have really received"
    );
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(disagreements.is_empty(), "{disagreements:?}");
    assert!(violations(&plan).is_empty(), "{:?}", violations(&plan));
}

/// The second null, the one `mailbox_capacity.rs` needed too. An empty mailbox
/// refuses every receiver under every order, so nobody won and no order decided
/// anything — even though the resource is contended-looking, four lanes drew on
/// it, and it refused all four.
#[test]
fn the_orders_agree_when_the_mailbox_is_empty() {
    let plan = drained(LaneOrder::Plan, 4, 0);
    let reverse = drained(LaneOrder::Reverse, 4, 0);
    assert!(lanes_that_received(&plan).is_empty());
    assert_eq!(blocked(&plan).len(), 4, "every receiver must be refused");
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

/// Clause 1 is blind here for the reason it is blind to a quota and to a
/// capacity: the dependence is carried by how many messages are in a mailbox,
/// and an occupancy is not an event.
#[test]
fn clause_1_does_not_see_it() {
    let kernel = one_message(LaneOrder::Plan);
    let edges = SemanticOrder::of(&kernel).cross_lane_edges();
    assert!(edges.is_empty(), "{edges:?}");
    assert!(
        !violations(&kernel).is_empty(),
        "and yet I25 must report it"
    );
}

// ---- the clause -----------------------------------------------------------

#[test]
fn i25_reports_a_drained_mailbox() {
    for order in [LaneOrder::Plan, LaneOrder::Reverse] {
        let reported = violations(&one_message(order));
        assert_eq!(reported.len(), 1, "one report per run: {reported:?}");
        assert!(
            reported[0].contains("received from one mailbox"),
            "{}",
            reported[0]
        );
    }
}

#[test]
fn the_report_names_the_two_lanes_and_is_the_same_text_every_run() {
    let first = violations(&one_message(LaneOrder::Plan));
    for _ in 0..8 {
        assert_eq!(
            violations(&one_message(LaneOrder::Plan)),
            first,
            "the pair the clause reports must not depend on hash iteration order"
        );
    }
}

// ---- the trace ------------------------------------------------------------

#[test]
fn a_blocked_receive_is_in_the_trace() {
    // A receive that found nothing used to leave no record at all: the trace
    // showed a continuation that started and then waited, and could not say
    // whether it was waiting on a future or on an empty mailbox.
    let kernel = one_message(LaneOrder::Plan);
    let events = blocked(&kernel);
    assert_eq!(events.len(), 3, "three of the four receivers lost");
    for event in events {
        assert_eq!(
            event.process.kind,
            soma::abi::Kind::Process,
            "a blocked receive names the mailbox's owner, as a completed one does"
        );
        assert_eq!(
            event.auxiliary, 0,
            "the occupancy that refused it, which emptiness makes constant"
        );
        assert_ne!(event.lane, soma::abi::traces::HOST_LANE);
    }
}

#[test]
fn a_blocked_receiver_parks_rather_than_spinning() {
    // The losers are registered as receiver waiters, so a later message wakes
    // one (§11). Without that the clause would be reporting a race for
    // something that was simply dropped.
    let kernel = one_message(LaneOrder::Plan);
    let receiver = kernel
        .trace_events()
        .iter()
        .find(|e| e.event_kind == EventKind::MessageReceived)
        .map(|e| e.process)
        .expect("one receiver got the message");
    assert_eq!(
        kernel.mailbox_recv_waiter_count(receiver),
        3,
        "a blocked receiver must be parked on the mailbox"
    );
    let other: Vec<_> = check(&kernel)
        .into_iter()
        .filter(|v| v.invariant != Invariant::LaneIndependence)
        .collect();
    assert!(other.is_empty(), "{other:?}");
}
