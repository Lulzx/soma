//! I25, sixth decision: a future read without being awaited
//! (`docs/SOMA-v0.3.md` §4.16).
//!
//! §4.15 widened the question to "which operations have a result another lane
//! can decide". Re-walking `LaneView`'s fifteen under that question turns up one
//! the narrower question had no reason to look at, because it cannot fail and
//! it does not block: `future_value`. It was also the one read on the view that
//! was neither authorized nor traced.
//!
//! A resolver lane and a polling lane, one future, one epoch. The poll returns
//! the value under `Plan` and nothing under `Reverse`, and before this section
//! **nothing reported that at all** — not clause 1, which has no edge to find
//! because nothing was woken; not clause 2, which had no event to key on; and
//! not the run comparison, because what the poll saw went into a frame and a
//! frame is not observable behaviour. Two runs, I18-equivalent, leaving
//! different state.
//!
//! `POLL_ACT` is what makes that undeniable rather than merely true: the poller
//! sends a message in a later epoch if it saw a value. Then the runs do
//! disagree — one epoch after the epoch that decided it, which by every clause
//! was clean.
//!
//! The fix is the one the machine already applies to its other reads. Looking at
//! a future is now a governed effect: it authorizes `AWAIT`, the right that
//! already means "may observe this future", and it emits `FutureStateObserved`
//! recording *what it saw*. Both outcomes are recorded, because a poll that
//! found the future still pending was decided by the resolver's lane exactly as
//! much as one that found the value.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ObjectKind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::{ByteCursor, Frame};
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, POLL_FUTURE, SEARCH_HEURISTIC};
use soma::compiler::state_machine_lowering::{HeuristicFrame, JoinFrame};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order, SemanticOrder};

#[derive(Clone, Copy, PartialEq)]
enum Setup {
    /// A resolver lane and a polling lane, on one future, in one epoch.
    Contested,
    /// The value is published before the epoch, so the poll sees it under every
    /// order and no lane resolved anything.
    SettledFirst,
    /// The resolver publishes into a future of its own; the polled one is never
    /// resolved.
    SeparateFutures,
    /// Nobody resolves anything, so the poll sees nothing under every order.
    NoResolver,
    /// The poller holds no capability on the future it is about to look at.
    Unauthorized,
}

/// The run, plus the poller's process and continuation — the frame is where the
/// interesting part of the result is, and it is not in the trace.
struct Run {
    kernel: Kernel,
    poller: Ref64,
    polling: Ref64,
}

fn workload(order: LaneOrder, setup: Setup) -> Run {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let polled = kernel.create_future(owner);

    if setup != Setup::NoResolver {
        let target = if setup == Setup::SeparateFutures {
            kernel.create_future(owner)
        } else {
            polled
        };
        let resolver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .grant_capability(owner, resolver, target, Rights::RESOLVE, 0, 0)
            .expect("the owner holds the genesis capability");
        let mut bytes = Vec::new();
        HeuristicFrame {
            future: target,
            input: 7,
        }
        .encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                resolver,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    SEARCH_HEURISTIC,
                    SEARCH_HEURISTIC,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create the initial continuation");
    }

    let poller = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    if setup != Setup::Unauthorized {
        // A poll is the non-blocking form of an await, so it asks for the same
        // right. Without this grant the read is denied, which is the null that
        // says the read is governed at all.
        kernel
            .grant_capability(owner, poller, polled, Rights::AWAIT, 0, 0)
            .expect("the owner holds the genesis capability");
    }
    let mut bytes = Vec::new();
    JoinFrame {
        future: polled,
        observed: Ref64::NULL,
    }
    .encode(&mut bytes);
    let polling = kernel
        .create_continuation(
            SYSTEM_PRINCIPAL,
            poller,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                POLL_FUTURE,
                POLL_FUTURE,
                bytes,
                DEFAULT_MAX_STEPS,
            ),
        )
        .expect("system may create the initial continuation");

    if setup == Setup::SettledFirst {
        let value = kernel.create_object(
            SYSTEM_PRINCIPAL,
            ObjectKind::FutureValue,
            0u64.to_le_bytes().to_vec(),
        );
        kernel
            .resolve_future(owner, polled, value)
            .expect("the future is pending before the epoch");
    }

    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(30);
    Run {
        kernel,
        poller,
        polling,
    }
}

impl Run {
    /// What the poll put in its frame. The part of the result the trace does not
    /// carry, which is the whole reason this section exists.
    fn observed(&mut self) -> Ref64 {
        let frame = self
            .kernel
            .continuation_frame(self.polling)
            .expect("the poller's continuation outlives the run");
        let bytes = self
            .kernel
            .object_bytes(self.poller, frame)
            .expect("a process may read its own frame")
            .to_vec();
        let mut cursor = ByteCursor::new(&bytes);
        <JoinFrame as Frame>::decode(&mut cursor)
            .expect("the frame is a JoinFrame")
            .observed
    }

    fn violations(&self) -> Vec<String> {
        check(&self.kernel)
            .into_iter()
            .filter(|v| v.invariant == Invariant::LaneIndependence)
            .map(|v| v.detail)
            .collect()
    }

    fn events(&self, kind: EventKind) -> Vec<&soma::abi::TraceEvent> {
        self.kernel
            .trace_events()
            .iter()
            .filter(|e| e.event_kind == kind)
            .collect()
    }
}

fn disagreements(a: &Run, b: &Run) -> Vec<soma::semantics::order::OrderViolation> {
    conforms_traces(
        &in_position_order(&a.kernel.trace_snapshot()),
        &in_position_order(&b.kernel.trace_snapshot()),
    )
}

// ---- the counterexample ---------------------------------------------------

/// The two orders leave different state. This is the assertion the whole
/// section turns on, and it reads a frame to make it, because a frame is where
/// the difference is.
#[test]
fn the_two_orders_leave_the_poller_holding_different_things() {
    let mut plan = workload(LaneOrder::Plan, Setup::Contested);
    let mut reverse = workload(LaneOrder::Reverse, Setup::Contested);
    assert_eq!(
        plan.observed().kind,
        soma::abi::Kind::Object,
        "the poll ran after the resolve and saw the value"
    );
    assert_eq!(
        reverse.observed(),
        Ref64::NULL,
        "and before it, under the other order, saw nothing"
    );
}

/// Clause 1 has nothing to find, in either order, and this is the one race in
/// the series where that is not a limitation of the trace but of the workload:
/// nothing parks, so nothing is woken, so no event names another lane's
/// continuation. §4.15's edge does not exist here.
#[test]
fn clause_1_is_blind_in_both_orders() {
    for order in [LaneOrder::Plan, LaneOrder::Reverse] {
        let run = workload(order, Setup::Contested);
        let edges = SemanticOrder::of(&run.kernel).cross_lane_edges();
        assert!(edges.is_empty(), "{order:?}: {edges:?}");
    }
}

/// The deciding epoch is epoch 0, and clause 2 reports it there — in both
/// orders, because the poll records what it saw either way.
#[test]
fn the_epoch_that_was_decided_is_the_one_that_reports() {
    for order in [
        LaneOrder::Plan,
        LaneOrder::Reverse,
        LaneOrder::Permuted(3),
        LaneOrder::Permuted(4),
    ] {
        let run = workload(order, Setup::Contested);
        let reported = run.violations();
        assert_eq!(reported.len(), 1, "{order:?}: {reported:?}");
        assert!(reported[0].starts_with("epoch 0:"), "{}", reported[0]);
        assert!(
            reported[0].contains("read whether it had been resolved"),
            "{}",
            reported[0]
        );
    }
}

/// What the trace comparison sees, and when. `POLL_ACT` sends a message if the
/// poll saw a value, so the divergence reaches the trace — an epoch after the
/// one that caused it. Without the event this section adds, that later epoch is
/// the *only* place any checker could notice, and by then the epoch whose lane
/// order decided it has already been declared clean.
#[test]
fn the_divergence_reaches_the_trace_an_epoch_late() {
    let plan = workload(LaneOrder::Plan, Setup::Contested);
    let reverse = workload(LaneOrder::Reverse, Setup::Contested);

    let sent = plan.events(EventKind::MessageSent);
    assert_eq!(sent.len(), 1, "the poll saw a value and acted on it");
    assert_eq!(sent[0].epoch, 1, "in the epoch after the one that decided");
    assert!(
        reverse.events(EventKind::MessageSent).is_empty(),
        "and under the other order there was nothing to act on"
    );
    assert!(!disagreements(&plan, &reverse).is_empty());
}

/// And the deciding epoch itself: the two runs emit the same event *kinds* in
/// the same order there, and differ in the fields of exactly one event — the
/// one recording what the poll saw. That is the precise sense in which nothing
/// else could have reported this: strip this event's payload and epoch 0 is
/// identical under both orders.
#[test]
fn the_deciding_epoch_differs_in_one_events_fields_and_nothing_else() {
    let plan = workload(LaneOrder::Plan, Setup::Contested);
    let reverse = workload(LaneOrder::Reverse, Setup::Contested);

    let kinds = |run: &Run| -> Vec<EventKind> {
        let mut rows: Vec<_> = run
            .kernel
            .trace_events()
            .iter()
            .filter(|e| e.epoch == 0)
            .collect();
        rows.sort_by_key(|e| (e.lane, e.lane_sequence));
        rows.into_iter().map(|e| e.event_kind).collect()
    };
    assert_eq!(kinds(&plan), kinds(&reverse), "same events, in the same places");

    let observed = |run: &Run| {
        let events = run.events(EventKind::FutureStateObserved);
        assert_eq!(events.len(), 1);
        (events[0].auxiliary, events[0].subject)
    };
    let (plan_aux, plan_subject) = observed(&plan);
    let (reverse_aux, reverse_subject) = observed(&reverse);
    assert_eq!((plan_aux, reverse_aux), (1, 0), "resolved, then pending");
    assert_ne!(plan_subject, reverse_subject);
}

// ---- the nulls ------------------------------------------------------------

/// Settled before the epoch: the poll sees the value under every order, and no
/// lane resolved anything, so nothing was decided.
#[test]
fn the_orders_agree_when_the_value_was_published_before_the_epoch() {
    let plan = workload(LaneOrder::Plan, Setup::SettledFirst);
    let reverse = workload(LaneOrder::Reverse, Setup::SettledFirst);
    assert_eq!(plan.events(EventKind::FutureStateObserved).len(), 1);
    assert!(disagreements(&plan, &reverse).is_empty());
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
}

/// Nobody resolves: the poll sees nothing under every order.
#[test]
fn the_orders_agree_when_nobody_resolves() {
    let plan = workload(LaneOrder::Plan, Setup::NoResolver);
    let reverse = workload(LaneOrder::Reverse, Setup::NoResolver);
    assert_eq!(plan.events(EventKind::FutureStateObserved)[0].auxiliary, 0);
    assert!(disagreements(&plan, &reverse).is_empty());
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
}

/// The clause is about one future: a resolver publishing into its own decides
/// nothing for a poll of another.
#[test]
fn the_orders_agree_when_the_lanes_touch_different_futures() {
    let plan = workload(LaneOrder::Plan, Setup::SeparateFutures);
    let reverse = workload(LaneOrder::Reverse, Setup::SeparateFutures);
    assert_eq!(plan.events(EventKind::FutureResolved).len(), 1);
    assert!(disagreements(&plan, &reverse).is_empty());
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
}

// ---- the read is governed -------------------------------------------------

/// Without `AWAIT` the poll is refused, which it never used to be: reading a
/// future's state took no capability at all. A step that is denied faults
/// rather than reading the denial as "not resolved yet" — those are different
/// answers, and the ungoverned read collapsed them.
#[test]
fn a_poll_without_the_right_is_denied() {
    let run = workload(LaneOrder::Plan, Setup::Unauthorized);
    assert!(
        run.events(EventKind::FutureStateObserved).is_empty(),
        "a denied read observes nothing"
    );
    assert_eq!(run.events(EventKind::AuthorityDenied).len(), 1);
    assert_eq!(run.events(EventKind::ProcessFailed).len(), 1);
}

/// And an authorized one is a governed effect like every other read: the
/// decision is traced, and the effect is adjacent to it (I10c).
#[test]
fn an_authorized_poll_is_a_governed_effect() {
    let run = workload(LaneOrder::Plan, Setup::Contested);
    let mut rows: Vec<_> = run
        .kernel
        .trace_events()
        .iter()
        .filter(|e| e.epoch == 0)
        .collect();
    rows.sort_by_key(|e| (e.lane, e.lane_sequence));
    let observed = rows
        .iter()
        .position(|e| e.event_kind == EventKind::FutureStateObserved)
        .expect("the poll observed something");
    assert_eq!(rows[observed - 1].event_kind, EventKind::AuthorityEffect);
    assert_eq!(rows[observed - 2].event_kind, EventKind::AuthorityGranted);
    assert_eq!(rows[observed].causal.kind, soma::abi::Kind::Future);
}
