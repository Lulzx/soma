//! I25, fifth decision: an await that reads a future another lane is resolving
//! (`docs/SOMA-v0.3.md` §4.15).
//!
//! §4.14 walked the fifteen operations `LaneView` offers and found two that
//! read a decision no workload could reach — `await_future` returning
//! `AlreadySettled`, and `future_value` — and was careful to say why: the only
//! awaiting handler creates the future in the step it awaits, so no resolver
//! can have run yet. That is a fact about the handler set, and this is the
//! handler that changes it. `JOIN_AWAIT` awaits a future named by its frame,
//! which somebody else made and somebody else resolves.
//!
//! One resolver and one awaiter, one future, one epoch. Under `Plan` the
//! resolver is lane 1, so the awaiter finds the value published and continues
//! without parking. Under `Reverse` the awaiter goes first, parks, and is woken
//! by the resolver. Both runs leave the same state — the awaiter runs
//! `JOIN_RESUME` in epoch 1 and reads the same value either way — so what
//! differs is only how the epoch got there, which is exactly what I18 is about.
//!
//! **The two clauses split this one between them**, and neither covers it
//! alone:
//!
//! * Under `Reverse` the awaiter parks and the resolver's wake names the parked
//!   continuation, so ≺ has an edge from one lane to another and clause 1
//!   reports it. This is the first of the five races where clause 1 is not
//!   blind, and the reason is that a wake is the only one of these events that
//!   names *another lane's continuation* rather than only a resource.
//! * Under `Plan` nothing is woken, so there is no edge and clause 1 sees
//!   nothing. What there is instead is `FutureAwaitSettled`, and clause 2 reads
//!   it.
//!
//! So the reference order is the one clause 1 cannot see, which is worth
//! stating plainly: a single plan-order run of this workload passing clause 1
//! is not evidence that its lanes were independent. §4.6's discipline — run it
//! again in another order and compare — is what makes the pair of clauses
//! cover both.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ObjectKind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, JOIN_AWAIT, JOIN_RESUME, SEARCH_HEURISTIC};
use soma::compiler::state_machine_lowering::{HeuristicFrame, JoinFrame};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order, SemanticOrder};

/// What the workload does with the future before the epoch runs.
#[derive(Clone, Copy, PartialEq)]
enum Setup {
    /// A resolver lane and an awaiter lane, contesting one future in one epoch.
    Contested,
    /// The host publishes the value before the epoch, so the awaiter finds it
    /// settled under every order and no lane resolved anything.
    SettledFirst,
    /// The awaiter's future is nobody's this epoch: the resolver has one of its
    /// own, and the one being awaited is never resolved at all.
    SeparateFutures,
    /// No resolver at all, so the awaiter parks and stays parked.
    NoResolver,
}

/// Build the epoch. The resolver is a `SEARCH_HEURISTIC` continuation — the
/// existing handler that publishes into a future named by its frame — and the
/// awaiter is a `JOIN_AWAIT`, which is the handler this section adds.
fn workload(order: LaneOrder, setup: Setup) -> Kernel {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let awaited = kernel.create_future(owner);

    if setup != Setup::NoResolver {
        // The future the resolver publishes into: the contested one, unless the
        // null wants the two lanes touching different futures.
        let target = if setup == Setup::SeparateFutures {
            kernel.create_future(owner)
        } else {
            awaited
        };
        let resolver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        // A future reference carries no authority; RESOLVE is delegated by the
        // process that minted it.
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

    let joiner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .grant_capability(owner, joiner, awaited, Rights::AWAIT, 0, 0)
        .expect("the owner holds the genesis capability");
    let mut bytes = Vec::new();
    JoinFrame {
        future: awaited,
        observed: Ref64::NULL,
    }
    .encode(&mut bytes);
    let joining = kernel
        .create_continuation(
            SYSTEM_PRINCIPAL,
            joiner,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                JOIN_AWAIT,
                JOIN_AWAIT,
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
            .resolve_future(owner, awaited, value)
            .expect("the future is pending before the epoch");
    }

    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(30);
    let _ = joining;
    kernel
}

fn violations(kernel: &Kernel) -> Vec<String> {
    check(kernel)
        .into_iter()
        .filter(|v| v.invariant == Invariant::LaneIndependence)
        .map(|v| v.detail)
        .collect()
}

fn events_of(kernel: &Kernel, kind: EventKind) -> Vec<&soma::abi::TraceEvent> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == kind)
        .collect()
}

fn disagreements(a: &Kernel, b: &Kernel) -> Vec<soma::semantics::order::OrderViolation> {
    conforms_traces(
        &in_position_order(&a.trace_snapshot()),
        &in_position_order(&b.trace_snapshot()),
    )
}

// ---- the counterexample ---------------------------------------------------

#[test]
fn two_lane_orders_disagree_over_whether_an_await_parks() {
    let plan = workload(LaneOrder::Plan, Setup::Contested);
    let reverse = workload(LaneOrder::Reverse, Setup::Contested);

    // The two branches of the await, one per order.
    assert_eq!(events_of(&plan, EventKind::FutureAwaitSettled).len(), 1);
    assert!(events_of(&plan, EventKind::ContinuationWaiting).is_empty());
    assert!(events_of(&reverse, EventKind::FutureAwaitSettled).is_empty());
    assert_eq!(events_of(&reverse, EventKind::ContinuationWaiting).len(), 1);

    assert!(
        !disagreements(&plan, &reverse).is_empty(),
        "an await raced against its resolver must make the epoch depend on the lane order"
    );
}

/// And the state they leave is the same, which is what makes this a schedule
/// dependence rather than a bug: both runs finish the awaiter in the same epoch
/// at the same resume point, having read the same published value. Only the
/// route differs, and only the trace records the route.
#[test]
fn both_routes_leave_the_same_state() {
    let plan = workload(LaneOrder::Plan, Setup::Contested);
    let reverse = workload(LaneOrder::Reverse, Setup::Contested);
    for kernel in [&plan, &reverse] {
        let completed: Vec<_> = events_of(kernel, EventKind::ContinuationCompleted)
            .into_iter()
            .filter(|e| e.run_class == 0)
            .collect();
        assert_eq!(completed.len(), 2, "the resolver and the awaiter both finish");
        let resumed = events_of(kernel, EventKind::ContinuationStarted)
            .into_iter()
            .filter(|e| e.run_class == JOIN_RESUME)
            .count();
        assert_eq!(resumed, 1, "the awaiter reaches JOIN_RESUME by either route");
    }
    assert_eq!(
        events_of(&plan, EventKind::FutureResolved)[0].subject,
        events_of(&reverse, EventKind::FutureResolved)[0].subject,
        "and the value published is the same one"
    );
}

// ---- which clause sees which order ----------------------------------------

/// Clause 1 is not blind here — the first of the five races where it is not —
/// because a wake names the woken lane's continuation, so ≺ really does have an
/// edge from one lane of the epoch to another. It sees only the order in which
/// the awaiter parked, because in the other one nothing was woken.
#[test]
fn clause_1_sees_the_order_in_which_the_awaiter_parked() {
    let reverse = workload(LaneOrder::Reverse, Setup::Contested);
    let edges = SemanticOrder::of(&reverse).cross_lane_edges();
    assert_eq!(edges.len(), 1, "{edges:?}");

    let plan = workload(LaneOrder::Plan, Setup::Contested);
    assert!(
        SemanticOrder::of(&plan).cross_lane_edges().is_empty(),
        "and nothing at all in the order where the resolver went first"
    );
}

/// Clause 2 sees the other one, and needs `FutureAwaitSettled` to do it: the
/// awaiter that did not park is otherwise indistinguishable from a continuation
/// that yielded for its own reasons.
#[test]
fn clause_2_sees_the_order_in_which_the_resolver_went_first() {
    let plan = workload(LaneOrder::Plan, Setup::Contested);
    let reported = violations(&plan);
    assert_eq!(reported.len(), 1, "{reported:?}");
    assert!(
        reported[0].contains("read whether it had been resolved"),
        "{}",
        reported[0]
    );
}

/// Between them, every order reports. That is the property the workload owes:
/// I25 is checked per run, so a race no run reports is a race the checker
/// cannot be said to catch.
#[test]
fn i25_reports_the_run_whichever_order_it_ran_in() {
    for order in [
        LaneOrder::Plan,
        LaneOrder::Reverse,
        LaneOrder::Permuted(11),
        LaneOrder::Permuted(12),
    ] {
        let reported = violations(&workload(order, Setup::Contested));
        assert_eq!(reported.len(), 1, "{order:?}: {reported:?}");
    }
}

// ---- the nulls ------------------------------------------------------------

/// A future settled before the epoch began. The awaiter finds it settled under
/// every order, and no lane resolved anything — so nobody won it, and a
/// resource nobody got out of decided nothing. This is the null every resource
/// in the clause has, and here it is also the run that proves
/// `FutureAwaitSettled` alone is not the report: the event is emitted and
/// nothing is reported.
#[test]
fn the_orders_agree_when_the_future_was_settled_before_the_epoch() {
    let plan = workload(LaneOrder::Plan, Setup::SettledFirst);
    let reverse = workload(LaneOrder::Reverse, Setup::SettledFirst);
    assert_eq!(events_of(&plan, EventKind::FutureAwaitSettled).len(), 1);
    assert!(disagreements(&plan, &reverse).is_empty());
    assert!(violations(&plan).is_empty(), "{:?}", violations(&plan));
    assert!(violations(&reverse).is_empty(), "{:?}", violations(&reverse));
}

/// The null that says the clause is about *one* future: the resolver publishes
/// into one of its own and the awaited future is untouched, so the awaiter
/// parks under both orders and nothing was decided.
#[test]
fn the_orders_agree_when_the_lanes_touch_different_futures() {
    let plan = workload(LaneOrder::Plan, Setup::SeparateFutures);
    let reverse = workload(LaneOrder::Reverse, Setup::SeparateFutures);
    assert_eq!(events_of(&plan, EventKind::FutureResolved).len(), 1);
    assert_eq!(events_of(&plan, EventKind::ContinuationWaiting).len(), 1);
    assert!(events_of(&plan, EventKind::FutureAwaitSettled).is_empty());
    assert!(disagreements(&plan, &reverse).is_empty());
    assert!(violations(&plan).is_empty(), "{:?}", violations(&plan));
}

/// The null that says it is about one *epoch*: with no resolver, the awaiter
/// parks and stays parked, and a wake that never happens orders nothing.
#[test]
fn the_orders_agree_when_nobody_resolves() {
    let plan = workload(LaneOrder::Plan, Setup::NoResolver);
    let reverse = workload(LaneOrder::Reverse, Setup::NoResolver);
    assert_eq!(events_of(&plan, EventKind::ContinuationWaiting).len(), 1);
    assert!(disagreements(&plan, &reverse).is_empty());
    assert!(violations(&plan).is_empty(), "{:?}", violations(&plan));
}

// ---- the trace ------------------------------------------------------------

#[test]
fn an_await_that_found_the_value_published_is_in_the_trace() {
    // Until this event existed, the branch left no record: `ContinuationWaiting`
    // marks the await that parked, and the await that did not park emitted the
    // same `ContinuationYielded` as any other yield.
    let kernel = workload(LaneOrder::Plan, Setup::Contested);
    let events = events_of(&kernel, EventKind::FutureAwaitSettled);
    assert_eq!(events.len(), 1);
    let event = events[0];
    assert_eq!(
        event.causal.kind,
        soma::abi::Kind::Future,
        "a settled await names its future, as a resolution does"
    );
    assert_eq!(
        event.subject,
        events_of(&kernel, EventKind::FutureResolved)[0].subject,
        "and the value it found published"
    );
    assert_eq!(event.run_class, JOIN_RESUME, "with where the await continues");
    assert_ne!(event.lane, soma::abi::traces::HOST_LANE);
}

/// The `AlreadySettled` arm is reached by a handler, which is the whole point:
/// §4.14's blind spot was a property of the handler set and this is the handler
/// that moves it. `future_value` goes with it — `JOIN_RESUME` reads a
/// resolution it did not produce, and that read has no event of its own.
#[test]
fn the_awaiting_handler_reads_a_future_it_did_not_create() {
    let kernel = workload(LaneOrder::Plan, Setup::Contested);
    let started: Vec<_> = events_of(&kernel, EventKind::ContinuationStarted)
        .into_iter()
        .filter(|e| e.run_class == JOIN_AWAIT || e.run_class == JOIN_RESUME)
        .collect();
    assert_eq!(started.len(), 2, "both halves of the join ran");
    // Nothing in the trace says `future_value` was called, which is the reason
    // §4.14 could only reach it by reading the handlers.
    assert!(events_of(&kernel, EventKind::FutureAwaitSettled).len() == 1);
}
