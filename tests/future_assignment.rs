//! I25 clause 2, fourth resource: a future's one assignment
//! (`docs/SOMA-v0.3.md` §4.14).
//!
//! §4.13 corrected the method §4.12 got wrong: enumerate the *operations* that
//! can say no, not the resources. `LaneView` offers fifteen and the list is
//! closed by the compiler, so the enumeration is finite. `resolve_future` is on
//! it, single assignment (§12) is exactly an operation that can say no, and this
//! is the run where two lanes of one epoch say it to each other.
//!
//! Four processes hold `RESOLVE` on one future and each tries to publish a
//! different value in one epoch. The value the future takes is the one whose
//! lane ran first — lane 1 under `Plan`, lane 4 under `Reverse` — and the other
//! three fault. Both runs leave a legal state; only comparing them reports it.
//! Clause 1 is blind for the fourth time: `FutureResolved` names no other lane,
//! the losers emit no delivery, and `cross_lane_edges()` is empty.
//!
//! This is the resource that does not discriminate between clause 2's two
//! conditions. A future has one unit, so there is never a second winner, and
//! "any other lane that drew" and "a lane that was refused" name the same set.
//! It is registered as dispensing interchangeable units because what a resolver
//! wins is a permission to publish and one such permission is like another.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_HEURISTIC};
use soma::compiler::state_machine_lowering::HeuristicFrame;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order, SemanticOrder};

/// `resolvers` processes, each with `RESOLVE` on the same future and a
/// continuation that computes a value from its own input and publishes it. They
/// run as lanes of one epoch; one of them succeeds.
///
/// `settled_first` resolves the future from the host before the epoch, so every
/// lane is refused.
fn contested(order: LaneOrder, resolvers: u64, settled_first: bool) -> Kernel {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let future = kernel.create_future(owner);

    for input in 0..resolvers {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        // A future reference carries no authority; RESOLVE is delegated by the
        // process that minted it.
        kernel
            .grant_capability(owner, process, future, Rights::RESOLVE, 0, 0)
            .expect("the owner holds the genesis capability");
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
                    SEARCH_HEURISTIC,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create the initial continuation");
    }
    if settled_first {
        let value = kernel.create_object(
            SYSTEM_PRINCIPAL,
            soma::abi::ObjectKind::FutureValue,
            0u64.to_le_bytes().to_vec(),
        );
        kernel
            .resolve_future(owner, future, value)
            .expect("the future is pending before the epoch");
    }
    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(30);
    kernel
}

/// Four resolvers, one pending future.
fn one_future(order: LaneOrder) -> Kernel {
    contested(order, 4, false)
}

/// Four resolvers, each with a future of its own.
fn separate(order: LaneOrder) -> Kernel {
    let mut kernel = Kernel::new();
    for input in 0..4u64 {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let future = kernel.create_future(process);
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
                    SEARCH_HEURISTIC,
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

fn violations(kernel: &Kernel) -> Vec<String> {
    check(kernel)
        .into_iter()
        .filter(|v| v.invariant == Invariant::LaneIndependence)
        .map(|v| v.detail)
        .collect()
}

/// The lane of every resolve that took. A lane number is a position in the
/// epoch's plan and does not move when the walk order does (§4.6).
fn lanes_that_resolved(kernel: &Kernel) -> Vec<u32> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::FutureResolved)
        // A host resolve is the plan's, not a lane's, which is the same
        // exclusion the clause makes.
        .filter(|e| e.lane != soma::abi::traces::HOST_LANE)
        .map(|e| e.lane)
        .collect()
}

fn refused(kernel: &Kernel) -> Vec<&soma::abi::TraceEvent> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::FutureResolutionRefused)
        .collect()
}

// ---- the counterexample ---------------------------------------------------

#[test]
fn two_lane_orders_disagree_over_which_value_a_future_takes() {
    let plan = one_future(LaneOrder::Plan);
    let reverse = one_future(LaneOrder::Reverse);

    assert_eq!(lanes_that_resolved(&plan).len(), 1, "single assignment");
    assert_ne!(
        lanes_that_resolved(&plan),
        lanes_that_resolved(&reverse),
        "the same lane published under both orders, so nothing was raced for"
    );
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(
        !disagreements.is_empty(),
        "a contested future must make the epoch's outcome depend on the lane order"
    );
}

/// The null: a future each, and the orders agree.
#[test]
fn the_orders_agree_when_each_lane_has_its_own_future() {
    let plan = separate(LaneOrder::Plan);
    let reverse = separate(LaneOrder::Reverse);
    assert_eq!(
        lanes_that_resolved(&plan).len(),
        4,
        "the null needs every lane to have really resolved"
    );
    assert!(refused(&plan).is_empty());
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(disagreements.is_empty(), "{disagreements:?}");
    assert!(violations(&plan).is_empty(), "{:?}", violations(&plan));
}

/// The second null, the one every resource in this clause has. A future settled
/// before the epoch began refuses all four lanes under every order, so nobody
/// won and no order decided anything.
#[test]
fn the_orders_agree_when_the_future_was_already_settled() {
    let plan = contested(LaneOrder::Plan, 4, true);
    let reverse = contested(LaneOrder::Reverse, 4, true);
    assert!(lanes_that_resolved(&plan).is_empty(), "the host resolved it");
    assert_eq!(refused(&plan).len(), 4, "every lane must be refused");
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

/// Clause 1 is blind here for the fourth time, and for a variant of one reason:
/// the dependence is carried by whether a cell has been written, and a write
/// that did not happen is not an event anyone can draw an edge from.
#[test]
fn clause_1_does_not_see_it() {
    let kernel = one_future(LaneOrder::Plan);
    let edges = SemanticOrder::of(&kernel).cross_lane_edges();
    assert!(edges.is_empty(), "{edges:?}");
    assert!(
        !violations(&kernel).is_empty(),
        "and yet I25 must report it"
    );
}

// ---- the clause -----------------------------------------------------------

#[test]
fn i25_reports_a_contested_future() {
    for order in [LaneOrder::Plan, LaneOrder::Reverse] {
        let reported = violations(&one_future(order));
        assert_eq!(reported.len(), 1, "one report per run: {reported:?}");
        assert!(reported[0].contains("one future"), "{}", reported[0]);
    }
}

// ---- the trace ------------------------------------------------------------

#[test]
fn a_refused_resolve_is_in_the_trace() {
    // Single assignment used to enforce itself silently: the loser's step
    // faulted, and `ProcessFailed` does not say whether a program went wrong or
    // was told the value had already been published.
    let kernel = one_future(LaneOrder::Plan);
    let events = refused(&kernel);
    assert_eq!(events.len(), 3, "three of the four lanes lost");
    for event in events {
        assert_eq!(
            event.causal.kind,
            soma::abi::Kind::Future,
            "a refused resolve names its future, as a successful one does"
        );
        assert_ne!(
            event.subject,
            Ref64::NULL,
            "and the value it built and did not publish"
        );
        assert_ne!(event.lane, soma::abi::traces::HOST_LANE);
    }
}

#[test]
fn a_refused_resolver_faults_and_the_state_stays_legal() {
    // The losers fault rather than proceeding as though they had published,
    // which is what makes the refusal a decision rather than a dropped write.
    let kernel = one_future(LaneOrder::Plan);
    let failures = kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::ProcessFailed)
        .count();
    assert_eq!(failures, 3);
    let other: Vec<_> = check(&kernel)
        .into_iter()
        .filter(|v| v.invariant != Invariant::LaneIndependence)
        .collect();
    assert!(other.is_empty(), "{other:?}");
}
