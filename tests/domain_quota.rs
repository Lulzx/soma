//! I25 clause 2: an epoch's lanes must not be decided by one domain's quota.
//!
//! §4.6 reordered an epoch's lanes and reported that it found nothing. That was
//! true of the workloads it ran, and all of them allocate in the root domain,
//! which is unbounded. A bounded domain is the case it did not have: a step
//! creating a process consumes its domain's quota, so two lanes of one epoch
//! creating processes in one bounded domain race for it, and the loser faults.
//!
//! `two_lane_orders_disagree_when_the_quota_binds` is the test that carries the
//! weight — it is the counterexample, and it was found by looking for one rather
//! than by a failure. Everything else either shows the clause is not vacuous or
//! shows what it deliberately does not report.
//!
//! The dependence carries no ≺ edge: nothing is sent, resolved or woken between
//! the lanes. `clause_1_does_not_see_it` is the reason clause 2 exists at all.
//!
//! A quota is one bounded resource and not the only one. `tests/mailbox_capacity.rs`
//! is the same experiment on a receiver's mailbox, and it is where the clause's
//! condition got its final form: a winner *and* a different loser.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ProcessMode, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::DEFAULT_MAX_STEPS;
use soma::compiler::state_machine_lowering::SearchFrame;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::lane_order::LaneOrder;
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{conforms_traces, in_position_order, SemanticOrder};

/// `parents` processes in one domain, each spawning two children in its first
/// step. The domain holds `parents` already, so a quota of `parents + 2 *
/// parents` is exactly enough and anything less binds.
fn bounded(order: LaneOrder, quota: u32, parents: u64) -> Kernel {
    let mut kernel = Kernel::new();
    let root = kernel.root_domain();
    let domain = kernel
        .create_domain(SYSTEM_PRINCIPAL, root, quota)
        .expect("system may create a domain");

    for value in 0..parents {
        let process = kernel
            .create_process_in_domain(SYSTEM_PRINCIPAL, domain, ProcessMode::Serial)
            .expect("the quota holds the parents");
        let frame = SearchFrame {
            value,
            depth: 1,
            branching: 2,
            work_iters: 0,
            class_count: 1,
        };
        let run_class = frame.run_class();
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    run_class,
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create the initial continuation");
    }
    kernel.configure_cohorts(4, PartialCohortPolicy::RunPartial);
    kernel.configure_lane_order(order);
    kernel.run_to_quiescence(20);
    kernel
}

/// Four parents wanting eight children, in a domain with room for six of them.
fn binding(order: LaneOrder) -> Kernel {
    bounded(order, 10, 4)
}

/// The same workload with room for every child.
fn slack(order: LaneOrder) -> Kernel {
    bounded(order, 12, 4)
}

fn violations(kernel: &Kernel) -> Vec<String> {
    check(kernel)
        .into_iter()
        .filter(|v| v.invariant == Invariant::LaneIndependence)
        .map(|v| v.detail)
        .collect()
}

fn kinds(kernel: &Kernel, kind: EventKind) -> Vec<&soma::abi::TraceEvent> {
    kernel
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == kind)
        .collect()
}

// ---- the counterexample ---------------------------------------------------

#[test]
fn two_lane_orders_disagree_when_the_quota_binds() {
    let plan = binding(LaneOrder::Plan);
    let reverse = binding(LaneOrder::Reverse);

    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(
        !disagreements.is_empty(),
        "a bounded domain must make the epoch's outcome depend on the lane order; \
         if this passes the workload stopped exercising the quota"
    );

    // And concretely: a different set of processes is refused.
    let refused = |k: &Kernel| {
        kinds(k, EventKind::ProcessCreationRefused)
            .iter()
            .map(|e| e.process.slot)
            .collect::<Vec<_>>()
    };
    assert_ne!(
        refused(&plan),
        refused(&reverse),
        "the same processes were refused under both orders, so nothing was raced for"
    );
}

/// The null for the counterexample. Same workload, same lanes, same domain —
/// only the bound has room, and the orders agree again. Without this the test
/// above could be reporting any difference two orders happen to produce.
#[test]
fn the_orders_agree_when_the_quota_has_room() {
    let plan = slack(LaneOrder::Plan);
    let reverse = slack(LaneOrder::Reverse);
    let disagreements = conforms_traces(
        &in_position_order(&plan.trace_snapshot()),
        &in_position_order(&reverse.trace_snapshot()),
    );
    assert!(
        disagreements.is_empty(),
        "a bound with room decides nothing, so the orders must agree: {disagreements:?}"
    );
    assert!(kinds(&plan, EventKind::ProcessCreationRefused).is_empty());
}

/// Why clause 2 is not clause 1 restated. The lanes depend on each other and
/// the semantic order has no edge to show for it — nothing was sent, resolved
/// or woken, so the dependence is carried by a counter.
#[test]
fn clause_1_does_not_see_it() {
    let kernel = binding(LaneOrder::Plan);
    let edges = SemanticOrder::of(&kernel).cross_lane_edges();
    assert!(
        edges.is_empty(),
        "this run has a cross-lane ≺ edge, so it is not the case clause 2 was written for: \
         {edges:?}"
    );
    assert!(
        !violations(&kernel).is_empty(),
        "and yet I25 must report it"
    );
}

// ---- the clause -----------------------------------------------------------

#[test]
fn i25_reports_a_contended_quota() {
    for order in [LaneOrder::Plan, LaneOrder::Reverse] {
        let reported = violations(&binding(order));
        assert_eq!(reported.len(), 1, "one report per run: {reported:?}");
        assert!(reported[0].contains("bounded domain"), "{}", reported[0]);
    }
}

#[test]
fn i25_is_silent_when_the_bound_has_room() {
    // Two lanes draw on the domain and neither is refused. The increment is
    // shared; the decision is not, and clause 2 is about the decision.
    let kernel = slack(LaneOrder::Plan);
    assert!(kinds(&kernel, EventKind::ProcessCreated).len() > 4);
    assert!(violations(&kernel).is_empty());
}

#[test]
fn i25_is_silent_when_one_lane_is_refused() {
    // One parent, so one lane draws on the domain, takes what is left and is
    // then refused. A lane that exhausts a resource against nobody has raced
    // nobody, whatever order it ran in.
    let kernel = bounded(LaneOrder::Plan, 2, 1);
    assert!(
        !kinds(&kernel, EventKind::ProcessCreationRefused).is_empty(),
        "the bound must actually bind for this to be the null it claims to be"
    );
    assert!(violations(&kernel).is_empty());
}

#[test]
fn an_unbounded_domain_is_never_contended() {
    let kernel = bounded(LaneOrder::Plan, 0, 4);
    assert!(kinds(&kernel, EventKind::ProcessCreationRefused).is_empty());
    assert!(violations(&kernel).is_empty());
}

// ---- the fault ------------------------------------------------------------

#[test]
fn a_refused_step_faults_rather_than_aborting_the_machine() {
    // This used to abort: `create_process` on the lane surface was infallible,
    // so `DomainQuotaExceeded` reached an `expect` inside a handler. Reaching
    // this assertion at all is half the test.
    let kernel = binding(LaneOrder::Plan);
    let failed = kinds(&kernel, EventKind::ProcessFailed);
    assert!(
        !failed.is_empty(),
        "a step that cannot allocate must fault, not vanish"
    );
    // The failure is contained: every other invariant holds, and the run
    // reaches quiescence.
    let other: Vec<_> = check(&kernel)
        .into_iter()
        .filter(|v| v.invariant != Invariant::LaneIndependence)
        .collect();
    assert!(other.is_empty(), "{other:?}");
}

#[test]
fn the_refusal_names_the_domain_that_refused() {
    let kernel = binding(LaneOrder::Plan);
    let refusals = kinds(&kernel, EventKind::ProcessCreationRefused);
    assert!(!refusals.is_empty());
    for event in refusals {
        assert_eq!(
            event.subject.kind,
            soma::abi::Kind::Domain,
            "a refusal must name the domain as an entity, not as a bare number"
        );
        assert_ne!(
            event.lane,
            soma::abi::traces::HOST_LANE,
            "these refusals happen inside a step"
        );
    }
}

#[test]
fn a_created_process_records_the_domain_it_drew_on() {
    // The counterpart to the refusal, and what makes clause 2 checkable after
    // the process itself has been reclaimed.
    let kernel = binding(LaneOrder::Plan);
    for event in kinds(&kernel, EventKind::ProcessCreated) {
        assert_eq!(
            event.subject.kind,
            soma::abi::Kind::Domain,
            "process {} was created without recording its domain",
            event.process.slot
        );
        // Stronger where it can be: a live process still knows its domain, and
        // the event has to agree with it.
        if let Ok(domain) = kernel.process_domain(event.process) {
            assert_eq!(domain, event.subject);
        }
    }
}
