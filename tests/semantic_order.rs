//! I18 (schedule conformance) and I19 (placement neutrality).
//!
//! These two clauses weaken `docs/SOMA-v0.2.md` §1.2 from trace *equality* to
//! trace *refinement*, so that a concurrent implementation can conform at all.
//! A weakened equivalence is only worth having if it still rejects things, so
//! every acceptance test here is paired with a rejection test. The order of
//! obligations matters:
//!
//! 1. the reference interpreter satisfies its own derived order (if it does
//!    not, the derivation is wrong, and nothing downstream means anything);
//! 2. reorderings the model permits are accepted;
//! 3. reorderings the model forbids are rejected.

use soma::abi::{EventKind, ObjectKind, ProcessMode, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::order::{
    conforms, conforms_traces, is_placement_event, placement_neutral, semantic_projection,
    EdgeReason, SemanticOrder,
};

fn search_workload() -> Kernel {
    let knobs = ControlKnobs {
        branching_factor: 3,
        depth: 3,
        process_count: 2,
        class_count: 3,
        ..ControlKnobs::default()
    };
    let mut kernel = build(&knobs);
    kernel.run_to_quiescence(200);
    kernel
}

fn messaging_workload() -> Kernel {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 7);
    kernel.run_to_quiescence(200);
    kernel
}

/// A run containing a genuine mailbox send *and* its matching receive.
///
/// The `Expand` workload is ingested from outside and replies to a process
/// with no receiver, so it produces an unpaired receive and an unpaired send —
/// no delivery edge at all. Pairing needs both halves inside one run.
fn delivery_workload() -> Kernel {
    let mut kernel = Kernel::new();
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    let sender_cont = kernel
        .create_continuation(
            sender,
            sender,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_BRANCH,
                0,
                bytes.clone(),
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();
    let receiver_cont = kernel
        .create_continuation(
            receiver,
            receiver,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_BRANCH,
                0,
                bytes,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();

    kernel
        .grant_capability(SYSTEM_PRINCIPAL, sender, receiver, Rights::SEND, 0, 0)
        .unwrap();

    for value in 0..3u8 {
        let payload = kernel.create_object(sender, ObjectKind::MessagePayload, vec![value]);
        kernel
            .enqueue_message(sender, receiver, payload, sender_cont)
            .unwrap();
        kernel.receive_message(receiver, receiver_cont).unwrap();
    }
    kernel
}

// ---- obligation 1: the reference satisfies its own order ------------------

#[test]
fn the_reference_run_is_a_linear_extension_of_its_own_order() {
    for kernel in [search_workload(), messaging_workload()] {
        let order = SemanticOrder::of(&kernel);
        assert!(
            order.is_self_consistent(),
            "the sequential interpreter violated an order edge it generated itself"
        );
    }
}

#[test]
fn the_derived_order_is_not_empty() {
    // A relation with no edges accepts every permutation, which would make
    // I18 decorative. This is the guard against that.
    let order = SemanticOrder::of(&messaging_workload());
    assert!(
        order.edges().len() > 10,
        "expected a populated order, got {} edges",
        order.edges().len()
    );

    let reasons: Vec<EdgeReason> = order.edges().iter().map(|e| e.reason).collect();
    assert!(reasons.contains(&EdgeReason::ContinuationProgram));
    assert!(reasons.contains(&EdgeReason::FutureResolution));

    let delivery = SemanticOrder::of(&delivery_workload());
    assert!(delivery
        .edges()
        .iter()
        .any(|edge| edge.reason == EdgeReason::MessageDelivery));
}

#[test]
fn a_run_conforms_to_itself() {
    let kernel = search_workload();
    assert!(conforms(&kernel, &kernel).is_empty());
}

#[test]
fn two_identical_runs_conform() {
    let a = search_workload();
    let b = search_workload();
    assert!(
        conforms(&a, &b).is_empty(),
        "{:?}",
        conforms(&a, &b).first().map(|v| v.to_string())
    );
}

// ---- obligation 2: permitted reorderings are accepted ---------------------

#[test]
fn independent_events_may_be_reordered() {
    // Two events with no edge between them may appear in either order. This is
    // the reordering a cohort executing its lanes simultaneously produces, and
    // the whole reason trace equality had to go.
    //
    // Rather than guess which pair is independent, swap every adjacent
    // same-epoch pair and require that at least one swap is accepted. A
    // relation that rejected all of them would be trace equality wearing a
    // different name.
    let kernel = search_workload();
    let projected = semantic_projection(&kernel.trace_snapshot());

    let accepted = (0..projected.len().saturating_sub(1))
        .filter(|index| projected[*index].epoch == projected[index + 1].epoch)
        .filter(|index| {
            let mut swapped = projected.clone();
            swapped.swap(*index, index + 1);
            conforms_traces(&projected, &swapped).is_empty()
        })
        .count();

    assert!(
        accepted > 0,
        "every adjacent reordering was rejected, so the relation is still trace equality"
    );
}

// ---- obligation 3: forbidden reorderings are rejected --------------------

#[test]
fn reordering_one_continuations_own_events_is_rejected() {
    // A continuation is sequential by definition (§1.1). No implementation,
    // however parallel, may run its resume points out of order.
    let kernel = search_workload();
    let projected = semantic_projection(&kernel.trace_snapshot());

    let mut broken = projected.clone();
    let pair = same_continuation_pair(&projected).expect("workload should have a continuation with two events");
    broken.swap(pair.0, pair.1);

    let violations = conforms_traces(&projected, &broken);
    assert!(
        !violations.is_empty(),
        "swapping two events of one continuation was accepted"
    );
}

/// Two events of one continuation *within a single epoch*.
///
/// The same-epoch restriction matters: a pair straddling an epoch boundary
/// would be caught by the epoch clause no matter what program order said, and
/// the test would pass while proving nothing about program order.
fn same_continuation_pair(events: &[soma::kernel::TraceSnapshotRow]) -> Option<(usize, usize)> {
    for i in 0..events.len() {
        for j in (i + 1)..events.len() {
            if events[i].continuation != 0
                && events[i].epoch == events[j].epoch
                && events[i].continuation == events[j].continuation
                && matches!(
                    events[i].event_kind,
                    EventKind::ContinuationStarted
                        | EventKind::ContinuationReady
                        | EventKind::ContinuationYielded
                )
                && matches!(
                    events[j].event_kind,
                    EventKind::ContinuationCompleted
                        | EventKind::ContinuationYielded
                        | EventKind::ContinuationWaiting
                )
            {
                return Some((i, j));
            }
        }
    }
    None
}

#[test]
fn delivering_a_message_before_it_is_sent_is_rejected() {
    let kernel = delivery_workload();
    let projected = semantic_projection(&kernel.trace_snapshot());

    let send = projected
        .iter()
        .position(|row| row.event_kind == EventKind::MessageSent)
        .expect("the expand workload sends a message");
    let receive = projected
        .iter()
        .enumerate()
        .position(|(index, row)| {
            index > send
                && row.event_kind == EventKind::MessageReceived
                && row.causal == projected[send].process
                && row.auxiliary == projected[send].auxiliary
        })
        .expect("that message is received");

    let mut broken = projected.clone();
    broken.swap(send, receive);

    let violations = conforms_traces(&projected, &broken);
    assert!(
        !violations.is_empty(),
        "a receive placed before its send was accepted"
    );
}

#[test]
fn dropping_an_event_is_rejected() {
    // The ordering half of I18 alone would accept an implementation that
    // silently did less work. Clause 1 is what stops that.
    let kernel = search_workload();
    let projected = semantic_projection(&kernel.trace_snapshot());
    let mut truncated = projected.clone();
    truncated.remove(projected.len() / 2);

    assert!(
        !conforms_traces(&projected, &truncated).is_empty(),
        "a run that lost an event was accepted as conforming"
    );
}

#[test]
fn moving_an_event_into_another_epoch_is_rejected() {
    // An epoch boundary is a consistent cut (§3.5). Doing the same work a
    // whole epoch early is a different run, not a reordering.
    let kernel = search_workload();
    let projected = semantic_projection(&kernel.trace_snapshot());

    let mut broken = projected.clone();
    let target = broken
        .iter()
        .position(|row| row.epoch > 0)
        .expect("the workload runs more than one epoch");
    broken[target].epoch -= 1;

    assert!(
        !conforms_traces(&projected, &broken).is_empty(),
        "an event moved across an epoch boundary was accepted"
    );
}

#[test]
fn epochs_may_not_run_backwards() {
    let kernel = search_workload();
    let projected = semantic_projection(&kernel.trace_snapshot());
    let mut broken = projected.clone();
    broken.sort_by(|a, b| b.epoch.cmp(&a.epoch));

    assert!(
        !conforms_traces(&projected, &broken).is_empty(),
        "a trace with descending epochs was accepted"
    );
}

// ---- I19: placement neutrality -------------------------------------------

#[test]
fn cohort_width_does_not_change_observable_behaviour() {
    // Cohort width is the placement knob the model most obviously permits
    // (§4, "Cohorting"). Changing it changes how work is grouped and must not
    // change what the program observes.
    let widths = [1u16, 2, 4, 16];
    let kernels: Vec<Kernel> = widths
        .iter()
        .map(|width| {
            let knobs = ControlKnobs {
                branching_factor: 3,
                depth: 3,
                process_count: 2,
                class_count: 3,
                ..ControlKnobs::default()
            };
            let mut kernel = build(&knobs);
            kernel.configure_cohorts(*width, Default::default());
            kernel.run_to_quiescence(200);
            kernel
        })
        .collect();

    let refs: Vec<&Kernel> = kernels.iter().collect();
    let violations = placement_neutral(&refs);
    assert!(
        violations.is_empty(),
        "cohort width leaked into observable behaviour: {:?}",
        violations.first().map(|v| v.to_string())
    );
}

#[test]
fn cohort_width_really_does_change_the_placement_trace() {
    // The null for the test above. If widths 1 and 16 produced identical raw
    // traces, I19 would be passing because nothing varied, and it would prove
    // nothing about placement at all.
    let mut narrow = build(&ControlKnobs::default());
    narrow.configure_cohorts(1, Default::default());
    narrow.run_to_quiescence(200);

    let mut wide = build(&ControlKnobs::default());
    wide.configure_cohorts(16, Default::default());
    wide.run_to_quiescence(200);

    let narrow_cohorts = narrow
        .trace_snapshot()
        .iter()
        .filter(|row| is_placement_event(row.event_kind))
        .count();
    let wide_cohorts = wide
        .trace_snapshot()
        .iter()
        .filter(|row| is_placement_event(row.event_kind))
        .count();

    assert!(
        narrow_cohorts > wide_cohorts,
        "expected width 1 to produce more cohorts than width 16, got {narrow_cohorts} and {wide_cohorts}"
    );
    assert_ne!(
        narrow.trace_snapshot(),
        wide.trace_snapshot(),
        "the two placements produced byte-identical traces, so I19 was vacuous"
    );
}

#[test]
fn placement_neutrality_catches_a_placement_that_changes_behaviour() {
    // Fault injection for I19: two runs of genuinely different programs must
    // be reported as diverging, or the checker would accept any placement.
    let mut one = Kernel::new();
    create_expand(&mut one, 7);
    one.run_to_quiescence(200);

    let mut other = Kernel::new();
    create_expand(&mut other, 9);
    let process = other.create_process(soma::kernel::SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    other
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_BRANCH,
                0,
                bytes,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();
    other.run_to_quiescence(200);

    assert!(
        !placement_neutral(&[&one, &other]).is_empty(),
        "two different programs were reported as placement-equivalent"
    );
}
