//! I25 (lane independence), the clause canonical commit is paid for with.
//!
//! `docs/SOMA-v0.3.md` §4.3 (3) measured that no ≺ edge joins two lanes of one
//! epoch and declined to call it an invariant, on the honest grounds that
//! nothing at the time depended on it. `kernel/epochs.rs` now applies an
//! epoch's effects once, after every lane, so something does: a run with a
//! cross-lane edge is a run whose lanes observed each other in an order the
//! commit no longer reproduces.
//!
//! The obligations here are the pair the repository asks of every invariant.
//! The reference satisfies it, on workloads that genuinely exercise the
//! relations it is about — an empty order would satisfy it vacuously, so each
//! acceptance carries a null showing the order is not empty. And a state that
//! violates it is reported, built by moving a real delivery across lanes rather
//! than by fabricating a trace, so the rejected thing is one the derivation
//! would really have produced.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{EventKind, ObjectKind, ProcessMode, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::{ContinuationSpec, Kernel, TraceSnapshotRow, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order::{EdgeReason, SemanticOrder};

fn search_workload(width: u16) -> Kernel {
    let knobs = ControlKnobs {
        branching_factor: 3,
        depth: 3,
        process_count: 2,
        class_count: 3,
        ..ControlKnobs::default()
    };
    let mut kernel = build(&knobs);
    kernel.configure_cohorts(width, PartialCohortPolicy::RunPartial);
    kernel.run_to_quiescence(200);
    kernel
}

fn expand_workload(width: u16) -> Kernel {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 7);
    kernel.configure_cohorts(width, PartialCohortPolicy::RunPartial);
    kernel.run_to_quiescence(200);
    kernel
}

/// A run containing a genuine mailbox send *and* its matching receive, so the
/// trace has a delivery edge to move. Both halves are driven from outside a
/// lane, which is why the negative test has to place the lanes itself.
fn delivery_workload() -> Kernel {
    let mut kernel = Kernel::new();
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    let spec = || {
        ContinuationSpec::new(
            StateAccess::ReadOnly,
            SEARCH_BRANCH,
            0,
            bytes.clone(),
            DEFAULT_MAX_STEPS,
        )
    };
    let sender_cont = kernel.create_continuation(sender, sender, spec()).unwrap();
    let receiver_cont = kernel
        .create_continuation(receiver, receiver, spec())
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

// ---- the reference satisfies it -------------------------------------------

#[test]
fn no_lane_of_an_epoch_observes_another() {
    // Cohort width is the knob that decides how many lanes an epoch has. At
    // width 1 every epoch is one lane and the clause is trivially true, so the
    // widths that matter are the ones that put several continuations in an
    // epoch side by side.
    for width in [1, 2, 4, 16] {
        for kernel in [search_workload(width), expand_workload(width)] {
            let violations: Vec<_> = check(&kernel)
                .into_iter()
                .filter(|v| v.invariant == Invariant::LaneIndependence)
                .collect();
            assert!(
                violations.is_empty(),
                "width {width}: {:?}",
                violations.first()
            );
        }
    }
}

#[test]
fn the_null_is_that_these_runs_have_an_order_and_several_lanes() {
    // Without this, `no_lane_of_an_epoch_observes_another` would pass on a run
    // with no edges and one lane per epoch, which is the vacuous truth the
    // clause is least useful as.
    for width in [2, 16] {
        for kernel in [search_workload(width), expand_workload(width)] {
            let order = SemanticOrder::of(&kernel);
            assert!(
                !order.edges().is_empty(),
                "width {width}: derived order is empty, so I25 checked nothing"
            );

            let mut lanes_per_epoch: std::collections::BTreeMap<
                u32,
                std::collections::BTreeSet<u32>,
            > = std::collections::BTreeMap::new();
            for row in order.events() {
                if row.lane != soma::abi::traces::HOST_LANE {
                    lanes_per_epoch
                        .entry(row.epoch)
                        .or_default()
                        .insert(row.lane);
                }
            }
            assert!(
                lanes_per_epoch.values().any(|lanes| lanes.len() > 1),
                "width {width}: no epoch ran more than one lane, so there were no two lanes to \
                 be independent of each other"
            );
        }
    }
}

// ---- a state that violates it is reported ---------------------------------

/// Move a real delivery so that its send and its receive land in one epoch on
/// two different lanes.
///
/// This is the workload §4.3 (3) said was reachable in principle and that no
/// run in the suite performs: a lane sends, and a *later lane of the same
/// epoch* receives. Under canonical commit the receiving lane's behaviour
/// depends on the sending lane having gone first, which is precisely what the
/// epoch-boundary applier stops preserving.
fn deliver_across_two_lanes(rows: &mut [TraceSnapshotRow]) -> (usize, usize) {
    let sent = rows
        .iter()
        .position(|row| row.event_kind == EventKind::MessageSent)
        .expect("the delivery workload sends");
    let received = rows
        .iter()
        .position(|row| row.event_kind == EventKind::MessageReceived)
        .expect("the delivery workload receives");

    let epoch = rows[sent].epoch;
    rows[sent].lane = 1;
    rows[sent].lane_sequence = 0;
    rows[received].epoch = epoch;
    rows[received].lane = 2;
    rows[received].lane_sequence = 0;
    (sent, received)
}

#[test]
fn a_delivery_between_two_lanes_of_one_epoch_is_reported() {
    let kernel = delivery_workload();
    let mut rows = kernel.trace_snapshot();

    // The null: as run, this workload drives both halves from the host, so
    // there is no cross-lane edge to find and the assertion below would pass
    // for the wrong reason.
    assert!(
        SemanticOrder::from_trace(rows.clone())
            .cross_lane_edges()
            .is_empty(),
        "the workload already violated I25 before the injection"
    );

    let (sent, received) = deliver_across_two_lanes(&mut rows);
    let order = SemanticOrder::from_trace(rows);
    let cross = order.cross_lane_edges();

    assert!(
        cross.iter().any(|edge| edge.earlier == sent
            && edge.later == received
            && edge.reason == EdgeReason::MessageDelivery),
        "moving a delivery across two lanes of one epoch was not reported: {cross:?}"
    );
}

#[test]
fn an_edge_touching_the_host_lane_is_not_a_violation() {
    // The host's part of an epoch runs strictly before and after the lanes, so
    // an order between it and a lane is the plan's rather than a race. Without
    // this exclusion every run would report, since the host emits an epoch's
    // cohort records and deferrals into the same epoch the lanes run in.
    let kernel = delivery_workload();
    let mut rows = kernel.trace_snapshot();

    let sent = rows
        .iter()
        .position(|row| row.event_kind == EventKind::MessageSent)
        .expect("the delivery workload sends");
    let received = rows
        .iter()
        .position(|row| row.event_kind == EventKind::MessageReceived)
        .expect("the delivery workload receives");

    let epoch = rows[sent].epoch;
    rows[sent].lane = soma::abi::traces::HOST_LANE;
    rows[received].epoch = epoch;
    rows[received].lane = 2;

    let order = SemanticOrder::from_trace(rows);
    assert!(
        order.cross_lane_edges().is_empty(),
        "an edge from the host lane was reported as a race between two lanes"
    );
}
