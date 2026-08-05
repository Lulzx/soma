//! I23: the trace's order is recoverable from event positions, with no clock.
//!
//! `docs/SOMA-v0.3.md` §4 lists trace emission as the fourth thing a
//! device-resident scheduler must preserve. `logical_time` comes from a single
//! counter, and concurrent lanes have none to share. The clause is that they do
//! not need one: every event carries `(epoch, lane, lane_sequence)`, those
//! positions are unique, and sorting by them reproduces the run.
//!
//! The test that carries the weight is `sorting_by_position_reconstructs_the_run`
//! — it throws the trace's order away and rebuilds it. Everything else either
//! shows the reconstruction is not trivial or shows the checker can fail.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::traces::HOST_LANE;
use soma::abi::{EventKind, ProcessMode, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::kernel::{raw, ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::{assert_legal, check, Invariant};

fn leaf_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    bytes
}

/// A workload with several epochs, several lanes per epoch, and lanes that emit
/// more than one event each — everything the reconstruction needs in order to
/// be saying something.
fn busy_kernel(cohort_width: u16) -> Kernel {
    let mut kernel = Kernel::new();
    kernel.configure_cohorts(cohort_width, PartialCohortPolicy::RunPartial);
    create_expand(&mut kernel, 3);
    for _ in 0..4 {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .create_continuation(
                process,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    SEARCH_BRANCH,
                    0,
                    leaf_bytes(),
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel.run_to_quiescence(64);
    kernel
}

fn violated(kernel: &Kernel, invariant: Invariant) -> bool {
    check(kernel).iter().any(|v| v.invariant == invariant)
}

// ---- the property ---------------------------------------------------------

#[test]
fn sorting_by_position_reconstructs_the_run() {
    let kernel = busy_kernel(4);
    let emitted: Vec<_> = kernel.trace_events().to_vec();

    // Throw the emission order away, the way a concurrent append would.
    let mut scrambled = emitted.clone();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..scrambled.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        scrambled.swap(i, (state >> 33) as usize % (i + 1));
    }
    assert_ne!(
        scrambled.iter().map(|e| e.logical_time).collect::<Vec<_>>(),
        emitted.iter().map(|e| e.logical_time).collect::<Vec<_>>(),
        "the scramble must actually disturb the order"
    );

    scrambled.sort_by_key(|event| event.position());
    assert_eq!(
        scrambled.iter().map(|e| e.logical_time).collect::<Vec<_>>(),
        emitted.iter().map(|e| e.logical_time).collect::<Vec<_>>(),
        "position alone must recover the order the run was emitted in"
    );
}

#[test]
fn every_cohort_width_emits_in_position_order() {
    for width in [1u16, 2, 4, 16] {
        let kernel = busy_kernel(width);
        assert!(
            !violated(&kernel, Invariant::PositionDerivedEmission),
            "width {width} emitted out of position order"
        );
        assert_legal(&kernel);
    }
}

// ---- the nulls: the reconstruction is not trivial -------------------------

#[test]
fn the_run_really_uses_several_lanes_across_several_epochs() {
    let kernel = busy_kernel(4);
    let events = kernel.trace_events();

    let epochs: std::collections::BTreeSet<u32> = events.iter().map(|e| e.epoch).collect();
    let lanes: std::collections::BTreeSet<u32> = events
        .iter()
        .map(|e| e.lane)
        .filter(|lane| *lane != HOST_LANE)
        .collect();
    let chatty = events.iter().filter(|e| e.lane_sequence > 0).count();

    assert!(epochs.len() > 1, "one epoch makes the epoch key redundant");
    assert!(lanes.len() > 1, "one lane makes the lane key redundant");
    assert!(
        chatty > 0,
        "no lane emitted more than one event, so the sequence key is redundant"
    );
}

#[test]
fn host_events_precede_the_lanes_of_their_epoch() {
    // The host's part of an epoch — its cohort records — is emitted before any
    // lane runs, so `HOST_LANE` being zero is the ordering and not a coincidence
    // of the sentinel value.
    let kernel = busy_kernel(2);
    for event in kernel.trace_events() {
        if event.event_kind != EventKind::CohortCreated {
            continue;
        }
        assert_eq!(
            event.lane, HOST_LANE,
            "a cohort record is a plan, not a lane's work"
        );
    }
}

// ---- the fault injections -------------------------------------------------

#[test]
fn i23_catches_two_events_at_one_position() {
    let mut kernel = busy_kernel(2);
    assert!(!violated(&kernel, Invariant::PositionDerivedEmission));

    let trace = unsafe { raw::state(&mut kernel) }.trace;
    let stolen = trace[1].position();
    let victim = trace
        .iter_mut()
        .rfind(|event| event.position() != stolen)
        .expect("the run emitted more than one position");
    victim.epoch = stolen.0;
    victim.lane = stolen.1;
    victim.lane_sequence = stolen.2;

    assert!(
        violated(&kernel, Invariant::PositionDerivedEmission),
        "two events at one position leave their order unrecoverable"
    );
}

#[test]
fn i23_catches_an_event_emitted_out_of_position_order() {
    let mut kernel = busy_kernel(2);
    assert!(!violated(&kernel, Invariant::PositionDerivedEmission));

    // A lane that counted from a stale sequence: the event was appended after
    // its neighbour but claims to have come before it. This is exactly what a
    // concurrent implementation gets wrong when a lane's counter is shared.
    let trace = unsafe { raw::state(&mut kernel) }.trace;
    let last = trace.len() - 1;
    trace[last].lane_sequence = 0;
    trace[last].lane = HOST_LANE;

    assert!(
        violated(&kernel, Invariant::PositionDerivedEmission),
        "an event that sorts before the one it followed must be reported"
    );
}

#[test]
fn one_lane_carries_one_continuation() {
    // A lane number is a position in the epoch's plan, and each position holds
    // one continuation. If two continuations shared a lane, their events would
    // interleave in one sequence space and program order would stop being
    // recoverable from position — which is what I18 clause 2 relies on.
    let kernel = busy_kernel(4);
    let mut occupant: std::collections::BTreeMap<(u32, u32), u64> =
        std::collections::BTreeMap::new();
    let mut lanes = 0;
    for event in kernel.trace_events() {
        if event.lane == HOST_LANE || event.event_kind != EventKind::ContinuationStarted {
            continue;
        }
        lanes += 1;
        let previous = occupant.insert((event.epoch, event.lane), event.continuation.to_u64());
        assert_eq!(
            previous, None,
            "epoch {} lane {} started a second continuation",
            event.epoch, event.lane
        );
    }
    assert!(lanes > 1, "the run must have used more than one lane");
}
