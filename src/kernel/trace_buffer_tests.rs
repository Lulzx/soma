//! The lane-local trace buffer (`docs/SOMA-v0.3.md` §4.11).
//!
//! These are in-crate because what they check is structural rather than
//! behavioural: no run changes, so nothing about the buffer is visible from
//! outside the kernel. `enter_lane`, `leave_lane` and `trace` are `pub(crate)`
//! or private, and the point of the exercise is that a step *cannot* reach the
//! shared trace, so there is no public path to observe the difference from.
//! Every other property of trace emission is checked from `tests/`, over
//! whole runs, where it belongs.

use super::*;
use crate::abi::traces::HOST_LANE;

fn emit(kernel: &mut Kernel, kind: EventKind) {
    kernel.trace(kind, Ref64::NULL, Ref64::NULL, 0, 0);
}

/// The property. Between `enter_lane` and `leave_lane` the shared trace does
/// not move.
#[test]
fn a_lanes_events_do_not_reach_the_trace_until_the_lane_ends() {
    let mut kernel = Kernel::new();
    let before = kernel.trace_events().len();
    let clock_before = kernel.logical_time;

    kernel.enter_lane(1);
    emit(&mut kernel, EventKind::AuthorityEffect);
    emit(&mut kernel, EventKind::AuthorityReleased);
    emit(&mut kernel, EventKind::AuthorityEffect);

    assert_eq!(
        kernel.trace_events().len(),
        before,
        "a lane appended to the run's trace while it was running"
    );
    assert_eq!(
        kernel.logical_time, clock_before,
        "a lane took a number from the run's clock while it was running"
    );
    assert_eq!(kernel.lane_trace.len(), 3, "the events went somewhere else");

    kernel.leave_lane();
    assert_eq!(kernel.trace_events().len(), before + 3);
    assert!(kernel.lane_trace.is_empty());
}

/// The null. Without this the test above passes for a kernel that never
/// appends anything at all.
#[test]
fn a_host_event_is_appended_where_it_is_emitted() {
    let mut kernel = Kernel::new();
    let before = kernel.trace_events().len();
    emit(&mut kernel, EventKind::AuthorityEffect);
    assert_eq!(
        kernel.trace_events().len(),
        before + 1,
        "an event emitted outside a lane has no boundary to wait for"
    );
    assert!(kernel.lane_trace.is_empty());
}

/// The clock is handed out at the drain, in the order the lane emitted — which
/// is what makes the buffer change no run.
#[test]
fn draining_hands_out_the_clock_in_emission_order() {
    let mut kernel = Kernel::new();
    kernel.enter_lane(1);
    let kinds = [
        EventKind::AuthorityEffect,
        EventKind::AuthorityReleased,
        EventKind::AuthorityGranted,
    ];
    for kind in kinds {
        emit(&mut kernel, kind);
    }
    kernel.leave_lane();

    let tail = &kernel.trace_events()[kernel.trace_events().len() - 3..];
    assert_eq!(
        tail.iter().map(|e| e.event_kind).collect::<Vec<_>>(),
        kinds.to_vec(),
        "the drain reordered the lane"
    );
    for (i, event) in tail.iter().enumerate() {
        assert_eq!(event.lane, 1);
        assert_eq!(
            event.lane_sequence, i as u32,
            "positions are assigned at emission"
        );
        assert_eq!(
            event.logical_time,
            tail[0].logical_time + i as u64,
            "the clock is contiguous over a lane's events"
        );
    }
}

/// An event a lane holds is retained, not lost. The census compares emissions
/// against what is still held, and the buffer is somewhere it can be held.
#[test]
fn an_undrained_event_is_counted_as_retained() {
    let mut kernel = Kernel::new();
    kernel.enter_lane(1);
    emit(&mut kernel, EventKind::AuthorityEffect);
    let census = kernel.log_accounting().trace;
    assert!(
        census.is_balanced(),
        "emitted={} retained={} taken={} dropped={}",
        census.emitted,
        census.retained,
        census.taken,
        census.dropped
    );
    assert!(census.is_complete(), "nothing was dropped");
    kernel.leave_lane();
    assert!(kernel.log_accounting().trace.is_balanced());
}

/// The fault injection: skipping the drain is caught where it happens rather
/// than showing up later as a lane credited with another lane's work.
#[test]
#[should_panic(expected = "undrained")]
fn entering_a_lane_over_an_undrained_buffer_is_rejected() {
    let mut kernel = Kernel::new();
    kernel.enter_lane(1);
    emit(&mut kernel, EventKind::AuthorityEffect);
    // What `leave_lane` would have done, not done.
    kernel.enter_lane(2);
}

/// A lane's events carry its number whether they are held or appended. The
/// buffer is where an event waits, not what decides its position.
#[test]
fn the_buffer_does_not_decide_a_position() {
    let mut kernel = Kernel::new();
    kernel.enter_lane(7);
    emit(&mut kernel, EventKind::AuthorityEffect);
    assert_eq!(kernel.lane_trace[0].lane, 7);
    assert_eq!(kernel.lane_trace[0].lane_sequence, 0);
    kernel.leave_lane();
    emit(&mut kernel, EventKind::AuthorityEffect);
    let last = kernel.trace_events().last().copied().unwrap();
    assert_eq!(last.lane, HOST_LANE);
}
