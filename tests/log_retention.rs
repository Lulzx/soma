//! Bounded log retention (`kernel::retention`).
//!
//! The append-only logs grow with the run, which is correct for whole-run
//! invariant checking and fatal for a long one. `LogRetention::PerEpoch` lets a
//! consumer stream them out instead. Two things have to be true for that to be
//! worth having: streaming must reconstruct exactly the trace a retaining run
//! kept, and a consumer that fails to drain must be *told*, not quietly handed a
//! log that looks complete.
//!
//! Per the repository's test discipline, the second property has a failing case:
//! `undrained_records_are_reported_as_dropped` is the run that loses records, and
//! it asserts that the census says so.

use soma::abi::TraceEvent;
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::retention::LogRetention;
use soma::kernel::Kernel;

/// A workload with enough epochs that retention has something to do.
fn knobs() -> ControlKnobs {
    ControlKnobs {
        branching_factor: 3,
        depth: 4,
        process_count: 3,
        class_count: 4,
        ..ControlKnobs::default()
    }
}

/// The comparable projection of a trace event. `logical_time` and the lane
/// position are included: a streamed trace that reordered anything, or restarted
/// a counter, diverges here.
type Row = (u64, u32, u32, u32, u16, u64, u64, u32, u32);

fn row(e: &TraceEvent) -> Row {
    (
        e.logical_time,
        e.epoch,
        e.lane,
        e.lane_sequence,
        e.event_kind as u16,
        e.process.to_u64(),
        e.continuation.to_u64(),
        e.run_class,
        e.auxiliary,
    )
}

fn run_to_quiescence_stepwise(kernel: &mut Kernel, max_epochs: u32) -> u32 {
    let mut epochs = 0;
    while kernel.total_pending() > 0 && epochs < max_epochs {
        kernel.run_epoch();
        epochs += 1;
    }
    epochs
}

#[test]
fn retain_is_the_default_and_keeps_the_whole_run() {
    let mut kernel = build(&knobs());
    assert_eq!(kernel.log_retention(), LogRetention::Retain);

    run_to_quiescence_stepwise(&mut kernel, 10_000);

    let census = kernel.log_accounting();
    assert!(census.is_balanced(), "every record must be accounted for");
    assert!(
        census.is_complete(),
        "the default policy must not drop anything"
    );
    assert_eq!(census.trace.dropped, 0);
    assert_eq!(census.trace.taken, 0);
    assert_eq!(census.trace.retained, census.trace.emitted);
    assert_eq!(kernel.trace_events().len() as u64, census.trace.emitted);
    // The logs the whole-run checks read are all still whole.
    assert_eq!(
        kernel.admission_log().len() as u64,
        census.admissions.emitted
    );
    assert_eq!(kernel.effect_log().len() as u64, census.effects.emitted);
}

/// The property that makes `PerEpoch` a retention policy rather than a
/// semantic one: draining epoch by epoch reconstructs the retaining run's trace
/// exactly — same events, same order, same positions, nothing lost.
#[test]
fn streaming_reconstructs_the_retained_trace_exactly() {
    let mut retained = build(&knobs());
    let retained_epochs = run_to_quiescence_stepwise(&mut retained, 10_000);
    let expected: Vec<Row> = retained.trace_events().iter().map(row).collect();

    let mut streamed = build(&knobs());
    streamed.set_log_retention(LogRetention::PerEpoch);
    // Setup events belong to no epoch and the first `run_epoch` would discard
    // them, so a streaming consumer drains before it runs. That is the contract.
    let mut collected: Vec<Row> = streamed.take_trace_events().iter().map(row).collect();
    let _ = streamed.take_effect_log();
    let _ = streamed.take_admission_log();
    let mut streamed_epochs = 0;
    while streamed.total_pending() > 0 && streamed_epochs < 10_000 {
        streamed.run_epoch();
        collected.extend(streamed.take_trace_events().iter().map(row));
        // A complete consumer drains all three logs, not just the one it reads.
        // Leaving either of the others undrained is the `dropped` case, which
        // `undrained_records_are_reported_as_dropped` covers.
        let _ = streamed.take_effect_log();
        let _ = streamed.take_admission_log();
        streamed_epochs += 1;
    }
    // Anything produced after the last epoch ran.
    collected.extend(streamed.take_trace_events().iter().map(row));
    let _ = streamed.take_effect_log();
    let _ = streamed.take_admission_log();

    assert_eq!(
        retained_epochs, streamed_epochs,
        "retention must not change how the run schedules"
    );
    assert_eq!(
        collected.len(),
        expected.len(),
        "a streamed trace must hold as many events as a retained one"
    );
    assert_eq!(
        collected, expected,
        "a streamed trace must be the retained trace, event for event"
    );

    let census = streamed.log_accounting();
    assert!(census.is_balanced());
    assert!(
        census.is_complete(),
        "a consumer that drained every epoch must have dropped nothing"
    );
    assert_eq!(census.trace.taken, census.trace.emitted);
    assert_eq!(census.trace.retained, 0);
}

/// The failing case. A `PerEpoch` run that never drains loses records — that is
/// the point of the policy — and the thing being checked is that it says so.
/// Without this, a truncated log is indistinguishable from a complete one.
#[test]
fn undrained_records_are_reported_as_dropped() {
    let mut kernel = build(&knobs());
    kernel.set_log_retention(LogRetention::PerEpoch);

    let epochs = run_to_quiescence_stepwise(&mut kernel, 10_000);
    assert!(epochs > 1, "the workload must span several epochs");

    let census = kernel.log_accounting();
    assert!(census.is_balanced(), "dropped records are still counted");
    assert!(
        !census.is_complete(),
        "a run that never drained must report incompleteness"
    );
    assert!(
        census.trace.dropped > 0,
        "trace records were discarded and must be counted as dropped"
    );
    assert_eq!(census.trace.taken, 0);
    // The log itself looks small and healthy. Only the census reveals the loss,
    // which is exactly why the census exists.
    assert!(
        (kernel.trace_events().len() as u64) < census.trace.emitted,
        "the retained log must be shorter than the run that produced it"
    );
}

/// Retention must not touch what the machine does — only what it remembers.
#[test]
fn retention_does_not_change_the_run() {
    let mut retained = build(&knobs());
    let retained_epochs = run_to_quiescence_stepwise(&mut retained, 10_000);

    let mut bounded = build(&knobs());
    bounded.set_log_retention(LogRetention::PerEpoch);
    let bounded_epochs = run_to_quiescence_stepwise(&mut bounded, 10_000);

    assert_eq!(retained_epochs, bounded_epochs);
    assert_eq!(retained.accounting(), bounded.accounting());
    assert_eq!(retained.process_count(), bounded.process_count());
    assert_eq!(retained.continuation_count(), bounded.continuation_count());
}

/// The logs stop growing. This is the whole reason the policy exists, so it is
/// checked directly rather than inferred from the counters.
#[test]
fn bounded_retention_bounds_the_logs() {
    let mut kernel = build(&knobs());
    kernel.set_log_retention(LogRetention::PerEpoch);

    let mut high_water = 0usize;
    let mut epochs = 0;
    while kernel.total_pending() > 0 && epochs < 10_000 {
        kernel.run_epoch();
        high_water = high_water.max(kernel.trace_events().len());
        // The admission log holds one record per epoch, and under this policy
        // that record is the current epoch's.
        assert!(
            kernel.admission_log().len() <= 1,
            "at most the running epoch's admission is retained"
        );
        epochs += 1;
    }

    let census = kernel.log_accounting();
    assert!(
        (high_water as u64) < census.trace.emitted,
        "a bounded log must never hold the whole run"
    );
}

