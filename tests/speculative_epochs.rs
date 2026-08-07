use soma::abi::{ProcessMode, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_HEURISTIC};
use soma::compiler::state_machine_lowering::{create_expand, HeuristicFrame, SearchFrame};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::speculation::EpochExecutive;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;
use soma::semantics::order::conforms_traces;

fn leaf_knobs() -> ControlKnobs {
    ControlKnobs {
        branching_factor: 0,
        depth: 0,
        process_count: 8,
        class_count: 3,
        arithmetic_ops: 2_000,
        ..ControlKnobs::default()
    }
}

#[test]
fn disjoint_lanes_commit_the_reference_run() {
    let knobs = leaf_knobs();
    let mut reference = build(&knobs);
    let mut speculative = build(&knobs);
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });

    assert_eq!(reference.run_epoch(), knobs.process_count as usize);
    assert_eq!(speculative.run_epoch(), knobs.process_count as usize);

    let stats = speculative.speculation_stats();
    assert_eq!(stats.attempted_epochs, 1);
    assert_eq!(stats.committed_epochs, 1);
    assert_eq!(stats.fallback_epochs, 0);
    assert_eq!(stats.committed_lanes, knobs.process_count as u64);
    assert!(
        conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot()).is_empty(),
        "canonical commit must reproduce the reference trace"
    );
    assert_legal(&speculative);
}

fn two_leaves_in_one_process() -> Kernel {
    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    for value in [11, 29] {
        let frame = SearchFrame {
            value,
            depth: 0,
            branching: 0,
            work_iters: 64,
            class_count: 1,
        };
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    frame.run_class(),
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel
}

#[test]
fn a_process_commit_conflict_replays_the_whole_epoch() {
    let mut reference = two_leaves_in_one_process();
    let mut speculative = two_leaves_in_one_process();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    let stats = speculative.speculation_stats();
    assert_eq!(stats.committed_epochs, 0);
    assert_eq!(stats.fallback_epochs, 1);
    assert_eq!(stats.conflict_fallbacks, 1);
    let disagreements =
        conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
    assert_legal(&speculative);
}

#[test]
fn contended_allocation_falls_back_before_any_snapshot_effect_can_escape() {
    let knobs = ControlKnobs {
        depth: 1,
        process_count: 3,
        branching_factor: 2,
        ..leaf_knobs()
    };
    let mut reference = build(&knobs);
    let mut speculative = build(&knobs);
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    let stats = speculative.speculation_stats();
    assert_eq!(stats.fallback_epochs, 1);
    assert_eq!(stats.conflict_fallbacks, 1);
    assert!(
        conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot()).is_empty()
    );
    assert_legal(&speculative);
}

fn independent_expands() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(16);
    for value in 1..=4 {
        create_expand(&mut kernel, value);
    }
    kernel
}

#[test]
fn independent_mailboxes_futures_and_allocations_commit() {
    let mut reference = independent_expands();
    let mut speculative = independent_expands();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });

    reference.run_to_quiescence(64);
    speculative.run_to_quiescence(64);

    let stats = speculative.speculation_stats();
    assert!(stats.committed_epochs >= 2, "{stats:?}");
    assert!(stats.committed_lanes >= 8, "{stats:?}");
    let disagreements =
        conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
    assert_legal(&speculative);
}

fn contested_future() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(8);
    let future = kernel.create_future(SYSTEM_PRINCIPAL);
    for input in [7, 13] {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .grant_capability(
                SYSTEM_PRINCIPAL,
                process,
                future,
                Rights::RESOLVE,
                0,
                0,
            )
            .unwrap();
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
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel
}

#[test]
fn two_future_writers_conflict_and_replay_in_plan_order() {
    let mut reference = contested_future();
    let mut speculative = contested_future();
    speculative.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    speculative.run_epoch();

    let stats = speculative.speculation_stats();
    assert_eq!(stats.committed_epochs, 0);
    assert_eq!(stats.conflict_fallbacks, 1);
    let disagreements =
        conforms_traces(&reference.trace_snapshot(), &speculative.trace_snapshot());
    assert!(disagreements.is_empty(), "{disagreements:#?}");
}
