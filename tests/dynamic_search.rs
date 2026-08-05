//! Step 2/3 verification: the synthetic branching-search workload (§25.1)
//! terminates, visits exactly the expected number of process nodes, groups them
//! by run class into the runnable bins, and is fully deterministic.

use soma::experiments::dynamic_search::{build, run, ControlKnobs};
use soma::replay::trace_reader::same_trace;

#[test]
fn search_terminates() {
    let knobs = ControlKnobs::default();
    let epochs = run(&knobs);
    assert!(epochs > 0, "search must make progress");
    assert!(epochs <= knobs.depth + 2, "tree must finish within ~depth epochs");
}

#[test]
fn search_visits_expected_number_of_processes() {
    let knobs = ControlKnobs::default();
    let mut kernel = build(&knobs);
    let expected = knobs.node_count();
    // All roots are pending before the first epoch.
    assert_eq!(kernel.total_pending(), knobs.process_count as usize);
    kernel.run_to_quiescence(100_000);

    assert_eq!(kernel.processes.len() as u64, expected);
    assert_eq!(kernel.continuations.len() as u64, expected);
    // Every node terminated; the search reached quiescence.
    assert_eq!(kernel.total_pending(), 0);
}

#[test]
fn search_is_deterministic() {
    let knobs = ControlKnobs::default();
    let mut k1 = build(&knobs);
    let mut k2 = build(&knobs);
    k1.run_to_quiescence(100_000);
    k2.run_to_quiescence(100_000);
    assert!(same_trace(&k1, &k2), "identical seeds must yield identical traces");
}

#[test]
fn node_count_scales_with_branching_and_depth() {
    // depth 0, branching 4 -> 1 node per root.
    let small = ControlKnobs {
        branching_factor: 4,
        depth: 0,
        process_count: 3,
        ..ControlKnobs::default()
    };
    assert_eq!(small.node_count(), 3);

    // branching 2, depth 2 -> 1 + 2 + 4 = 7 per root.
    let med = ControlKnobs {
        branching_factor: 2,
        depth: 2,
        process_count: 1,
        ..ControlKnobs::default()
    };
    assert_eq!(med.node_count(), 7);
}

#[test]
fn runnable_bins_group_by_run_class() {
    // The root runs in epoch 1 and produces two children. After that epoch all
    // pending work shares one run class (SEARCH_BRANCH) — the run-class grouping
    // that is the seed of continuation cohorting (§9).
    let knobs = ControlKnobs {
        process_count: 1,
        depth: 1,
        branching_factor: 2,
        ..ControlKnobs::default()
    };
    let mut kernel = build(&knobs);
    kernel.run_epoch();
    let counts = kernel.pending_counts();
    assert_eq!(counts.len(), 1, "all live work shares one run class");
    assert_eq!(counts[0].0, soma::compiler::run_classes::SEARCH_BRANCH);
    assert!(counts[0].1 >= 2, "two children were produced by the root");
}
