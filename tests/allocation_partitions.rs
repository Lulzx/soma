//! Partitioned allocation is placement (v0.3 §4.3).
//!
//! A device's lanes and a cluster's nodes cannot share an allocator without
//! contending on it, so each mints references from its own partition. Which
//! partition an entity lands in is decided by the epoch's plan and says nothing
//! about the entity, so spreading a run across partitions must rename entities
//! and change nothing else.
//!
//! That is I19's shape, applied to allocation instead of cohort width, and it
//! is only checkable because I18 compares up to a correspondence between names
//! (§2.6) — against raw references every partitioned run diverges immediately,
//! which is the null here.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{ProcessMode, Ref64, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;
use soma::semantics::order::{conforms_traces, placement_neutral};

/// A workload that allocates from inside lanes — processes, continuations,
/// futures, objects and capabilities — so partitioning has something to spread.
fn workload(partitions: u8, cohort_width: u16) -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(partitions);
    kernel.configure_cohorts(cohort_width, PartialCohortPolicy::RunPartial);
    create_expand(&mut kernel, 3);
    for _ in 0..4 {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let mut bytes = Vec::new();
        SearchFrame {
            value: 1,
            depth: 2,
            branching: 2,
            work_iters: 1,
            class_count: 1,
        }
        .encode(&mut bytes);
        kernel
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
    }
    kernel.run_to_quiescence(256);
    kernel
}

fn partitions_used(kernel: &Kernel) -> std::collections::BTreeSet<u8> {
    kernel
        .trace_snapshot()
        .iter()
        .flat_map(|row| [row.process, row.continuation, row.subject, row.causal])
        .filter(|encoded| *encoded != 0)
        .map(|encoded| Ref64::from_u64(encoded).partition)
        .collect()
}

// ---- the property ---------------------------------------------------------

#[test]
fn spreading_allocation_across_partitions_is_not_observable() {
    let runs: Vec<Kernel> = [1u8, 2, 4, 8].into_iter().map(|p| workload(p, 4)).collect();
    let borrowed: Vec<&Kernel> = runs.iter().collect();

    assert_eq!(
        placement_neutral(&borrowed),
        Vec::new(),
        "the number of allocator partitions must not change what a program observes"
    );
    for kernel in &runs {
        assert_legal(kernel);
    }
}

#[test]
fn partitioning_composes_with_cohort_width() {
    // The two placement knobs are independent, so varying both at once must
    // still land on one behaviour.
    let runs: Vec<Kernel> = [(1u8, 1u16), (2, 4), (4, 2), (8, 16)]
        .into_iter()
        .map(|(p, w)| workload(p, w))
        .collect();
    let borrowed: Vec<&Kernel> = runs.iter().collect();
    assert_eq!(placement_neutral(&borrowed), Vec::new());
}

// ---- the nulls ------------------------------------------------------------

#[test]
fn partitioning_really_spreads_the_allocations() {
    // Without this the agreement above would be the agreement of four identical
    // runs.
    let single = partitions_used(&workload(1, 4));
    let spread = partitions_used(&workload(4, 4));

    assert_eq!(
        single,
        std::collections::BTreeSet::from([0]),
        "one partition must mean partition 0 only"
    );
    assert!(
        spread.len() > 1,
        "four partitions produced entities in only {spread:?}"
    );
}

#[test]
fn a_partitioned_run_really_names_its_entities_differently() {
    // And this is why §2.6 had to come first: compared as raw references, the
    // partitioned run diverges from the reference on nearly every event.
    let reference = workload(1, 4);
    let partitioned = workload(4, 4);

    let raw_reference = reference.trace_snapshot();
    let raw_partitioned = partitioned.trace_snapshot();
    assert_eq!(
        raw_reference.len(),
        raw_partitioned.len(),
        "the two runs did the same amount of work"
    );
    let differing = raw_reference
        .iter()
        .zip(&raw_partitioned)
        .filter(|(a, b)| a.process != b.process || a.continuation != b.continuation)
        .count();
    assert!(
        differing > raw_reference.len() / 4,
        "only {differing} of {} events named entities differently, so the renaming \
         is doing nothing",
        raw_reference.len()
    );

    // Same traces, compared through the correspondence: no divergence.
    assert_eq!(
        conforms_traces(&raw_reference, &raw_partitioned),
        Vec::new()
    );
}

// ---- the mechanism --------------------------------------------------------

#[test]
fn slot_numbers_repeat_across_partitions() {
    // The point of partitioning: two allocators mint slot 7 at the same time and
    // mean different entities. If slots stayed globally unique, nothing would
    // have been decoupled.
    let kernel = workload(4, 4);
    let mut by_slot: std::collections::BTreeMap<(u8, u32), std::collections::BTreeSet<u8>> =
        std::collections::BTreeMap::new();
    for row in kernel.trace_snapshot() {
        for encoded in [row.process, row.continuation] {
            if encoded == 0 {
                continue;
            }
            let r = Ref64::from_u64(encoded);
            if r.slot == 0 {
                continue;
            }
            by_slot
                .entry((r.kind.as_u8(), r.slot))
                .or_default()
                .insert(r.partition);
        }
    }
    assert!(
        by_slot.values().any(|partitions| partitions.len() > 1),
        "no slot number was reused across partitions"
    );
}

#[test]
fn a_lanes_partition_comes_from_its_position_not_its_identity() {
    // Determinism rests on this: the partition is a function of the plan, so
    // two runs of one program partition identically however the lanes were
    // dispatched. Re-running must reproduce the assignment exactly.
    let first = workload(4, 4);
    let second = workload(4, 4);
    assert_eq!(
        first.trace_snapshot(),
        second.trace_snapshot(),
        "partitioned allocation must be reproducible, not merely deterministic-looking"
    );
}
