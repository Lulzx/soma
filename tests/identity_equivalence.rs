//! I18 up to a renaming of entity names.
//!
//! An identity is a table position, and a table position is an implementation
//! detail. A device whose lanes allocate from separate partitions, and a
//! cluster whose nodes do, name the same entity differently and behave
//! identically. Comparing raw `Ref64`s makes both non-conforming by
//! construction — the same defect trace equality had for *ordering* before
//! v0.3 §2, one level down.
//!
//! The correspondence is forced rather than chosen: entities pair in order of
//! first appearance within their kind. That is what keeps the widening from
//! swallowing real defects, and it is what these tests are mostly about — a
//! checker free to pick the correspondence could pair whatever made two traces
//! agree.

use soma::abi::{ProcessMode, Ref64, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::kernel::{ContinuationSpec, Kernel, TraceSnapshotRow, SYSTEM_PRINCIPAL};
use soma::semantics::order::conforms_traces;

fn workload() -> Kernel {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 3);
    for _ in 0..3 {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let mut bytes = Vec::new();
        SearchFrame::leaf(1, 0).encode(&mut bytes);
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
    kernel.run_to_quiescence(64);
    kernel
}

/// Apply `f` to every entity-shaped field of every row. A uniform bijection
/// stands in for the run an allocator with different partitions would produce.
fn rename(events: &[TraceSnapshotRow], mut f: impl FnMut(u64) -> u64) -> Vec<TraceSnapshotRow> {
    let mut apply = move |encoded: u64| {
        if encoded == 0 || Ref64::from_u64(encoded).slot == 0 {
            encoded
        } else {
            f(encoded)
        }
    };
    events
        .iter()
        .map(|row| TraceSnapshotRow {
            process: apply(row.process),
            continuation: apply(row.continuation),
            subject: apply(row.subject),
            causal: apply(row.causal),
            ..*row
        })
        .collect()
}

/// Shift every slot by a constant: a different partition's names for the same
/// entities, in the same order.
fn shifted(encoded: u64) -> u64 {
    let mut r = Ref64::from_u64(encoded);
    r.slot += 1_000;
    r.to_u64()
}

// ---- the property ---------------------------------------------------------

#[test]
fn a_uniformly_renamed_run_conforms() {
    let kernel = workload();
    let reference = kernel.trace_snapshot();
    let renamed = rename(&reference, shifted);

    assert_ne!(
        renamed, reference,
        "the renaming must actually change the trace"
    );
    assert_eq!(
        conforms_traces(&reference, &renamed),
        Vec::new(),
        "a run that differs only in what it calls its entities must conform"
    );
}

#[test]
fn renaming_across_partitions_conforms() {
    // Closer to what a partitioned allocator produces: the same entities, named
    // in a different space, with slot numbers that collide across partitions.
    let kernel = workload();
    let reference = kernel.trace_snapshot();
    let mut counter = 0u32;
    let mut assigned: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    let renamed = rename(&reference, |encoded| {
        *assigned.entry(encoded).or_insert_with(|| {
            let mut r = Ref64::from_u64(encoded);
            r.partition = (counter % 4) as u8;
            r.slot = counter / 4 + 1;
            counter += 1;
            r.to_u64()
        })
    });

    assert_eq!(conforms_traces(&reference, &renamed), Vec::new());
}

// ---- the null -------------------------------------------------------------

#[test]
fn the_reference_run_really_names_several_entities_per_kind() {
    // If every kind had one entity, any correspondence would be the only
    // correspondence and pairing could not be wrong.
    let kernel = workload();
    let mut per_kind: std::collections::BTreeMap<u8, std::collections::BTreeSet<u64>> =
        std::collections::BTreeMap::new();
    for row in kernel.trace_snapshot() {
        for encoded in [row.process, row.continuation, row.subject, row.causal] {
            if encoded == 0 || Ref64::from_u64(encoded).slot == 0 {
                continue;
            }
            per_kind
                .entry(Ref64::from_u64(encoded).kind.as_u8())
                .or_default()
                .insert(encoded);
        }
    }
    assert!(
        per_kind.values().filter(|names| names.len() > 1).count() >= 2,
        "at least two kinds must name several entities: {:?}",
        per_kind
            .iter()
            .map(|(k, v)| (k, v.len()))
            .collect::<Vec<_>>()
    );
}

// ---- the fault injections -------------------------------------------------

#[test]
fn merging_two_entities_is_reported() {
    // The failure the widening most obviously risks: a run that used one entity
    // where the reference used two. Positional pairing of distinct names is a
    // bijection, so the merged run simply has fewer names.
    let kernel = workload();
    let reference = kernel.trace_snapshot();

    let victim = reference
        .iter()
        .map(|row| row.continuation)
        .find(|c| *c != 0 && Ref64::from_u64(*c).slot != 0)
        .expect("the run named a continuation");
    let survivor = reference
        .iter()
        .map(|row| row.continuation)
        .find(|c| *c != 0 && Ref64::from_u64(*c).slot != 0 && *c != victim)
        .expect("the run named a second continuation");

    let merged = rename(&reference, |encoded| {
        if encoded == victim {
            survivor
        } else {
            encoded
        }
    });

    assert!(
        !conforms_traces(&reference, &merged).is_empty(),
        "a run that collapsed two continuations into one must be reported"
    );
}

#[test]
fn dropping_an_entity_is_reported() {
    let kernel = workload();
    let reference = kernel.trace_snapshot();
    let victim = reference
        .iter()
        .map(|row| row.continuation)
        .find(|c| *c != 0 && Ref64::from_u64(*c).slot != 0)
        .expect("the run named a continuation");

    let dropped: Vec<TraceSnapshotRow> = reference
        .iter()
        .filter(|row| row.continuation != victim)
        .copied()
        .collect();

    assert!(
        !conforms_traces(&reference, &dropped).is_empty(),
        "a run that never mentions one of the reference's entities must be reported"
    );
}

#[test]
fn an_inconsistent_renaming_is_reported() {
    // Renaming only *some* occurrences of an entity is the case a map built by
    // first appearance cannot paper over: the map pairs the name it met first,
    // and the untranslated occurrences stop matching.
    let kernel = workload();
    let reference = kernel.trace_snapshot();
    let victim = reference
        .iter()
        .map(|row| row.continuation)
        .find(|c| *c != 0 && Ref64::from_u64(*c).slot != 0)
        .expect("the run named a continuation");

    let mut seen = 0;
    let mut patched = reference.clone();
    for row in patched.iter_mut() {
        if row.continuation == victim {
            seen += 1;
            if seen > 1 {
                row.continuation = shifted(victim);
            }
        }
    }
    assert!(seen > 1, "the victim must appear more than once");

    assert!(
        !conforms_traces(&reference, &patched).is_empty(),
        "renaming an entity in some events but not others must be reported"
    );
}

#[test]
fn swapping_two_entities_uniformly_is_a_renaming_and_conforms() {
    // The boundary, recorded because it is the one that looks like a defect and
    // is not. Exchanging two entities' names *everywhere* produces a trace whose
    // first-appearance order is exchanged too, so the forced correspondence
    // pairs them the other way round and undoes the swap. That is right: the
    // run did the same things and called them different names.
    //
    // What a swap must not survive is being applied to only part of the run —
    // that is `an_inconsistent_renaming_is_reported`, and it is the case that
    // separates a renaming from a behavioural difference.
    let kernel = workload();
    let reference = kernel.trace_snapshot();

    let mut names: Vec<u64> = Vec::new();
    for row in &reference {
        for encoded in [row.process, row.continuation, row.subject, row.causal] {
            if encoded != 0 && Ref64::from_u64(encoded).slot != 0 && !names.contains(&encoded) {
                names.push(encoded);
            }
        }
    }
    let first = names[0];
    let second = *names
        .iter()
        .find(|n| Ref64::from_u64(**n).kind == Ref64::from_u64(first).kind && **n != first)
        .expect("two entities of one kind");

    let swapped = rename(&reference, |encoded| {
        if encoded == first {
            second
        } else if encoded == second {
            first
        } else {
            encoded
        }
    });

    assert_ne!(swapped, reference, "the swap must change the trace");
    assert_eq!(
        conforms_traces(&reference, &swapped),
        Vec::new(),
        "a uniform exchange of two names is a renaming, not a behavioural difference"
    );
}
