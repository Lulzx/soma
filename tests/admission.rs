//! I22: admission decides from the epoch's candidate set, not from its order.
//!
//! `docs/SOMA-v0.3.md` §4 names this as the first obligation a device-resident
//! scheduler has to meet. The host used to resolve I13's per-process mutable
//! claim with a `HashSet` taken as it scanned the bins, so the winner was
//! whichever continuation the scan met first. That is a race on a device, where
//! the epoch's runnable set is claimed concurrently.
//!
//! The positive cases here are cheap; the ones that matter are the null (the
//! workload really does make two mutable continuations compete, so the check is
//! not vacuous) and the fault injection (the rule this replaced *fails* the
//! check, so the check has teeth).

use soma::abi::continuations::ContinuationState;
use soma::abi::{Kind, ProcessMode, Ref64, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::SearchFrame;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::admission::{admit, Candidate, Decision};
use soma::semantics::invariants::assert_legal;
use soma::semantics::schedule::{admission_determinism, decision_is_order_independent};

fn leaf_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    bytes
}

fn spawn_leaf(kernel: &mut Kernel, process: Ref64, state_access: StateAccess) -> Ref64 {
    kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(
                state_access,
                SEARCH_BRANCH,
                0,
                leaf_bytes(),
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap()
}

/// Three processes, each with several mutable continuations plus a read-only
/// one. Every epoch has a contested claim to resolve, and the read-only
/// continuations are there so the deferral is visibly selective rather than a
/// blanket one-per-process rule.
fn contended_kernel() -> Kernel {
    let mut kernel = Kernel::new();
    for _ in 0..3 {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        for _ in 0..3 {
            spawn_leaf(&mut kernel, process, StateAccess::Mutable);
        }
        spawn_leaf(&mut kernel, process, StateAccess::ReadOnly);
    }
    kernel
}

/// A synthetic candidate. Slot and generation are the identity `admit` breaks
/// ties on, so the tests set them explicitly rather than borrowing a kernel's.
fn candidate(slot: u32, process: u32, state_access: StateAccess, waiting_since: u32) -> Candidate {
    Candidate {
        bin: SEARCH_BRANCH,
        continuation: Ref64::new(slot, 1, Kind::Continuation),
        process: Ref64::new(process, 1, Kind::Process),
        run_class: SEARCH_BRANCH,
        state_access,
        waiting_since,
    }
}

// ---- the property ---------------------------------------------------------

#[test]
fn admission_decides_the_same_thing_in_any_discovery_order() {
    let mut kernel = contended_kernel();
    kernel.run_to_quiescence(64);

    assert_eq!(
        admission_determinism(&kernel),
        Vec::new(),
        "the admission decision must not depend on the order candidates were drained in"
    );
    assert_legal(&kernel);
}

#[test]
fn every_workload_in_the_suite_admits_deterministically() {
    // I22 is part of `invariants::check`, so this is really a check that the
    // seam is wired: a kernel that ran real work reports no violation, and the
    // clause is evaluated rather than skipped.
    let mut kernel = contended_kernel();
    kernel.run_to_quiescence(64);
    assert!(
        !kernel.admission_log().is_empty(),
        "the run must have admitted something for the clause to mean anything"
    );
    assert_legal(&kernel);
}

// ---- the null: the check is not vacuous ----------------------------------

#[test]
fn the_workload_really_makes_mutable_continuations_compete() {
    let mut kernel = contended_kernel();
    kernel.run_to_quiescence(64);

    let contested = kernel
        .admission_log()
        .iter()
        .filter(|record| {
            let mut mutable_per_process: std::collections::BTreeMap<u32, usize> =
                std::collections::BTreeMap::new();
            for candidate in record.candidates.iter() {
                if candidate.state_access == StateAccess::Mutable {
                    *mutable_per_process
                        .entry(candidate.process.slot)
                        .or_default() += 1;
                }
            }
            mutable_per_process.values().any(|n| *n > 1)
        })
        .count();

    assert!(
        contested > 0,
        "no epoch had two mutable continuations of one process, so I22 was checked \
         against a decision that had nothing to decide"
    );
    assert!(
        kernel.accounting().serial_deferrals > 0,
        "no candidate was ever deferred by the I13 claim"
    );
}

#[test]
fn permutation_really_reorders_the_candidates() {
    // The checker's permutations are fixed and deterministic; if they ever
    // degenerated to the identity every I22 check would pass for free.
    let candidates = vec![
        candidate(10, 1, StateAccess::Mutable, 0),
        candidate(11, 1, StateAccess::Mutable, 0),
        candidate(12, 2, StateAccess::ReadOnly, 0),
    ];
    let observed = std::cell::RefCell::new(Vec::new());
    let _ = decision_is_order_independent(&candidates, |offered| {
        observed.borrow_mut().push(
            offered
                .iter()
                .map(|c| c.continuation.slot)
                .collect::<Vec<_>>(),
        );
        admit(offered).decision()
    });
    let observed = observed.into_inner();
    assert!(
        observed.iter().any(|order| *order != vec![10, 11, 12]),
        "the checker never presented the candidates in a different order"
    );
}

// ---- the fault injection: the rule this replaced fails --------------------

/// Admission as it was written before v0.3 §4: claim the process slot as the
/// scan reaches it, first one wins.
fn first_come_decision(candidates: &[Candidate]) -> Decision {
    let mut claimed: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut admitted = Vec::new();
    let mut deferred = Vec::new();
    for c in candidates {
        if c.state_access == StateAccess::Mutable && !claimed.insert(c.process.slot) {
            deferred.push(c.continuation.to_u64());
        } else {
            admitted.push(c.continuation.to_u64());
        }
    }
    admitted.sort_unstable();
    deferred.sort_unstable();
    (admitted, deferred)
}

#[test]
fn first_come_admission_is_order_dependent() {
    let candidates = vec![
        candidate(10, 1, StateAccess::Mutable, 0),
        candidate(11, 1, StateAccess::Mutable, 0),
    ];
    assert_eq!(
        decision_is_order_independent(&candidates, |offered| admit(offered).decision()),
        Vec::new(),
        "the state-derived rule must survive its own check"
    );
    assert!(
        !decision_is_order_independent(&candidates, first_come_decision).is_empty(),
        "a first-come claim decides by position and must be rejected"
    );
}

#[test]
fn an_epoch_whose_decision_does_not_follow_from_its_candidates_is_reported() {
    // The other half of the clause. Half two says the rule is order-independent;
    // this says the epoch's decision follows from its candidates. In-crate the
    // question is settled by construction — `Admission` is sealed, so an epoch
    // that claimed inline would not compile — but the record is what makes the
    // clause askable of an implementation this crate did not run, and a check
    // that cannot fail is not a check.
    let mut kernel = contended_kernel();
    kernel.run_epoch();
    assert_eq!(admission_determinism(&kernel), Vec::new());

    {
        let log = unsafe { soma::kernel::raw::state(&mut kernel) }.admission_log;
        let record = log.first_mut().expect("the epoch admitted work");
        assert!(
            record.candidates.len() > 1,
            "the epoch must have offered more than one candidate"
        );
        record.candidates.pop();
    }
    assert!(
        !admission_determinism(&kernel).is_empty(),
        "a decision that does not follow from the epoch's candidates must be reported"
    );
}

// ---- the decision itself --------------------------------------------------

#[test]
fn the_longest_waiting_mutable_continuation_wins_the_claim() {
    // Identity alone would be deterministic and unfair: slot 10 would win every
    // epoch and slot 11 would starve until I21 reported it. The waiting term is
    // what stops that, so it gets a case where the two disagree.
    let candidates = vec![
        candidate(10, 1, StateAccess::Mutable, 5),
        candidate(11, 1, StateAccess::Mutable, 2),
    ];
    let decided = admit(&candidates);
    assert_eq!(
        decided.deferred(),
        [(SEARCH_BRANCH, Ref64::new(10, 1, Kind::Continuation))],
        "the continuation that has waited longer must take the claim"
    );
}

#[test]
fn identity_breaks_a_tie_between_equally_starved_continuations() {
    let candidates = vec![
        candidate(11, 1, StateAccess::Mutable, 3),
        candidate(10, 1, StateAccess::Mutable, 3),
    ];
    let decided = admit(&candidates);
    assert_eq!(
        decided.deferred(),
        [(SEARCH_BRANCH, Ref64::new(11, 1, Kind::Continuation))],
        "with nothing else to separate them, the lower identity wins"
    );
}

#[test]
fn read_only_continuations_of_one_process_are_never_deferred() {
    let candidates = vec![
        candidate(10, 1, StateAccess::ReadOnly, 0),
        candidate(11, 1, StateAccess::ReadOnly, 0),
        candidate(12, 1, StateAccess::Mutable, 0),
    ];
    let decided = admit(&candidates);
    assert!(
        decided.deferred().is_empty(),
        "I13 serialises mutable declarations only"
    );
    let (admitted, _) = decided.decision();
    assert_eq!(admitted.len(), 3);
}

#[test]
fn a_deferred_claim_loser_runs_in_the_following_epoch() {
    // The end-to-end shape of the same fairness property: deferral delays work,
    // it does not withhold it (I21).
    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let first = spawn_leaf(&mut kernel, process, StateAccess::Mutable);
    let second = spawn_leaf(&mut kernel, process, StateAccess::Mutable);

    kernel.run_epoch();
    assert_eq!(
        kernel.continuation_state(first).unwrap(),
        ContinuationState::Completed
    );
    assert_eq!(
        kernel.continuation_state(second).unwrap(),
        ContinuationState::Runnable
    );

    kernel.run_epoch();
    assert_eq!(
        kernel.continuation_state(second).unwrap(),
        ContinuationState::Completed
    );
    assert_legal(&kernel);
}
