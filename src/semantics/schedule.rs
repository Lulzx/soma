//! **I22. Admission determinism.**
//!
//! `docs/SOMA-v0.3.md` §4 lists four things a device-resident scheduler must
//! preserve. The first is admission: I13's per-process mutable claim is taken
//! with a host `HashSet` as the bins are scanned, and on a device that claim is
//! concurrent, so it cannot be "whoever gets there first".
//!
//! This module states the property that replaces "whoever gets there first" and
//! checks it: **the admission decision is a function of the epoch's candidate
//! set, not of the order that set is presented in.** A host scanner and a
//! device claiming the same work concurrently then decide the same thing, and
//! the reference run stays the reference under both.
//!
//! ## What it does not say
//!
//! It constrains *which* continuations run, not the order they run in.
//! `Admission::decision` projects lane order out on purpose. Lane order within
//! a bin is arrival order — a real scheduling policy, and observable, since two
//! lanes receiving from one mailbox get different messages depending on which
//! goes first. A device gets that order only if it commits an epoch's effects
//! in canonical lane order, which is a separate obligation B has not
//! discharged. Folding lane order into I22 would state a property the host
//! satisfies and no device could, which is the failure mode v0.3 §2 removed
//! from the equivalence relation; leaving it out here keeps the clause honest
//! rather than aspirational.
//!
//! ## Why the clause has two halves
//!
//! `admit` is short enough to read and believe. So was the first pairing scan
//! in `semantics::order`, which was wrong in a way that made its checker
//! silently accept inverted deliveries. A rule that decides from state is easy
//! to write and just as easy to regress into one that decides from position:
//! the regression compiles, passes every behavioural test, and only shows up on
//! hardware. Running the rule over permutations of its own input is the cheapest
//! thing that fails when that happens.
//!
//! But a checker that only exercises `admit` says nothing about the scheduler.
//! An implementation that took its own first-come claim inline — which is
//! exactly what a device is tempted to do, and exactly what the host used to do
//! — would leave such a checker green while breaking the property it is named
//! after. So the epoch records the decision it took, and the first half of the
//! clause is that the decision matches the rule.

use crate::abi::Ref64;
use crate::kernel::Kernel;
use crate::scheduler::admission::{admit, Candidate, Decision};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScheduleClause {
    /// I22. Admission decides from the candidate set, not from its order.
    AdmissionDeterminism,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScheduleViolation {
    pub clause: ScheduleClause,
    pub detail: String,
}

impl ScheduleViolation {
    fn new(clause: ScheduleClause, detail: impl Into<String>) -> Self {
        Self {
            clause,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ScheduleViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.clause, self.detail)
    }
}

/// Check I22 for every epoch of a run, in two halves.
///
/// 1. **The run decided what the rule decides.** Re-running `admit` over the
///    epoch's candidates must reproduce the decision the epoch actually took.
///    Without this the clause would be about a function nobody is obliged to
///    call: a scheduler that inlined a first-come claim would pass a checker
///    that only exercises `admit`, which is the same way `semantics::order`'s
///    first pairing scan managed to accept inverted deliveries.
/// 2. **The rule decides from the set.** Permuting the candidates must not
///    change the decision.
///
/// Both halves run over every epoch, not the last one. An admission rule that
/// is order-dependent for one arrangement of one epoch's work is order-dependent.
pub fn admission_determinism(kernel: &Kernel) -> Vec<ScheduleViolation> {
    let mut out = Vec::new();
    for (epoch, record) in kernel.admission_log().iter().enumerate() {
        let specified = admit(&record.candidates).decision();
        let taken = record.decision.decision();
        if specified != taken {
            out.push(ScheduleViolation::new(
                ScheduleClause::AdmissionDeterminism,
                format!(
                    "epoch {epoch}: admitted {} and deferred {} where the rule specifies {} and {}",
                    taken.0.len(),
                    taken.1.len(),
                    specified.0.len(),
                    specified.1.len(),
                ),
            ));
        }
        for mut violation in
            decision_is_order_independent(&record.candidates, |offered| admit(offered).decision())
        {
            violation.detail = format!("epoch {epoch}: {}", violation.detail);
            out.push(violation);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Whether `rule` decides the same thing however its input is ordered.
///
/// Generic in the rule so that the negative case has something to test: a
/// first-come claim is a legal `Fn(&[Candidate]) -> Decision` and this must
/// reject it. The rule returns a `Decision` rather than an `Admission` because
/// `Admission` is sealed — only `admit` can build one — and a checker that
/// could only be handed the answer it is checking would have nothing to fail
/// on.
pub fn decision_is_order_independent<F>(candidates: &[Candidate], rule: F) -> Vec<ScheduleViolation>
where
    F: Fn(&[Candidate]) -> Decision,
{
    let mut out = Vec::new();
    if candidates.len() < 2 {
        return out;
    }
    let (expected_admitted, expected_deferred) = rule(candidates);

    for permutation in permutations(candidates.len()) {
        let reordered: Vec<Candidate> = permutation.iter().map(|i| candidates[*i]).collect();
        let (admitted, deferred) = rule(&reordered);
        if admitted != expected_admitted || deferred != expected_deferred {
            out.push(ScheduleViolation::new(
                ScheduleClause::AdmissionDeterminism,
                format!(
                    "reordering the candidate set changed the decision: {} admitted / {} deferred \
                     became {} admitted / {} deferred; first disagreement at {}",
                    expected_admitted.len(),
                    expected_deferred.len(),
                    admitted.len(),
                    deferred.len(),
                    first_disagreement(&expected_admitted, &admitted)
                        .map(|c| format!("continuation {}", Ref64::from_u64(c).slot))
                        .unwrap_or_else(|| "the deferred set".to_string()),
                ),
            ));
            break;
        }
    }
    out
}

fn first_disagreement(expected: &[u64], actual: &[u64]) -> Option<u64> {
    expected
        .iter()
        .find(|c| !actual.contains(c))
        .or_else(|| actual.iter().find(|c| !expected.contains(c)))
        .copied()
}

/// A fixed family of permutations of `0..n`: reverse, short rotations, and a
/// few shuffles from a fixed-seed generator.
///
/// Deterministic on purpose. A checker seeded from the clock reports a failure
/// the next run cannot reproduce, which in a suite this size means the failure
/// gets rerun away rather than fixed.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();

    out.push((0..n).rev().collect());
    for shift in 1..n.min(8) {
        out.push((0..n).map(|i| (i + shift) % n).collect());
    }
    for seed in 1..=8u64 {
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(i, (state >> 33) as usize % (i + 1));
        }
        out.push(order);
    }

    out
}
