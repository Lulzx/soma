//! Admission (§18 Phase C; `docs/SOMA-v0.3.md` §4).
//!
//! Admission decides which of an epoch's runnable continuations run and which
//! wait. The only thing it decides is I13's claim: at most one continuation
//! declaring `Mutable` access to a process's canonical state runs per process
//! per epoch.
//!
//! The host used to decide that with a `HashSet` claim taken as it scanned the
//! bins — whoever was met first won. That is a decision for a scanner and no
//! decision at all for a device, where an epoch's runnable set is claimed
//! concurrently and "met first" is a race. v0.3 §4 names it as B's first
//! obligation: admission must be deterministic, so it cannot be whoever gets
//! there first.
//!
//! `admit` decides from *state* instead of from position. It is a pure function
//! of the candidate set, and its decision is invariant under permutation of
//! that set — which is I22, checked in `semantics::schedule`.
//!
//! **What this does not settle.** The decision is order-independent; the lane
//! *order* within a bin is still the order the candidates arrived in, because
//! arrival order is a real scheduling policy and not an artifact. A device
//! whose bins are appended concurrently does not get that order for free: it
//! has to commit an epoch's effects in canonical lane order so the next epoch's
//! bins are built in a reproducible one. That is a separate obligation, and B
//! has not discharged it — see `docs/SOMA-v0.3.md` §4.

use std::collections::BTreeMap;

use crate::abi::{Ref64, StateAccess};

/// One runnable continuation offered to admission, with everything the decision
/// depends on. Nothing here is a position: two candidate lists holding the same
/// candidates decide identically however they are ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The bin the continuation was drained from.
    pub bin: u32,
    pub continuation: Ref64,
    pub process: Ref64,
    pub run_class: u32,
    pub state_access: StateAccess,
    /// The epoch since which this continuation has been runnable. Same quantity
    /// I21 measures starvation against.
    pub waiting_since: u32,
}

impl Candidate {
    /// Priority for the per-process mutable claim: least key wins.
    ///
    /// Longest-waiting first, ties broken by continuation identity. Identity
    /// alone would be deterministic but not fair — a process whose lower-slot
    /// mutable continuation stays runnable would win every epoch and the other
    /// would never run, which I21's starvation bound would eventually report as
    /// a violation the scheduler chose.
    ///
    /// The scan rule got fairness right by accident: a deferred continuation is
    /// re-enqueued during admission, ahead of the appends this epoch's commits
    /// make, so it led the next epoch's bin. Once bin order is no longer
    /// trustworthy that accident is gone, so the property becomes part of the
    /// key.
    fn claim_key(&self) -> (u32, u64) {
        (self.waiting_since, self.continuation.to_u64())
    }

    fn is_mutable(&self) -> bool {
        self.state_access == StateAccess::Mutable
    }
}

/// Which continuations an admission ran and which it held back, as sorted
/// identities. Lane order is projected out — that is what makes this the
/// *decision* rather than the schedule.
pub type Decision = (Vec<u64>, Vec<u64>);

/// What admission decided.
///
/// The fields are private and the only constructor is [`admit`]. That is not
/// tidiness: an epoch that builds its own `Admission` is an epoch that took its
/// own claim, which is precisely what I22 exists to forbid, and a checker
/// cannot reliably tell the two apart on a run where the rules happen to agree
/// — as they do on the reference interpreter's discovery order. Sealing the
/// type turns "the scheduler ran the rule" from something a test hopes to catch
/// into a compile error, the same way `docs/SOMA-CAPABILITIES.md` closes the
/// operation set by making kernel state private.
///
/// ```compile_fail
/// let claim = soma::scheduler::admission::Admission {
///     bins: Vec::new(),
///     deferred: Vec::new(),
/// };
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Admission {
    bins: Vec<(u32, Vec<(Ref64, u32)>)>,
    deferred: Vec<(u32, Ref64)>,
}

impl Admission {
    /// Admitted lanes per bin, in bin order. Within a bin, lanes keep the order
    /// the candidates arrived in — that is the arrival order cohorting groups.
    pub fn bins(&self) -> &[(u32, Vec<(Ref64, u32)>)] {
        &self.bins
    }

    /// Continuations held back by the I13 claim, as `(run_class, continuation)`
    /// so they can be returned to their bins.
    pub fn deferred(&self) -> &[(u32, Ref64)] {
        &self.deferred
    }

    /// Consume the admission, yielding its bins.
    pub fn into_bins(self) -> Vec<(u32, Vec<(Ref64, u32)>)> {
        self.bins
    }

    /// The decision proper: which continuations run and which wait, with lane
    /// order projected out.
    ///
    /// This is what I22 constrains. Comparing whole `Admission` values instead
    /// would fold in lane order, and lane order legitimately follows the order
    /// candidates were discovered in.
    pub fn decision(&self) -> Decision {
        let mut admitted: Vec<u64> = self
            .bins
            .iter()
            .flat_map(|(_, lanes)| lanes.iter().map(|(c, _)| c.to_u64()))
            .collect();
        let mut deferred: Vec<u64> = self.deferred.iter().map(|(_, c)| c.to_u64()).collect();
        admitted.sort_unstable();
        deferred.sort_unstable();
        (admitted, deferred)
    }
}

/// One epoch's admission, as it actually happened.
///
/// The decision is recorded next to the candidates it was made from, not just
/// derived from them on demand, so I22 can ask whether this epoch decided what
/// the rule says it should have. In-crate that question is already answered by
/// `Admission` being sealed; the record is what lets it be asked of an
/// implementation whose admission this crate did not run — which is the case
/// the clause is ultimately for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionRecord {
    pub candidates: Vec<Candidate>,
    pub decision: Admission,
}

/// Decide an epoch's admission from its candidate set.
///
/// Two passes, and the second is the point: the claim winner is chosen from the
/// whole candidate set before any candidate is admitted, so no candidate's fate
/// depends on how far the scan had got when it was reached.
pub fn admit(candidates: &[Candidate]) -> Admission {
    // Pass 1: the mutable claim, per process. `claim_key` is a total order over
    // distinct continuations, so the winner is a function of the set.
    let mut claim: BTreeMap<u64, Candidate> = BTreeMap::new();
    for candidate in candidates.iter().filter(|c| c.is_mutable()) {
        claim
            .entry(candidate.process.key())
            .and_modify(|best| {
                if candidate.claim_key() < best.claim_key() {
                    *best = *candidate;
                }
            })
            .or_insert(*candidate);
    }

    // Pass 2: place every candidate. Read-only continuations of one process may
    // share an epoch — I13 serialises only mutable ones.
    let mut bins: BTreeMap<u32, Vec<(Ref64, u32)>> = BTreeMap::new();
    let mut deferred = Vec::new();
    for candidate in candidates {
        let won = !candidate.is_mutable()
            || claim
                .get(&candidate.process.key())
                .is_some_and(|winner| winner.continuation == candidate.continuation);
        if won {
            bins.entry(candidate.bin)
                .or_default()
                .push((candidate.continuation, candidate.run_class));
        } else {
            deferred.push((candidate.run_class, candidate.continuation));
        }
    }

    Admission {
        bins: bins.into_iter().collect(),
        deferred,
    }
}
