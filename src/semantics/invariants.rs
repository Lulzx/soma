//! Executable well-formedness invariants for the SOMA abstract machine.
//!
//! These are the machine-checked half of the semantic specification in
//! `docs/SOMA-v0.2.md`. Every invariant here has an identifier (`I1`, `I2`, …)
//! that matches a numbered clause in that document, so the prose and the code
//! cannot drift apart without a test failing.
//!
//! The checker is a predicate over a whole machine state, not over a
//! transition. It answers "is this a legal state", which is what makes it
//! usable as a postcondition after *any* transition: run it after every epoch
//! and any rule that can produce an illegal state gets caught, without having
//! to anticipate which rule.
//!
//! Capability safety is split into structural attenuation/integrity checks and
//! a trace-level effect check. The latter rejects every governed effect that is
//! not immediately paired with the matching successful authority decision.

use crate::abi::continuations::ContinuationState;
use crate::abi::{CollectiveState, FutureState, Kind, ProcessState, Ref64, StateAccess};
use crate::kernel::Kernel;

/// Which clause of the specification a violation belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Invariant {
    /// I1. Every reference held in a live descriptor resolves.
    ReferenceIntegrity,
    /// I2. No continuation is left mid-execution at a quiescent state.
    NoContinuationLeftRunning,
    /// I3. A continuation's process is live; a terminated process has no
    /// schedulable continuations.
    ProcessContinuationConsistency,
    /// I4. Futures are single-assignment, and nothing waits on a settled one.
    FutureSingleAssignment,
    /// I5. Mailboxes respect their declared bound.
    MailboxBound,
    /// I6. Messages from one sender to one receiver stay in send order.
    MessageOrdering,
    /// I7. Everything in a runnable bin is live, runnable, and correctly binned.
    SchedulerWellFormed,
    /// I8. No two continuations share a frame object.
    FrameExclusivity,
    /// I10a. Derived capabilities never amplify rights or byte range.
    CapabilityAttenuation,
    /// I10b. Capability targets and parent links resolve with valid rights.
    CapabilityIntegrity,
    /// I10c. Every governed effect immediately follows matching authority.
    NoUnauthorizedEffect,
    /// I11. The trace is a strictly increasing logical clock.
    TraceMonotonicity,
    /// I12. Accounting counters are mutually consistent.
    AccountingConsistency,
    /// I13. At most one mutable process-state continuation starts per epoch.
    SerialProcessExecution,
    /// I15. Supervision links and queued exit notices are structurally sound.
    SupervisionIntegrity,
    /// I16. Domain membership/quotas and attached contracts are valid.
    DomainContractIntegrity,
    /// I17. Loaded module manifests and linked evaluator uses agree.
    ModuleIntegrity,
    /// I21. Admitted work is dispatched, and no runnable continuation waits
    /// longer than the declared deferral bound.
    BoundedProgress,
    /// I22. Admission decides from the epoch's candidate set, not from the
    /// order that set is discovered in.
    AdmissionDeterminism,
    /// I23. The trace's order is recoverable from event positions alone, with
    /// no shared clock.
    PositionDerivedEmission,
    /// I24. Every runnable-bin entry is an effect a step produced and the
    /// kernel applied, in the order the plan puts the producing lanes in.
    EffectMediatedCommit,
    /// I25. No lane of an epoch observes another lane of the same epoch.
    LaneIndependence,
}

/// A specific way in which a state was illegal.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Violation {
    pub invariant: Invariant,
    pub detail: String,
}

impl Violation {
    fn new(invariant: Invariant, detail: impl Into<String>) -> Violation {
        Violation {
            invariant,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.invariant, self.detail)
    }
}

/// Check every invariant, returning all violations rather than the first, so a
/// broken transition reports its full damage in one pass.
pub fn check(kernel: &Kernel) -> Vec<Violation> {
    let mut v = Vec::new();
    reference_integrity(kernel, &mut v);
    no_continuation_left_running(kernel, &mut v);
    process_continuation_consistency(kernel, &mut v);
    future_single_assignment(kernel, &mut v);
    mailbox_bound(kernel, &mut v);
    message_ordering(kernel, &mut v);
    scheduler_well_formed(kernel, &mut v);
    frame_exclusivity(kernel, &mut v);
    capability_attenuation(kernel, &mut v);
    capability_integrity(kernel, &mut v);
    no_unauthorized_effect(kernel, &mut v);
    trace_monotonicity(kernel, &mut v);
    accounting_consistency(kernel, &mut v);
    serial_process_execution(kernel, &mut v);
    supervision_integrity(kernel, &mut v);
    domain_contract_integrity(kernel, &mut v);
    module_integrity(kernel, &mut v);
    bounded_progress(kernel, &mut v);
    admission_determinism(kernel, &mut v);
    position_derived_emission(kernel, &mut v);
    effect_mediated_commit(kernel, &mut v);
    lane_independence(kernel, &mut v);
    v.sort();
    v
}

// ---- I24 -----------------------------------------------------------------

/// **I24. Effect-mediated commit.**
///
/// `docs/SOMA-v0.3.md` §4.3 names the obstacle to canonical commit: handlers
/// take `&mut Kernel` and allocate their effects as they run, so execute and
/// commit are fused, and an epoch's lanes therefore touch shared state in
/// whatever order they were scheduled in. §4.4 unfuses the part every lane
/// writes — entry into a runnable bin, which v0.2 §3.4 already makes commit's
/// exclusive right. A step now *produces* its bin entries and the kernel
/// applies them.
///
/// Three clauses:
///
/// 1. **Nothing is applied twice and nothing is lost.** The applied indices are
///    exactly `0..n`, each once.
/// 2. **Application order is plan order.** Sorting the log by the position the
///    effect was produced at — its epoch, its lane, and that lane's own count —
///    puts the applied indices in increasing order. This is what "in lane
///    order" means as a property of a record: a lane number is a position in
///    the plan (§4.2), so the clause is independent of when a lane ran.
/// 3. **No bin entry arrived any other way.** The scheduler counts every entry
///    it has ever made; the log accounts for all of them.
///
/// **Clauses 1 and 2 are, on this interpreter, satisfied by construction**, in
/// the same way and for the same reason as I23's clause 2: a sequential applier
/// draining one lane's journal at a time cannot produce an out-of-order log. As
/// with I22's first half, the record is not there to catch this crate. It is
/// there so the clause can be asked of an implementation whose lanes append
/// concurrently, where the log's row order is *not* its application order and
/// the question has content. `kernel::raw` supplies the failing case in the
/// meantime.
///
/// Clause 3 is the one with teeth here, and it is the runtime half of a
/// compile-time guarantee: `Scheduler::enqueue` demands a `Committing` token
/// that only `kernel::effects` can build, so an unmediated bin write does not
/// compile outside `kernel::raw`. That is the technique §4.1 used to seal
/// `Admission`; the count is what carries it to an implementation this crate
/// did not compile.
///
/// **What the clause does not say.** Only bin entry is mediated. Mailboxes,
/// futures, capability spaces and the object tables are still written as the
/// step runs, and allocation is still eager — §4.3 (2) shows it has to be. The
/// applier does now run at the epoch boundary rather than at the end of each
/// lane (§4.5), so "effects are applied after the epoch" is true of the effects
/// this clause covers and of no others. I25 is the clause that says what the
/// rest of a step's writes owe in exchange.
fn effect_mediated_commit(kernel: &Kernel, out: &mut Vec<Violation>) {
    let log = kernel.effect_log();

    // Clause 1.
    let mut applied: Vec<u64> = log.iter().map(|record| record.applied).collect();
    applied.sort_unstable();
    for (expected, actual) in applied.iter().enumerate() {
        if *actual != expected as u64 {
            out.push(Violation::new(
                Invariant::EffectMediatedCommit,
                format!(
                    "the effect log applies index {actual} where {expected} is missing, so an \
                     effect was applied twice or not at all"
                ),
            ));
            break;
        }
    }

    // Clause 2. Positions are compared, not the rows: a concurrent applier's
    // log is appended in finish order, and what it owes is that the sort by
    // position recovers the order it applied in.
    let mut by_position: Vec<&crate::kernel::effects::EffectRecord> = log.iter().collect();
    by_position.sort_by_key(|record| record.position());
    for window in by_position.windows(2) {
        if window[0].position() == window[1].position() {
            let (epoch, lane, sequence) = window[0].position();
            out.push(Violation::new(
                Invariant::EffectMediatedCommit,
                format!(
                    "two effects were produced at (epoch {epoch}, lane {lane}, sequence \
                     {sequence}), so the order to apply them in is not determined"
                ),
            ));
            break;
        }
        if window[0].applied > window[1].applied {
            let (epoch, lane, sequence) = window[0].position();
            let (next_epoch, next_lane, next_sequence) = window[1].position();
            out.push(Violation::new(
                Invariant::EffectMediatedCommit,
                format!(
                    "the effect produced at (epoch {epoch}, lane {lane}, sequence {sequence}) \
                     was applied at {} but the one at (epoch {next_epoch}, lane \
                     {next_lane}, sequence {next_sequence}) was applied at {}, so commit \
                     did not run in plan order",
                    window[0].applied, window[1].applied
                ),
            ));
            break;
        }
    }

    // Clause 3.
    let admissions = kernel.scheduler().admissions();
    if admissions != log.len() as u64 {
        out.push(Violation::new(
            Invariant::EffectMediatedCommit,
            format!(
                "{admissions} continuation(s) entered a runnable bin but the effect log \
                 accounts for {}, so commit is not the only path into one",
                log.len()
            ),
        ));
    }
}

// ---- I25 -----------------------------------------------------------------

/// **I25. Lane independence.** Two clauses:
///
/// 1. no ≺ edge joins two distinct lanes of one epoch; and
/// 2. no two lanes of one epoch draw on the same bounded domain.
///
/// Clause 2 is not a second way of saying clause 1, and it was added because
/// clause 1 alone accepted a run whose lanes are demonstrably not reorderable.
/// A step creating a process consumes its domain's quota. Two lanes of an epoch
/// creating processes in one bounded domain therefore race for it: the same
/// workload run under `LaneOrder::Plan` and `LaneOrder::Reverse` faults a
/// *different* pair of processes, and both runs leave a legal state, so nothing
/// but a comparison of the two runs reports it. Clause 1 does not, because it
/// reads the semantic order and the dependence carries no ≺ edge — nothing is
/// delivered, resolved or woken. It is carried by a counter.
///
/// That is the general shape of what clause 1 misses: a dependence through
/// *state* rather than through an event. A quota was the first case; a mailbox's
/// capacity, a mailbox's occupancy, a future's one assignment and a future's
/// settled state followed (§4.12–§4.15), each found by asking which results of
/// the operations a step can perform another lane can decide. Anything else with
/// that property belongs here as it arrives.
///
/// The clause reads `ProcessCreated`, whose `subject` is the domain the process
/// was allocated in, so it survives the process being reclaimed. `HOST_LANE` is
/// excluded for clause 1's reason: the host's allocations are the plan's, and
/// they run strictly before or after the lanes rather than beside them.
///
/// This is the invariant canonical commit is paid for with, and it is worth
/// being precise about which direction that goes. Applying an epoch's effects
/// at the epoch boundary rather than at the end of each lane does not *make*
/// lanes independent — mailboxes, futures, capability spaces and the object
/// tables are still written as a step runs, and §4.3 (2) shows allocation has
/// to stay eager. What it does is remove the executive's own reason for lanes
/// to have run in a particular order, leaving the workload's. I25 is the
/// question of whether the workload has one.
///
/// `docs/SOMA-v0.3.md` §4.3 (3) measured the answer across the Expand workload
/// at three cohort widths — 1025 events, 441 edges, no cross-lane edge — and
/// gave the structural reason: the wake events (`MessageReceived`,
/// `ContinuationReady`, `ChannelReceived`) are emitted by the *acting* lane, so
/// a delivery edge is either inside one lane or across epochs.
///
/// **That reason is sound for delivery edges and does not cover program-order
/// ones**, which §4.15 is the workload that shows. A wake is emitted by the
/// acting lane and names the *woken* continuation, so it joins that
/// continuation's own history — and if the woken lane parked earlier in the
/// same epoch, the ContinuationProgram edge between its park and its wake spans
/// two lanes. `cross_lane_edges()` is non-empty there, which is the first time
/// in five races it has been. The measurement stands; what it measured was a
/// workload in which nothing parked and was woken within one epoch.
///
/// §4.3 then
/// declined to call this an invariant, correctly, because at the time nothing
/// depended on it: the applier ran per lane, so a run with a cross-lane edge
/// was merely a run whose lanes could not be reordered.
///
/// The applier now runs once per epoch, so such a run is a run this executive
/// commits differently from the order its lanes actually observed each other
/// in. That is the difference between a measurement and a requirement, and it
/// is why the clause moves here rather than staying prose.
///
/// **What it does not claim.** It is a property of a *run*, not of the model —
/// exactly the standing §4.3 gave it. A workload that drives `channel_send`
/// from one lane and receives it in a later lane of the same epoch is still
/// expressible, and I25 reports it rather than the kernel refusing it. The
/// report is the useful outcome: it names the workload the concurrent executive
/// cannot take, at the point the workload does it, rather than leaving it to
/// surface as a nondeterministic result on hardware.
///
/// Edges touching `HOST_LANE` are excluded and are not an exemption. The host's
/// part of an epoch — admission's deferrals, the cohort records, the deferred
/// lanes — runs strictly before or strictly after the lanes, so an order
/// between it and a lane is the plan's and not a race between two things that
/// could have gone either way.
fn lane_independence(kernel: &Kernel, out: &mut Vec<Violation>) {
    let order = crate::semantics::order::SemanticOrder::of(kernel);
    let events = order.events();
    for edge in order.cross_lane_edges() {
        let (Some(a), Some(b)) = (events.get(edge.earlier), events.get(edge.later)) else {
            continue;
        };
        out.push(Violation::new(
            Invariant::LaneIndependence,
            format!(
                "epoch {} lane {} is ordered before lane {} by {:?}, so one lane of the epoch \
                 observed another and the epoch's lanes are not reorderable",
                a.epoch, a.lane, b.lane, edge.reason
            ),
        ));
        // One report per run. The edges come in families — a chatty workload
        // produces one per delivery — and a hundred lines all naming the same
        // defect buries the other invariants' reports.
        break;
    }
    bounded_resource_independence(kernel, out);
}

/// Which thing an epoch's lanes were decided by.
///
/// The five the machine has. A step can exhaust a domain's process quota, it
/// can fill a receiver's mailbox, it can take the message another lane was about
/// to receive, it can publish the one value a future accepts, and it can find
/// that future already published and not park. They were found by walking the
/// operations a step can perform, which is a finite list because `LaneView`
/// closes it (§4.10), and asking of each which of its results another lane can
/// decide. The first four rounds asked the narrower question — which of them
/// can say *no* — and the fifth is what showed that to be a special case
/// (§4.15): `await_future` never fails and is decided all the same.
///
/// The third is the second one read from the other end, and it is a separate
/// variant because it is contended differently rather than merely elsewhere. A
/// capacity and a quota hand out **interchangeable** units: a slot in a mailbox
/// is like every other slot, and which one a sender gets is not in the trace. An
/// occupancy hands out **identified** ones — this message, from that sender,
/// with that sequence number — so two lanes that both succeed still got
/// different things, and which lane got which is decided by the order.
///
/// That is why `Dispenses` exists rather than one condition for all three. It
/// is also the first place clause 2's original condition, "a winner and a
/// different loser", turned out to be a statement about the two resources that
/// happened to exist when it was written.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Bounded {
    DomainQuota,
    MailboxCapacity,
    MailboxOccupancy,
    /// A future's one assignment (§12). The bound is one, which is why it is
    /// the case that does not discriminate between the two conditions below:
    /// with at most one winner ever, "any other lane that drew" and "a lane
    /// that was refused" are the same set. It is called interchangeable
    /// because what a resolver gets is the permission to publish and there is
    /// nothing to tell one such permission from another — not because the run
    /// could not tell the difference.
    FutureAssignment,
    /// The *state* of that same future, read by a lane that is not writing it
    /// (v0.3 §4.15). An `await_future` parks or continues depending on whether
    /// the value has been published, and a resolving lane of the same epoch
    /// decides which.
    ///
    /// This is the first entry that is not a bounded resource at all. Nothing
    /// is dispensed and the awaiter draws nothing: it reads which of two states
    /// the future is in. It is in the same clause because the clause's subject
    /// was never really boundedness — it is a lane's outcome being decided by
    /// another lane of its epoch, and a refusal was only the first way that was
    /// found to happen.
    FutureSettlement,
}

/// What a resource hands a lane that draws on it successfully.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dispenses {
    /// Units no run can tell apart. Two lanes that both succeed are not decided
    /// by their order, so contention needs a refusal: a winner and a *different*
    /// loser.
    Interchangeable,
    /// Distinct items with identities of their own. Two lanes that both succeed
    /// took different items, so a second lane drawing at all is enough — a
    /// refusal is one way to lose and getting the other message is another.
    Identified,
}

impl Bounded {
    fn dispenses(&self) -> Dispenses {
        match self {
            Bounded::DomainQuota
            | Bounded::MailboxCapacity
            | Bounded::FutureAssignment
            // A settled future hands the awaiter the same news whoever it is,
            // and there is only ever one lane resolving. The reading below is
            // "a winner and a lane decided against", which is what
            // interchangeable already means; a settled await is a lane decided
            // against without having drawn.
            | Bounded::FutureSettlement => Dispenses::Interchangeable,
            Bounded::MailboxOccupancy => Dispenses::Identified,
        }
    }

    fn describe(&self) -> &'static str {
        match self {
            Bounded::DomainQuota => {
                "allocated in one bounded domain, and it refused at least one, so which lane is \
                 refused"
            }
            Bounded::MailboxCapacity => {
                "sent to one mailbox, and it was full for at least one, so which lane is refused"
            }
            Bounded::MailboxOccupancy => {
                "received from one mailbox, so which lane gets which message"
            }
            Bounded::FutureAssignment => {
                "resolved one future, and it takes one value, so which lane's value it takes"
            }
            Bounded::FutureSettlement => {
                "touched one future, and one of them awaited it, so whether that await parks"
            }
        }
    }
}

/// I25 clause 2: no two lanes of one epoch are decided by one bounded resource.
///
/// The condition depends on what the resource hands out, and that is a
/// correction rather than a complication.
///
/// For a resource dispensing **interchangeable** units — a domain's quota, a
/// mailbox's capacity, a future's one assignment — it is that **one lane got the resource and a different
/// lane was refused it**, in the same epoch. Each half is doing work. Two lanes
/// drawing on a resource with room for both decide nothing — every lane succeeds
/// under every order, and nothing in the trace says which slot each took. A
/// refusal with nobody succeeding decides nothing either: a mailbox that was
/// already full at the start of the epoch refuses all four of its senders
/// whatever order they run in, and the run is reorderable. So is one lane
/// refused after succeeding itself, which is a lane exhausting a resource
/// against nobody.
///
/// For a resource dispensing **identified** items — the messages in a mailbox —
/// a refusal is not required, and requiring one would miss the common case.
/// Four receivers and four messages is not "room for everyone": each lane takes
/// a different message, and which lane takes which is exactly what the order
/// decides. The condition there is a winner and any other lane that drew, won
/// or lost.
///
/// The distinction is not a special case for mailboxes. It is what "bounded
/// resource" was hiding: the original condition was written when both resources
/// counted units, and it silently assumed the units were anonymous.
///
/// The decision has to be in the trace for this to be checkable at all, which is
/// what `ProcessCreationRefused`, `MessageSendBlocked`, `MessageReceiveBlocked`,
/// `FutureResolutionRefused` and `FutureAwaitSettled` are for. Without them the
/// clause can only ask whether two lanes touched the resource, which is true of
/// a great many runs that are perfectly reorderable.
///
/// The last of the five is not a refusal, and `FutureSettlement` is not a
/// bounded resource. Both are the same correction: what this clause is about is
/// one lane's outcome being decided by another lane of its epoch, and a bounded
/// resource refusing somebody was the first mechanism found for that rather
/// than the definition of it (v0.3 §4.15).
///
/// **What is not this clause.** The counters themselves — `processes_created`
/// is incremented by every allocation, in the root domain as much as a bounded
/// one — are writes two lanes make and a journal fixes, because they commute
/// (v0.3 §4.12). What does not commute is the decision read off them.
fn bounded_resource_independence(kernel: &Kernel, out: &mut Vec<Violation>) {
    use std::collections::{HashMap, HashSet};
    type Key = (u32, Bounded, u64);
    // Per (epoch, resource): the lanes that got it, and the lanes that were
    // refused it.
    let mut won: HashMap<Key, HashSet<u32>> = HashMap::new();
    let mut lost: HashMap<Key, HashSet<u32>> = HashMap::new();

    for event in kernel.trace_events() {
        // `HOST_LANE` is excluded for clause 1's reason: the host's part of an
        // epoch runs strictly before or after the lanes, so an order between it
        // and a lane is the plan's and not a race.
        if event.lane == crate::abi::traces::HOST_LANE {
            continue;
        }
        // An event can draw on more than one of these — a resolve both takes
        // the future's one assignment and publishes the state an awaiter
        // reads — so this is a list rather than a tuple.
        let draws: Vec<(Bounded, crate::abi::Ref64, bool)> = match event.event_kind {
            // A domain names itself in `subject`; an unbounded one refuses
            // nobody, and a domain reclaimed since is treated as unbounded
            // rather than guessed at — the run it constrained is over.
            crate::abi::EventKind::ProcessCreated
            | crate::abi::EventKind::ProcessCreationRefused => {
                let bounded = kernel
                    .domains()
                    .get(event.subject)
                    .map(|d| d.max_processes != 0)
                    .unwrap_or(false);
                if !bounded {
                    continue;
                }
                vec![(
                    Bounded::DomainQuota,
                    event.subject,
                    event.event_kind == crate::abi::EventKind::ProcessCreationRefused,
                )]
            }
            // A mailbox is named by its receiver, which both events carry in
            // `causal`. Every mailbox is bounded, so there is no exemption.
            crate::abi::EventKind::MessageSent | crate::abi::EventKind::MessageSendBlocked => {
                vec![(
                    Bounded::MailboxCapacity,
                    event.causal,
                    event.event_kind == crate::abi::EventKind::MessageSendBlocked,
                )]
            }
            // A future takes one value and refuses every later write. Both
            // events name it in `causal`.
            crate::abi::EventKind::FutureResolved => vec![
                (Bounded::FutureAssignment, event.causal, false),
                // The publication an awaiter of the same epoch may read. A
                // resolve is the only way to win this one: nothing else moves
                // a future out of `Pending`.
                (Bounded::FutureSettlement, event.causal, false),
            ],
            crate::abi::EventKind::FutureResolutionRefused => {
                vec![(Bounded::FutureAssignment, event.causal, true)]
            }
            // The same future read rather than written. The awaiter is the
            // lane decided against — it found the value published and did not
            // park — and the resolver is the lane that decided it. A resolve
            // is therefore *both* a draw on the assignment and a write to the
            // state, and is counted under both keys, which is why this arm
            // sits after the one above rather than being merged into it.
            crate::abi::EventKind::FutureAwaitSettled => {
                vec![(Bounded::FutureSettlement, event.causal, true)]
            }
            // The same mailbox from the other end, where what is contended is
            // the messages in it rather than the room left. Both events name
            // the mailbox's owner in `process`.
            crate::abi::EventKind::MessageReceived
            | crate::abi::EventKind::MessageReceiveBlocked => {
                vec![(
                    Bounded::MailboxOccupancy,
                    event.process,
                    event.event_kind == crate::abi::EventKind::MessageReceiveBlocked,
                )]
            }
            _ => continue,
        };
        for (resource, target, is_refusal) in draws {
            let key = (event.epoch, resource, target.key());
            if is_refusal {
                lost.entry(key).or_default().insert(event.lane);
            } else {
                won.entry(key).or_default().insert(event.lane);
            }
        }
    }

    // A winner is required either way: a resource nobody got out of decided
    // nothing between the lanes that were refused it.
    let no_losers: HashSet<u32> = HashSet::new();
    let mut contended: Vec<(&Key, u32, u32)> = won
        .iter()
        .filter_map(|(key, winners)| {
            let (_, resource, _) = *key;
            let losers = lost.get(key).unwrap_or(&no_losers);
            // Who counts as the *other* lane is the whole of the distinction
            // `Dispenses` draws. Against an interchangeable unit only a refused
            // lane was decided against; against an identified one a second
            // winner was too, because it got the other message.
            let others: Vec<u32> = match resource.dispenses() {
                Dispenses::Interchangeable => losers.iter().copied().collect(),
                Dispenses::Identified => winners.iter().chain(losers.iter()).copied().collect(),
            };
            // Every pair is considered rather than the lowest-numbered winner
            // and someone unequal to it. Those are not the same test: a lane
            // that draws twice can both win and lose, and if it is the lowest
            // winner then asking only about *it* misses a higher-numbered
            // winner that raced it. Taking the minimum over pairs also makes
            // the report the same text every run, which picking out of a
            // `HashSet` did not.
            others
                .iter()
                .flat_map(|other| {
                    winners
                        .iter()
                        .filter(move |winner| *winner != other)
                        .map(move |winner| {
                            (key, (*winner).min(*other), (*winner).max(*other))
                        })
                })
                .min_by_key(|(_, first, second)| (*first, *second))
        })
        .collect();
    contended.sort();
    // One report per run, as in clause 1.
    if let Some((key, first, second)) = contended.first() {
        let (epoch, resource, _) = **key;
        out.push(Violation::new(
            Invariant::LaneIndependence,
            format!(
                "epoch {epoch}: lanes {first} and {second} both {} depends on which lane ran first",
                resource.describe()
            ),
        ));
    }
}

// ---- I23 -----------------------------------------------------------------

/// **I23. Position-derived emission.**
///
/// `docs/SOMA-v0.3.md` §4 lists trace emission as the fourth thing a
/// device-resident scheduler has to preserve: it becomes a concurrent append,
/// and logical time must still satisfy I11 and I18. The difficulty is that
/// `logical_time` is drawn from a single counter, and concurrent lanes have no
/// counter to share.
///
/// The clause is that they do not need one. Every event carries the position it
/// was emitted at — its epoch, the lane that emitted it, and that lane's own
/// count — and the run's order is exactly the order of those positions:
///
/// 1. positions are unique, so the ordering is total;
/// 2. sorting the trace by position reproduces the trace as emitted; and
/// 3. work that ran in a lane is attributed to one, and no two continuations
///    share a lane within an epoch.
///
/// Together the first two say `logical_time` is derived rather than
/// load-bearing. A concurrent implementation counts locally, appends in
/// whatever order it finishes, and the reference order is recovered by a sort —
/// which is what makes I11 and I18 checkable against it at all.
///
/// Clause 3 is what stops that from being free. Without it, an implementation
/// that emitted every event from `HOST_LANE` off a single counter would satisfy
/// clauses 1 and 2 exactly — positions unique, sorted order equal to emitted
/// order — while being the shared-clock design the clause exists to replace.
/// Requiring each executing continuation to hold a lane of its own means the
/// sequence space really is partitioned, which is the thing a device needs.
///
/// **Clause 2 is a statement about the reference.** It holds for a run whose
/// append order *is* its emission order, which is what a sequential interpreter
/// produces. A concurrent implementation appends interleaved and will not
/// satisfy it on its raw trace; what it owes is clauses 1 and 3, and I18 after
/// sorting by position. Requiring clause 2 of such an implementation would
/// re-import the assumption §2 removed from the equivalence relation.
///
/// The clause deliberately does *not* say that lanes may run in any order. It
/// says the record of a run is reconstructible without a clock. What a lane
/// observes still depends on when it ran relative to other lanes: I24 moved bin
/// entry to an applier, but the applier runs at the end of each lane and
/// everything else is still written as the step runs. Canonical commit is a
/// different obligation and is not met.
fn position_derived_emission(kernel: &Kernel, out: &mut Vec<Violation>) {
    let events = kernel.trace_events();

    let mut seen: std::collections::BTreeSet<(u32, u32, u32)> = std::collections::BTreeSet::new();
    for event in events {
        if !seen.insert(event.position()) {
            let (epoch, lane, sequence) = event.position();
            out.push(Violation::new(
                Invariant::PositionDerivedEmission,
                format!(
                    "two events share position (epoch {epoch}, lane {lane}, sequence {sequence}), \
                     so their order is not recoverable"
                ),
            ));
        }
    }

    // Clause 2, and only of an executive that runs its lanes in plan order.
    //
    // §4.2 wrote this exemption before there was anything to exempt: "a
    // concurrent implementation appends interleaved and will not satisfy it on
    // its raw trace; what it owes is clauses 1 and 3, and I18 after sorting by
    // position." A reordering executive (§4.6) is the first thing in this crate
    // that appends out of position order, and it owes exactly that instead.
    //
    // Skipping the clause rather than weakening it is the point. Weakened to
    // "sorting by position gives *a* total order" it would be clause 1 again
    // and would hold of anything; the content moves to `tests/lane_order.rs`,
    // where a reordered run is compared against the plan-order run it must
    // reproduce.
    //
    // Comparing positions pairwise rather than sorting a copy keeps the report
    // specific: the first inversion names the two events, where a sort would
    // only report that the orders differ.
    for (index, window) in events.windows(2).enumerate() {
        if !kernel.lane_order().is_plan_order() {
            break;
        }
        if window[0].position() >= window[1].position() {
            let (epoch, lane, sequence) = window[0].position();
            let (next_epoch, next_lane, next_sequence) = window[1].position();
            out.push(Violation::new(
                Invariant::PositionDerivedEmission,
                format!(
                    "event {index} at (epoch {epoch}, lane {lane}, sequence {sequence}) was \
                     emitted before event {} at (epoch {next_epoch}, lane {next_lane}, sequence \
                     {next_sequence}), which sorts earlier",
                    index + 1
                ),
            ));
            break;
        }
    }

    // Clause 3. `ContinuationStarted` is the one event that marks a lane
    // actually running something, so it is where attribution is checkable.
    let mut occupant: std::collections::BTreeMap<(u32, u32), Ref64> =
        std::collections::BTreeMap::new();
    for event in events {
        if event.event_kind != crate::abi::EventKind::ContinuationStarted {
            continue;
        }
        if event.lane == crate::abi::traces::HOST_LANE {
            out.push(Violation::new(
                Invariant::PositionDerivedEmission,
                format!(
                    "continuation {} ran without a lane of its own, so emission is still \
                     serialised through the host",
                    event.continuation.slot
                ),
            ));
            continue;
        }
        if let Some(previous) = occupant.insert((event.epoch, event.lane), event.continuation) {
            if previous != event.continuation {
                out.push(Violation::new(
                    Invariant::PositionDerivedEmission,
                    format!(
                        "epoch {} lane {} carried continuations {} and {}, so their events share \
                         one sequence space",
                        event.epoch, event.lane, previous.slot, event.continuation.slot
                    ),
                ));
            }
        }
    }
}

// ---- I22 -----------------------------------------------------------------

/// **I22. Admission determinism.**
///
/// Stated and checked in `semantics::schedule`, which owns the permutation
/// machinery; this is the seam that makes it part of `check`, so a run that is
/// checked at all is checked for it.
fn admission_determinism(kernel: &Kernel, out: &mut Vec<Violation>) {
    for violation in crate::semantics::schedule::admission_determinism(kernel) {
        out.push(Violation::new(
            Invariant::AdmissionDeterminism,
            violation.detail,
        ));
    }
}

// ---- I21 -----------------------------------------------------------------

/// **I21. Bounded progress.**
///
/// This replaces v0.2's I14, which was the specification's only `[modelled]`
/// clause — verified by targeted test rather than by a predicate. Two halves:
///
/// 1. **No withholding.** An epoch that admitted work dispatched some of it.
///    The reference interpreter guarantees this by re-planning under
///    `RunPartial` when a deferral policy would otherwise idle the epoch; the
///    counter is what makes a future policy unable to break it quietly.
/// 2. **No starvation.** No runnable continuation has sat in a bin for longer
///    than the kernel's deferral bound. v0.2 §4 declined to promise this and
///    allowed one run class to starve another. Under placement policies that
///    bind classes to territories, starvation stops being a scheduling detail
///    and becomes a silent correctness surprise, so it gets a bound.
fn bounded_progress(kernel: &Kernel, out: &mut Vec<Violation>) {
    if kernel.accounting().stalled_epochs > 0 {
        out.push(Violation::new(
            Invariant::BoundedProgress,
            format!(
                "{} epoch(s) admitted work and dispatched none",
                kernel.accounting().stalled_epochs
            ),
        ));
    }

    let bound = kernel.deferral_bound();
    let epoch = kernel.epoch_number();
    for (r, c) in kernel.continuations().iter() {
        if c.status != ContinuationState::Runnable {
            continue;
        }
        // A continuation enqueued this epoch has waited zero epochs. Only the
        // gap since it last changed state counts, so work created during the
        // current epoch is never reported.
        let waiting_since = c.last_run_epoch.max(c.created_epoch);
        let waited = epoch.saturating_sub(waiting_since);
        if waited > bound {
            out.push(Violation::new(
                Invariant::BoundedProgress,
                format!(
                    "continuation {} of run class {} has been runnable for {} epochs, bound is {}",
                    r.slot, c.run_class, waited, bound
                ),
            ));
        }
    }
}

// ---- I17 -----------------------------------------------------------------

fn module_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (module_ref, module) in kernel.modules().iter() {
        let valid = kernel
            .module_manifest(module_ref)
            .map(|manifest| {
                module.id == module_ref
                    && module.evaluator_count == manifest.len() as u32
                    && !manifest.is_empty()
                    && manifest.iter().all(|(id, stride)| *id != 0 && *stride != 0)
                    && manifest.windows(2).all(|pair| pair[0].0 < pair[1].0)
            })
            .unwrap_or(false);
        if !valid {
            out.push(Violation::new(
                Invariant::ModuleIntegrity,
                format!("module {} has an invalid manifest", module_ref.slot),
            ));
        }
    }
    for (collective_ref, collective) in kernel.collectives().iter() {
        if collective.module.is_null() {
            continue;
        }
        let linked = kernel
            .module_manifest(collective.module)
            .map(|manifest| {
                manifest.iter().any(|(id, stride)| {
                    *id == collective.evaluator_id && *stride == collective.element_stride
                })
            })
            .unwrap_or(false);
        if !linked {
            out.push(Violation::new(
                Invariant::ModuleIntegrity,
                format!(
                    "collective {} is not linked to its module evaluator",
                    collective_ref.slot
                ),
            ));
        }
    }
}

// ---- I16 -----------------------------------------------------------------

fn domain_contract_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    let root = kernel.root_domain();
    if kernel
        .domains()
        .get(root)
        .map(|domain| !domain.parent.is_null())
        .unwrap_or(true)
    {
        out.push(Violation::new(
            Invariant::DomainContractIntegrity,
            "root domain is missing or has a parent",
        ));
    }

    for (domain_ref, domain) in kernel.domains().iter() {
        let actual = kernel
            .processes()
            .iter()
            .filter(|(_, process)| process.domain == domain_ref)
            .count() as u32;
        if domain.parent == domain_ref
            || domain.processes_created != actual
            || (domain.max_processes != 0 && actual > domain.max_processes)
        {
            out.push(Violation::new(
                Invariant::DomainContractIntegrity,
                format!(
                    "domain {} has invalid parent, count, or quota",
                    domain_ref.slot
                ),
            ));
        }
    }

    for (contract_ref, contract) in kernel.contracts().iter() {
        if !Kernel::contract_is_valid(contract) {
            out.push(Violation::new(
                Invariant::DomainContractIntegrity,
                format!("contract {} is invalid for this machine", contract_ref.slot),
            ));
        }
    }

    for (continuation_ref, continuation) in kernel.continuations().iter() {
        if continuation.execution_contract.is_null() {
            continue;
        }
        let valid = kernel
            .contracts()
            .get(continuation.execution_contract)
            .ok()
            .and_then(|contract| {
                kernel.objects().get(continuation.frame).ok().map(|frame| {
                    continuation.remaining_steps <= contract.maximum_steps
                        && (contract.local_memory_bytes == 0
                            || frame.byte_length <= u64::from(contract.local_memory_bytes))
                })
            })
            .unwrap_or(false);
        if !valid {
            out.push(Violation::new(
                Invariant::DomainContractIntegrity,
                format!(
                    "continuation {} violates its execution contract",
                    continuation_ref.slot
                ),
            ));
        }
    }
}

/// Panic with every violation, for use as a test postcondition.
pub fn assert_legal(kernel: &Kernel) {
    let violations = check(kernel);
    assert!(
        violations.is_empty(),
        "illegal machine state at epoch {}:\n{}",
        kernel.epoch_number(),
        violations
            .iter()
            .map(|v| format!("  {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ---- I1 ------------------------------------------------------------------

fn live(kernel: &Kernel, r: Ref64) -> bool {
    match r.kind {
        Kind::Process => kernel.processes().get(r).is_ok(),
        Kind::Domain => kernel.domains().get(r).is_ok(),
        Kind::Object => kernel.objects().get(r).is_ok(),
        Kind::Continuation => kernel.continuations().get(r).is_ok(),
        Kind::Future => kernel.futures().get(r).is_ok(),
        // Capability references are actor-relative and are checked by I10b in
        // the space that owns them; there is no meaningful global lookup.
        Kind::Capability => true,
        Kind::Channel => kernel.channels().get(r).is_ok(),
        Kind::Collective => kernel.collectives().get(r).is_ok(),
        Kind::Contract => kernel.contracts().get(r).is_ok(),
        Kind::Module => kernel.modules().get(r).is_ok(),
    }
}

fn reference_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, p) in kernel.processes().iter() {
        if !p.id.is_null() && p.id != r {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("process {} carries id {}", r.slot, p.id.slot),
            ));
        }
        if !p.state.is_null() && !live(kernel, p.state) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("process {} has a dangling state object", r.slot),
            ));
        }
        if !live(kernel, p.domain) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("process {} has a dangling domain", r.slot),
            ));
        }
        if !p.supervisor.is_null() && !live(kernel, p.supervisor) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("process {} has a dangling supervisor", r.slot),
            ));
        }
    }

    for (r, c) in kernel.continuations().iter() {
        if !live(kernel, c.process) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} references a dead process", r.slot),
            ));
        }
        if !c.frame.is_null() && !live(kernel, c.frame) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} has a dangling frame", r.slot),
            ));
        }
        if !c.dependency.is_null() && !live(kernel, c.dependency) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} depends on a dead entity", r.slot),
            ));
        }
        if !c.execution_contract.is_null() && !live(kernel, c.execution_contract) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} has a dangling contract", r.slot),
            ));
        }
    }

    for (r, f) in kernel.futures().iter() {
        if !f.owner_process.is_null() && !live(kernel, f.owner_process) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("future {} references a dead owner process", r.slot),
            ));
        }
        if f.state == FutureState::Resolved && !f.value.is_null() && !live(kernel, f.value) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("future {} resolved to a dead object", r.slot),
            ));
        }
    }

    for (r, object) in kernel.objects().iter() {
        if !live(kernel, object.owner_domain) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("object {} has a dangling owner domain", r.slot),
            ));
        }
    }

    for (r, domain) in kernel.domains().iter() {
        if domain.id != r || (!domain.parent.is_null() && !live(kernel, domain.parent)) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("domain {} has an invalid identity or parent", r.slot),
            ));
        }
    }

    for (r, contract) in kernel.contracts().iter() {
        if contract.id != r {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("contract {} has an invalid identity", r.slot),
            ));
        }
    }

    for queue in kernel.supervision_queues().values() {
        for notice in &queue.notices {
            if !live(kernel, notice.child)
                || (!notice.replacement.is_null() && !live(kernel, notice.replacement))
            {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!(
                        "supervision notice for child {} has a dangling reference",
                        notice.child.slot
                    ),
                ));
            }
        }
    }

    for (slot, mailbox) in kernel.mailboxes() {
        for m in &mailbox.entries {
            if !m.payload.is_null() && !live(kernel, m.payload) {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!("mailbox {slot} holds a message with a dead payload"),
                ));
            }
        }
    }

    for (r, channel) in kernel.channels().iter() {
        if channel.id != r || channel.closed > 1 {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("channel {} contains an invalid id or closed state", r.slot),
            ));
        }
    }

    for queue in kernel.channel_queue_snapshots() {
        for (payload, _, escrow_target) in queue.entries {
            if !live(kernel, payload) || escrow_target != payload {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!(
                        "channel {} holds an invalid escrowed payload",
                        queue.channel.slot
                    ),
                ));
            }
        }
        for waiter in queue.send_waiters.into_iter().chain(queue.receive_waiters) {
            if !live(kernel, waiter) {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!("channel {} holds a dead waiter", queue.channel.slot),
                ));
            }
        }
    }

    for (r, collective) in kernel.collectives().iter() {
        if collective.id != r
            || (!collective.owner_process.is_null() && !live(kernel, collective.owner_process))
            || !live(kernel, collective.inputs)
            || !live(kernel, collective.completion_future)
            || (!collective.module.is_null() && !live(kernel, collective.module))
            || (!collective.outputs.is_null() && !live(kernel, collective.outputs))
        {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!(
                    "collective {} contains a dangling or inconsistent reference",
                    r.slot
                ),
            ));
        }
    }
}

// ---- I15 -----------------------------------------------------------------

fn supervision_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (process_ref, process) in kernel.processes().iter() {
        if process.supervisor == process_ref {
            out.push(Violation::new(
                Invariant::SupervisionIntegrity,
                format!("process {} supervises itself", process_ref.slot),
            ));
        }
        if process.supervision_policy == crate::abi::SupervisionPolicy::Restart {
            if process.supervisor.is_null()
                || process.restart_limit == 0
                || process.restart_attempt > process.restart_limit
                || !kernel.has_restart_blueprint(process_ref)
            {
                out.push(Violation::new(
                    Invariant::SupervisionIntegrity,
                    format!(
                        "process {} has an invalid restart contract",
                        process_ref.slot
                    ),
                ));
            }
        } else if !process.restart_of.is_null()
            || process.restart_attempt != 0
            || process.restart_limit != 0
        {
            out.push(Violation::new(
                Invariant::SupervisionIntegrity,
                format!(
                    "process {} has restart lineage without restart policy",
                    process_ref.slot
                ),
            ));
        }
        if !process.restart_of.is_null() {
            let valid_predecessor = kernel
                .processes()
                .get(process.restart_of)
                .map(|predecessor| {
                    predecessor.status == ProcessState::Failed as u32
                        && predecessor.supervisor == process.supervisor
                        && predecessor.restart_attempt + 1 == process.restart_attempt
                })
                .unwrap_or(false);
            if !valid_predecessor {
                out.push(Violation::new(
                    Invariant::SupervisionIntegrity,
                    format!("process {} has invalid restart lineage", process_ref.slot),
                ));
            }
        }
    }

    for (supervisor_key, queue) in kernel.supervision_queues() {
        for notice in &queue.notices {
            let Ok(child) = kernel.processes().get(notice.child) else {
                // The dangling child is also reported by the more general
                // reference checker when held from a descriptor.
                out.push(Violation::new(
                    Invariant::SupervisionIntegrity,
                    format!("supervisor {supervisor_key} has notice for a dead child"),
                ));
                continue;
            };
            let expected_status = match notice.reason {
                crate::abi::ExitReason::Completed => ProcessState::Terminated,
                crate::abi::ExitReason::Failed => ProcessState::Failed,
                crate::abi::ExitReason::Cancelled => ProcessState::Cancelled,
            };
            if child.supervisor.key() != *supervisor_key
                || child.status != expected_status as u32
                || notice.failure_count != child.failure_count
            {
                out.push(Violation::new(
                    Invariant::SupervisionIntegrity,
                    format!(
                        "supervisor {supervisor_key} has an inconsistent notice for child {}",
                        notice.child.slot
                    ),
                ));
            }
            if notice.reason == crate::abi::ExitReason::Failed
                && child.supervision_policy == crate::abi::SupervisionPolicy::Escalate
            {
                let supervisor_failed = kernel
                    .processes()
                    .get(child.supervisor)
                    .map(|supervisor| supervisor.status == ProcessState::Failed as u32)
                    .unwrap_or(false);
                if !supervisor_failed {
                    out.push(Violation::new(
                        Invariant::SupervisionIntegrity,
                        format!(
                            "failed child {} required escalation but supervisor {} did not fail",
                            notice.child.slot, supervisor_key
                        ),
                    ));
                }
            }
            if notice.reason == crate::abi::ExitReason::Failed
                && child.supervision_policy == crate::abi::SupervisionPolicy::Restart
            {
                if notice.replacement.is_null() {
                    let supervisor_failed = kernel
                        .processes()
                        .get(child.supervisor)
                        .map(|supervisor| supervisor.status == ProcessState::Failed as u32)
                        .unwrap_or(false);
                    if !supervisor_failed {
                        out.push(Violation::new(
                            Invariant::SupervisionIntegrity,
                            format!(
                                "failed child {} was neither restarted nor escalated",
                                notice.child.slot
                            ),
                        ));
                    }
                } else {
                    let valid_replacement = kernel
                        .processes()
                        .get(notice.replacement)
                        .map(|replacement| {
                            replacement.restart_of == notice.child
                                && replacement.supervisor == child.supervisor
                                && replacement.process_mode == child.process_mode
                                && replacement.supervision_policy
                                    == crate::abi::SupervisionPolicy::Restart
                                && replacement.restart_attempt == child.restart_attempt + 1
                                && replacement.restart_limit == child.restart_limit
                        })
                        .unwrap_or(false);
                    if !valid_replacement {
                        out.push(Violation::new(
                            Invariant::SupervisionIntegrity,
                            format!(
                                "failed child {} has an invalid replacement",
                                notice.child.slot
                            ),
                        ));
                    }
                }
            } else if !notice.replacement.is_null() {
                out.push(Violation::new(
                    Invariant::SupervisionIntegrity,
                    format!("child {} has an unexpected replacement", notice.child.slot),
                ));
            }
        }
        for waiter in &queue.waiters {
            let valid = kernel
                .continuations()
                .get(*waiter)
                .map(|continuation| continuation.process.key() == *supervisor_key)
                .unwrap_or(false);
            if !valid {
                out.push(Violation::new(
                    Invariant::SupervisionIntegrity,
                    format!("supervisor {supervisor_key} has a foreign or dead waiter"),
                ));
            }
        }
    }
}

// ---- I2 ------------------------------------------------------------------

fn no_continuation_left_running(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, c) in kernel.continuations().iter() {
        if c.status == ContinuationState::Running {
            out.push(Violation::new(
                Invariant::NoContinuationLeftRunning,
                format!("continuation {} is still RUNNING between epochs", r.slot),
            ));
        }
    }
}

// ---- I3 ------------------------------------------------------------------

fn process_continuation_consistency(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, c) in kernel.continuations().iter() {
        let process = match kernel.processes().get(c.process) {
            Ok(p) => p,
            // Already reported by I1.
            Err(_) => continue,
        };
        let schedulable = matches!(
            c.status,
            ContinuationState::Runnable | ContinuationState::Waiting
        );
        let terminal = process.status == ProcessState::Failed as u32
            || process.status == ProcessState::Terminated as u32
            || process.status == ProcessState::Cancelled as u32;
        if schedulable && terminal {
            out.push(Violation::new(
                Invariant::ProcessContinuationConsistency,
                format!(
                    "continuation {} is {:?} but its process {} is terminal",
                    r.slot, c.status, c.process.slot
                ),
            ));
        }
    }

    // `live_continuations` replaced a per-completion table scan, so it is
    // derived state that can silently drift. Recompute it the slow way and
    // compare: a status write that bypasses `set_continuation_status` is a
    // bug that would otherwise surface much later as a stranded process.
    let mut actual: std::collections::BTreeMap<u64, u32> = std::collections::BTreeMap::new();
    for (_, c) in kernel.continuations().iter() {
        if c.status.is_live() {
            *actual.entry(c.process.key()).or_insert(0) += 1;
        }
    }
    for (r, p) in kernel.processes().iter() {
        let counted = actual.get(&r.key()).copied().unwrap_or(0);
        if p.live_continuations != counted {
            out.push(Violation::new(
                Invariant::ProcessContinuationConsistency,
                format!(
                    "process {} caches {} live continuations but {} are live",
                    r.slot, p.live_continuations, counted
                ),
            ));
        }
    }
}

// ---- I4 ------------------------------------------------------------------

fn future_single_assignment(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, f) in kernel.futures().iter() {
        match f.state {
            FutureState::Pending => {
                if !f.value.is_null() {
                    out.push(Violation::new(
                        Invariant::FutureSingleAssignment,
                        format!("future {} is pending but carries a value", r.slot),
                    ));
                }
            }
            FutureState::Resolved | FutureState::Failed | FutureState::Cancelled => {
                if f.resolved_epoch > kernel.epoch_number() {
                    out.push(Violation::new(
                        Invariant::FutureSingleAssignment,
                        format!("future {} resolved in a future epoch", r.slot),
                    ));
                }
                // A settled future's waiter list has already been drained, so a
                // continuation registered on it would never wake.
                if let Some(waiters) = kernel.future_waiters().get(&r.key()) {
                    if !waiters.is_empty() {
                        out.push(Violation::new(
                            Invariant::FutureSingleAssignment,
                            format!(
                                "{} continuations wait on already-settled future {}",
                                waiters.len(),
                                r.slot
                            ),
                        ));
                    }
                }
            }
        }
    }

    for (r, collective) in kernel.collectives().iter() {
        let Ok(completion) = kernel.futures().get(collective.completion_future) else {
            continue;
        };
        let consistent = match collective.state {
            CollectiveState::Pending => {
                completion.state == FutureState::Pending && collective.outputs.is_null()
            }
            CollectiveState::Completed => {
                completion.state == FutureState::Resolved
                    && !collective.outputs.is_null()
                    && completion.value == collective.outputs
            }
            CollectiveState::Failed => completion.state == FutureState::Failed,
            CollectiveState::Cancelled => completion.state == FutureState::Cancelled,
        };
        if !consistent {
            out.push(Violation::new(
                Invariant::FutureSingleAssignment,
                format!("collective {} disagrees with its completion future", r.slot),
            ));
        }
    }
}

// ---- I5 ------------------------------------------------------------------

fn mailbox_bound(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (slot, mailbox) in kernel.mailboxes() {
        if mailbox.entries.len() > mailbox.capacity {
            out.push(Violation::new(
                Invariant::MailboxBound,
                format!(
                    "mailbox {slot} holds {} messages over a capacity of {}",
                    mailbox.entries.len(),
                    mailbox.capacity
                ),
            ));
        }
    }
    for queue in kernel.channel_queue_snapshots() {
        let Ok(descriptor) = kernel.channels().get(queue.channel) else {
            continue;
        };
        if queue.entries.len() > descriptor.capacity as usize {
            out.push(Violation::new(
                Invariant::MailboxBound,
                format!(
                    "channel {} holds {} messages over a capacity of {}",
                    queue.channel.slot,
                    queue.entries.len(),
                    descriptor.capacity
                ),
            ));
        }
    }
}

// ---- I6 ------------------------------------------------------------------

fn message_ordering(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (slot, mailbox) in kernel.mailboxes() {
        let mut last: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for m in &mailbox.entries {
            if let Some(previous) = last.get(&m.sender.key()) {
                if m.sender_sequence <= *previous {
                    out.push(Violation::new(
                        Invariant::MessageOrdering,
                        format!(
                            "mailbox {slot}: sender {} delivered sequence {} after {}",
                            m.sender.slot, m.sender_sequence, previous
                        ),
                    ));
                }
            }
            last.insert(m.sender.key(), m.sender_sequence);
        }
    }
    for queue in kernel.channel_queue_snapshots() {
        for pair in queue.entries.windows(2) {
            if pair[1].1 <= pair[0].1 {
                out.push(Violation::new(
                    Invariant::MessageOrdering,
                    format!(
                        "channel {} sequence {} follows {}",
                        queue.channel.slot, pair[1].1, pair[0].1
                    ),
                ));
            }
        }
    }
}

// ---- I7 ------------------------------------------------------------------

fn scheduler_well_formed(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (bin, cont) in kernel.scheduler().pending_entries() {
        let c = match kernel.continuations().get(cont) {
            Ok(c) => c,
            Err(_) => {
                out.push(Violation::new(
                    Invariant::SchedulerWellFormed,
                    format!("bin {bin} holds dead continuation {}", cont.slot),
                ));
                continue;
            }
        };
        if c.status != ContinuationState::Runnable {
            out.push(Violation::new(
                Invariant::SchedulerWellFormed,
                format!(
                    "bin {bin} holds continuation {} in state {:?}",
                    cont.slot, c.status
                ),
            ));
        }
        let expected = kernel.scheduler().bin_of(c.run_class);
        if expected != bin {
            out.push(Violation::new(
                Invariant::SchedulerWellFormed,
                format!(
                    "continuation {} of run class {} sits in bin {bin}, not {expected}",
                    cont.slot, c.run_class
                ),
            ));
        }
    }
}

// ---- I8 ------------------------------------------------------------------

fn frame_exclusivity(kernel: &Kernel, out: &mut Vec<Violation>) {
    // Keyed by full identity, not slot: two partitions each mint a slot 7, and
    // comparing bare slots would report those two frames as one shared frame.
    let mut owner: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for (r, c) in kernel.continuations().iter() {
        if c.frame.is_null() {
            continue;
        }
        if let Some(previous) = owner.insert(c.frame.key(), r.slot) {
            out.push(Violation::new(
                Invariant::FrameExclusivity,
                format!(
                    "frame {} is shared by continuations {} and {}",
                    c.frame.slot, previous, r.slot
                ),
            ));
        }
    }
}

// ---- I10 -----------------------------------------------------------------

fn capability_attenuation(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (holder, space) in kernel.capability_spaces() {
        for (r, cap) in space.iter() {
            if cap.parent_capability.is_null() {
                continue;
            }
            let Ok(parent) = space.get(cap.parent_capability) else {
                // I10b reports the broken link.
                continue;
            };
            let cap_end = cap.offset.checked_add(cap.length);
            let parent_end = parent.offset.checked_add(parent.length);
            if cap.rights & !parent.rights != 0
                || cap.offset < parent.offset
                || cap_end.is_none()
                || parent_end.is_none()
                || cap_end > parent_end
            {
                out.push(Violation::new(
                    Invariant::CapabilityAttenuation,
                    format!(
                        "capability {} in space {holder} amplifies its parent",
                        r.slot
                    ),
                ));
            }
        }
    }
}

fn capability_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    use crate::abi::Rights;

    for (holder, space) in kernel.capability_spaces() {
        for (r, cap) in space.iter() {
            let target_live = match cap.target.kind {
                Kind::Process => kernel.processes().get(cap.target).is_ok(),
                Kind::Object => kernel.objects().get(cap.target).is_ok(),
                Kind::Continuation => kernel.continuations().get(cap.target).is_ok(),
                Kind::Future => kernel.futures().get(cap.target).is_ok(),
                Kind::Channel => kernel.channels().get(cap.target).is_ok(),
                Kind::Collective => kernel.collectives().get(cap.target).is_ok(),
                Kind::Domain => kernel.domains().get(cap.target).is_ok(),
                Kind::Contract => kernel.contracts().get(cap.target).is_ok(),
                Kind::Module => kernel.modules().get(cap.target).is_ok(),
                Kind::Capability => space.get(cap.target).is_ok(),
            };
            if !target_live {
                out.push(Violation::new(
                    Invariant::CapabilityIntegrity,
                    format!(
                        "capability {} in space {holder} has a dead or unsupported target",
                        r.slot
                    ),
                ));
            }
            if cap.rights & !Rights::for_target(cap.target.kind) != 0 {
                out.push(Violation::new(
                    Invariant::CapabilityIntegrity,
                    format!(
                        "capability {} in space {holder} has rights invalid for {:?}",
                        r.slot, cap.target.kind
                    ),
                ));
            }
            if !cap.parent_capability.is_null() && space.get(cap.parent_capability).is_err() {
                out.push(Violation::new(
                    Invariant::CapabilityIntegrity,
                    format!("capability {} in space {holder} has a dead parent", r.slot),
                ));
            }
        }
    }

    for (object, _) in kernel.objects().iter() {
        let writers = kernel.authority_holder_count(object, Rights::WRITE);
        if writers > 1 {
            out.push(Violation::new(
                Invariant::CapabilityIntegrity,
                format!(
                    "object {} has {writers} mutable authority holders",
                    object.slot
                ),
            ));
        }
    }
}

fn no_unauthorized_effect(kernel: &Kernel, out: &mut Vec<Violation>) {
    use crate::abi::EventKind;

    for (index, effect) in kernel.trace_events().iter().enumerate() {
        if effect.event_kind != EventKind::AuthorityEffect {
            continue;
        }
        let authorized = index.checked_sub(1).and_then(|previous| {
            let decision = &kernel.trace_events()[previous];
            (decision.event_kind == EventKind::AuthorityGranted
                && decision.process == effect.process
                && decision.continuation == effect.continuation
                && decision.run_class == effect.run_class)
                .then_some(())
        });
        if authorized.is_none() {
            out.push(Violation::new(
                Invariant::NoUnauthorizedEffect,
                format!(
                    "trace event {index} applies right {} by actor {} to target {} without an adjacent grant",
                    effect.run_class, effect.process.slot, effect.continuation.slot
                ),
            ));
        }
    }
}

// ---- I11 -----------------------------------------------------------------

fn trace_monotonicity(kernel: &Kernel, out: &mut Vec<Violation>) {
    let mut previous_time = 0u64;
    let mut previous_epoch = 0u32;
    for (i, e) in kernel.trace_events().iter().enumerate() {
        if e.logical_time <= previous_time && i > 0 {
            out.push(Violation::new(
                Invariant::TraceMonotonicity,
                format!(
                    "trace event {i} has logical time {} after {}",
                    e.logical_time, previous_time
                ),
            ));
        }
        if e.epoch < previous_epoch {
            out.push(Violation::new(
                Invariant::TraceMonotonicity,
                format!("trace event {i} moves backward to epoch {}", e.epoch),
            ));
        }
        previous_time = e.logical_time;
        previous_epoch = e.epoch;
    }
}

// ---- I12 -----------------------------------------------------------------

fn accounting_consistency(kernel: &Kernel, out: &mut Vec<Violation>) {
    let a = kernel.accounting();
    if a.useful_lane_slots > a.lane_slots {
        out.push(Violation::new(
            Invariant::AccountingConsistency,
            format!(
                "useful lane slots {} exceed issued lane slots {}",
                a.useful_lane_slots, a.lane_slots
            ),
        ));
    }
    if a.full_cohorts > a.cohorts {
        out.push(Violation::new(
            Invariant::AccountingConsistency,
            format!(
                "full cohorts {} exceed total cohorts {}",
                a.full_cohorts, a.cohorts
            ),
        ));
    }
    if a.lane_slots != a.useful_lane_slots + a.idle_lane_slots {
        out.push(Violation::new(
            Invariant::AccountingConsistency,
            format!(
                "lane slots {} do not split into {} useful and {} idle",
                a.lane_slots, a.useful_lane_slots, a.idle_lane_slots
            ),
        ));
    }
}

// ---- I13 -----------------------------------------------------------------

fn serial_process_execution(kernel: &Kernel, out: &mut Vec<Violation>) {
    use crate::abi::EventKind;

    let mut claimed = std::collections::HashSet::new();
    for (index, event) in kernel.trace_events().iter().enumerate() {
        if event.event_kind != EventKind::ContinuationStarted {
            continue;
        }
        let Ok(continuation) = kernel.continuations().get(event.continuation) else {
            continue;
        };
        if continuation.state_access != StateAccess::Mutable {
            continue;
        }
        let key = (event.epoch, continuation.process.slot);
        if !claimed.insert(key) {
            out.push(Violation::new(
                Invariant::SerialProcessExecution,
                format!(
                    "trace event {index} starts a second mutable continuation for process {} in epoch {}",
                    continuation.process.slot, event.epoch
                ),
            ));
        }
    }
}

/// Whether a continuation declares mutable access to its process state.
pub fn mutates_process_state(kernel: &Kernel, continuation: Ref64) -> bool {
    kernel
        .continuations()
        .get(continuation)
        .map(|c| c.state_access == StateAccess::Mutable)
        .unwrap_or(false)
}
