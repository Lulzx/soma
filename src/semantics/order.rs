//! The semantic order ≺, and conformance of a trace to it (I18, I19).
//!
//! `docs/SOMA-v0.2.md` §1.2 defines two runs as equivalent when their traces
//! are *equal*. That is the right relation for the sequential interpreter and
//! the wrong one for anything else. Trace equality is defined over a total
//! order on logical time, and the interpreter's total order is an artifact of
//! running one continuation at a time: two lanes of a single cohort have no
//! semantic order between them, but a `for` loop must emit one of them first.
//! Under equality, every parallel implementation of SOMA is non-conforming by
//! construction.
//!
//! This module replaces equality with refinement. It derives a partial order ≺
//! from the transition rules, and a trace conforms when it is a linear
//! extension of ≺ that agrees with the reference run on what actually
//! happened. Equality becomes the special case where ≺ happens to be total.
//!
//! **On the direction of edges.** ≺ is an ordering constraint read off the
//! transition rules, not a claim about physical causation. §3.3 emits a
//! future's waiter wakes *before* the `FutureResolved` event, so ≺ orders them
//! that way too. Orienting an edge "the intuitive way" against the rule that
//! emits it would make the reference interpreter fail its own checker, which
//! is the first thing `reference_run_is_a_linear_extension` verifies.

use std::collections::BTreeMap;

use crate::abi::{EventKind, Ref64};
use crate::kernel::{Kernel, TraceSnapshotRow};

/// How a trace failed to conform.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrderViolation {
    pub clause: OrderClause,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderClause {
    /// I18. The trace is not a linear extension of ≺.
    ScheduleConformance,
    /// I19. Two placements of the same program disagree observably.
    PlacementNeutrality,
}

impl OrderViolation {
    fn new(clause: OrderClause, detail: impl Into<String>) -> Self {
        Self {
            clause,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for OrderViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.clause, self.detail)
    }
}

/// One edge of ≺, as a pair of indices into the trace it was derived from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge {
    pub earlier: usize,
    pub later: usize,
    pub reason: EdgeReason,
}

/// Why two events are ordered. Recorded so a violation can say which rule the
/// implementation broke rather than only that something was out of order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeReason {
    /// Both events belong to one continuation, which is sequential (§1.1).
    ContinuationProgram,
    /// A future's wakes precede its resolution event (§3.3).
    FutureResolution,
    /// A mailbox message is sent before it is received (§3.2).
    MessageDelivery,
    /// A channel message is sent before it is received (§3.2).
    ChannelDelivery,
    /// A collective is created before it completes (§3.3).
    CollectiveLifecycle,
    /// A child becomes terminal before its supervisor is notified (§3.2).
    SupervisionNotice,
    /// A governed effect immediately follows its authority decision (I10c).
    AuthorityEffect,
}

/// A trace paired with the order derived from it.
pub struct SemanticOrder {
    events: Vec<TraceSnapshotRow>,
    edges: Vec<Edge>,
}

impl SemanticOrder {
    pub fn of(kernel: &Kernel) -> Self {
        Self::from_trace(kernel.trace_snapshot())
    }

    pub fn from_trace(events: Vec<TraceSnapshotRow>) -> Self {
        let edges = derive_edges(&events);
        Self { events, edges }
    }

    pub fn events(&self) -> &[TraceSnapshotRow] {
        &self.events
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Whether this trace is itself consistent with the order derived from it.
    /// The reference interpreter must satisfy this or the derivation is wrong.
    pub fn is_self_consistent(&self) -> bool {
        self.edges.iter().all(|edge| edge.earlier < edge.later)
    }

    /// Every ≺ edge that joins two distinct lanes of one epoch.
    ///
    /// An edge like this is a lane observing another lane's write within the
    /// epoch they share. That is what canonical commit forbids: the applier
    /// runs once, after every lane, so the only inter-lane ordering an epoch
    /// has is the plan's — and a run containing such an edge is one whose
    /// result depends on which lane went first, which is the schedule
    /// dependence §2 removed from the equivalence relation.
    ///
    /// `docs/SOMA-v0.3.md` §4.3 (3) measured this across the Expand workload
    /// and found none, and was careful to call the result a precondition to
    /// check per run rather than a property of the model. It is still exactly
    /// that. What changed is that the executive now *relies* on it, so I25 asks
    /// it of every run instead of it having been asked once.
    ///
    /// Edges within one lane are the common case and are not reported: a lane
    /// is sequential, so `ContinuationProgram` and `AuthorityEffect` edges
    /// cannot join two lanes at all. Edges touching `HOST_LANE` are not
    /// reported either — the host's part of an epoch runs before and after the
    /// lanes rather than beside them, so an order between it and a lane is the
    /// plan's own and is not a race.
    pub fn cross_lane_edges(&self) -> Vec<Edge> {
        self.edges
            .iter()
            .copied()
            .filter(|edge| {
                let (Some(a), Some(b)) =
                    (self.events.get(edge.earlier), self.events.get(edge.later))
                else {
                    return false;
                };
                a.epoch == b.epoch
                    && a.lane != b.lane
                    && a.lane != crate::abi::traces::HOST_LANE
                    && b.lane != crate::abi::traces::HOST_LANE
            })
            .collect()
    }
}

/// Key identifying the position of an event within its entity's history, used
/// to match an implementation's events against the reference run's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EventIdentity {
    epoch: u32,
    event_kind: EventKind,
    process: u64,
    continuation: u64,
    run_class: u32,
    auxiliary: u32,
    subject: u64,
    causal: u64,
}

impl EventIdentity {
    fn of_renamed(row: &TraceSnapshotRow, names: &IdentityMap) -> Self {
        Self {
            epoch: row.epoch,
            event_kind: row.event_kind,
            process: names.rename(row.process),
            continuation: names.rename(row.continuation),
            run_class: row.run_class,
            auxiliary: row.auxiliary,
            subject: names.rename(row.subject),
            causal: names.rename(row.causal),
        }
    }
}

/// A correspondence between two runs' entity names.
///
/// An identity is a table position, and a table position is an implementation
/// detail: two runs of one program that allocate from different partitions —
/// which is what a device's lanes and a cluster's nodes must do — name the same
/// entity differently and behave identically. Comparing raw `Ref64`s makes such
/// an implementation non-conforming by construction, in exactly the way trace
/// equality did for ordering before §2.
///
/// The map is *forced*, not chosen: entities are paired in the order they first
/// appear in each trace, within their kind. A checker that got to pick the
/// correspondence could pair whatever made the traces agree; this one cannot.
/// Pairing two sequences of distinct names positionally is a bijection, so an
/// implementation that dropped an entity or merged two into one produces
/// sequences of different lengths and is reported rather than renamed away.
#[derive(Clone, Debug, Default)]
pub struct IdentityMap {
    forward: BTreeMap<u64, u64>,
}

impl IdentityMap {
    pub fn identity() -> Self {
        Self::default()
    }

    /// Translate a candidate-run name into the reference run's namespace.
    /// Unmapped names — the null reference, and anything the map never saw —
    /// pass through, so a missing entry shows up as a mismatch rather than as
    /// a silent success.
    pub fn rename(&self, encoded: u64) -> u64 {
        self.forward.get(&encoded).copied().unwrap_or(encoded)
    }

    /// Pair `candidate`'s entity names with `reference`'s by order of first
    /// appearance, within each kind.
    pub fn between(
        reference: &[TraceSnapshotRow],
        candidate: &[TraceSnapshotRow],
    ) -> Result<Self, OrderViolation> {
        let mut forward = BTreeMap::new();
        let expected = first_appearances(reference);
        let actual = first_appearances(candidate);

        for (kind, wanted) in &expected {
            let found = actual.get(kind).map(Vec::as_slice).unwrap_or(&[]);
            if found.len() != wanted.len() {
                return Err(OrderViolation::new(
                    OrderClause::ScheduleConformance,
                    format!(
                        "the run named {} entities of kind {:?} where the reference named {}, so \
                         no correspondence between them exists",
                        found.len(),
                        kind,
                        wanted.len()
                    ),
                ));
            }
            for (theirs, ours) in found.iter().zip(wanted) {
                forward.insert(*theirs, *ours);
            }
        }
        for (kind, found) in &actual {
            if !expected.contains_key(kind) && !found.is_empty() {
                return Err(OrderViolation::new(
                    OrderClause::ScheduleConformance,
                    format!(
                        "the run named {} entities of kind {kind:?}, which the reference never \
                         mentions",
                        found.len()
                    ),
                ));
            }
        }
        Ok(Self { forward })
    }
}

/// Every entity name a trace mentions, bucketed by kind, in order of first
/// appearance.
///
/// The four reference-shaped fields are read in a fixed order per event, so the
/// enumeration is a function of the trace and nothing else. `auxiliary` is
/// deliberately absent: it is numeric, and an entity recorded there as a bare
/// slot would be untranslatable — which is why `subject` exists.
fn first_appearances(events: &[TraceSnapshotRow]) -> BTreeMap<u8, Vec<u64>> {
    let mut order: BTreeMap<u8, Vec<u64>> = BTreeMap::new();
    let mut seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for row in events {
        for encoded in [row.process, row.continuation, row.subject, row.causal] {
            if encoded == 0 || slot(encoded) == 0 || !seen.insert(encoded) {
                continue;
            }
            let kind = Ref64::from_u64(encoded).kind.as_u8();
            order.entry(kind).or_default().push(encoded);
        }
    }
    order
}

fn slot(encoded: u64) -> u32 {
    Ref64::from_u64(encoded).slot
}

/// Events that report *where and how* work was grouped rather than *what
/// happened*.
///
/// A cohort is an implementation strategy the model enables but does not
/// require (§4, "Cohorting"). Running the same program at cohort width 1 and
/// width 16 produces a different number of `CohortCreated` records and
/// identical behaviour, so these records cannot be part of observable
/// behaviour without making §4 false. They stay in the trace — accounting and
/// the occupancy studies need them — but the semantic projection drops them.
///
/// This is the one place where widening the equivalence could hide a real
/// defect, so the rule is narrow: an event is placement-only when removing it
/// from every conforming implementation's trace loses no information about
/// process, continuation, message, future, capability, or collective state.
pub fn is_placement_event(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::CohortCreated | EventKind::ContinuationPlaced
    )
}

/// The observable part of a trace: everything except placement reporting.
pub fn semantic_projection(events: &[TraceSnapshotRow]) -> Vec<TraceSnapshotRow> {
    events
        .iter()
        .filter(|row| !is_placement_event(row.event_kind))
        .copied()
        .collect()
}

/// Re-sort a trace into the order its positions put it in.
///
/// I18 compares traces as *emitted*, which is the right relation for an
/// executive whose append order is its plan order and the wrong one for
/// anything else. §4.2 said so when I23 was written: a concurrent
/// implementation appends interleaved, and "what it owes is clauses 1 and 3,
/// and I18 after sorting by position". This is that sort, made an explicit step
/// rather than folded into `conforms`.
///
/// Explicit because the two relations are genuinely different obligations and
/// collapsing them would hide which one an implementation met. A plan-order run
/// is unchanged by this — its emission order already *is* its position order,
/// which I23's clause 2 checks — so an implementation that quietly appended out
/// of order would go from failing clause 2 to passing a silently-sorted I18,
/// and the clause would stop meaning anything.
///
/// The sort is total and needs nothing but the trace: a position is
/// `(epoch, lane, sequence)`, positions are unique (I23 clause 1), and every
/// one of those is decided before the work runs. So this is a derivation any
/// implementation can perform on its own output, not a privilege of the
/// reference.
pub fn in_position_order(events: &[TraceSnapshotRow]) -> Vec<TraceSnapshotRow> {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|row| (row.epoch, row.lane, row.lane_sequence));
    sorted
}

/// Build ≺ from a trace.
///
/// Two structural relations plus five causal ones. Deliberately not included:
/// per-*process* program order. Two read-only continuations of one process may
/// legitimately run in the same epoch (I13 serialises only mutable ones), so
/// their events are genuinely unordered and an edge between them would reject
/// correct implementations.
fn derive_edges(events: &[TraceSnapshotRow]) -> Vec<Edge> {
    let mut edges = Vec::new();

    // (a) Program order within one continuation.
    let mut previous_of_continuation: BTreeMap<u64, usize> = BTreeMap::new();
    for (index, row) in events.iter().enumerate() {
        if row.continuation == 0 || slot(row.continuation) == 0 {
            continue;
        }
        // A continuation reference is only meaningful for events that name a
        // continuation as their subject. Several kinds reuse the field to
        // carry the entity they act on (a channel, a collective, a child
        // process), which is a different namespace.
        if !names_a_continuation(row.event_kind) {
            continue;
        }
        if let Some(previous) = previous_of_continuation.insert(row.continuation, index) {
            edges.push(Edge {
                earlier: previous,
                later: index,
                reason: EdgeReason::ContinuationProgram,
            });
        }
    }

    // The remaining relations pair two events through a shared key. Every one
    // of them collects both sides first and joins afterwards, rather than
    // matching a "pending" side as the scan walks forward.
    //
    // That distinction is the difference between a checker and a decoration.
    // A forward scan that consumes a pending send when it meets a receive
    // finds no pair at all when the two are inverted — so it emits no edge,
    // and an inverted delivery is silently accepted. Collecting both sides by
    // key means the edge exists regardless of position, and the inversion is
    // exactly what shows up as `earlier > later`.
    let mut produced: BTreeMap<(EdgeReason, u64, u64, u32), usize> = BTreeMap::new();
    let mut consumed: BTreeMap<(EdgeReason, u64, u64, u32), usize> = BTreeMap::new();

    for (index, row) in events.iter().enumerate() {
        match row.event_kind {
            // (b) Futures: every wake attributed to a future precedes that
            // future's resolution event (§3.3 emits the wakes first, so the
            // wake is the "producer" side here).
            EventKind::ContinuationReady if row.causal != 0 => {
                produced.insert(
                    (
                        EdgeReason::FutureResolution,
                        row.causal,
                        row.continuation,
                        0,
                    ),
                    index,
                );
            }
            EventKind::FutureResolved | EventKind::FutureFailed | EventKind::FutureCancelled => {
                if row.causal != 0 {
                    consumed.insert((EdgeReason::FutureResolution, row.causal, 0, 0), index);
                }
            }

            // (c) Mailbox delivery, keyed on (sender, receiver, sequence). The
            // send records the receiver in `causal` and the receive records the
            // sender, so the pair is exact even when several senders target one
            // mailbox.
            EventKind::MessageSent => {
                produced.insert(
                    (
                        EdgeReason::MessageDelivery,
                        row.process,
                        row.causal,
                        row.auxiliary,
                    ),
                    index,
                );
            }
            EventKind::MessageReceived => {
                consumed.insert(
                    (
                        EdgeReason::MessageDelivery,
                        row.causal,
                        row.process,
                        row.auxiliary,
                    ),
                    index,
                );
            }

            // (d) Channel delivery, keyed on (channel, sequence). Both events
            // already carry the channel in the entity field and the sequence in
            // `auxiliary`.
            EventKind::ChannelSent => {
                produced.insert(
                    (
                        EdgeReason::ChannelDelivery,
                        row.continuation,
                        0,
                        row.auxiliary,
                    ),
                    index,
                );
            }
            EventKind::ChannelReceived => {
                consumed.insert(
                    (
                        EdgeReason::ChannelDelivery,
                        row.continuation,
                        0,
                        row.auxiliary,
                    ),
                    index,
                );
            }

            // (e) Collective lifecycle: creation precedes settlement.
            EventKind::CollectiveCreated => {
                produced.insert(
                    (EdgeReason::CollectiveLifecycle, row.continuation, 0, 0),
                    index,
                );
            }
            EventKind::CollectiveCompleted
            | EventKind::CollectiveFailed
            | EventKind::CollectiveCancelled => {
                consumed.insert(
                    (EdgeReason::CollectiveLifecycle, row.continuation, 0, 0),
                    index,
                );
            }

            // (f) Supervision: a child's terminal event precedes the notice
            // reporting it. The notice names the child in its entity field.
            EventKind::ProcessFailed | EventKind::ProcessCancelled => {
                produced
                    .entry((EdgeReason::SupervisionNotice, row.process, 0, 0))
                    .or_insert(index);
            }
            EventKind::SupervisionNotified => {
                consumed.insert(
                    (EdgeReason::SupervisionNotice, row.continuation, 0, 0),
                    index,
                );
            }
            _ => {}
        }
    }

    // Futures fan out: many wakes share one resolution, so their producer keys
    // carry the woken continuation and must be matched on the future alone.
    for ((reason, entity, secondary, sequence), earlier) in &produced {
        let key = if *reason == EdgeReason::FutureResolution {
            (*reason, *entity, 0, 0)
        } else {
            (*reason, *entity, *secondary, *sequence)
        };
        if let Some(later) = consumed.get(&key) {
            edges.push(Edge {
                earlier: *earlier,
                later: *later,
                reason: *reason,
            });
        }
    }

    // (g) Authority: a governed effect follows its grant (I10c already
    // requires adjacency; the edge makes the requirement survive reordering).
    for (index, row) in events.iter().enumerate() {
        if row.event_kind == EventKind::AuthorityEffect && index > 0 {
            edges.push(Edge {
                earlier: index - 1,
                later: index,
                reason: EdgeReason::AuthorityEffect,
            });
        }
    }

    edges.sort();
    edges
}

/// Event kinds whose `continuation` field really names a continuation.
///
/// The trace ABI reuses that field to carry whatever entity an event is about.
/// Treating a channel slot as a continuation slot would invent program-order
/// edges between unrelated events, which is the easiest way to make this
/// checker reject correct implementations.
fn names_a_continuation(kind: EventKind) -> bool {
    matches!(
        kind,
        EventKind::ContinuationReady
            | EventKind::ContinuationStarted
            | EventKind::ContinuationYielded
            | EventKind::ContinuationWaiting
            | EventKind::ContinuationCompleted
            | EventKind::ContinuationCancelled
            | EventKind::MessageSent
            | EventKind::MessageReceived
    )
}

/// **I18. Schedule conformance.**
///
/// `candidate` conforms to `reference` when:
///
/// 1. it contains exactly the same events, per epoch — same kinds, same
///    subjects, same auxiliary data, same causal attribution;
/// 2. epochs do not move backwards; and
/// 3. it is a linear extension of ≺.
///
/// Clause 1 is what stops the relation from being vacuous: a weakened
/// equivalence that only checked ordering would accept an implementation that
/// silently dropped work. Clause 3 is what admits parallelism: events with no
/// edge between them may appear in either order.
pub fn conforms(reference: &Kernel, candidate: &Kernel) -> Vec<OrderViolation> {
    conforms_traces(&reference.trace_snapshot(), &candidate.trace_snapshot())
}

pub fn conforms_traces(
    reference: &[TraceSnapshotRow],
    candidate: &[TraceSnapshotRow],
) -> Vec<OrderViolation> {
    let mut out = Vec::new();
    let reference = &semantic_projection(reference);
    let candidate = &semantic_projection(candidate);

    // Clause 0: a correspondence between the two runs' entity names exists.
    // Without one, an implementation that allocates from a different partition
    // fails every later clause for a reason that has nothing to do with what it
    // did. With one, a dropped or merged entity fails *here*, which is the
    // stronger report.
    let names = match IdentityMap::between(reference, candidate) {
        Ok(names) => names,
        Err(violation) => {
            out.push(violation);
            return out;
        }
    };

    // Clause 1: same events, grouped by epoch. Comparing sorted multisets per
    // epoch is exactly "the same things happened in the same epoch, in some
    // order".
    let reference_by_epoch = group_by_epoch(reference, &IdentityMap::identity());
    let candidate_by_epoch = group_by_epoch(candidate, &names);
    for (epoch, expected) in &reference_by_epoch {
        match candidate_by_epoch.get(epoch) {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                out.push(OrderViolation::new(
                    OrderClause::ScheduleConformance,
                    format!(
                        "epoch {} ran {} events where the reference ran {}, or they differ in kind",
                        epoch,
                        actual.len(),
                        expected.len()
                    ),
                ));
            }
            None => out.push(OrderViolation::new(
                OrderClause::ScheduleConformance,
                format!("epoch {epoch} is missing entirely"),
            )),
        }
    }
    for epoch in candidate_by_epoch.keys() {
        if !reference_by_epoch.contains_key(epoch) {
            out.push(OrderViolation::new(
                OrderClause::ScheduleConformance,
                format!("epoch {epoch} has no counterpart in the reference run"),
            ));
        }
    }

    // Clause 1b: per-continuation sequence equality. A continuation is
    // sequential by definition (§1.1), so its own events must appear in the
    // reference's order — not merely be present.
    //
    // This is stated as a projection rather than as an edge because an edge
    // derived from the candidate's own positions can never be inverted: swap
    // two of a continuation's events and the derivation simply reads them in
    // the new order. Comparing against the reference is what makes program
    // order enforceable at all.
    let reference_by_continuation = group_by_continuation(reference, &IdentityMap::identity());
    let candidate_by_continuation = group_by_continuation(candidate, &names);
    for (continuation, expected) in &reference_by_continuation {
        match candidate_by_continuation.get(continuation) {
            Some(actual) if actual == expected => {}
            _ => out.push(OrderViolation::new(
                OrderClause::ScheduleConformance,
                format!(
                    "continuation {} ran its own events out of the reference's order",
                    slot(*continuation)
                ),
            )),
        }
    }

    // Clause 2: epochs never move backwards (I11's second half, restated here
    // because a concurrent implementation could satisfy I11's clock condition
    // while interleaving two epochs).
    for window in candidate.windows(2) {
        if window[1].epoch < window[0].epoch {
            out.push(OrderViolation::new(
                OrderClause::ScheduleConformance,
                format!(
                    "epoch moved backwards from {} to {}",
                    window[0].epoch, window[1].epoch
                ),
            ));
            break;
        }
    }

    // Clause 3: a linear extension of ≺ derived from the candidate's own
    // events. Deriving from the candidate rather than the reference is
    // deliberate — an implementation must respect the order implied by what it
    // did, not by what some other run did.
    let order = SemanticOrder::from_trace(candidate.to_vec());
    for edge in order.edges() {
        if edge.earlier >= edge.later {
            out.push(OrderViolation::new(
                OrderClause::ScheduleConformance,
                format!(
                    "{:?} edge is inverted: event {} must precede event {}",
                    edge.reason, edge.earlier, edge.later
                ),
            ));
        }
    }

    out.sort();
    out.dedup();
    out
}

/// Each continuation's own events, in trace order.
fn group_by_continuation(
    events: &[TraceSnapshotRow],
    names: &IdentityMap,
) -> BTreeMap<u64, Vec<EventIdentity>> {
    let mut grouped: BTreeMap<u64, Vec<EventIdentity>> = BTreeMap::new();
    for row in events {
        if row.continuation == 0 || slot(row.continuation) == 0 {
            continue;
        }
        if !names_a_continuation(row.event_kind) {
            continue;
        }
        grouped
            .entry(names.rename(row.continuation))
            .or_default()
            .push(EventIdentity::of_renamed(row, names));
    }
    grouped
}

fn group_by_epoch(
    events: &[TraceSnapshotRow],
    names: &IdentityMap,
) -> BTreeMap<u32, Vec<EventIdentity>> {
    let mut grouped: BTreeMap<u32, Vec<EventIdentity>> = BTreeMap::new();
    for row in events {
        grouped
            .entry(row.epoch)
            .or_default()
            .push(EventIdentity::of_renamed(row, names));
    }
    for group in grouped.values_mut() {
        group.sort();
    }
    grouped
}

/// **I19. Placement neutrality.**
///
/// Every run in `placements` must be I18-equivalent to the first. If changing
/// where work runs — territory assignment, cohort width, backend selection —
/// changes what a program observes, the placement layer has leaked into the
/// semantics.
///
/// This is the control `docs/HANDOFF.md` §6 demands, promoted to an invariant:
/// a result that only holds under one placement is a result about that
/// placement.
pub fn placement_neutral(placements: &[&Kernel]) -> Vec<OrderViolation> {
    let Some((reference, rest)) = placements.split_first() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (index, candidate) in rest.iter().enumerate() {
        for violation in conforms(reference, candidate) {
            out.push(OrderViolation::new(
                OrderClause::PlacementNeutrality,
                format!("placement {} diverges: {}", index + 1, violation.detail),
            ));
        }
    }
    out.sort();
    out.dedup();
    out
}
