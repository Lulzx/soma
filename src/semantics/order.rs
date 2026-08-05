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
    causal: u64,
}

impl EventIdentity {
    fn of(row: &TraceSnapshotRow) -> Self {
        Self {
            epoch: row.epoch,
            event_kind: row.event_kind,
            process: row.process,
            continuation: row.continuation,
            run_class: row.run_class,
            auxiliary: row.auxiliary,
            causal: row.causal,
        }
    }
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
    matches!(kind, EventKind::CohortCreated | EventKind::ContinuationPlaced)
}

/// The observable part of a trace: everything except placement reporting.
pub fn semantic_projection(events: &[TraceSnapshotRow]) -> Vec<TraceSnapshotRow> {
    events
        .iter()
        .filter(|row| !is_placement_event(row.event_kind))
        .copied()
        .collect()
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
                    (EdgeReason::FutureResolution, row.causal, row.continuation, 0),
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

    // Clause 1: same events, grouped by epoch. Comparing sorted multisets per
    // epoch is exactly "the same things happened in the same epoch, in some
    // order".
    let reference_by_epoch = group_by_epoch(reference);
    let candidate_by_epoch = group_by_epoch(candidate);
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
    let reference_by_continuation = group_by_continuation(reference);
    let candidate_by_continuation = group_by_continuation(candidate);
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
fn group_by_continuation(events: &[TraceSnapshotRow]) -> BTreeMap<u64, Vec<EventIdentity>> {
    let mut grouped: BTreeMap<u64, Vec<EventIdentity>> = BTreeMap::new();
    for row in events {
        if row.continuation == 0 || slot(row.continuation) == 0 {
            continue;
        }
        if !names_a_continuation(row.event_kind) {
            continue;
        }
        grouped
            .entry(row.continuation)
            .or_default()
            .push(EventIdentity::of(row));
    }
    grouped
}

fn group_by_epoch(events: &[TraceSnapshotRow]) -> BTreeMap<u32, Vec<EventIdentity>> {
    let mut grouped: BTreeMap<u32, Vec<EventIdentity>> = BTreeMap::new();
    for row in events {
        grouped
            .entry(row.epoch)
            .or_default()
            .push(EventIdentity::of(row));
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
