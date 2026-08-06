//! The effect log (`docs/SOMA-v0.3.md` §4.4).
//!
//! §4.3 leaves canonical commit as the piece of B that gates a concurrent
//! executive, and names the obstacle: the executive's handlers take
//! `&mut Kernel` and allocate their effects as they run, so execute and commit
//! are fused. A step that *performs* its effects can only be run one at a time,
//! because the order two lanes touch shared state in is the order they happened
//! to be scheduled in.
//!
//! This module is the unfusing. A step no longer writes the scheduler's bins;
//! it **produces** the entries it wants written, and the kernel applies them
//! afterwards. "Apply" is a place — one function, over a list, with the order
//! it applies in supplied by the plan rather than by the clock.
//!
//! Application happens **at the end of the epoch** (§4.5). It happened at the
//! end of each lane when this module was written, which is exactly where a
//! sequential interpreter already wrote, so §4.4 could introduce the machinery
//! without changing any run. Moving the call out of the lane loop is canonical
//! commit: an epoch's bin entries now land in the order the plan puts the
//! producing lanes in and not in the order those lanes happened to run.
//!
//! What that costs is I25. No lane can observe another lane's bin entry or
//! status write any more, so a workload in which one lane depended on another
//! within an epoch is one this executive no longer reproduces — and I25 is the
//! checker asking, per run, whether the workload does that. §4.3 (3) measured
//! the answer and called it a precondition; it is a requirement now.
//!
//! ## What is mediated, and what is not
//!
//! Entry into a runnable bin, and the status transition that goes with it.
//! That is deliberately the smallest thing worth doing first and not an
//! arbitrary subset: v0.2 §3.4 makes commit the sole path to `Runnable`, which
//! is what makes I7 checkable, and the bins are the one structure every lane of
//! an epoch writes. Mailboxes, futures, capability spaces and the object tables
//! are still written as the step runs, and §4.3 (3) says why deferring them is
//! harder: a deferred mailbox pop changes what the *same* step reads back,
//! which is a semantic change and not a refactor.
//!
//! They are not the next slice, though, which is the thing §4.5 settled.
//! Deferring them would buy lane independence by construction; I25 buys it by
//! checking, and checking is what the model can actually afford — §4.3 (2)
//! establishes that allocation has to stay eager, so a step will keep writing
//! tables as it runs no matter how much of commit moves.
//!
//! Allocation stays eager, and that is not an omission either — §4.3 (2)
//! establishes that a step allocates entities and then stores them in opaque
//! frame bytes, so a symbolic reference cannot survive the step. Partitioned
//! allocation is what makes eager allocation safe for concurrent lanes.
//!
//! ## Why there is a record
//!
//! `Kernel::effect_log` keeps every applied effect with the position it was
//! produced at and the index it was applied at. In-crate, [`Committing`] already
//! makes an unmediated bin write impossible to compile. The record is what lets
//! I24 be asked of an implementation this crate did not run — the same reason
//! `AdmissionRecord` exists next to a sealed `Admission` (§4.1).

use std::collections::HashSet;

use crate::abi::continuations::ContinuationState;
use crate::abi::Ref64;
use crate::kernel::Kernel;

/// Proof that a runnable bin is being written by the effect applier.
///
/// [`crate::scheduler::runnable_bins::Scheduler::enqueue`] demands one, the
/// field is private, and [`apply`](Kernel::apply_lane_effects) is the only
/// place that constructs it. A step that writes a bin as it runs is therefore a
/// compile error rather than a test that might notice, which is the technique
/// `docs/SOMA-CAPABILITIES.md` used to close the operation set and §4.1 used to
/// seal `Admission`.
///
/// ```compile_fail
/// let _ = soma::kernel::effects::Committing(());
/// ```
pub struct Committing(());

/// One scheduling effect a step produced.
///
/// The four variants are the four shapes that exist in the kernel, not a
/// generalisation: they differ in how the continuation's status is written, and
/// collapsing them would change which continuations look long-waiting to
/// admission (§4.1 decides its claim on `waiting_since`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    /// Commit re-binds the continuation that just ran: its next run class, its
    /// status through `set_continuation_status`, and its bin.
    Resume { continuation: Ref64, run_class: u32 },
    /// A waiter woken by a resolution, a delivery, or released capacity. Its
    /// status is written directly: `Waiting` and `Runnable` are both live, so
    /// the live-continuation count does not move, and the wait it has already
    /// served is not reset.
    Wake { continuation: Ref64, run_class: u32 },
    /// A freshly created continuation takes its first bin. It is already
    /// `Runnable` — allocation is eager — so only the bin entry is deferred.
    Bin { continuation: Ref64, run_class: u32 },
    /// A continuation returned to its bin unrun: deferred by the I13 claim, or
    /// held back by a partial-cohort policy. Its status never changed.
    Requeue { continuation: Ref64, run_class: u32 },
}

impl Effect {
    pub(crate) fn continuation(&self) -> Ref64 {
        match self {
            Effect::Resume { continuation, .. }
            | Effect::Wake { continuation, .. }
            | Effect::Bin { continuation, .. }
            | Effect::Requeue { continuation, .. } => *continuation,
        }
    }

    fn run_class(&self) -> u32 {
        match self {
            Effect::Resume { run_class, .. }
            | Effect::Wake { run_class, .. }
            | Effect::Bin { run_class, .. }
            | Effect::Requeue { run_class, .. } => *run_class,
        }
    }

    fn kind(&self) -> EffectKind {
        match self {
            Effect::Resume { .. } => EffectKind::Resume,
            Effect::Wake { .. } => EffectKind::Wake,
            Effect::Bin { .. } => EffectKind::Bin,
            Effect::Requeue { .. } => EffectKind::Requeue,
        }
    }
}

/// Which shape an applied effect had. Public so the record is readable without
/// exposing the effect vocabulary itself, which is kernel-internal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectKind {
    Resume,
    Wake,
    Bin,
    Requeue,
}

/// One applied effect, with where it was produced and when it was applied.
///
/// `position` is the producing lane's, taken when the step produced the effect.
/// `applied` is the applier's, taken when the kernel wrote the bin. I24 asks
/// whether the second is the first sorted — that is what "in lane order" means
/// as a property of a record rather than as a property of a sequential
/// interpreter that never had the chance to do otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectRecord {
    pub epoch: u32,
    pub lane: u32,
    pub sequence: u32,
    pub applied: u64,
    pub kind: EffectKind,
    pub continuation: Ref64,
    pub run_class: u32,
}

impl EffectRecord {
    /// Where the step that produced this effect was in the epoch's plan.
    pub fn position(&self) -> (u32, u32, u32) {
        (self.epoch, self.lane, self.sequence)
    }
}

impl Kernel {
    /// Produce a scheduling effect.
    ///
    /// Inside a lane the effect is journalled and applied when the lane ends.
    /// Outside one — a caller building work between epochs, an epoch phase
    /// running on the host — there is no lane to order against, so it is
    /// applied at once. Both paths go through `apply_effect`, so both are
    /// recorded.
    pub(crate) fn emit(&mut self, effect: Effect) {
        // A continuation enters a bin at most once per epoch. While the status
        // write was immediate this was enforced by the guards at the production
        // sites (`wake_waiting_continuation` returns unless the continuation is
        // `Waiting`); with the write deferred those guards still see the old
        // status, so the deferred form of the guard lives here.
        //
        // The journal now spans the epoch rather than the lane, so this guard
        // widened with it — and that is the point rather than a side effect. A
        // continuation woken by lane 1 stays `Waiting` in the table until the
        // epoch ends, so lane 5 resolving a second future it waits on finds the
        // same stale status and would wake it again. Under the per-lane applier
        // lane 5 saw `Runnable` and stopped at the production site; the epoch
        // journal is where that check has to live once no lane can see another
        // lane's status write.
        if self
            .lane_effects
            .iter()
            .any(|(_, pending)| pending.continuation() == effect.continuation())
        {
            return;
        }
        let position = self.next_effect_position();
        if self.current_lane == crate::abi::traces::HOST_LANE || self.applying_effects {
            self.apply_effect(position, effect);
        } else {
            self.lane_effects.push((position, effect));
        }
    }

    /// Apply everything the epoch's lanes produced, in plan order.
    ///
    /// This is canonical commit. The applier used to run at the end of each
    /// lane, which is where a sequential interpreter already wrote, so the
    /// order effects landed in was the order the lanes happened to run in.
    /// Running it once, here, after every lane of the epoch has finished, makes
    /// the order a property of the plan instead: `position` is
    /// `(epoch, lane, sequence)`, and a lane number is decided before anything
    /// runs (§4.2).
    ///
    /// The sort is not decoration on a sequential interpreter — this crate
    /// appends in plan order already, so it is a no-op here for the same reason
    /// I24's clauses 1 and 2 are satisfied by construction. It is what the
    /// clause *means* for an implementation whose lanes append concurrently,
    /// where journal order is arrival order and the plan is the only thing that
    /// still says what "in lane order" was.
    pub(crate) fn apply_epoch_effects(&mut self) {
        if self.lane_effects.is_empty() {
            return;
        }
        let mut produced = std::mem::take(&mut self.lane_effects);
        produced.sort_by_key(|(position, _)| *position);
        self.applying_effects = true;
        for (position, effect) in produced {
            self.apply_effect(position, effect);
        }
        self.applying_effects = false;
    }

    /// Drop every effect the epoch has produced for one of `continuations`,
    /// before it is applied.
    ///
    /// Cancellation empties the bins of everything it cancels. While the
    /// applier ran per lane, "the bins" was the whole story: the entry had been
    /// made and `remove_all` took it out again. With the applier at the epoch
    /// boundary an entry produced this epoch has not landed anywhere yet, so
    /// emptying the bins is two moves — this one for what is still in the
    /// journal, and `remove_all` for what earlier epochs left. §4.4 named this
    /// as the alternative it did not take, on the grounds that no handler could
    /// reach it; the epoch-boundary applier is what makes it the only option.
    pub(crate) fn withdraw_effects(&mut self, continuations: &HashSet<Ref64>) {
        self.lane_effects
            .retain(|(_, effect)| !continuations.contains(&effect.continuation()));
    }

    fn apply_effect(&mut self, position: (u32, u32, u32), effect: Effect) {
        let (epoch, lane, sequence) = position;
        let applied = self.effect_log.len() as u64;
        self.effect_counters.emit();
        self.effect_log.push(EffectRecord {
            epoch,
            lane,
            sequence,
            applied,
            kind: effect.kind(),
            continuation: effect.continuation(),
            run_class: effect.run_class(),
        });

        let token = Committing(());
        match effect {
            Effect::Resume {
                continuation,
                run_class,
            } => {
                if let Ok(descriptor) = self.continuations.get_mut(continuation) {
                    descriptor.run_class = run_class;
                }
                self.set_continuation_status(continuation, ContinuationState::Runnable);
                self.scheduler.enqueue(run_class, continuation, &token);
            }
            Effect::Wake {
                continuation,
                run_class,
            } => {
                if let Ok(descriptor) = self.continuations.get_mut(continuation) {
                    descriptor.status = ContinuationState::Runnable;
                }
                self.scheduler.enqueue(run_class, continuation, &token);
            }
            Effect::Bin {
                continuation,
                run_class,
            }
            | Effect::Requeue {
                continuation,
                run_class,
            } => {
                self.scheduler.enqueue(run_class, continuation, &token);
            }
        }
    }

    /// Take the next `(epoch, lane, sequence)` for a produced effect.
    ///
    /// Structurally identical to `next_position`, and deliberately a separate
    /// counter: an effect's position says where in the plan it was produced,
    /// and stamping it from the trace counter would make the effect log's order
    /// a consequence of how many events happened to be emitted.
    fn next_effect_position(&mut self) -> (u32, u32, u32) {
        if self.current_lane == crate::abi::traces::HOST_LANE {
            let sequence = self.host_effect_sequence;
            self.host_effect_sequence = self.host_effect_sequence.saturating_add(1);
            (self.epoch, crate::abi::traces::HOST_LANE, sequence)
        } else {
            let sequence = self.lane_effect_sequence;
            self.lane_effect_sequence = self.lane_effect_sequence.saturating_add(1);
            (self.epoch, self.current_lane, sequence)
        }
    }
}
