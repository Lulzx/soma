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
//! afterwards. Application happens at the end of the lane today, which is
//! exactly where a sequential interpreter already was, so no run changes. What
//! changes is that "apply" is now a place — one function, over a list, with the
//! order it applies in supplied by the plan rather than by the clock. Moving
//! that call from the end of a lane to the end of an epoch is what canonical
//! commit is, and it is a change to *one line* of `epochs.rs` once the read
//! visibility of §4.3 (3) is settled.
//!
//! ## What is mediated, and what is not
//!
//! Entry into a runnable bin, and the status transition that goes with it.
//! That is deliberately the smallest thing worth doing first and not an
//! arbitrary subset: v0.2 §3.4 makes commit the sole path to `Runnable`, which
//! is what makes I7 checkable, and the bins are the one structure every lane of
//! an epoch writes. Mailboxes, futures, capability spaces and the object tables
//! are still written as the step runs. They are the next slice, and §4.3 (3)
//! says why they are harder: deferring a mailbox pop changes what the *same*
//! step reads back, which is a semantic change and not a refactor.
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
        // A continuation enters a bin at most once per lane. While the status
        // write was immediate this was enforced by the guards at the production
        // sites (`wake_waiting_continuation` returns unless the continuation is
        // `Waiting`); with the write deferred those guards still see the old
        // status, so the deferred form of the guard lives here.
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

    /// Apply everything the lane produced, in the order it produced it.
    ///
    /// Called between `enter_lane` and `leave_lane` so that anything the
    /// application traces is still attributed to the lane that caused it (I23
    /// clause 3).
    pub(crate) fn apply_lane_effects(&mut self) {
        if self.lane_effects.is_empty() {
            return;
        }
        let produced = std::mem::take(&mut self.lane_effects);
        self.applying_effects = true;
        for (position, effect) in produced {
            self.apply_effect(position, effect);
        }
        self.applying_effects = false;
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
