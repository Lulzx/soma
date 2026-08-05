//! The epoch lifecycle (§18): Ingest (nothing external yet) → Validate/Admit
//! (per-process serial guard) → Cohort → Execute → Commit → Account, then swap
//! the runnable-bin buffers and advance the epoch.
//!
//! Phase E (Cohort) partitions each bin's admitted continuations into width-`W`
//! SIMD dispatches (§14). Execution itself is still the CPU scalar executive, so
//! a cohort's lanes run one after another rather than simultaneously — but the
//! cohorts are real, and the lane-occupancy they imply is what §28.1 measures.
//! With the default `cohort_width` of 1 every dispatch is a single lane, which
//! reduces exactly to the pre-cohorting behaviour.

use std::collections::HashSet;

use crate::abi::cohorts::PartialCohortPolicy;
use crate::abi::continuations::ContinuationState;
use crate::abi::ProcessMode;
use crate::abi::Ref64;
use crate::executives::cpu_scalar;
use crate::kernel::commit;
use crate::kernel::Kernel;
use crate::scheduler::cohorts::{build_cohorts, CohortPlan};

impl Kernel {
    /// Run one epoch. Returns the number of continuation steps executed.
    ///
    /// The epoch boundary runs *first*: work produced last epoch (in the `next`
    /// buffer) is promoted to `current`, executed, and any work it produces
    /// lands in `next` for the following epoch (§13, §18).
    pub fn run_epoch(&mut self) -> usize {
        // Epoch boundary: promote `next` → `current` (§18 Phase H → next epoch).
        self.scheduler.swap_all();

        // Phases B/C: Validate and Admit. Everything admitted this epoch is
        // collected per bin before any of it runs, so cohort construction sees
        // the whole epoch's eligible work rather than a prefix of it.
        //
        // Serial-process invariant (§19): at most one mutating continuation of a
        // serial process runs per epoch. Sequential execution cannot overlap two
        // continuations *within* a step, but a parallel executive would run a
        // whole epoch's cohorts concurrently — so the invariant must hold at
        // epoch granularity, and a slot claimed here is held for the rest of the
        // epoch. Later continuations of the same process defer to the next epoch.
        let mut claimed_procs: HashSet<u32> = HashSet::new();
        let mut admitted: Vec<(u32, Vec<(Ref64, u32)>)> = Vec::new();

        let bins: Vec<u32> = self
            .scheduler
            .runnable_counts()
            .iter()
            .map(|(b, _)| *b)
            .collect();
        for bin in bins {
            let mut lanes: Vec<(Ref64, u32)> = Vec::new();
            for cont in self.scheduler.drain(bin) {
                let (process, run_class, mode, status) = match self.continuations.get(cont) {
                    Ok(c) => (
                        c.process,
                        c.run_class,
                        self.process_mode(c.process),
                        c.status,
                    ),
                    Err(_) => continue,
                };
                // Only runnable continuations execute. Nothing in the current
                // single-threaded path enqueues a non-runnable continuation —
                // the budget check faults before any commit can requeue — so
                // this is a guard for the executives to come, where a cohort can
                // be cancelled or faulted after its bin was filled.
                if status != ContinuationState::Runnable {
                    continue;
                }
                let mutating = mode == ProcessMode::Serial || mode == ProcessMode::System;
                if mutating && !claimed_procs.insert(process.slot) {
                    self.scheduler.enqueue(run_class, cont);
                    self.accounting.serial_deferrals += 1;
                    continue;
                }
                lanes.push((cont, run_class));
            }
            if !lanes.is_empty() {
                admitted.push((bin, lanes));
            }
        }

        // Phase E: Cohort (§14).
        let mut plans: Vec<CohortPlan> = admitted
            .iter()
            .map(|(_, lanes)| build_cohorts(lanes, self.cohort_width, self.partial_policy))
            .collect();

        // Forward-progress guard for `Defer`: if the policy held everything back
        // this epoch, nothing would run and the next epoch would face the same
        // choice forever. Re-plan under `RunPartial` so the epoch does work.
        if plans.iter().all(|p| p.is_empty()) && !admitted.is_empty() {
            plans = admitted
                .iter()
                .map(|(_, lanes)| {
                    build_cohorts(lanes, self.cohort_width, PartialCohortPolicy::RunPartial)
                })
                .collect();
        }

        // Phase F: Execute (CPU scalar — a cohort's lanes run in lane order).
        let mut steps = 0;
        for plan in &plans {
            for cohort in &plan.cohorts {
                self.accounting.cohorts += 1;
                self.accounting.lane_slots += cohort.width as u64;
                self.accounting.useful_lane_slots += cohort.active_lanes as u64;
                self.accounting.idle_lane_slots += cohort.idle_lanes() as u64;
                if cohort.is_full() {
                    self.accounting.full_cohorts += 1;
                }
                self.trace(
                    crate::abi::EventKind::CohortCreated,
                    Ref64::NULL,
                    *cohort.lanes().first().unwrap_or(&Ref64::NULL),
                    cohort.run_class,
                    cohort.active_lanes as u32,
                );

                for cont in cohort.lanes() {
                    let process = match self.continuations.get(*cont) {
                        Ok(c) => c.process,
                        Err(_) => continue,
                    };
                    steps += self.execute_cont(*cont, process);
                }
            }
            // Deferred lanes return to their bins for a later epoch.
            for cont in &plan.deferred {
                let run_class = self.continuations.get(*cont).map(|c| c.run_class).unwrap_or(0);
                self.scheduler.enqueue(run_class, *cont);
                self.accounting.deferred_lanes += 1;
            }
        }

        // Phase H: Account.
        let total = self.scheduler.total_pending();
        self.epoch_runnable.push(total);
        self.accounting.epochs += 1;
        self.accounting.steps += steps as u64;

        self.epoch = self.epoch.wrapping_add(1);

        steps
    }

    /// Run epochs until no work remains anywhere (bounded by `max_epochs`).
    /// Returns the number of epochs run.
    pub fn run_to_quiescence(&mut self, max_epochs: u32) -> u32 {
        let mut n = 0;
        while self.scheduler.total_pending() > 0 && n < max_epochs {
            self.run_epoch();
            n += 1;
        }
        n
    }

    /// Total work outstanding anywhere (current + next buffers).
    pub fn total_pending(&self) -> usize {
        self.scheduler.total_pending()
    }

    /// Per-class counts of all pending work, in class order.
    pub fn pending_counts(&self) -> Vec<(u32, usize)> {
        self.scheduler.pending_counts()
    }

    /// Total runnable continuations across all classes.
    pub fn total_runnable(&self) -> usize {
        self.scheduler.total_runnable()
    }

    /// Per-class runnable counts for the current (pre-swap) epoch.
    pub fn runnable_counts(&self) -> Vec<(u32, usize)> {
        self.scheduler.runnable_counts()
    }

    fn process_mode(&self, p: Ref64) -> ProcessMode {
        self.processes
            .get(p)
            .map(|pd| pd.process_mode)
            .unwrap_or(ProcessMode::System)
    }

    /// Execute a single continuation: enforce the step budget, dispatch to the
    /// interpreter, and commit the result.
    fn execute_cont(&mut self, cont: Ref64, process: Ref64) -> usize {
        let (run_class, remaining) = self
            .continuations
            .get(cont)
            .map(|c| (c.run_class, c.remaining_steps))
            .unwrap_or((0, 0));

        // Step budget (§8): a continuation must not exceed its declared maximum.
        // The check happens *before* dispatch, so an exhausted continuation is
        // faulted without running and — crucially — without having first been
        // re-enqueued by a commit. Faulting after the commit would leave a
        // faulted continuation sitting live in a runnable bin.
        if remaining == 0 {
            let over = crate::abi::StepResult::fault(process, run_class);
            let _ = commit::apply_step_result(self, cont, process, over);
            return 0;
        }

        self.trace(
            crate::abi::EventKind::ContinuationStarted,
            process,
            cont,
            run_class,
            0,
        );
        let result = cpu_scalar::dispatch(self, cont, process);
        let consumed = commit::apply_step_result(self, cont, process, result);

        if let Ok(c) = self.continuations.get_mut(cont) {
            c.remaining_steps = c.remaining_steps.saturating_sub(consumed as u32);
        }

        consumed
    }
}
