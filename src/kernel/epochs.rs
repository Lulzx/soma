//! The epoch lifecycle (§18): Ingest (nothing external yet) → Validate/Admit
//! (per-process mutable-state guard) → Cohort → Execute → Commit → Account,
//! then swap the runnable-bin buffers and advance the epoch.
//!
//! Phase E (Cohort) partitions each bin's admitted continuations into width-`W`
//! SIMD dispatches (§14). Execution itself is still the CPU scalar executive, so
//! a cohort's lanes run one after another rather than simultaneously — but the
//! cohorts are real, and the lane-occupancy they imply is what §28.1 measures.
//! With the default `cohort_width` of 1 every dispatch is a single lane, which
//! reduces exactly to the pre-cohorting behaviour.

use crate::abi::cohorts::PartialCohortPolicy;
use crate::abi::continuations::ContinuationState;
use crate::abi::Ref64;
use crate::executives::cpu_scalar;
use crate::kernel::commit;
use crate::kernel::Kernel;
use crate::scheduler::admission::{admit, AdmissionRecord, Candidate};
use crate::scheduler::cohorts::{build_cohorts, CohortPlan};

impl Kernel {
    /// Run one epoch. Returns the number of continuation steps executed.
    ///
    /// The epoch boundary runs *first*: work produced last epoch (in the `next`
    /// buffer) is promoted to `current`, executed, and any work it produces
    /// lands in `next` for the following epoch (§13, §18).
    pub fn run_epoch(&mut self) -> usize {
        // Under a bounded retention policy the logs carry only the epoch that
        // just ran, so this is where last epoch's records stop being the
        // caller's to drain. Under the default `Retain` it does nothing. A
        // dropped record is counted (`kernel::retention`), so a run that meant
        // to stream its whole trace can tell that it did not.
        self.release_epoch_logs();

        // Epoch boundary: promote `next` → `current` (§18 Phase H → next epoch).
        self.scheduler.swap_all();

        // Phase B: Validate. Drain every bin into one candidate set before
        // anything is decided, so cohort construction sees the whole epoch's
        // eligible work rather than a prefix of it.
        let bins: Vec<u32> = self
            .scheduler
            .runnable_counts()
            .iter()
            .map(|(b, _)| *b)
            .collect();
        let mut candidates: Vec<Candidate> = Vec::new();
        for bin in bins {
            for cont in self.scheduler.drain(bin) {
                let Ok(descriptor) = self.continuations.get(cont) else {
                    continue;
                };
                // Only runnable continuations execute. Nothing in the current
                // single-threaded path enqueues a non-runnable continuation —
                // the budget check faults before any commit can requeue — so
                // this is a guard for the executives to come, where a cohort can
                // be cancelled or faulted after its bin was filled.
                if descriptor.status != ContinuationState::Runnable {
                    continue;
                }
                candidates.push(Candidate {
                    bin,
                    continuation: cont,
                    process: descriptor.process,
                    run_class: descriptor.run_class,
                    state_access: descriptor.state_access,
                    waiting_since: descriptor.last_run_epoch.max(descriptor.created_epoch),
                });
            }
        }

        // Phase C: Admit. I13 permits at most one continuation declaring
        // mutable state access to run per process per epoch. Sequential
        // execution cannot overlap two continuations *within* a step, but a
        // parallel executive runs a whole epoch's cohorts concurrently, so the
        // invariant holds at epoch granularity: a claim taken here stands for
        // the rest of the epoch and the losers defer to the next one.
        //
        // The decision is `scheduler::admission`'s, computed from the candidate
        // set rather than taken as the scan walks it. On a device that scan is
        // a concurrent claim, and a rule that resolves it by arrival resolves it
        // by race (v0.3 §4). I22 checks that the decision survives permutation
        // of the candidates.
        let decision = admit(&candidates);
        self.admission_counters.emit();
        self.admission_log.push(AdmissionRecord {
            candidates,
            decision: decision.clone(),
        });

        for (run_class, cont) in decision.deferred() {
            self.emit(crate::kernel::effects::Effect::Requeue {
                continuation: *cont,
                run_class: *run_class,
            });
            self.accounting.serial_deferrals += 1;
        }
        let admitted = decision.into_bins();

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

        // I21, first half: an epoch that admitted work must dispatch some of
        // it. The re-plan above is what makes this hold under `Defer`; the
        // counter exists so that a future policy which withholds work cannot
        // do so silently. A deferral policy may delay work, not withhold it.
        if !admitted.is_empty() && plans.iter().all(|plan| plan.is_empty()) {
            self.accounting.stalled_epochs += 1;
        }

        // Phase E, recorded: the whole epoch's dispatch shape, before any of it
        // runs. These used to be emitted one cohort at a time as execution
        // reached it, which made a placement record depend on when a lane ran.
        // The plan is complete before Phase F starts, so the records belong to
        // the host's part of the epoch and are emitted there (I23).
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
            }
            // Lanes the partial-cohort policy held back return to their bins.
            // Which they are is a property of the plan, so this belongs here
            // rather than after execution, for the reason the cohort records
            // moved here in §4.2: a host effect produced after the lanes ran
            // would carry a position that sorts ahead of theirs and be applied
            // behind them, which is I24's clause 2 exactly.
            for cont in &plan.deferred {
                let run_class = self
                    .continuations
                    .get(*cont)
                    .map(|c| c.run_class)
                    .unwrap_or(0);
                self.emit(crate::kernel::effects::Effect::Requeue {
                    continuation: *cont,
                    run_class,
                });
                self.accounting.deferred_lanes += 1;
            }
        }

        // Phase F: Execute (CPU scalar — a cohort's lanes run in lane order).
        //
        // A lane's number comes from its position in the plan, not from when it
        // ran, and every event it emits is stamped with that number and a
        // counter local to it. Nothing here consults a shared clock, so a
        // concurrent executive can run these lanes in any order and the trace
        // still sorts back into this one.
        // The plan is numbered first and walked second, so that the order the
        // executive runs lanes in is a separate decision from what their
        // numbers are. A lane's number is its position here, before any of
        // this runs, which is what keeps a reordered run comparable to a
        // plan-order one: the same continuation gets the same number, the same
        // allocation partition, and the same position space either way.
        let mut lanes: Vec<(u32, Ref64)> = Vec::new();
        let mut lane_number = 0u32;
        for plan in &plans {
            for cohort in &plan.cohorts {
                for cont in cohort.lanes() {
                    lane_number += 1;
                    lanes.push((lane_number, *cont));
                }
            }
        }
        self.lane_order.arrange(&mut lanes, self.epoch);

        let mut steps = 0;
        for (lane_number, cont) in lanes {
            let process = match self.continuations.get(cont) {
                Ok(c) => c.process,
                Err(_) => continue,
            };
            self.enter_lane(lane_number);
            steps += self.execute_cont(cont, process);
            self.leave_lane();
        }

        // Phase G: Commit. Every lane of the epoch has finished; apply what
        // they produced, in plan order.
        //
        // This call used to sit inside the loop above, one lane at a time,
        // which is where a sequential interpreter already wrote and is why
        // §4.4 could say no run changed. Out here it is canonical commit: the
        // order the epoch's bin entries land in comes from the plan and not
        // from the order the lanes ran, so a lane may be run whenever and
        // wherever and the epoch commits the same.
        //
        // What this costs is stated as an invariant rather than left implicit.
        // No lane can now observe another lane's bin entry or status write, so
        // a run in which one lane's behaviour depended on another's is a run
        // this executive no longer reproduces. §4.3 (3) measured that no ≺ edge
        // joins two lanes of one epoch and was careful to call that a
        // precondition to check per run; I25 is that precondition promoted to
        // something the checker asks, which is what moving this line requires.
        self.apply_epoch_effects();

        // Phase H: Account.
        let total = self.scheduler.total_pending();
        self.epoch_runnable.push(total);
        self.accounting.epochs += 1;
        self.accounting.steps += steps as u64;

        self.epoch = self.epoch.wrapping_add(1);
        self.open_epoch_positions();

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

    /// Execute a single continuation: enforce the step budget, dispatch to the
    /// interpreter, and commit the result.
    fn execute_cont(&mut self, cont: Ref64, process: Ref64) -> usize {
        if self
            .continuations
            .get(cont)
            .map(|descriptor| descriptor.status)
            .ok()
            != Some(ContinuationState::Runnable)
        {
            return 0;
        }
        let (run_class, remaining, dependency) = self
            .continuations
            .get(cont)
            .map(|c| (c.run_class, c.remaining_steps, c.dependency))
            .unwrap_or((0, 0, Ref64::NULL));

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

        // Authority is checked again at resume, not captured when the
        // continuation parked. Revoking AWAIT therefore takes effect before
        // any resumed instruction executes (§5.1).
        if dependency.kind == crate::abi::Kind::Future
            && !dependency.is_null()
            && self
                .authorize(process, crate::abi::Rights::AWAIT, dependency)
                .is_err()
        {
            let denied = crate::abi::StepResult::fault(process, run_class);
            let _ = commit::apply_step_result(self, cont, process, denied);
            return 0;
        }

        self.trace(
            crate::abi::EventKind::ContinuationStarted,
            process,
            cont,
            run_class,
            0,
        );
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.active_continuation = cont;
        }
        // The step gets a lane, not the kernel. `LaneView` offers the fifteen
        // operations a handler actually performs and nothing else, so an
        // operation with no lane-local form is a compile error inside a step
        // rather than something to discover by audit later (v0.3 §4.10).
        let result = {
            let mut lane = crate::executives::lane::LaneView::new(self);
            cpu_scalar::dispatch(&mut lane, cont, process)
        };
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.active_continuation = Ref64::NULL;
        }
        let consumed = commit::apply_step_result(self, cont, process, result);

        if let Ok(c) = self.continuations.get_mut(cont) {
            c.remaining_steps = c.remaining_steps.saturating_sub(consumed as u32);
        }

        consumed
    }
}
