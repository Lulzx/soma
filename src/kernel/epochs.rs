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
use crate::abi::{EventKind, ProcessState, Ref64, StepKind, StepResult};
use crate::executives::cpu_scalar;
use crate::kernel::commit;
use crate::kernel::payload::Payload;
use crate::kernel::speculation::{EpochExecutive, LaneJournal, LaneOperation, Resource};
use crate::kernel::Kernel;
use crate::scheduler::admission::{admit, AdmissionRecord, Candidate};
use crate::scheduler::cohorts::{build_cohorts, CohortPlan};

#[derive(Clone, Copy, Debug)]
struct EvaluatedStep {
    result: StepResult,
    executed: bool,
}

#[derive(Clone, Debug)]
struct SpeculativeLane {
    lane: u32,
    continuation: Ref64,
    process: Ref64,
    evaluated: EvaluatedStep,
    journal: LaneJournal,
    payloads: Vec<(Ref64, Vec<u8>)>,
}

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

        let steps = match self.epoch_executive {
            EpochExecutive::Speculative { max_lanes } => self
                .execute_lanes_speculatively(&lanes, max_lanes)
                .unwrap_or_else(|| self.execute_lanes_reference(&lanes)),
            EpochExecutive::Reference => self.execute_lanes_reference(&lanes),
        };

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
        let Some(evaluated) = self.evaluate_cont(cont, process) else {
            return 0;
        };
        self.commit_evaluated(cont, process, evaluated)
    }

    fn evaluate_cont(&mut self, cont: Ref64, process: Ref64) -> Option<EvaluatedStep> {
        if self
            .continuations
            .get(cont)
            .map(|descriptor| descriptor.status)
            .ok()
            != Some(ContinuationState::Runnable)
        {
            return None;
        }
        // `frame` joins the three because the step needs it and no longer has a
        // way to look it up: `LaneView` stopped offering the continuation table
        // in §4.17, so the frame reference is read here, once, on the host's
        // side of the epoch, and carried into the lane.
        let (run_class, remaining, dependency, frame) = self
            .continuations
            .get(cont)
            .map(|c| (c.run_class, c.remaining_steps, c.dependency, c.frame))
            .unwrap_or((0, 0, Ref64::NULL, Ref64::NULL));

        // Step budget (§8): a continuation must not exceed its declared maximum.
        // The check happens *before* dispatch, so an exhausted continuation is
        // faulted without running and — crucially — without having first been
        // re-enqueued by a commit. Faulting after the commit would leave a
        // faulted continuation sitting live in a runnable bin.
        if remaining == 0 {
            return Some(EvaluatedStep {
                result: crate::abi::StepResult::fault(process, run_class),
                executed: false,
            });
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
            return Some(EvaluatedStep {
                result: crate::abi::StepResult::fault(process, run_class),
                executed: false,
            });
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
        //
        // The run class is passed rather than looked up. It is the same value
        // `dispatch` used to read back out of the descriptor for itself, and
        // nothing between that read and the one above can change it — which is
        // why §4.17 could take the read away without changing a run.
        let result = {
            let mut lane = crate::executives::lane::LaneView::new(self, frame);
            cpu_scalar::dispatch(&mut lane, cont, process, run_class)
        };
        if let Ok(descriptor) = self.processes.get_mut(process) {
            descriptor.active_continuation = Ref64::NULL;
        }
        Some(EvaluatedStep {
            result,
            executed: true,
        })
    }

    fn commit_evaluated(&mut self, cont: Ref64, process: Ref64, evaluated: EvaluatedStep) -> usize {
        let consumed = commit::apply_step_result(self, cont, process, evaluated.result);
        if evaluated.executed {
            if let Ok(c) = self.continuations.get_mut(cont) {
                c.remaining_steps = c.remaining_steps.saturating_sub(consumed as u32);
            }
            consumed
        } else {
            0
        }
    }

    fn execute_lanes_reference(&mut self, lanes: &[(u32, Ref64)]) -> usize {
        let mut steps = 0;
        for &(lane, cont) in lanes {
            let process = match self.continuations.get(cont) {
                Ok(c) => c.process,
                Err(_) => continue,
            };
            self.enter_lane(lane);
            steps += self.execute_cont(cont, process);
            self.leave_lane();
        }
        steps
    }

    fn execute_lanes_speculatively(
        &mut self,
        lanes: &[(u32, Ref64)],
        max_lanes: usize,
    ) -> Option<usize> {
        if lanes.len() < 2 || lanes.len() > max_lanes.max(1) {
            return None;
        }

        self.speculation_stats.attempted_epochs += 1;
        self.speculation_stats.speculative_lanes += lanes.len() as u64;

        // Each worker receives the exact same pre-Phase-F state. Cloning is
        // intentionally outside the timed handler threads: it is isolation,
        // not work whose completion order may affect a lane.
        let snapshots: Vec<_> = lanes.iter().map(|_| self.clone()).collect();
        let outcomes = std::thread::scope(|scope| {
            let handles: Vec<_> = snapshots
                .into_iter()
                .zip(lanes.iter().copied())
                .map(|(mut snapshot, (lane, continuation))| {
                    scope.spawn(move || {
                        let process = snapshot.continuations.get(continuation).ok()?.process;
                        let process_descriptor = snapshot.processes.get(process).ok()?.clone();
                        snapshot.enter_lane(lane);
                        snapshot.begin_speculative_recording();
                        snapshot.record_speculative_write(Resource::Process(process));
                        if !process_descriptor.supervisor.is_null() {
                            snapshot.record_speculative_write(Resource::Process(
                                process_descriptor.supervisor,
                            ));
                        }
                        if process_descriptor.status == ProcessState::CancelPending as u32 {
                            snapshot.mark_speculation_unsupported();
                        }
                        let evaluated = snapshot.evaluate_cont(continuation, process)?;
                        let mut journal = snapshot.finish_speculative_recording();
                        if evaluated.result.kind == StepKind::Fault {
                            journal.unsupported = true;
                        }
                        let payloads = journal
                            .mutated_objects
                            .iter()
                            .filter_map(|object| {
                                snapshot
                                    .object_payloads
                                    .get(&object.key())
                                    .map(|payload| (*object, payload.as_slice().to_vec()))
                            })
                            .collect();
                        Some(SpeculativeLane {
                            lane,
                            continuation,
                            process,
                            evaluated,
                            journal,
                            payloads,
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().ok().flatten())
                .collect::<Option<Vec<_>>>()
        });

        let Some(mut outcomes) = outcomes else {
            self.speculation_stats.fallback_epochs += 1;
            self.speculation_stats.unsupported_fallbacks += 1;
            return None;
        };

        if outcomes.iter().any(|outcome| outcome.journal.unsupported) {
            self.speculation_stats.fallback_epochs += 1;
            self.speculation_stats.unsupported_fallbacks += 1;
            return None;
        }
        for left in 0..outcomes.len() {
            for right in left + 1..outcomes.len() {
                if outcomes[left]
                    .journal
                    .conflicts_with(&outcomes[right].journal)
                {
                    self.speculation_stats.fallback_epochs += 1;
                    self.speculation_stats.conflict_fallbacks += 1;
                    return None;
                }
            }
        }

        // Execution order is deliberately discarded here. Lane number is the
        // canonical position assigned by the epoch plan.
        outcomes.sort_by_key(|outcome| outcome.lane);
        // Validate replay against a disposable copy before touching the real
        // kernel. A mismatch means an access declaration was incomplete; no
        // partial canonical commit may escape in that case.
        let mut validation = self.clone();
        if validation.commit_speculative_lanes(&outcomes).is_none() {
            self.speculation_stats.fallback_epochs += 1;
            self.speculation_stats.unsupported_fallbacks += 1;
            return None;
        }
        let steps = self
            .commit_speculative_lanes(&outcomes)
            .expect("validated speculative replay must remain deterministic");
        self.speculation_stats.committed_epochs += 1;
        self.speculation_stats.committed_lanes += lanes.len() as u64;
        Some(steps)
    }

    fn commit_speculative_lanes(&mut self, outcomes: &[SpeculativeLane]) -> Option<usize> {
        let mut steps = 0;
        for outcome in outcomes {
            self.enter_lane(outcome.lane);
            let (run_class, dependency) = self
                .continuations
                .get(outcome.continuation)
                .map(|continuation| (continuation.run_class, continuation.dependency))
                .ok()?;
            if dependency.kind == crate::abi::Kind::Future
                && !dependency.is_null()
                && self
                    .authorize(outcome.process, crate::abi::Rights::AWAIT, dependency)
                    .is_err()
            {
                return None;
            }
            self.trace(
                EventKind::ContinuationStarted,
                outcome.process,
                outcome.continuation,
                run_class,
                0,
            );
            if let Ok(process) = self.processes.get_mut(outcome.process) {
                process.active_continuation = outcome.continuation;
            }
            if !self.replay_lane_operations(&outcome.journal.operations) {
                return None;
            }
            if let Ok(process) = self.processes.get_mut(outcome.process) {
                process.active_continuation = Ref64::NULL;
            }
            for (object, bytes) in &outcome.payloads {
                self.object_payloads
                    .insert(object.key(), Payload::Host(bytes.clone()));
            }
            steps +=
                self.commit_evaluated(outcome.continuation, outcome.process, outcome.evaluated);
            self.leave_lane();
        }
        Some(steps)
    }

    fn replay_lane_operations(&mut self, operations: &[LaneOperation]) -> bool {
        operations.iter().all(|operation| match operation {
            LaneOperation::ObserveFuture {
                actor,
                future,
                result,
            } => self.observe_future(*actor, *future) == *result,
            LaneOperation::ReadObject { actor, object } => {
                let _ = self.object_bytes(*actor, *object);
                true
            }
            LaneOperation::CreateProcess {
                actor,
                mode,
                result,
            } => self.try_create_process(*actor, *mode) == *result,
            LaneOperation::CreateContinuation {
                actor,
                process,
                spec,
                result,
            } => self.create_continuation(*actor, *process, spec.clone()) == *result,
            LaneOperation::CreateFuture { actor, result } => self.create_future(*actor) == *result,
            LaneOperation::CreateObject {
                actor,
                kind,
                bytes,
                result,
            } => self.create_object(*actor, *kind, bytes.clone()) == *result,
            LaneOperation::WriteObject {
                actor,
                object,
                growable,
            } => {
                if *growable {
                    let _ = self.host_payload_mut(*actor, *object);
                } else {
                    let _ = self.object_bytes_mut(*actor, *object);
                }
                true
            }
            LaneOperation::EnqueueMessage {
                actor,
                receiver,
                payload,
                sender_continuation,
                result,
            } => self.enqueue_message(*actor, *receiver, *payload, *sender_continuation) == *result,
            LaneOperation::ReceiveMessage {
                actor,
                continuation,
                result,
            } => self.receive_message(*actor, *continuation) == *result,
            LaneOperation::ResolveFuture {
                actor,
                future,
                value,
                result,
            } => self.resolve_future(*actor, *future, *value) == *result,
            LaneOperation::AwaitFuture {
                actor,
                continuation,
                future,
                next_run_class,
                result,
            } => self.await_future(*actor, *continuation, *future, *next_run_class) == *result,
        })
    }
}
