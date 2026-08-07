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
use crate::scheduler::device::{
    reference_lane_conflicts, DeviceEpochBackend, DeviceEvaluatorLane, LaneConflictValidator,
};
use crate::scheduler::device_ops::{
    await_result, decode_spec, message_result, object_kind, observe_result, process_mode,
    ref_result, unit_result, DeviceOperationJournal, OP_AWAIT_FUTURE, OP_CREATE_CONTINUATION,
    OP_CREATE_FUTURE, OP_CREATE_OBJECT, OP_CREATE_PROCESS, OP_ENQUEUE_MESSAGE, OP_OBSERVE_FUTURE,
    OP_READ_OBJECT, OP_RECEIVE_MESSAGE, OP_RESOLVE_FUTURE, OP_WRITE_OBJECT, RESULT_OK,
};

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
    device_operations: DeviceOperationJournal,
    payloads: Vec<(Ref64, Vec<u8>)>,
}

impl Kernel {
    /// Run one epoch. Returns the number of continuation steps executed.
    ///
    /// The epoch boundary runs *first*: work produced last epoch (in the `next`
    /// buffer) is promoted to `current`, executed, and any work it produces
    /// lands in `next` for the following epoch (§13, §18).
    pub fn run_epoch(&mut self) -> usize {
        self.run_epoch_validated(None, None)
    }

    /// Run one epoch and use `validator` for the complete speculative lane
    /// journal before canonical commit. A validation error or conflict causes
    /// the same whole-epoch reference fallback as the in-process validator.
    pub fn run_epoch_with_lane_validator(
        &mut self,
        validator: &mut dyn LaneConflictValidator,
    ) -> usize {
        self.run_epoch_validated(Some(validator), None)
    }

    /// Run one epoch with a physical backend that evaluates supported handler
    /// bodies and validates their complete journals before canonical commit.
    pub fn run_epoch_with_device_backend(&mut self, backend: &mut dyn DeviceEpochBackend) -> usize {
        self.run_epoch_validated(None, Some(backend))
    }

    fn run_epoch_validated(
        &mut self,
        mut validator: Option<&mut dyn LaneConflictValidator>,
        mut backend: Option<&mut dyn DeviceEpochBackend>,
    ) -> usize {
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
            EpochExecutive::Speculative { max_lanes } => {
                let speculative = match backend.take() {
                    Some(backend) => {
                        self.execute_lanes_speculatively(&lanes, max_lanes, None, Some(backend))
                    }
                    None => match validator.take() {
                        Some(validator) => self.execute_lanes_speculatively(
                            &lanes,
                            max_lanes,
                            Some(validator),
                            None,
                        ),
                        None => self.execute_lanes_speculatively(&lanes, max_lanes, None, None),
                    },
                };
                speculative.unwrap_or_else(|| self.execute_lanes_reference(&lanes))
            }
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

        if self.remote_lane_programs.contains_key(&run_class) {
            let entries = self
                .remote_lane_outbox_base_entries
                .saturating_add(self.remote_lane_emissions.len());
            let bytes = self.remote_lane_outbox_base_bytes.saturating_add(
                self.remote_lane_emissions
                    .iter()
                    .map(|emission| emission.batch.encode().len())
                    .sum::<usize>(),
            );
            if entries >= crate::distributed::remote_lane_effect::MAX_REMOTE_LANE_OUTBOX_ENTRIES
                || bytes.saturating_add(
                    crate::distributed::remote_lane_effect::MAX_REMOTE_LANE_PROGRAM_FRAME_BYTES,
                ) > crate::distributed::remote_lane_effect::MAX_REMOTE_LANE_OUTBOX_BYTES
            {
                return Some(EvaluatedStep {
                    result: crate::abi::StepResult::fault(process, run_class),
                    executed: false,
                });
            }
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
        let frame_evaluator = self.frame_evaluators.get(&run_class).cloned();
        let frame_effect = self.frame_evaluator_effects.get(&run_class).copied();
        let remote_program = self.remote_lane_programs.get(&run_class).cloned();
        let remote_completed = self.remote_lane_completed.remove(&cont.key());
        let remote_failed = self.remote_lane_failed.remove(&cont.key());
        let mut remote_instantiated = if let Some(program) = remote_program.as_ref() {
            if remote_completed || remote_failed {
                Ok(None)
            } else {
                program
                    .instantiate(self.epoch, self.current_lane, process, |target, send| {
                        let key = (
                            process.to_u64(),
                            target.node.0,
                            target.entity.to_u64(),
                            send,
                        );
                        let sequence = self.remote_lane_sequences.entry(key).or_insert(0);
                        let value = *sequence;
                        *sequence = sequence.saturating_add(1);
                        value
                    })
                    .map(Some)
            }
        } else {
            Ok(None)
        };
        if let Ok(Some(batch)) = &remote_instantiated {
            let entries = self
                .remote_lane_outbox_base_entries
                .saturating_add(self.remote_lane_emissions.len())
                .saturating_add(1);
            let bytes = self
                .remote_lane_outbox_base_bytes
                .saturating_add(
                    self.remote_lane_emissions
                        .iter()
                        .map(|emission| emission.batch.encode().len())
                        .sum::<usize>(),
                )
                .saturating_add(batch.encode().len());
            if entries > crate::distributed::remote_lane_effect::MAX_REMOTE_LANE_OUTBOX_ENTRIES
                || bytes > crate::distributed::remote_lane_effect::MAX_REMOTE_LANE_OUTBOX_BYTES
            {
                remote_instantiated =
                    Err(crate::distributed::remote_lane_effect::RemoteLaneError::JournalFull);
            }
        }
        let result = {
            let mut lane = crate::executives::lane::LaneView::new(self, frame);
            if remote_program.is_some() {
                match &remote_instantiated {
                    Ok(Some(_)) => StepResult::yield_next(run_class),
                    Ok(None) if remote_completed => StepResult::complete(),
                    Ok(None) if remote_failed => StepResult::fault(process, run_class),
                    _ => StepResult::fault(process, run_class),
                }
            } else if let Some(program) = frame_evaluator {
                let input = lane
                    .object_bytes(process, frame)
                    .map(|bytes| bytes.to_vec());
                match input {
                    Ok(input) if input.len() == program.stride() as usize => {
                        let mut output = input.clone();
                        program.evaluate_at(&input, 1, 0, &mut output);
                        let frame_written = match lane.host_payload_mut(process, frame) {
                            Ok(payload) => {
                                *payload = output.clone();
                                true
                            }
                            Err(_) => false,
                        };
                        let effect_result = match frame_effect {
                            Some(crate::kernel::FrameEvaluatorEffect::ResolveFuture {
                                future_offset,
                                value_offset,
                            }) => {
                                let read_ref = |offset: u32| {
                                    output
                                        .get(offset as usize..offset as usize + 8)
                                        .and_then(|bytes| bytes.try_into().ok())
                                        .map(u64::from_le_bytes)
                                        .map(Ref64::from_u64)
                                };
                                match (read_ref(future_offset), read_ref(value_offset)) {
                                    (Some(future), Some(value)) => {
                                        lane.resolve_future(process, future, value).map(|_| ())
                                    }
                                    _ => Err(crate::kernel::RuntimeError::InvalidModule),
                                }
                            }
                            None => Ok(()),
                        };
                        if frame_written && effect_result.is_ok() {
                            StepResult::complete()
                        } else {
                            StepResult::fault(process, run_class)
                        }
                    }
                    _ => StepResult::fault(process, run_class),
                }
            } else {
                cpu_scalar::dispatch(&mut lane, cont, process, run_class)
            }
        };
        if let Ok(Some(batch)) = remote_instantiated {
            self.remote_lane_emissions.push(
                crate::distributed::remote_lane_effect::KernelRemoteLaneEmission {
                    continuation: cont,
                    process,
                    run_class,
                    batch,
                },
            );
        }
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

    fn evaluate_lanes_on_device(
        &mut self,
        lanes: &[(u32, Ref64)],
        backend: &mut dyn DeviceEpochBackend,
    ) -> Option<Vec<SpeculativeLane>> {
        let mut inputs = Vec::with_capacity(lanes.len());
        let mut frames = Vec::new();
        let mut metadata = Vec::with_capacity(lanes.len());
        for &(lane, continuation) in lanes {
            let descriptor = self.continuations.get(continuation).ok()?;
            if descriptor.status != ContinuationState::Runnable
                || descriptor.remaining_steps == 0
                || !descriptor.dependency.is_null()
            {
                return None;
            }
            let process = descriptor.process;
            let frame = descriptor.frame;
            let process_descriptor = self.processes.get(process).ok()?;
            if process_descriptor.status == ProcessState::CancelPending as u32 {
                return None;
            }
            let bytes = self.object_payloads.get(&frame.key())?.as_slice();
            let frame_offset: u32 = frames.len().try_into().ok()?;
            let frame_len: u32 = bytes.len().try_into().ok()?;
            frames.extend_from_slice(bytes);
            inputs.push(DeviceEvaluatorLane {
                continuation: continuation.to_u64(),
                process: process.to_u64(),
                frame: frame.to_u64(),
                lane,
                run_class: descriptor.run_class,
                frame_offset,
                frame_len,
            });
            metadata.push((continuation, process, frame, process_descriptor.supervisor));
        }

        let evaluation = backend.evaluate_lanes(&inputs, &frames).ok()?;
        if evaluation.results.len() != inputs.len() {
            return None;
        }
        let mut outcomes = Vec::with_capacity(inputs.len());
        for (index, result) in evaluation.results.iter().copied().enumerate() {
            let input = inputs[index];
            if result.status != 1 || result.lane != input.lane {
                return None;
            }
            let start = result.frame_offset as usize;
            let end = start.checked_add(result.frame_len as usize)?;
            let output_frame = evaluation.frames.get(start..end)?.to_vec();
            if result.frame_len != input.frame_len {
                return None;
            }
            let kind = match result.step_kind {
                1 => StepKind::Complete,
                2 => StepKind::Yield,
                3 => StepKind::Await,
                4 => StepKind::Send,
                5 => StepKind::Spawn,
                6 => StepKind::Fault,
                _ => return None,
            };
            let (continuation, process, frame, supervisor) = metadata[index];
            let mut journal = LaneJournal::default();
            journal.write(Resource::Process(process));
            if !supervisor.is_null() {
                journal.write(Resource::Process(supervisor));
            }
            journal.read(Resource::Object(frame));
            journal.write(Resource::Object(frame));
            journal.mutated_objects.insert(frame);
            journal.push(LaneOperation::ReadObject {
                actor: process,
                object: frame,
            });
            journal.push(LaneOperation::WriteObject {
                actor: process,
                object: frame,
                growable: true,
            });
            if let Some(crate::kernel::FrameEvaluatorEffect::ResolveFuture {
                future_offset,
                value_offset,
            }) = self.frame_evaluator_effects.get(&input.run_class).copied()
            {
                let read_ref = |offset: u32| {
                    output_frame
                        .get(offset as usize..offset as usize + 8)
                        .and_then(|bytes| bytes.try_into().ok())
                        .map(u64::from_le_bytes)
                        .map(Ref64::from_u64)
                };
                let future = read_ref(future_offset)?;
                let value = read_ref(value_offset)?;
                journal.write(Resource::Future(future));
                journal.read(Resource::Object(value));
                journal.push(LaneOperation::ResolveFuture {
                    actor: process,
                    future,
                    value,
                    result: Ok(()),
                });
            }
            let device_operations = journal.device_operations(input.lane)?;
            outcomes.push(SpeculativeLane {
                lane: input.lane,
                continuation,
                process,
                evaluated: EvaluatedStep {
                    result: StepResult {
                        kind,
                        next_run_class: result.next_run_class,
                        target: Ref64::from_u64(result.target),
                        value: Ref64::from_u64(result.value),
                        consumed_steps: result.consumed_steps,
                        flags: result.flags,
                    },
                    executed: true,
                },
                journal,
                device_operations,
                payloads: vec![(frame, output_frame)],
            });
        }
        self.speculation_stats.device_evaluated_epochs += 1;
        self.speculation_stats.device_evaluated_lanes += outcomes.len() as u64;
        Some(outcomes)
    }

    fn execute_lanes_speculatively(
        &mut self,
        lanes: &[(u32, Ref64)],
        max_lanes: usize,
        validator: Option<&mut dyn LaneConflictValidator>,
        mut backend: Option<&mut dyn DeviceEpochBackend>,
    ) -> Option<usize> {
        if lanes.len() < 2 || lanes.len() > max_lanes.max(1) {
            return None;
        }

        self.speculation_stats.attempted_epochs += 1;
        self.speculation_stats.speculative_lanes += lanes.len() as u64;

        // A physical backend receives only pointer-free lane descriptors and
        // frame bytes. If it declines the batch, evaluation continues through
        // isolated CPU snapshots below without any semantic state changed.
        let device_outcomes = backend
            .as_deref_mut()
            .and_then(|backend| self.evaluate_lanes_on_device(lanes, backend));

        // Each worker receives the exact same pre-Phase-F state. Cloning is
        // intentionally outside the timed handler threads: it is isolation,
        // not work whose completion order may affect a lane.
        let outcomes = device_outcomes.or_else(|| {
            let snapshots: Vec<_> = lanes.iter().map(|_| self.clone()).collect();
            std::thread::scope(|scope| {
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
                            let device_operations = journal.device_operations(lane)?;
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
                                device_operations,
                                payloads,
                            })
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().ok().flatten())
                    .collect::<Option<Vec<_>>>()
            })
        });

        let Some(mut outcomes) = outcomes else {
            self.speculation_stats.fallback_epochs += 1;
            self.speculation_stats.unsupported_fallbacks += 1;
            return None;
        };

        for operation in outcomes
            .iter()
            .flat_map(|outcome| &outcome.device_operations.operations)
        {
            if (1..=32).contains(&operation.opcode) {
                self.speculation_stats.device_operation_kinds |= 1 << (operation.opcode - 1);
            }
        }

        if outcomes.iter().any(|outcome| outcome.journal.unsupported) {
            self.speculation_stats.fallback_epochs += 1;
            self.speculation_stats.unsupported_fallbacks += 1;
            return None;
        }
        // Execution order is deliberately discarded here. Lane number is the
        // canonical position assigned by the epoch plan.
        outcomes.sort_by_key(|outcome| outcome.lane);
        if backend.is_some() || validator.is_some() {
            let accesses: Vec<_> = outcomes
                .iter()
                .enumerate()
                .flat_map(|(lane, outcome)| outcome.journal.device_accesses(lane as u32))
                .collect();
            let operations: Vec<_> = outcomes
                .iter()
                .map(|outcome| &outcome.device_operations)
                .collect();
            let validated = match backend {
                Some(backend) => {
                    backend.validate_epoch(&accesses, outcomes.len() as u32, &operations)
                }
                None => validator.expect("checked above").validate_epoch(
                    &accesses,
                    outcomes.len() as u32,
                    &operations,
                ),
            };
            let conflicts = match validated {
                Ok(conflicts) if conflicts.len() == outcomes.len() => conflicts,
                _ => {
                    self.speculation_stats.fallback_epochs += 1;
                    self.speculation_stats.unsupported_fallbacks += 1;
                    return None;
                }
            };
            debug_assert_eq!(
                conflicts,
                reference_lane_conflicts(&accesses, outcomes.len() as u32)
            );
            if conflicts.iter().any(|conflict| conflict.conflicts != 0) {
                self.speculation_stats.fallback_epochs += 1;
                self.speculation_stats.conflict_fallbacks += 1;
                return None;
            }
        } else {
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
        }

        /*
         * Device validation and host validation both end at this exact gate.
         * Nothing from a snapshot is replayed before it passes.
         */
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
            if !self.replay_device_operations(&outcome.device_operations) {
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

    fn replay_device_operations(&mut self, journal: &DeviceOperationJournal) -> bool {
        let mut lane = None;
        for (ordinal, operation) in journal.operations.iter().copied().enumerate() {
            if operation.ordinal != ordinal as u32
                || lane.is_some_and(|expected| expected != operation.lane)
            {
                return false;
            }
            lane = Some(operation.lane);
            let actor = Ref64::from_u64(operation.actor);
            let target = Ref64::from_u64(operation.target);
            let value = Ref64::from_u64(operation.value);
            let payload = match journal.payload(operation) {
                Some(payload) => payload,
                None => return false,
            };
            let matches = match operation.opcode {
                OP_OBSERVE_FUTURE => {
                    let actual = observe_result(&self.observe_future(actor, target));
                    actual == (operation.result_code, operation.result_ref) && payload.is_empty()
                }
                OP_READ_OBJECT => {
                    let _ = self.object_bytes(actor, target);
                    payload.is_empty()
                }
                OP_CREATE_PROCESS => process_mode(operation.flags).is_some_and(|mode| {
                    ref_result(&self.try_create_process(actor, mode))
                        == (operation.result_code, operation.result_ref)
                        && payload.is_empty()
                }),
                OP_CREATE_CONTINUATION => decode_spec(payload).is_some_and(|spec| {
                    ref_result(&self.create_continuation(actor, target, spec))
                        == (operation.result_code, operation.result_ref)
                }),
                OP_CREATE_FUTURE => {
                    operation.result_code == RESULT_OK
                        && self.create_future(actor).to_u64() == operation.result_ref
                        && payload.is_empty()
                }
                OP_CREATE_OBJECT => object_kind(operation.flags).is_some_and(|kind| {
                    operation.result_code == RESULT_OK
                        && self.create_object(actor, kind, payload.to_vec()).to_u64()
                            == operation.result_ref
                }),
                OP_WRITE_OBJECT => {
                    if operation.flags == 0 {
                        let _ = self.object_bytes_mut(actor, target);
                    } else if operation.flags == 1 {
                        let _ = self.host_payload_mut(actor, target);
                    } else {
                        return false;
                    }
                    payload.is_empty()
                }
                OP_ENQUEUE_MESSAGE => {
                    unit_result(&self.enqueue_message(
                        actor,
                        target,
                        value,
                        Ref64::from_u64(operation.auxiliary),
                    )) == operation.result_code
                        && payload.is_empty()
                }
                OP_RECEIVE_MESSAGE => {
                    let (code, bytes) = message_result(&self.receive_message(actor, target));
                    code == operation.result_code && bytes == payload
                }
                OP_RESOLVE_FUTURE => {
                    unit_result(&self.resolve_future(actor, target, value)) == operation.result_code
                        && payload.is_empty()
                }
                OP_AWAIT_FUTURE => {
                    await_result(&self.await_future(actor, target, value, operation.flags))
                        == (operation.result_code, operation.result_aux)
                        && payload.is_empty()
                }
                _ => false,
            };
            if !matches {
                return false;
            }
        }
        true
    }
}
