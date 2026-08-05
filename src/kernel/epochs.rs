//! The epoch lifecycle (§18). Phase 1 runs on a single CPU scalar executive, so
//! the phases reduce to: Ingest (nothing external yet) → Validate/Admit (light,
//! per-process serial guard) → Execute → Commit → Account, then swap the
//! runnable-bin buffers and advance the epoch.
//!
//! Phase E (Cohort) is a stub here: everything is scalar, `cohort_width = 1`.
//! The run-class grouping already performed by `Scheduler::enqueue` is the seed
//! of cohorting (§9); real SIMD cohort construction is a later slice.

use std::collections::HashSet;

use crate::abi::ProcessMode;
use crate::abi::Ref64;
use crate::executives::cpu_scalar;
use crate::kernel::commit;
use crate::kernel::Kernel;

impl Kernel {
    /// Run one epoch. Returns the number of continuation steps executed.
    ///
    /// The epoch boundary runs *first*: work produced last epoch (in the `next`
    /// buffer) is promoted to `current`, executed, and any work it produces
    /// lands in `next` for the following epoch (§13, §18).
    pub fn run_epoch(&mut self) -> usize {
        // Epoch boundary: promote `next` → `current` (§18 Phase H → next epoch).
        self.scheduler.swap_all();

        // Phase F: Execute (CPU scalar).
        let mut steps = 0;

        // Serial-process invariant (§19): at most one mutating continuation of a
        // serial process RUNNING at a time. Sequential execution can't overlap,
        // but we still encode the check so a later parallel executive must obey
        // it. A continuation of an already-running serial process is deferred.
        let mut running_procs: HashSet<u32> = HashSet::new();

        let classes: Vec<u32> = self.scheduler.runnable_counts().iter().map(|(c, _)| *c).collect();
        for rc in classes {
            let batch = self.scheduler.drain(rc);
            for cont in batch {
                let (process, mode) = match self.continuations.get(cont) {
                    Ok(c) => (c.process, self.process_mode(c.process)),
                    Err(_) => continue,
                };
                let mutating = mode == ProcessMode::Serial || mode == ProcessMode::System;
                if mutating && !running_procs.insert(process.slot) {
                    // Another continuation of this serial process is running
                    // this epoch; defer to the next-epoch buffer.
                    self.scheduler.enqueue(rc, cont);
                    continue;
                }
                steps += self.execute_cont(cont, process);
                if mutating {
                    running_procs.remove(&process.slot);
                }
            }
        }

        // Phase H: Account.
        let total = self.scheduler.total_pending();
        self.epoch_runnable.push(total);

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

    /// Execute a single continuation: dispatch to the interpreter, enforce the
    /// step budget, and commit the result.
    fn execute_cont(&mut self, cont: Ref64, process: Ref64) -> usize {
        let run_class = self
            .continuations
            .get(cont)
            .map(|c| c.run_class)
            .unwrap_or(0);
        self.trace(
            crate::abi::EventKind::ContinuationStarted,
            process,
            cont,
            run_class,
            0,
        );
        let result = cpu_scalar::dispatch(self, cont, process);
        let consumed = commit::apply_step_result(self, cont, process, result);

        // Step budget (§8): a continuation must not exceed its declared maximum.
        // If a hand-written handler ever overruns, fault it.
        let remaining = self
            .continuations
            .get(cont)
            .map(|c| c.remaining_steps)
            .unwrap_or(0);
        if remaining <= consumed as u32 {
            if let Ok(c) = self.continuations.get_mut(cont) {
                if c.status != crate::abi::continuations::ContinuationState::Completed
                    && c.status != crate::abi::continuations::ContinuationState::Faulted
                {
                    let fake = crate::abi::StepResult::fault(process, run_class);
                    let _ = commit::apply_step_result(self, cont, process, fake);
                }
            }
        } else if let Ok(c) = self.continuations.get_mut(cont) {
            c.remaining_steps = c.remaining_steps.saturating_sub(consumed as u32);
        }

        consumed
    }
}
