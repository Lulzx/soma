//! Commit: atomically apply a continuation's `StepResult` (§8, §18 Phase G).
//!
//! A continuation never becomes runnable or terminal except through this
//! function. The handler has already performed its side-effect *allocation*
//! (creating child continuations/processes, delivering messages, resolving
//! futures); this phase finalizes scheduling state, status transitions, and
//! tracing. All of it is deterministic.

use crate::abi::continuations::ContinuationState;
use crate::abi::{EventKind, ProcessState, Ref64, StepKind, StepResult};
use crate::kernel::Kernel;

/// Terminate `process` only if it has no continuation left that could still
/// run.
///
/// A `Complete` result is a statement about one continuation, not about the
/// process that owns it. A process may legitimately have several continuations
/// alive at once — `Expand` spawns its heuristic as a second continuation of
/// the same process — and retiring the process when the first of them finishes
/// strands the others: they stay schedulable, wake on their futures, and run
/// against a terminated process. The semantic checker reports that as an I3
/// violation, which is how this was found.
///
/// This is a linear scan of the continuation table per completion. The
/// reference model is written for clarity over speed; a real implementation
/// would keep a per-process count.
fn retire_process_if_idle(kernel: &mut Kernel, process: Ref64) {
    let still_live = kernel.continuations.iter().any(|(_, c)| {
        c.process == process
            && matches!(
                c.status,
                ContinuationState::New
                    | ContinuationState::Runnable
                    | ContinuationState::Waiting
                    | ContinuationState::Running
            )
    });
    if let Ok(p) = kernel.processes.get_mut(process) {
        p.active_continuation = Ref64::NULL;
        if !still_live {
            p.status = ProcessState::Terminated as u32;
        }
    }
}

/// Apply `result` for `cont` (owned by `process`). Returns steps consumed.
pub fn apply_step_result(kernel: &mut Kernel, cont: Ref64, process: Ref64, result: StepResult) -> usize {
    match result.kind {
        StepKind::Complete => {
            if let Ok(c) = kernel.continuations.get_mut(cont) {
                c.status = ContinuationState::Completed;
                c.last_run_epoch = kernel.epoch;
            }
            retire_process_if_idle(kernel, process);
            kernel.trace(EventKind::ContinuationCompleted, process, cont, result.next_run_class, 0);
        }
        StepKind::Yield => {
            let rc = result.next_run_class;
            if let Ok(c) = kernel.continuations.get_mut(cont) {
                c.run_class = rc;
                c.status = ContinuationState::Runnable;
                c.last_run_epoch = kernel.epoch;
            }
            kernel.scheduler.enqueue(rc, cont);
            kernel.trace(EventKind::ContinuationYielded, process, cont, rc, 0);
        }
        StepKind::Await => {
            let rc = result.next_run_class;
            if let Ok(c) = kernel.continuations.get_mut(cont) {
                c.run_class = rc;
                c.status = ContinuationState::Waiting;
                c.last_run_epoch = kernel.epoch;
            }
            // No immediate re-enqueue: the waiter is woken later by
            // `resolve_future` / `enqueue_message`. This is the single
            // `ContinuationWaiting` emission for every await path; `auxiliary`
            // carries the slot of whatever is being awaited.
            kernel.trace(
                EventKind::ContinuationWaiting,
                process,
                cont,
                rc,
                result.target.slot,
            );
        }
        StepKind::Send | StepKind::Spawn => {
            let kind = if result.kind == StepKind::Send {
                EventKind::MessageSent
            } else {
                EventKind::ContinuationReady
            };
            if result.next_run_class != 0 {
                // Continue in the next run class.
                if let Ok(c) = kernel.continuations.get_mut(cont) {
                    c.status = ContinuationState::Runnable;
                    c.run_class = result.next_run_class;
                    c.last_run_epoch = kernel.epoch;
                }
                kernel.scheduler.enqueue(result.next_run_class, cont);
            } else {
                // No continuation remains: the node is done.
                if let Ok(c) = kernel.continuations.get_mut(cont) {
                    c.status = ContinuationState::Completed;
                    c.last_run_epoch = kernel.epoch;
                }
                retire_process_if_idle(kernel, process);
            }
            kernel.trace(kind, process, cont, result.next_run_class, 0);
        }
        StepKind::Fault => {
            if let Ok(c) = kernel.continuations.get_mut(cont) {
                c.status = ContinuationState::Faulted;
                c.last_run_epoch = kernel.epoch;
            }
            if let Ok(p) = kernel.processes.get_mut(process) {
                p.status = ProcessState::Failed as u32;
                p.failure_count = p.failure_count.wrapping_add(1);
            }
            // Failure reclaims the holder's local authority. Capabilities
            // previously exported into other spaces are independent roots and
            // deliberately survive (§7.2 of the capability design note).
            kernel.capability_spaces.remove(&process.slot);
            kernel.trace(EventKind::ProcessFailed, process, cont, result.next_run_class, 0);
        }
    }
    result.consumed_steps.max(1) as usize
}
