//! Commit: atomically apply a continuation's `StepResult` (§8, §18 Phase G).
//!
//! A continuation never becomes runnable or terminal except through this
//! function. The handler has already performed its side-effect *allocation*
//! (creating child continuations/processes, delivering messages, resolving
//! futures); this phase finalizes scheduling state, status transitions, and
//! tracing. All of it is deterministic.
//!
//! Since v0.3 §4.4 the scheduling half of that is *produced* rather than
//! performed: a resume becomes an `Effect` the kernel applies when the lane
//! ends. The status transitions that take a continuation out of the schedulable
//! set stay immediate — nothing else in the lane can observe them, because the
//! lane is over — and they are named in §4.4 as the part still to move.

use crate::abi::continuations::ContinuationState;
use crate::abi::{EventKind, ExitReason, ProcessState, Ref64, StepKind, StepResult};
use crate::kernel::effects::Effect;
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
/// The answer comes from `ProcessDescriptor::live_continuations`, maintained
/// incrementally by `Kernel::set_continuation_status`. This used to be a linear
/// scan of the continuation table on every completion, which is O(n) per commit
/// and not survivable under a concurrent scheduler. I3 checks the count against
/// an actual scan, so the fast path cannot drift from the truth unnoticed.
fn retire_process_if_idle(kernel: &mut Kernel, process: Ref64) {
    let still_live = kernel.has_live_continuation(process);
    let mut terminated = false;
    if let Ok(p) = kernel.processes.get_mut(process) {
        p.active_continuation = Ref64::NULL;
        if !still_live
            && p.status != ProcessState::CancelPending as u32
            && p.status != ProcessState::Terminated as u32
        {
            p.status = ProcessState::Terminated as u32;
            terminated = true;
        }
    }
    if terminated {
        kernel.notify_supervisor(process, ExitReason::Completed);
    }
}

/// Apply `result` for `cont` (owned by `process`). Returns steps consumed.
pub fn apply_step_result(
    kernel: &mut Kernel,
    cont: Ref64,
    process: Ref64,
    result: StepResult,
) -> usize {
    match result.kind {
        StepKind::Complete => {
            kernel.set_continuation_status(cont, ContinuationState::Completed);
            retire_process_if_idle(kernel, process);
            kernel.trace(
                EventKind::ContinuationCompleted,
                process,
                cont,
                result.next_run_class,
                0,
            );
        }
        StepKind::Yield => {
            let rc = result.next_run_class;
            kernel.emit(Effect::Resume {
                continuation: cont,
                run_class: rc,
            });
            kernel.trace(EventKind::ContinuationYielded, process, cont, rc, 0);
        }
        StepKind::Await => {
            let rc = result.next_run_class;
            if let Ok(c) = kernel.continuations.get_mut(cont) {
                c.run_class = rc;
            }
            kernel.set_continuation_status(cont, ContinuationState::Waiting);
            // No immediate re-enqueue: the waiter is woken later by
            // `resolve_future` / `enqueue_message`. This is the single
            // `ContinuationWaiting` emission for every await path; `auxiliary`
            // carries the slot of whatever is being awaited.
            kernel.trace_about(
                EventKind::ContinuationWaiting,
                process,
                cont,
                rc,
                result.target,
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
                kernel.emit(Effect::Resume {
                    continuation: cont,
                    run_class: result.next_run_class,
                });
            } else {
                // No continuation remains: the node is done.
                kernel.set_continuation_status(cont, ContinuationState::Completed);
                retire_process_if_idle(kernel, process);
            }
            kernel.trace(kind, process, cont, result.next_run_class, 0);
        }
        StepKind::Fault => {
            kernel.set_continuation_status(cont, ContinuationState::Faulted);
            if let Ok(p) = kernel.processes.get_mut(process) {
                p.status = ProcessState::Failed as u32;
                p.failure_count = p.failure_count.wrapping_add(1);
            }
            kernel.contain_process_failure(process, cont);
            kernel.trace(
                EventKind::ProcessFailed,
                process,
                cont,
                result.next_run_class,
                0,
            );
            kernel.notify_supervisor(process, ExitReason::Failed);
        }
    }
    if result.kind != StepKind::Fault
        && kernel.process_state(process).ok() == Some(ProcessState::CancelPending)
    {
        kernel.finalize_process_cancellation(process);
    }
    result.consumed_steps.max(1) as usize
}
