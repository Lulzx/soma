use soma::abi::continuations::ContinuationState;
use soma::abi::{EventKind, ExitReason, ProcessMode, ProcessState, Ref64, StateAccess, StepResult};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::SearchFrame;
use soma::kernel::raw;
use soma::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn leaf_spec(budget: u32) -> ContinuationSpec {
    let mut frame = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut frame);
    ContinuationSpec::new(StateAccess::ReadOnly, SEARCH_BRANCH, 0, frame, budget)
}

fn leaf(kernel: &mut Kernel, process: Ref64, budget: u32) -> Ref64 {
    kernel
        .create_continuation(process, process, leaf_spec(budget))
        .unwrap()
}

#[test]
fn child_failure_wakes_supervisor_and_delivers_one_notice() {
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let child = kernel
        .create_supervised_process(supervisor, supervisor, ProcessMode::Serial)
        .unwrap();
    let waiter = leaf(&mut kernel, supervisor, DEFAULT_MAX_STEPS);
    let faulting = leaf(&mut kernel, child, 0);

    assert_eq!(
        kernel.receive_supervision(supervisor, waiter).unwrap(),
        None
    );
    {
        let state = unsafe { raw::state(&mut kernel) };
        state.scheduler.remove(waiter);
        state.continuations.get_mut(waiter).unwrap().status = ContinuationState::Waiting;
    }

    kernel.run_epoch();

    assert_eq!(kernel.process_state(child).unwrap(), ProcessState::Failed);
    assert_eq!(
        kernel.continuation_state(faulting).unwrap(),
        ContinuationState::Faulted
    );
    assert_eq!(
        kernel.continuation_state(waiter).unwrap(),
        ContinuationState::Runnable
    );
    assert_eq!(kernel.pending_supervision_notices(supervisor), 1);
    let notice = kernel
        .receive_supervision(supervisor, Ref64::NULL)
        .unwrap()
        .unwrap();
    assert_eq!(notice.child, child);
    assert!(notice.replacement.is_null());
    assert_eq!(notice.reason, ExitReason::Failed);
    assert_eq!(notice.failure_count, 1);
    assert_eq!(kernel.pending_supervision_notices(supervisor), 0);
    assert_eq!(
        kernel.receive_supervision(supervisor, Ref64::NULL).unwrap(),
        None
    );
    assert_legal(&kernel);
}

#[test]
fn restart_replaces_identity_and_exhaustion_escalates() {
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let original = kernel
        .create_restartable_process(supervisor, supervisor, ProcessMode::Serial, 1, leaf_spec(0))
        .unwrap();

    kernel.run_epoch();

    let first = kernel
        .receive_supervision(supervisor, Ref64::NULL)
        .unwrap()
        .unwrap();
    let replacement = first.replacement;
    assert_eq!(first.child, original);
    assert_eq!(first.reason, ExitReason::Failed);
    assert!(!replacement.is_null());
    assert_ne!(replacement, original);
    assert_eq!(
        kernel.process_restart_lineage(replacement).unwrap(),
        (original, 1, 1)
    );
    assert_eq!(
        kernel.process_state(original).unwrap(),
        ProcessState::Failed
    );
    assert_eq!(
        kernel.process_state(replacement).unwrap(),
        ProcessState::Created
    );
    assert_eq!(
        kernel
            .trace_events()
            .iter()
            .filter(|event| event.event_kind == EventKind::ProcessRestarted)
            .count(),
        1
    );
    assert_legal(&kernel);

    kernel.run_epoch();

    assert_eq!(
        kernel.process_state(replacement).unwrap(),
        ProcessState::Failed
    );
    assert_eq!(
        kernel.process_state(supervisor).unwrap(),
        ProcessState::Failed
    );
    assert_eq!(kernel.pending_supervision_notices(supervisor), 1);
    assert_eq!(
        kernel
            .trace_events()
            .iter()
            .filter(|event| event.event_kind == EventKind::ProcessRestarted)
            .count(),
        1,
        "the exhausted retry budget must not create a second replacement"
    );
    assert_legal(&kernel);
}

#[test]
fn restart_requires_a_nonzero_budget() {
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    assert_eq!(
        kernel.create_restartable_process(
            supervisor,
            supervisor,
            ProcessMode::Serial,
            0,
            leaf_spec(DEFAULT_MAX_STEPS),
        ),
        Err(RuntimeError::InvalidSupervisionPolicy)
    );
    assert_legal(&kernel);
}

#[test]
fn completion_and_cancellation_are_distinct_terminal_notices() {
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let completed = kernel
        .create_supervised_process(supervisor, supervisor, ProcessMode::Serial)
        .unwrap();
    let cancelled = kernel
        .create_supervised_process(supervisor, supervisor, ProcessMode::Serial)
        .unwrap();
    let completed_continuation = leaf(&mut kernel, completed, DEFAULT_MAX_STEPS);
    leaf(&mut kernel, cancelled, DEFAULT_MAX_STEPS);

    {
        let state = unsafe { raw::state(&mut kernel) };
        state.scheduler.remove(completed_continuation);
    }
    soma::kernel::commit::apply_step_result(
        &mut kernel,
        completed_continuation,
        completed,
        StepResult::complete(),
    );
    kernel.cancel_process(supervisor, cancelled).unwrap();

    let first = kernel
        .receive_supervision(supervisor, Ref64::NULL)
        .unwrap()
        .unwrap();
    let second = kernel
        .receive_supervision(supervisor, Ref64::NULL)
        .unwrap()
        .unwrap();
    assert_eq!(
        (first.child, first.reason),
        (completed, ExitReason::Completed)
    );
    assert_eq!(
        (second.child, second.reason),
        (cancelled, ExitReason::Cancelled)
    );
    assert_legal(&kernel);
}

#[test]
fn supervision_relationship_requires_the_supervisor_or_system() {
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    assert_eq!(
        kernel.create_supervised_process(stranger, supervisor, ProcessMode::Serial),
        Err(RuntimeError::AuthorityDenied)
    );
    assert_legal(&kernel);
}
