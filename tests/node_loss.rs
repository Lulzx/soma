use soma::abi::{EventKind, ExitReason, ProcessMode, ProcessState, StateAccess};
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

#[test]
fn declared_node_loss_is_not_a_program_fault() {
    let mut kernel = Kernel::new();
    let supervisor = kernel
        .create_process_on_node(SYSTEM_PRINCIPAL, ProcessMode::Serial, 1)
        .unwrap();
    let remote = kernel
        .create_supervised_process_on_node(supervisor, supervisor, ProcessMode::Serial, 2)
        .unwrap();
    let unaffected = kernel
        .create_supervised_process_on_node(supervisor, supervisor, ProcessMode::Serial, 3)
        .unwrap();
    kernel
        .create_continuation(
            remote,
            remote,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_BRANCH,
                0,
                Vec::new(),
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();

    let lost = kernel.declare_node_lost(SYSTEM_PRINCIPAL, 2).unwrap();
    assert_eq!(lost, vec![remote]);
    assert_eq!(kernel.process_state(remote), Ok(ProcessState::Failed));
    assert_eq!(kernel.process_state(supervisor), Ok(ProcessState::Created));
    assert_eq!(kernel.process_state(unaffected), Ok(ProcessState::Created));
    let notice = kernel
        .receive_supervision(supervisor, soma::abi::Ref64::NULL)
        .unwrap()
        .unwrap();
    assert_eq!(notice.child, remote);
    assert_eq!(notice.reason, ExitReason::NodeLost);
    assert!(kernel
        .trace_events()
        .iter()
        .any(|event| event.event_kind == EventKind::ProcessLost && event.process == remote));
    assert!(!kernel
        .trace_events()
        .iter()
        .any(|event| event.event_kind == EventKind::ProcessFailed && event.process == remote));
    assert_legal(&kernel);
}

#[test]
fn a_partition_does_nothing_until_loss_is_explicitly_declared() {
    let mut kernel = Kernel::new();
    let remote = kernel
        .create_process_on_node(SYSTEM_PRINCIPAL, ProcessMode::Serial, 9)
        .unwrap();
    assert_eq!(kernel.process_node(remote), Ok(9));
    assert_eq!(kernel.process_state(remote), Ok(ProcessState::Created));
    assert!(kernel
        .trace_events()
        .iter()
        .all(|event| event.event_kind != EventKind::ProcessLost));
}

#[test]
fn loss_declaration_is_system_only_idempotent_and_prevents_new_placement() {
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    assert_eq!(
        kernel.declare_node_lost(actor, 7),
        Err(RuntimeError::AuthorityDenied)
    );
    assert!(kernel
        .declare_node_lost(SYSTEM_PRINCIPAL, 7)
        .unwrap()
        .is_empty());
    assert!(kernel
        .declare_node_lost(SYSTEM_PRINCIPAL, 7)
        .unwrap()
        .is_empty());
    assert_eq!(
        kernel.create_process_on_node(SYSTEM_PRINCIPAL, ProcessMode::Serial, 7),
        Err(RuntimeError::NodeUnavailable)
    );
}
