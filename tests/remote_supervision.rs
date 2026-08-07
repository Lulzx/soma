use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::continuations::ContinuationState;
use soma::abi::{
    ExitReason, Kind, ProcessMode, ProcessState, Ref64, Rights, StateAccess, SupervisionPolicy,
};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_supervision::{
    RemoteSupervisionBridge, RemoteSupervisionBridgeError, RemoteSupervisionClient,
    RemoteSupervisionError, RemoteSupervisionServer, RemoteSupervisionService,
    RemoteSupervisionState, RemoteTerminalNotice,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::raw;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

struct Fixture {
    target: RemoteRef,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    service: Arc<Mutex<RemoteSupervisionService>>,
    publisher: RemoteSupervisionClient,
    observer: RemoteSupervisionClient,
    observer_nonce: u64,
    listener: TcpListener,
}

fn fixture() -> Fixture {
    let issuer = NodeId(80);
    let owner = NodeId(81);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(7, 1, Kind::Process),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x51; 32])));
    let actor = Ref64::new(3, 1, Kind::Process);
    let grant = |rights, authority: &Arc<Mutex<RemoteAuthorityStore>>| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights,
            object_version: 4,
            valid_from_epoch: 0,
            valid_until_epoch: 100,
        })
    };
    let publish_grant = grant(Rights::WRITE, &authority);
    let observe_grant = grant(Rights::AWAIT, &authority);
    let observer_nonce = observe_grant.nonce;
    let service = Arc::new(Mutex::new(RemoteSupervisionService::new(
        owner,
        target,
        4,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    Fixture {
        target,
        authority,
        service,
        publisher: RemoteSupervisionClient::new(endpoint, publish_grant, 0),
        observer: RemoteSupervisionClient::new(endpoint, observe_grant, 0),
        observer_nonce,
        listener,
    }
}

fn waiter(kernel: &mut Kernel, supervisor: Ref64) -> Ref64 {
    let continuation = kernel
        .create_continuation(
            supervisor,
            supervisor,
            ContinuationSpec::new(StateAccess::ReadOnly, 19, 0, Vec::new(), 8),
        )
        .unwrap();
    assert_eq!(
        kernel
            .receive_supervision(supervisor, continuation)
            .unwrap(),
        None
    );
    let state = unsafe { raw::state(kernel) };
    state.scheduler.remove(continuation);
    state.continuations.get_mut(continuation).unwrap().status = ContinuationState::Waiting;
    continuation
}

#[test]
fn owner_terminal_publication_is_apply_once_and_wakes_remote_supervisor_at_next_boundary() {
    let Fixture {
        target,
        service,
        publisher,
        observer,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteSupervisionServer::serve_n(listener, server_service, 4).unwrap()
    });

    let mut supervisor_kernel = Kernel::new();
    let supervisor = supervisor_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let waiting = waiter(&mut supervisor_kernel, supervisor);
    let mut bridge =
        RemoteSupervisionBridge::new(supervisor, SupervisionPolicy::Notify, observer).unwrap();
    assert_legal(&supervisor_kernel);
    assert_eq!(
        bridge.sync_epoch_boundary(&mut supervisor_kernel),
        Ok(RemoteSupervisionState::Running)
    );

    let mut terminal = RemoteTerminalNotice::new(target, ExitReason::Failed, 1, 9);
    terminal.restart_of = RemoteRef {
        node: target.node,
        entity: Ref64::new(5, 1, Kind::Process),
    };
    terminal.restart_attempt = 2;
    assert_eq!(
        publisher.publish(terminal),
        Ok(RemoteSupervisionState::Terminal(terminal))
    );
    // A transport-level retry has the same content-derived request id.
    assert_eq!(
        publisher.publish(terminal),
        Ok(RemoteSupervisionState::Terminal(terminal))
    );
    assert_eq!(service.lock().unwrap().applied_publications(), 1);

    // Epoch-zero observation is frozen even though publication raced after it.
    assert_eq!(
        bridge.sync_epoch_boundary(&mut supervisor_kernel),
        Ok(RemoteSupervisionState::Running)
    );
    supervisor_kernel.run_epoch();
    assert_legal(&supervisor_kernel);
    assert_eq!(
        bridge.sync_epoch_boundary(&mut supervisor_kernel),
        Ok(RemoteSupervisionState::Terminal(terminal))
    );
    assert_eq!(
        supervisor_kernel.continuation_state(waiting),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(supervisor_kernel.pending_supervision_notices(supervisor), 0);
    assert_legal(&supervisor_kernel);
    assert!(bridge.has_terminal_receipt());
    let notice = bridge.receive_terminal().unwrap();
    assert_eq!(notice, terminal);
    assert_legal(&supervisor_kernel);
    // Duplicate sync applies neither a second receipt nor a second wake.
    assert_eq!(
        bridge.sync_epoch_boundary(&mut supervisor_kernel),
        Ok(RemoteSupervisionState::Terminal(terminal))
    );
    assert_eq!(supervisor_kernel.pending_supervision_notices(supervisor), 0);
    assert!(!bridge.has_terminal_receipt());
    assert_legal(&supervisor_kernel);
    server.join().unwrap();
}

#[test]
fn notify_and_escalate_match_local_i18_terminal_outcomes_without_shadow_child_state() {
    let Fixture {
        target,
        service,
        publisher,
        observer,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteSupervisionServer::serve_n(listener, server_service, 2).unwrap()
    });

    // Kernel one owns the child outcome; kernel two owns only the supervisor.
    let mut child_owner = Kernel::new();
    let child = child_owner.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    child_owner
        .create_continuation(
            child,
            child,
            ContinuationSpec::new(StateAccess::ReadOnly, 77, 0, Vec::new(), 0),
        )
        .unwrap();
    child_owner.run_epoch();
    assert_eq!(child_owner.process_state(child), Ok(ProcessState::Failed));
    let terminal =
        RemoteTerminalNotice::new(target, ExitReason::Failed, 1, child_owner.current_epoch());
    assert_eq!(
        publisher.publish(terminal),
        Ok(RemoteSupervisionState::Terminal(terminal))
    );

    let mut supervisor_kernel = Kernel::new();
    let supervisor = supervisor_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let process_count = supervisor_kernel.process_count();
    let mut bridge =
        RemoteSupervisionBridge::new(supervisor, SupervisionPolicy::Escalate, observer).unwrap();
    assert_legal(&supervisor_kernel);
    assert_eq!(
        bridge.sync_epoch_boundary(&mut supervisor_kernel),
        Ok(RemoteSupervisionState::Terminal(terminal))
    );
    assert_eq!(
        supervisor_kernel.process_count(),
        process_count,
        "no coordinator-side child descriptor"
    );
    assert_eq!(
        supervisor_kernel.process_state(supervisor),
        Ok(ProcessState::Failed)
    );
    assert_eq!(supervisor_kernel.pending_supervision_notices(supervisor), 0);
    assert_legal(&supervisor_kernel);
    assert_eq!(bridge.receive_terminal(), Some(terminal));
    assert_legal(&supervisor_kernel);

    // I18 observable outcome agrees with an equivalent local Escalate tree,
    // up to the expected correspondence between child identities.
    let mut local = Kernel::new();
    let local_supervisor = local.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let local_child = local
        .create_supervised_process_with_policy(
            SYSTEM_PRINCIPAL,
            local_supervisor,
            ProcessMode::Serial,
            SupervisionPolicy::Escalate,
        )
        .unwrap();
    local
        .create_continuation(
            local_child,
            local_child,
            ContinuationSpec::new(StateAccess::ReadOnly, 77, 0, Vec::new(), 0),
        )
        .unwrap();
    local.run_epoch();
    assert_eq!(local.pending_supervision_notices(local_supervisor), 1);
    assert_legal(&local);
    let remote_event = supervisor_kernel
        .trace_events()
        .iter()
        .find(|e| e.event_kind == soma::abi::EventKind::SupervisionNotified)
        .unwrap();
    let local_event = local
        .trace_events()
        .iter()
        .find(|e| e.event_kind == soma::abi::EventKind::SupervisionNotified)
        .unwrap();
    assert_eq!(remote_event.auxiliary, local_event.auxiliary);
    assert_eq!(
        local.process_state(local_supervisor),
        supervisor_kernel.process_state(supervisor)
    );
    server.join().unwrap();
}

#[test]
fn revocation_and_node_loss_remain_distinct_and_restart_is_explicitly_owner_orchestrated() {
    let Fixture {
        target,
        service,
        observer,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteSupervisionServer::serve_n(listener, server_service, 1).unwrap()
    });
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mut bridge =
        RemoteSupervisionBridge::new(supervisor, SupervisionPolicy::Notify, observer).unwrap();
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Ok(RemoteSupervisionState::Running)
    );
    server.join().unwrap();
    kernel.run_epoch();
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Err(RemoteSupervisionBridgeError::Remote(
            RemoteSupervisionError::NodeLost
        ))
    );

    // Before first successful contact the same transport condition is merely
    // unavailable, not an owner that disappeared after accepting supervision.
    let Fixture {
        observer: never_contacted_client,
        listener: dead,
        ..
    } = fixture();
    drop(dead);
    let mut never = RemoteSupervisionBridge::new(
        supervisor,
        SupervisionPolicy::Notify,
        never_contacted_client,
    )
    .unwrap();
    assert_eq!(
        never.sync_epoch_boundary(&mut kernel),
        Err(RemoteSupervisionBridgeError::Remote(
            RemoteSupervisionError::NodeUnavailable
        ))
    );

    // Authorization is rechecked before replay/observation.
    let Fixture {
        authority,
        service,
        observer,
        observer_nonce,
        listener,
        ..
    } = fixture();
    authority.lock().unwrap().revoke(observer_nonce);
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteSupervisionServer::serve_n(listener, server_service, 1).unwrap()
    });
    assert_eq!(
        observer.poll(),
        Err(RemoteSupervisionError::AuthorityDenied)
    );
    server.join().unwrap();

    // Restart requires an owner-side replacement protocol; the observing
    // kernel refuses to forge a canonical child identity.
    let Fixture {
        observer: restart_observer,
        listener: unused,
        ..
    } = fixture();
    drop(unused);
    assert_eq!(
        RemoteSupervisionBridge::new(supervisor, SupervisionPolicy::Restart, restart_observer)
            .err(),
        Some(RemoteSupervisionError::UnsupportedPolicy)
    );
    let _ = target;
}
