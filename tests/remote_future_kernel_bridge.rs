use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::continuations::ContinuationState;
use soma::abi::{EventKind, Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::{
    RemoteAwaitOutcome, RemoteFutureBridge, RemoteFutureBridgeError, RemoteFutureClient,
    RemoteFutureError, RemoteFutureServer, RemoteFutureService, RemoteFutureState,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn continuation(kernel: &mut Kernel) -> Ref64 {
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(StateAccess::ReadOnly, 17, 0, Vec::new(), 8),
        )
        .unwrap()
}

struct Fixture {
    target: RemoteRef,
    service: Arc<Mutex<RemoteFutureService>>,
    observer: RemoteFutureClient,
    resolver: RemoteFutureClient,
    listener: TcpListener,
}

fn fixture() -> Fixture {
    let issuer = NodeId(41);
    let owner = NodeId(42);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(77, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x91; 32])));
    let actor = Ref64::new(2, 1, Kind::Process);
    let issue = |rights| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights,
            object_version: 3,
            valid_from_epoch: 0,
            valid_until_epoch: 100,
        })
    };
    let observer_grant = issue(Rights::AWAIT);
    let resolver_grant = issue(Rights::RESOLVE);
    let service = Arc::new(Mutex::new(RemoteFutureService::new(
        owner, target, 3, authority,
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    Fixture {
        target,
        service,
        observer: RemoteFutureClient::new(endpoint, observer_grant, 0),
        resolver: RemoteFutureClient::new(endpoint, resolver_grant, 0),
        listener,
    }
}

#[test]
fn remote_resolution_wakes_a_local_continuation_without_a_local_future_copy() {
    let Fixture {
        target,
        service,
        observer,
        mut resolver,
        listener,
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteFutureServer::serve_n(listener, server_service, 3).unwrap()
    });

    let mut kernel = Kernel::new();
    let cont = continuation(&mut kernel);
    let mut bridge = RemoteFutureBridge::new(target, observer);
    assert_eq!(
        bridge.await_at_epoch_boundary(&mut kernel, cont, 29),
        Ok(RemoteAwaitOutcome::Registered)
    );
    assert_eq!(
        kernel.continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );

    let value = Ref64::new(9, 1, Kind::Object);
    resolver.set_epoch(kernel.current_epoch());
    assert!(resolver.resolve(value).is_ok());

    // The observation for epoch zero is frozen. Resolution races within the
    // epoch cannot alter its scheduling decision.
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Ok(RemoteFutureState::Pending)
    );
    assert_eq!(
        kernel.continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );

    kernel.run_epoch();
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Ok(RemoteFutureState::Resolved {
            value,
            resolved_epoch: 0,
        })
    );
    assert_eq!(
        kernel.continuation_state(cont),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(bridge.waiter_count(), 0);
    assert_eq!(kernel.future_value(target.entity), None);
    assert_eq!(service.lock().unwrap().applied_resolutions(), 1);
    server.join().unwrap();
}

#[test]
fn registration_and_wakeup_are_idempotent_at_an_epoch_boundary() {
    let Fixture {
        target,
        service,
        observer,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        // One poll per boundary, despite duplicate registration and sync.
        RemoteFutureServer::serve_n(listener, server_service, 2).unwrap()
    });
    let mut kernel = Kernel::new();
    let cont = continuation(&mut kernel);
    let mut bridge = RemoteFutureBridge::new(target, observer);
    assert_eq!(
        bridge.await_at_epoch_boundary(&mut kernel, cont, 4),
        Ok(RemoteAwaitOutcome::Registered)
    );
    assert_eq!(
        bridge.await_at_epoch_boundary(&mut kernel, cont, 99),
        Ok(RemoteAwaitOutcome::Registered)
    );
    assert_eq!(bridge.waiter_count(), 1);
    let waiting_events: Vec<_> = kernel
        .trace_events()
        .iter()
        .filter(|event| {
            event.event_kind == EventKind::ContinuationWaiting && event.continuation == cont
        })
        .collect();
    assert_eq!(waiting_events.len(), 1, "registration traces exactly once");
    assert_eq!(waiting_events[0].subject, target.entity);

    // Boundary parking removes the continuation's pre-existing runnable-bin
    // entry immediately. The checker must accept the state before an epoch
    // drain, and the stale entry must never execute.
    assert_legal(&kernel);
    let starts_before = kernel
        .trace_events()
        .iter()
        .filter(|event| event.event_kind == EventKind::ContinuationStarted)
        .count();
    kernel.run_epoch();
    assert_eq!(
        kernel.continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );
    assert_eq!(
        kernel
            .trace_events()
            .iter()
            .filter(|event| event.event_kind == EventKind::ContinuationStarted)
            .count(),
        starts_before
    );
    assert_legal(&kernel);

    // This is now epoch one, so the first sync is a new authoritative poll.
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Ok(RemoteFutureState::Pending)
    );
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Ok(RemoteFutureState::Pending)
    );
    assert_eq!(bridge.waiter_count(), 1);
    server.join().unwrap();
}

#[test]
fn a_failure_before_contact_is_unavailable_but_loss_after_registration_is_node_lost() {
    let Fixture {
        target,
        service,
        observer,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        // Accept the registration poll and then drop the listening socket.
        RemoteFutureServer::serve_n(listener, server_service, 1).unwrap()
    });
    let mut kernel = Kernel::new();
    let cont = continuation(&mut kernel);
    let mut bridge = RemoteFutureBridge::new(target, observer);
    assert_eq!(
        bridge.await_at_epoch_boundary(&mut kernel, cont, 5),
        Ok(RemoteAwaitOutcome::Registered)
    );
    server.join().unwrap();
    kernel.run_epoch();
    assert_eq!(
        bridge.sync_epoch_boundary(&mut kernel),
        Err(RemoteFutureBridgeError::Remote(RemoteFutureError::NodeLost))
    );
    assert_eq!(
        kernel.continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );

    let Fixture {
        target: other_target,
        observer: other_observer,
        listener: dead_listener,
        ..
    } = fixture();
    drop(dead_listener);
    let mut never_contacted = RemoteFutureBridge::new(other_target, other_observer);
    let other = continuation(&mut kernel);
    assert_eq!(
        never_contacted.await_at_epoch_boundary(&mut kernel, other, 5),
        Err(RemoteFutureBridgeError::Remote(
            RemoteFutureError::NodeUnavailable
        ))
    );
}

#[test]
fn a_late_remote_await_observes_settlement_and_never_parks() {
    let Fixture {
        target,
        service,
        observer,
        resolver,
        listener,
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteFutureServer::serve_n(listener, server_service, 2).unwrap()
    });
    let value = Ref64::new(31, 1, Kind::Object);
    assert_eq!(
        resolver.resolve(value),
        Ok(RemoteFutureState::Resolved {
            value,
            resolved_epoch: 0,
        })
    );

    let mut kernel = Kernel::new();
    let cont = continuation(&mut kernel);
    let mut bridge = RemoteFutureBridge::new(target, observer);
    assert_eq!(
        bridge.await_at_epoch_boundary(&mut kernel, cont, 8),
        Ok(RemoteAwaitOutcome::AlreadySettled {
            value,
            resolved_epoch: 0,
        })
    );
    assert_eq!(
        kernel.continuation_state(cont),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(bridge.waiter_count(), 0);
    server.join().unwrap();
}
