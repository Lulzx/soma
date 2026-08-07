use std::net::TcpListener;
use std::sync::{mpsc, Arc, Mutex};

use soma::abi::continuations::ContinuationState;
use soma::abi::{EventKind, Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::RemoteFutureError;
use soma::distributed::remote_future::{
    RemoteAwaitOutcome, RemoteFutureClient, RemoteFutureService, RemoteFutureState,
};
use soma::distributed::remote_node_runtime::{RemoteNodeRuntime, RemoteNodeRuntimeError};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn continuation(kernel: &mut Kernel) -> Ref64 {
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .create_continuation(
            p,
            p,
            ContinuationSpec::new(StateAccess::ReadOnly, 17, 0, vec![], 8),
        )
        .unwrap()
}

#[test]
fn two_owner_epoch_threads_park_resolve_wake_without_a_foreign_descriptor() {
    let owner = NodeId(201);
    let client_node = NodeId(202);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(77, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(200),
        [0x51; 32],
    )));
    let actor = Ref64::new(4, 1, Kind::Process);
    let issue = |rights| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 20,
        })
    };
    let await_grant = issue(Rights::AWAIT);
    let resolve_grant = issue(Rights::RESOLVE);
    let service = Arc::new(Mutex::new(RemoteFutureService::new(
        owner, target, 1, authority,
    )));

    let mut owner_kernel = Kernel::new();
    let producer = continuation(&mut owner_kernel);
    let mut owner_runtime = RemoteNodeRuntime::new(owner, owner_kernel);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = owner_runtime
        .register_owned_future(target, service.clone(), listener)
        .unwrap();
    let value = Ref64::new(91, 1, Kind::Object);
    owner_runtime
        .resolve_after_continuation_runs(
            target,
            producer,
            RemoteFutureClient::new(endpoint, resolve_grant, 0),
            value,
        )
        .unwrap();

    let mut client_kernel = Kernel::new();
    let consumer = continuation(&mut client_kernel);
    let mut client_runtime = RemoteNodeRuntime::new(client_node, client_kernel);
    client_runtime
        .register_foreign_future(target, RemoteFutureClient::new(endpoint, await_grant, 0))
        .unwrap();
    assert_eq!(owner_runtime.kernel().future_count(), 0);
    assert_eq!(client_runtime.kernel().future_count(), 0);
    assert_legal(owner_runtime.kernel());
    assert_legal(client_runtime.kernel());

    let (parked_tx, parked_rx) = mpsc::channel();
    let (resolved_tx, resolved_rx) = mpsc::channel();
    let client_thread = std::thread::spawn(move || {
        assert_eq!(
            client_runtime
                .park_on_remote_future(target, consumer, 17)
                .unwrap(),
            RemoteAwaitOutcome::Registered
        );
        assert_eq!(
            client_runtime.kernel().continuation_state(consumer),
            Ok(ContinuationState::Waiting)
        );
        assert_legal(client_runtime.kernel());
        parked_tx.send(()).unwrap();
        resolved_rx.recv().unwrap();
        assert_eq!(
            client_runtime.run_epoch().unwrap(),
            vec![(
                target,
                RemoteFutureState::Resolved {
                    value,
                    resolved_epoch: 1
                }
            )]
        );
        assert_eq!(
            client_runtime.kernel().continuation_state(consumer),
            Ok(ContinuationState::Runnable)
        );
        assert_legal(client_runtime.kernel());
        client_runtime.run_epoch().unwrap();
        assert_legal(client_runtime.kernel());
        assert_ne!(
            client_runtime.kernel().continuation_state(consumer),
            Ok(ContinuationState::Waiting)
        );
        assert_eq!(client_runtime.kernel().future_count(), 0);
        let starts = client_runtime
            .kernel()
            .trace_events()
            .iter()
            .filter(|e| {
                e.event_kind == EventKind::ContinuationStarted && e.continuation == consumer
            })
            .count();
        assert_eq!(
            starts, 1,
            "remote wake follows ordinary apply-once runnable semantics"
        );
        client_runtime
    });
    let owner_thread = std::thread::spawn(move || {
        parked_rx.recv().unwrap();
        owner_runtime.run_epoch().unwrap();
        assert_eq!(owner_runtime.kernel().future_count(), 0);
        assert_legal(owner_runtime.kernel());
        resolved_tx.send(()).unwrap();
        owner_runtime
    });
    let client_runtime = client_thread.join().unwrap();
    let mut owner_runtime = owner_thread.join().unwrap();
    owner_runtime.join_servers().unwrap();
    assert_eq!(
        service.lock().unwrap().state(),
        RemoteFutureState::Resolved {
            value,
            resolved_epoch: 1
        }
    );
    assert_eq!(service.lock().unwrap().applied_resolutions(), 1);
    assert_eq!(client_runtime.kernel().future_value(target.entity), None);
    assert_eq!(owner_runtime.kernel().future_value(target.entity), None);
    // I18 control: once runnable, the same continuation body has exactly the
    // ordinary local single-start behavior and reaches the same terminal state.
    let mut local = Kernel::new();
    let local_cont = continuation(&mut local);
    local.run_epoch();
    let local_starts = local
        .trace_events()
        .iter()
        .filter(|e| e.event_kind == EventKind::ContinuationStarted && e.continuation == local_cont)
        .count();
    assert_eq!(local_starts, 1);
    assert_eq!(
        client_runtime.kernel().continuation_state(consumer),
        local.continuation_state(local_cont)
    );
}

#[test]
fn accepted_remote_wait_becomes_node_lost_without_fabricating_progress() {
    let owner = NodeId(211);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(8, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(210), [7; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor: Ref64::new(1, 1, Kind::Process),
        target,
        rights: Rights::AWAIT,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 9,
    });
    let service = Arc::new(Mutex::new(RemoteFutureService::new(
        owner, target, 1, authority,
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    owner_runtime
        .register_owned_future(target, service, listener)
        .unwrap();
    let mut kernel = Kernel::new();
    let cont = continuation(&mut kernel);
    let mut client = RemoteNodeRuntime::new(NodeId(212), kernel);
    client
        .register_foreign_future(target, RemoteFutureClient::new(endpoint, grant, 0))
        .unwrap();
    assert_eq!(
        client.park_on_remote_future(target, cont, 3),
        Ok(RemoteAwaitOutcome::Registered)
    );
    owner_runtime.join_servers().unwrap();
    assert_eq!(
        client.run_epoch(),
        Err(RemoteNodeRuntimeError::Remote(RemoteFutureError::NodeLost))
    );
    assert_eq!(
        client.kernel().continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );
    assert_eq!(client.kernel().future_count(), 0);
}
