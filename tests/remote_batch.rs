use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{Kind, Ref64, Rights};
use soma::compiler::examples;
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_batch::{RemoteBatchBackend, RemoteBatchServer, RemoteBatchService};
use soma::distributed::{NodeId, RemoteRef};
use soma::executives::batch::{BackendError, BackendKind, BatchBackend, CpuReferenceBackend};

fn input(count: u32) -> Vec<u8> {
    (0..count).flat_map(u32::to_le_bytes).collect()
}

struct Fixture {
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    service: Arc<Mutex<RemoteBatchService>>,
    client: RemoteBatchBackend,
    listener: TcpListener,
}

fn fixture() -> Fixture {
    let module = examples::module();
    let program = module.program(examples::DOUBLE_PLUS_ONE).unwrap();
    let issuer = NodeId(101);
    let worker = NodeId(202);
    let target = RemoteRef {
        node: worker,
        entity: Ref64::new(9, 1, Kind::Module),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x5A; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: worker,
        actor: Ref64::new(1, 1, Kind::Process),
        target,
        rights: Rights::READ,
        object_version: 4,
        valid_from_epoch: 3,
        valid_until_epoch: 30,
    });
    let service = Arc::new(Mutex::new(RemoteBatchService::with(
        worker,
        target,
        4,
        authority.clone(),
        &[program],
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client =
        RemoteBatchBackend::with(listener.local_addr().unwrap(), grant, 3, &[program]).unwrap();
    Fixture {
        authority,
        service,
        client,
        listener,
    }
}

#[test]
fn authenticated_tcp_backend_agrees_with_the_reference() {
    let Fixture {
        service,
        mut client,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteBatchServer::serve_n(listener, server_service, 1).unwrap()
    });
    let bytes = input(1024);
    let module = examples::module();
    let program = module.program(examples::DOUBLE_PLUS_ONE).unwrap();
    let mut reference = CpuReferenceBackend::with(&[program]);
    assert_eq!(client.kind(), BackendKind::Remote);
    assert_eq!(
        client
            .evaluate(examples::DOUBLE_PLUS_ONE, &bytes, 1024, 4)
            .unwrap(),
        reference
            .evaluate(examples::DOUBLE_PLUS_ONE, &bytes, 1024, 4)
            .unwrap()
    );
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().applied_requests(), 1);
}

#[test]
fn duplicate_content_addressed_requests_apply_once() {
    let Fixture {
        service,
        mut client,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteBatchServer::serve_n(listener, server_service, 2).unwrap()
    });
    let bytes = input(64);
    let first = client
        .evaluate(examples::DOUBLE_PLUS_ONE, &bytes, 64, 4)
        .unwrap();
    let second = client
        .evaluate(examples::DOUBLE_PLUS_ONE, &bytes, 64, 4)
        .unwrap();
    assert_eq!(first, second);
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().applied_requests(), 1);
}

#[test]
fn revocation_is_checked_before_the_response_ledger() {
    let Fixture {
        authority,
        service,
        mut client,
        listener,
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteBatchServer::serve_n(listener, server_service, 2).unwrap()
    });
    let bytes = input(64);
    client
        .evaluate(examples::DOUBLE_PLUS_ONE, &bytes, 64, 4)
        .unwrap();
    let nonce = {
        // The client's grant is deliberately opaque; this fixture issued the
        // first grant from a fresh store, whose nonce is one.
        1
    };
    assert!(authority.lock().unwrap().revoke(nonce));
    assert_eq!(
        client.evaluate(examples::DOUBLE_PLUS_ONE, &bytes, 64, 4),
        Err(BackendError::AuthorityDenied)
    );
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().applied_requests(), 1);
}

#[test]
fn an_unreachable_node_is_not_reported_as_bad_evaluator_bytes() {
    let Fixture {
        mut client,
        listener,
        ..
    } = fixture();
    drop(listener);
    assert_eq!(
        client.evaluate(examples::DOUBLE_PLUS_ONE, &input(1), 1, 4),
        Err(BackendError::NodeUnavailable)
    );
}
