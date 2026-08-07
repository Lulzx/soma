use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::{
    RemoteFutureClient, RemoteFutureError, RemoteFutureServer, RemoteFutureService,
    RemoteFutureState,
};
use soma::distributed::{NodeId, RemoteRef};

struct Fixture {
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    service: Arc<Mutex<RemoteFutureService>>,
    resolver: RemoteFutureClient,
    observer: RemoteFutureClient,
    listener: TcpListener,
}

fn fixture() -> Fixture {
    let issuer = NodeId(11);
    let worker = NodeId(22);
    let target = RemoteRef {
        node: worker,
        entity: Ref64::new(7, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x35; 32])));
    let actor = Ref64::new(3, 1, Kind::Process);
    let resolver_grant = authority.lock().unwrap().issue(GrantSpec {
        audience: worker,
        actor,
        target,
        rights: Rights::RESOLVE,
        object_version: 2,
        valid_from_epoch: 4,
        valid_until_epoch: 40,
    });
    let observer_grant = authority.lock().unwrap().issue(GrantSpec {
        audience: worker,
        actor,
        target,
        rights: Rights::AWAIT,
        object_version: 2,
        valid_from_epoch: 4,
        valid_until_epoch: 40,
    });
    let service = Arc::new(Mutex::new(RemoteFutureService::new(
        worker,
        target,
        2,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    Fixture {
        authority,
        service,
        resolver: RemoteFutureClient::new(endpoint, resolver_grant, 5),
        observer: RemoteFutureClient::new(endpoint, observer_grant, 5),
        listener,
    }
}

#[test]
fn remote_node_owns_and_publishes_canonical_future_state() {
    let Fixture {
        service,
        resolver,
        observer,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteFutureServer::serve_n(listener, server_service, 3).unwrap()
    });
    assert_eq!(observer.poll(), Ok(RemoteFutureState::Pending));
    let value = Ref64::new(91, 2, Kind::Object);
    assert_eq!(
        resolver.resolve(value),
        Ok(RemoteFutureState::Resolved {
            value,
            resolved_epoch: 5
        })
    );
    assert_eq!(
        observer.poll(),
        Ok(RemoteFutureState::Resolved {
            value,
            resolved_epoch: 5
        })
    );
    server.join().unwrap();
    assert_eq!(
        service.lock().unwrap().state(),
        RemoteFutureState::Resolved {
            value,
            resolved_epoch: 5
        }
    );
}

#[test]
fn exact_retry_is_idempotent_but_a_distinct_writer_loses_single_assignment() {
    let Fixture {
        service,
        resolver,
        listener,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteFutureServer::serve_n(listener, server_service, 3).unwrap()
    });
    let first = Ref64::new(1, 1, Kind::Object);
    assert!(resolver.resolve(first).is_ok());
    assert!(
        resolver.resolve(first).is_ok(),
        "same content-addressed request must replay success"
    );
    assert_eq!(
        resolver.resolve(Ref64::new(2, 1, Kind::Object)),
        Err(RemoteFutureError::AlreadyResolved)
    );
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().applied_resolutions(), 1);
}

#[test]
fn operation_specific_rights_are_enforced_on_the_owner() {
    let Fixture {
        resolver,
        observer,
        listener,
        service,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteFutureServer::serve_n(listener, server_service, 2).unwrap()
    });
    assert_eq!(resolver.poll(), Err(RemoteFutureError::AuthorityDenied));
    assert_eq!(
        observer.resolve(Ref64::new(1, 1, Kind::Object)),
        Err(RemoteFutureError::AuthorityDenied)
    );
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().state(), RemoteFutureState::Pending);
}

#[test]
fn revocation_is_rechecked_before_idempotent_replay() {
    let Fixture {
        authority,
        resolver,
        listener,
        service,
        ..
    } = fixture();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteFutureServer::serve_n(listener, server_service, 2).unwrap()
    });
    let value = Ref64::new(4, 1, Kind::Object);
    assert!(resolver.resolve(value).is_ok());
    assert!(authority.lock().unwrap().revoke(1));
    assert_eq!(
        resolver.resolve(value),
        Err(RemoteFutureError::AuthorityDenied)
    );
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().applied_resolutions(), 1);
}
