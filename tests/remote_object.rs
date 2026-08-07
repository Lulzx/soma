use std::net::TcpListener;
use std::sync::{Arc, Barrier, Mutex};

use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_object::{
    RemoteObjectClient, RemoteObjectError, RemoteObjectServer, RemoteObjectService,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::Kernel;

struct Fixture {
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    service: Arc<Mutex<RemoteObjectService>>,
    reader: RemoteObjectClient,
    writer: RemoteObjectClient,
    writer_grant_nonce: u64,
    listener: TcpListener,
}
fn fixture() -> Fixture {
    let issuer = NodeId(90);
    let owner = NodeId(91);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(4, 1, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x33; 32])));
    let issue = |rights| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor: Ref64::new(2, 1, Kind::Process),
            target,
            rights,
            object_version: 7,
            valid_from_epoch: 0,
            valid_until_epoch: 20,
        })
    };
    let read = issue(Rights::READ);
    let write = issue(Rights::WRITE);
    let service = Arc::new(Mutex::new(RemoteObjectService::new(
        owner,
        target,
        7,
        b"abc".to_vec(),
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let ep = listener.local_addr().unwrap();
    Fixture {
        authority,
        service,
        reader: RemoteObjectClient::new(ep, read, 1),
        writer: RemoteObjectClient::new(ep, write, 1),
        writer_grant_nonce: write.nonce,
        listener,
    }
}
fn serve(f: &Fixture, n: usize) -> std::thread::JoinHandle<()> {
    let l = f.listener.try_clone().unwrap();
    let s = f.service.clone();
    std::thread::spawn(move || RemoteObjectServer::serve_n(l, s, n).unwrap())
}

#[test]
fn canonical_bytes_grow_only_at_owner_and_client_creates_no_kernel_shadow() {
    let f = fixture();
    let server = serve(&f, 2);
    let kernel = Kernel::new();
    let before = kernel.object_count();
    assert_eq!(f.writer.write(0, 3, b"def").unwrap().version, 1);
    assert_eq!(kernel.object_count(), before);
    assert_eq!(f.reader.read(0, 6).unwrap().bytes, b"abcdef");
    assert_eq!(kernel.object_count(), before);
    server.join().unwrap();
    assert_eq!(f.service.lock().unwrap().bytes(), b"abcdef");
}

#[test]
fn competing_writers_cannot_both_commit_the_same_version() {
    let f = fixture();
    let server = serve(&f, 2);
    let ep = f.listener.local_addr().unwrap();
    // Both proxies carry the same grant and optimistic base version.
    let grant = f.authority.lock().unwrap().issue(GrantSpec {
        audience: NodeId(91),
        actor: Ref64::new(3, 1, Kind::Process),
        target: RemoteRef {
            node: NodeId(91),
            entity: Ref64::new(4, 1, Kind::Object),
        },
        rights: Rights::WRITE,
        object_version: 7,
        valid_from_epoch: 0,
        valid_until_epoch: 20,
    });
    let barrier = Arc::new(Barrier::new(3));
    let handles: Vec<_> = [b"X".as_slice(), b"Y".as_slice()]
        .into_iter()
        .map(|bytes| {
            let b = barrier.clone();
            let c = RemoteObjectClient::new(ep, grant, 1);
            std::thread::spawn(move || {
                b.wait();
                c.write(0, 0, bytes)
            })
        })
        .collect();
    barrier.wait();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    server.join().unwrap();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(
                r,
                Err(RemoteObjectError::StaleVersion {
                    expected: 0,
                    actual: 1
                })
            ))
            .count(),
        1
    );
    assert_eq!(f.service.lock().unwrap().applied_writes(), 1);
}

#[test]
fn exact_retry_is_apply_once_but_a_revoked_grant_cannot_replay() {
    let f = fixture();
    let server = serve(&f, 3);
    let first = f.writer.write(0, 3, b"!").unwrap();
    assert_eq!(f.writer.write(0, 3, b"!").unwrap(), first);
    assert_eq!(f.service.lock().unwrap().applied_writes(), 1);
    assert!(f.authority.lock().unwrap().revoke(f.writer_grant_nonce));
    assert_eq!(
        f.writer.write(0, 3, b"!"),
        Err(RemoteObjectError::AuthorityDenied)
    );
    server.join().unwrap();
    assert_eq!(f.service.lock().unwrap().bytes(), b"abc!");
}

#[test]
fn stale_version_does_not_overwrite_newer_bytes() {
    let f = fixture();
    let server = serve(&f, 3);
    assert_eq!(f.writer.write(0, 0, b"A").unwrap().version, 1);
    assert_eq!(
        f.writer.write(0, 1, b"B"),
        Err(RemoteObjectError::StaleVersion {
            expected: 0,
            actual: 1
        })
    );
    assert_eq!(f.reader.read(0, 3).unwrap().bytes, b"Abc");
    server.join().unwrap();
}

#[test]
fn transport_distinguishes_unavailable_lost_and_protocol() {
    let unavailable = fixture();
    let ep = unavailable.listener.local_addr().unwrap();
    drop(unavailable.listener);
    assert_eq!(
        unavailable.reader.read(0, 1),
        Err(RemoteObjectError::NodeUnavailable)
    );
    let lost = fixture();
    let l = lost.listener.try_clone().unwrap();
    let peer = std::thread::spawn(move || {
        let _ = l.accept().unwrap();
    });
    assert_eq!(lost.reader.read(0, 1), Err(RemoteObjectError::NodeLost));
    peer.join().unwrap();
    let protocol = fixture();
    let l = protocol.listener.try_clone().unwrap();
    let peer = std::thread::spawn(move || {
        use std::io::{Read, Write};
        let (mut s, _) = l.accept().unwrap();
        let mut n = [0; 8];
        s.read_exact(&mut n).unwrap();
        let mut b = vec![0; u64::from_le_bytes(n) as usize];
        s.read_exact(&mut b).unwrap();
        s.write_all(&3u64.to_le_bytes()).unwrap();
        s.write_all(b"bad").unwrap();
    });
    assert_eq!(
        protocol.reader.read(0, 1),
        Err(RemoteObjectError::ProtocolError)
    );
    peer.join().unwrap();
    let _ = ep;
}
