use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_channel::{
    RemoteChannelClient, RemoteChannelEntry, RemoteChannelError, RemoteChannelServer,
    RemoteChannelService, RemoteCloseOutcome, RemoteReceiveOutcome, RemoteSendOutcome,
};
use soma::distributed::{NodeId, RemoteRef};

struct Fixture {
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    service: Arc<Mutex<RemoteChannelService>>,
    sender: RemoteChannelClient,
    receiver: RemoteChannelClient,
    closer: RemoteChannelClient,
    listener: TcpListener,
}
fn fixture(capacity: usize) -> Fixture {
    let issuer = NodeId(51);
    let owner = NodeId(52);
    let actor = Ref64::new(1, 1, Kind::Process);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(9, 1, Kind::Channel),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x71; 32])));
    let issue = |rights| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights,
            object_version: 3,
            valid_from_epoch: 1,
            valid_until_epoch: 30,
        })
    };
    let send = issue(Rights::SEND);
    let receive = issue(Rights::RECEIVE);
    let close = issue(Rights::DESTROY);
    let service = Arc::new(Mutex::new(RemoteChannelService::new(
        owner,
        target,
        3,
        capacity,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let ep = listener.local_addr().unwrap();
    Fixture {
        authority,
        service,
        sender: RemoteChannelClient::new(ep, send, 4),
        receiver: RemoteChannelClient::new(ep, receive, 4),
        closer: RemoteChannelClient::new(ep, close, 4),
        listener,
    }
}
fn serve(f: &Fixture, n: usize) -> std::thread::JoinHandle<()> {
    let listener = f.listener.try_clone().unwrap();
    let service = f.service.clone();
    std::thread::spawn(move || RemoteChannelServer::serve_n(listener, service, n).unwrap())
}
#[test]
fn owner_keeps_bounded_fifo_and_sequence() {
    let f = fixture(2);
    let server = serve(&f, 6);
    let a = Ref64::new(20, 1, Kind::Object);
    let b = Ref64::new(21, 1, Kind::Object);
    assert!(matches!(
        f.sender.send(0, a),
        Ok(RemoteSendOutcome::Sent { .. })
    ));
    assert!(matches!(
        f.sender.send(1, b),
        Ok(RemoteSendOutcome::Sent { .. })
    ));
    assert_eq!(f.sender.send(2, a), Ok(RemoteSendOutcome::Full));
    assert_eq!(
        f.receiver.receive(0),
        Ok(RemoteReceiveOutcome::Received(RemoteChannelEntry {
            value: a,
            sender_sequence: 0
        }))
    );
    assert!(matches!(
        f.sender.send(2, a),
        Ok(RemoteSendOutcome::Sent { .. })
    ));
    assert_eq!(
        f.receiver.receive(1),
        Ok(RemoteReceiveOutcome::Received(RemoteChannelEntry {
            value: b,
            sender_sequence: 1
        }))
    );
    server.join().unwrap();
    assert_eq!(f.service.lock().unwrap().entries()[0].sender_sequence, 2);
}
#[test]
fn successful_mutations_apply_once_and_revocation_precedes_replay() {
    let f = fixture(2);
    let server = serve(&f, 4);
    let v = Ref64::new(22, 1, Kind::Object);
    assert!(f.sender.send(0, v).is_ok());
    assert!(f.sender.send(0, v).is_ok());
    assert_eq!(f.service.lock().unwrap().applied_sends(), 1);
    assert!(f.authority.lock().unwrap().revoke(1));
    assert_eq!(
        f.sender.send(0, v),
        Err(RemoteChannelError::AuthorityDenied)
    );
    assert_eq!(
        f.receiver.receive(0),
        Ok(RemoteReceiveOutcome::Received(RemoteChannelEntry {
            value: v,
            sender_sequence: 0
        }))
    );
    server.join().unwrap();
    assert_eq!(f.service.lock().unwrap().applied_receives(), 1);
}
#[test]
fn close_drains_then_reports_closed_and_rights_are_specific() {
    let f = fixture(2);
    let server = serve(&f, 7);
    let v = Ref64::new(23, 1, Kind::Object);
    assert_eq!(f.receiver.close(), Err(RemoteChannelError::AuthorityDenied));
    assert_eq!(f.closer.close(), Ok(RemoteCloseOutcome::Closed));
    assert_eq!(f.closer.close(), Ok(RemoteCloseOutcome::Closed));
    assert_eq!(f.sender.send(0, v), Ok(RemoteSendOutcome::Closed));
    assert_eq!(f.receiver.receive(0), Ok(RemoteReceiveOutcome::Closed));
    assert_eq!(
        f.sender.receive(0),
        Err(RemoteChannelError::AuthorityDenied)
    );
    assert_eq!(
        f.receiver.send(0, v),
        Err(RemoteChannelError::AuthorityDenied)
    );
    server.join().unwrap();
    assert_eq!(f.service.lock().unwrap().applied_closes(), 1);
}

#[test]
fn transport_failures_distinguish_unavailable_lost_and_protocol() {
    // No listener ever accepted this endpoint.
    let unavailable = fixture(1);
    let endpoint = unavailable.listener.local_addr().unwrap();
    drop(unavailable.listener);
    assert_eq!(
        unavailable.sender.send(0, Ref64::NULL),
        Err(RemoteChannelError::NodeUnavailable)
    );

    let lost = fixture(1);
    let listener = lost.listener.try_clone().unwrap();
    let peer = std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
    });
    assert_eq!(
        lost.sender.send(0, Ref64::NULL),
        Err(RemoteChannelError::NodeLost)
    );
    peer.join().unwrap();

    let protocol = fixture(1);
    let listener = protocol.listener.try_clone().unwrap();
    let peer = std::thread::spawn(move || {
        use std::io::{Read, Write};
        let (mut stream, _) = listener.accept().unwrap();
        let mut len = [0u8; 8];
        stream.read_exact(&mut len).unwrap();
        let mut request = vec![0; u64::from_le_bytes(len) as usize];
        stream.read_exact(&mut request).unwrap();
        stream.write_all(&3u64.to_le_bytes()).unwrap();
        stream.write_all(b"bad").unwrap();
    });
    assert_eq!(
        protocol.sender.send(0, Ref64::NULL),
        Err(RemoteChannelError::ProtocolError)
    );
    peer.join().unwrap();
    let _ = endpoint;
}
