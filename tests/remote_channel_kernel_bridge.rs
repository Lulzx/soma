use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::continuations::ContinuationState;
use soma::abi::{Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_channel::{
    RemoteChannelBridge, RemoteChannelClient, RemoteChannelServer, RemoteChannelService,
    RemoteChannelWaitKind,
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

#[test]
fn epoch_probe_wakes_only_the_locally_ready_operation_without_shadow_channel_state() {
    let issuer = NodeId(61);
    let owner = NodeId(62);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(44, 1, Kind::Channel),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0x42; 32])));
    let actor = Ref64::new(3, 1, Kind::Process);
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
    let send_grant = issue(Rights::SEND);
    let receive_grant = issue(Rights::RECEIVE);
    let service = Arc::new(Mutex::new(RemoteChannelService::new(
        owner, target, 1, 1, authority,
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteChannelServer::serve_n(listener, server_service, 2).unwrap()
    });
    let mut bridge = RemoteChannelBridge::new(
        target,
        RemoteChannelClient::new(endpoint, send_grant, 0),
        RemoteChannelClient::new(endpoint, receive_grant, 0),
    );
    let mut kernel = Kernel::new();
    let sender = continuation(&mut kernel);
    let receiver = continuation(&mut kernel);
    bridge
        .register(&mut kernel, RemoteChannelWaitKind::Send, sender, 21)
        .unwrap();
    bridge
        .register(&mut kernel, RemoteChannelWaitKind::Receive, receiver, 22)
        .unwrap();
    assert_eq!(
        kernel.continuation_state(sender),
        Ok(ContinuationState::Waiting)
    );
    assert_eq!(
        kernel.continuation_state(receiver),
        Ok(ContinuationState::Waiting)
    );
    assert_legal(&kernel);
    bridge.sync_epoch_boundary(&mut kernel).unwrap();
    assert_eq!(
        kernel.continuation_state(sender),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(
        kernel.continuation_state(receiver),
        Ok(ContinuationState::Waiting)
    );
    assert_legal(&kernel);
    server.join().unwrap();
}
