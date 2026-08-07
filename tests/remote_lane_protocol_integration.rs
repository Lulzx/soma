use soma::abi::{Kind, ProcessMode, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_channel::{RemoteChannelClient, RemoteChannelService};
use soma::distributed::remote_future::{RemoteFutureClient, RemoteFutureService};
use soma::distributed::remote_lane_effect::*;
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::remote_object::{
    RemoteObjectClient, RemoteObjectServer, RemoteObjectService,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::{invariants, order};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
fn rr(k: Kind, s: u32) -> Ref64 {
    Ref64::new(s, 1, k)
}
#[test]
fn protocol_integration_has_exact_results_and_no_shadow_kernel_control() {
    let owner = NodeId(80);
    let worker = NodeId(81);
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [8; 32])));
    let future = RemoteRef {
        node: owner,
        entity: rr(Kind::Future, 70),
    };
    let channel = RemoteRef {
        node: owner,
        entity: rr(Kind::Channel, 71),
    };
    let object = RemoteRef {
        node: owner,
        entity: rr(Kind::Object, 72),
    };
    let mut worker_kernel = Kernel::new();
    let actor = worker_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mut worker_runtime = RemoteNodeRuntime::new(worker, worker_kernel);
    let grant = |target, rights| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 5,
        })
    };
    let fg = grant(future, Rights::AWAIT | Rights::RESOLVE);
    let cg = grant(channel, Rights::SEND | Rights::RECEIVE);
    let og = grant(object, Rights::READ | Rights::WRITE);
    let fs = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        future,
        1,
        authority.clone(),
    )));
    let cs = Arc::new(Mutex::new(RemoteChannelService::new(
        owner,
        channel,
        1,
        2,
        authority.clone(),
    )));
    let os = Arc::new(Mutex::new(RemoteObjectService::new(
        owner,
        object,
        1,
        b"a".to_vec(),
        authority.clone(),
    )));
    let fl = TcpListener::bind("127.0.0.1:0").unwrap();
    let cl = TcpListener::bind("127.0.0.1:0").unwrap();
    let ol = TcpListener::bind("127.0.0.1:0").unwrap();
    let oe = ol.local_addr().unwrap();
    let os2 = os.clone();
    let ot = std::thread::spawn(move || RemoteObjectServer::serve_n(ol, os2, 2));
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let fe = owner_runtime
        .register_owned_future(future, fs.clone(), fl)
        .unwrap();
    let ce = owner_runtime
        .register_owned_channel(channel, cs.clone(), cl)
        .unwrap();
    let mut service = RemoteLaneEffectService::new(owner, authority);
    for t in [future, channel, object] {
        service.register_target(t, 1).unwrap()
    }
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(future, RemoteFutureClient::new(fe, fg, 0))
        .unwrap();
    router
        .register_channel(channel, RemoteChannelClient::new(ce, cg, 0))
        .unwrap();
    router
        .register_object(object, RemoteObjectClient::new(oe, og, 0))
        .unwrap();
    owner_runtime
        .install_remote_lane_owner(service, router)
        .unwrap();
    let mut lane = RemoteLaneApi::new(worker_runtime.kernel().current_epoch(), 5, actor);
    lane.emit(
        channel,
        cg,
        RemoteLaneOperation::ChannelSend {
            sequence: 0,
            value: rr(Kind::Object, 90),
        },
    )
    .unwrap();
    lane.emit(
        channel,
        cg,
        RemoteLaneOperation::ChannelReceive { sequence: 0 },
    )
    .unwrap();
    lane.emit(
        object,
        og,
        RemoteLaneOperation::ObjectWrite {
            expected_version: 0,
            offset: 0,
            bytes: b"z".to_vec(),
        },
    )
    .unwrap();
    lane.emit(
        object,
        og,
        RemoteLaneOperation::ObjectRead {
            offset: 0,
            length: 1,
        },
    )
    .unwrap();
    lane.emit(
        future,
        fg,
        RemoteLaneOperation::FutureResolve {
            value: rr(Kind::Object, 91),
        },
    )
    .unwrap();
    lane.emit(future, fg, RemoteLaneOperation::FutureAwait)
        .unwrap();
    let batch = lane.finish();
    let ids: Vec<_> = batch.effects().iter().map(|e| e.request_id).collect();
    let frame = batch.encode();
    assert_eq!(ids.len(), 6);
    owner_runtime.stage_remote_lane_effects(&frame).unwrap();
    owner_runtime.run_epoch().unwrap();
    let outcomes = owner_runtime.drain_remote_lane_outcomes();
    assert_eq!(outcomes.len(), 6);
    assert!(outcomes
        .iter()
        .all(|o| matches!(o.result, Ok(RemoteLaneApply::Applied(_)))));
    assert_eq!(cs.lock().unwrap().applied_sends(), 1);
    assert_eq!(cs.lock().unwrap().applied_receives(), 1);
    assert_eq!(os.lock().unwrap().bytes(), b"z");
    assert_eq!(os.lock().unwrap().applied_writes(), 1);
    assert_eq!(fs.lock().unwrap().applied_resolutions(), 1);
    // The worker Kernel only supplies/cancels an actor identity; RemoteLaneApi is
    // called directly and the frame is staged in-process. This comparison proves
    // only that the otherwise-empty owner Kernel epoch is unaffected and that no
    // foreign descriptor shadow entered either Kernel; it is not distributed I18.
    let mut reference = Kernel::new();
    reference.run_epoch();
    assert!(order::conforms(&reference, owner_runtime.kernel()).is_empty());
    invariants::assert_legal(owner_runtime.kernel());
    invariants::assert_legal(worker_runtime.kernel());
    worker_runtime
        .kernel_mut()
        .cancel_process(SYSTEM_PRINCIPAL, actor)
        .unwrap();
    invariants::assert_legal(worker_runtime.kernel());
    owner_runtime.join_servers().unwrap();
    ot.join().unwrap().unwrap();
}
