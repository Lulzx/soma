use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::{
    RemoteFutureClient, RemoteFutureService, RemoteFutureState,
};
use soma::distributed::remote_lane_effect::{
    RemoteLaneApi, RemoteLaneApply, RemoteLaneClientRouter, RemoteLaneEffectService,
    RemoteLaneOperation,
};
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::Kernel;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
fn r(kind: Kind, slot: u32) -> Ref64 {
    Ref64::new(slot, 1, kind)
}
#[test]
fn generic_protocol_reaches_owner_only_at_runtime_boundary() {
    let owner = NodeId(51);
    let issuer = NodeId(52);
    let actor = r(Kind::Process, 7);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Future, 2),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [7; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::RESOLVE,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 4,
    });
    let canonical = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        target,
        1,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let endpoint = runtime
        .register_owned_future(target, canonical.clone(), listener)
        .unwrap();
    let mut lane_service = RemoteLaneEffectService::new(owner, authority);
    lane_service.register_target(target, 1).unwrap();
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(target, RemoteFutureClient::new(endpoint, grant, 0))
        .unwrap();
    runtime
        .install_remote_lane_owner(lane_service, router)
        .unwrap();
    let mut lane = RemoteLaneApi::new(0, 3, actor);
    let id = lane
        .emit(
            target,
            grant,
            RemoteLaneOperation::FutureResolve {
                value: r(Kind::Object, 99),
            },
        )
        .unwrap();
    runtime
        .stage_remote_lane_effects(&lane.finish().encode())
        .unwrap();
    assert_eq!(
        canonical.lock().unwrap().state(),
        RemoteFutureState::Pending
    );
    runtime.run_epoch().unwrap();
    let out = runtime.drain_remote_lane_outcomes();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].request_id, id);
    assert!(matches!(out[0].result, Ok(RemoteLaneApply::Applied(_))));
    assert_eq!(canonical.lock().unwrap().applied_resolutions(), 1);
    runtime.join_servers().unwrap();
}
