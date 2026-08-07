use soma::abi::{Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::{RemoteFutureClient, RemoteFutureService};
use soma::distributed::remote_lane_effect::{
    RemoteLaneClientRouter, RemoteLaneEffectService, RemoteLaneInstruction, RemoteLaneProgram,
    PROGRAM_FUTURE_AWAIT,
};
use soma::distributed::remote_lane_transport::{
    RemoteLaneClientSession, RemoteLaneOwnerSession, RemoteLaneTransportClient,
    RemoteLaneTransportError, RemoteLaneTransportServer,
};
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

#[test]
fn signed_exact_response_wakes_a_real_worker_runtime_over_tcp_once() {
    let owner = NodeId(401);
    let worker = NodeId(402);
    let key = [0x5a; 32];
    let mut wk = Kernel::new();
    let actor = wk.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(7, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [9; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::AWAIT | Rights::RESOLVE,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 4,
    });
    let instruction = RemoteLaneInstruction {
        opcode: PROGRAM_FUTURE_AWAIT,
        reserved: 0,
        target_node: owner.0,
        target_entity: target.entity.to_u64(),
        grant: grant.encode(),
        argument0: 0,
        argument1: 0,
        value: 0,
        payload_offset: 0,
        payload_len: 0,
    };
    let program = RemoteLaneProgram::validate(vec![instruction], vec![]).unwrap();
    let run_class = 9001;
    let continuation = wk
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, run_class, vec![], 4),
        )
        .unwrap();
    let mut worker_runtime = RemoteNodeRuntime::new(worker, wk);
    worker_runtime
        .install_remote_lane_program(run_class, program)
        .unwrap();

    let canonical = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        target,
        1,
        authority.clone(),
    )));
    let future_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let future_endpoint = owner_runtime
        .register_owned_future(target, canonical.clone(), future_listener)
        .unwrap();
    let mut lane_service = RemoteLaneEffectService::new(owner, authority);
    lane_service.register_target(target, 1).unwrap();
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(target, RemoteFutureClient::new(future_endpoint, grant, 0))
        .unwrap();
    let lane_service = Arc::new(Mutex::new(lane_service));
    let lane_observe = lane_service.clone();
    let router = Arc::new(Mutex::new(router));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            listener,
            lane_service,
            router,
            RemoteLaneOwnerSession::new([3; 16], worker, owner, key),
            3,
        )
    });

    worker_runtime.run_epoch().unwrap();
    let emissions = worker_runtime.pending_outbound_remote_lane();
    let mut client = RemoteLaneTransportClient::new(
        endpoint,
        RemoteLaneClientSession::new([3; 16], worker, owner, key),
    );
    let verified = client.exchange(0, &[emissions[0].batch.clone()]).unwrap();
    let nonce = verified.nonce();
    assert!(matches!(
        verified.outcomes()[0].result,
        Ok(soma::distributed::remote_lane_effect::RemoteLaneApply::WouldBlock)
    ));
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(verified)
        .unwrap();
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(continuation)
            .unwrap(),
        soma::abi::ContinuationState::Waiting
    );

    let resolver = RemoteFutureClient::new(future_endpoint, grant, 0);
    resolver.resolve(Ref64::new(8, 1, Kind::Object)).unwrap();
    client.send_without_receiving(nonce).unwrap();
    let verified = client.retry(nonce).unwrap();
    assert!(matches!(
        verified.outcomes()[0].result,
        Ok(soma::distributed::remote_lane_effect::RemoteLaneApply::Applied(_))
    ));
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(verified)
        .unwrap();
    worker_runtime.run_epoch().unwrap();
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(continuation)
            .unwrap(),
        soma::abi::ContinuationState::Completed
    );
    assert!(matches!(
        client.retry(nonce),
        Err(RemoteLaneTransportError::Replay)
    ));
    server.join().unwrap().unwrap();
    assert_eq!(lane_observe.lock().unwrap().applied_len(), 1);
    owner_runtime.join_servers().unwrap();
}
