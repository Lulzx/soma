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
    let correct_service = lane_service.clone();
    let correct_router = router.clone();
    let server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            listener,
            correct_service,
            correct_router,
            RemoteLaneOwnerSession::new([3; 16], worker, owner, key),
            3,
        )
    });
    // This is a genuinely authenticated second session on the same node route,
    // not a forged response. Its matching deterministic request must still be
    // unusable for a waiter explicitly bound to the first session.
    let other_key = [0x6b; 32];
    let other_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let other_endpoint = other_listener.local_addr().unwrap();
    let other_server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            other_listener,
            lane_service,
            router,
            RemoteLaneOwnerSession::new([4; 16], worker, owner, other_key),
            1,
        )
    });

    worker_runtime.run_epoch().unwrap();
    let emissions = worker_runtime.pending_outbound_remote_lane();
    let request_id = emissions[0].batch.effects()[0].request_id;
    let session = RemoteLaneClientSession::new([3; 16], worker, owner, key);
    worker_runtime
        .bind_remote_lane_waiter_session(request_id, &session)
        .unwrap();

    let mut other_client = RemoteLaneTransportClient::new(
        other_endpoint,
        RemoteLaneClientSession::new([4; 16], worker, owner, other_key),
    );
    let other_verified = other_client
        .exchange(0, &[emissions[0].batch.clone()])
        .unwrap();
    let state_before = worker_runtime
        .kernel()
        .continuation_state(continuation)
        .unwrap();
    let trace_len_before = worker_runtime.kernel().trace_events().len();
    let outbox_before: Vec<_> = worker_runtime
        .pending_outbound_remote_lane()
        .iter()
        .flat_map(|emission| emission.batch.effects())
        .map(|effect| effect.request_id)
        .collect();
    assert!(worker_runtime
        .accept_authenticated_remote_lane_outcomes(other_verified)
        .is_err());
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(continuation)
            .unwrap(),
        state_before
    );
    assert_eq!(
        worker_runtime.kernel().trace_events().len(),
        trace_len_before
    );
    assert_eq!(
        worker_runtime
            .pending_outbound_remote_lane()
            .iter()
            .flat_map(|emission| emission.batch.effects())
            .map(|effect| effect.request_id)
            .collect::<Vec<_>>(),
        outbox_before
    );

    let mut client = RemoteLaneTransportClient::new(endpoint, session);
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
    other_server.join().unwrap().unwrap();
    assert_eq!(lane_observe.lock().unwrap().applied_len(), 1);
    owner_runtime.join_servers().unwrap();
}
