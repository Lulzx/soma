use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{
    ContinuationState, EventKind, ExitReason, Kind, ProcessMode, Ref64, Rights, StateAccess,
};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore, RemoteGrant};
use soma::distributed::remote_channel::{
    RemoteChannelClient, RemoteChannelService, RemoteSendOutcome,
};
use soma::distributed::remote_future::{
    RemoteFutureClient, RemoteFutureService, RemoteFutureState,
};
use soma::distributed::remote_lane_effect::{
    RemoteLaneApply, RemoteLaneClientRouter, RemoteLaneEffectService, RemoteLaneInstruction,
    RemoteLaneProgram, PROGRAM_CHANNEL_RECEIVE, PROGRAM_FUTURE_AWAIT, PROGRAM_OBJECT_WRITE,
};
use soma::distributed::remote_lane_transport::{
    RemoteLaneClientSession, RemoteLaneOwnerSession, RemoteLaneTransportClient,
    RemoteLaneTransportError, RemoteLaneTransportServer,
};
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::remote_object::{
    RemoteObjectClient, RemoteObjectServer, RemoteObjectService,
};
use soma::distributed::remote_process::{
    create_request_id, RemoteProcessResponse, RemoteProcessService, RemoteProcessStatus,
    RemoteProcessTcpClient, RemoteProcessTemplate,
};
use soma::distributed::remote_supervision::{
    RemoteSupervisionClient, RemoteSupervisionServer, RemoteSupervisionService,
    RemoteSupervisionState, RemoteTerminalNotice,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn issue(
    authority: &Arc<Mutex<RemoteAuthorityStore>>,
    audience: NodeId,
    actor: Ref64,
    target: RemoteRef,
    rights: u32,
) -> RemoteGrant {
    authority.lock().unwrap().issue(GrantSpec {
        audience,
        actor,
        target,
        rights,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 50,
    })
}

fn instruction(opcode: u16, target: RemoteRef, grant: RemoteGrant) -> RemoteLaneInstruction {
    RemoteLaneInstruction {
        opcode,
        reserved: 0,
        target_node: target.node.0,
        target_entity: target.entity.to_u64(),
        grant: grant.encode(),
        argument0: 0,
        argument1: 0,
        value: 0,
        payload_offset: 0,
        payload_len: 0,
    }
}

fn continuation(kernel: &mut Kernel, actor: Ref64, run_class: u32) -> Ref64 {
    kernel
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, run_class, vec![], 4),
        )
        .unwrap()
}

#[test]
fn authenticated_two_owner_multi_resource_wake_fault_and_lifecycle_are_exact() {
    let worker = NodeId(700);
    let channel_owner = NodeId(701);
    let data_owner = NodeId(702);
    let channel_ref = RemoteRef {
        node: channel_owner,
        entity: Ref64::new(10, 1, Kind::Channel),
    };
    let future_ref = RemoteRef {
        node: data_owner,
        entity: Ref64::new(11, 1, Kind::Future),
    };
    let object_ref = RemoteRef {
        node: data_owner,
        entity: Ref64::new(12, 1, Kind::Object),
    };

    let mut worker_kernel = Kernel::new();
    let actor = worker_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let channel_continuation = continuation(&mut worker_kernel, actor, 7101);
    let future_continuation = continuation(&mut worker_kernel, actor, 7102);
    let initial_processes = worker_kernel.process_count();
    let initial_objects = worker_kernel.object_count();

    let channel_authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [0x31; 32])));
    let data_authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [0x32; 32])));
    let channel_grant = issue(
        &channel_authority,
        channel_owner,
        actor,
        channel_ref,
        Rights::SEND | Rights::RECEIVE,
    );
    let future_grant = issue(
        &data_authority,
        data_owner,
        actor,
        future_ref,
        Rights::AWAIT,
    );
    let object_grant = issue(
        &data_authority,
        data_owner,
        actor,
        object_ref,
        Rights::READ | Rights::WRITE,
    );

    let mut worker_runtime = RemoteNodeRuntime::new(worker, worker_kernel);
    worker_runtime
        .install_remote_lane_program(
            7101,
            RemoteLaneProgram::validate(
                vec![instruction(
                    PROGRAM_CHANNEL_RECEIVE,
                    channel_ref,
                    channel_grant,
                )],
                vec![],
            )
            .unwrap(),
        )
        .unwrap();
    worker_runtime
        .install_remote_lane_program(
            7102,
            RemoteLaneProgram::validate(
                vec![
                    instruction(PROGRAM_FUTURE_AWAIT, future_ref, future_grant),
                    RemoteLaneInstruction {
                        opcode: PROGRAM_OBJECT_WRITE,
                        argument0: 0,
                        argument1: 0,
                        payload_len: 4,
                        ..instruction(PROGRAM_OBJECT_WRITE, object_ref, object_grant)
                    },
                ],
                b"DATA".to_vec(),
            )
            .unwrap(),
        )
        .unwrap();

    let channel_service = Arc::new(Mutex::new(RemoteChannelService::new(
        channel_owner,
        channel_ref,
        1,
        1,
        channel_authority.clone(),
    )));
    let channel_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut channel_owner_runtime = RemoteNodeRuntime::new(channel_owner, Kernel::new());
    let channel_endpoint = channel_owner_runtime
        .register_owned_channel(channel_ref, channel_service.clone(), channel_listener)
        .unwrap();

    let future_service = Arc::new(Mutex::new(RemoteFutureService::new(
        data_owner,
        future_ref,
        1,
        data_authority.clone(),
    )));
    let future_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut data_owner_runtime = RemoteNodeRuntime::new(data_owner, Kernel::new());
    let future_endpoint = data_owner_runtime
        .register_owned_future(future_ref, future_service.clone(), future_listener)
        .unwrap();

    let object_service = Arc::new(Mutex::new(RemoteObjectService::new(
        data_owner,
        object_ref,
        1,
        b"seed".to_vec(),
        data_authority.clone(),
    )));
    let object_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let object_endpoint = object_listener.local_addr().unwrap();
    let object_for_server = object_service.clone();
    let object_server = std::thread::spawn(move || {
        RemoteObjectServer::serve_n(object_listener, object_for_server, 1)
    });

    let channel_lane_service = Arc::new(Mutex::new(RemoteLaneEffectService::new(
        channel_owner,
        channel_authority.clone(),
    )));
    channel_lane_service
        .lock()
        .unwrap()
        .register_target(channel_ref, 1)
        .unwrap();
    let mut channel_router = RemoteLaneClientRouter::default();
    channel_router
        .register_channel(
            channel_ref,
            RemoteChannelClient::new(channel_endpoint, channel_grant, 0),
        )
        .unwrap();
    let channel_router = Arc::new(Mutex::new(channel_router));
    let channel_lane_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let channel_lane_endpoint = channel_lane_listener.local_addr().unwrap();
    let channel_lane_observe = channel_lane_service.clone();
    let channel_lane_server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            channel_lane_listener,
            channel_lane_service,
            channel_router,
            RemoteLaneOwnerSession::new([1; 16], worker, channel_owner, [0xa1; 32]),
            3,
        )
    });

    let data_lane_service = Arc::new(Mutex::new(RemoteLaneEffectService::new(
        data_owner,
        data_authority.clone(),
    )));
    {
        let mut service = data_lane_service.lock().unwrap();
        service.register_target(future_ref, 1).unwrap();
        service.register_target(object_ref, 1).unwrap();
    }
    let mut data_router = RemoteLaneClientRouter::default();
    data_router
        .register_future(
            future_ref,
            RemoteFutureClient::new(future_endpoint, future_grant, 0),
        )
        .unwrap();
    data_router
        .register_object(
            object_ref,
            RemoteObjectClient::new(object_endpoint, object_grant, 0),
        )
        .unwrap();
    let data_router = Arc::new(Mutex::new(data_router));
    let data_lane_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let data_lane_endpoint = data_lane_listener.local_addr().unwrap();
    let data_lane_observe = data_lane_service.clone();
    let data_lane_server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            data_lane_listener,
            data_lane_service,
            data_router,
            RemoteLaneOwnerSession::new([2; 16], worker, data_owner, [0xa2; 32]),
            3,
        )
    });

    // Both continuations take the real Kernel special-dispatch path. The data
    // program emits one wait and one bounded object write as a single batch.
    worker_runtime.run_epoch().unwrap();
    let emissions = worker_runtime.pending_outbound_remote_lane();
    assert_eq!(emissions.len(), 2);
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(channel_continuation),
        Ok(ContinuationState::Waiting)
    );
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(future_continuation),
        Ok(ContinuationState::Waiting)
    );
    let channel_batch = emissions
        .iter()
        .find(|e| e.continuation == channel_continuation)
        .unwrap()
        .batch
        .clone();
    let future_batch = emissions
        .iter()
        .find(|e| e.continuation == future_continuation)
        .unwrap()
        .batch
        .clone();

    let channel_session = RemoteLaneClientSession::new([1; 16], worker, channel_owner, [0xa1; 32]);
    worker_runtime
        .bind_remote_lane_waiter_session(channel_batch.effects()[0].request_id, &channel_session)
        .unwrap();
    let mut channel_transport =
        RemoteLaneTransportClient::new(channel_lane_endpoint, channel_session);
    let blocked_channel = channel_transport.exchange(0, &[channel_batch]).unwrap();
    let channel_nonce = blocked_channel.nonce();
    assert_eq!(blocked_channel.outcomes().len(), 1);
    assert!(matches!(
        blocked_channel.outcomes()[0].result,
        Ok(RemoteLaneApply::WouldBlock)
    ));
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(blocked_channel)
        .unwrap();

    let data_session = RemoteLaneClientSession::new([2; 16], worker, data_owner, [0xa2; 32]);
    worker_runtime
        .bind_remote_lane_waiter_session(future_batch.effects()[0].request_id, &data_session)
        .unwrap();
    let mut data_transport = RemoteLaneTransportClient::new(data_lane_endpoint, data_session);
    let first_data = data_transport.exchange(0, &[future_batch]).unwrap();
    let data_nonce = first_data.nonce();
    assert_eq!(first_data.outcomes().len(), 2);
    assert_eq!(
        first_data
            .outcomes()
            .iter()
            .filter(|o| matches!(o.result, Ok(RemoteLaneApply::WouldBlock)))
            .count(),
        1
    );
    assert_eq!(
        first_data
            .outcomes()
            .iter()
            .filter(|o| matches!(o.result, Ok(RemoteLaneApply::Applied(_))))
            .count(),
        1
    );
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(first_data)
        .unwrap();
    assert!(worker_runtime
        .pending_outbound_remote_lane()
        .iter()
        .any(|emission| emission.continuation == future_continuation));

    // Make the channel receive ready, then deliberately drop one authenticated
    // response. Retrying retains byte-identical request/nonce and applies once.
    let payload = Ref64::new(44, 1, Kind::Object);
    assert!(matches!(
        RemoteChannelClient::new(channel_endpoint, channel_grant, 0)
            .send(0, payload)
            .unwrap(),
        RemoteSendOutcome::Sent { .. }
    ));
    channel_transport
        .send_without_receiving(channel_nonce)
        .unwrap();
    let ready_channel = channel_transport.retry(channel_nonce).unwrap();
    assert!(matches!(
        ready_channel.outcomes()[0].result,
        Ok(RemoteLaneApply::Applied(_))
    ));
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(ready_channel)
        .unwrap();
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(channel_continuation),
        Ok(ContinuationState::Runnable)
    );
    assert_eq!(channel_transport.pending_nonces(), Vec::<u64>::new());
    assert!(matches!(
        channel_transport.retry(channel_nonce),
        Err(RemoteLaneTransportError::Replay)
    ));

    // Live revocation is checked before both the lane ledger and routed service.
    // The object outcome remains an exact cached success while the future wait
    // becomes a terminal authenticated denial, faulting (not waking) its worker.
    assert!(data_authority.lock().unwrap().revoke(future_grant.nonce));
    data_transport.send_without_receiving(data_nonce).unwrap();
    let denied_data = data_transport.retry(data_nonce).unwrap();
    assert_eq!(denied_data.outcomes().len(), 2);
    assert!(denied_data.outcomes().iter().any(|o| {
        o.target == future_ref
            && matches!(
                o.result,
                Err(soma::distributed::remote_lane_effect::RemoteLaneError::Authority(_))
            )
    }));
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(denied_data)
        .unwrap();
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(future_continuation),
        Ok(ContinuationState::Runnable)
    );
    worker_runtime.run_epoch().unwrap();
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(channel_continuation),
        Ok(ContinuationState::Completed)
    );
    assert_eq!(
        worker_runtime
            .kernel()
            .continuation_state(future_continuation),
        Ok(ContinuationState::Faulted)
    );

    // Existing authoritative process lifecycle runs on a second real owner
    // kernel. Its exact create retry yields one immutable terminal receipt.
    let process_ref = RemoteRef {
        node: channel_owner,
        entity: Ref64::new(90, 1, Kind::Object),
    };
    let process_authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [0x41; 32])));
    let process_grant = issue(
        &process_authority,
        channel_owner,
        Ref64::NULL,
        process_ref,
        Rights::WRITE,
    );
    let process_service = Arc::new(Mutex::new(
        RemoteProcessService::new(channel_owner, process_ref, 1, process_authority.clone())
            .unwrap(),
    ));
    process_service
        .lock()
        .unwrap()
        .register_template(RemoteProcessTemplate {
            id: 5,
            mode: ProcessMode::Serial,
            entry: ContinuationSpec::new(StateAccess::ReadOnly, 0xffff_ff00, 0, vec![], 1),
            restart_limit: 0,
        })
        .unwrap();
    let process_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let process_endpoint = channel_owner_runtime
        .register_owned_process_server(process_ref, process_service.clone(), process_listener)
        .unwrap();
    let process_client = RemoteProcessTcpClient::new(process_endpoint, channel_owner);
    let create_id = create_request_id(5, 0, &process_grant);
    assert_eq!(
        process_client
            .create(create_id, 5, 0, &process_grant)
            .unwrap(),
        None
    );
    channel_owner_runtime.run_epoch().unwrap();
    let process_receipt = match process_client
        .create(create_id, 5, 0, &process_grant)
        .unwrap()
        .unwrap()
    {
        RemoteProcessResponse::Created(receipt) => receipt,
        other => panic!("unexpected lifecycle response: {other:?}"),
    };
    assert_eq!(
        process_client
            .create(create_id, 5, 0, &process_grant)
            .unwrap(),
        Some(RemoteProcessResponse::Created(process_receipt))
    );
    let read_grant = issue(
        &process_authority,
        channel_owner,
        Ref64::NULL,
        process_receipt.process,
        Rights::READ,
    );
    assert_eq!(
        process_client
            .query(process_receipt, 1, &read_grant)
            .unwrap()
            .status,
        RemoteProcessStatus::Terminal(ExitReason::Failed)
    );

    // Publish that owner-authored terminal fact through the existing
    // authoritative supervision service; exact retry cannot duplicate it.
    let supervision_authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [0x42; 32])));
    let supervision_grant = issue(
        &supervision_authority,
        channel_owner,
        Ref64::NULL,
        process_receipt.process,
        Rights::AWAIT | Rights::WRITE,
    );
    let supervision_service = Arc::new(Mutex::new(RemoteSupervisionService::new(
        channel_owner,
        process_receipt.process,
        process_receipt.version,
        supervision_authority,
    )));
    let supervision_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let supervision_endpoint = supervision_listener.local_addr().unwrap();
    let supervision_for_server = supervision_service.clone();
    let supervision_server = std::thread::spawn(move || {
        RemoteSupervisionServer::serve_n(supervision_listener, supervision_for_server, 3)
    });
    let supervision_client =
        RemoteSupervisionClient::new(supervision_endpoint, supervision_grant, 1);
    let notice = RemoteTerminalNotice::new(
        process_receipt.process,
        ExitReason::Failed,
        1,
        channel_owner_runtime.kernel().current_epoch(),
    );
    let terminal = RemoteSupervisionState::Terminal(notice);
    assert_eq!(supervision_client.publish(notice).unwrap(), terminal);
    assert_eq!(supervision_client.publish(notice).unwrap(), terminal);
    assert_eq!(supervision_client.poll().unwrap(), terminal);

    // Exact final resources, receipts, trace, and bounded ledgers.
    data_owner_runtime.run_epoch().unwrap();
    assert_eq!(
        future_service.lock().unwrap().state(),
        RemoteFutureState::Pending
    );
    assert_eq!(future_service.lock().unwrap().applied_resolutions(), 0);
    assert_eq!(object_service.lock().unwrap().bytes(), b"DATA");
    assert_eq!(object_service.lock().unwrap().version(), 1);
    assert_eq!(object_service.lock().unwrap().applied_writes(), 1);
    assert_eq!(channel_service.lock().unwrap().applied_sends(), 1);
    assert_eq!(channel_service.lock().unwrap().applied_receives(), 1);
    assert!(channel_service.lock().unwrap().entries().is_empty());
    assert_eq!(channel_lane_observe.lock().unwrap().pending_len(), 0);
    assert_eq!(channel_lane_observe.lock().unwrap().applied_len(), 1);
    assert_eq!(data_lane_observe.lock().unwrap().pending_len(), 0);
    assert_eq!(data_lane_observe.lock().unwrap().applied_len(), 1);
    assert_eq!(process_service.lock().unwrap().process_count(), 1);
    assert_eq!(process_service.lock().unwrap().ledger_len(), 1);
    assert_eq!(
        supervision_service.lock().unwrap().applied_publications(),
        1
    );
    assert_eq!(worker_runtime.kernel().process_count(), initial_processes);
    assert_eq!(worker_runtime.kernel().object_count(), initial_objects);
    assert_eq!(worker_runtime.kernel().future_count(), 0);
    assert!(worker_runtime
        .kernel()
        .channel_len(channel_ref.entity)
        .is_err());
    assert_eq!(channel_owner_runtime.kernel().future_count(), 0);
    assert!(data_owner_runtime
        .kernel()
        .channel_len(channel_ref.entity)
        .is_err());
    let starts: Vec<_> = worker_runtime
        .kernel()
        .trace_events()
        .iter()
        .filter(|event| event.event_kind == EventKind::ContinuationStarted)
        .map(|event| event.continuation)
        .collect();
    assert_eq!(
        starts,
        vec![
            channel_continuation,
            future_continuation,
            channel_continuation,
            future_continuation,
        ]
    );
    assert_legal(worker_runtime.kernel());
    assert_legal(channel_owner_runtime.kernel());
    assert_legal(data_owner_runtime.kernel());

    channel_lane_server.join().unwrap().unwrap();
    data_lane_server.join().unwrap().unwrap();
    object_server.join().unwrap().unwrap();
    supervision_server.join().unwrap().unwrap();
    channel_owner_runtime.join_servers().unwrap();
    data_owner_runtime.join_servers().unwrap();
}
