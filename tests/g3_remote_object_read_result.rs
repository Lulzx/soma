use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{ContinuationState, Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore, RemoteGrant};
use soma::distributed::remote_future::{RemoteFutureClient, RemoteFutureService};
use soma::distributed::remote_lane_effect::{
    RemoteLaneApply, RemoteLaneClientRouter, RemoteLaneEffectService, RemoteLaneInstruction,
    RemoteLaneProgram, RemoteLaneValue, PROGRAM_FUTURE_AWAIT, PROGRAM_OBJECT_READ,
};
use soma::distributed::remote_lane_transport::{
    RemoteLaneClientSession, RemoteLaneOwnerSession, RemoteLaneTransportClient,
    RemoteLaneTransportServer,
};
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::remote_object::{
    RemoteObjectClient, RemoteObjectServer, RemoteObjectService,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};

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
        valid_until_epoch: 20,
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

#[test]
fn authenticated_mixed_reads_publish_exact_versioned_frame_records() {
    let worker = NodeId(810);
    let owner = NodeId(811);
    let future = RemoteRef {
        node: owner,
        entity: Ref64::new(1, 1, Kind::Future),
    };
    let object = RemoteRef {
        node: owner,
        entity: Ref64::new(2, 1, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [0x51; 32])));

    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let continuation = kernel
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, 8201, 8201, vec![0xcc; 48], 3),
        )
        .unwrap();
    let future_grant = issue(
        &authority,
        owner,
        actor,
        future,
        Rights::AWAIT | Rights::RESOLVE,
    );
    let object_grant = issue(&authority, owner, actor, object, Rights::READ);
    let program = RemoteLaneProgram::validate(
        vec![
            instruction(PROGRAM_FUTURE_AWAIT, future, future_grant),
            RemoteLaneInstruction {
                opcode: PROGRAM_OBJECT_READ,
                argument0: 1,
                argument1: 4,
                value: 0,
                ..instruction(PROGRAM_OBJECT_READ, object, object_grant)
            },
            RemoteLaneInstruction {
                opcode: PROGRAM_OBJECT_READ,
                argument0: 5,
                argument1: 3,
                value: 20,
                ..instruction(PROGRAM_OBJECT_READ, object, object_grant)
            },
        ],
        vec![],
    )
    .unwrap();
    let mut worker_runtime = RemoteNodeRuntime::new(worker, kernel);
    worker_runtime
        .install_remote_lane_program(8201, program)
        .unwrap();

    let future_service = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        future,
        1,
        authority.clone(),
    )));
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let future_endpoint = owner_runtime
        .register_owned_future(
            future,
            future_service,
            TcpListener::bind("127.0.0.1:0").unwrap(),
        )
        .unwrap();
    let resolved = Ref64::new(9, 1, Kind::Object);
    RemoteFutureClient::new(future_endpoint, future_grant, 0)
        .resolve(resolved)
        .unwrap();

    let object_service = Arc::new(Mutex::new(RemoteObjectService::new(
        owner,
        object,
        1,
        b"abcdefghij".to_vec(),
        authority.clone(),
    )));
    let object_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let object_endpoint = object_listener.local_addr().unwrap();
    let object_server = {
        let service = object_service.clone();
        std::thread::spawn(move || RemoteObjectServer::serve_n(object_listener, service, 2))
    };

    let service = Arc::new(Mutex::new(RemoteLaneEffectService::new(
        owner,
        authority.clone(),
    )));
    {
        let mut service = service.lock().unwrap();
        service.register_target(future, 1).unwrap();
        service.register_target(object, 1).unwrap();
    }
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(
            future,
            RemoteFutureClient::new(future_endpoint, future_grant, 0),
        )
        .unwrap();
    router
        .register_object(
            object,
            RemoteObjectClient::new(object_endpoint, object_grant, 0),
        )
        .unwrap();
    let lane_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let lane_endpoint = lane_listener.local_addr().unwrap();
    let session_id = [7; 16];
    let session_key = [0xa7; 32];
    let lane_server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            lane_listener,
            service,
            Arc::new(Mutex::new(router)),
            RemoteLaneOwnerSession::new(session_id, worker, owner, session_key),
            1,
        )
    });

    worker_runtime.run_epoch().unwrap();
    let emission = worker_runtime.pending_outbound_remote_lane()[0].clone();
    assert_eq!(emission.batch.effects().len(), 3);
    let session = RemoteLaneClientSession::new(session_id, worker, owner, session_key);
    worker_runtime
        .bind_remote_lane_waiter_session(emission.batch.effects()[0].request_id, &session)
        .unwrap();
    let outcomes = RemoteLaneTransportClient::new(lane_endpoint, session)
        .exchange(0, &[emission.batch])
        .unwrap();
    assert_eq!(
        outcomes
            .outcomes()
            .iter()
            .filter(|outcome| matches!(
                outcome.result,
                Ok(RemoteLaneApply::Applied(RemoteLaneValue::Bytes { .. }))
            ))
            .count(),
        2
    );
    worker_runtime
        .accept_authenticated_remote_lane_outcomes(outcomes)
        .unwrap();
    assert_eq!(
        worker_runtime.kernel().continuation_state(continuation),
        Ok(ContinuationState::Runnable)
    );
    let frame = worker_runtime
        .kernel()
        .continuation_frame(continuation)
        .unwrap();
    let bytes = worker_runtime
        .kernel_mut()
        .object_bytes(actor, frame)
        .unwrap()
        .to_vec();
    assert_eq!(&bytes[0..8], &0u64.to_le_bytes());
    assert_eq!(&bytes[8..12], b"bcde");
    assert_eq!(&bytes[20..28], &0u64.to_le_bytes());
    assert_eq!(&bytes[28..31], b"fgh");
    assert!(bytes[12..20].iter().all(|byte| *byte == 0xcc));
    worker_runtime.run_epoch().unwrap();
    assert_eq!(
        worker_runtime.kernel().continuation_state(continuation),
        Ok(ContinuationState::Completed)
    );

    lane_server.join().unwrap().unwrap();
    object_server.join().unwrap().unwrap();
    drop(owner_runtime);
}

#[test]
fn read_result_destinations_are_bounded_nonempty_and_nonoverlapping() {
    let worker = NodeId(820);
    let owner = NodeId(821);
    let actor = Ref64::new(1, 1, Kind::Process);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(2, 1, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [3; 32])));
    let grant = issue(
        &authority,
        owner,
        actor,
        target,
        Rights::READ | Rights::AWAIT,
    );
    let await_target = RemoteRef {
        node: owner,
        entity: Ref64::new(3, 1, Kind::Future),
    };
    let await_grant = issue(&authority, owner, actor, await_target, Rights::AWAIT);
    let base = vec![instruction(PROGRAM_FUTURE_AWAIT, await_target, await_grant)];
    for (destination, length) in [(0, 0), (131_068, 8)] {
        let mut instructions = base.clone();
        instructions.push(RemoteLaneInstruction {
            opcode: PROGRAM_OBJECT_READ,
            argument1: length,
            value: destination,
            ..instruction(PROGRAM_OBJECT_READ, target, grant)
        });
        assert!(RemoteLaneProgram::validate(instructions, vec![]).is_err());
    }
    let mut overlapping = base;
    overlapping.extend([
        RemoteLaneInstruction {
            opcode: PROGRAM_OBJECT_READ,
            argument1: 8,
            value: 4,
            ..instruction(PROGRAM_OBJECT_READ, target, grant)
        },
        RemoteLaneInstruction {
            opcode: PROGRAM_OBJECT_READ,
            argument1: 4,
            value: 12,
            ..instruction(PROGRAM_OBJECT_READ, target, grant)
        },
    ]);
    assert!(RemoteLaneProgram::validate(overlapping, vec![]).is_err());
}

#[test]
fn read_destination_outside_actual_private_frame_refuses_before_emission() {
    let worker = NodeId(830);
    let owner = NodeId(831);
    let future = RemoteRef {
        node: owner,
        entity: Ref64::new(1, 1, Kind::Future),
    };
    let object = RemoteRef {
        node: owner,
        entity: Ref64::new(2, 1, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [8; 32])));
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let continuation = kernel
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, 8301, 8301, vec![0x5a; 16], 1),
        )
        .unwrap();
    let future_grant = issue(&authority, owner, actor, future, Rights::AWAIT);
    let object_grant = issue(&authority, owner, actor, object, Rights::READ);
    let program = RemoteLaneProgram::validate(
        vec![
            instruction(PROGRAM_FUTURE_AWAIT, future, future_grant),
            RemoteLaneInstruction {
                opcode: PROGRAM_OBJECT_READ,
                argument1: 1,
                value: 8,
                ..instruction(PROGRAM_OBJECT_READ, object, object_grant)
            },
        ],
        vec![],
    )
    .unwrap();
    let mut runtime = RemoteNodeRuntime::new(worker, kernel);
    runtime.install_remote_lane_program(8301, program).unwrap();
    runtime.run_epoch().unwrap();
    assert!(runtime.pending_outbound_remote_lane().is_empty());
    assert_eq!(
        runtime.kernel().continuation_state(continuation),
        Ok(ContinuationState::Faulted)
    );
}
