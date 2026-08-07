use soma::abi::{Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_channel::{
    RemoteChannelClient, RemoteChannelService, RemoteSendOutcome,
};
use soma::distributed::remote_future::{RemoteFutureClient, RemoteFutureService};
use soma::distributed::remote_lane_effect::*;
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
#[test]
fn real_continuation_emits_future_await_and_owner_retries_exact_receipt() {
    let owner = NodeId(91);
    let worker = NodeId(92);
    let mut wk = Kernel::new();
    let actor = wk.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(90, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [9; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::AWAIT | Rights::RESOLVE,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 9,
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
    let run_class = 4000;
    let cont = wk
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, run_class, vec![], 8),
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
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let endpoint = owner_runtime
        .register_owned_future(target, canonical, listener)
        .unwrap();
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(target, RemoteFutureClient::new(endpoint, grant, 0))
        .unwrap();
    owner_runtime
        .install_remote_lane_owner(service, router)
        .unwrap();
    worker_runtime.run_epoch().unwrap();
    assert_eq!(
        worker_runtime.kernel().continuation_state(cont).unwrap(),
        soma::abi::ContinuationState::Waiting
    );
    let emitted = worker_runtime.pending_outbound_remote_lane();
    assert_eq!(emitted.len(), 1);
    assert_eq!(
        worker_runtime.pending_outbound_remote_lane()[0]
            .batch
            .effects()[0]
            .request_id,
        emitted[0].batch.effects()[0].request_id
    );
    assert_eq!(emitted[0].continuation, cont);
    let started = worker_runtime
        .kernel()
        .trace_events()
        .iter()
        .find(|event| {
            event.event_kind == soma::abi::EventKind::ContinuationStarted
                && event.continuation == cont
        })
        .unwrap();
    assert_eq!(emitted[0].batch.effects()[0].lane, started.lane);
    assert_eq!(emitted[0].batch.effects()[0].epoch, started.epoch);
    owner_runtime
        .stage_remote_lane_effects(&emitted[0].batch.encode())
        .unwrap();
    owner_runtime.run_epoch().unwrap();
    let pending = owner_runtime.drain_remote_lane_outcomes();
    assert!(matches!(pending[0].result, Ok(RemoteLaneApply::WouldBlock)));
    assert_eq!(
        worker_runtime.kernel().continuation_state(cont).unwrap(),
        soma::abi::ContinuationState::Waiting
    );
    let resolver = RemoteFutureClient::new(endpoint, grant, owner_runtime.kernel().current_epoch());
    resolver.resolve(Ref64::new(99, 1, Kind::Object)).unwrap();
    owner_runtime.run_epoch().unwrap();
    let ready = owner_runtime.drain_remote_lane_outcomes();
    assert!(matches!(ready[0].result, Ok(RemoteLaneApply::Applied(_))));
    owner_runtime.join_servers().unwrap();
}

#[test]
fn real_continuation_channel_backpressure_retries_at_owner() {
    let owner = NodeId(101);
    let worker = NodeId(102);
    let mut wk = Kernel::new();
    let actor = wk.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let filler = Ref64::new(77, 1, Kind::Process);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(76, 1, Kind::Channel),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [4; 32])));
    let issue = |actor| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights: Rights::SEND | Rights::RECEIVE,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 9,
        })
    };
    let grant = issue(actor);
    let filler_grant = issue(filler);
    let instruction = RemoteLaneInstruction {
        opcode: PROGRAM_CHANNEL_SEND,
        reserved: 0,
        target_node: owner.0,
        target_entity: target.entity.to_u64(),
        grant: grant.encode(),
        argument0: 0,
        argument1: 0,
        value: Ref64::new(88, 1, Kind::Object).to_u64(),
        payload_offset: 0,
        payload_len: 0,
    };
    let program = RemoteLaneProgram::validate(vec![instruction], vec![]).unwrap();
    let run_class = 4001;
    let cont = wk
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, run_class, vec![], 8),
        )
        .unwrap();
    let mut worker_runtime = RemoteNodeRuntime::new(worker, wk);
    worker_runtime
        .install_remote_lane_program(run_class, program)
        .unwrap();
    let canonical = Arc::new(Mutex::new(RemoteChannelService::new(
        owner,
        target,
        1,
        1,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let endpoint = owner_runtime
        .register_owned_channel(target, canonical.clone(), listener)
        .unwrap();
    let filler_client = RemoteChannelClient::new(endpoint, filler_grant, 0);
    assert!(matches!(
        filler_client
            .send(0, Ref64::new(87, 1, Kind::Object))
            .unwrap(),
        RemoteSendOutcome::Sent { .. }
    ));
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_channel(target, RemoteChannelClient::new(endpoint, grant, 0))
        .unwrap();
    owner_runtime
        .install_remote_lane_owner(service, router)
        .unwrap();
    worker_runtime.run_epoch().unwrap();
    let emitted = worker_runtime.pending_outbound_remote_lane();
    owner_runtime
        .stage_remote_lane_effects(&emitted[0].batch.encode())
        .unwrap();
    owner_runtime.run_epoch().unwrap();
    let blocked = owner_runtime.drain_remote_lane_outcomes();
    assert!(matches!(blocked[0].result, Ok(RemoteLaneApply::WouldBlock)));
    assert_eq!(
        worker_runtime.kernel().continuation_state(cont).unwrap(),
        soma::abi::ContinuationState::Waiting
    );
    filler_client.receive(0).unwrap();
    owner_runtime.run_epoch().unwrap();
    let sent = owner_runtime.drain_remote_lane_outcomes();
    assert!(matches!(sent[0].result, Ok(RemoteLaneApply::Applied(_))));
    assert_eq!(canonical.lock().unwrap().applied_sends(), 2);
    owner_runtime.join_servers().unwrap();
}

#[test]
fn invalid_multi_wait_program_is_refused_before_kernel_execution() {
    let owner = NodeId(111);
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(1, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(112), [1; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::AWAIT,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 2,
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
    assert!(matches!(
        RemoteLaneProgram::validate(vec![instruction, instruction], vec![]),
        Err(RemoteLaneError::InvalidProgram)
    ));
    assert!(matches!(
        RemoteLaneProgram::validate(vec![instruction], vec![1]),
        Err(RemoteLaneError::InvalidProgram)
    ));
    assert!(matches!(
        RemoteLaneProgram::validate(
            vec![instruction; MAX_REMOTE_LANE_PROGRAM_INSTRUCTIONS + 1],
            vec![],
        ),
        Err(RemoteLaneError::InvalidProgram)
    ));
    assert!(matches!(
        RemoteLaneProgram::validate(
            vec![instruction],
            vec![0; MAX_REMOTE_LANE_PROGRAM_PAYLOAD + 1],
        ),
        Err(RemoteLaneError::InvalidProgram)
    ));
    let mut malformed = instruction;
    malformed.argument0 = 1;
    assert!(matches!(
        RemoteLaneProgram::validate(vec![malformed], vec![]),
        Err(RemoteLaneError::InvalidProgram)
    ));
    let read_grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::READ,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 2,
    });
    let mut read = instruction;
    read.opcode = PROGRAM_OBJECT_READ;
    read.grant = read_grant.encode();
    assert!(matches!(
        RemoteLaneProgram::validate(vec![read], vec![]),
        Err(RemoteLaneError::InvalidProgram)
    ));
    assert_eq!(kernel.current_epoch(), 0);
    assert!(kernel
        .trace_events()
        .iter()
        .all(|event| event.event_kind != soma::abi::EventKind::ContinuationStarted));
}

#[test]
fn unrecoverable_stage_refusal_fault_receipt_does_not_strand_waiter() {
    let owner = NodeId(121);
    let worker = NodeId(122);
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(2, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [2; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::AWAIT,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 3,
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
    let run_class = 4002;
    let cont = kernel
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, run_class, vec![], 8),
        )
        .unwrap();
    let mut runtime = RemoteNodeRuntime::new(worker, kernel);
    runtime
        .install_remote_lane_program(run_class, program)
        .unwrap();
    runtime.run_epoch().unwrap();
    let emission = runtime.pending_outbound_remote_lane();
    let id = emission[0].batch.effects()[0].request_id;
    runtime.fail_outbound_remote_lane(id).unwrap();
    assert_eq!(
        runtime.kernel().continuation_state(cont).unwrap(),
        soma::abi::ContinuationState::Runnable
    );
    assert!(runtime.pending_outbound_remote_lane().is_empty());
    runtime.run_epoch().unwrap();
    assert_eq!(
        runtime.kernel().continuation_state(cont).unwrap(),
        soma::abi::ContinuationState::Faulted
    );
}

#[test]
fn full_outbox_refuses_before_effect_publication() {
    let owner = NodeId(131);
    let worker = NodeId(132);
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(3, 1, Kind::Future),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(worker, [3; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::AWAIT,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 3,
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
    let run_class = 4003;
    let cont = kernel
        .create_continuation(
            actor,
            actor,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, run_class, vec![], 8),
        )
        .unwrap();
    kernel
        .install_remote_lane_program(run_class, program)
        .unwrap();
    kernel.set_remote_lane_outbox_usage(MAX_REMOTE_LANE_OUTBOX_ENTRIES, 0);
    kernel.run_epoch();
    assert_eq!(
        kernel.continuation_state(cont).unwrap(),
        soma::abi::ContinuationState::Faulted
    );
    assert!(kernel.drain_remote_lane_emissions().is_empty());
    assert!(kernel
        .trace_events()
        .iter()
        .all(|event| event.event_kind != soma::abi::EventKind::ContinuationStarted));
}
