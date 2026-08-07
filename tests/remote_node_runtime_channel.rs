use std::net::TcpListener;
use std::sync::{mpsc, Arc, Mutex};

use soma::abi::continuations::ContinuationState;
use soma::abi::{Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_channel::{
    RemoteChannelClient, RemoteChannelEntry, RemoteChannelError, RemoteChannelService,
    RemoteChannelWaitKind, RemoteReceiveOutcome, RemoteSendOutcome,
};
use soma::distributed::remote_node_runtime::{
    RemoteChannelEffectOutcome, RemoteNodeRuntime, RemoteNodeRuntimeError,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn continuation(kernel: &mut Kernel) -> Ref64 {
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .create_continuation(
            p,
            p,
            ContinuationSpec::new(StateAccess::ReadOnly, 17, 0, vec![], 8),
        )
        .unwrap()
}
fn grant(
    store: &Arc<Mutex<RemoteAuthorityStore>>,
    owner: NodeId,
    target: RemoteRef,
    actor: Ref64,
    rights: u32,
) -> soma::distributed::authority::RemoteGrant {
    store.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 30,
    })
}

#[test]
fn two_kernel_threads_send_backpressure_wake_and_receive_from_owner_queue() {
    let producer_node = NodeId(301);
    let consumer_node = NodeId(302);
    let target = RemoteRef {
        node: consumer_node,
        entity: Ref64::new(55, 1, Kind::Channel),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(300),
        [0x61; 32],
    )));
    let producer_actor = Ref64::new(1, 1, Kind::Process);
    let consumer_actor = Ref64::new(2, 1, Kind::Process);
    let producer_send = grant(
        &authority,
        consumer_node,
        target,
        producer_actor,
        Rights::SEND,
    );
    let consumer_send = grant(
        &authority,
        consumer_node,
        target,
        consumer_actor,
        Rights::SEND,
    );
    let consumer_receive = grant(
        &authority,
        consumer_node,
        target,
        consumer_actor,
        Rights::RECEIVE,
    );
    let service = Arc::new(Mutex::new(RemoteChannelService::new(
        consumer_node,
        target,
        1,
        1,
        authority,
    )));

    let mut consumer_kernel = Kernel::new();
    let consumer = continuation(&mut consumer_kernel);
    let mut consumer_runtime = RemoteNodeRuntime::new(consumer_node, consumer_kernel);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = consumer_runtime
        .register_owned_channel(target, service.clone(), listener)
        .unwrap();
    consumer_runtime
        .register_channel_bridge(
            target,
            RemoteChannelClient::new(endpoint, consumer_send, 0),
            RemoteChannelClient::new(endpoint, consumer_receive, 0),
        )
        .unwrap();
    consumer_runtime
        .receive_after_continuation_runs(
            target,
            consumer,
            RemoteChannelClient::new(endpoint, consumer_receive, 0),
            0,
        )
        .unwrap();

    let mut producer_kernel = Kernel::new();
    let producer = continuation(&mut producer_kernel);
    let mut producer_runtime = RemoteNodeRuntime::new(producer_node, producer_kernel);
    let payload = Ref64::new(90, 1, Kind::Object);
    let overflow = Ref64::new(91, 1, Kind::Object);
    // The duplicate is byte-identical at one fixed effect epoch; the third
    // operation demonstrates capacity backpressure rather than sequencing it.
    producer_runtime
        .send_after_continuation_runs(
            target,
            producer,
            RemoteChannelClient::new(endpoint, producer_send, 0),
            0,
            payload,
        )
        .unwrap();
    producer_runtime
        .send_after_continuation_runs(
            target,
            producer,
            RemoteChannelClient::new(endpoint, producer_send, 0),
            0,
            payload,
        )
        .unwrap();
    producer_runtime
        .send_after_continuation_runs(
            target,
            producer,
            RemoteChannelClient::new(endpoint, producer_send, 0),
            1,
            overflow,
        )
        .unwrap();
    assert!(producer_runtime
        .kernel()
        .channel_len(target.entity)
        .is_err());
    assert!(consumer_runtime
        .kernel()
        .channel_len(target.entity)
        .is_err());
    assert_legal(producer_runtime.kernel());
    assert_legal(consumer_runtime.kernel());

    // Operation-specific rights remain live at the runtime-owned server.
    let wrong = RemoteChannelClient::new(endpoint, consumer_receive, 0);
    assert_eq!(
        wrong.send(0, payload),
        Err(RemoteChannelError::AuthorityDenied)
    );

    let (sent_tx, sent_rx) = mpsc::channel();
    let producer_thread = std::thread::spawn(move || {
        producer_runtime.run_epoch().unwrap();
        assert_legal(producer_runtime.kernel());
        let outcomes = producer_runtime.drain_channel_outcomes();
        assert_eq!(outcomes.len(), 3);
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(
                    outcome,
                    RemoteChannelEffectOutcome::Send {
                        outcome: RemoteSendOutcome::Sent { .. },
                        ..
                    }
                ))
                .count(),
            2
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| **outcome
                    == RemoteChannelEffectOutcome::Send {
                        target,
                        outcome: RemoteSendOutcome::Full
                    })
                .count(),
            1
        );
        sent_tx.send(()).unwrap();
        producer_runtime
    });
    let consumer_thread = std::thread::spawn(move || {
        assert_eq!(
            consumer_runtime.park_on_remote_channel(
                target,
                RemoteChannelWaitKind::Receive,
                consumer,
                17
            ),
            Ok(())
        );
        assert_eq!(
            consumer_runtime.kernel().continuation_state(consumer),
            Ok(ContinuationState::Waiting)
        );
        assert_legal(consumer_runtime.kernel());
        sent_rx.recv().unwrap();
        consumer_runtime.run_epoch().unwrap();
        assert_eq!(
            consumer_runtime.kernel().continuation_state(consumer),
            Ok(ContinuationState::Runnable)
        );
        assert_legal(consumer_runtime.kernel());
        consumer_runtime.run_epoch().unwrap();
        assert_legal(consumer_runtime.kernel());
        assert_eq!(
            consumer_runtime.drain_channel_outcomes(),
            vec![RemoteChannelEffectOutcome::Receive {
                target,
                outcome: RemoteReceiveOutcome::Received(RemoteChannelEntry {
                    value: payload,
                    sender_sequence: 0
                })
            }]
        );
        consumer_runtime
    });
    let producer_runtime = producer_thread.join().unwrap();
    let mut consumer_runtime = consumer_thread.join().unwrap();
    consumer_runtime.join_servers().unwrap();
    assert_eq!(service.lock().unwrap().applied_sends(), 1);
    assert_eq!(service.lock().unwrap().applied_receives(), 1);
    assert!(service.lock().unwrap().entries().is_empty());
    assert!(producer_runtime
        .kernel()
        .channel_len(target.entity)
        .is_err());
    assert!(consumer_runtime
        .kernel()
        .channel_len(target.entity)
        .is_err());
    let mut local = Kernel::new();
    let local_cont = continuation(&mut local);
    local.run_epoch();
    assert_eq!(
        producer_runtime.kernel().continuation_state(producer),
        local.continuation_state(local_cont)
    );
    assert_eq!(
        consumer_runtime.kernel().continuation_state(consumer),
        local.continuation_state(local_cont)
    );
}

#[test]
fn contacted_channel_owner_loss_keeps_waiter_parked() {
    let owner = NodeId(312);
    let target = RemoteRef {
        node: owner,
        entity: Ref64::new(5, 1, Kind::Channel),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(310), [9; 32])));
    let actor = Ref64::new(1, 1, Kind::Process);
    let send = grant(&authority, owner, target, actor, Rights::SEND);
    let receive = grant(&authority, owner, target, actor, Rights::RECEIVE);
    let service = Arc::new(Mutex::new(RemoteChannelService::new(
        owner, target, 1, 1, authority,
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    owner_runtime
        .register_owned_channel(target, service, listener)
        .unwrap();
    let mut kernel = Kernel::new();
    let cont = continuation(&mut kernel);
    let mut client = RemoteNodeRuntime::new(NodeId(311), kernel);
    client
        .register_channel_bridge(
            target,
            RemoteChannelClient::new(endpoint, send, 0),
            RemoteChannelClient::new(endpoint, receive, 0),
        )
        .unwrap();
    client
        .park_on_remote_channel(target, RemoteChannelWaitKind::Receive, cont, 17)
        .unwrap();
    client.run_epoch().unwrap();
    assert_eq!(
        client.kernel().continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );
    owner_runtime.join_servers().unwrap();
    assert_eq!(
        client.run_epoch(),
        Err(RemoteNodeRuntimeError::Channel(
            RemoteChannelError::NodeLost
        ))
    );
    assert_eq!(
        client.kernel().continuation_state(cont),
        Ok(ContinuationState::Waiting)
    );
    assert_legal(client.kernel());
}
