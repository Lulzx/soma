use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, EXPAND_RESUME_0, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{ExpandFrame, SearchFrame};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_mailbox_ingress::{
    RemoteMailboxApplyStatus, RemoteMailboxClient, RemoteMailboxEnvelope, RemoteMailboxError,
    RemoteMailboxIngress, RemoteMailboxReceipt, RemoteMailboxSendOutcome,
};
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

fn leaf(kernel: &mut Kernel, process: Ref64) -> Ref64 {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                SEARCH_BRANCH,
                SEARCH_BRANCH,
                bytes,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap()
}

fn send_grant(
    authority: &Arc<Mutex<RemoteAuthorityStore>>,
    owner: NodeId,
    actor: Ref64,
    target: RemoteRef,
) -> soma::distributed::authority::RemoteGrant {
    authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::SEND,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 30,
    })
}

#[test]
fn two_kernels_tcp_wakes_the_real_receiver_and_applies_duplicate_once() {
    let owner = NodeId(701);
    let sender_node = NodeId(702);
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(700),
        [0x71; 32],
    )));

    let mut owner_kernel = Kernel::new();
    let requester = owner_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = owner_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    owner_kernel
        .grant_capability(SYSTEM_PRINCIPAL, receiver, requester, Rights::SEND, 0, 0)
        .unwrap();
    let mut frame = Vec::new();
    ExpandFrame::initial(0, requester).encode(&mut frame);
    let receiver_continuation = owner_kernel
        .create_continuation(
            receiver,
            receiver,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                EXPAND_RESUME_0,
                EXPAND_RESUME_0,
                frame,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();
    let target = RemoteRef {
        node: owner,
        entity: receiver,
    };
    let ingress = Arc::new(Mutex::new(RemoteMailboxIngress::new(
        owner,
        target,
        1,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, owner_kernel);
    let endpoint = owner_runtime
        .register_mailbox_ingress(target, ingress.clone(), listener)
        .unwrap();
    owner_runtime.run_epoch().unwrap();
    assert_eq!(
        owner_runtime.kernel().mailbox_recv_waiter_count(receiver),
        1
    );

    let mut sender_kernel = Kernel::new();
    let sender = sender_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sender_continuation = leaf(&mut sender_kernel, sender);
    let mut sender_runtime = RemoteNodeRuntime::new(sender_node, sender_kernel);
    sender_runtime.run_epoch().unwrap();
    assert!(!sender_runtime
        .kernel()
        .continuation_state(sender_continuation)
        .unwrap()
        .is_live());
    let sender_grant = send_grant(&authority, owner, sender, target);
    let client = RemoteMailboxClient::new(
        endpoint,
        sender_grant,
        owner_runtime.kernel().current_epoch(),
    );
    let value = 42u64.to_le_bytes().to_vec();
    let first = client.send(0, value.clone(), None, false).unwrap();
    let duplicate = client.send(0, value, None, false).unwrap();
    assert!(matches!(first, RemoteMailboxSendOutcome::Staged(_)));
    assert!(matches!(duplicate, RemoteMailboxSendOutcome::Duplicate(_)));

    owner_runtime.run_epoch().unwrap();
    assert!(matches!(
        client
            .send(0, 42u64.to_le_bytes().to_vec(), None, false)
            .unwrap(),
        RemoteMailboxSendOutcome::Applied(_)
    ));
    assert_eq!(ingress.lock().unwrap().applied_count(), 1);
    assert_eq!(owner_runtime.drain_mailbox_outcomes().len(), 1);
    let receiver_starts = owner_runtime
        .kernel()
        .trace_events()
        .iter()
        .filter(|event| {
            event.event_kind == soma::abi::EventKind::ContinuationStarted
                && event.continuation == receiver_continuation
        })
        .count();
    assert_eq!(
        receiver_starts, 2,
        "the remote message resumed the real handler"
    );
    assert_eq!(sender_runtime.kernel().process_count(), 1);
    assert_eq!(owner_runtime.kernel().process_count(), 2);
    assert_legal(sender_runtime.kernel());
    assert_legal(owner_runtime.kernel());

    // I18 control: the same boundary input through the local canonical path
    // leaves the actual receiver continuation in the same state.
    let mut local = Kernel::new();
    let local_requester = local.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let local_receiver = local.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    local
        .grant_capability(
            SYSTEM_PRINCIPAL,
            local_receiver,
            local_requester,
            Rights::SEND,
            0,
            0,
        )
        .unwrap();
    let mut local_frame = Vec::new();
    ExpandFrame::initial(0, local_requester).encode(&mut local_frame);
    let local_continuation = local
        .create_continuation(
            local_receiver,
            local_receiver,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                EXPAND_RESUME_0,
                EXPAND_RESUME_0,
                local_frame,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();
    local.run_epoch();
    local
        .ingest_remote_message(
            local_receiver,
            RemoteMailboxEnvelope {
                receipt: RemoteMailboxReceipt {
                    actor_node: sender_grant.issuer,
                    actor: sender,
                    sender_sequence: 0,
                    capability: None,
                },
                value: 42u64.to_le_bytes().to_vec(),
            }
            .encode(),
            false,
        )
        .unwrap();
    local.run_epoch();
    assert_eq!(
        owner_runtime
            .kernel()
            .continuation_state(receiver_continuation),
        local.continuation_state(local_continuation)
    );
    assert_legal(&local);
    owner_runtime.join_servers().unwrap();
}

#[test]
fn revocation_is_checked_before_staged_replay_and_sequence_failures_do_not_enqueue() {
    let owner = NodeId(711);
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(710),
        [0x72; 32],
    )));
    let mut kernel = Kernel::new();
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let actor = Ref64::new(88, 1, Kind::Process);
    let target = RemoteRef {
        node: owner,
        entity: receiver,
    };
    let grant = send_grant(&authority, owner, actor, target);
    let ingress = Arc::new(Mutex::new(RemoteMailboxIngress::new(
        owner,
        target,
        1,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut runtime = RemoteNodeRuntime::new(owner, kernel);
    let endpoint = runtime
        .register_mailbox_ingress(target, ingress.clone(), listener)
        .unwrap();
    let client = RemoteMailboxClient::new(endpoint, grant, 0);
    assert_eq!(
        client.send(1, vec![1], None, false),
        Err(RemoteMailboxError::InvalidSequence)
    );
    client.send(0, vec![2], None, false).unwrap();
    assert!(authority.lock().unwrap().revoke(grant.nonce));
    runtime.run_epoch().unwrap();
    let outcomes = runtime.drain_mailbox_outcomes();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(
        outcomes[0].status,
        RemoteMailboxApplyStatus::AuthorityDenied
    );
    assert!(runtime
        .kernel()
        .mailbox_entries(receiver)
        .unwrap()
        .is_empty());
    assert_eq!(
        client.send(0, vec![2], None, false),
        Err(RemoteMailboxError::AuthorityDenied)
    );
    runtime.join_servers().unwrap();
}

#[test]
fn urgent_ordering_and_owner_mailbox_backpressure_are_preserved() {
    let owner = NodeId(721);
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(720),
        [0x73; 32],
    )));
    let mut kernel = Kernel::new();
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let actor = Ref64::new(89, 1, Kind::Process);
    let target = RemoteRef {
        node: owner,
        entity: receiver,
    };
    let grant = send_grant(&authority, owner, actor, target);
    let transferred = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target: RemoteRef {
            node: NodeId(799),
            entity: Ref64::new(4, 1, Kind::Object),
        },
        rights: Rights::TRANSFER,
        object_version: 3,
        valid_from_epoch: 0,
        valid_until_epoch: 30,
    });
    let ingress = Arc::new(Mutex::new(RemoteMailboxIngress::new(
        owner, target, 1, authority,
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut runtime = RemoteNodeRuntime::new(owner, kernel);
    let endpoint = runtime
        .register_owned_mailbox(target, ingress, listener)
        .unwrap();
    let client = RemoteMailboxClient::new(endpoint, grant, 0);
    client.send(0, vec![0], None, false).unwrap();
    client.send(1, vec![1], Some(transferred), true).unwrap();
    runtime.run_epoch().unwrap();

    let entries = runtime.kernel().mailbox_entries(receiver).unwrap();
    assert_eq!(entries.len(), 2);
    assert_ne!(
        entries[0].flags & soma::abi::messages::MESSAGE_FLAG_URGENT,
        0
    );
    assert_eq!(entries[0].sender_sequence, 0);
    assert_eq!(entries[1].sender_sequence, 1);
    assert_eq!(entries[0].sender, SYSTEM_PRINCIPAL);
    let receipt_target = runtime
        .kernel()
        .capability_entry(receiver, entries[0].transferred_capability)
        .unwrap()
        .target;
    assert_eq!(receipt_target, entries[0].payload);
    let envelope = RemoteMailboxEnvelope::decode(
        runtime
            .kernel_mut()
            .object_bytes(receiver, receipt_target)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(envelope.value, vec![1]);
    assert_eq!(envelope.receipt.actor, actor);
    assert_eq!(envelope.receipt.actor_node, NodeId(720));
    assert_eq!(envelope.receipt.sender_sequence, 1);
    assert_eq!(envelope.receipt.capability, Some(transferred));

    for value in 2..8u8 {
        let payload = runtime.kernel_mut().create_object(
            SYSTEM_PRINCIPAL,
            soma::abi::ObjectKind::MessagePayload,
            vec![value],
        );
        runtime
            .kernel_mut()
            .ingest_message(
                SYSTEM_PRINCIPAL,
                SYSTEM_PRINCIPAL,
                receiver,
                payload,
                Ref64::NULL,
            )
            .unwrap();
    }
    client.send(2, vec![8], None, false).unwrap();
    runtime.run_epoch().unwrap();
    assert_eq!(runtime.kernel().mailbox_entries(receiver).unwrap().len(), 8);
    assert!(runtime
        .drain_mailbox_outcomes()
        .iter()
        .any(|outcome| outcome.sender_sequence == 2
            && outcome.status == RemoteMailboxApplyStatus::Backpressured));
    assert!(matches!(
        client.send(2, vec![8], None, false).unwrap(),
        RemoteMailboxSendOutcome::Backpressured(_)
    ));
    assert_legal(runtime.kernel());
    runtime.join_servers().unwrap();
}

#[test]
fn identical_actor_bits_from_two_issuers_have_independent_sequence_zero() {
    let owner = NodeId(731);
    let first_authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(732),
        [0x74; 32],
    )));
    let second_authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(
        NodeId(733),
        [0x75; 32],
    )));
    let mut kernel = Kernel::new();
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let actor = Ref64::new(1, 1, Kind::Process);
    let target = RemoteRef {
        node: owner,
        entity: receiver,
    };
    let first_grant = send_grant(&first_authority, owner, actor, target);
    let second_grant = send_grant(&second_authority, owner, actor, target);
    let ingress = Arc::new(Mutex::new(RemoteMailboxIngress::new(
        owner,
        target,
        1,
        first_authority,
    )));
    ingress.lock().unwrap().trust_authority(second_authority);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut runtime = RemoteNodeRuntime::new(owner, kernel);
    let endpoint = runtime
        .register_owned_mailbox(target, ingress, listener)
        .unwrap();
    RemoteMailboxClient::new(endpoint, first_grant, 0)
        .send(0, vec![1], None, false)
        .unwrap();
    RemoteMailboxClient::new(endpoint, second_grant, 0)
        .send(0, vec![2], None, false)
        .unwrap();
    runtime.run_epoch().unwrap();
    let outcomes = runtime.drain_mailbox_outcomes();
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes.iter().all(|outcome| outcome.sender_sequence == 0));
    assert_eq!(outcomes[0].actor_node, NodeId(732));
    assert_eq!(outcomes[1].actor_node, NodeId(733));
    assert_legal(runtime.kernel());
    runtime.join_servers().unwrap();
}
