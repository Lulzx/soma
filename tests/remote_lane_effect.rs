use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::{
    RemoteFutureClient, RemoteFutureServer, RemoteFutureService,
};
use soma::distributed::remote_lane_effect::*;
use soma::distributed::{NodeId, RemoteRef};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

fn r(kind: Kind, slot: u64) -> Ref64 {
    Ref64::new(slot as u32, 1, kind)
}
struct Exec {
    calls: Vec<RemoteLaneRequestId>,
    block: bool,
}
impl RemoteLaneExecutor for Exec {
    fn supports(&self, o: &RemoteLaneOperation) -> bool {
        !matches!(
            o,
            RemoteLaneOperation::MailboxSend { .. } | RemoteLaneOperation::ObserveTerminal
        )
    }
    fn apply(&mut self, e: &RemoteLaneEffect) -> Result<RemoteLaneApply, RemoteLaneError> {
        self.calls.push(e.request_id);
        if self.block {
            return Ok(RemoteLaneApply::WouldBlock);
        };
        Ok(match &e.operation {
            RemoteLaneOperation::ObjectRead { .. } => {
                RemoteLaneApply::Applied(RemoteLaneValue::Bytes {
                    version: 2,
                    bytes: b"ok".to_vec(),
                })
            }
            _ => RemoteLaneApply::Applied(RemoteLaneValue::Unit),
        })
    }
}
#[test]
fn bounded_authenticated_multiplexed_apply_once_and_canonical_order() {
    let owner = NodeId(2);
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(owner, [9; 32])));
    let targets = [
        RemoteRef {
            node: owner,
            entity: r(Kind::Future, 1),
        },
        RemoteRef {
            node: owner,
            entity: r(Kind::Channel, 2),
        },
        RemoteRef {
            node: owner,
            entity: r(Kind::Object, 3),
        },
    ];
    let rights = [Rights::RESOLVE, Rights::SEND, Rights::READ];
    let mut grants = Vec::new();
    for (target, right) in targets.into_iter().zip(rights) {
        grants.push(authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor: r(Kind::Process, 7),
            target,
            rights: right,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 9,
        }));
    }
    let mut api = RemoteLaneApi::new(3, 4, r(Kind::Process, 7));
    let a = api
        .emit(
            targets[0],
            grants[0],
            RemoteLaneOperation::FutureResolve {
                value: r(Kind::Object, 11),
            },
        )
        .unwrap();
    let b = api
        .emit(
            targets[1],
            grants[1],
            RemoteLaneOperation::ChannelSend {
                sequence: 0,
                value: r(Kind::Object, 12),
            },
        )
        .unwrap();
    let c = api
        .emit(
            targets[2],
            grants[2],
            RemoteLaneOperation::ObjectRead {
                offset: 0,
                length: 2,
            },
        )
        .unwrap();
    assert_ne!(a, b);
    assert_ne!(b, c);
    let frame = api.finish().encode();
    let mut service = RemoteLaneEffectService::new(owner, authority.clone());
    for t in targets {
        service.register_target(t, 1).unwrap()
    }
    let mut exec = Exec {
        calls: vec![],
        block: false,
    };
    service.stage(&frame, &exec).unwrap();
    let out = service.apply_epoch(3, &mut exec);
    assert_eq!(out.len(), 3);
    assert_eq!(exec.calls, vec![a, b, c]);
    assert_eq!(service.applied_len(), 3);
    // Exact retry is served from the ledger without a second executor mutation.
    service.stage(&frame, &exec).unwrap();
    let _ = service.apply_epoch(3, &mut exec);
    assert_eq!(exec.calls.len(), 3);
    // Revocation precedes replay lookup.
    authority.lock().unwrap().revoke(grants[0].nonce);
    service
        .stage(
            &RemoteLaneEffectBatch::decode(&frame).unwrap().encode(),
            &exec,
        )
        .unwrap_err();
}
#[test]
fn unsupported_is_rejected_before_any_apply_and_would_block_retries() {
    let owner = NodeId(2);
    let actor = r(Kind::Process, 1);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Process, 2),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(owner, [3; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::SEND,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 5,
    });
    let mut api = RemoteLaneApi::new(0, 0, actor);
    api.emit(
        target,
        grant,
        RemoteLaneOperation::MailboxSend {
            sender_sequence: 0,
            bytes: b"x".to_vec(),
        },
    )
    .unwrap();
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let exec = Exec {
        calls: vec![],
        block: false,
    };
    assert_eq!(
        service.stage(&api.finish().encode(), &exec),
        Err(RemoteLaneError::Unsupported)
    );
    assert!(exec.calls.is_empty());
    assert_eq!(service.pending_len(), 0);
}

#[test]
fn exact_effect_grant_is_rebound_and_signed_issuer_qualifies_actor() {
    let owner = NodeId(40);
    let actor = r(Kind::Process, 9);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Future, 2),
    };
    let first = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(41), [1; 32])));
    let second = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(42), [2; 32])));
    let issue = |a: &Arc<Mutex<RemoteAuthorityStore>>| {
        a.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights: Rights::RESOLVE,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 3,
        })
    };
    let good = issue(&first);
    let other = issue(&second);
    let e1 = RemoteLaneEffect::new(
        0,
        0,
        0,
        actor,
        target,
        good,
        RemoteLaneOperation::FutureResolve {
            value: r(Kind::Object, 3),
        },
    );
    let e2 = RemoteLaneEffect::new(
        0,
        0,
        0,
        actor,
        target,
        other,
        RemoteLaneOperation::FutureResolve {
            value: r(Kind::Object, 3),
        },
    );
    assert_eq!(e1.actor_node, NodeId(41));
    assert_eq!(e2.actor_node, NodeId(42));
    assert_ne!(e1.request_id, e2.request_id);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let future = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        target,
        1,
        first.clone(),
    )));
    let server = std::thread::spawn(move || RemoteFutureServer::serve_n(listener, future, 1));
    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(target, RemoteFutureClient::new(endpoint, good, 0))
        .unwrap();
    let mut service = RemoteLaneEffectService::new(owner, first);
    service.trust_authority(second).unwrap();
    service.register_target(target, 1).unwrap();
    let mut batch = RemoteLaneEffectBatch::default();
    batch.push(e2).unwrap();
    service.stage(&batch.encode(), &router).unwrap();
    let out = service.apply_epoch(0, &mut router);
    assert_eq!(out[0].result, Err(RemoteLaneError::AuthorityDenied));
    server.join().unwrap().unwrap();
}

#[test]
fn positional_collision_and_late_new_effect_are_rejected() {
    let owner = NodeId(60);
    let actor = r(Kind::Process, 1);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Future, 2),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(61), [6; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::RESOLVE,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 5,
    });
    let first = RemoteLaneEffect::new(
        0,
        0,
        0,
        actor,
        target,
        grant,
        RemoteLaneOperation::FutureResolve {
            value: r(Kind::Object, 1),
        },
    );
    let mut b = RemoteLaneEffectBatch::default();
    b.push(first.clone()).unwrap();
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let mut exec = Exec {
        calls: vec![],
        block: false,
    };
    service.stage(&b.encode(), &exec).unwrap();
    service.apply_epoch(0, &mut exec);
    let collision = RemoteLaneEffect::new(
        0,
        0,
        0,
        actor,
        target,
        grant,
        RemoteLaneOperation::FutureResolve {
            value: r(Kind::Object, 2),
        },
    );
    let mut b = RemoteLaneEffectBatch::default();
    b.push(collision).unwrap();
    assert_eq!(
        service.stage(&b.encode(), &exec),
        Err(RemoteLaneError::InvalidEnvelope)
    );
    let late = RemoteLaneEffect::new(
        0,
        1,
        0,
        actor,
        target,
        grant,
        RemoteLaneOperation::FutureResolve {
            value: r(Kind::Object, 3),
        },
    );
    let mut b = RemoteLaneEffectBatch::default();
    b.push(late).unwrap();
    assert_eq!(
        service.stage(&b.encode(), &exec),
        Err(RemoteLaneError::InvalidEnvelope)
    );
    let mut replay = RemoteLaneEffectBatch::default();
    replay.push(first).unwrap();
    service.stage(&replay.encode(), &exec).unwrap();
    service.apply_epoch(0, &mut exec);
    assert_eq!(exec.calls.len(), 1);
}

struct Flaky {
    calls: usize,
}
impl RemoteLaneExecutor for Flaky {
    fn supports(&self, _: &RemoteLaneOperation) -> bool {
        true
    }
    fn apply(&mut self, _: &RemoteLaneEffect) -> Result<RemoteLaneApply, RemoteLaneError> {
        self.calls += 1;
        if self.calls == 1 {
            Err(RemoteLaneError::NodeUnavailable)
        } else {
            Ok(RemoteLaneApply::Applied(RemoteLaneValue::Unit))
        }
    }
}
#[test]
fn temporary_transport_failure_retries_exact_id_and_ledger_capacity_refuses_transactionally() {
    let owner = NodeId(70);
    let actor = r(Kind::Process, 1);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Object, 2),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(71), [7; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::READ,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 20,
    });
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let mut flaky = Flaky { calls: 0 };
    let e = RemoteLaneEffect::new(
        0,
        0,
        0,
        actor,
        target,
        grant,
        RemoteLaneOperation::ObjectRead {
            offset: 0,
            length: 1,
        },
    );
    let id = e.request_id;
    let mut one = RemoteLaneEffectBatch::default();
    one.push(e).unwrap();
    service.stage(&one.encode(), &flaky).unwrap();
    let first = service.apply_epoch(0, &mut flaky);
    assert_eq!(first[0].result, Err(RemoteLaneError::NodeUnavailable));
    assert_eq!(service.pending_len(), 1);
    let second = service.apply_epoch(1, &mut flaky);
    assert!(matches!(second[0].result, Ok(RemoteLaneApply::Applied(_))));
    assert_eq!(second[0].request_id, id);
    assert_eq!(flaky.calls, 2);
    assert_eq!(service.applied_len(), 1);
    let mut oversized = RemoteLaneEffectBatch::default();
    for ordinal in 0..9 {
        oversized
            .push(RemoteLaneEffect::new(
                2,
                1,
                ordinal,
                actor,
                target,
                grant,
                RemoteLaneOperation::ObjectRead {
                    offset: 0,
                    length: 1024 * 1024,
                },
            ))
            .unwrap();
    }
    assert_eq!(
        service.stage(&oversized.encode(), &flaky),
        Err(RemoteLaneError::JournalFull)
    );
    // Refusal did not reserve a position: a smaller request at the same position is accepted.
    let mut small = RemoteLaneEffectBatch::default();
    small
        .push(RemoteLaneEffect::new(
            2,
            1,
            0,
            actor,
            target,
            grant,
            RemoteLaneOperation::ObjectRead {
                offset: 0,
                length: 1,
            },
        ))
        .unwrap();
    service.stage(&small.encode(), &flaky).unwrap();
}

struct BadProtocol;
impl RemoteLaneExecutor for BadProtocol {
    fn supports(&self, _: &RemoteLaneOperation) -> bool {
        true
    }
    fn apply(&mut self, _: &RemoteLaneEffect) -> Result<RemoteLaneApply, RemoteLaneError> {
        Err(RemoteLaneError::Protocol)
    }
}
#[test]
fn malformed_protocol_response_is_terminal_not_unbounded_retry() {
    let owner = NodeId(75);
    let actor = r(Kind::Process, 1);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Object, 2),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(NodeId(76), [5; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::READ,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 2,
    });
    let effect = RemoteLaneEffect::new(
        0,
        0,
        0,
        actor,
        target,
        grant,
        RemoteLaneOperation::ObjectRead {
            offset: 0,
            length: 1,
        },
    );
    let mut batch = RemoteLaneEffectBatch::default();
    batch.push(effect).unwrap();
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let mut executor = BadProtocol;
    service.stage(&batch.encode(), &executor).unwrap();
    let out = service.apply_epoch(0, &mut executor);
    assert_eq!(out[0].result, Err(RemoteLaneError::Protocol));
    assert_eq!(service.pending_len(), 0);
    assert_eq!(service.applied_len(), 1);
}
