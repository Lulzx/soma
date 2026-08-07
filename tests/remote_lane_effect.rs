use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_future::{
    RemoteFutureClient, RemoteFutureServer, RemoteFutureService,
};
use soma::distributed::remote_lane_effect::*;
use soma::distributed::remote_lane_transport::{
    RemoteLaneClientSession, RemoteLaneOwnerSession, RemoteLaneTransportClient,
    RemoteLaneTransportError, RemoteLaneTransportServer,
};
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

#[test]
fn stage_many_is_atomic_and_refuses_effects_ahead_of_the_boundary() {
    let owner = NodeId(2);
    let actor = r(Kind::Process, 31);
    let target = RemoteRef {
        node: owner,
        entity: r(Kind::Future, 32),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(owner, [31; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: owner,
        actor,
        target,
        rights: Rights::RESOLVE | Rights::SEND,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: 9,
    });
    let mut valid = RemoteLaneApi::new(2, 0, actor);
    valid
        .emit(
            target,
            grant,
            RemoteLaneOperation::FutureResolve {
                value: r(Kind::Object, 33),
            },
        )
        .unwrap();
    let valid = valid.finish().encode();
    let mut invalid = RemoteLaneApi::new(2, 1, actor);
    invalid
        .emit(
            target,
            grant,
            RemoteLaneOperation::MailboxSend {
                sender_sequence: 0,
                bytes: vec![1],
            },
        )
        .unwrap();
    let invalid = invalid.finish().encode();
    let mut service = RemoteLaneEffectService::new(owner, authority);
    service.register_target(target, 1).unwrap();
    let mut exec = Exec {
        calls: vec![],
        block: false,
    };
    assert_eq!(
        service.stage_many(&[&valid, &invalid], 2, &exec),
        Err(RemoteLaneError::Unsupported)
    );
    assert_eq!(service.pending_len(), 0);
    assert!(exec.calls.is_empty());

    assert_eq!(
        service.stage_many(&[&valid], 1, &exec),
        Err(RemoteLaneError::InvalidEnvelope)
    );
    assert_eq!(service.pending_len(), 0);
    assert!(service.apply_epoch(9, &mut exec).is_empty());
    assert!(exec.calls.is_empty());
}

#[test]
fn exact_retry_retires_revoked_effect_and_reapplies_live_effect_in_mixed_batch() {
    let owner = NodeId(90);
    let issuer = NodeId(91);
    let actor = r(Kind::Process, 90);
    let revoked_target = RemoteRef {
        node: owner,
        entity: r(Kind::Future, 91),
    };
    let live_target = RemoteRef {
        node: owner,
        entity: r(Kind::Future, 92),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [90; 32])));
    let issue = |target| {
        authority.lock().unwrap().issue(GrantSpec {
            audience: owner,
            actor,
            target,
            rights: Rights::AWAIT | Rights::RESOLVE,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 4,
        })
    };
    let revoked_grant = issue(revoked_target);
    let live_grant = issue(live_target);

    let revoked_future = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        revoked_target,
        1,
        authority.clone(),
    )));
    let revoked_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let revoked_endpoint = revoked_listener.local_addr().unwrap();
    let revoked_server = {
        let future = revoked_future.clone();
        std::thread::spawn(move || RemoteFutureServer::serve_n(revoked_listener, future, 1))
    };
    let live_future = Arc::new(Mutex::new(RemoteFutureService::new(
        owner,
        live_target,
        1,
        authority.clone(),
    )));
    let live_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let live_endpoint = live_listener.local_addr().unwrap();
    let live_server = {
        let future = live_future.clone();
        std::thread::spawn(move || RemoteFutureServer::serve_n(live_listener, future, 3))
    };

    let mut router = RemoteLaneClientRouter::default();
    router
        .register_future(
            revoked_target,
            RemoteFutureClient::new(revoked_endpoint, revoked_grant, 0),
        )
        .unwrap();
    router
        .register_future(
            live_target,
            RemoteFutureClient::new(live_endpoint, live_grant, 0),
        )
        .unwrap();
    let mut lane_service = RemoteLaneEffectService::new(owner, authority.clone());
    lane_service.register_target(revoked_target, 1).unwrap();
    lane_service.register_target(live_target, 1).unwrap();
    let lane_service = Arc::new(Mutex::new(lane_service));
    let lane_observe = lane_service.clone();
    let lane_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let lane_endpoint = lane_listener.local_addr().unwrap();
    let lane_server = std::thread::spawn(move || {
        RemoteLaneTransportServer::serve_n(
            lane_listener,
            lane_service,
            Arc::new(Mutex::new(router)),
            RemoteLaneOwnerSession::new([90; 16], issuer, owner, [91; 32]),
            2,
        )
    });

    let mut api = RemoteLaneApi::new(0, 0, actor);
    let revoked_id = api
        .emit(
            revoked_target,
            revoked_grant,
            RemoteLaneOperation::FutureAwait,
        )
        .unwrap();
    let live_id = api
        .emit(live_target, live_grant, RemoteLaneOperation::FutureAwait)
        .unwrap();
    let batch = api.finish();
    let mut client = RemoteLaneTransportClient::new(
        lane_endpoint,
        RemoteLaneClientSession::new([90; 16], issuer, owner, [91; 32]),
    );
    let first = client.exchange(0, &[batch]).unwrap();
    let nonce = first.nonce();
    assert!(first
        .outcomes()
        .iter()
        .all(|outcome| matches!(outcome.result, Ok(RemoteLaneApply::WouldBlock))));
    assert_eq!(lane_observe.lock().unwrap().pending_len(), 2);

    assert!(authority.lock().unwrap().revoke(revoked_grant.nonce));
    RemoteFutureClient::new(live_endpoint, live_grant, 0)
        .resolve(r(Kind::Object, 93))
        .unwrap();
    let retried = client.retry(nonce).unwrap();
    assert_eq!(
        retried
            .outcomes()
            .iter()
            .map(|outcome| outcome.request_id)
            .collect::<Vec<_>>(),
        vec![revoked_id, live_id]
    );
    assert!(matches!(
        retried.outcomes()[0].result,
        Err(RemoteLaneError::Authority(_))
    ));
    assert!(matches!(
        retried.outcomes()[1].result,
        Ok(RemoteLaneApply::Applied(_))
    ));
    assert!(client.pending_nonces().is_empty());
    assert!(matches!(
        client.retry(nonce),
        Err(RemoteLaneTransportError::Replay)
    ));

    let service = lane_observe.lock().unwrap();
    assert_eq!(service.pending_len(), 0);
    assert_eq!(service.applied_len(), 1);
    drop(service);
    assert_eq!(live_future.lock().unwrap().applied_resolutions(), 1);
    lane_server.join().unwrap().unwrap();
    revoked_server.join().unwrap().unwrap();
    live_server.join().unwrap().unwrap();
}
