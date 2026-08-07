use soma::abi::{Kind, Ref64, Rights};
use soma::distributed::authority::{
    GrantSpec, RemoteAuthorityError, RemoteAuthorityStore, RemoteGrant,
};
use soma::distributed::{NodeId, RemoteRef};

fn fixture() -> (RemoteAuthorityStore, GrantSpec) {
    let target = RemoteRef {
        node: NodeId(2),
        entity: Ref64::new(7, 3, Kind::Module),
    };
    (
        RemoteAuthorityStore::new(NodeId(1), [0xA5; 32]),
        GrantSpec {
            audience: NodeId(2),
            actor: Ref64::new(4, 1, Kind::Process),
            target,
            rights: Rights::READ,
            object_version: 9,
            valid_from_epoch: 10,
            valid_until_epoch: 20,
        },
    )
}

#[test]
fn a_remote_reference_is_not_authority() {
    let (mut store, spec) = fixture();
    let grant = store.issue(spec);
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::READ, 9, 10),
        Ok(())
    );
    assert_eq!(
        store.authorize(&grant, NodeId(3), spec.target, Rights::READ, 9, 10),
        Err(RemoteAuthorityError::WrongAudience)
    );
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::WRITE, 9, 10),
        Err(RemoteAuthorityError::InsufficientRights)
    );
}

#[test]
fn revocation_is_observed_at_remote_use() {
    let (mut store, spec) = fixture();
    let grant = store.issue(spec);
    assert!(store.revoke(grant.nonce));
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::READ, 9, 11),
        Err(RemoteAuthorityError::Revoked)
    );
}

#[test]
fn signed_grants_round_trip_and_tampering_fails_closed() {
    let (mut store, spec) = fixture();
    let grant = store.issue(spec);
    let encoded = grant.encode();
    assert_eq!(RemoteGrant::decode(&encoded), Some(grant));
    let mut tampered = encoded;
    tampered[42] ^= Rights::WRITE as u8;
    let tampered = RemoteGrant::decode(&tampered).unwrap();
    assert_eq!(
        store.authorize(&tampered, NodeId(2), spec.target, Rights::READ, 9, 11),
        Err(RemoteAuthorityError::InvalidSignature)
    );
}

#[test]
fn logical_epoch_bounds_are_inclusive_and_enforced() {
    let (mut store, spec) = fixture();
    let grant = store.issue(spec);
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::READ, 9, 9),
        Err(RemoteAuthorityError::NotYetValid)
    );
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::READ, 9, 20),
        Ok(())
    );
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::READ, 9, 21),
        Err(RemoteAuthorityError::Expired)
    );
}

#[test]
fn protocol_and_object_versions_are_pinned_at_use() {
    let (mut store, spec) = fixture();
    let grant = store.issue(spec);
    assert_eq!(
        store.authorize(&grant, NodeId(2), spec.target, Rights::READ, 8, 11),
        Err(RemoteAuthorityError::ObjectVersionMismatch)
    );
    let mut wrong_protocol = grant;
    wrong_protocol.version += 1;
    assert_eq!(
        store.authorize(&wrong_protocol, NodeId(2), spec.target, Rights::READ, 9, 11),
        Err(RemoteAuthorityError::UnsupportedVersion)
    );
}
