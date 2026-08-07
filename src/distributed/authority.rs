use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use super::{NodeId, RemoteRef};
use crate::abi::Ref64;

pub const REMOTE_AUTH_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteAuthorityError {
    UnsupportedVersion,
    InvalidSignature,
    WrongIssuer,
    WrongAudience,
    WrongTarget,
    ObjectVersionMismatch,
    InsufficientRights,
    NotYetValid,
    Expired,
    Revoked,
    UnknownGrant,
}

/// Wire-stable proof that one node delegated attenuated authority to another.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RemoteGrant {
    pub version: u16,
    pub issuer: NodeId,
    pub audience: NodeId,
    pub actor: Ref64,
    pub target: RemoteRef,
    pub rights: u32,
    pub object_version: u32,
    pub valid_from_epoch: u32,
    pub valid_until_epoch: u32,
    pub nonce: u64,
    pub signature: [u8; 32],
}

impl std::fmt::Debug for RemoteGrant {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RemoteGrant")
            .field("issuer", &self.issuer)
            .field("audience", &self.audience)
            .field("actor", &self.actor)
            .field("target", &self.target)
            .field("rights", &self.rights)
            .field("nonce", &self.nonce)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GrantSpec {
    pub audience: NodeId,
    pub actor: Ref64,
    pub target: RemoteRef,
    pub rights: u32,
    pub object_version: u32,
    pub valid_from_epoch: u32,
    pub valid_until_epoch: u32,
}

/// Issuer-side live grant registry. Signed bytes prove provenance; membership
/// here makes revocation observable at the remote operation, rather than only
/// when a token happens to expire.
pub struct RemoteAuthorityStore {
    node: NodeId,
    secret: [u8; 32],
    next_nonce: u64,
    issued: HashMap<u64, RemoteGrant>,
    revoked: HashSet<u64>,
}

impl RemoteAuthorityStore {
    pub fn new(node: NodeId, secret: [u8; 32]) -> Self {
        Self {
            node,
            secret,
            next_nonce: 1,
            issued: HashMap::new(),
            revoked: HashSet::new(),
        }
    }

    pub fn issue(&mut self, spec: GrantSpec) -> RemoteGrant {
        let nonce = self.next_nonce;
        self.next_nonce = self.next_nonce.wrapping_add(1).max(1);
        let mut grant = RemoteGrant {
            version: REMOTE_AUTH_VERSION,
            issuer: self.node,
            audience: spec.audience,
            actor: spec.actor,
            target: spec.target,
            rights: spec.rights,
            object_version: spec.object_version,
            valid_from_epoch: spec.valid_from_epoch,
            valid_until_epoch: spec.valid_until_epoch,
            nonce,
            signature: [0; 32],
        };
        grant.signature = hmac_sha256(&self.secret, &grant.signing_bytes());
        self.issued.insert(nonce, grant);
        grant
    }

    pub fn revoke(&mut self, nonce: u64) -> bool {
        if self.issued.contains_key(&nonce) {
            self.revoked.insert(nonce);
            true
        } else {
            false
        }
    }

    pub fn authorize(
        &self,
        grant: &RemoteGrant,
        audience: NodeId,
        target: RemoteRef,
        required_rights: u32,
        object_version: u32,
        epoch: u32,
    ) -> Result<(), RemoteAuthorityError> {
        if grant.version != REMOTE_AUTH_VERSION {
            return Err(RemoteAuthorityError::UnsupportedVersion);
        }
        if grant.issuer != self.node {
            return Err(RemoteAuthorityError::WrongIssuer);
        }
        let expected = hmac_sha256(&self.secret, &grant.signing_bytes());
        if !constant_time_eq(&expected, &grant.signature) {
            return Err(RemoteAuthorityError::InvalidSignature);
        }
        let Some(issued) = self.issued.get(&grant.nonce) else {
            return Err(RemoteAuthorityError::UnknownGrant);
        };
        if issued != grant {
            return Err(RemoteAuthorityError::InvalidSignature);
        }
        if self.revoked.contains(&grant.nonce) {
            return Err(RemoteAuthorityError::Revoked);
        }
        if grant.audience != audience {
            return Err(RemoteAuthorityError::WrongAudience);
        }
        if grant.target != target {
            return Err(RemoteAuthorityError::WrongTarget);
        }
        if grant.object_version != object_version {
            return Err(RemoteAuthorityError::ObjectVersionMismatch);
        }
        if grant.rights & required_rights != required_rights {
            return Err(RemoteAuthorityError::InsufficientRights);
        }
        if epoch < grant.valid_from_epoch {
            return Err(RemoteAuthorityError::NotYetValid);
        }
        if epoch > grant.valid_until_epoch {
            return Err(RemoteAuthorityError::Expired);
        }
        Ok(())
    }
}

impl RemoteGrant {
    pub const ENCODED_LEN: usize = 98;

    pub fn encode(self) -> [u8; Self::ENCODED_LEN] {
        let mut out = [0u8; Self::ENCODED_LEN];
        out[..66].copy_from_slice(&self.signing_bytes());
        out[66..].copy_from_slice(&self.signature);
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::ENCODED_LEN {
            return None;
        }
        Some(Self {
            version: u16::from_le_bytes(bytes[0..2].try_into().ok()?),
            issuer: NodeId(u64::from_le_bytes(bytes[2..10].try_into().ok()?)),
            audience: NodeId(u64::from_le_bytes(bytes[10..18].try_into().ok()?)),
            actor: Ref64::from_u64(u64::from_le_bytes(bytes[18..26].try_into().ok()?)),
            target: RemoteRef {
                node: NodeId(u64::from_le_bytes(bytes[26..34].try_into().ok()?)),
                entity: Ref64::from_u64(u64::from_le_bytes(bytes[34..42].try_into().ok()?)),
            },
            rights: u32::from_le_bytes(bytes[42..46].try_into().ok()?),
            object_version: u32::from_le_bytes(bytes[46..50].try_into().ok()?),
            valid_from_epoch: u32::from_le_bytes(bytes[50..54].try_into().ok()?),
            valid_until_epoch: u32::from_le_bytes(bytes[54..58].try_into().ok()?),
            nonce: u64::from_le_bytes(bytes[58..66].try_into().ok()?),
            signature: bytes[66..98].try_into().ok()?,
        })
    }

    fn signing_bytes(self) -> [u8; 66] {
        let mut out = [0u8; 66];
        out[0..2].copy_from_slice(&self.version.to_le_bytes());
        out[2..10].copy_from_slice(&self.issuer.0.to_le_bytes());
        out[10..18].copy_from_slice(&self.audience.0.to_le_bytes());
        out[18..26].copy_from_slice(&self.actor.to_u64().to_le_bytes());
        out[26..34].copy_from_slice(&self.target.node.0.to_le_bytes());
        out[34..42].copy_from_slice(&self.target.entity.to_u64().to_le_bytes());
        out[42..46].copy_from_slice(&self.rights.to_le_bytes());
        out[46..50].copy_from_slice(&self.object_version.to_le_bytes());
        out[50..54].copy_from_slice(&self.valid_from_epoch.to_le_bytes());
        out[54..58].copy_from_slice(&self.valid_until_epoch.to_le_bytes());
        out[58..66].copy_from_slice(&self.nonce.to_le_bytes());
        out
    }
}

fn hmac_sha256(key: &[u8; 32], message: &[u8]) -> [u8; 32] {
    let mut inner_key = [0x36u8; 64];
    let mut outer_key = [0x5cu8; 64];
    for (index, byte) in key.iter().enumerate() {
        inner_key[index] ^= byte;
        outer_key[index] ^= byte;
    }
    let mut inner = Sha256::new();
    inner.update(inner_key);
    inner.update(message);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_key);
    outer.update(inner);
    outer.finalize().into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}
