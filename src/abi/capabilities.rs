//! Capability ABI (§5).

use super::refs::Ref64;
use super::AbiHeader;

/// Phase-1 rights (§5).
#[allow(non_snake_case)]
pub mod Rights {
    pub const READ: u32 = 1 << 0;
    pub const WRITE: u32 = 1 << 1;
    pub const FREEZE: u32 = 1 << 2;
    pub const TRANSFER: u32 = 1 << 3;
    pub const SEND: u32 = 1 << 4;
    pub const RECEIVE: u32 = 1 << 5;
    pub const RESOLVE: u32 = 1 << 6;
    pub const AWAIT: u32 = 1 << 7;
    pub const DESTROY: u32 = 1 << 8;

    /// All Phase-1 rights.
    pub const ALL: u32 = READ | WRITE | FREEZE | TRANSFER | SEND | RECEIVE | RESOLVE | AWAIT | DESTROY;
}

/// Phase-1 ownership modes (§5).
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipMode {
    UniqueMutable = 1,
    FrozenShared = 2,
    KernelOwned = 3,
}

/// Capability entry (§5). Application code receives a `CapRef` which resolves
/// through the calling domain's capability table. Capability derivation can
/// only reduce authority — it can never amplify it.
#[derive(Clone, Debug)]
pub struct CapabilityEntry {
    pub header: AbiHeader,

    pub object: Ref64,

    pub offset: u64,
    pub length: u64,

    pub rights: u32,
    pub ownership_mode: OwnershipMode,
    pub transfer_policy: u16,

    pub object_version: u32,
    pub valid_until_epoch: u32,

    pub parent_capability: Ref64,
}

impl CapabilityEntry {
    pub fn new(object: Ref64, rights: u32) -> CapabilityEntry {
        CapabilityEntry {
            header: AbiHeader::new(4, std::mem::size_of::<CapabilityEntry>() as u32),
            object,
            offset: 0,
            length: 0,
            rights,
            ownership_mode: OwnershipMode::UniqueMutable,
            transfer_policy: 0,
            object_version: 0,
            valid_until_epoch: u32::MAX,
            parent_capability: Ref64::NULL,
        }
    }
}
