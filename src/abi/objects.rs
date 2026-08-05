//! Object ABI (§6).

use super::refs::Ref64;
use super::AbiHeader;

/// Phase-1 object kinds (§6).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectKind {
    RawBytes = 1,
    ProcessState = 2,
    ContinuationFrame = 3,
    MessagePayload = 4,
    FrozenArray = 5,
    FutureValue = 6,
    TraceBuffer = 7,
}

/// Phase-1 ownership states. A frozen object cannot return to mutable state in
/// Phase 1; mutation requires allocating a new object (§6.2).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipState {
    UniqueMutable = 1,
    FrozenShared = 2,
    KernelOwned = 3,
}

/// Object descriptor (§6). `physical_mapping_token` is private to the kernel
/// and engine executives; user programs cannot inspect or construct it.
#[derive(Clone, Debug)]
pub struct ObjectDescriptor {
    pub header: AbiHeader,

    pub id: Ref64,
    pub owner_domain: Ref64,

    pub byte_length: u64,
    pub physical_mapping_token: u64,

    pub version: u32,
    pub object_kind: ObjectKind,
    pub ownership_state: OwnershipState,

    pub unique_owner: Ref64,
    pub reader_count: u32,
    pub flags: u32,
}

impl ObjectDescriptor {
    pub fn new(object_kind: ObjectKind, byte_length: u64) -> ObjectDescriptor {
        ObjectDescriptor {
            header: AbiHeader::new(3, std::mem::size_of::<ObjectDescriptor>() as u32),
            id: Ref64::NULL,
            owner_domain: Ref64::NULL,
            byte_length,
            physical_mapping_token: 0,
            version: 0,
            object_kind,
            ownership_state: OwnershipState::UniqueMutable,
            unique_owner: Ref64::NULL,
            reader_count: 0,
            flags: 0,
        }
    }
}
