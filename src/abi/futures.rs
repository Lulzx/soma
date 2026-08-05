//! Future ABI (§12).

use super::refs::Ref64;
use super::AbiHeader;

/// Future states (§12). Resolution is single-assignment.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FutureState {
    Pending = 1,
    Resolved = 2,
    Failed = 3,
    Cancelled = 4,
}

/// Future descriptor (§12).
#[derive(Clone, Debug)]
pub struct FutureDescriptor {
    pub header: AbiHeader,

    pub id: Ref64,
    pub owner_domain: Ref64,

    pub state: FutureState,
    pub waiter_count: u32,

    pub value: Ref64, // CapRef
    pub failure: Ref64,

    pub waiter_list: Ref64,
    pub resolved_epoch: u32,
    pub flags: u32,
}

impl FutureDescriptor {
    pub fn new() -> FutureDescriptor {
        FutureDescriptor {
            header: AbiHeader::new(7, std::mem::size_of::<FutureDescriptor>() as u32),
            id: Ref64::NULL,
            owner_domain: Ref64::NULL,
            state: FutureState::Pending,
            waiter_count: 0,
            value: Ref64::NULL,
            failure: Ref64::NULL,
            waiter_list: Ref64::NULL,
            resolved_epoch: 0,
            flags: 0,
        }
    }
}

impl Default for FutureDescriptor {
    fn default() -> Self {
        Self::new()
    }
}
