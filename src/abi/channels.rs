//! First-class bounded channel ABI.

use super::{AbiHeader, Ref64};

#[derive(Clone, Debug)]
pub struct ChannelDescriptor {
    pub header: AbiHeader,
    pub id: Ref64,
    pub capacity: u32,
    pub closed: u32,
}

impl ChannelDescriptor {
    pub fn new(capacity: u32) -> Self {
        Self {
            header: AbiHeader::new(10, std::mem::size_of::<Self>() as u32),
            id: Ref64::NULL,
            capacity,
            closed: 0,
        }
    }
}
