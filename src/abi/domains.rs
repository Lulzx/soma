//! Logical protection-domain ABI.

use super::{AbiHeader, Ref64};

/// A logical authority and allocation boundary. A zero process limit is
/// unbounded. Counts are monotonic because the reference interpreter retains
/// terminal process descriptors for traceability.
#[derive(Clone, Debug)]
pub struct DomainDescriptor {
    pub header: AbiHeader,
    pub id: Ref64,
    pub parent: Ref64,
    pub max_processes: u32,
    pub processes_created: u32,
}

impl DomainDescriptor {
    pub fn new(parent: Ref64, max_processes: u32) -> Self {
        Self {
            header: AbiHeader::new(1, std::mem::size_of::<Self>() as u32),
            id: Ref64::NULL,
            parent,
            max_processes,
            processes_created: 0,
        }
    }
}
