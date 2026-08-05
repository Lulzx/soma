//! Loaded module ABI.

use super::{AbiHeader, Ref64};

#[derive(Clone, Debug)]
pub struct ModuleDescriptor {
    pub header: AbiHeader,
    pub id: Ref64,
    pub name_hash: u64,
    pub evaluator_count: u32,
}

impl ModuleDescriptor {
    pub fn new(name_hash: u64, evaluator_count: u32) -> Self {
        Self {
            header: AbiHeader::new(10, std::mem::size_of::<Self>() as u32),
            id: Ref64::NULL,
            name_hash,
            evaluator_count,
        }
    }
}
