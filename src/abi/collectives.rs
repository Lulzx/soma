//! Collective-operation ABI. Phase 1 exposes one `BatchEvaluate` lifecycle.

use super::{AbiHeader, Ref64};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveKind {
    BatchEvaluate = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveState {
    Pending = 1,
    Completed = 2,
    Failed = 3,
    Cancelled = 4,
}

#[derive(Clone, Debug)]
pub struct CollectiveDescriptor {
    pub header: AbiHeader,
    pub id: Ref64,
    pub owner_process: Ref64,
    pub evaluator_id: u32,
    pub collective_kind: CollectiveKind,
    pub state: CollectiveState,
    pub inputs: Ref64,
    pub outputs: Ref64,
    pub element_count: u32,
    pub element_stride: u32,
    pub completion_future: Ref64,
}

impl CollectiveDescriptor {
    pub fn batch_evaluate(
        owner_process: Ref64,
        evaluator_id: u32,
        inputs: Ref64,
        element_count: u32,
        element_stride: u32,
        completion_future: Ref64,
    ) -> Self {
        Self {
            header: AbiHeader::new(23, std::mem::size_of::<Self>() as u32),
            id: Ref64::NULL,
            owner_process,
            evaluator_id,
            collective_kind: CollectiveKind::BatchEvaluate,
            state: CollectiveState::Pending,
            inputs,
            outputs: Ref64::NULL,
            element_count,
            element_stride,
            completion_future,
        }
    }
}
