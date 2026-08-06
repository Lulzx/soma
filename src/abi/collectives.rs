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
    pub module: Ref64,
    pub evaluator_id: u32,
    pub collective_kind: CollectiveKind,
    pub state: CollectiveState,
    pub inputs: Ref64,
    pub outputs: Ref64,
    pub element_count: u32,
    pub element_stride: u32,
    /// A second, read-only array the evaluator body gathers from.
    ///
    /// `NULL` with a zero stride is a collective binding one array, which is
    /// every collective whose body has no `gatheraux`. The binding is part of
    /// the collective rather than of the call because it is what the capability
    /// escrow freezes: an array a body reads has to be frozen for as long as
    /// the collective can run, and there is nowhere else that knows both facts.
    pub aux_inputs: Ref64,
    pub aux_count: u32,
    pub aux_stride: u32,
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
            module: Ref64::NULL,
            evaluator_id,
            collective_kind: CollectiveKind::BatchEvaluate,
            state: CollectiveState::Pending,
            inputs,
            outputs: Ref64::NULL,
            element_count,
            element_stride,
            aux_inputs: Ref64::NULL,
            aux_count: 0,
            aux_stride: 0,
            completion_future,
        }
    }
}
