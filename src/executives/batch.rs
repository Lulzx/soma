//! Physical batch-evaluation backend boundary.
//!
//! Backends operate on frozen bytes and never mutate kernel state directly.
//! The common publication path creates and freezes the output object, then
//! completes the semantic `BatchEvaluate` collective.

use std::collections::HashMap;

use crate::abi::{ObjectKind, Ref64};
use crate::kernel::ownership::freeze;
use crate::kernel::{Kernel, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Accelerator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendError {
    Unavailable,
    UnsupportedEvaluator,
    InvalidInput,
    ExecutionFailed,
    Runtime(RuntimeError),
}

impl From<RuntimeError> for BackendError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

pub trait BatchBackend {
    fn kind(&self) -> BackendKind;

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError>;
}

/// Dependency-free scalar reference backend for the example evaluator:
/// transform the first little-endian u32 in each element as `2*x + 1` and
/// preserve any remaining bytes in the element.
#[derive(Debug, Default)]
pub struct CpuReferenceBackend;

impl BatchBackend for CpuReferenceBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn evaluate(
        &mut self,
        _evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        let stride = element_stride as usize;
        let required = (element_count as usize)
            .checked_mul(stride)
            .ok_or(BackendError::InvalidInput)?;
        if stride < 4 || inputs.len() < required {
            return Err(BackendError::InvalidInput);
        }
        let mut outputs = inputs[..required].to_vec();
        for element in outputs.chunks_exact_mut(stride) {
            let value = u32::from_le_bytes(
                element[..4]
                    .try_into()
                    .map_err(|_| BackendError::InvalidInput)?,
            );
            element[..4].copy_from_slice(&value.wrapping_mul(2).wrapping_add(1).to_le_bytes());
        }
        Ok(outputs)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlacementStats {
    pub cpu_executions: u64,
    pub accelerator_executions: u64,
    pub cpu_spills: u64,
    pub migrations: u64,
    last_backend: HashMap<u32, BackendKind>,
}

impl PlacementStats {
    fn record(&mut self, evaluator_id: u32, kind: BackendKind, spilled: bool) {
        match kind {
            BackendKind::Cpu => self.cpu_executions += 1,
            BackendKind::Accelerator => self.accelerator_executions += 1,
        }
        if spilled {
            self.cpu_spills += 1;
        }
        if self
            .last_backend
            .insert(evaluator_id, kind)
            .is_some_and(|previous| previous != kind)
        {
            self.migrations += 1;
        }
    }
}

fn publish(
    kernel: &mut Kernel,
    actor: Ref64,
    collective: Ref64,
    outputs: Vec<u8>,
) -> Result<Ref64, BackendError> {
    let output = kernel.create_object(actor, ObjectKind::FrozenArray, outputs);
    freeze(kernel, actor, output)?;
    kernel.complete_batch_evaluate(actor, collective, output)?;
    Ok(output)
}

pub fn execute_with_spill(
    kernel: &mut Kernel,
    actor: Ref64,
    collective: Ref64,
    minimum_accelerator_batch: u32,
    accelerator: &mut dyn BatchBackend,
    cpu: &mut dyn BatchBackend,
    stats: &mut PlacementStats,
) -> Result<Ref64, BackendError> {
    let (evaluator, inputs, count, stride) = kernel.batch_evaluate_request(collective)?;
    let input_bytes = kernel.object_bytes(actor, inputs)?.to_vec();
    let (outputs, kind, spilled) = if count >= minimum_accelerator_batch {
        match accelerator.evaluate(evaluator, &input_bytes, count, stride) {
            Ok(outputs) => (outputs, accelerator.kind(), false),
            Err(BackendError::Unavailable) => (
                cpu.evaluate(evaluator, &input_bytes, count, stride)?,
                cpu.kind(),
                true,
            ),
            Err(error) => return Err(error),
        }
    } else {
        (
            cpu.evaluate(evaluator, &input_bytes, count, stride)?,
            cpu.kind(),
            true,
        )
    };
    let required = (count as usize)
        .checked_mul(stride as usize)
        .ok_or(BackendError::InvalidInput)?;
    if outputs.len() < required {
        return Err(BackendError::InvalidInput);
    }
    let output = publish(kernel, actor, collective, outputs)?;
    stats.record(evaluator, kind, spilled);
    Ok(output)
}
