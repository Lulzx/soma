//! Minimal, hardware-neutral IR for naming batch evaluators.
//!
//! The IR deliberately contains no device, lane width, placement, or launch
//! concept. It identifies frozen-array shape and continuation resume points,
//! then instantiates the semantic collective.

use std::collections::HashSet;

use crate::abi::{Ref64, StateAccess};
use crate::kernel::{Kernel, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenArraySchema {
    pub element_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResumePoint {
    pub id: u32,
    pub run_class: u32,
    pub state_access: StateAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchEvaluator {
    pub id: u32,
    pub name: String,
    pub schema: FrozenArraySchema,
    pub entry: ResumePoint,
    pub completion: ResumePoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    evaluators: Vec<BatchEvaluator>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrError {
    EmptyName,
    ZeroIdentifier,
    ZeroStride,
    DuplicateEvaluator,
    DuplicateResumePoint,
    DuplicateRunClass,
    UnknownEvaluator,
    Runtime(RuntimeError),
}

impl From<RuntimeError> for IrError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl Module {
    pub fn new(evaluators: Vec<BatchEvaluator>) -> Result<Self, IrError> {
        let mut evaluator_ids = HashSet::new();
        let mut names = HashSet::new();
        let mut resume_points = HashSet::new();
        let mut run_classes = HashSet::new();
        for evaluator in &evaluators {
            if evaluator.name.trim().is_empty() {
                return Err(IrError::EmptyName);
            }
            if evaluator.id == 0
                || evaluator.entry.id == 0
                || evaluator.entry.run_class == 0
                || evaluator.completion.id == 0
                || evaluator.completion.run_class == 0
            {
                return Err(IrError::ZeroIdentifier);
            }
            if evaluator.schema.element_stride == 0 {
                return Err(IrError::ZeroStride);
            }
            if !evaluator_ids.insert(evaluator.id) || !names.insert(evaluator.name.clone()) {
                return Err(IrError::DuplicateEvaluator);
            }
            for resume in [evaluator.entry, evaluator.completion] {
                if !resume_points.insert(resume.id) {
                    return Err(IrError::DuplicateResumePoint);
                }
                if !run_classes.insert(resume.run_class) {
                    return Err(IrError::DuplicateRunClass);
                }
            }
        }
        Ok(Self { evaluators })
    }

    pub fn evaluators(&self) -> &[BatchEvaluator] {
        &self.evaluators
    }

    pub fn instantiate_batch(
        &self,
        kernel: &mut Kernel,
        actor: Ref64,
        evaluator_id: u32,
        inputs: Ref64,
        element_count: u32,
    ) -> Result<(Ref64, Ref64), IrError> {
        let evaluator = self
            .evaluators
            .iter()
            .find(|evaluator| evaluator.id == evaluator_id)
            .ok_or(IrError::UnknownEvaluator)?;
        kernel
            .create_batch_evaluate_for(
                actor,
                evaluator.id,
                inputs,
                element_count,
                evaluator.schema.element_stride,
            )
            .map_err(Into::into)
    }
}
