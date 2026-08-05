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
    name: String,
    evaluators: Vec<BatchEvaluator>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrError {
    EmptyName,
    EmptyModule,
    ZeroIdentifier,
    ZeroStride,
    DuplicateEvaluator,
    DuplicateResumePoint,
    DuplicateRunClass,
    UnknownEvaluator,
    ModuleMismatch,
    Syntax,
    InvalidAccess,
    Runtime(RuntimeError),
}

impl From<RuntimeError> for IrError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl Module {
    pub fn new(evaluators: Vec<BatchEvaluator>) -> Result<Self, IrError> {
        Self::named("anonymous", evaluators)
    }

    pub fn named(
        name: impl Into<String>,
        evaluators: Vec<BatchEvaluator>,
    ) -> Result<Self, IrError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(IrError::EmptyName);
        }
        if evaluators.is_empty() {
            return Err(IrError::EmptyModule);
        }
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
        Ok(Self { name, evaluators })
    }

    /// Parse the deliberately small textual surface:
    ///
    /// `module NAME`
    /// `evaluator ID NAME STRIDE ENTRY_ID ENTRY_CLASS ro|rw DONE_ID DONE_CLASS ro|rw`
    pub fn parse(source: &str) -> Result<Self, IrError> {
        let mut lines = source
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'));
        let header = lines.next().ok_or(IrError::Syntax)?;
        let header = header.split_whitespace().collect::<Vec<_>>();
        if header.len() != 2 || header[0] != "module" {
            return Err(IrError::Syntax);
        }
        let mut evaluators = Vec::new();
        for line in lines {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 10 || fields[0] != "evaluator" {
                return Err(IrError::Syntax);
            }
            let number = |index: usize| fields[index].parse::<u32>().map_err(|_| IrError::Syntax);
            let access = |index: usize| match fields[index] {
                "ro" => Ok(StateAccess::ReadOnly),
                "rw" => Ok(StateAccess::Mutable),
                _ => Err(IrError::InvalidAccess),
            };
            evaluators.push(BatchEvaluator {
                id: number(1)?,
                name: fields[2].to_string(),
                schema: FrozenArraySchema {
                    element_stride: number(3)?,
                },
                entry: ResumePoint {
                    id: number(4)?,
                    run_class: number(5)?,
                    state_access: access(6)?,
                },
                completion: ResumePoint {
                    id: number(7)?,
                    run_class: number(8)?,
                    state_access: access(9)?,
                },
            });
        }
        Self::named(header[1], evaluators)
    }

    pub fn name(&self) -> &str {
        &self.name
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

    pub fn load(&self, kernel: &mut Kernel, actor: Ref64) -> Result<Ref64, IrError> {
        let manifest = self.manifest();
        kernel
            .load_module(actor, &self.name, &manifest)
            .map_err(Into::into)
    }

    pub fn instantiate_loaded_batch(
        &self,
        kernel: &mut Kernel,
        actor: Ref64,
        module: Ref64,
        evaluator_id: u32,
        inputs: Ref64,
        element_count: u32,
    ) -> Result<(Ref64, Ref64), IrError> {
        if !self
            .evaluators
            .iter()
            .any(|evaluator| evaluator.id == evaluator_id)
        {
            return Err(IrError::UnknownEvaluator);
        }
        if !kernel.module_matches(module, &self.name, &self.manifest()) {
            return Err(IrError::ModuleMismatch);
        }
        kernel
            .create_batch_evaluate_in_module(actor, module, evaluator_id, inputs, element_count)
            .map_err(Into::into)
    }

    fn manifest(&self) -> Vec<(u32, u32)> {
        self.evaluators
            .iter()
            .map(|evaluator| (evaluator.id, evaluator.schema.element_stride))
            .collect()
    }
}
