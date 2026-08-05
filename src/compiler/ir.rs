//! Minimal, hardware-neutral IR for naming batch evaluators.
//!
//! The IR deliberately contains no device, lane width, placement, or launch
//! concept. It identifies frozen-array shape and continuation resume points,
//! then instantiates the semantic collective.

use std::collections::HashSet;

use crate::abi::{Ref64, StateAccess};
use crate::compiler::body::{BodyError, EvaluatorProgram};
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
    /// What this evaluator computes, when the module declares it.
    ///
    /// `None` names an evaluator without describing it, which is all this IR
    /// could do before v0.3. A module in that form can still be loaded and
    /// linked — I17 covers identity and stride either way — but no backend can
    /// realize it, so `execute_with_spill` reports `UnsupportedEvaluator`
    /// rather than applying whatever it happened to have compiled in.
    pub body: Option<EvaluatorProgram>,
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
    Body(BodyError),
}

impl From<RuntimeError> for IrError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<BodyError> for IrError {
    fn from(value: BodyError) -> Self {
        Self::Body(value)
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
            // A body derives its own stride from its declared element layout.
            // If that disagrees with the stride the evaluator declares, one of
            // the two is wrong and the module must not load — a backend
            // striding differently from the collective would read across
            // element boundaries and produce plausible garbage.
            if let Some(body) = &evaluator.body {
                if body.stride() != evaluator.schema.element_stride {
                    return Err(IrError::Body(BodyError::StrideMismatch));
                }
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
    /// ```text
    /// module NAME
    /// evaluator ID NAME STRIDE ENTRY_ID ENTRY_CLASS ro|rw DONE_ID DONE_CLASS ro|rw
    ///   field u32
    ///   op 0 load 0
    ///   op 1 const 2
    ///   op 2 mul 0 1
    ///   store 0 2
    /// ```
    ///
    /// The `field`/`op`/`store` lines following an `evaluator` line are its
    /// body and are optional. Omitting them keeps the pre-v0.3 form, which
    /// names an evaluator without saying what it computes.
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

        // Group each `evaluator` line with the body lines that follow it.
        let mut groups: Vec<(&str, Vec<&str>)> = Vec::new();
        for line in lines {
            if line.starts_with("evaluator ") {
                groups.push((line, Vec::new()));
            } else {
                match groups.last_mut() {
                    Some((_, body)) => body.push(line),
                    // A body line before any evaluator has nothing to attach to.
                    None => return Err(IrError::Syntax),
                }
            }
        }

        let mut evaluators = Vec::new();
        for (declaration, body_lines) in groups {
            let fields = declaration.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 10 {
                return Err(IrError::Syntax);
            }
            let number = |index: usize| fields[index].parse::<u32>().map_err(|_| IrError::Syntax);
            let access = |index: usize| match fields[index] {
                "ro" => Ok(StateAccess::ReadOnly),
                "rw" => Ok(StateAccess::Mutable),
                _ => Err(IrError::InvalidAccess),
            };
            let id = number(1)?;
            let name = fields[2].to_string();
            let body = if body_lines.is_empty() {
                None
            } else {
                Some(EvaluatorProgram::parse_lines(id, &name, &body_lines)?)
            };
            evaluators.push(BatchEvaluator {
                id,
                name,
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
                body,
            });
        }
        Self::named(header[1], evaluators)
    }

    /// Every declared body in this module, in evaluator order.
    pub fn programs(&self) -> Vec<&EvaluatorProgram> {
        self.evaluators
            .iter()
            .filter_map(|evaluator| evaluator.body.as_ref())
            .collect()
    }

    pub fn program(&self, evaluator_id: u32) -> Option<&EvaluatorProgram> {
        self.evaluators
            .iter()
            .find(|evaluator| evaluator.id == evaluator_id)
            .and_then(|evaluator| evaluator.body.as_ref())
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
