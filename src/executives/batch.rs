//! Physical batch-evaluation backend boundary.
//!
//! Backends operate on frozen bytes and never mutate kernel state directly.
//! The common publication path creates and freezes the output object, then
//! completes the semantic `BatchEvaluate` collective.

use std::collections::HashMap;

use crate::abi::{ObjectKind, Ref64};
use crate::compiler::body::EvaluatorProgram;
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

    /// Make `program` available to this backend under its evaluator id.
    ///
    /// A backend answers only for bodies it has been given. Before v0.3 the
    /// trait took an `evaluator_id` that every implementation ignored while
    /// hardcoding one function, so nothing could tell a correct backend from
    /// one returning arbitrary bytes. Installation is what makes
    /// `UnsupportedEvaluator` an honest answer rather than a guess.
    fn install(&mut self, program: &EvaluatorProgram) -> Result<(), BackendError>;

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError>;
}

/// Split `inputs` into elements, apply `element` to each, and return the
/// result. Shared by every backend's argument checking so that two backends
/// cannot disagree about what a malformed request is.
pub fn evaluate_elementwise(
    inputs: &[u8],
    element_count: u32,
    element_stride: u32,
    mut element: impl FnMut(&[u8], &mut [u8]),
) -> Result<Vec<u8>, BackendError> {
    let stride = element_stride as usize;
    if stride == 0 {
        return Err(BackendError::InvalidInput);
    }
    let required = (element_count as usize)
        .checked_mul(stride)
        .ok_or(BackendError::InvalidInput)?;
    if inputs.len() < required {
        return Err(BackendError::InvalidInput);
    }
    // The output starts as a copy of the input, so fields a body does not
    // store keep their incoming bytes.
    let mut outputs = inputs[..required].to_vec();
    for index in 0..element_count as usize {
        let range = index * stride..(index + 1) * stride;
        let source = inputs[range.clone()].to_vec();
        element(&source, &mut outputs[range]);
    }
    Ok(outputs)
}

/// Dependency-free scalar backend. It interprets whatever body it was given,
/// and under I20 it is the definition every other backend is checked against.
#[derive(Debug, Default)]
pub struct CpuReferenceBackend {
    programs: HashMap<u32, EvaluatorProgram>,
}

impl CpuReferenceBackend {
    pub fn with(programs: &[&EvaluatorProgram]) -> Self {
        let mut backend = Self::default();
        for program in programs {
            let _ = backend.install(program);
        }
        backend
    }
}

impl BatchBackend for CpuReferenceBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn install(&mut self, program: &EvaluatorProgram) -> Result<(), BackendError> {
        self.programs.insert(program.id(), program.clone());
        Ok(())
    }

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        let program = self
            .programs
            .get(&evaluator_id)
            .ok_or(BackendError::UnsupportedEvaluator)?;
        if program.stride() != element_stride {
            return Err(BackendError::InvalidInput);
        }
        evaluate_elementwise(inputs, element_count, element_stride, |source, target| {
            program.evaluate_element(source, target)
        })
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

/// One way two backends disagreed about a body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgreementViolation {
    pub evaluator: u32,
    pub detail: String,
}

impl std::fmt::Display for AgreementViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "evaluator {}: {}", self.evaluator, self.detail)
    }
}

/// **I20. Backend agreement.**
///
/// For a given evaluator and frozen input, every backend claiming to realize
/// that evaluator produces identical output bytes. A backend that cannot
/// realize a body must return `UnsupportedEvaluator` rather than an
/// approximation — an approximation is indistinguishable from a correct answer
/// to every other invariant in the machine, which is exactly why this clause
/// has to exist separately.
///
/// The first backend in `backends` is the definition; the rest are checked
/// against it. Ordering matters and the CPU interpreter should come first,
/// because it is the one whose behaviour the body language specifies.
pub fn check_agreement(
    program: &EvaluatorProgram,
    inputs: &[u8],
    element_count: u32,
    backends: &mut [&mut dyn BatchBackend],
) -> Vec<AgreementViolation> {
    let mut out = Vec::new();
    let stride = program.stride();
    let Some((first, rest)) = backends.split_first_mut() else {
        return out;
    };
    let expected = match first.evaluate(program.id(), inputs, element_count, stride) {
        Ok(bytes) => bytes,
        Err(error) => {
            out.push(AgreementViolation {
                evaluator: program.id(),
                detail: format!("the defining backend could not evaluate it: {error:?}"),
            });
            return out;
        }
    };
    for backend in rest.iter_mut() {
        match backend.evaluate(program.id(), inputs, element_count, stride) {
            Ok(actual) if actual == expected => {}
            Ok(actual) => {
                let position = actual
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| a != b)
                    .unwrap_or(expected.len().min(actual.len()));
                out.push(AgreementViolation {
                    evaluator: program.id(),
                    detail: format!(
                        "{:?} backend differs from the definition at byte {}",
                        backend.kind(),
                        position
                    ),
                });
            }
            // Declining is allowed. Answering wrongly is not.
            Err(BackendError::UnsupportedEvaluator) => {}
            Err(error) => out.push(AgreementViolation {
                evaluator: program.id(),
                detail: format!("{:?} backend failed: {error:?}", backend.kind()),
            }),
        }
    }
    out
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
