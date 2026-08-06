//! Physical batch-evaluation backend boundary.
//!
//! Backends operate on frozen bytes and never mutate kernel state directly.
//! The common publication path creates and freezes the output object, then
//! completes the semantic `BatchEvaluate` collective.

use std::collections::HashMap;

use crate::abi::{ObjectKind, Ref64};
use crate::compiler::body::EvaluatorProgram;
use crate::kernel::ownership::freeze;
use crate::kernel::payload::Payload;
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

/// One batch a backend has been asked to evaluate.
#[derive(Clone, Copy, Debug)]
pub struct BatchRequest<'a> {
    pub evaluator_id: u32,
    pub inputs: &'a [u8],
    pub element_count: u32,
    pub element_stride: u32,
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

    /// Evaluate, returning bytes the kernel can take ownership of wherever
    /// they already are.
    ///
    /// `evaluate` returns a `Vec`, which on Apple silicon means a backend that
    /// has just written its answer into memory the CPU can already read copies
    /// it into memory the CPU can also already read, so that it has the right
    /// type. This is the same operation without that copy: a backend holding
    /// its result in an allocation it is willing to give away hands the
    /// allocation over instead.
    ///
    /// The default is the copy, so a backend need not implement it, and I20
    /// still compares backends through `evaluate` — agreement is about bytes,
    /// not about where they live.
    fn evaluate_payload(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Payload, BackendError> {
        self.evaluate(evaluator_id, inputs, element_count, element_stride)
            .map(Payload::from)
    }

    /// Evaluate every request in `requests`, which an epoch offers together.
    ///
    /// The default runs them one at a time, which is what a backend with no
    /// notion of submission should do. It matters for the ones that have one:
    /// `examples/metal_overhead` prices sixty-four cohorts at 9897µs when each
    /// is committed and waited on separately against 757µs encoded into a
    /// single command buffer, because a round trip per cohort is a round trip
    /// the GPU spends idle.
    ///
    /// Either every request succeeds or the call fails. A partial epoch would
    /// leave the caller holding some published outputs and some unstarted
    /// collectives with no way to say which.
    fn evaluate_epoch(
        &mut self,
        requests: &[BatchRequest<'_>],
    ) -> Result<Vec<Payload>, BackendError> {
        requests
            .iter()
            .map(|request| {
                self.evaluate_payload(
                    request.evaluator_id,
                    request.inputs,
                    request.element_count,
                    request.element_stride,
                )
            })
            .collect()
    }
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
    outputs: Payload,
) -> Result<Ref64, BackendError> {
    let output = kernel.create_object_from_payload(actor, ObjectKind::FrozenArray, outputs);
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
    // Borrowed, not copied. This was a `to_vec` because the borrow of the
    // kernel has to end before `publish` takes it mutably again, and copying
    // the batch is the way to end a borrow without thinking about it. The
    // copy is a whole pass over the input — at a million 8-byte elements it
    // is 8MB, which `examples/backend_bench` measured as a third of the
    // published path's time against Metal. The borrow ends at the last use of
    // `input_bytes` instead, which is before `publish`.
    let input_bytes = kernel.object_bytes(actor, inputs)?;
    let (outputs, kind, spilled) = if count >= minimum_accelerator_batch {
        match accelerator.evaluate_payload(evaluator, input_bytes, count, stride) {
            Ok(outputs) => (outputs, accelerator.kind(), false),
            Err(BackendError::Unavailable) => (
                cpu.evaluate_payload(evaluator, input_bytes, count, stride)?,
                cpu.kind(),
                true,
            ),
            Err(error) => return Err(error),
        }
    } else {
        (
            cpu.evaluate_payload(evaluator, input_bytes, count, stride)?,
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

/// Execute every ready `BatchEvaluate` collective in one epoch, giving the
/// accelerator all of them at once.
///
/// `execute_with_spill` is this for a single collective, and running it in a
/// loop is what makes an epoch cost one GPU round trip per collective. Here
/// the requests are gathered first, handed to the backend together, and only
/// then published, so a backend that can submit them as one unit gets the
/// chance to.
///
/// Placement is still per-collective: a batch below `minimum_accelerator_batch`
/// goes to the CPU, and the two groups run separately. Spilling is all or
/// nothing for the accelerator group, because a backend that reports itself
/// unavailable partway through an epoch has not told us which requests it
/// completed.
///
/// Publication order follows `collectives`, not the order the backend
/// finished, so the trace does not depend on how the work was submitted.
pub fn execute_epoch_with_spill(
    kernel: &mut Kernel,
    actor: Ref64,
    collectives: &[Ref64],
    minimum_accelerator_batch: u32,
    accelerator: &mut dyn BatchBackend,
    cpu: &mut dyn BatchBackend,
    stats: &mut PlacementStats,
) -> Result<Vec<Ref64>, BackendError> {
    if collectives.is_empty() {
        return Ok(Vec::new());
    }

    let mut plans = Vec::with_capacity(collectives.len());
    for collective in collectives {
        let (evaluator, inputs, count, stride) = kernel.batch_evaluate_request(*collective)?;
        plans.push((*collective, evaluator, inputs, count, stride));
    }

    let inputs: Vec<Ref64> = plans.iter().map(|plan| plan.2).collect();
    let bytes = kernel.object_bytes_many(actor, &inputs)?;

    let mut accelerated = Vec::new();
    let mut on_cpu = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        let request = BatchRequest {
            evaluator_id: plan.1,
            inputs: bytes[index],
            element_count: plan.3,
            element_stride: plan.4,
        };
        if plan.3 >= minimum_accelerator_batch {
            accelerated.push((index, request));
        } else {
            on_cpu.push((index, request));
        }
    }

    let accelerator_requests: Vec<BatchRequest<'_>> =
        accelerated.iter().map(|(_, request)| *request).collect();
    let (accelerator_outputs, accelerator_kind, spilled) =
        match accelerator.evaluate_epoch(&accelerator_requests) {
            Ok(outputs) => (outputs, accelerator.kind(), false),
            Err(BackendError::Unavailable) => {
                (cpu.evaluate_epoch(&accelerator_requests)?, cpu.kind(), true)
            }
            Err(error) => return Err(error),
        };

    let cpu_requests: Vec<BatchRequest<'_>> = on_cpu.iter().map(|(_, request)| *request).collect();
    let cpu_outputs = cpu.evaluate_epoch(&cpu_requests)?;

    // Reassemble into the caller's order before anything is published, so the
    // trace records the epoch's collectives in the order it offered them.
    let mut ordered: Vec<Option<(Payload, BackendKind, bool)>> =
        (0..plans.len()).map(|_| None).collect();
    for ((index, _), payload) in accelerated.iter().zip(accelerator_outputs) {
        ordered[*index] = Some((payload, accelerator_kind, spilled));
    }
    for ((index, _), payload) in on_cpu.iter().zip(cpu_outputs) {
        ordered[*index] = Some((payload, cpu.kind(), true));
    }

    let mut published = Vec::with_capacity(plans.len());
    for (plan, slot) in plans.iter().zip(ordered) {
        let (payload, kind, spilled) = slot.ok_or(BackendError::ExecutionFailed)?;
        let required = (plan.3 as usize)
            .checked_mul(plan.4 as usize)
            .ok_or(BackendError::InvalidInput)?;
        if payload.len() < required {
            return Err(BackendError::InvalidInput);
        }
        published.push(publish(kernel, actor, plan.0, payload)?);
        stats.record(plan.1, kind, spilled);
    }
    Ok(published)
}
