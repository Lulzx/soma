//! Physical batch-evaluation backend boundary.
//!
//! Backends operate on frozen bytes and never mutate kernel state directly.
//! The common publication path creates and freezes the output object, then
//! completes the semantic `BatchEvaluate` collective.

use std::collections::HashMap;

use crate::abi::{ObjectKind, Ref64};
use crate::compiler::body::{Arrays, EvaluatorProgram};
use crate::kernel::ownership::freeze;
use crate::kernel::payload::Payload;
use crate::kernel::{Kernel, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Accelerator,
    Remote,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendError {
    Unavailable,
    UnsupportedEvaluator,
    InvalidInput,
    ExecutionFailed,
    AuthorityDenied,
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    Runtime(RuntimeError),
}

impl From<RuntimeError> for BackendError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

/// A second, read-only array bound alongside a batch's input array.
///
/// Zero `element_stride` means nothing is bound, which is every batch whose
/// body does not gather from a second array. It is a stride rather than an
/// `Option` because that is the form the value already travels in through the
/// collective descriptor and the Metal buffer bindings, and one spelling of
/// "not bound" is worth more than a tidier type at one of the three layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct AuxArray<'a> {
    pub bytes: &'a [u8],
    pub element_count: u32,
    pub element_stride: u32,
}

impl<'a> AuxArray<'a> {
    pub const NONE: AuxArray<'static> = AuxArray {
        bytes: &[],
        element_count: 0,
        element_stride: 0,
    };

    pub fn new(bytes: &'a [u8], element_count: u32, element_stride: u32) -> Self {
        Self {
            bytes,
            element_count,
            element_stride,
        }
    }

    pub fn is_bound(&self) -> bool {
        self.element_stride != 0
    }
}

/// One batch a backend has been asked to evaluate.
#[derive(Clone, Copy, Debug)]
pub struct BatchRequest<'a> {
    pub evaluator_id: u32,
    pub inputs: &'a [u8],
    pub aux: AuxArray<'a>,
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

    /// Evaluate with a second read-only array bound.
    ///
    /// The default declines when one is bound, and is `evaluate` when one is
    /// not. Declining is the important half: a backend that has not been taught
    /// the second binding must not quietly evaluate the body against the input
    /// array alone and return bytes. That answer is indistinguishable from a
    /// correct one to every other invariant in the machine, which is the exact
    /// failure I20 exists to catch and the reason `UnsupportedEvaluator` is a
    /// legal answer at all.
    fn evaluate_with_aux(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Vec<u8>, BackendError> {
        if aux.is_bound() {
            Err(BackendError::UnsupportedEvaluator)
        } else {
            self.evaluate(evaluator_id, inputs, element_count, element_stride)
        }
    }

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
        aux: AuxArray<'_>,
    ) -> Result<Payload, BackendError> {
        self.evaluate_with_aux(evaluator_id, inputs, element_count, element_stride, aux)
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
                    request.aux,
                )
            })
            .collect()
    }
}

/// Apply `element` to each element of `inputs` and return the result. Shared
/// by every backend's argument checking so that two backends cannot disagree
/// about what a malformed request is.
///
/// `element` receives an element's *index* rather than its bytes, because a
/// body that gathers needs the whole array and slicing one element out would
/// hide the rest of it. It also removes a per-element copy that existed only
/// to hand the interpreter an isolated slice.
pub fn evaluate_elementwise(
    inputs: &[u8],
    element_count: u32,
    element_stride: u32,
    mut element: impl FnMut(u32, &mut [u8]),
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
    for index in 0..element_count {
        let at = index as usize * stride;
        element(index, &mut outputs[at..at + stride]);
    }
    Ok(outputs)
}

/// The same, across `threads` OS threads.
///
/// This is the one place in the machine where parallelism needs no argument
/// beyond the rules the body language already has. `compiler::body` requires a
/// body to be pure, to read the frozen *input* array and never the output, and
/// to write only its own element. Those are not stated for the sake of
/// threading — they are what makes I19 true of a gathering body, since a body
/// that could observe another element's store would make the published result
/// depend on the schedule. Having paid for them, an element's output is a
/// function of the frozen input and its own index, so splitting the elements
/// across threads cannot change a byte.
///
/// The split is `chunks_mut`, so each thread owns a disjoint run of output
/// elements and shares an immutable borrow of the input. There is no
/// synchronisation inside the loop and none is needed; the only ordering is the
/// scope's join.
///
/// Chunking is by element rather than by byte. A chunk boundary inside an
/// element would hand two threads halves of one element's output, and the
/// interpreter writes an element as a unit.
pub fn evaluate_elementwise_threaded(
    inputs: &[u8],
    element_count: u32,
    element_stride: u32,
    threads: usize,
    element: impl Fn(u32, &mut [u8]) + Sync,
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
    let mut outputs = inputs[..required].to_vec();

    let threads = threads.max(1);
    if threads == 1 || element_count < 2 {
        for index in 0..element_count {
            let at = index as usize * stride;
            element(index, &mut outputs[at..at + stride]);
        }
        return Ok(outputs);
    }

    // Ceiling division, so `threads` chunks cover the array and the last is the
    // short one. A floor would leave a remainder chunk and one more thread than
    // asked for.
    let per_thread = (element_count as usize).div_ceil(threads);
    let element = &element;
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in outputs.chunks_mut(per_thread * stride).enumerate() {
            let first = (chunk_index * per_thread) as u32;
            scope.spawn(move || {
                for offset in 0..(chunk.len() / stride) {
                    let at = offset * stride;
                    element(first + offset as u32, &mut chunk[at..at + stride]);
                }
            });
        }
    });
    Ok(outputs)
}

/// Dependency-free scalar backend. It interprets whatever body it was given,
/// and under I20 it is the definition every other backend is checked against.
#[derive(Debug)]
pub struct CpuReferenceBackend {
    programs: HashMap<u32, EvaluatorProgram>,
    threads: usize,
}

impl Default for CpuReferenceBackend {
    fn default() -> Self {
        Self {
            programs: HashMap::new(),
            // One thread by default, so this backend stays what I20 calls the
            // definition: the simplest possible reading of a body, with no
            // decisions in it. A threaded run has to agree with *this*, and a
            // default that already threaded would leave nothing to agree with.
            threads: 1,
        }
    }
}

impl CpuReferenceBackend {
    pub fn with(programs: &[&EvaluatorProgram]) -> Self {
        let mut backend = Self::default();
        for program in programs {
            let _ = backend.install(program);
        }
        backend
    }

    /// Evaluate each batch across `threads` OS threads.
    ///
    /// Safe for any body the language admits, and not because the
    /// implementation is careful: `compiler::body` requires a body to read the
    /// frozen input and never the output and to write only its own element, so
    /// an element's result is a function of the input and its index. Those
    /// rules exist for I19, and threading is what they were already paying for.
    ///
    /// It is a knob rather than a default because I20 makes this backend the
    /// definition every other is checked against, and a definition should be
    /// the plainest reading of a body available.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    pub fn threads(&self) -> usize {
        self.threads
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
        self.evaluate_with_aux(
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            AuxArray::NONE,
        )
    }

    fn evaluate_with_aux(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Vec<u8>, BackendError> {
        self.evaluate_shared(
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            aux,
            self.threads,
        )
    }

    /// Run every request of an epoch, one thread per group of requests.
    ///
    /// The parallelism is across *requests* here rather than across the
    /// elements of one, and the two are alternatives rather than layers. An
    /// epoch of sixty-four small cohorts gets nothing from splitting each
    /// cohort's elements — there are not enough of them to fill a thread — and
    /// everything from running the cohorts side by side, which is the shape
    /// `examples/metal_overhead` prices. An epoch holding one large collective
    /// is the other way round. Nesting both would oversubscribe by the product
    /// of the two counts, so each request runs single-threaded here.
    ///
    /// Requests are independent by construction: each names its own frozen
    /// input and its own output, and a body writes only its own element. That
    /// is the same argument element threading makes, applied one level out.
    fn evaluate_epoch(
        &mut self,
        requests: &[BatchRequest<'_>],
    ) -> Result<Vec<Payload>, BackendError> {
        if self.threads <= 1 || requests.len() < 2 {
            return requests
                .iter()
                .map(|request| {
                    self.evaluate_shared(
                        request.evaluator_id,
                        request.inputs,
                        request.element_count,
                        request.element_stride,
                        request.aux,
                        self.threads,
                    )
                    .map(Payload::from)
                })
                .collect();
        }

        let per_thread = requests.len().div_ceil(self.threads);
        let backend = &*self;
        // Collected as bytes and wrapped afterwards, so nothing about
        // `Payload`'s foreign variant has to cross a thread boundary. This
        // backend only ever produces host bytes anyway.
        let grouped: Vec<Result<Vec<Vec<u8>>, BackendError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = requests
                .chunks(per_thread)
                .map(|group| {
                    scope.spawn(move || {
                        group
                            .iter()
                            .map(|request| {
                                backend.evaluate_shared(
                                    request.evaluator_id,
                                    request.inputs,
                                    request.element_count,
                                    request.element_stride,
                                    request.aux,
                                    1,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("an evaluator body cannot panic"))
                .collect()
        });

        // Either every request succeeds or the call fails, which is the
        // contract the sequential path already has: a partial epoch leaves the
        // caller holding some published outputs and some unstarted collectives
        // with no way to say which.
        let mut out = Vec::with_capacity(requests.len());
        for group in grouped {
            out.extend(group?.into_iter().map(Payload::from));
        }
        Ok(out)
    }
}

impl CpuReferenceBackend {
    /// The evaluation itself, over `&self` so several threads can run it.
    fn evaluate_shared(
        &self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
        threads: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let program = self
            .programs
            .get(&evaluator_id)
            .ok_or(BackendError::UnsupportedEvaluator)?;
        if program.stride() != element_stride {
            return Err(BackendError::InvalidInput);
        }
        // The binding is checked in both directions, at the boundary, so that
        // the interpreter never has to decide what to do about a body reading
        // an array it was not given. A body that gathers from an aux array and
        // was handed none is a malformed request; so is an aux array bound to a
        // body with no name for it, because the caller froze something for
        // nothing and would never find out.
        if program.binds_aux() != aux.is_bound() || program.aux_stride() != aux.element_stride {
            return Err(BackendError::InvalidInput);
        }
        if aux.is_bound() {
            let required = (aux.element_count as usize)
                .checked_mul(aux.element_stride as usize)
                .ok_or(BackendError::InvalidInput)?;
            if aux.bytes.len() < required {
                return Err(BackendError::InvalidInput);
            }
        }
        let arrays = Arrays::of(inputs, element_count).with_aux(aux.bytes, aux.element_count);
        evaluate_elementwise_threaded(
            inputs,
            element_count,
            element_stride,
            threads,
            |index, target| program.evaluate_bound(arrays, index, target),
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlacementStats {
    pub cpu_executions: u64,
    pub accelerator_executions: u64,
    pub remote_executions: u64,
    pub cpu_spills: u64,
    pub migrations: u64,
    last_backend: HashMap<u32, BackendKind>,
}

impl PlacementStats {
    fn record(&mut self, evaluator_id: u32, kind: BackendKind, spilled: bool) {
        match kind {
            BackendKind::Cpu => self.cpu_executions += 1,
            BackendKind::Accelerator => self.accelerator_executions += 1,
            BackendKind::Remote => self.remote_executions += 1,
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
    // The second array is bound to the collective rather than passed in, so
    // this is where it is read. Both arrays come out of one
    // `object_bytes_many` because `object_bytes` borrows the kernel mutably —
    // it records the authority decision — and two of those cannot overlap.
    // An unbound aux slot is NULL and resolves to empty bytes.
    let binding = kernel.batch_evaluate_aux(collective)?;
    // An unbound slot is not fetched rather than fetched as NULL: every
    // reference handed to `object_bytes_many` is authorized, and NULL is not
    // something an actor can hold READ on.
    let refs: Vec<Ref64> = if binding.is_bound() {
        vec![inputs, binding.inputs]
    } else {
        vec![inputs]
    };
    let fetched = kernel.object_bytes_many(actor, &refs)?;
    let input_bytes = fetched[0];
    let aux_bytes: &[u8] = if binding.is_bound() { fetched[1] } else { &[] };
    let aux = AuxArray::new(aux_bytes, binding.element_count, binding.element_stride);
    let (outputs, kind, spilled) = if count >= minimum_accelerator_batch {
        match accelerator.evaluate_payload(evaluator, input_bytes, count, stride, aux) {
            Ok(outputs) => (outputs, accelerator.kind(), false),
            Err(BackendError::Unavailable) => (
                cpu.evaluate_payload(evaluator, input_bytes, count, stride, aux)?,
                cpu.kind(),
                true,
            ),
            Err(error) => return Err(error),
        }
    } else {
        (
            cpu.evaluate_payload(evaluator, input_bytes, count, stride, aux)?,
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
        let aux = kernel.batch_evaluate_aux(*collective)?;
        plans.push((*collective, evaluator, inputs, count, stride, aux));
    }

    // One `object_bytes_many` over every array of every collective rather than
    // one call per array, because it exists to take a single borrow of the
    // kernel for the whole epoch's inputs. Bound aux arrays are appended after
    // the inputs and `aux_slot` records where each went; unbound collectives
    // contribute nothing, since every reference passed here is authorized and
    // NULL is not something an actor holds READ on.
    let mut refs: Vec<Ref64> = plans.iter().map(|plan| plan.2).collect();
    let mut aux_slot: Vec<Option<usize>> = Vec::with_capacity(plans.len());
    for plan in &plans {
        if plan.5.is_bound() {
            aux_slot.push(Some(refs.len()));
            refs.push(plan.5.inputs);
        } else {
            aux_slot.push(None);
        }
    }
    let bytes = kernel.object_bytes_many(actor, &refs)?;

    let mut accelerated = Vec::new();
    let mut on_cpu = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        let request = BatchRequest {
            evaluator_id: plan.1,
            inputs: bytes[index],
            aux: AuxArray::new(
                aux_slot[index].map(|slot| bytes[slot]).unwrap_or(&[]),
                plan.5.element_count,
                plan.5.element_stride,
            ),
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
