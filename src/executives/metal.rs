//! Apple Metal implementation of the physical batch backend.
//!
//! This module is intentionally behind the `metal` feature. The semantic core
//! remains dependency-free, while macOS deployments can execute sufficiently
//! large batches on a real GPU without giving the backend direct kernel access.
//!
//! The shader is generated from the evaluator body by
//! `compiler::body::EvaluatorProgram::metal_source`, not written here. That is
//! the whole point of the body language: this file used to contain a
//! hand-written kernel computing `2*x + 1`, which agreed with a hand-written
//! CPU function computing `2*x + 1`, and the agreement was evidence about
//! nothing. Now both sides are lowered from one source and I20 checks them.

use std::collections::HashMap;
use std::ffi::c_void;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::batch::{AuxArray, BackendError, BackendKind, BatchBackend, BatchRequest};
use crate::compiler::body::EvaluatorProgram;
use crate::kernel::payload::{ForeignPayload, Payload};

/// Non-semantic Metal choices exposed to configuration search.
///
/// `None` uses the pipeline's SIMD width. An explicit threadgroup width is
/// clamped to both the pipeline maximum and the number of elements, so every
/// positive value is executable and changes placement only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalTuning {
    pub threadgroup_width: Option<u64>,
    pub reuse_scratch_buffers: bool,
}

impl Default for MetalTuning {
    fn default() -> Self {
        Self {
            threadgroup_width: None,
            reuse_scratch_buffers: true,
        }
    }
}

/// An object whose bytes are an `MTLBuffer` the GPU wrote and the backend has
/// given away.
///
/// The buffer is shared storage, so the CPU can read these bytes without any
/// transfer; that is the entire point. `len` is the batch, which can be less
/// than the buffer's capacity, and the slice is clipped to it so a published
/// object never exposes whatever the allocation happens to be carrying past
/// the end of the batch.
/// Where a batch's output should be written: into the buffer the backend
/// reuses, or into one allocated for this batch and given away.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    Reused,
    Owned,
}

struct MetalPayload {
    buffer: Buffer,
    len: usize,
}

// Safety: the buffer is exclusively owned — `MetalBatchBackend` allocates it
// for one batch, hands it over, and never writes to it again. Metal objects
// are internally reference counted, and no thread mutates this one after the
// command buffer that wrote it has completed, which `evaluate_payload` waits
// for before constructing this.
unsafe impl Send for MetalPayload {}

impl ForeignPayload for MetalPayload {
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u8>(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents().cast::<u8>(), self.len) }
    }

    fn provenance(&self) -> &'static str {
        "metal-shared"
    }
}

pub struct MetalBatchBackend {
    device: Device,
    /// Created once. `evaluate` used to create a command queue per call, which
    /// `examples/metal_overhead` measures at 68µs at best and past 400µs when
    /// queues are created in a loop, against a total call cost of ~330µs. It
    /// was the single largest fixed cost in the backend and it bought nothing:
    /// a queue is a submission channel, not per-batch state.
    queue: CommandQueue,
    /// One compiled pipeline per installed evaluator, with the stride the body
    /// declared so a mismatched request is rejected rather than misread.
    /// Pipeline, element stride, and aux element stride (zero when the body
    /// binds one array).
    pipelines: HashMap<u32, (ComputePipelineState, u32, u32)>,
    /// Grown to the largest batch seen and then reused.
    ///
    /// This looks like a fixed cost and is not one. `new_buffer` itself is
    /// about 1.4µs at any size, but a fresh shared buffer is fresh pages, and
    /// the fill that follows faults every one of them:
    /// `examples/metal_overhead` measures allocate-and-fill against memcpy
    /// into a warm buffer at 59µs vs 13µs for 1MB and 2016µs vs 503µs for
    /// 32MB. Reusing the allocation is therefore a per-byte saving of roughly
    /// three quarters, not a per-call saving, and it moves the fitted
    /// per-element cost rather than the intercept.
    scratch: Option<(Buffer, Buffer, u64)>,
    tuning: MetalTuning,
}

impl MetalBatchBackend {
    pub fn new() -> Result<Self, BackendError> {
        Self::new_with_tuning(MetalTuning::default())
    }

    pub fn new_with_tuning(tuning: MetalTuning) -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines: HashMap::new(),
            scratch: None,
            tuning,
        })
    }

    pub fn tuning(&self) -> MetalTuning {
        self.tuning
    }

    /// Change dispatch policy without recompiling the installed evaluators.
    ///
    /// Configuration searches use one backend so every candidate sees the
    /// same compiled pipelines. Scratch allocations deliberately survive a
    /// switch: reuse-on candidates measure the steady-state warm-buffer path,
    /// while reuse-off candidates ignore the cache and allocate afresh.
    pub fn set_tuning(&mut self, tuning: MetalTuning) {
        self.tuning = tuning;
    }

    /// Input and output buffers of at least `bytes`, allocating only when the
    /// batch is larger than any seen so far.
    fn scratch_for(&mut self, bytes: u64) -> (Buffer, Buffer) {
        if !self.tuning.reuse_scratch_buffers {
            return (
                self.device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                self.device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            );
        }
        let big_enough = matches!(&self.scratch, Some((_, _, capacity)) if *capacity >= bytes);
        if !big_enough {
            let input = self
                .device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            let output = self
                .device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            self.scratch = Some((input, output, bytes));
        }
        let (input, output, _) = self.scratch.as_ref().expect("just populated");
        (input.clone(), output.clone())
    }

    fn threadgroup_width(&self, pipeline: &ComputePipelineState, elements: u32) -> u64 {
        let requested = self
            .tuning
            .threadgroup_width
            .unwrap_or_else(|| pipeline.thread_execution_width())
            .max(1);
        requested
            .min(pipeline.max_total_threads_per_threadgroup())
            .min(u64::from(elements).max(1))
    }

    /// Run one batch and return the buffer holding its output, or `None` when
    /// the batch is empty.
    ///
    /// `Output::Reused` writes into the scratch buffer the backend keeps, for
    /// callers that are about to copy the bytes out anyway. `Output::Owned`
    /// allocates an output buffer for this batch alone, because a caller that
    /// publishes the buffer as a SOMA object takes ownership of it: the
    /// backend must not be able to overwrite a frozen object on its next
    /// batch. The input side is reused either way, since nothing outlives the
    /// call.
    fn run(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
        output_kind: Output,
    ) -> Result<Option<(Buffer, usize)>, BackendError> {
        // Cloned rather than borrowed: `scratch_for` needs `&mut self`, and a
        // `ComputePipelineState` clone is a retain on an object the backend
        // already owns.
        let (pipeline, declared_stride, declared_aux_stride) = self
            .pipelines
            .get(&evaluator_id)
            .cloned()
            .ok_or(BackendError::UnsupportedEvaluator)?;
        if declared_stride != element_stride {
            return Err(BackendError::InvalidInput);
        }
        // The same both-directions check the reference backend makes, and it
        // has to be made here too rather than trusted: a kernel compiled with
        // the aux parameters reads whatever is bound at buffer 4, so a body
        // expecting a second array and dispatched without one would read
        // uninitialised device memory and return plausible bytes.
        if declared_aux_stride != aux.element_stride {
            return Err(BackendError::InvalidInput);
        }
        let aux_required = (aux.element_count as usize)
            .checked_mul(aux.element_stride as usize)
            .ok_or(BackendError::InvalidInput)?;
        if aux.bytes.len() < aux_required {
            return Err(BackendError::InvalidInput);
        }

        let stride = element_stride as usize;
        let required = (element_count as usize)
            .checked_mul(stride)
            .ok_or(BackendError::InvalidInput)?;
        if inputs.len() < required {
            return Err(BackendError::InvalidInput);
        }
        if required == 0 {
            return Ok(None);
        }

        // Safe to overwrite in place: this backend waits for completion below,
        // so no previously submitted command buffer can still be reading it.
        // That stops being true the moment submission becomes asynchronous,
        // which is why a ring rather than one slot is the next shape.
        let (input, scratch_output) = self.scratch_for(required as u64);
        unsafe {
            std::ptr::copy_nonoverlapping(inputs.as_ptr(), input.contents().cast::<u8>(), required);
        }
        let output = match output_kind {
            Output::Reused => scratch_output,
            Output::Owned => self
                .device
                .new_buffer(required as u64, MTLResourceOptions::StorageModeShared),
        };

        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&element_count as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&element_stride as *const u32).cast::<c_void>(),
        );
        // Buffers 4-6 exist only in kernels generated for a body that reads a
        // second array; `metal_source` omits the parameters otherwise, so
        // binding them unconditionally would be binding to a slot the shader
        // does not declare.
        // Bound to a name so the allocation outlives the dispatch below. The
        // encoder holds a reference the driver honours, but the Rust binding is
        // what stops the buffer being released before `wait_until_completed`.
        let _aux_buffer = if aux_required > 0 {
            let buffer = self.device.new_buffer_with_data(
                aux.bytes.as_ptr().cast::<c_void>(),
                aux_required as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(4, Some(&buffer), 0);
            encoder.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                (&aux.element_count as *const u32).cast::<c_void>(),
            );
            encoder.set_bytes(
                6,
                std::mem::size_of::<u32>() as u64,
                (&aux.element_stride as *const u32).cast::<c_void>(),
            );
            Some(buffer)
        } else {
            None
        };
        encoder.dispatch_threads(
            MTLSize {
                width: element_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.threadgroup_width(&pipeline, element_count),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }
        Ok(Some((output, required)))
    }

    /// Compile every body in one pass, so a codegen defect surfaces at
    /// installation rather than in the middle of a collective.
    pub fn with(programs: &[&EvaluatorProgram]) -> Result<Self, BackendError> {
        Self::with_tuning(programs, MetalTuning::default())
    }

    pub fn with_tuning(
        programs: &[&EvaluatorProgram],
        tuning: MetalTuning,
    ) -> Result<Self, BackendError> {
        let mut backend = Self::new_with_tuning(tuning)?;
        for program in programs {
            backend.install(program)?;
        }
        Ok(backend)
    }
}

impl BatchBackend for MetalBatchBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Accelerator
    }

    fn install(&mut self, program: &EvaluatorProgram) -> Result<(), BackendError> {
        let source = program.metal_source();
        // I20 requires strict operation boundaries. In particular, f32
        // add/mul must not contract or reassociate into a different result.
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = self
            .device
            .new_library_with_source(&source, &options)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        let function = library
            .get_function(&program.metal_entry_point(), None)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        self.pipelines.insert(
            program.id(),
            (pipeline, program.stride(), program.aux_stride()),
        );
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
        let Some((output, required)) = self.run(
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            aux,
            Output::Reused,
        )?
        else {
            return Ok(Vec::new());
        };
        let bytes = unsafe { std::slice::from_raw_parts(output.contents().cast::<u8>(), required) };
        Ok(bytes.to_vec())
    }

    fn evaluate_payload(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Payload, BackendError> {
        let Some((buffer, len)) = self.run(
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            aux,
            Output::Owned,
        )?
        else {
            return Ok(Payload::Host(Vec::new()));
        };
        Ok(Payload::Foreign(Box::new(MetalPayload { buffer, len })))
    }

    /// Encode every request in the epoch into one command buffer.
    ///
    /// The default implementation would commit and wait once per request, and
    /// the wait is the expensive half: the GPU finishes a cohort, the CPU
    /// wakes, encodes the next, and the GPU idles through all of it. Sixty-four
    /// 8192-element cohorts cost 9897µs that way and 757µs encoded together
    /// (`examples/metal_overhead`).
    ///
    /// Inputs are staged into one reused buffer at aligned offsets, so the
    /// epoch costs one copy of its total bytes rather than one allocation per
    /// request. Outputs are separate owned allocations because each becomes a
    /// published object, and objects cannot share a buffer the backend would
    /// later reuse.
    fn evaluate_epoch(
        &mut self,
        requests: &[BatchRequest<'_>],
    ) -> Result<Vec<Payload>, BackendError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Validate everything before touching the GPU, so a bad request in an
        // epoch is rejected rather than half-executed.
        let mut plans = Vec::with_capacity(requests.len());
        let mut staged = 0usize;
        for request in requests {
            let (pipeline, declared_stride, declared_aux_stride) = self
                .pipelines
                .get(&request.evaluator_id)
                .cloned()
                .ok_or(BackendError::UnsupportedEvaluator)?;
            if declared_stride != request.element_stride
                || declared_aux_stride != request.aux.element_stride
            {
                return Err(BackendError::InvalidInput);
            }
            let aux_required = (request.aux.element_count as usize)
                .checked_mul(request.aux.element_stride as usize)
                .ok_or(BackendError::InvalidInput)?;
            if request.aux.bytes.len() < aux_required {
                return Err(BackendError::InvalidInput);
            }
            let required = (request.element_count as usize)
                .checked_mul(request.element_stride as usize)
                .ok_or(BackendError::InvalidInput)?;
            if request.inputs.len() < required {
                return Err(BackendError::InvalidInput);
            }
            let offset = staged;
            staged = staged
                .checked_add(align_up(required))
                .ok_or(BackendError::InvalidInput)?;
            plans.push((pipeline, required, offset, aux_required));
        }
        if staged == 0 {
            return Ok(requests.iter().map(|_| Payload::Host(Vec::new())).collect());
        }

        let (input, _) = self.scratch_for(staged as u64);
        for (request, (_, required, offset, _)) in requests.iter().zip(&plans) {
            if *required == 0 {
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    request.inputs.as_ptr(),
                    input.contents().cast::<u8>().add(*offset),
                    *required,
                );
            }
        }

        let outputs: Vec<Buffer> = plans
            .iter()
            .map(|(_, required, _, _)| {
                self.device.new_buffer(
                    (*required).max(1) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        // An epoch's aux arrays are separate buffers rather than staged into
        // the shared input allocation. They are read-only and shared between
        // requests in the common case -- every ant cohort of one epoch senses
        // the same grid -- so staging them would copy one grid per cohort to
        // gain an offset nothing needs. The vector holds them alive until the
        // command buffer completes.
        let mut aux_buffers: Vec<Option<Buffer>> = Vec::with_capacity(plans.len());
        for (request, (_, _, _, aux_required)) in requests.iter().zip(&plans) {
            aux_buffers.push(if *aux_required > 0 {
                Some(self.device.new_buffer_with_data(
                    request.aux.bytes.as_ptr().cast::<c_void>(),
                    *aux_required as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            } else {
                None
            });
        }

        for (((request, (pipeline, required, offset, _)), output), aux_buffer) in
            requests.iter().zip(&plans).zip(&outputs).zip(&aux_buffers)
        {
            if *required == 0 {
                continue;
            }
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&input), *offset as u64);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                (&request.element_count as *const u32).cast::<c_void>(),
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                (&request.element_stride as *const u32).cast::<c_void>(),
            );
            if let Some(buffer) = aux_buffer {
                encoder.set_buffer(4, Some(buffer), 0);
                encoder.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    (&request.aux.element_count as *const u32).cast::<c_void>(),
                );
                encoder.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    (&request.aux.element_stride as *const u32).cast::<c_void>(),
                );
            }
            encoder.dispatch_threads(
                MTLSize {
                    width: request.element_count as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.threadgroup_width(pipeline, request.element_count),
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }

        Ok(outputs
            .into_iter()
            .zip(&plans)
            .map(|(buffer, (_, required, _, _))| {
                if *required == 0 {
                    Payload::Host(Vec::new())
                } else {
                    Payload::Foreign(Box::new(MetalPayload {
                        buffer,
                        len: *required,
                    }))
                }
            })
            .collect())
    }
}

/// Round a staging offset up to a buffer-binding boundary. Metal requires a
/// bound offset to be aligned, and 256 is the conservative choice across the
/// families this may run on.
fn align_up(bytes: usize) -> usize {
    const ALIGNMENT: usize = 256;
    bytes.div_ceil(ALIGNMENT) * ALIGNMENT
}
