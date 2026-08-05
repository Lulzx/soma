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

use super::batch::{BackendError, BackendKind, BatchBackend};
use crate::compiler::body::EvaluatorProgram;

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
    pipelines: HashMap<u32, (ComputePipelineState, u32)>,
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
}

impl MetalBatchBackend {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines: HashMap::new(),
            scratch: None,
        })
    }

    /// Input and output buffers of at least `bytes`, allocating only when the
    /// batch is larger than any seen so far.
    fn scratch_for(&mut self, bytes: u64) -> (Buffer, Buffer) {
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

    /// Compile every body in one pass, so a codegen defect surfaces at
    /// installation rather than in the middle of a collective.
    pub fn with(programs: &[&EvaluatorProgram]) -> Result<Self, BackendError> {
        let mut backend = Self::new()?;
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
        let library = self
            .device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        let function = library
            .get_function(&program.metal_entry_point(), None)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        self.pipelines
            .insert(program.id(), (pipeline, program.stride()));
        Ok(())
    }

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        // Cloned rather than borrowed: `scratch_for` needs `&mut self`, and a
        // `ComputePipelineState` clone is a retain on an object the backend
        // already owns.
        let (pipeline, declared_stride) = self
            .pipelines
            .get(&evaluator_id)
            .cloned()
            .ok_or(BackendError::UnsupportedEvaluator)?;
        if declared_stride != element_stride {
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
            return Ok(Vec::new());
        }

        // Safe to overwrite in place: this backend waits for completion below,
        // so no previously submitted command buffer can still be reading it.
        // That stops being true the moment submission becomes asynchronous,
        // which is why a ring rather than one slot is the next shape.
        let (input, output) = self.scratch_for(required as u64);
        unsafe {
            std::ptr::copy_nonoverlapping(inputs.as_ptr(), input.contents().cast::<u8>(), required);
        }

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
        encoder.dispatch_threads(
            MTLSize {
                width: element_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: pipeline.thread_execution_width().min(element_count as u64),
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

        let bytes = unsafe { std::slice::from_raw_parts(output.contents().cast::<u8>(), required) };
        Ok(bytes.to_vec())
    }
}
