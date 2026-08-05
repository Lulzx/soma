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
    CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};

use super::batch::{BackendError, BackendKind, BatchBackend};
use crate::compiler::body::EvaluatorProgram;

pub struct MetalBatchBackend {
    device: Device,
    /// One compiled pipeline per installed evaluator, with the stride the body
    /// declared so a mismatched request is rejected rather than misread.
    pipelines: HashMap<u32, (ComputePipelineState, u32)>,
}

impl MetalBatchBackend {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        Ok(Self {
            device,
            pipelines: HashMap::new(),
        })
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
        let (pipeline, declared_stride) = self
            .pipelines
            .get(&evaluator_id)
            .ok_or(BackendError::UnsupportedEvaluator)?;
        if *declared_stride != element_stride {
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

        let input = self.device.new_buffer_with_data(
            inputs.as_ptr().cast::<c_void>(),
            required as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output = self
            .device
            .new_buffer(required as u64, MTLResourceOptions::StorageModeShared);
        let queue = self.device.new_command_queue();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
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
