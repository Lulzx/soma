//! Apple Metal implementation of the physical batch backend.
//!
//! This module is intentionally behind the `metal` feature. The semantic core
//! remains dependency-free, while macOS deployments can execute sufficiently
//! large batches on a real GPU without giving the backend direct kernel access.

use std::ffi::c_void;

use metal::{
    CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};

use super::batch::{BackendError, BackendKind, BatchBackend};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void soma_reference_batch(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint base = gid * stride;
    for (uint byte = 0; byte < stride; ++byte) {
        output[base + byte] = input[base + byte];
    }
    uint value = uint(input[base])
        | (uint(input[base + 1]) << 8)
        | (uint(input[base + 2]) << 16)
        | (uint(input[base + 3]) << 24);
    value = value * 2u + 1u;
    output[base] = uchar(value);
    output[base + 1] = uchar(value >> 8);
    output[base + 2] = uchar(value >> 16);
    output[base + 3] = uchar(value >> 24);
}
"#;

pub struct MetalBatchBackend {
    device: Device,
    pipeline: ComputePipelineState,
}

impl MetalBatchBackend {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        let library = device
            .new_library_with_source(SOURCE, &CompileOptions::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let function = library
            .get_function("soma_reference_batch", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self { device, pipeline })
    }
}

impl BatchBackend for MetalBatchBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Accelerator
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
        encoder.set_compute_pipeline_state(&self.pipeline);
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
                width: self
                    .pipeline
                    .thread_execution_width()
                    .min(element_count as u64),
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
