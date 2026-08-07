//! Concurrent Metal implementation of the device scheduling ABI.
//!
//! Candidate and placement buffers grow when necessary and otherwise remain
//! resident across epochs. Admission and placement are separate GPU dispatches
//! in one command buffer, so the device reaches the scheduling decision without
//! an intervening host read or round trip.

use std::ffi::c_void;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use crate::abi::cohorts::PartialCohortPolicy;
use crate::scheduler::admission::Candidate;
use crate::scheduler::device::{DeviceCandidate, DevicePlacement, DeviceSchedule};
use crate::scheduler::device::{ResidentSearchConfig, ResidentSearchResult};

use super::batch::BackendError;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Candidate {
    ulong continuation;
    ulong process;
    uint bin;
    uint run_class;
    uint waiting_since;
    uint state_access;
    uint input_order;
    uint reserved;
};

struct Placement {
    uint disposition;
    uint bin;
    uint run_class;
    uint bin_rank;
    uint cohort;
    uint lane_in_cohort;
    uint input_order;
    uint reserved;
};

inline bool wins_mutable_claim(
    device const Candidate* candidates,
    uint count,
    uint at
) {
    Candidate candidate = candidates[at];
    if (candidate.state_access != 2u) return true;
    for (uint other = 0u; other < count; ++other) {
        Candidate rival = candidates[other];
        if (rival.state_access != 2u || rival.process != candidate.process) continue;
        bool earlier = rival.waiting_since < candidate.waiting_since;
        bool identity_tie = rival.waiting_since == candidate.waiting_since
            && rival.continuation < candidate.continuation;
        if (earlier || identity_tie) return false;
    }
    return true;
}

kernel void soma_device_admit(
    device const Candidate* candidates [[buffer(0)]],
    device Placement* placements [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    Candidate candidate = candidates[gid];
    Placement placement;
    placement.disposition = wins_mutable_claim(candidates, count, gid) ? 1u : 0u;
    placement.bin = candidate.bin;
    placement.run_class = candidate.run_class;
    placement.bin_rank = 0u;
    placement.cohort = 0u;
    placement.lane_in_cohort = 0u;
    placement.input_order = candidate.input_order;
    placement.reserved = 0u;
    placements[gid] = placement;
}

kernel void soma_device_place(
    device const Candidate* candidates [[buffer(0)]],
    device Placement* placements [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& policy [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count || placements[gid].disposition == 0u) return;
    uint rank = 0u;
    uint bin_count = 0u;
    uint bin = candidates[gid].bin;
    for (uint other = 0u; other < count; ++other) {
        if (placements[other].disposition == 1u && candidates[other].bin == bin) {
            if (other < gid) ++rank;
            ++bin_count;
        }
    }
    uint remainder = bin_count % width;
    uint full_count = bin_count - remainder;
    uint disposition = 1u;
    if (remainder != 0u && rank >= full_count) {
        if (policy == 0u) disposition = 2u;
        if (policy == 1u) disposition = 3u;
    }
    placements[gid].disposition = disposition;
    placements[gid].bin_rank = rank;
    placements[gid].cohort = rank / width;
    placements[gid].lane_in_cohort = rank % width;
}
"#;

pub struct MetalDeviceScheduler {
    device: Device,
    queue: CommandQueue,
    admission: ComputePipelineState,
    placement: ComputePipelineState,
    resident: Option<(Buffer, Buffer, usize)>,
}

impl MetalDeviceScheduler {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_source(SOURCE, &CompileOptions::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let admission_fn = library
            .get_function("soma_device_admit", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let placement_fn = library
            .get_function("soma_device_place", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let admission = device
            .new_compute_pipeline_state_with_function(&admission_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let placement = device
            .new_compute_pipeline_state_with_function(&placement_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self {
            device,
            queue,
            admission,
            placement,
            resident: None,
        })
    }

    pub fn resident_capacity(&self) -> usize {
        self.resident
            .as_ref()
            .map(|(_, _, capacity)| *capacity)
            .unwrap_or(0)
    }

    fn buffers(&mut self, count: usize) -> (&Buffer, &Buffer) {
        let capacity = count.max(1).next_power_of_two();
        let needs_growth = self
            .resident
            .as_ref()
            .is_none_or(|(_, _, resident_capacity)| *resident_capacity < count);
        if needs_growth {
            let candidates = self.device.new_buffer(
                (capacity * std::mem::size_of::<DeviceCandidate>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let placements = self.device.new_buffer(
                (capacity * std::mem::size_of::<DevicePlacement>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.resident = Some((candidates, placements, capacity));
        }
        let (candidates, placements, _) = self.resident.as_ref().expect("resident buffers");
        (candidates, placements)
    }

    pub fn schedule(
        &mut self,
        candidates: &[Candidate],
        width: u16,
        policy: PartialCohortPolicy,
    ) -> Result<DeviceSchedule, BackendError> {
        if candidates.len() > u32::MAX as usize {
            return Err(BackendError::InvalidInput);
        }
        if candidates.is_empty() {
            return Ok(DeviceSchedule::default());
        }
        let width = u32::from(width.max(1)).min(crate::abi::cohorts::MAX_COHORT_WIDTH as u32);
        let count = candidates.len() as u32;
        let policy = policy_code(policy);
        let device_candidates: Vec<_> = candidates
            .iter()
            .copied()
            .enumerate()
            .map(|(index, candidate)| DeviceCandidate::from_candidate(candidate, index))
            .collect();
        let (candidate_buffer, placement_buffer) = self.buffers(candidates.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                device_candidates.as_ptr().cast::<u8>(),
                candidate_buffer.contents().cast::<u8>(),
                std::mem::size_of_val(device_candidates.as_slice()),
            );
        }
        let candidate_buffer = candidate_buffer.clone();
        let placement_buffer = placement_buffer.clone();

        let command = self.queue.new_command_buffer();
        let admit = command.new_compute_command_encoder();
        admit.set_compute_pipeline_state(&self.admission);
        admit.set_buffer(0, Some(&candidate_buffer), 0);
        admit.set_buffer(1, Some(&placement_buffer), 0);
        admit.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&count as *const u32).cast::<c_void>(),
        );
        dispatch(admit, &self.admission, count);
        admit.end_encoding();

        let place = command.new_compute_command_encoder();
        place.set_compute_pipeline_state(&self.placement);
        place.set_buffer(0, Some(&candidate_buffer), 0);
        place.set_buffer(1, Some(&placement_buffer), 0);
        place.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&count as *const u32).cast::<c_void>(),
        );
        place.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&width as *const u32).cast::<c_void>(),
        );
        place.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            (&policy as *const u32).cast::<c_void>(),
        );
        dispatch(place, &self.placement, count);
        place.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }

        let placements = unsafe {
            std::slice::from_raw_parts(
                placement_buffer.contents().cast::<DevicePlacement>(),
                candidates.len(),
            )
        }
        .to_vec();
        Ok(DeviceSchedule { placements })
    }
}

fn policy_code(policy: PartialCohortPolicy) -> u32 {
    match policy {
        PartialCohortPolicy::Defer => 0,
        PartialCohortPolicy::SendToCpu => 1,
        PartialCohortPolicy::RunPartial => 2,
        PartialCohortPolicy::MergeWithGenericClass => 3,
    }
}

fn dispatch(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    count: u32,
) {
    let width = pipeline
        .thread_execution_width()
        .min(pipeline.max_total_threads_per_threadgroup())
        .min(u64::from(count))
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

const RESIDENT_SEARCH_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Node { ulong value; uint depth; uint branching; uint work_iters; uint class_count; };
struct Config { uint capacity; uint branching; uint work_iters; uint class_count; };
struct State {
    atomic_uint current_count; atomic_uint next_count; atomic_uint active_count;
    atomic_uint nodes; atomic_uint epochs; atomic_uint checksum_sum;
    atomic_uint checksum_xor; atomic_uint overflow;
};
kernel void resident_reset(device State& state [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    atomic_store_explicit(&state.active_count, atomic_load_explicit(&state.current_count, memory_order_relaxed), memory_order_relaxed);
    atomic_store_explicit(&state.next_count, 0u, memory_order_relaxed);
}
kernel void resident_execute(
    device const Node* input [[buffer(0)]], device Node* output [[buffer(1)]],
    device State& state [[buffer(2)]], constant Config& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint count = atomic_load_explicit(&state.active_count, memory_order_relaxed);
    if (gid >= count) return;
    Node node = input[gid];
    uint classes = max(node.class_count, 1u);
    uint cls = uint(node.value % ulong(classes));
    ulong multiplier = 31ul + ulong(cls) * 2ul;
    ulong addend = 7ul + ulong(cls);
    ulong value = node.value;
    for (uint i = 0u; i < node.work_iters; ++i) value = value * multiplier + addend;
    uint word = uint(value) ^ uint(value >> 32) ^ (node.depth * 0x9E3779B9u);
    atomic_fetch_add_explicit(&state.nodes, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&state.checksum_sum, word, memory_order_relaxed);
    atomic_fetch_xor_explicit(&state.checksum_xor, word, memory_order_relaxed);
    if (node.depth == 0u || node.branching == 0u) return;
    uint first = atomic_fetch_add_explicit(&state.next_count, node.branching, memory_order_relaxed);
    if (first > config.capacity || node.branching > config.capacity - first) {
        atomic_store_explicit(&state.overflow, 1u, memory_order_relaxed);
        return;
    }
    for (uint child = 0u; child < node.branching; ++child) {
        Node next = { value + ulong(child), node.depth - 1u, node.branching, node.work_iters, node.class_count };
        output[first + child] = next;
    }
}
kernel void resident_finish(device State& state [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    uint active = atomic_load_explicit(&state.active_count, memory_order_relaxed);
    if (active != 0u) atomic_fetch_add_explicit(&state.epochs, 1u, memory_order_relaxed);
    atomic_store_explicit(&state.current_count, atomic_load_explicit(&state.next_count, memory_order_relaxed), memory_order_relaxed);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ResidentNode {
    value: u64,
    depth: u32,
    branching: u32,
    work_iters: u32,
    class_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ResidentConfig {
    capacity: u32,
    branching: u32,
    work_iters: u32,
    class_count: u32,
}

/// A bounded multi-epoch command graph whose scheduler state never returns to
/// the host between epochs.
pub struct MetalResidentSearch {
    device: Device,
    queue: CommandQueue,
    reset: ComputePipelineState,
    execute: ComputePipelineState,
    finish: ComputePipelineState,
}

impl MetalResidentSearch {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        let library = device
            .new_library_with_source(RESIDENT_SEARCH_SOURCE, &CompileOptions::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = |name| -> Result<ComputePipelineState, BackendError> {
            let function = library
                .get_function(name, None)
                .map_err(|_| BackendError::ExecutionFailed)?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| BackendError::ExecutionFailed)
        };
        Ok(Self {
            queue: device.new_command_queue(),
            reset: pipeline("resident_reset")?,
            execute: pipeline("resident_execute")?,
            finish: pipeline("resident_finish")?,
            device,
        })
    }

    pub fn run(
        &mut self,
        config: ResidentSearchConfig,
    ) -> Result<ResidentSearchResult, BackendError> {
        let capacity = config
            .node_count()
            .ok_or(BackendError::InvalidInput)?
            .max(config.roots)
            .max(1);
        let bytes = u64::from(capacity) * std::mem::size_of::<ResidentNode>() as u64;
        let buffers = [
            self.device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            self.device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
        ];
        let state = self
            .device
            .new_buffer(8 * 4, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(state.contents(), 0, 8 * 4) };
        unsafe {
            let words = std::slice::from_raw_parts_mut(state.contents().cast::<u32>(), 8);
            words[0] = config.roots;
            let roots = std::slice::from_raw_parts_mut(
                buffers[0].contents().cast::<ResidentNode>(),
                config.roots as usize,
            );
            for (root, node) in roots.iter_mut().enumerate() {
                *node = ResidentNode {
                    value: root as u64 + 1,
                    depth: config.depth,
                    branching: config.branching,
                    work_iters: config.work_iters,
                    class_count: config.class_count,
                };
            }
        }
        let constants = ResidentConfig {
            capacity,
            branching: config.branching,
            work_iters: config.work_iters,
            class_count: config.class_count,
        };
        let command = self.queue.new_command_buffer();
        for epoch in 0..=config.depth {
            let reset = command.new_compute_command_encoder();
            reset.set_compute_pipeline_state(&self.reset);
            reset.set_buffer(0, Some(&state), 0);
            dispatch(reset, &self.reset, 1);
            reset.end_encoding();
            let execute = command.new_compute_command_encoder();
            execute.set_compute_pipeline_state(&self.execute);
            execute.set_buffer(0, Some(&buffers[(epoch & 1) as usize]), 0);
            execute.set_buffer(1, Some(&buffers[((epoch + 1) & 1) as usize]), 0);
            execute.set_buffer(2, Some(&state), 0);
            execute.set_bytes(
                3,
                std::mem::size_of::<ResidentConfig>() as u64,
                (&constants as *const ResidentConfig).cast(),
            );
            dispatch(execute, &self.execute, capacity);
            execute.end_encoding();
            let finish = command.new_compute_command_encoder();
            finish.set_compute_pipeline_state(&self.finish);
            finish.set_buffer(0, Some(&state), 0);
            dispatch(finish, &self.finish, 1);
            finish.end_encoding();
        }
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }
        let words = unsafe { std::slice::from_raw_parts(state.contents().cast::<u32>(), 8) };
        Ok(ResidentSearchResult {
            nodes: words[3],
            epochs: words[4],
            checksum_sum: words[5],
            checksum_xor: words[6],
            overflow: words[7],
        })
    }
}
