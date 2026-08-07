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
use crate::scheduler::device::{
    DeviceCandidate, DeviceLaneAccess, DeviceLaneConflict, DevicePlacement, DeviceSchedule,
    LaneConflictValidator, LaneValidationError,
};
use crate::scheduler::device::{ResidentSearchConfig, ResidentSearchResult};
use crate::scheduler::device_ops::{
    DeviceLaneOperation, DeviceOperationJournal, DeviceOperationSlice,
};

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

struct LaneAccess {
    ulong resource;
    uint lane;
    uint resource_kind;
    uint mode;
    uint ordinal;
};

struct LaneConflict {
    uint lane;
    uint conflicts;
    uint first_other_lane;
    uint reserved;
};

struct LaneOperation {
    uint lane; uint ordinal; uint opcode; uint flags;
    ulong actor; ulong target; ulong value; ulong auxiliary; ulong result_ref;
    uint payload_offset; uint payload_len; uint result_code; uint result_aux;
};

struct OperationSlice {
    uint operation_offset;
    uint operation_count;
    uint payload_len;
    uint expected_lane;
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

inline bool index_after(device const Candidate* candidates, uint left, uint right, uint mode) {
    if (left == 0xFFFFFFFFu) return right != 0xFFFFFFFFu;
    if (right == 0xFFFFFFFFu) return false;
    Candidate a = candidates[left];
    Candidate b = candidates[right];
    if (mode == 0u) {
        if (a.process != b.process) return a.process > b.process;
        if (a.state_access != b.state_access) return a.state_access < b.state_access;
        if (a.waiting_since != b.waiting_since) return a.waiting_since > b.waiting_since;
        if (a.continuation != b.continuation) return a.continuation > b.continuation;
        return a.input_order > b.input_order;
    }
    return a.bin > b.bin || (a.bin == b.bin && a.input_order > b.input_order);
}

kernel void soma_device_index_init(
    device const Placement* placements [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& padded [[buffer(3)]],
    constant uint& admitted_only [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= padded) return;
    indices[gid] = gid < count && (admitted_only == 0u || placements[gid].disposition == 1u)
        ? gid : 0xFFFFFFFFu;
}

kernel void soma_device_index_sort(
    device const Candidate* candidates [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& padded [[buffer(2)]],
    constant uint& merge_width [[buffer(3)]],
    constant uint& compare_distance [[buffer(4)]],
    constant uint& mode [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= padded) return;
    uint partner = gid ^ compare_distance;
    if (partner <= gid || partner >= padded) return;
    uint left = indices[gid];
    uint right = indices[partner];
    bool descending = (gid & merge_width) != 0u;
    bool swap = descending ? index_after(candidates, right, left, mode)
                           : index_after(candidates, left, right, mode);
    if (swap) {
        indices[gid] = right;
        indices[partner] = left;
    }
}

kernel void soma_device_admit_sorted(
    device const Candidate* candidates [[buffer(0)]],
    device Placement* placements [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint sorted_at [[thread_position_in_grid]]
) {
    if (sorted_at >= count) return;
    uint original = indices[sorted_at];
    Candidate candidate = candidates[original];
    bool wins = true;
    if (candidate.state_access == 2u && sorted_at > 0u) {
        Candidate previous = candidates[indices[sorted_at - 1u]];
        wins = previous.process != candidate.process || previous.state_access != 2u;
    }
    Placement placement;
    placement.disposition = wins ? 1u : 0u;
    placement.bin = candidate.bin;
    placement.run_class = candidate.run_class;
    placement.bin_rank = 0u;
    placement.cohort = 0u;
    placement.lane_in_cohort = 0u;
    placement.input_order = candidate.input_order;
    placement.reserved = 0u;
    placements[original] = placement;
}

kernel void soma_device_place_sorted(
    device const Candidate* candidates [[buffer(0)]],
    device Placement* placements [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& policy [[buffer(5)]],
    uint sorted_at [[thread_position_in_grid]]
) {
    if (sorted_at >= count) return;
    uint original = indices[sorted_at];
    if (original == 0xFFFFFFFFu) return;
    uint bin = candidates[original].bin;
    uint low = 0u;
    uint high = sorted_at;
    while (low < high) {
        uint mid = low + (high - low) / 2u;
        uint candidate_index = indices[mid];
        if (candidate_index != 0xFFFFFFFFu && candidates[candidate_index].bin < bin) low = mid + 1u;
        else high = mid;
    }
    uint first = low;
    low = sorted_at + 1u;
    high = count;
    while (low < high) {
        uint mid = low + (high - low) / 2u;
        uint candidate_index = indices[mid];
        if (candidate_index != 0xFFFFFFFFu && candidates[candidate_index].bin <= bin) low = mid + 1u;
        else high = mid;
    }
    uint bin_count = low - first;
    uint rank = sorted_at - first;
    uint remainder = bin_count % width;
    uint full_count = bin_count - remainder;
    uint disposition = 1u;
    if (remainder != 0u && rank >= full_count) {
        if (policy == 0u) disposition = 2u;
        if (policy == 1u) disposition = 3u;
    }
    placements[original].disposition = disposition;
    placements[original].bin_rank = rank;
    placements[original].cohort = rank / width;
    placements[original].lane_in_cohort = rank % width;
}

kernel void soma_device_validate_journals(
    device const LaneAccess* accesses [[buffer(0)]],
    device LaneConflict* conflicts [[buffer(1)]],
    constant uint& access_count [[buffer(2)]],
    constant uint& lane_count [[buffer(3)]],
    uint lane [[thread_position_in_grid]]
) {
    if (lane >= lane_count) return;
    uint first = 0xFFFFFFFFu;
    for (uint left = 0u; left < access_count; ++left) {
        LaneAccess access = accesses[left];
        if (access.lane != lane) continue;
        for (uint right = 0u; right < access_count; ++right) {
            LaneAccess other = accesses[right];
            if (other.lane == lane || other.resource != access.resource
                || other.resource_kind != access.resource_kind) continue;
            if (access.mode == 2u || other.mode == 2u) first = min(first, other.lane);
        }
    }
    LaneConflict result = {lane, first != 0xFFFFFFFFu, first, 0u};
    conflicts[lane] = result;
}

kernel void soma_device_validate_operations(
    device const LaneOperation* operations [[buffer(0)]],
    device const OperationSlice* slices [[buffer(1)]],
    device uint* valid [[buffer(2)]],
    constant uint& slice_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= slice_count) return;
    OperationSlice slice = slices[gid];
    uint ok = 1u;
    for (uint ordinal = 0u; ordinal < slice.operation_count; ++ordinal) {
        LaneOperation operation = operations[slice.operation_offset + ordinal];
        bool payload_fits = operation.payload_offset <= slice.payload_len
            && operation.payload_len <= slice.payload_len - operation.payload_offset;
        if (operation.ordinal != ordinal || operation.opcode < 1u || operation.opcode > 11u
            || operation.lane != slice.expected_lane || !payload_fits) ok = 0u;
    }
    valid[gid] = ok;
}
"#;

pub struct MetalDeviceScheduler {
    device: Device,
    queue: CommandQueue,
    admission: ComputePipelineState,
    sorted_admission: ComputePipelineState,
    index_init: ComputePipelineState,
    index_sort: ComputePipelineState,
    placement: ComputePipelineState,
    journal_validation: ComputePipelineState,
    operation_validation: ComputePipelineState,
    resident: Option<(Buffer, Buffer, Buffer, usize)>,
    journal_resident: Option<(Buffer, Buffer, usize, usize)>,
    operation_resident: Option<(Buffer, Buffer, Buffer, usize, usize)>,
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
        let sorted_admission_fn = library
            .get_function("soma_device_admit_sorted", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let index_init_fn = library
            .get_function("soma_device_index_init", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let index_sort_fn = library
            .get_function("soma_device_index_sort", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let placement_fn = library
            .get_function("soma_device_place_sorted", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let journal_validation_fn = library
            .get_function("soma_device_validate_journals", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let operation_validation_fn = library
            .get_function("soma_device_validate_operations", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let admission = device
            .new_compute_pipeline_state_with_function(&admission_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let sorted_admission = device
            .new_compute_pipeline_state_with_function(&sorted_admission_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let index_init = device
            .new_compute_pipeline_state_with_function(&index_init_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let index_sort = device
            .new_compute_pipeline_state_with_function(&index_sort_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let placement = device
            .new_compute_pipeline_state_with_function(&placement_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let journal_validation = device
            .new_compute_pipeline_state_with_function(&journal_validation_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let operation_validation = device
            .new_compute_pipeline_state_with_function(&operation_validation_fn)
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self {
            device,
            queue,
            admission,
            sorted_admission,
            index_init,
            index_sort,
            placement,
            journal_validation,
            operation_validation,
            resident: None,
            journal_resident: None,
            operation_resident: None,
        })
    }

    pub fn resident_capacity(&self) -> usize {
        self.resident
            .as_ref()
            .map(|(_, _, _, capacity)| *capacity)
            .unwrap_or(0)
    }

    pub fn journal_resident_capacity(&self) -> (usize, usize) {
        self.journal_resident
            .as_ref()
            .map(|(_, _, accesses, lanes)| (*accesses, *lanes))
            .unwrap_or((0, 0))
    }

    fn buffers(&mut self, count: usize) -> (&Buffer, &Buffer, &Buffer) {
        let capacity = count.max(1).next_power_of_two();
        let needs_growth = self
            .resident
            .as_ref()
            .is_none_or(|(_, _, _, resident_capacity)| *resident_capacity < count);
        if needs_growth {
            let candidates = self.device.new_buffer(
                (capacity * std::mem::size_of::<DeviceCandidate>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let placements = self.device.new_buffer(
                (capacity * std::mem::size_of::<DevicePlacement>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let indices = self.device.new_buffer(
                (capacity * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.resident = Some((candidates, placements, indices, capacity));
        }
        let (candidates, placements, indices, _) =
            self.resident.as_ref().expect("resident buffers");
        (candidates, placements, indices)
    }

    fn journal_buffers(&mut self, access_count: usize, lane_count: usize) -> (&Buffer, &Buffer) {
        let access_capacity = access_count.max(1).next_power_of_two();
        let lane_capacity = lane_count.max(1).next_power_of_two();
        let needs_growth = self.journal_resident.as_ref().is_none_or(
            |(_, _, resident_accesses, resident_lanes)| {
                *resident_accesses < access_count || *resident_lanes < lane_count
            },
        );
        if needs_growth {
            let accesses = self.device.new_buffer(
                (access_capacity * std::mem::size_of::<DeviceLaneAccess>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let conflicts = self.device.new_buffer(
                (lane_capacity * std::mem::size_of::<DeviceLaneConflict>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.journal_resident = Some((accesses, conflicts, access_capacity, lane_capacity));
        }
        let (accesses, conflicts, _, _) = self
            .journal_resident
            .as_ref()
            .expect("resident journal buffers");
        (accesses, conflicts)
    }

    fn operation_buffers(
        &mut self,
        operation_count: usize,
        slice_count: usize,
    ) -> (&Buffer, &Buffer, &Buffer) {
        let operation_capacity = operation_count.max(1).next_power_of_two();
        let slice_capacity = slice_count.max(1).next_power_of_two();
        let needs_growth = self.operation_resident.as_ref().is_none_or(
            |(_, _, _, resident_operations, resident_slices)| {
                *resident_operations < operation_count || *resident_slices < slice_count
            },
        );
        if needs_growth {
            let operations = self.device.new_buffer(
                (operation_capacity * std::mem::size_of::<DeviceLaneOperation>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let slices = self.device.new_buffer(
                (slice_capacity * std::mem::size_of::<DeviceOperationSlice>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let valid = self.device.new_buffer(
                (slice_capacity * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            self.operation_resident = Some((
                operations,
                slices,
                valid,
                operation_capacity,
                slice_capacity,
            ));
        }
        let (operations, slices, valid, _, _) = self
            .operation_resident
            .as_ref()
            .expect("resident operation buffers");
        (operations, slices, valid)
    }

    /// Validate a complete epoch's lane access journals concurrently.
    /// Results are indexed by canonical lane number, not device completion
    /// order, and the buffers remain resident for later epochs.
    pub fn validate_lane_journals(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
    ) -> Result<Vec<DeviceLaneConflict>, BackendError> {
        if accesses.len() > u32::MAX as usize
            || accesses.iter().any(|access| access.lane >= lane_count)
        {
            return Err(BackendError::InvalidInput);
        }
        if lane_count == 0 {
            return if accesses.is_empty() {
                Ok(Vec::new())
            } else {
                Err(BackendError::InvalidInput)
            };
        }
        let access_count = accesses.len() as u32;
        let (access_buffer, conflict_buffer) =
            self.journal_buffers(accesses.len(), lane_count as usize);
        if !accesses.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    accesses.as_ptr().cast::<u8>(),
                    access_buffer.contents().cast::<u8>(),
                    std::mem::size_of_val(accesses),
                );
            }
        }
        let access_buffer = access_buffer.clone();
        let conflict_buffer = conflict_buffer.clone();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.journal_validation);
        encoder.set_buffer(0, Some(&access_buffer), 0);
        encoder.set_buffer(1, Some(&conflict_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&access_count as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&lane_count as *const u32).cast::<c_void>(),
        );
        dispatch(encoder, &self.journal_validation, lane_count);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                conflict_buffer.contents().cast::<DeviceLaneConflict>(),
                lane_count as usize,
            )
        }
        .to_vec())
    }

    /// Validate fixed-width operation records and every byte-arena bound on
    /// Metal. Payload bytes do not need to be copied for a structural check;
    /// each slice supplies the authoritative arena length.
    pub fn validate_operation_journals(
        &mut self,
        journals: &[&DeviceOperationJournal],
    ) -> Result<(), BackendError> {
        if journals.is_empty() {
            return Ok(());
        }
        let operation_count = journals.iter().try_fold(0usize, |total, journal| {
            total.checked_add(journal.operations.len())
        });
        let Some(operation_count) = operation_count else {
            return Err(BackendError::InvalidInput);
        };
        if operation_count > u32::MAX as usize || journals.len() > u32::MAX as usize {
            return Err(BackendError::InvalidInput);
        }
        let mut operations = Vec::with_capacity(operation_count);
        let mut slices = Vec::with_capacity(journals.len());
        for journal in journals {
            let operation_offset: u32 = operations
                .len()
                .try_into()
                .map_err(|_| BackendError::InvalidInput)?;
            let operation_count: u32 = journal
                .operations
                .len()
                .try_into()
                .map_err(|_| BackendError::InvalidInput)?;
            let payload_len: u32 = journal
                .payload
                .len()
                .try_into()
                .map_err(|_| BackendError::InvalidInput)?;
            let expected_lane = journal
                .operations
                .first()
                .map(|operation| operation.lane)
                .unwrap_or(u32::MAX);
            operations.extend_from_slice(&journal.operations);
            slices.push(DeviceOperationSlice {
                operation_offset,
                operation_count,
                payload_len,
                expected_lane,
            });
        }
        let (operation_buffer, slice_buffer, valid_buffer) =
            self.operation_buffers(operations.len(), slices.len());
        unsafe {
            if !operations.is_empty() {
                std::ptr::copy_nonoverlapping(
                    operations.as_ptr().cast::<u8>(),
                    operation_buffer.contents().cast::<u8>(),
                    std::mem::size_of_val(operations.as_slice()),
                );
            }
            std::ptr::copy_nonoverlapping(
                slices.as_ptr().cast::<u8>(),
                slice_buffer.contents().cast::<u8>(),
                std::mem::size_of_val(slices.as_slice()),
            );
        }
        let operation_buffer = operation_buffer.clone();
        let slice_buffer = slice_buffer.clone();
        let valid_buffer = valid_buffer.clone();
        let slice_count = slices.len() as u32;
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.operation_validation);
        encoder.set_buffer(0, Some(&operation_buffer), 0);
        encoder.set_buffer(1, Some(&slice_buffer), 0);
        encoder.set_buffer(2, Some(&valid_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&slice_count as *const u32).cast::<c_void>(),
        );
        dispatch(encoder, &self.operation_validation, slice_count);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }
        let valid = unsafe {
            std::slice::from_raw_parts(valid_buffer.contents().cast::<u32>(), slices.len())
        };
        if valid.iter().all(|valid| *valid == 1) {
            Ok(())
        } else {
            Err(BackendError::InvalidInput)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_index_sort(
        &self,
        command: &metal::CommandBufferRef,
        candidate_buffer: &Buffer,
        placement_buffer: &Buffer,
        index_buffer: &Buffer,
        count: u32,
        padded: u32,
        admitted_only: u32,
        mode: u32,
    ) {
        let initialize = command.new_compute_command_encoder();
        initialize.set_compute_pipeline_state(&self.index_init);
        initialize.set_buffer(0, Some(placement_buffer), 0);
        initialize.set_buffer(1, Some(index_buffer), 0);
        initialize.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&count as *const u32).cast::<c_void>(),
        );
        initialize.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&padded as *const u32).cast::<c_void>(),
        );
        initialize.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            (&admitted_only as *const u32).cast::<c_void>(),
        );
        dispatch(initialize, &self.index_init, padded);
        initialize.end_encoding();

        let mut merge_width = 2u32;
        while merge_width <= padded {
            let mut compare_distance = merge_width / 2;
            while compare_distance > 0 {
                let sort = command.new_compute_command_encoder();
                sort.set_compute_pipeline_state(&self.index_sort);
                sort.set_buffer(0, Some(candidate_buffer), 0);
                sort.set_buffer(1, Some(index_buffer), 0);
                sort.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    (&padded as *const u32).cast::<c_void>(),
                );
                sort.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    (&merge_width as *const u32).cast::<c_void>(),
                );
                sort.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    (&compare_distance as *const u32).cast::<c_void>(),
                );
                sort.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    (&mode as *const u32).cast::<c_void>(),
                );
                dispatch(sort, &self.index_sort, padded);
                sort.end_encoding();
                compare_distance /= 2;
            }
            merge_width *= 2;
        }
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
        let (candidate_buffer, placement_buffer, index_buffer) = self.buffers(candidates.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                device_candidates.as_ptr().cast::<u8>(),
                candidate_buffer.contents().cast::<u8>(),
                std::mem::size_of_val(device_candidates.as_slice()),
            );
        }
        let candidate_buffer = candidate_buffer.clone();
        let placement_buffer = placement_buffer.clone();
        let index_buffer = index_buffer.clone();

        let padded = candidates.len().next_power_of_two() as u32;
        let command = self.queue.new_command_buffer();
        let mutable_count = candidates
            .iter()
            .filter(|candidate| candidate.state_access == crate::abi::StateAccess::Mutable)
            .count();
        if mutable_count >= 128 {
            self.encode_index_sort(
                command,
                &candidate_buffer,
                &placement_buffer,
                &index_buffer,
                count,
                padded,
                0,
                0,
            );
            let admit = command.new_compute_command_encoder();
            admit.set_compute_pipeline_state(&self.sorted_admission);
            admit.set_buffer(0, Some(&candidate_buffer), 0);
            admit.set_buffer(1, Some(&placement_buffer), 0);
            admit.set_buffer(2, Some(&index_buffer), 0);
            admit.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                (&count as *const u32).cast::<c_void>(),
            );
            dispatch(admit, &self.sorted_admission, count);
            admit.end_encoding();
        } else {
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
        }

        self.encode_index_sort(
            command,
            &candidate_buffer,
            &placement_buffer,
            &index_buffer,
            count,
            padded,
            1,
            1,
        );

        let place = command.new_compute_command_encoder();
        place.set_compute_pipeline_state(&self.placement);
        place.set_buffer(0, Some(&candidate_buffer), 0);
        place.set_buffer(1, Some(&placement_buffer), 0);
        place.set_buffer(2, Some(&index_buffer), 0);
        place.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&count as *const u32).cast::<c_void>(),
        );
        place.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            (&width as *const u32).cast::<c_void>(),
        );
        place.set_bytes(
            5,
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

impl LaneConflictValidator for MetalDeviceScheduler {
    fn validate_lane_journals(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError> {
        MetalDeviceScheduler::validate_lane_journals(self, accesses, lane_count)
            .map_err(lane_validation_error)
    }

    fn validate_epoch(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
        operations: &[&DeviceOperationJournal],
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError> {
        if operations.len() != lane_count as usize {
            return Err(LaneValidationError::InvalidInput);
        }
        self.validate_operation_journals(operations)
            .map_err(lane_validation_error)?;
        self.validate_lane_journals(accesses, lane_count)
            .map_err(lane_validation_error)
    }
}

fn lane_validation_error(error: BackendError) -> LaneValidationError {
    match error {
        BackendError::InvalidInput => LaneValidationError::InvalidInput,
        BackendError::Unavailable => LaneValidationError::Unavailable,
        _ => LaneValidationError::ExecutionFailed,
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
struct Config { uint capacity; uint branching; uint work_iters; uint class_count; uint cohort_width; };
struct State {
    atomic_uint current_count; atomic_uint next_count; atomic_uint active_count;
    atomic_uint nodes; atomic_uint epochs; atomic_uint checksum_sum;
    atomic_uint checksum_xor; atomic_uint overflow; atomic_uint cohorts;
    atomic_uint lane_slots; atomic_uint useful_lane_slots;
};
kernel void resident_reset(device State& state [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    atomic_store_explicit(&state.active_count, atomic_load_explicit(&state.current_count, memory_order_relaxed), memory_order_relaxed);
    atomic_store_explicit(&state.next_count, 0u, memory_order_relaxed);
}
kernel void resident_place(
    device const Node* input [[buffer(0)]], device Node* placed [[buffer(1)]],
    device State& state [[buffer(2)]], constant Config& config [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint count = atomic_load_explicit(&state.active_count, memory_order_relaxed);
    if (gid >= count) return;
    uint classes = max(input[gid].class_count, 1u);
    uint cls = uint(input[gid].value % ulong(classes));
    uint before_classes = 0u;
    uint rank = 0u;
    uint class_count = 0u;
    for (uint other = 0u; other < count; ++other) {
        uint other_classes = max(input[other].class_count, 1u);
        uint other_class = uint(input[other].value % ulong(other_classes));
        if (other_class < cls) ++before_classes;
        if (other_class == cls) {
            if (other < gid) ++rank;
            ++class_count;
        }
    }
    placed[before_classes + rank] = input[gid];
    uint width = max(config.cohort_width, 1u);
    if ((rank % width) == 0u) {
        atomic_fetch_add_explicit(&state.cohorts, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&state.lane_slots, width, memory_order_relaxed);
        atomic_fetch_add_explicit(&state.useful_lane_slots, min(width, class_count - rank), memory_order_relaxed);
    }
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
    cohort_width: u32,
}

/// A bounded multi-epoch command graph whose scheduler state never returns to
/// the host between epochs.
pub struct MetalResidentSearch {
    device: Device,
    queue: CommandQueue,
    reset: ComputePipelineState,
    place: ComputePipelineState,
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
            place: pipeline("resident_place")?,
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
        let placed = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let state = self
            .device
            .new_buffer(11 * 4, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(state.contents(), 0, 11 * 4) };
        unsafe {
            let words = std::slice::from_raw_parts_mut(state.contents().cast::<u32>(), 11);
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
            cohort_width: config.cohort_width.max(1),
        };
        let command = self.queue.new_command_buffer();
        for epoch in 0..=config.depth {
            let reset = command.new_compute_command_encoder();
            reset.set_compute_pipeline_state(&self.reset);
            reset.set_buffer(0, Some(&state), 0);
            dispatch(reset, &self.reset, 1);
            reset.end_encoding();
            let place = command.new_compute_command_encoder();
            place.set_compute_pipeline_state(&self.place);
            place.set_buffer(0, Some(&buffers[(epoch & 1) as usize]), 0);
            place.set_buffer(1, Some(&placed), 0);
            place.set_buffer(2, Some(&state), 0);
            place.set_bytes(
                3,
                std::mem::size_of::<ResidentConfig>() as u64,
                (&constants as *const ResidentConfig).cast(),
            );
            dispatch(place, &self.place, capacity);
            place.end_encoding();
            let execute = command.new_compute_command_encoder();
            execute.set_compute_pipeline_state(&self.execute);
            execute.set_buffer(0, Some(&placed), 0);
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
        let words = unsafe { std::slice::from_raw_parts(state.contents().cast::<u32>(), 11) };
        Ok(ResidentSearchResult {
            nodes: words[3],
            epochs: words[4],
            checksum_sum: words[5],
            checksum_xor: words[6],
            overflow: words[7],
            cohorts: words[8],
            lane_slots: words[9],
            useful_lane_slots: words[10],
        })
    }
}
