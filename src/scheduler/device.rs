//! Fixed-width device scheduling ABI and its executable reference lowering.
//!
//! A device scheduler cannot borrow the kernel's Rust tables. It receives the
//! complete candidate set for one epoch in this pointer-free representation,
//! decides I13 admission from the set, and assigns every admitted candidate a
//! stable position in its run-class bin. The Metal implementation runs the two
//! passes concurrently on GPU threads; this module is the independent oracle.

use crate::abi::cohorts::PartialCohortPolicy;
use crate::abi::{Ref64, StateAccess};
use crate::scheduler::admission::{admit, Candidate};
use crate::scheduler::device_ops::DeviceOperationJournal;

pub const DEVICE_DEFERRED: u32 = 0;
pub const DEVICE_RUN: u32 = 1;
pub const DEVICE_POLICY_DEFERRED: u32 = 2;
pub const DEVICE_SEND_TO_CPU: u32 = 3;

pub const DEVICE_ACCESS_READ: u32 = 1;
pub const DEVICE_ACCESS_WRITE: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneValidationError {
    InvalidInput,
    AuthorityDenied,
    Unavailable,
    NodeLost,
    ProtocolError,
    ExecutionFailed,
}

/// Pointer-free input for one continuation evaluator lane. Variable frame
/// bytes live in the epoch arena named by `frame_offset`/`frame_len`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceEvaluatorLane {
    pub continuation: u64,
    pub process: u64,
    pub frame: u64,
    pub lane: u32,
    pub run_class: u32,
    pub frame_offset: u32,
    pub frame_len: u32,
}

const _: () = assert!(std::mem::size_of::<DeviceEvaluatorLane>() == 40);

/// Pointer-free result of one evaluator lane. The returned frame is a slice
/// of `DeviceEvaluation::frames`; step fields mirror `StepResult` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceEvaluatorResult {
    pub target: u64,
    pub value: u64,
    pub lane: u32,
    pub status: u32,
    pub step_kind: u32,
    pub next_run_class: u32,
    pub consumed_steps: u32,
    pub flags: u32,
    pub frame_offset: u32,
    pub frame_len: u32,
}

const _: () = assert!(std::mem::size_of::<DeviceEvaluatorResult>() == 48);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceEvaluation {
    pub results: Vec<DeviceEvaluatorResult>,
    pub frames: Vec<u8>,
}

/// An epoch backend may decline a body without changing semantic state. The
/// kernel then runs the same lanes through the reference evaluator.
pub trait DeviceEpochBackend: LaneConflictValidator {
    fn evaluate_lanes(
        &mut self,
        lanes: &[DeviceEvaluatorLane],
        frames: &[u8],
    ) -> Result<DeviceEvaluation, LaneValidationError>;
}

/// Pluggable validator used at the speculative epoch's pre-commit boundary.
pub trait LaneConflictValidator {
    fn validate_lane_journals(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError>;

    /// Validate the complete epoch boundary. Validators concerned only with
    /// conflicts may use the default; remote validators can authenticate and
    /// ledger the operation payloads before returning the same decision.
    fn validate_epoch(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
        operations: &[&DeviceOperationJournal],
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError> {
        let _ = operations;
        self.validate_lane_journals(accesses, lane_count)
    }
}

#[derive(Default)]
pub struct ReferenceLaneConflictValidator;

impl LaneConflictValidator for ReferenceLaneConflictValidator {
    fn validate_lane_journals(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError> {
        if accesses.iter().any(|access| access.lane >= lane_count) {
            return Err(LaneValidationError::InvalidInput);
        }
        Ok(reference_lane_conflicts(accesses, lane_count))
    }
}

/// One semantic resource access emitted by a lane.
///
/// The representation is pointer-free and has the same layout in Rust, Metal,
/// and the distributed journal wire format. `resource_kind` is kept separate
/// from the reference because two resource namespaces may intentionally name
/// the same `Ref64` bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLaneAccess {
    pub resource: u64,
    pub lane: u32,
    pub resource_kind: u32,
    pub mode: u32,
    pub ordinal: u32,
}

const _: () = assert!(std::mem::size_of::<DeviceLaneAccess>() == 24);

impl DeviceLaneAccess {
    pub fn new(lane: u32, resource_kind: u32, resource: u64, mode: u32, ordinal: u32) -> Self {
        Self {
            resource,
            lane,
            resource_kind,
            mode,
            ordinal,
        }
    }

    pub fn read(lane: u32, resource_kind: u32, resource: Ref64, ordinal: u32) -> Self {
        Self {
            resource: resource.to_u64(),
            lane,
            resource_kind,
            mode: DEVICE_ACCESS_READ,
            ordinal,
        }
    }

    pub fn write(lane: u32, resource_kind: u32, resource: Ref64, ordinal: u32) -> Self {
        Self {
            resource: resource.to_u64(),
            lane,
            resource_kind,
            mode: DEVICE_ACCESS_WRITE,
            ordinal,
        }
    }
}

/// Conflict decision for one canonical lane.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLaneConflict {
    pub lane: u32,
    pub conflicts: u32,
    pub first_other_lane: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<DeviceLaneConflict>() == 16);

/// Independent oracle for device-side lane-journal validation.
///
/// Two different lanes conflict when they name the same resource namespace
/// and identity and at least one access is a write. Physical access order and
/// duplicate records within one lane cannot affect the result.
pub fn reference_lane_conflicts(
    accesses: &[DeviceLaneAccess],
    lane_count: u32,
) -> Vec<DeviceLaneConflict> {
    let mut conflicts: Vec<_> = (0..lane_count)
        .map(|lane| DeviceLaneConflict {
            lane,
            conflicts: 0,
            first_other_lane: u32::MAX,
            reserved: 0,
        })
        .collect();

    // Resource and lane grouping turns the old all-pairs comparison into one
    // sort followed by linear scans. An access's ordinal and physical input
    // position are intentionally absent from the key: duplicate records in a
    // lane collapse to the same semantic read/write claim.
    let mut sorted: Vec<_> = accesses.iter().collect();
    sorted.sort_unstable_by_key(|access| (access.resource_kind, access.resource, access.lane));

    let mut resource_start = 0;
    while resource_start < sorted.len() {
        let resource_kind = sorted[resource_start].resource_kind;
        let resource = sorted[resource_start].resource;
        let mut resource_end = resource_start + 1;
        while resource_end < sorted.len()
            && sorted[resource_end].resource_kind == resource_kind
            && sorted[resource_end].resource == resource
        {
            resource_end += 1;
        }

        // The first two lanes and first two writing lanes are sufficient to
        // answer "smallest conflicting other lane" for every member. Keeping
        // two handles the case where the smallest one is the member itself.
        let mut first_lane = None;
        let mut second_lane = None;
        let mut first_writer = None;
        let mut second_writer = None;
        let mut lane_start = resource_start;
        while lane_start < resource_end {
            let lane = sorted[lane_start].lane;
            let mut lane_end = lane_start + 1;
            let mut writes = sorted[lane_start].mode == DEVICE_ACCESS_WRITE;
            while lane_end < resource_end && sorted[lane_end].lane == lane {
                writes |= sorted[lane_end].mode == DEVICE_ACCESS_WRITE;
                lane_end += 1;
            }
            if first_lane.is_none() {
                first_lane = Some(lane);
            } else if second_lane.is_none() {
                second_lane = Some(lane);
            }
            if writes {
                if first_writer.is_none() {
                    first_writer = Some(lane);
                } else if second_writer.is_none() {
                    second_writer = Some(lane);
                }
            }
            lane_start = lane_end;
        }

        lane_start = resource_start;
        while lane_start < resource_end {
            let lane = sorted[lane_start].lane;
            let mut lane_end = lane_start + 1;
            let mut writes = sorted[lane_start].mode == DEVICE_ACCESS_WRITE;
            while lane_end < resource_end && sorted[lane_end].lane == lane {
                writes |= sorted[lane_end].mode == DEVICE_ACCESS_WRITE;
                lane_end += 1;
            }
            let candidates = if writes {
                (first_lane, second_lane)
            } else {
                (first_writer, second_writer)
            };
            let other = if candidates.0 != Some(lane) {
                candidates.0
            } else {
                candidates.1
            };
            if let (Some(other), Some(result)) = (other, conflicts.get_mut(lane as usize)) {
                result.conflicts = 1;
                result.first_other_lane = result.first_other_lane.min(other);
            }
            lane_start = lane_end;
        }
        resource_start = resource_end;
    }

    conflicts
}

/// Candidate data copied into persistent device-visible storage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceCandidate {
    pub continuation: u64,
    pub process: u64,
    pub bin: u32,
    pub run_class: u32,
    pub waiting_since: u32,
    pub state_access: u32,
    pub input_order: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<DeviceCandidate>() == 40);

impl DeviceCandidate {
    pub fn from_candidate(candidate: Candidate, input_order: usize) -> Self {
        Self {
            continuation: candidate.continuation.to_u64(),
            process: candidate.process.to_u64(),
            bin: candidate.bin,
            run_class: candidate.run_class,
            waiting_since: candidate.waiting_since,
            state_access: candidate.state_access as u32,
            input_order: input_order as u32,
            reserved: 0,
        }
    }
}

/// One candidate's device-side scheduling result.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevicePlacement {
    pub disposition: u32,
    pub bin: u32,
    pub run_class: u32,
    pub bin_rank: u32,
    pub cohort: u32,
    pub lane_in_cohort: u32,
    pub input_order: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<DevicePlacement>() == 32);

impl DevicePlacement {
    pub fn runs_on_device(self) -> bool {
        self.disposition == DEVICE_RUN
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceSchedule {
    /// Placements are returned in candidate input order.
    pub placements: Vec<DevicePlacement>,
}

impl DeviceSchedule {
    pub fn disposition_counts(&self) -> [usize; 4] {
        let mut counts = [0; 4];
        for placement in &self.placements {
            if let Some(slot) = counts.get_mut(placement.disposition as usize) {
                *slot += 1;
            }
        }
        counts
    }
}

/// Reference implementation of the device scheduling contract.
pub fn reference_device_schedule(
    candidates: &[Candidate],
    width: u16,
    policy: PartialCohortPolicy,
) -> DeviceSchedule {
    let admission = admit(candidates);
    let mut placements: Vec<DevicePlacement> = candidates
        .iter()
        .enumerate()
        .map(|(input_order, candidate)| DevicePlacement {
            disposition: DEVICE_DEFERRED,
            bin: candidate.bin,
            run_class: candidate.run_class,
            input_order: input_order as u32,
            ..DevicePlacement::default()
        })
        .collect();

    for (bin, lanes) in admission.bins() {
        let group = usize::from(width.max(1)).min(crate::abi::cohorts::MAX_COHORT_WIDTH);
        let remainder = lanes.len() % group;
        let full_len = lanes.len() - remainder;
        for (rank, (continuation, _)) in lanes.iter().enumerate() {
            let input_order = candidates
                .iter()
                .position(|candidate| candidate.continuation == *continuation)
                .expect("admission only returns input candidates");
            let disposition = if remainder > 0 && rank >= full_len {
                match policy {
                    PartialCohortPolicy::Defer => DEVICE_POLICY_DEFERRED,
                    PartialCohortPolicy::SendToCpu => DEVICE_SEND_TO_CPU,
                    PartialCohortPolicy::RunPartial
                    | PartialCohortPolicy::MergeWithGenericClass => DEVICE_RUN,
                }
            } else {
                DEVICE_RUN
            };
            placements[input_order].disposition = disposition;
            placements[input_order].bin = *bin;
            placements[input_order].bin_rank = rank as u32;
            placements[input_order].cohort = (rank / group) as u32;
            placements[input_order].lane_in_cohort = (rank % group) as u32;
        }
    }

    DeviceSchedule { placements }
}

pub fn state_access_code(access: StateAccess) -> u32 {
    access as u32
}

pub fn continuation_from_device(value: u64) -> Ref64 {
    Ref64::from_u64(value)
}

/// Bounded dynamic search used to verify that scheduler state survives several
/// device epochs without an epoch-by-epoch host decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentSearchConfig {
    pub roots: u32,
    pub branching: u32,
    pub depth: u32,
    pub class_count: u32,
    pub work_iters: u32,
    pub cohort_width: u32,
}

impl ResidentSearchConfig {
    pub fn node_count(self) -> Option<u32> {
        let mut total = 0u64;
        let mut level = u64::from(self.roots);
        for _ in 0..=self.depth {
            total = total.checked_add(level)?;
            level = level.checked_mul(u64::from(self.branching))?;
        }
        total.try_into().ok()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentSearchResult {
    pub nodes: u32,
    pub epochs: u32,
    pub checksum_sum: u32,
    pub checksum_xor: u32,
    pub overflow: u32,
    pub cohorts: u32,
    pub lane_slots: u32,
    pub useful_lane_slots: u32,
}

pub fn reference_resident_search(config: ResidentSearchConfig) -> ResidentSearchResult {
    reference_resident_search_with_trace(config).0
}

/// Canonical lane-local event emitted by the bounded resident graph.
///
/// The physical Metal thread writes directly to its epoch/lane slot; no atomic
/// append or shared clock participates. `lane_sequence` is therefore local to
/// the lane, matching the trace-position rule used by the general executive.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentTraceEvent {
    pub value: u64,
    pub epoch: u32,
    pub lane: u32,
    pub lane_sequence: u32,
    pub run_class: u32,
    pub word: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<ResidentTraceEvent>() == 32);

/// One root in a compiled resident frame graph. Each lane owns its frame for
/// the lifetime of the graph; `actor`/`target` are retained for the canonical
/// host write journal emitted after the last evaluator dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentFrameBinding {
    pub continuation: u64,
    pub process: u64,
    pub frame: u64,
    pub actor: u64,
    pub target: u64,
}

const _: () = assert!(std::mem::size_of::<ResidentFrameBinding>() == 40);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentFrameGraphConfig {
    pub run_class: u32,
    pub epochs: u32,
    pub cohort_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentFrameTraceEvent {
    pub epoch: u32,
    pub lane: u32,
    pub run_class: u32,
    pub word: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentFrameGraphResult {
    pub frames: Vec<u8>,
    pub accesses: Vec<DeviceLaneAccess>,
    pub operations: Vec<DeviceOperationJournal>,
    pub trace: Vec<ResidentFrameTraceEvent>,
}

/// CPU oracle for the general compiled resident handler path.
///
/// This intentionally uses the evaluator interpreter, rather than duplicating
/// its arithmetic. The operation/access ABI is the same one accepted by the
/// epoch validator; final private frames remain available to the host snapshot
/// commit that performs each `OP_WRITE_OBJECT`.
pub fn reference_resident_frame_graph(
    program: &crate::compiler::body::EvaluatorProgram,
    config: ResidentFrameGraphConfig,
    bindings: &[ResidentFrameBinding],
    input_frames: &[u8],
) -> Option<ResidentFrameGraphResult> {
    use crate::scheduler::device_ops::{DeviceLaneOperation, OP_WRITE_OBJECT};
    let stride = program.stride() as usize;
    if config.run_class < 1024
        || config.cohort_width == 0
        || program.binds_aux()
        || program.ops().iter().any(|op| {
            matches!(
                op,
                crate::compiler::body::Op::Gather(_, _)
                    | crate::compiler::body::Op::GatherAux(_, _)
            )
        })
        || stride == 0
        || input_frames.len() != bindings.len().checked_mul(stride)?
    {
        return None;
    }
    let mut frames = input_frames.to_vec();
    let mut trace = Vec::with_capacity(bindings.len().checked_mul(config.epochs as usize)?);
    for epoch in 0..config.epochs {
        let input = frames.clone();
        for lane in 0..bindings.len() {
            let range = lane * stride..(lane + 1) * stride;
            program.evaluate_at(
                &input,
                bindings.len() as u32,
                lane as u32,
                &mut frames[range.clone()],
            );
            let word = frames[range].iter().fold(2_166_136_261u32, |hash, byte| {
                (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
            });
            trace.push(ResidentFrameTraceEvent {
                epoch,
                lane: lane as u32 + 1,
                run_class: config.run_class,
                word,
            });
        }
    }
    let accesses = bindings
        .iter()
        .enumerate()
        .flat_map(|(lane, binding)| {
            [
                DeviceLaneAccess::new(lane as u32, 1, binding.target, DEVICE_ACCESS_READ, 0),
                DeviceLaneAccess::new(lane as u32, 1, binding.target, DEVICE_ACCESS_WRITE, 1),
            ]
        })
        .collect();
    let operations = bindings
        .iter()
        .enumerate()
        .map(|(lane, binding)| DeviceOperationJournal {
            operations: vec![
                DeviceLaneOperation {
                    lane: lane as u32,
                    ordinal: 0,
                    opcode: crate::scheduler::device_ops::OP_READ_OBJECT,
                    actor: binding.actor,
                    target: binding.target,
                    ..DeviceLaneOperation::default()
                },
                DeviceLaneOperation {
                    lane: lane as u32,
                    ordinal: 1,
                    opcode: OP_WRITE_OBJECT,
                    actor: binding.actor,
                    target: binding.target,
                    ..DeviceLaneOperation::default()
                },
            ],
            payload: Vec::new(),
        })
        .collect();
    Some(ResidentFrameGraphResult {
        frames,
        accesses,
        operations,
        trace,
    })
}

/// Independent transition and trace oracle for the no-round-trip graph.
pub fn reference_resident_search_with_trace(
    config: ResidentSearchConfig,
) -> (ResidentSearchResult, Vec<ResidentTraceEvent>) {
    let mut current: Vec<(u64, u32)> = (0..config.roots)
        .map(|root| (u64::from(root) + 1, config.depth))
        .collect();
    let mut result = ResidentSearchResult::default();
    let mut trace = Vec::new();
    while !current.is_empty() {
        let width = config.cohort_width.max(1);
        let classes = u64::from(config.class_count.max(1));
        let mut class_counts = std::collections::BTreeMap::<u32, u32>::new();
        for (value, _) in &current {
            *class_counts.entry((*value % classes) as u32).or_default() += 1;
        }
        for count in class_counts.values() {
            let cohorts = count.div_ceil(width);
            result.cohorts = result.cohorts.wrapping_add(cohorts);
            result.lane_slots = result.lane_slots.wrapping_add(cohorts.wrapping_mul(width));
            result.useful_lane_slots = result.useful_lane_slots.wrapping_add(*count);
        }
        // resident_place is a stable class ordering. Child publication uses
        // this canonical lane order, rather than physical completion order.
        current.sort_by_key(|(value, _)| *value % classes);
        let mut next = Vec::new();
        for (lane, (input_value, depth)) in current.into_iter().enumerate() {
            let class = (input_value % classes) as u32;
            let value = crate::compiler::state_machine_lowering::search_step(
                input_value,
                config.work_iters,
                class,
            );
            let word = value as u32 ^ (value >> 32) as u32 ^ depth.wrapping_mul(0x9E37_79B9);
            trace.push(ResidentTraceEvent {
                value,
                epoch: result.epochs,
                lane: lane as u32 + 1,
                lane_sequence: 0,
                run_class: class,
                word,
                reserved: 0,
            });
            result.nodes = result.nodes.wrapping_add(1);
            result.checksum_sum = result.checksum_sum.wrapping_add(word);
            result.checksum_xor ^= word;
            if depth > 0 {
                next.extend(
                    (0..config.branching)
                        .map(|child| (value.wrapping_add(u64::from(child)), depth - 1)),
                );
            }
        }
        current = next;
        result.epochs += 1;
    }
    (result, trace)
}
