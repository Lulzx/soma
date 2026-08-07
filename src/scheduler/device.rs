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

pub const DEVICE_DEFERRED: u32 = 0;
pub const DEVICE_RUN: u32 = 1;
pub const DEVICE_POLICY_DEFERRED: u32 = 2;
pub const DEVICE_SEND_TO_CPU: u32 = 3;

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
            process: candidate.process.key(),
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
    let mut current: Vec<(u64, u32)> = (0..config.roots)
        .map(|root| (u64::from(root) + 1, config.depth))
        .collect();
    let mut result = ResidentSearchResult::default();
    while !current.is_empty() {
        let width = config.cohort_width.max(1);
        let mut class_counts = std::collections::BTreeMap::<u32, u32>::new();
        for (value, _) in &current {
            *class_counts
                .entry((*value % u64::from(config.class_count.max(1))) as u32)
                .or_default() += 1;
        }
        for count in class_counts.values() {
            let cohorts = count.div_ceil(width);
            result.cohorts = result.cohorts.wrapping_add(cohorts);
            result.lane_slots = result.lane_slots.wrapping_add(cohorts.wrapping_mul(width));
            result.useful_lane_slots = result.useful_lane_slots.wrapping_add(*count);
        }
        let mut next = Vec::new();
        for (value, depth) in current {
            let class = (value % u64::from(config.class_count.max(1))) as u32;
            let value = crate::compiler::state_machine_lowering::search_step(
                value,
                config.work_iters,
                class,
            );
            let word = value as u32 ^ (value >> 32) as u32 ^ depth.wrapping_mul(0x9E37_79B9);
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
    result
}
