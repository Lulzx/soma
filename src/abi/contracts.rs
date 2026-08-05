//! Execution-contract ABI (§10).

use super::refs::Ref64;
use super::AbiHeader;

/// Phase-1 execution shapes (§10).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Scalar = 1,
    Lanes = 2,
}

/// Phase-1 placement policies (§10, §17). The runtime may override a preference
/// but never a requirement.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementPolicy {
    Any = 1,
    PreferCpu = 2,
    PreferGpu = 3,
    RequireCpu = 4,
    RequireGpu = 5,
}

/// Phase-1 precision policies.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionPolicy {
    Any = 1,
    Float32 = 2,
    Float64 = 3,
}

/// Phase-1 determinism policies.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeterminismPolicy {
    Relaxed = 1,
    Deterministic = 2,
}

/// Execution-contract flags.
#[allow(non_snake_case)]
pub mod ContractFlags {
    pub const NONE: u32 = 0;
    pub const TRACE_EXECUTION: u32 = 1 << 0;
    pub const STRICT_DETERMINISM: u32 = 1 << 1;
}

/// Execution contract (§10).
#[derive(Clone, Debug)]
pub struct ExecutionContract {
    pub header: AbiHeader,

    pub id: Ref64,

    pub shape: Shape,
    pub placement_policy: PlacementPolicy,
    pub precision_policy: PrecisionPolicy,
    pub determinism_policy: DeterminismPolicy,

    pub minimum_parallelism: u16,
    pub preferred_parallelism: u16,

    pub maximum_steps: u32,
    pub local_memory_bytes: u32,

    pub deadline_ns: u64,
    pub expected_read_bytes: u64,
    pub expected_write_bytes: u64,

    pub objective_flags: u32,
    pub contract_flags: u32,
}

impl ExecutionContract {
    pub fn new(shape: Shape, placement: PlacementPolicy) -> ExecutionContract {
        ExecutionContract {
            header: AbiHeader::new(8, std::mem::size_of::<ExecutionContract>() as u32),
            id: Ref64::NULL,
            shape,
            placement_policy: placement,
            precision_policy: PrecisionPolicy::Any,
            determinism_policy: DeterminismPolicy::Deterministic,
            minimum_parallelism: 1,
            preferred_parallelism: 1,
            maximum_steps: 64,
            local_memory_bytes: 0,
            deadline_ns: 0,
            expected_read_bytes: 0,
            expected_write_bytes: 0,
            objective_flags: 0,
            contract_flags: ContractFlags::NONE,
        }
    }
}
