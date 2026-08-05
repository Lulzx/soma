//! Fixed ABI surface for SOMA-P1.
//!
//! This module defines the shared structures every engine executive, the
//! scheduler, and replay can rely on: the header every ABI object begins with,
//! the generational reference format, and all descriptor structs. It mirrors
//! `soma/abi/*` in §24 of the contract.

pub mod capabilities;
pub mod cohorts;
pub mod contracts;
pub mod continuations;
pub mod futures;
pub mod messages;
pub mod objects;
pub mod processes;
pub mod refs;
pub mod traces;

pub use capabilities::{CapabilityEntry, OwnershipMode, Rights};
pub use cohorts::{CohortDescriptor, PartialCohortPolicy, MAX_COHORT_WIDTH};
pub use contracts::{
    ContractFlags, DeterminismPolicy, ExecutionContract, PlacementPolicy, PrecisionPolicy, Shape,
};
pub use continuations::{
    ContinuationDescriptor, ContinuationState, RunClassDescriptor, StepKind, StepResult,
};
pub use futures::{FutureDescriptor, FutureState};
pub use messages::MessageDescriptor;
pub use objects::{ObjectDescriptor, ObjectKind, OwnershipState};
pub use processes::{ProcessDescriptor, ProcessMode, ProcessState};
pub use refs::{Kind, Ref64};
pub use traces::{EventKind, TraceEvent};

/// Header every shared ABI object begins with (§3).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbiHeader {
    pub abi_version: u16,
    pub structure_kind: u16,
    pub byte_length: u32,
}

impl AbiHeader {
    pub const ABI_VERSION: u16 = 1;

    pub fn new(structure_kind: u16, byte_length: u32) -> AbiHeader {
        AbiHeader {
            abi_version: Self::ABI_VERSION,
            structure_kind,
            byte_length,
        }
    }
}

/// Errors raised while resolving references or applying ABI-level checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbiError {
    /// `slot` did not exist in the target table.
    BadSlot,
    /// `ref.kind` did not match the table it was dereferenced against.
    KindMismatch,
    /// `ref.generation` did not equal the current slot generation (§4).
    StaleReference,
    /// An operation required authority the caller did not hold.
    NoAuthority,
    /// A bounds or length check failed.
    BoundsViolation,
    /// A structural kind field was unknown.
    UnknownKind,
}
