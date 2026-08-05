//! Continuation and run-class ABI (§8, §9).

use super::refs::Ref64;
use super::AbiHeader;

/// A continuation's declared access to its process's canonical state object.
/// Private continuation frames are governed separately by I8.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateAccess {
    ReadOnly = 1,
    Mutable = 2,
}

/// Continuation states (§8).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuationState {
    New = 1,
    Runnable = 2,
    Running = 3,
    Waiting = 4,
    Completed = 5,
    Cancelled = 6,
    Faulted = 7,
}

/// Result kinds a continuation may return (§8).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    Complete = 1,
    Yield = 2,
    Await = 3,
    Send = 4,
    Spawn = 5,
    Fault = 6,
}

/// The result of one continuation step (§8). A continuation must return control
/// before exhausting its declared maximum step budget; in Phase 1 this is
/// compiler- or programmer-enforced, not hardware-enforced.
#[derive(Clone, Copy, Debug)]
pub struct StepResult {
    pub kind: StepKind,
    pub next_run_class: u32,

    /// Target of the side effect (for Send/Spawn/Await/Fault).
    pub target: Ref64,
    /// Capability carried by the side effect (for Send/Spawn).
    pub value: Ref64,

    pub consumed_steps: u32,
    pub flags: u32,
}

impl StepResult {
    pub fn complete() -> StepResult {
        StepResult {
            kind: StepKind::Complete,
            next_run_class: 0,
            target: Ref64::NULL,
            value: Ref64::NULL,
            consumed_steps: 1,
            flags: 0,
        }
    }

    pub fn yield_next(next_run_class: u32) -> StepResult {
        StepResult {
            kind: StepKind::Yield,
            next_run_class,
            target: Ref64::NULL,
            value: Ref64::NULL,
            consumed_steps: 1,
            flags: 0,
        }
    }

    pub fn await_on(target: Ref64, next_run_class: u32) -> StepResult {
        StepResult {
            kind: StepKind::Await,
            next_run_class,
            target,
            value: Ref64::NULL,
            consumed_steps: 1,
            flags: 0,
        }
    }

    pub fn send(target: Ref64, value: Ref64) -> StepResult {
        StepResult {
            kind: StepKind::Send,
            next_run_class: 0,
            target,
            value,
            consumed_steps: 1,
            flags: 0,
        }
    }

    pub fn spawn(target: Ref64, next_run_class: u32) -> StepResult {
        StepResult {
            kind: StepKind::Spawn,
            next_run_class,
            target,
            value: Ref64::NULL,
            consumed_steps: 1,
            flags: 0,
        }
    }

    pub fn fault(target: Ref64, next_run_class: u32) -> StepResult {
        StepResult {
            kind: StepKind::Fault,
            next_run_class,
            target,
            value: Ref64::NULL,
            consumed_steps: 1,
            flags: 0,
        }
    }
}

/// Continuation descriptor (§8). The continuation is the actual schedulable unit.
#[derive(Clone, Debug)]
pub struct ContinuationDescriptor {
    pub header: AbiHeader,

    pub id: Ref64,
    pub process: Ref64,

    pub run_class: u32,
    pub resume_point: u32,
    pub execution_contract: Ref64,

    pub frame: Ref64, // CapRef
    pub dependency: Ref64,

    pub deadline_ns: u64,

    pub remaining_steps: u32,
    pub priority: u16,
    pub status: ContinuationState,
    pub state_access: StateAccess,

    pub created_epoch: u32,
    pub last_run_epoch: u32,
}

impl ContinuationDescriptor {
    pub fn new(
        process: Ref64,
        state_access: StateAccess,
        run_class: u32,
        resume_point: u32,
    ) -> ContinuationDescriptor {
        ContinuationDescriptor {
            header: AbiHeader::new(5, std::mem::size_of::<ContinuationDescriptor>() as u32),
            id: Ref64::NULL,
            process,
            run_class,
            resume_point,
            execution_contract: Ref64::NULL,
            frame: Ref64::NULL,
            dependency: Ref64::NULL,
            deadline_ns: 0,
            remaining_steps: 0,
            priority: 0,
            status: ContinuationState::New,
            state_access,
            created_epoch: 0,
            last_run_epoch: 0,
        }
    }
}

/// Run-class descriptor (§9). Captures the code implementation, resume point,
/// execution shape, frame layout, precision mode, resource class, and supported
/// engines — so the scheduler need not inspect arbitrary continuation metadata.
#[derive(Clone, Debug)]
pub struct RunClassDescriptor {
    pub header: AbiHeader,

    pub id: u32,
    pub module_function: u32,

    pub execution_contract: Ref64,

    pub engine_mask: u16,
    pub cohort_width: u16,
    pub priority_class: u16,
    pub flags: u16,

    pub frame_type: u32,
    pub result_type: u32,
}

impl RunClassDescriptor {
    pub fn new(id: u32) -> RunClassDescriptor {
        RunClassDescriptor {
            header: AbiHeader::new(5, std::mem::size_of::<RunClassDescriptor>() as u32),
            id,
            module_function: 0,
            execution_contract: Ref64::NULL,
            engine_mask: 1, // bit 0 = CPU scalar
            cohort_width: 1,
            priority_class: 0,
            flags: 0,
            frame_type: 0,
            result_type: 0,
        }
    }
}
