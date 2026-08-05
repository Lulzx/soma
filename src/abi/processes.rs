//! Process ABI (§7).

use super::refs::Ref64;
use super::AbiHeader;

/// Phase-1 process modes (§7).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessMode {
    Serial = 1,
    Pure = 2,
    System = 3,
}

/// Process states (§7). A process contains no permanently assigned execution
/// thread.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessState {
    Created = 1,
    Runnable = 2,
    Running = 3,
    Waiting = 4,
    CancelPending = 5,
    Failed = 6,
    Terminated = 7,
    Cancelled = 8,
}

/// Why a supervised process reached a terminal state.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    Completed = 1,
    Failed = 2,
    Cancelled = 3,
}

/// Kernel response when this process fails under a direct supervisor.
/// `Notify` contains the failure at the child; `Escalate` additionally fails
/// the direct supervisor after delivering the child's notice; `Restart`
/// creates a new process identity from a registered entry template and
/// escalates after its bounded retry budget is exhausted.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SupervisionPolicy {
    #[default]
    Notify = 1,
    Escalate = 2,
    Restart = 3,
}

/// Kernel-owned notification delivered to a process when one of its directly
/// supervised children exits. It is deliberately separate from the bounded
/// user mailbox: supervision cannot be lost because ordinary traffic filled
/// the inbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupervisionNotice {
    pub child: Ref64,
    pub replacement: Ref64,
    pub reason: ExitReason,
    pub failure_count: u32,
}

/// Process descriptor (§7).
#[derive(Clone, Debug)]
pub struct ProcessDescriptor {
    pub header: AbiHeader,

    pub id: Ref64,
    pub domain: Ref64,
    pub supervisor: Ref64,
    pub supervision_policy: SupervisionPolicy,
    pub restart_of: Ref64,
    pub restart_attempt: u32,
    pub restart_limit: u32,

    pub state: super::refs::Ref64, // CapRef
    pub inbox: Ref64,
    pub urgent_inbox: Ref64,

    pub active_continuation: Ref64,
    pub waiting_on: Ref64,

    pub status: u32,
    pub process_mode: ProcessMode,
    pub base_priority: u16,

    pub compute_quota: u64,
    pub memory_quota: u64,
    pub deadline_ns: u64,

    pub last_committed_epoch: u32,
    pub failure_count: u32,

    /// Continuations of this process in a live state (`New`, `Runnable`,
    /// `Waiting`, or `Running`), maintained incrementally.
    ///
    /// A process retires when its last continuation does (I3), and the
    /// reference implementation used to answer "is any left?" by scanning the
    /// whole continuation table on every completion. That is O(continuations)
    /// per commit, which a concurrent scheduler cannot afford. The count is
    /// derived state, so I3 checks it against an actual scan.
    pub live_continuations: u32,
}

impl ProcessDescriptor {
    pub fn new(mode: ProcessMode) -> ProcessDescriptor {
        ProcessDescriptor {
            header: AbiHeader::new(2, std::mem::size_of::<ProcessDescriptor>() as u32),
            id: Ref64::NULL,
            domain: Ref64::NULL,
            supervisor: Ref64::NULL,
            supervision_policy: SupervisionPolicy::Notify,
            restart_of: Ref64::NULL,
            restart_attempt: 0,
            restart_limit: 0,
            state: Ref64::NULL,
            inbox: Ref64::NULL,
            urgent_inbox: Ref64::NULL,
            active_continuation: Ref64::NULL,
            waiting_on: Ref64::NULL,
            status: 0,
            process_mode: mode,
            base_priority: 0,
            compute_quota: 0,
            memory_quota: 0,
            deadline_ns: 0,
            last_committed_epoch: 0,
            failure_count: 0,
            live_continuations: 0,
        }
    }
}

impl ContinuationState {
    /// Whether a continuation in this state could still run, and therefore
    /// keeps its process alive (I3).
    pub fn is_live(self) -> bool {
        matches!(
            self,
            ContinuationState::New
                | ContinuationState::Runnable
                | ContinuationState::Waiting
                | ContinuationState::Running
        )
    }
}

use super::continuations::ContinuationState;
