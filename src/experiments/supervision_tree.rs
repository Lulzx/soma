//! Controlled actor-tree validation for supervision semantics.
//!
//! The tree has a root, two branch supervisors, and one worker under each
//! branch. The left worker may fault; the right worker always completes. This
//! supplies both an unaffected sibling branch and a fault-free control while
//! comparing notification-only containment with automatic escalation.

use crate::abi::{EventKind, ProcessMode, ProcessState, Ref64, StateAccess, SupervisionPolicy};
use crate::compiler::frame::Frame;
use crate::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use crate::compiler::state_machine_lowering::SearchFrame;
use crate::kernel::{ContinuationSpec, Kernel, TraceSnapshotRow, SYSTEM_PRINCIPAL};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeKnobs {
    pub worker_policy: SupervisionPolicy,
    pub inject_left_failure: bool,
}

impl Default for TreeKnobs {
    fn default() -> Self {
        Self {
            worker_policy: SupervisionPolicy::Notify,
            inject_left_failure: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeOutcome {
    pub root: ProcessState,
    pub left_branch: ProcessState,
    pub right_branch: ProcessState,
    pub left_worker: ProcessState,
    pub right_worker: ProcessState,
    pub root_notices: usize,
    pub left_branch_notices: usize,
    pub right_branch_notices: usize,
    pub restarts: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeRun {
    pub outcome: TreeOutcome,
    pub trace: Vec<TraceSnapshotRow>,
    pub legal: bool,
}

fn worker_spec(budget: u32) -> ContinuationSpec {
    let mut frame = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut frame);
    ContinuationSpec::new(StateAccess::ReadOnly, SEARCH_BRANCH, 0, frame, budget)
}

fn worker(kernel: &mut Kernel, supervisor: Ref64, policy: SupervisionPolicy, budget: u32) -> Ref64 {
    if policy == SupervisionPolicy::Restart {
        return kernel
            .create_restartable_process(
                supervisor,
                supervisor,
                ProcessMode::Serial,
                1,
                worker_spec(budget),
            )
            .expect("live branch may create restartable worker");
    }
    let process = kernel
        .create_supervised_process_with_policy(supervisor, supervisor, ProcessMode::Serial, policy)
        .expect("live branch may create worker");
    kernel
        .create_continuation(process, process, worker_spec(budget))
        .expect("a worker may create its own continuation");
    process
}

pub fn run(knobs: &TreeKnobs) -> TreeRun {
    let mut kernel = Kernel::new();
    let root = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let left_branch = kernel
        .create_supervised_process(root, root, ProcessMode::Serial)
        .expect("live root may create branch");
    let right_branch = kernel
        .create_supervised_process(root, root, ProcessMode::Serial)
        .expect("live root may create branch");
    let left_worker = worker(
        &mut kernel,
        left_branch,
        knobs.worker_policy,
        if knobs.inject_left_failure {
            0
        } else {
            DEFAULT_MAX_STEPS
        },
    );
    let right_worker = worker(
        &mut kernel,
        right_branch,
        knobs.worker_policy,
        DEFAULT_MAX_STEPS,
    );
    kernel.run_epoch();

    let outcome = TreeOutcome {
        root: kernel.process_state(root).expect("root resolves"),
        left_branch: kernel
            .process_state(left_branch)
            .expect("left branch resolves"),
        right_branch: kernel
            .process_state(right_branch)
            .expect("right branch resolves"),
        left_worker: kernel
            .process_state(left_worker)
            .expect("left worker resolves"),
        right_worker: kernel
            .process_state(right_worker)
            .expect("right worker resolves"),
        root_notices: kernel.pending_supervision_notices(root),
        left_branch_notices: kernel.pending_supervision_notices(left_branch),
        right_branch_notices: kernel.pending_supervision_notices(right_branch),
        restarts: kernel
            .trace_events()
            .iter()
            .filter(|event| event.event_kind == EventKind::ProcessRestarted)
            .count(),
    };
    TreeRun {
        outcome,
        trace: kernel.trace_snapshot(),
        legal: crate::semantics::invariants::check(&kernel).is_empty(),
    }
}

pub fn report() -> String {
    let cases = [
        (
            "control",
            TreeKnobs {
                worker_policy: SupervisionPolicy::Notify,
                inject_left_failure: false,
            },
        ),
        (
            "notify",
            TreeKnobs {
                worker_policy: SupervisionPolicy::Notify,
                inject_left_failure: true,
            },
        ),
        (
            "escalate",
            TreeKnobs {
                worker_policy: SupervisionPolicy::Escalate,
                inject_left_failure: true,
            },
        ),
        (
            "restart",
            TreeKnobs {
                worker_policy: SupervisionPolicy::Restart,
                inject_left_failure: true,
            },
        ),
    ];
    let mut output = String::from(
        "case       left-worker left-branch right-worker right-branch root root-notices restarts\n",
    );
    for (name, knobs) in cases {
        let outcome = run(&knobs).outcome;
        output.push_str(&format!(
            "{name:<10} {:?} {:?} {:?} {:?} {:?} {} {}\n",
            outcome.left_worker,
            outcome.left_branch,
            outcome.right_worker,
            outcome.right_branch,
            outcome.root,
            outcome.root_notices,
            outcome.restarts,
        ));
    }
    output
}
