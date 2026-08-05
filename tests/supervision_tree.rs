use soma::abi::{ProcessState, SupervisionPolicy};
use soma::experiments::supervision_tree::{run, TreeKnobs};

#[test]
fn fault_free_control_does_not_escalate() {
    for worker_policy in [
        SupervisionPolicy::Notify,
        SupervisionPolicy::Escalate,
        SupervisionPolicy::Restart,
    ] {
        let run = run(&TreeKnobs {
            worker_policy,
            inject_left_failure: false,
        });
        assert!(run.legal);
        assert_eq!(run.outcome.left_worker, ProcessState::Terminated);
        assert_eq!(run.outcome.right_worker, ProcessState::Terminated);
        assert_eq!(run.outcome.left_branch, ProcessState::Created);
        assert_eq!(run.outcome.right_branch, ProcessState::Created);
        assert_eq!(run.outcome.root, ProcessState::Created);
        assert_eq!(run.outcome.root_notices, 0);
    }
}

#[test]
fn notification_only_contains_the_worker_failure() {
    let run = run(&TreeKnobs {
        worker_policy: SupervisionPolicy::Notify,
        inject_left_failure: true,
    });
    assert!(run.legal);
    assert_eq!(run.outcome.left_worker, ProcessState::Failed);
    assert_eq!(run.outcome.left_branch, ProcessState::Created);
    assert_eq!(run.outcome.left_branch_notices, 1);
    assert_eq!(run.outcome.root_notices, 0);
    assert_eq!(run.outcome.restarts, 0);
}

#[test]
fn escalation_propagates_one_level_and_preserves_the_sibling_branch() {
    let run = run(&TreeKnobs {
        worker_policy: SupervisionPolicy::Escalate,
        inject_left_failure: true,
    });
    assert!(run.legal);
    assert_eq!(run.outcome.left_worker, ProcessState::Failed);
    assert_eq!(run.outcome.left_branch, ProcessState::Failed);
    assert_eq!(run.outcome.root, ProcessState::Created);
    assert_eq!(run.outcome.root_notices, 1);
    assert_eq!(run.outcome.right_worker, ProcessState::Terminated);
    assert_eq!(run.outcome.right_branch, ProcessState::Created);
    assert_eq!(run.outcome.restarts, 0);
}

#[test]
fn restart_replaces_the_failed_worker_without_failing_its_branch() {
    let run = run(&TreeKnobs {
        worker_policy: SupervisionPolicy::Restart,
        inject_left_failure: true,
    });
    assert!(run.legal);
    assert_eq!(run.outcome.left_worker, ProcessState::Failed);
    assert_eq!(run.outcome.left_branch, ProcessState::Created);
    assert_eq!(run.outcome.root, ProcessState::Created);
    assert_eq!(run.outcome.root_notices, 0);
    assert_eq!(run.outcome.restarts, 1);
    assert_eq!(run.outcome.right_worker, ProcessState::Terminated);
    assert_eq!(run.outcome.right_branch, ProcessState::Created);
}

#[test]
fn supervision_tree_runs_are_deterministic() {
    let knobs = TreeKnobs {
        worker_policy: SupervisionPolicy::Escalate,
        inject_left_failure: true,
    };
    assert_eq!(run(&knobs), run(&knobs));
}
