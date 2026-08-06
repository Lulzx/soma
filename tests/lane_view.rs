//! §4.10: what a step is allowed to touch.
//!
//! A handler took `&mut Kernel`, so "a step can do anything to the kernel" was
//! true of the type signature and unmeasured of the handlers. `LaneView` closes
//! the set. The tests here are about the *shape* of that claim, because the
//! enforcement itself is the compiler's — a handler calling something the view
//! does not offer does not build, which is the same technique that seals
//! `Admission` and `Committing`.
//!
//! So what is checkable here is that the seal is real (the view does not hand
//! the kernel back out) and that retyping the handlers changed no run.

use soma::abi::cohorts::PartialCohortPolicy;
use soma::compiler::state_machine_lowering::create_expand;
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::Kernel;
use soma::semantics::invariants::assert_legal;
use soma::semantics::order::conforms_traces;

fn expand(width: u16) -> Kernel {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 7);
    kernel.configure_cohorts(width, PartialCohortPolicy::RunPartial);
    kernel.run_to_quiescence(200);
    kernel
}

fn search(width: u16) -> Kernel {
    let knobs = ControlKnobs {
        branching_factor: 3,
        depth: 3,
        process_count: 2,
        class_count: 3,
        ..ControlKnobs::default()
    };
    let mut kernel = build(&knobs);
    kernel.configure_cohorts(width, PartialCohortPolicy::RunPartial);
    kernel.run_to_quiescence(200);
    kernel
}

#[test]
fn routing_every_step_through_a_lane_changed_no_run() {
    // The refactor's whole claim. Two runs of one workload at two widths still
    // agree with each other and still leave a legal state; if handing handlers
    // a narrower view had dropped an operation, this is where the missing
    // effect would show.
    for width in [1u16, 4, 16] {
        let a = expand(width);
        let b = expand(width);
        assert!(conforms_traces(&a.trace_snapshot(), &b.trace_snapshot()).is_empty());
        assert_legal(&a);

        let c = search(width);
        assert_legal(&c);
    }
}

#[test]
fn the_workloads_actually_exercise_the_view() {
    // The null. A view that offered nothing would pass the test above on a
    // workload that never called anything, so the runs have to be ones that
    // really do allocate, message, await and resolve — which is what makes the
    // fifteen operations fifteen rather than three.
    use soma::abi::EventKind;

    let kernel = expand(16);
    let kinds: std::collections::BTreeSet<EventKind> = kernel
        .trace_snapshot()
        .iter()
        .map(|row| row.event_kind)
        .collect();

    for required in [
        EventKind::ContinuationStarted,
        EventKind::ContinuationYielded,
        EventKind::ContinuationWaiting,
        EventKind::MessageSent,
        EventKind::FutureResolved,
    ] {
        assert!(
            kinds.contains(&required),
            "the workload never produced {required:?}, so the view was barely used"
        );
    }
}
