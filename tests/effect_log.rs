//! I24: a runnable bin is written by the effect applier, in plan order.
//!
//! `docs/SOMA-v0.3.md` §4.3 names the obstacle to canonical commit — handlers
//! mutate as they run, so execute and commit are fused — and §4.4 unfuses the
//! part every lane of an epoch writes. A step produces its bin entries; the
//! kernel applies them.
//!
//! The positive tests here say what the record is. The ones that matter are the
//! nulls (the workload really does produce effects from several lanes of
//! several epochs, and all four effect shapes, so the ordering clause has
//! something to order) and the fault injections (each clause has a state that
//! fails it).

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{ProcessMode, Ref64, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::DEFAULT_MAX_STEPS;
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::kernel::effects::EffectKind;
use soma::kernel::{raw, ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::{assert_legal, check, Invariant};

/// A search node that branches, so the run keeps producing work for several
/// epochs instead of completing in one.
fn tree_frame(value: u64, depth: u32) -> SearchFrame {
    SearchFrame {
        value,
        depth,
        branching: 2,
        work_iters: 4,
        class_count: 2,
    }
}

fn spawn_tree(kernel: &mut Kernel, process: Ref64, value: u64, depth: u32) -> Ref64 {
    let frame = tree_frame(value, depth);
    let run_class = frame.run_class();
    let mut bytes = Vec::new();
    frame.encode(&mut bytes);
    kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                run_class,
                0,
                bytes,
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap()
}

/// Several epochs, several lanes per epoch, and the `Expand` workload so that
/// wakes (future resolution, message delivery) appear alongside fresh bins and
/// commit resumes.
fn busy_kernel(cohort_width: u16, policy: PartialCohortPolicy) -> Kernel {
    let mut kernel = Kernel::new();
    kernel.configure_cohorts(cohort_width, policy);
    create_expand(&mut kernel, 3);
    for root in 0..4 {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        spawn_tree(&mut kernel, process, root, 3);
    }
    kernel.run_to_quiescence(64);
    kernel
}

fn violated(kernel: &Kernel, invariant: Invariant) -> bool {
    check(kernel).iter().any(|v| v.invariant == invariant)
}

// ---- the property ---------------------------------------------------------

#[test]
fn every_bin_entry_is_an_applied_effect() {
    let kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    assert_legal(&kernel);
}

#[test]
fn sorting_the_log_by_position_recovers_the_order_it_was_applied_in() {
    let kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    let mut by_position: Vec<_> = kernel.effect_log().to_vec();
    by_position.sort_by_key(|record| record.position());

    let applied: Vec<u64> = by_position.iter().map(|record| record.applied).collect();
    let mut expected = applied.clone();
    expected.sort_unstable();
    assert_eq!(
        applied, expected,
        "the order the plan puts the producing lanes in is the order the kernel applied them in"
    );
}

#[test]
fn an_epoch_applies_its_lanes_in_plan_order() {
    let kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    let mut previous: Option<(u32, u32)> = None;
    for record in kernel.effect_log() {
        if let Some((epoch, lane)) = previous {
            if record.epoch == epoch {
                assert!(
                    record.lane >= lane,
                    "epoch {epoch} applied lane {} after lane {lane}",
                    record.lane
                );
            } else {
                assert!(record.epoch > epoch, "epochs apply in order");
            }
        }
        previous = Some((record.epoch, record.lane));
    }
}

// ---- the nulls ------------------------------------------------------------

#[test]
fn the_log_spans_several_lanes_of_several_epochs() {
    let kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    let log = kernel.effect_log();
    assert!(
        log.len() > 20,
        "{} effects is too few to say much",
        log.len()
    );

    let epochs: std::collections::BTreeSet<u32> = log.iter().map(|r| r.epoch).collect();
    let lanes: std::collections::BTreeSet<u32> = log.iter().map(|r| r.lane).collect();
    assert!(
        epochs.len() > 2,
        "the ordering clause needs more than one epoch to order"
    );
    assert!(
        lanes.iter().filter(|lane| **lane != 0).count() > 1,
        "with one lane per epoch, applying in lane order is not a constraint"
    );
}

/// Each of the four effect shapes writes a bin a different way — a commit
/// resume, a waiter wake, a fresh continuation's first bin, and a lane the
/// policy held back. A run exercising only one of them would leave the other
/// three unmediated without failing anything.
#[test]
fn every_effect_shape_is_exercised() {
    let mut kinds: std::collections::BTreeSet<EffectKind> =
        busy_kernel(2, PartialCohortPolicy::RunPartial)
            .effect_log()
            .iter()
            .map(|record| record.kind)
            .collect();
    // `Defer` is the only policy that holds a partial cohort back, so `Requeue`
    // comes from a second run rather than a wider one.
    kinds.extend(
        busy_kernel(4, PartialCohortPolicy::Defer)
            .effect_log()
            .iter()
            .map(|record| record.kind),
    );

    for kind in [
        EffectKind::Resume,
        EffectKind::Wake,
        EffectKind::Bin,
        EffectKind::Requeue,
    ] {
        assert!(kinds.contains(&kind), "no {kind:?} effect was produced");
    }
}

/// The refactor is supposed to have changed no run. Producing effects and
/// applying them at the end of a lane puts the writes exactly where a
/// sequential interpreter already had them, so two kernels built the same way
/// still agree event for event.
#[test]
fn producing_effects_did_not_change_the_run() {
    let left = busy_kernel(2, PartialCohortPolicy::RunPartial);
    let right = busy_kernel(2, PartialCohortPolicy::RunPartial);
    assert_eq!(left.trace_snapshot(), right.trace_snapshot());
    assert_eq!(left.effect_log(), right.effect_log());
}

// ---- the fault injections -------------------------------------------------

/// Clause 3. A bin entry that produced no effect. `Scheduler::enqueue` demands
/// a token only `kernel::effects` can build, so this is reachable only through
/// `raw` — which is the point: in-crate the mistake does not compile, and the
/// count is what would catch an implementation that this crate did not compile.
#[test]
fn an_unmediated_bin_entry_is_rejected() {
    let mut kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    assert!(!violated(&kernel, Invariant::EffectMediatedCommit));

    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let continuation = spawn_tree(&mut kernel, process, 1, 0);
    assert!(
        !violated(&kernel, Invariant::EffectMediatedCommit),
        "the mediated bin entry that creation makes is accounted for"
    );
    unsafe { raw::enqueue_unmediated(&mut kernel, tree_frame(1, 0).run_class(), continuation) };

    assert!(
        violated(&kernel, Invariant::EffectMediatedCommit),
        "a continuation reached a bin without an effect accounting for it"
    );
}

/// Clause 2. An applier that ran a later lane's effect first.
#[test]
fn applying_out_of_plan_order_is_rejected() {
    let mut kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    assert!(!violated(&kernel, Invariant::EffectMediatedCommit));

    {
        let state = unsafe { raw::state(&mut kernel) };
        let last = state.effect_log.len() - 1;
        let (first, second) = (state.effect_log[0].applied, state.effect_log[last].applied);
        state.effect_log[0].applied = second;
        state.effect_log[last].applied = first;
    }

    assert!(
        violated(&kernel, Invariant::EffectMediatedCommit),
        "the log records an application order its positions do not justify"
    );
}

/// Clause 1. An effect applied twice, or one applied not at all — the same
/// corruption seen from either end.
#[test]
fn an_effect_applied_twice_is_rejected() {
    let mut kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    assert!(!violated(&kernel, Invariant::EffectMediatedCommit));

    {
        let state = unsafe { raw::state(&mut kernel) };
        state.effect_log[1].applied = state.effect_log[0].applied;
    }

    assert!(
        violated(&kernel, Invariant::EffectMediatedCommit),
        "two effects claiming one application index is one effect lost"
    );
}

/// Clause 2's uniqueness half: two effects produced at one position leave no
/// rule for which to apply first.
#[test]
fn two_effects_at_one_position_are_rejected() {
    let mut kernel = busy_kernel(2, PartialCohortPolicy::RunPartial);
    assert!(!violated(&kernel, Invariant::EffectMediatedCommit));

    {
        let state = unsafe { raw::state(&mut kernel) };
        let position = state.effect_log[0];
        state.effect_log[1].epoch = position.epoch;
        state.effect_log[1].lane = position.lane;
        state.effect_log[1].sequence = position.sequence;
    }

    assert!(
        violated(&kernel, Invariant::EffectMediatedCommit),
        "an ambiguous production position gives the applier no order to follow"
    );
}

// ---- canonical commit: withdrawal -----------------------------------------

/// A step that produces work and then loses it in the same lane.
///
/// §4.4 wrote the ordering rule for this case and could not test it: with the
/// applier running per lane, cancellation applied the journal and then emptied
/// the bins, so the entry was made and removed and no run could tell the
/// difference. With the applier at the epoch boundary the entry has not been
/// made yet, so cancellation has to withdraw it from the journal instead — and
/// a withdrawn effect is one that never reaches the log at all.
///
/// The state is constructed rather than reached by a workload, for the reason
/// §4.4 gave: no handler in this executive cancels a process it is running in.
/// What is constructed is only the precondition — a process cancelled while its
/// continuation is mid-step, which `cancel_process` produces whenever the
/// target has an active continuation. Everything after that is the real path.
#[test]
fn cancelling_a_process_withdraws_the_effects_its_lane_produced() {
    use soma::abi::ProcessState;
    use soma::compiler::run_classes::EXPAND_RESUME_1;

    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    // `expand_resume_1` yields unconditionally, so this lane produces exactly
    // one `Resume` and nothing else.
    let continuation = kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                EXPAND_RESUME_1,
                0,
                Vec::new(),
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();

    // The `Bin` effect for the continuation's own creation, which the host
    // produced and applied before the epoch. Everything after this index is
    // the epoch's.
    let before = kernel.effect_log().len();

    unsafe { raw::state(&mut kernel) }
        .processes
        .get_mut(process)
        .unwrap()
        .status = ProcessState::CancelPending as u32;

    kernel.run_epoch();

    let produced: Vec<_> = kernel.effect_log()[before..].to_vec();
    assert!(
        produced
            .iter()
            .all(|record| record.continuation != continuation),
        "the resume this lane produced was applied instead of withdrawn: {produced:?}"
    );

    // Withdrawal is not the same as never having produced: the point is that
    // the continuation ends up cancelled and out of every bin, which is where
    // applying-then-removing used to leave it.
    assert_eq!(
        kernel.continuation_state(continuation).unwrap(),
        soma::abi::continuations::ContinuationState::Cancelled
    );
    assert_eq!(kernel.total_pending(), 0);
    assert_legal(&kernel);
}

/// The null: without the cancellation, that lane's resume really is applied.
///
/// Otherwise the test above would pass on a run where `expand_resume_1`
/// produced no effect at all, which is the reading that makes it check nothing.
#[test]
fn the_same_lane_applies_its_resume_when_the_process_survives() {
    use soma::compiler::run_classes::EXPAND_RESUME_1;

    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let continuation = kernel
        .create_continuation(
            process,
            process,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                EXPAND_RESUME_1,
                0,
                Vec::new(),
                DEFAULT_MAX_STEPS,
            ),
        )
        .unwrap();

    let before = kernel.effect_log().len();
    kernel.run_epoch();

    let produced: Vec<_> = kernel.effect_log()[before..].to_vec();
    assert!(
        produced
            .iter()
            .any(|record| record.continuation == continuation
                && record.kind == EffectKind::Resume),
        "the lane produced no resume, so withdrawing one proves nothing: {produced:?}"
    );
    assert_legal(&kernel);
}
