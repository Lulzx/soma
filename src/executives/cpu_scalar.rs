//! CPU scalar executive (§16).
//!
//! Phase 1's only executive. It consumes scalar run classes via a uniform
//! `dispatch` on the run class — the static-dispatch seed of §15's uniform
//! cohort execution. Every yielded continuation already knows its next bin
//! (`next_run_class`), so scheduling needs no arbitrary metadata inspection.
//!
//! A continuation can move between executives only at a continuation boundary;
//! no register state is migrated, and its durable frame already resides in
//! shared memory. The same continuation descriptor, frame layout, capability
//! model, step result, and message semantics are used here as everywhere else.
//!
//! A step takes a [`LaneView`] rather than the kernel (v0.3 §4.10). That is
//! what makes the set of things a step does finite and visible: the view offers
//! fifteen operations, an operation with no lane-local form is a compile error
//! inside a handler, and the remaining work to run lanes concurrently is
//! therefore enumerable rather than an audit.

use crate::abi::Ref64;
use crate::abi::{MessageDescriptor, StateAccess, StepResult};
use crate::compiler::frame::{ByteCursor, Frame};
use crate::compiler::run_classes::{
    ant_class_index, search_class_index, COLONY_AGGREGATE, DEFAULT_MAX_STEPS, EXPAND_RESUME_0,
    EXPAND_RESUME_1, EXPAND_RESUME_2, JOIN_AWAIT, JOIN_RESUME, POLL_ACT, POLL_FUTURE, SEARCH_BRANCH,
    SEARCH_HEURISTIC, WORLD_STEP,
};
use crate::compiler::state_machine_lowering::{
    search_step, ExpandFrame, HeuristicFrame, JoinFrame, SearchFrame,
};
use crate::executives::ant_colony;
use crate::executives::lane::LaneView;
use crate::kernel::{AwaitOutcome, ContinuationSpec, RuntimeError};

/// Run-class identifiers recognized by this executive (mirrors §15's switch).
pub mod run_classes {
    pub use crate::compiler::run_classes::*;
}

/// Uniform dispatch over run classes. In a real SIMD cohort the same switch
/// executes uniformly for every lane because the whole cohort shares one run
/// class, so the branch introduces no intra-cohort divergence (§15).
///
/// The run class is an argument because the epoch loop already decided it:
/// Phase E built this lane's cohort out of it, and Phase F read it again on the
/// host's side of the step. Reading it back out of the continuation table from
/// in here needed the table, and the table was the last ungoverned read a step
/// could reach (v0.3 §4.17, and [`LaneView::frame`] for what it could have led
/// to).
///
/// Eight handlers now take `_cont`. The parameter stays because the signature
/// is what makes this a switch rather than eight unrelated calls, and it is
/// underscored rather than removed so that the count is visible: those eight
/// wanted the continuation only to ask the table about it, and the frame they
/// were asking for arrives another way.
pub fn dispatch(lane: &mut LaneView<'_>, cont: Ref64, process: Ref64, rc: u32) -> StepResult {
    match rc {
        EXPAND_RESUME_0 => expand_resume_0(lane, cont, process),
        EXPAND_RESUME_1 => expand_resume_1(lane, cont, process),
        EXPAND_RESUME_2 => expand_resume_2(lane, cont, process),
        SEARCH_HEURISTIC => heuristic(lane, cont, process),
        JOIN_AWAIT => join_await(lane, cont, process),
        JOIN_RESUME => join_resume(lane, cont, process),
        POLL_FUTURE => poll_future(lane, cont, process),
        POLL_ACT => poll_act(lane, cont, process),
        COLONY_AGGREGATE => ant_colony::colony_aggregate(lane, cont, process),
        WORLD_STEP => ant_colony::world_step(lane, cont, process),
        // The search classes occupy a contiguous block; each is a distinct
        // case of this switch with its own arithmetic (§25.1).
        rc => match search_class_index(rc) {
            Some(index) => search_branch(lane, cont, process, index),
            // The ant behaviours are their own contiguous block. Each is a
            // separate case for the same reason the search classes are: a
            // cohort really can only hold one of them.
            None => match ant_class_index(rc) {
                Some(behaviour) => ant_colony::ant_step(lane, cont, process, behaviour),
                None => StepResult::fault(process, rc),
            },
        },
    }
}

// ---- frame access helpers ------------------------------------------------

// These four used to take the continuation whose frame they wanted and look up
// its frame object in the table. They took `cont` for the whole of that reason
// and every call site passed the running continuation, so the parameter is
// gone: `lane.frame()` is that same object, copied in before the step began.
// The narrowing is the point rather than the tidying — with the table reachable,
// "whose frame" was a question a handler could answer with someone else's
// (v0.3 §4.17).

fn frame_bytes(lane: &mut LaneView<'_>, actor: Ref64) -> Vec<u8> {
    let obj = lane.frame();
    lane
        .object_bytes(actor, obj)
        .map(|b| b.to_vec())
        .unwrap_or_default()
}

fn set_frame_bytes(lane: &mut LaneView<'_>, actor: Ref64, bytes: Vec<u8>) {
    let obj = lane.frame();
    if obj.is_null() {
        return;
    }
    // A frame can change length between steps, so this needs the payload
    // as a `Vec` rather than the slice `object_bytes_mut` now returns.
    if let Ok(buf) = lane.host_payload_mut(actor, obj) {
        *buf = bytes;
    }
}

pub(crate) fn load_frame<F: Frame>(lane: &mut LaneView<'_>, actor: Ref64, fallback: F) -> F {
    let bytes = frame_bytes(lane, actor);
    let mut c = ByteCursor::new(&bytes);
    F::decode(&mut c).unwrap_or(fallback)
}

pub(crate) fn store_frame<F: Frame>(lane: &mut LaneView<'_>, actor: Ref64, frame: &F) {
    let mut bytes = Vec::new();
    frame.encode(&mut bytes);
    set_frame_bytes(lane, actor, bytes);
}

// ---- handlers (§22) -------------------------------------------------------

/// `Expand.resume_0`: receive the request, store it in the frame, spawn the
/// heuristic evaluation, and await its future. Continues at `EXPAND_RESUME_1`.
fn expand_resume_0(lane: &mut LaneView<'_>, cont: Ref64, process: Ref64) -> StepResult {
    match lane.receive_message(process, cont) {
        Ok(Some(msg)) => {
            let request_value = lane.read_u64_object(process, msg.payload).unwrap_or(0);
            let mut frame = ExpandFrame::initial(request_value, msg.sender);

            // Spawn the heuristic and await its future.
            let fut = lane.create_future(process);
            frame.heuristic_future = fut;
            let hframe = HeuristicFrame {
                future: fut,
                input: request_value,
            };
            let mut hb = Vec::new();
            hframe.encode(&mut hb);
            lane
                .create_continuation(
                    process,
                    process,
                    ContinuationSpec::new(
                        StateAccess::ReadOnly,
                        SEARCH_HEURISTIC,
                        0,
                        hb,
                        DEFAULT_MAX_STEPS,
                    ),
                )
                .expect("a process may create its own continuation");

            store_frame(lane, process, &frame);
            match lane.await_future(process, cont, fut, EXPAND_RESUME_1) {
                Ok(AwaitOutcome::Registered) => StepResult::await_on(fut, EXPAND_RESUME_1),
                // The heuristic already resolved, so there is nothing to wait
                // for; go straight to the next resume point.
                Ok(AwaitOutcome::AlreadySettled(_)) => StepResult::yield_next(EXPAND_RESUME_1),
                Err(_) => StepResult::fault(process, EXPAND_RESUME_0),
            }
        }
        // No message yet: `receive_message` registered us as a receiver waiter.
        Ok(None) => StepResult::await_on(process, EXPAND_RESUME_0),
        Err(_) => StepResult::fault(process, EXPAND_RESUME_0),
    }
}

/// `Expand.resume_1`: load the heuristic result and generate a bounded group of
/// moves, then yield to `EXPAND_RESUME_2`.
fn expand_resume_1(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64) -> StepResult {
    let mut frame: ExpandFrame = load_frame(lane, process, ExpandFrame::initial(0, process));
    // A denial faults rather than reading as "not resolved yet": those are
    // different answers, and collapsing them is what the ungoverned read did.
    // A *null* future is neither — the frame simply names no future to look at,
    // so there is nothing to observe and nothing to record.
    if !frame.heuristic_future.is_null() {
        match lane.future_value(process, frame.heuristic_future) {
            Ok(Some(vobj)) => {
                frame.heuristic_result = lane.read_u64_object(process, vobj).unwrap_or(0);
            }
            Ok(None) => {}
            Err(_) => return StepResult::fault(process, EXPAND_RESUME_1),
        }
    }
    // Bounded move generation: legal_moves(node) in the §22 sketch.
    frame.moves = (1..=3)
        .map(|i| frame.request_value.wrapping_mul(i as u64))
        .collect();
    store_frame(lane, process, &frame);
    StepResult::yield_next(EXPAND_RESUME_2)
}

/// `Expand.resume_2`: spawn a child process per move, send the reply, complete.
/// This resume point can be re-entered: a full reply mailbox parks it and it
/// runs again once capacity frees. Child creation and payload allocation are
/// therefore recorded in the frame as they happen, so re-entry resumes where it
/// left off rather than repeating side effects (§8).
fn expand_resume_2(lane: &mut LaneView<'_>, cont: Ref64, process: Ref64) -> StepResult {
    let mut frame: ExpandFrame = load_frame(lane, process, ExpandFrame::initial(0, process));

    while (frame.move_index as usize) < frame.moves.len() {
        let m = frame.moves[frame.move_index as usize];
        // A full domain faults this node rather than aborting the machine. The
        // frame is stored first: the children already spawned happened, and a
        // supervisor restarting this continuation must not spawn them twice
        // (§8).
        let Ok(child) = lane.create_process(process, crate::abi::ProcessMode::Serial) else {
            store_frame(lane, process, &frame);
            return StepResult::fault(process, 0);
        };
        let cframe = SearchFrame::leaf(m, 0);
        let mut cb = Vec::new();
        cframe.encode(&mut cb);
        lane
            .create_continuation(
                process,
                child,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    SEARCH_BRANCH,
                    0,
                    cb,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("the creator holds WRITE on the child process");
        frame.move_index += 1;
    }

    if frame.reply_payload.is_null() {
        frame.reply_payload = lane.create_object(
            process,
            crate::abi::ObjectKind::MessagePayload,
            frame.heuristic_result.to_le_bytes().to_vec(),
        );
    }
    store_frame(lane, process, &frame);

    match lane.enqueue_message(process, frame.reply_receiver, frame.reply_payload, cont) {
        Ok(()) => StepResult::complete(),
        Err(RuntimeError::MailboxFull) => {
            StepResult::await_on(frame.reply_receiver, EXPAND_RESUME_2)
        }
        Err(_) => StepResult::fault(process, EXPAND_RESUME_2),
    }
}

/// `SEARCH_HEURISTIC`: compute a deterministic result, publish it into the
/// future (single-assignment, §12), and complete.
fn heuristic(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64) -> StepResult {
    let hf: HeuristicFrame = load_frame(
        lane,
        process,
        HeuristicFrame {
            future: Ref64::NULL,
            input: 0,
        },
    );
    let result = hf.input.wrapping_mul(2).wrapping_add(1);
    let vobj = lane.create_object(
        process,
        crate::abi::ObjectKind::FutureValue,
        result.to_le_bytes().to_vec(),
    );
    match lane.resolve_future(process, hf.future, vobj) {
        Ok(()) => StepResult::complete(),
        Err(_) => StepResult::fault(process, SEARCH_HEURISTIC),
    }
}

/// `JOIN_AWAIT` (v0.3 §4.15): await a future this continuation did not create.
///
/// The one handler whose await can go either way. `expand_resume_0` creates the
/// future in the step it awaits, so no resolver can have run yet and the
/// `AlreadySettled` arm is dead there; here the future is named by the frame and
/// somebody else resolves it, so which arm runs is decided by whether the
/// resolving lane went first.
fn join_await(lane: &mut LaneView<'_>, cont: Ref64, process: Ref64) -> StepResult {
    let frame: JoinFrame = load_frame(
        lane,
        process,
        JoinFrame {
            future: Ref64::NULL,
            observed: Ref64::NULL,
        },
    );
    match lane.await_future(process, cont, frame.future, JOIN_RESUME) {
        Ok(AwaitOutcome::Registered) => StepResult::await_on(frame.future, JOIN_RESUME),
        // Already published: nothing will wake us, because `resolve_future`
        // drained the waiter list before we asked to be on it.
        Ok(AwaitOutcome::AlreadySettled(_)) => StepResult::yield_next(JOIN_RESUME),
        Err(_) => StepResult::fault(process, JOIN_AWAIT),
    }
}

/// `JOIN_RESUME`: read the value the future took, by either route.
fn join_resume(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64) -> StepResult {
    let mut frame: JoinFrame = load_frame(
        lane,
        process,
        JoinFrame {
            future: Ref64::NULL,
            observed: Ref64::NULL,
        },
    );
    // `future_value` is the second decision §4.14 could not reach: it reads a
    // resolution with no event of its own. This handler reaches it for the same
    // reason it reaches `AlreadySettled` — the future is somebody else's.
    if !frame.future.is_null() {
        match lane.future_value(process, frame.future) {
            Ok(observed) => frame.observed = observed.unwrap_or(Ref64::NULL),
            Err(_) => return StepResult::fault(process, JOIN_RESUME),
        }
    }
    store_frame(lane, process, &frame);
    StepResult::complete()
}

/// `POLL_FUTURE` (v0.3 §4.16): read a future's value without awaiting it.
fn poll_future(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64) -> StepResult {
    let mut frame: JoinFrame = load_frame(
        lane,
        process,
        JoinFrame {
            future: Ref64::NULL,
            observed: Ref64::NULL,
        },
    );
    if !frame.future.is_null() {
        match lane.future_value(process, frame.future) {
            Ok(observed) => frame.observed = observed.unwrap_or(Ref64::NULL),
            Err(_) => return StepResult::fault(process, POLL_FUTURE),
        }
    }
    store_frame(lane, process, &frame);
    StepResult::yield_next(POLL_ACT)
}

/// `POLL_ACT`: act on what the poll saw, in a later epoch.
///
/// Without this the divergence a poll causes stays inside a frame, and a frame
/// is not observable behaviour. With it, what one epoch's lane order decided
/// becomes a message another epoch either sends or does not.
fn poll_act(lane: &mut LaneView<'_>, cont: Ref64, process: Ref64) -> StepResult {
    let frame: JoinFrame = load_frame(
        lane,
        process,
        JoinFrame {
            future: Ref64::NULL,
            observed: Ref64::NULL,
        },
    );
    if frame.observed.is_null() {
        return StepResult::complete();
    }
    let payload = lane.create_object(
        process,
        crate::abi::ObjectKind::MessagePayload,
        frame.observed.slot.to_le_bytes().to_vec(),
    );
    // To its own mailbox: what matters is that the send is in the trace and
    // happens in one run and not the other, not who reads it.
    match lane.enqueue_message(process, process, payload, cont) {
        Ok(()) => StepResult::complete(),
        Err(_) => StepResult::fault(process, POLL_ACT),
    }
}

/// One synthetic branching-search node of class `index` (§25.1). Reads the
/// frame, does a bounded amount of deterministic arithmetic ("state duration"),
/// then either spawns `branching` child processes (internal node) or completes
/// (leaf).
///
/// `index` selects the arithmetic, so the search classes are distinct code
/// paths rather than aliases of one handler — a cohort really can only contain
/// one of them.
fn search_branch(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64, index: u32) -> StepResult {
    let mut sf: SearchFrame = load_frame(
        lane,
        process,
        SearchFrame {
            value: 0,
            depth: 0,
            branching: 0,
            work_iters: 0,
            class_count: 1,
        },
    );

    sf.value = search_step(sf.value, sf.work_iters, index);

    if sf.depth > 0 {
        for i in 0..sf.branching {
            // As in `expand_resume_2`: a full domain is a fault, not an abort.
            let Ok(child) = lane.create_process(process, crate::abi::ProcessMode::Serial) else {
                store_frame(lane, process, &sf);
                return StepResult::fault(process, 0);
            };
            let cframe = SearchFrame {
                value: sf.value.wrapping_add(i as u64),
                depth: sf.depth - 1,
                branching: sf.branching,
                work_iters: sf.work_iters,
                class_count: sf.class_count,
            };
            let run_class = cframe.run_class();
            let mut cb = Vec::new();
            cframe.encode(&mut cb);
            lane
                .create_continuation(
                    process,
                    child,
                    ContinuationSpec::new(
                        StateAccess::ReadOnly,
                        run_class,
                        0,
                        cb,
                        DEFAULT_MAX_STEPS,
                    ),
                )
                .expect("the creator holds WRITE on the child process");
        }
        // Spawn with no continuation: the commit phase terminates this node.
        StepResult::spawn(process, 0)
    } else {
        store_frame(lane, process, &sf);
        StepResult::complete()
    }
}

/// Result type alias to keep callers explicit about what a message receive
/// yields (used by tests / the harness to avoid guessing payload layouts).
pub type ReceivedMessage = MessageDescriptor;
