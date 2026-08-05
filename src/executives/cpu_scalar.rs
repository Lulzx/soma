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

use crate::abi::Ref64;
use crate::abi::{MessageDescriptor, StepResult};
use crate::compiler::frame::{ByteCursor, Frame};
use crate::compiler::run_classes::{
    DEFAULT_MAX_STEPS, EXPAND_RESUME_0, EXPAND_RESUME_1, EXPAND_RESUME_2, SEARCH_BRANCH,
    SEARCH_HEURISTIC,
};
use crate::compiler::state_machine_lowering::{
    ExpandFrame, HeuristicFrame, SearchFrame,
};
use crate::kernel::{Kernel, RuntimeError};

/// Run-class identifiers recognized by this executive (mirrors §15's switch).
pub mod run_classes {
    pub use crate::compiler::run_classes::*;
}

/// Uniform dispatch over run classes. In a real SIMD cohort the same switch
/// executes uniformly for every lane because the whole cohort shares one run
/// class, so the branch introduces no intra-cohort divergence (§15).
pub fn dispatch(kernel: &mut Kernel, cont: Ref64, process: Ref64) -> StepResult {
    let rc = kernel
        .continuations
        .get(cont)
        .map(|c| c.run_class)
        .unwrap_or(0);
    match rc {
        EXPAND_RESUME_0 => expand_resume_0(kernel, cont, process),
        EXPAND_RESUME_1 => expand_resume_1(kernel, cont, process),
        EXPAND_RESUME_2 => expand_resume_2(kernel, cont, process),
        SEARCH_HEURISTIC => heuristic(kernel, cont, process),
        SEARCH_BRANCH => search_branch(kernel, cont, process),
        _ => StepResult::fault(process, rc),
    }
}

// ---- frame access helpers ------------------------------------------------

fn frame_bytes(kernel: &Kernel, cont: Ref64) -> Vec<u8> {
    let obj = kernel
        .continuations
        .get(cont)
        .map(|c| c.frame)
        .unwrap_or(Ref64::NULL);
    kernel.object_bytes(obj).map(|b| b.to_vec()).unwrap_or_default()
}

fn set_frame_bytes(kernel: &mut Kernel, cont: Ref64, bytes: Vec<u8>) {
    if let Ok(obj) = kernel.continuations.get(cont).map(|c| c.frame) {
        if let Ok(buf) = kernel.object_bytes_mut(obj) {
            *buf = bytes;
        }
    }
}

fn load_frame<F: Frame>(kernel: &Kernel, cont: Ref64, fallback: F) -> F {
    let bytes = frame_bytes(kernel, cont);
    let mut c = ByteCursor::new(&bytes);
    F::decode(&mut c).unwrap_or(fallback)
}

fn store_frame<F: Frame>(kernel: &mut Kernel, cont: Ref64, frame: &F) {
    let mut bytes = Vec::new();
    frame.encode(&mut bytes);
    set_frame_bytes(kernel, cont, bytes);
}

// ---- handlers (§22) -------------------------------------------------------

/// `Expand.resume_0`: receive the request, store it in the frame, spawn the
/// heuristic evaluation, and await its future. Continues at `EXPAND_RESUME_1`.
fn expand_resume_0(kernel: &mut Kernel, cont: Ref64, process: Ref64) -> StepResult {
    match kernel.receive_message(cont) {
        Ok(Some(msg)) => {
            let request_value = kernel.read_u64_object(msg.payload).unwrap_or(0);
            let mut frame = ExpandFrame::initial(request_value, msg.sender);

            // Spawn the heuristic and await its future.
            let fut = kernel.create_future();
            frame.heuristic_future = fut;
            let hframe = HeuristicFrame {
                future: fut,
                input: request_value,
            };
            let mut hb = Vec::new();
            hframe.encode(&mut hb);
            kernel.create_continuation(process, SEARCH_HEURISTIC, 0, hb, DEFAULT_MAX_STEPS);

            store_frame(kernel, cont, &frame);
            let _ = kernel.await_future(cont, fut, EXPAND_RESUME_1);
            StepResult::await_on(fut, EXPAND_RESUME_1)
        }
        // No message yet: `receive_message` registered us as a receiver waiter.
        Ok(None) => StepResult::await_on(process, EXPAND_RESUME_0),
        Err(_) => StepResult::fault(process, EXPAND_RESUME_0),
    }
}

/// `Expand.resume_1`: load the heuristic result and generate a bounded group of
/// moves, then yield to `EXPAND_RESUME_2`.
fn expand_resume_1(kernel: &mut Kernel, cont: Ref64, process: Ref64) -> StepResult {
    let mut frame: ExpandFrame = load_frame(kernel, cont, ExpandFrame::initial(0, process));
    if let Some(vobj) = kernel.future_value(frame.heuristic_future) {
        frame.heuristic_result = kernel.read_u64_object(vobj).unwrap_or(0);
    }
    // Bounded move generation: legal_moves(node) in the §22 sketch.
    frame.moves = (1..=3)
        .map(|i| frame.request_value.wrapping_mul(i as u64))
        .collect();
    store_frame(kernel, cont, &frame);
    StepResult::yield_next(EXPAND_RESUME_2)
}

/// `Expand.resume_2`: spawn a child process per move, send the reply, complete.
fn expand_resume_2(kernel: &mut Kernel, cont: Ref64, process: Ref64) -> StepResult {
    let frame: ExpandFrame = load_frame(kernel, cont, ExpandFrame::initial(0, process));

    for m in &frame.moves {
        let child = kernel.create_process(crate::abi::ProcessMode::Serial);
        let cframe = SearchFrame::leaf(*m, 0);
        let mut cb = Vec::new();
        cframe.encode(&mut cb);
        kernel.create_continuation(child, SEARCH_BRANCH, 0, cb, DEFAULT_MAX_STEPS);
    }

    let reply_payload = kernel.create_object(
        crate::abi::ObjectKind::MessagePayload,
        frame.heuristic_result.to_le_bytes().to_vec(),
    );
    match kernel.enqueue_message(process, frame.reply_receiver, reply_payload, cont) {
        Ok(()) => StepResult::complete(),
        Err(RuntimeError::MailboxFull) => StepResult::await_on(frame.reply_receiver, EXPAND_RESUME_2),
        Err(_) => StepResult::fault(process, EXPAND_RESUME_2),
    }
}

/// `SEARCH_HEURISTIC`: compute a deterministic result, publish it into the
/// future (single-assignment, §12), and complete.
fn heuristic(kernel: &mut Kernel, cont: Ref64, process: Ref64) -> StepResult {
    let hf: HeuristicFrame = load_frame(
        kernel,
        cont,
        HeuristicFrame {
            future: Ref64::NULL,
            input: 0,
        },
    );
    let result = hf.input.wrapping_mul(2).wrapping_add(1);
    let vobj = kernel.create_object(
        crate::abi::ObjectKind::FutureValue,
        result.to_le_bytes().to_vec(),
    );
    match kernel.resolve_future(hf.future, vobj) {
        Ok(()) => StepResult::complete(),
        Err(_) => StepResult::fault(process, SEARCH_HEURISTIC),
    }
}

/// `SEARCH_BRANCH`: one synthetic branching-search node (§25.1). Reads the
/// frame, does a bounded amount of deterministic arithmetic ("state duration"),
/// then either spawns `branching` child processes (internal node) or completes
/// (leaf). Every resume point is a run class, so all siblings share this class.
fn search_branch(kernel: &mut Kernel, cont: Ref64, process: Ref64) -> StepResult {
    let mut sf: SearchFrame = load_frame(
        kernel,
        cont,
        SearchFrame {
            value: 0,
            depth: 0,
            branching: 0,
            work_iters: 0,
        },
    );
    let mut acc = sf.value;
    for _ in 0..sf.work_iters {
        acc = acc.wrapping_mul(31).wrapping_add(7);
    }
    sf.value = acc;

    if sf.depth > 0 {
        for i in 0..sf.branching {
            let child = kernel.create_process(crate::abi::ProcessMode::Serial);
            let cframe = SearchFrame {
                value: sf.value.wrapping_add(i as u64),
                depth: sf.depth - 1,
                branching: sf.branching,
                work_iters: sf.work_iters,
            };
            let mut cb = Vec::new();
            cframe.encode(&mut cb);
            kernel.create_continuation(child, SEARCH_BRANCH, 0, cb, DEFAULT_MAX_STEPS);
        }
        // Spawn with no continuation: the commit phase terminates this node.
        StepResult::spawn(process, 0)
    } else {
        store_frame(kernel, cont, &sf);
        StepResult::complete()
    }
}

/// Result type alias to keep callers explicit about what a message receive
/// yields (used by tests / the harness to avoid guessing payload layouts).
pub type ReceivedMessage = MessageDescriptor;
