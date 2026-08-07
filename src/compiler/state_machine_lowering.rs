//! Lowered state machines: the `Expand` example from §22.
//!
//! The source-level model
//!
//! ```text
//! process SearchState(state: Unique<State>) {
//!     on Expand(request) {
//!         let node = state.current;
//!         let score = await heuristic(node);
//!         for movement in legal_moves(node) { spawn SearchState(...); }
//!         send request.reply <- score;
//!     }
//! }
//! ```
//!
//! is lowered here into the three resume points `EXPAND_RESUME_0/1/2` plus a
//! `SEARCH_HEURISTIC` run class. Each resume point is its own run class and each
//! frame is a byte blob (see `frame`). The helpers in this module construct the
//! initial kernel state for a workload, so tests and experiments don't repeat
//! the ABI plumbing.

use crate::abi::{ObjectKind, ProcessMode, Ref64, StateAccess};
use crate::compiler::frame::{put_ref64, put_u64, put_vec_u64, ByteCursor, Frame, FrameError};
use crate::compiler::run_classes::{DEFAULT_MAX_STEPS, EXPAND_RESUME_0};
use crate::kernel::{ContinuationSpec, Kernel};

/// Frame for the `Expand` state machine across all three resume points.
///
/// Field layout is little-endian and stable so the frame can live in shared
/// memory and move between executives without register migration (§16).
#[derive(Clone, Debug)]
pub struct ExpandFrame {
    pub request_value: u64,
    pub heuristic_future: Ref64,
    pub heuristic_result: u64,
    pub reply_receiver: Ref64,
    pub moves: Vec<u64>,
    /// How many of `moves` have already had their child spawned. A resume point
    /// may re-enter after blocking, so this is what makes child creation
    /// idempotent across re-entries.
    pub move_index: u32,
    /// The reply payload object, allocated once and remembered so a re-entry
    /// reuses it instead of leaking a fresh object per attempt.
    pub reply_payload: Ref64,
}

impl ExpandFrame {
    pub fn initial(request_value: u64, reply_receiver: Ref64) -> ExpandFrame {
        ExpandFrame {
            request_value,
            heuristic_future: Ref64::NULL,
            heuristic_result: 0,
            reply_receiver,
            moves: Vec::new(),
            move_index: 0,
            reply_payload: Ref64::NULL,
        }
    }
}

impl Frame for ExpandFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_u64(out, self.request_value);
        put_ref64(out, self.heuristic_future);
        put_u64(out, self.heuristic_result);
        put_ref64(out, self.reply_receiver);
        put_vec_u64(out, &self.moves);
        crate::compiler::frame::put_u32(out, self.move_index);
        put_ref64(out, self.reply_payload);
    }

    fn decode(c: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(ExpandFrame {
            request_value: c.u64()?,
            heuristic_future: c.ref64()?,
            heuristic_result: c.u64()?,
            reply_receiver: c.ref64()?,
            moves: c.vec_u64()?,
            move_index: c.u32()?,
            reply_payload: c.ref64()?,
        })
    }
}

/// Frame for the heuristic evaluation continuation.
#[derive(Clone, Debug)]
pub struct HeuristicFrame {
    pub future: Ref64,
    pub input: u64,
}

/// Frame for a `Join` continuation (v0.3 §4.15): the future it awaits, which
/// it did not create, and the value it read once that future settled.
#[derive(Clone, Debug)]
pub struct JoinFrame {
    pub future: Ref64,
    /// The value the future took, as the awaiter saw it. A reference rather
    /// than its contents: the value object is minted by whichever process
    /// resolved the future, and a `LaneView` has no operation that would let
    /// that process delegate READ on it mid-epoch, so an awaiter can observe
    /// *that* a future resolved and to what, and not what is inside.
    pub observed: Ref64,
}

/// Frame for one synthetic branching-search node (§25.1).
#[derive(Clone, Debug)]
pub struct SearchFrame {
    pub value: u64,
    /// Remaining depth; a node at depth 0 is a leaf and completes.
    pub depth: u32,
    /// Number of children spawned by an internal node (the branching factor).
    pub branching: u32,
    /// Bounded arithmetic iterations per node (the "state duration" knob).
    pub work_iters: u32,
    /// How many distinct run classes this search spreads across (§25.1). Each
    /// node derives its own class from its value, so children scatter across
    /// classes and a mixed-arrival queue becomes genuinely divergent.
    pub class_count: u32,
}

/// The arithmetic one search node performs (§25.1's "state duration" knob).
///
/// Shared verbatim by the continuation interpreter and the bulk-frontier
/// baseline. Any comparison between them is only meaningful if both compute the
/// same thing, so both call this rather than reimplementing it.
///
/// `class_index` selects the constants, which is what makes the search run
/// classes genuinely distinct code paths rather than labels on one handler.
pub fn search_step(value: u64, work_iters: u32, class_index: u32) -> u64 {
    let multiplier = 31u64.wrapping_add(class_index as u64 * 2);
    let addend = 7u64.wrapping_add(class_index as u64);
    let mut acc = value;
    for _ in 0..work_iters {
        acc = acc.wrapping_mul(multiplier).wrapping_add(addend);
    }
    acc
}

impl SearchFrame {
    /// A single node with no children, confined to one run class.
    pub fn leaf(value: u64, work_iters: u32) -> SearchFrame {
        SearchFrame {
            value,
            depth: 0,
            branching: 0,
            work_iters,
            class_count: 1,
        }
    }

    /// The run class this node belongs to.
    pub fn run_class(&self) -> u32 {
        crate::compiler::run_classes::search_class(self.value, self.class_count)
    }
}

impl Frame for SearchFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_u64(out, self.value);
        crate::compiler::frame::put_u32(out, self.depth);
        crate::compiler::frame::put_u32(out, self.branching);
        crate::compiler::frame::put_u32(out, self.work_iters);
        crate::compiler::frame::put_u32(out, self.class_count);
    }

    fn decode(c: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(SearchFrame {
            value: c.u64()?,
            depth: c.u32()?,
            branching: c.u32()?,
            work_iters: c.u32()?,
            class_count: c.u32()?,
        })
    }
}

impl Frame for JoinFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_ref64(out, self.future);
        put_ref64(out, self.observed);
    }

    fn decode(c: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(JoinFrame {
            future: c.ref64()?,
            observed: c.ref64()?,
        })
    }
}

impl Frame for HeuristicFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_ref64(out, self.future);
        put_u64(out, self.input);
    }

    fn decode(c: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(HeuristicFrame {
            future: c.ref64()?,
            input: c.u64()?,
        })
    }
}

/// Construct the `Expand` workload: a requester process and a search process
/// whose initial continuation is `EXPAND_RESUME_0`, with an `ExpandRequest`
/// message already sitting in the search process's mailbox.
///
/// Returns `(expand_process, requester_process)`.
pub fn create_expand(kernel: &mut Kernel, request_value: u64) -> (Ref64, Ref64) {
    // Requester: an empty process that will receive the reply message.
    let requester = kernel.create_process(crate::kernel::SYSTEM_PRINCIPAL, ProcessMode::Serial);

    // Search process.
    let expand = kernel.create_process(crate::kernel::SYSTEM_PRINCIPAL, ProcessMode::Serial);

    // The search process may reply to the requester. A process reference alone
    // carries no authority, so delegate SEND explicitly.
    kernel
        .grant_capability(
            crate::kernel::SYSTEM_PRINCIPAL,
            expand,
            requester,
            crate::abi::Rights::SEND,
            0,
            0,
        )
        .expect("system created both processes");

    // Payload object for the ExpandRequest message.
    let payload = kernel.create_object(
        crate::kernel::SYSTEM_PRINCIPAL,
        ObjectKind::MessagePayload,
        request_value.to_le_bytes().to_vec(),
    );

    // Initial continuation (resume_0) with its frame.
    let frame = ExpandFrame::initial(request_value, requester);
    let mut bytes = Vec::new();
    frame.encode(&mut bytes);
    let cont = kernel
        .create_continuation(
            crate::kernel::SYSTEM_PRINCIPAL,
            expand,
            ContinuationSpec::new(
                StateAccess::ReadOnly,
                EXPAND_RESUME_0,
                EXPAND_RESUME_0,
                bytes,
                DEFAULT_MAX_STEPS,
            ),
        )
        .expect("system may create the initial continuation");

    // Ingest the request message into the search process's mailbox (§18 Phase A).
    // This wakes resume_0 if it is already waiting; otherwise it is consumed on
    // the next receive.
    let _ = kernel.ingest_message(
        crate::kernel::SYSTEM_PRINCIPAL,
        requester,
        expand,
        payload,
        cont,
    );

    (expand, requester)
}
