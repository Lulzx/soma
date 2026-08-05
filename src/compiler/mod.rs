//! Compiler / source-lowering layer (§22, §24).
//!
//! In Phase 1 the source model is realized through generated state machines
//! rather than a complete language. Each resume point becomes a run class
//! (§22); frames are byte blobs with a per-run-class layout (see `frame`).

pub mod frame;
pub mod state_machine_lowering;

/// Run-class identifiers. Every resume point of every state machine is its own
/// run class (§22), so the runnable-bin key and the interpreter dispatch key are
/// the same value — this is what lets the scheduler group continuations into
/// cohorts without inspecting arbitrary metadata (§9).
pub mod run_classes {
    /// `Expand` state machine (§22):
    /// - `resume_0`: receive request, store in frame, spawn heuristic, await.
    /// - `resume_1`: load heuristic result, generate a bounded group of moves.
    /// - `resume_2`: finish child creation, send reply, complete.
    pub const EXPAND_RESUME_0: u32 = 1;
    pub const EXPAND_RESUME_1: u32 = 2;
    pub const EXPAND_RESUME_2: u32 = 3;

    /// Standalone heuristic evaluation used by `Expand` (§22).
    pub const SEARCH_HEURISTIC: u32 = 4;

    /// Synthetic branching-search step (§25.1).
    pub const SEARCH_BRANCH: u32 = 10;

    /// Bounded step budget applied to runnable continuations (§10 `maximum_steps`).
    pub const DEFAULT_MAX_STEPS: u32 = 16;
}
