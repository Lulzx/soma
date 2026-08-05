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

    /// Synthetic branching-search step (§25.1). This is the *base* of a
    /// contiguous block of run classes: §25.1 exposes the number of distinct
    /// continuation classes as a control variable, so a search node of class
    /// index `i` runs as `SEARCH_BRANCH + i`. The classes do genuinely
    /// different work (see the executive), they are not relabelings of one
    /// handler — otherwise the divergence being measured would be fictional.
    pub const SEARCH_BRANCH: u32 = 10;

    /// How many distinct search classes the block reserves.
    pub const MAX_SEARCH_CLASSES: u32 = 8;

    /// The run class a search node of `value` belongs to, given how many
    /// classes the workload is configured to spread across. Deterministic in
    /// the node's own state, so no scheduling decision depends on arrival order.
    pub fn search_class(value: u64, class_count: u32) -> u32 {
        let n = class_count.clamp(1, MAX_SEARCH_CLASSES) as u64;
        SEARCH_BRANCH + (value % n) as u32
    }

    /// Whether `run_class` is one of the search classes, and its index if so.
    pub fn search_class_index(run_class: u32) -> Option<u32> {
        if (SEARCH_BRANCH..SEARCH_BRANCH + MAX_SEARCH_CLASSES).contains(&run_class) {
            Some(run_class - SEARCH_BRANCH)
        } else {
            None
        }
    }

    /// Bounded step budget applied to runnable continuations (§10 `maximum_steps`).
    pub const DEFAULT_MAX_STEPS: u32 = 16;
}
