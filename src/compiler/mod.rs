//! Compiler and source-lowering layer (§22 and §24 of the historical contract).
//!
//! Generated state machines cover continuation programs. The deliberately
//! small textual module surface in `ir` names batch evaluators and their resume
//! points; it is not a general-purpose language. Each resume point becomes a
//! run class; frames are byte blobs with a per-run-class layout (see `frame`).

pub mod body;
pub mod examples;
pub mod frame;
pub mod ir;
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

    /// `Join` (v0.3 §4.15): await a future named in the frame, then read it.
    ///
    /// Every other awaiting handler creates the future in the step it awaits,
    /// which is why `await_future`'s `AlreadySettled` outcome was unreachable —
    /// a resolver cannot have run before the future exists. This one is handed
    /// a future somebody else made, so whether the await registers a waiter or
    /// finds the value already published depends on which lane went first.
    /// `docs/SOMA-v0.3.md` §4.14 named that blind spot a property of the
    /// handler set; this is the handler that moves it.
    pub const JOIN_AWAIT: u32 = 5;
    /// Where a `JOIN_AWAIT` continues, by either route.
    pub const JOIN_RESUME: u32 = 6;

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

    /// Ant-colony workload (`experiments::ant_colony`). Each behaviour an ant can
    /// be in is its own resume point and therefore its own run class, which is
    /// the whole point of the workload: an ant's *next* class is decided by the
    /// ant, from its own state, and the scheduler groups on it without asking
    /// what the ant is doing.
    ///
    /// The block is contiguous so a bin index maps to a behaviour by subtraction,
    /// exactly as the search block does.
    pub const ANT_BASE: u32 = 20;
    /// Wander, biased by nothing but the ant's own generator.
    pub const ANT_EXPLORE: u32 = ANT_BASE;
    /// Climb the food-trail gradient.
    pub const ANT_FOLLOW_TRAIL: u32 = ANT_BASE + 1;
    /// Carry food home, laying a food trail behind.
    pub const ANT_CARRY_FOOD: u32 = ANT_BASE + 2;
    /// Blocked last step; pick a new heading before resuming.
    pub const ANT_AVOID_OBSTACLE: u32 = ANT_BASE + 3;
    /// Head for the nest without cargo.
    pub const ANT_RETURN_HOME: u32 = ANT_BASE + 4;
    /// Idle for a bounded number of epochs.
    pub const ANT_WAIT: u32 = ANT_BASE + 5;

    /// How many distinct ant behaviours the block reserves.
    pub const ANT_CLASS_COUNT: u32 = 6;

    /// Aggregate one colony's ants' deposits into the colony's summary.
    pub const COLONY_AGGREGATE: u32 = ANT_BASE + 6;
    /// Fold every colony summary into the pheromone field, decay it, and swap
    /// the double buffer.
    pub const WORLD_STEP: u32 = ANT_BASE + 7;

    /// Whether `run_class` is one of the ant behaviours, and its index if so.
    pub fn ant_class_index(run_class: u32) -> Option<u32> {
        if (ANT_BASE..ANT_BASE + ANT_CLASS_COUNT).contains(&run_class) {
            Some(run_class - ANT_BASE)
        } else {
            None
        }
    }

    /// A short name per ant behaviour, for reports and the trace export.
    pub fn ant_class_name(run_class: u32) -> Option<&'static str> {
        match run_class {
            ANT_EXPLORE => Some("Explore"),
            ANT_FOLLOW_TRAIL => Some("FollowTrail"),
            ANT_CARRY_FOOD => Some("CarryFood"),
            ANT_AVOID_OBSTACLE => Some("AvoidObstacle"),
            ANT_RETURN_HOME => Some("ReturnHome"),
            ANT_WAIT => Some("Wait"),
            COLONY_AGGREGATE => Some("ColonyAggregate"),
            WORLD_STEP => Some("WorldStep"),
            _ => None,
        }
    }
}
