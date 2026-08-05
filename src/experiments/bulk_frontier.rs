//! The bulk frontier baseline (§26).
//!
//! This is the strong manually-batched implementation: the way a competent
//! engineer writes this search on a GPU today, without any of SOMA's machinery.
//! There are no processes, no continuations, no mailboxes, no run-class bins and
//! no scheduler. There is a flat array of node states, and a level-synchronous
//! loop that the host drives:
//!
//! ```text
//! while frontier is not empty:
//!     host launches one kernel over the whole frontier   <- host round-trip
//!     every node does its work and writes its children
//!     global barrier; compact the children into the next frontier
//! ```
//!
//! The contract is emphatic that this baseline is essential — without it SOMA
//! might appear successful only because the baselines are weak (§26). So it is
//! implemented to be genuinely strong rather than as a straw man:
//!
//! * it does exactly the work SOMA does, by calling the same `search_step`;
//! * it is scored by exactly the same divergence model, `dispatch_cost`;
//! * and it comes in a **sorted** variant, which partitions each frontier by run
//!   class before dispatching. That variant is manual cohorting, performed by
//!   the host once per level, and on level-synchronous work it is expected to
//!   match or beat SOMA on lane occupancy. Reporting that honestly is the whole
//!   point of having the baseline.
//!
//! What it cannot do is run without the host in the loop, or absorb work that
//! arrives off the level boundary — which is where §28.4 and the irregular
//! regimes are argued, not here.

use crate::compiler::run_classes::{search_class, SEARCH_BRANCH};
use crate::compiler::state_machine_lowering::search_step;
use crate::experiments::dynamic_search::ControlKnobs;
use crate::scheduler::cohorts::{dispatch_cost, DispatchCost};

/// One node in a frontier: its pre-step value and remaining depth.
#[derive(Clone, Copy, Debug)]
struct Node {
    value: u64,
    depth: u32,
}

/// A completed bulk-frontier run.
#[derive(Clone, Copy, Debug)]
pub struct BulkFrontierRun {
    /// Whether each frontier was partitioned by run class before dispatch.
    pub sorted: bool,
    pub width: u16,
    /// Level-synchronous iterations, i.e. kernel launches.
    pub levels: u32,
    /// Host round-trips: one launch per level. SOMA's design target is zero
    /// (§28.4) — the host must not submit individual search operations.
    pub host_launches: u64,
    /// Global barriers: one per level, to compact the next frontier.
    pub global_barriers: u64,
    pub nodes_expanded: u64,
    pub cost: DispatchCost,
}

impl BulkFrontierRun {
    pub fn lane_occupancy(&self) -> f64 {
        self.cost.occupancy()
    }

    pub fn dispatches(&self) -> u64 {
        self.cost.dispatches
    }
}

/// The run class a node would belong to, matching the SOMA workload exactly.
fn class_of(value: u64, class_count: u32) -> u32 {
    search_class(value, class_count)
}

/// Run the level-synchronous bulk frontier over the same tree SOMA searches.
///
/// `sorted` partitions each frontier by run class before cutting it into lane
/// groups — the manual equivalent of cohorting, paid for with a host-side sort
/// per level.
pub fn run(knobs: &ControlKnobs, width: u16, sorted: bool) -> BulkFrontierRun {
    let mut frontier: Vec<Node> = (0..knobs.process_count)
        .map(|root| Node {
            value: root as u64 + 1,
            depth: knobs.depth,
        })
        .collect();

    let mut out = BulkFrontierRun {
        sorted,
        width,
        levels: 0,
        host_launches: 0,
        global_barriers: 0,
        nodes_expanded: 0,
        cost: DispatchCost::default(),
    };

    while !frontier.is_empty() {
        out.levels += 1;
        // The host submits this level and waits for it.
        out.host_launches += 1;

        let classes: Vec<u32> = frontier
            .iter()
            .map(|n| class_of(n.value, knobs.class_count))
            .collect();

        let level_cost = if sorted {
            // The host partitions the frontier by run class and launches each
            // segment separately. Segmenting matters: merely sorting and then
            // cutting into fixed lane groups would still straddle class
            // boundaries, and charging the baseline for straddles a real
            // implementation would not pay would make it a straw man.
            let mut distinct: Vec<u32> = classes.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let mut total = DispatchCost::default();
            for class in distinct {
                let n = classes.iter().filter(|c| **c == class).count();
                let segment = vec![class; n];
                let cost = dispatch_cost(&segment, width);
                total.dispatches += cost.dispatches;
                total.lane_slots += cost.lane_slots;
                total.useful_lane_slots += cost.useful_lane_slots;
                total.full_dispatches += cost.full_dispatches;
            }
            total
        } else {
            dispatch_cost(&classes, width)
        };

        out.cost.dispatches += level_cost.dispatches;
        out.cost.lane_slots += level_cost.lane_slots;
        out.cost.useful_lane_slots += level_cost.useful_lane_slots;
        out.cost.full_dispatches += level_cost.full_dispatches;
        out.nodes_expanded += frontier.len() as u64;

        // Expand the whole frontier, then compact the children.
        let mut next = Vec::new();
        for node in &frontier {
            let index = class_of(node.value, knobs.class_count) - SEARCH_BRANCH;
            let value = search_step(node.value, knobs.arithmetic_ops, index);
            if node.depth > 0 {
                for i in 0..knobs.branching_factor {
                    next.push(Node {
                        value: value.wrapping_add(i as u64),
                        depth: node.depth - 1,
                    });
                }
            }
        }

        // Global barrier before the next level can be launched.
        out.global_barriers += 1;
        frontier = next;
    }

    out
}
