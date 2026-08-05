//! Cohort construction (§14): turning a bin's runnable continuations into SIMD
//! dispatches.
//!
//! The algorithm is deliberately identical regardless of how continuations were
//! binned. It cuts the bin into width-`W` lane groups, then splits each group by
//! run class, because a dispatch is uniform by definition (§15) — a lane group
//! holding `k` distinct run classes costs `k` dispatches, not one.
//!
//! That is the whole experiment in one function. Run-class bins hand it lane
//! groups that are already uniform, so every group costs exactly one dispatch. A
//! persistent FIFO hands it lane groups drawn from mixed arrival order, so a
//! group costs as many dispatches as it holds distinct classes. Nothing about
//! the cohorting code differs between the two; only the binning does.

use crate::abi::cohorts::{CohortDescriptor, PartialCohortPolicy};
use crate::abi::Ref64;

/// The cohorts to dispatch, plus any continuations held back for a later epoch.
#[derive(Clone, Debug, Default)]
pub struct CohortPlan {
    pub cohorts: Vec<CohortDescriptor>,
    /// Continuations the partial policy declined to run this epoch.
    pub deferred: Vec<Ref64>,
}

impl CohortPlan {
    /// Total lane-slots consumed: every dispatch costs its full width, whether
    /// or not each lane carries work.
    pub fn lane_slots(&self) -> u64 {
        self.cohorts.iter().map(|c| c.width as u64).sum()
    }

    /// Lane-slots that carry a real continuation.
    pub fn useful_lane_slots(&self) -> u64 {
        self.cohorts.iter().map(|c| c.active_lanes as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.cohorts.is_empty()
    }
}

/// Partition one bin's `lanes` — `(continuation, run_class)` pairs in bin order
/// — into cohorts of `width`.
///
/// Ordering is positional, so the result is deterministic for a given bin order.
pub fn build_cohorts(
    lanes: &[(Ref64, u32)],
    width: u16,
    policy: PartialCohortPolicy,
) -> CohortPlan {
    let mut plan = CohortPlan::default();
    let group = (width.max(1) as usize).min(crate::abi::cohorts::MAX_COHORT_WIDTH);

    for lane_group in lanes.chunks(group) {
        // `chunks` yields a short group only as the last one.
        if lane_group.len() < group {
            match policy {
                PartialCohortPolicy::Defer => {
                    plan.deferred.extend(lane_group.iter().map(|(c, _)| *c));
                    continue;
                }
                PartialCohortPolicy::SendToCpu => {
                    for (cont, run_class) in lane_group {
                        plan.cohorts
                            .push(CohortDescriptor::scalar(*run_class, *cont));
                    }
                    continue;
                }
                // Both remaining policies dispatch the remainder as-is; Phase 1
                // has no generic class for the merge variant to fold into.
                PartialCohortPolicy::RunPartial
                | PartialCohortPolicy::MergeWithGenericClass => {}
            }
        }

        // One dispatch per distinct run class present, in first-appearance
        // order. A uniform group produces exactly one.
        let mut classes: Vec<u32> = Vec::new();
        for (_, run_class) in lane_group {
            if !classes.contains(run_class) {
                classes.push(*run_class);
            }
        }
        for run_class in classes {
            let members: Vec<Ref64> = lane_group
                .iter()
                .filter(|(_, rc)| *rc == run_class)
                .map(|(c, _)| *c)
                .collect();
            plan.cohorts
                .push(CohortDescriptor::new(run_class, width, &members));
        }
    }

    plan
}
