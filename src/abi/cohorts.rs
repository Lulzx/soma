//! Cohort ABI (§14).
//!
//! A cohort is one SIMD dispatch. Every active lane in it shares the same module
//! function, resume point, frame schema, and numerical policy, and differs only
//! in process and object state — which is exactly what a `run_class` names, so a
//! cohort is always the continuations of one run class.
//!
//! A cohort descriptor therefore models **one width-`W` dispatch**, of which
//! `active_lanes` do useful work. That single invariant is what makes lane
//! occupancy measurable: a lane group whose continuations span `k` run classes
//! cannot be one dispatch, it must be `k` of them, each with the lanes of the
//! other `k-1` classes masked off. Counting descriptors counts dispatches.

use super::refs::Ref64;
use super::AbiHeader;

/// Maximum lanes in one cohort. Sized to a typical SIMD/warp width so the
/// descriptor stays a fixed-width ABI structure with no native pointers (§3).
pub const MAX_COHORT_WIDTH: usize = 32;

/// Policy for the final, incompletely filled cohort of a run class (§14).
///
/// The contract is explicit that the choice is made on measured cost rather
/// than fixed ideology, so it is a runtime knob, not a compile-time decision.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PartialCohortPolicy {
    /// Dispatch the partial cohort anyway, wasting the inactive lanes.
    #[default]
    RunPartial = 1,
    /// Hold the remainder back for the next epoch, hoping it fills. Callers
    /// must guarantee forward progress themselves: a run class that never
    /// reaches `width` would otherwise starve forever.
    Defer = 2,
    /// Execute the remainder as scalar work — one width-1 dispatch per
    /// continuation, so no lane is wasted but no lane is shared either.
    SendToCpu = 3,
    /// Fold the remainder into a generic catch-all class. Phase 1 has no
    /// generic class to merge into, so this behaves as `RunPartial`; it exists
    /// so the policy set matches §14 and the choice stays measurable later.
    MergeWithGenericClass = 4,
}

/// One SIMD dispatch (§14).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CohortDescriptor {
    pub header: AbiHeader,

    pub run_class: u32,
    pub width: u16,
    pub active_lanes: u16,

    pub continuations: [Ref64; MAX_COHORT_WIDTH],
}

impl CohortDescriptor {
    /// Build a cohort of `width` lanes over `members`, all of `run_class`.
    /// `members` is truncated to the width; lanes beyond `active_lanes` are null.
    pub fn new(run_class: u32, width: u16, members: &[Ref64]) -> CohortDescriptor {
        let width = (width as usize).clamp(1, MAX_COHORT_WIDTH);
        let active = members.len().min(width);
        let mut continuations = [Ref64::NULL; MAX_COHORT_WIDTH];
        continuations[..active].copy_from_slice(&members[..active]);
        CohortDescriptor {
            header: AbiHeader::new(9, std::mem::size_of::<CohortDescriptor>() as u32),
            run_class,
            width: width as u16,
            active_lanes: active as u16,
            continuations,
        }
    }

    /// A single continuation executed on its own, as `SendToCpu` produces.
    pub fn scalar(run_class: u32, cont: Ref64) -> CohortDescriptor {
        CohortDescriptor::new(run_class, 1, &[cont])
    }

    /// The continuations that will actually execute, in lane order.
    pub fn lanes(&self) -> &[Ref64] {
        &self.continuations[..self.active_lanes as usize]
    }

    /// Whether every lane of this dispatch carries work.
    pub fn is_full(&self) -> bool {
        self.active_lanes == self.width
    }

    /// Lanes that will be masked off during this dispatch.
    pub fn idle_lanes(&self) -> u16 {
        self.width - self.active_lanes
    }
}
