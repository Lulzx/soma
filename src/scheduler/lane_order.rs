//! The order an epoch's lanes are *run* in, as distinct from the order the plan
//! lists them in (`docs/SOMA-v0.3.md` §4.6).
//!
//! Canonical commit (§4.5) made an epoch's commit independent of the order its
//! lanes ran: every lane finishes, then the journal is applied sorted by the
//! position each effect was produced at, and a position is a place in the plan
//! rather than a moment in time. I25 is the obligation that pays for it — no ≺
//! edge joins two lanes of one epoch, so no lane's behaviour depends on another
//! having gone first.
//!
//! Both of those are statements about a machine that still runs its lanes in a
//! `for` loop, in plan order, every time. So both were claims. This module is
//! what turns them into something a run can fail: the executive is told to run
//! the lanes in a different order, and the run has to come out the same.
//!
//! It is deliberately not threads. A permutation exercises exactly the property
//! threads need — that no lane observes another within its epoch, and that
//! commit does not care who finished first — while staying deterministic, so a
//! failure is a reproducible test rather than an intermittent one. A threaded
//! executive that is wrong about lane independence fails as a rare corruption;
//! this fails on every run, at the same place.

/// How to arrange an epoch's admitted lanes for execution.
///
/// A lane's *number* is unaffected by any of these. The number is its position
/// in the plan, it is decided before anything runs, and it is what stamps every
/// event and effect the lane produces (§4.2) and what chooses its allocation
/// partition (§4.3). Only the order the executive walks them in changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaneOrder {
    /// Run lanes in the order the plan lists them. What a sequential
    /// interpreter does, and the reference every other order is compared to.
    #[default]
    Plan,
    /// Run them backwards. The cheapest order that is wrong in the maximal way:
    /// every pair of lanes is inverted, so any dependence between two lanes of
    /// an epoch is a dependence this order violates.
    Reverse,
    /// Run them in a permutation seeded by `(seed, epoch)`.
    ///
    /// Reversal inverts every pair and therefore misses a defect that only
    /// shows up when lanes interleave rather than mirror — a three-lane
    /// dependence satisfied by both orderings of a pair. A permutation that
    /// changes per epoch also stops a workload from accidentally agreeing with
    /// one fixed shuffle.
    Permuted(u64),
}

impl LaneOrder {
    /// Rearrange `lanes` in place.
    pub fn arrange<T>(self, lanes: &mut [T], epoch: u32) {
        match self {
            LaneOrder::Plan => {}
            LaneOrder::Reverse => lanes.reverse(),
            LaneOrder::Permuted(seed) => {
                // Fisher-Yates, backwards, from a counter-based generator.
                // Deterministic in `(seed, epoch)` and nothing else — a shuffle
                // drawing on a clock or a global would make a failure
                // irreproducible, which is the property this whole module
                // exists to avoid.
                let mut state = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(u64::from(epoch));
                for index in (1..lanes.len()).rev() {
                    let pick = (split_mix(&mut state) % (index as u64 + 1)) as usize;
                    lanes.swap(index, pick);
                }
            }
        }
    }

    /// Whether this order runs lanes in the order the plan lists them.
    ///
    /// I23's clause 2 — the raw trace's append order equals its position order
    /// — is a statement about a sequential in-order executive and is not owed
    /// by anything else. §4.2 said so when the clause was written: a concurrent
    /// implementation appends interleaved and owes clauses 1 and 3, plus I18
    /// after sorting by position. This is the checker asking which case it is
    /// looking at.
    pub fn is_plan_order(self) -> bool {
        self == LaneOrder::Plan
    }
}

fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
