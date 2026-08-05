//! The ant colony scheduled two ways (`experiments::ant_colony`).
//!
//! The same population, the same seed, the same world — binned by run class in
//! one run and into a single persistent FIFO in the other. The control line
//! matters more than the ratio: if the two runs did not simulate the same
//! colony, the ratio is comparing two different workloads.

use soma::experiments::ant_colony::{report, ColonyKnobs};

fn main() {
    // `MAX_COHORT_WIDTH` is 32, so anything above it clamps and reports
    // the same numbers twice.
    for width in [8u16, 16, 32] {
        print!("{}", report(&ColonyKnobs::default(), width));
        println!();
    }
}
