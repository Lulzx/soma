//! Prints the §28.1 occupancy comparison across a sweep of regimes.

use soma::experiments::cohort_study::report;
use soma::experiments::dynamic_search::ControlKnobs;

fn main() {
    for classes in [1u32, 2, 4, 8] {
        for width in [8u16, 32] {
            print!(
                "{}",
                report(
                    &ControlKnobs {
                        class_count: classes,
                        depth: 5,
                        branching_factor: 3,
                        process_count: 4,
                        ..ControlKnobs::default()
                    },
                    width
                )
            );
        }
    }
}
