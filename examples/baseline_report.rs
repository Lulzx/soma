//! SOMA against both required baselines (§26) across homogeneous and
//! divergent regimes.

use soma::experiments::cohort_study::baseline_report;
use soma::experiments::dynamic_search::ControlKnobs;

fn main() {
    for classes in [1u32, 4, 8] {
        print!(
            "{}",
            baseline_report(
                &ControlKnobs {
                    class_count: classes,
                    depth: 5,
                    branching_factor: 3,
                    process_count: 4,
                    ..ControlKnobs::default()
                },
                32
            )
        );
    }
}
