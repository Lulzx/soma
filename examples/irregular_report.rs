//! Occupancy/latency frontiers under irregular arrival.

use soma::experiments::irregular_arrival::{report, IrregularKnobs};

fn main() {
    // Level-synchronous control: all roots at once, no jitter.
    print!(
        "{}",
        report(
            &IrregularKnobs { arrival_span: 0, jitter: 0, ..IrregularKnobs::default() },
            32
        )
    );
    println!();
    // Irregular: staggered roots and variable heuristic latency.
    print!("{}", report(&IrregularKnobs::default(), 32));
}
