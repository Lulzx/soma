//! Where continuation cohorting pays and where it does not (§25).
//!
//! Advantage is occupancy at a matched mean-wait budget of one tick.

use soma::experiments::irregular_arrival::{regime_map, IrregularKnobs};

fn main() {
    let map = regime_map(
        &IrregularKnobs::default(),
        &[0, 4, 12, 32],
        &[0, 1, 3, 8],
        &[1, 4],
        32,
        1.0,
    );
    println!("span jitter classes | adv   soma  bulk | wait-to-peak soma/bulk");
    for p in &map {
        println!(
            "{:>4} {:>6} {:>7} | {:>4.2}  {:.3} {:.3} | {:>5.2} {:>6.2}  {}",
            p.arrival_span,
            p.jitter,
            p.class_count,
            p.advantage,
            p.soma_occupancy,
            p.bulk_occupancy,
            p.soma_wait_at_peak,
            p.bulk_wait_at_peak,
            if p.is_level_synchronous() { "(level-synchronous)" } else { "" },
        );
    }
}
