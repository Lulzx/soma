//! Wall-clock regime map for the optimistic Phase-F executive.

use std::time::{Duration, Instant};

use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::speculation::EpochExecutive;

fn once(lanes: u32, arithmetic_ops: u32, speculative: bool) -> (Duration, u64) {
    let knobs = ControlKnobs {
        branching_factor: 0,
        depth: 0,
        process_count: lanes,
        class_count: lanes.min(4),
        arithmetic_ops,
        ..ControlKnobs::default()
    };
    let mut kernel = build(&knobs);
    if speculative {
        kernel.configure_epoch_executive(EpochExecutive::Speculative {
            max_lanes: lanes as usize,
        });
    }
    let started = Instant::now();
    let steps = kernel.run_epoch();
    let elapsed = started.elapsed();
    assert_eq!(steps, lanes as usize);
    (elapsed, kernel.speculation_stats().committed_lanes)
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    println!("speculative concurrent epoch wall-clock (median of 9, release recommended)");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>9} {:>10}",
        "lanes", "ops/lane", "reference", "optimistic", "speedup", "committed"
    );
    for lanes in [2, 4, 8] {
        for arithmetic_ops in [1_000, 10_000, 100_000, 1_000_000] {
            // Warm both paths before sampling allocation and thread startup.
            let _ = once(lanes, arithmetic_ops, false);
            let _ = once(lanes, arithmetic_ops, true);
            let mut reference = Vec::new();
            let mut optimistic = Vec::new();
            let mut committed = 0;
            for _ in 0..9 {
                reference.push(once(lanes, arithmetic_ops, false).0);
                let sample = once(lanes, arithmetic_ops, true);
                optimistic.push(sample.0);
                committed = sample.1;
            }
            let reference = median(reference);
            let optimistic = median(optimistic);
            let speedup = reference.as_secs_f64() / optimistic.as_secs_f64();
            println!(
                "{:>6} {:>10} {:>9.3} ms {:>9.3} ms {:>8.2}x {:>10}",
                lanes,
                arithmetic_ops,
                reference.as_secs_f64() * 1_000.0,
                optimistic.as_secs_f64() * 1_000.0,
                speedup,
                committed,
            );
        }
    }
}
