//! End-to-end wall-clock control for the two ant scheduling policies.
//!
//! This times the complete scalar executive, including scheduling and semantic
//! work. It is deliberately separate from structural occupancy: a result is
//! evidence only when both schedules leave the identical world.

use std::time::{Duration, Instant};

use soma::experiments::ant_colony::{run_mode, ColonyKnobs, ColonyRun};
use soma::scheduler::runnable_bins::SchedulingMode;

fn timed(knobs: &ColonyKnobs, mode: SchedulingMode, width: u16) -> (Duration, ColonyRun) {
    let start = Instant::now();
    let run = run_mode(knobs, mode, width);
    (start.elapsed(), run)
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}

fn raw(samples: &[Duration]) -> String {
    samples
        .iter()
        .map(|d| d.as_nanos().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn same_world(a: &ColonyRun, b: &ColonyRun) -> bool {
    a.accounting.steps == b.accounting.steps
        && a.accounting.useful_lane_slots == b.accounting.useful_lane_slots
        && a.epochs == b.epochs
        && a.delivered == b.delivered
        && a.food_trail == b.food_trail
        && a.home_trail == b.home_trail
        && a.population == b.population
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let full = args.iter().any(|arg| arg == "--full");
    let trials = args
        .iter()
        .find_map(|value| value.parse().ok())
        .unwrap_or(if full { 1usize } else { 3usize })
        .max(1);
    let knobs = if full {
        ColonyKnobs {
            colonies: 100,
            ants_per_colony: 100,
            width: 320,
            height: 320,
            epochs: 260,
            ..ColonyKnobs::default()
        }
    } else {
        ColonyKnobs::default()
    };
    let mut fifo = Vec::with_capacity(trials);
    let mut classes = Vec::with_capacity(trials);
    let mut reference = None;

    // Alternate order so a policy does not always inherit the first/second run.
    for trial in 0..trials {
        let ((fifo_time, fifo_run), (class_time, class_run)) = if trial % 2 == 0 {
            let f = timed(&knobs, SchedulingMode::PersistentFifo, 32);
            let c = timed(&knobs, SchedulingMode::RunClassBins, 32);
            (f, c)
        } else {
            let c = timed(&knobs, SchedulingMode::RunClassBins, 32);
            let f = timed(&knobs, SchedulingMode::PersistentFifo, 32);
            (f, c)
        };
        assert!(
            same_world(&fifo_run, &class_run),
            "schedules changed the world"
        );
        if let Some((expected_fifo, expected_class)) = &reference {
            assert!(same_world(expected_fifo, &fifo_run));
            assert!(same_world(expected_class, &class_run));
        } else {
            reference = Some((fifo_run, class_run));
        }
        fifo.push(fifo_time);
        classes.push(class_time);
    }

    let fifo_median = median(&fifo);
    let class_median = median(&classes);
    println!(
        "SOMA ant wall-clock control: ants={} epochs={} trials={trials}",
        knobs.ant_count(),
        knobs.epochs
    );
    println!(
        "persistent-fifo median_ms={:.3} raw_ns={}",
        fifo_median.as_secs_f64() * 1e3,
        raw(&fifo)
    );
    println!(
        "run-class      median_ms={:.3} raw_ns={}",
        class_median.as_secs_f64() * 1e3,
        raw(&classes)
    );
    println!(
        "fifo/run-class ratio={:.3}x identical_world=true",
        fifo_median.as_secs_f64() / class_median.as_secs_f64()
    );
}
