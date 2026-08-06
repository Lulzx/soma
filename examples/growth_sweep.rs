//! Does an operation cost more after the kernel has been running?
//!
//! Every other measurement in this repo asks what one operation costs right
//! now, on a kernel built moments earlier. That is the one regime where a
//! scan over accumulated state is invisible, and it is not the regime SOMA is
//! for: the machine is meant to run for a long time, publishing batches and
//! stepping continuations, and structures that only grow are exactly where an
//! O(n) scan turns a run into O(n²).
//!
//! That is not hypothetical. Authorization scanned the actor's whole
//! capability space and revocation scanned it again for children, which made
//! publishing a batch cost 6µs into a fresh kernel and 485µs into one that had
//! published sixteen thousand. Every test passed throughout, because nothing
//! about the result changes — only how long it takes.
//!
//! So each probe here fixes one operation, grows one structure underneath it,
//! and re-times the operation. A flat column is the answer you want. A column
//! that tracks the level is a scan.
//!
//!     cargo run --release --example growth_sweep
//!
//! The trace and effect logs are drained as the sweep runs, except in the
//! probe that is specifically about not draining them. Retaining ten million
//! trace rows would make this a memory benchmark.

use std::time::{Duration, Instant};

use soma::abi::{ObjectKind, ProcessMode, Ref64, StateAccess};
use soma::executives::batch::{execute_with_spill, CpuReferenceBackend, PlacementStats};
use soma::experiments::backend_bench::synthetic_program;
use soma::kernel::ownership::freeze;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};

/// Accumulated state at which each operation is re-timed.
const LEVELS: [usize; 10] = [
    0, 1_000, 4_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000,
];

/// Repetitions per measurement. Small, because the point is the trend across
/// levels rather than a precise figure at any one of them.
const REPS: usize = 100;

fn median(reps: usize, mut body: impl FnMut()) -> Duration {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed());
    }
    samples.sort();
    samples[samples.len() / 2]
}

/// Resident set size in megabytes, so a probe that is about to exhaust memory
/// says so before it does.
fn resident_megabytes() -> u64 {
    let pid = std::process::id();
    let Ok(output) = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    else {
        return 0;
    };
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        / 1024
}

struct Row {
    level: usize,
    cost: Duration,
    resident: u64,
    note: String,
}

fn report(title: &str, what: &str, rows: &[Row]) {
    println!("\n=== {title} ===");
    println!("timing: {what}");
    println!(
        "{:>12} {:>12} {:>10} {:>9}  state",
        "accumulated", "median", "vs first", "resident"
    );
    let first = rows.first().map(|row| row.cost.as_secs_f64()).unwrap_or(0.0);
    for row in rows {
        let ratio = if first > 0.0 {
            row.cost.as_secs_f64() / first
        } else {
            1.0
        };
        println!(
            "{:>12} {:>10.2}µs {:>9.1}x {:>7}MB  {}",
            row.level,
            row.cost.as_secs_f64() * 1e6,
            ratio,
            row.resident,
            row.note,
        );
    }
}

fn main() {
    if cfg!(debug_assertions) {
        println!("warning: debug build, re-run with --release");
    }
    published_batches();
    live_processes();
    pending_continuations();
    retained_logs();
}

/// Publishing one more batch, as published batches accumulate.
///
/// This is the one that was quadratic. It should now be flat, and the sweep
/// runs to a million because sixteen thousand was where the old cost was
/// merely bad rather than fatal.
fn published_batches() {
    let program = synthetic_program(830, 2, 8);
    let stride = program.stride();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = vec![5u8; 8 * stride as usize];
    let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, input).unwrap();

    let publish = |kernel: &mut Kernel,
                       accelerator: &mut CpuReferenceBackend,
                       cpu: &mut CpuReferenceBackend| {
        let (collective, _) = kernel
            .create_batch_evaluate_for(owner, program.id(), input, 8, stride)
            .unwrap();
        execute_with_spill(
            kernel,
            owner,
            collective,
            u32::MAX,
            accelerator,
            cpu,
            &mut PlacementStats::default(),
        )
        .unwrap();
    };

    let mut rows = Vec::new();
    let mut published = 0usize;
    for level in LEVELS {
        while published < level {
            publish(&mut kernel, &mut accelerator, &mut cpu);
            published += 1;
            if published.is_multiple_of(50_000) {
                kernel.take_trace_events();
                kernel.take_effect_log();
            }
        }
        let cost = median(REPS, || publish(&mut kernel, &mut accelerator, &mut cpu));
        published += REPS;
        rows.push(Row {
            level,
            cost,
            resident: resident_megabytes(),
            note: format!("{} capabilities", kernel.capability_count()),
        });
        kernel.take_trace_events();
        kernel.take_effect_log();
    }
    report(
        "published batches",
        "one more execute_with_spill",
        &rows,
    );
}

/// Running an epoch with a fixed, tiny amount of work, as idle processes
/// accumulate.
///
/// An epoch should cost what its runnable work costs. If it costs what the
/// process table holds, then a long-lived population makes every epoch slower
/// regardless of how much of it is actually running.
fn live_processes() {
    let mut kernel = Kernel::new();

    // A fresh worker per repetition. A process whose continuations have all
    // run is terminated, and `create_continuation` authorizes before it checks
    // process state, so reusing one worker reports `AuthorityDenied` on the
    // second step rather than saying the process is gone.
    let step = |kernel: &mut Kernel| {
        let worker = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .create_continuation(
                worker,
                worker,
                ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
            )
            .unwrap();
        // Two epochs: one to promote the continuation from `next` to
        // `current`, one to run it.
        kernel.run_epoch();
        kernel.run_epoch();
    };

    let mut rows = Vec::new();
    let mut processes = 0usize;
    for level in LEVELS {
        while processes < level {
            kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            processes += 1;
            if processes.is_multiple_of(50_000) {
                kernel.take_trace_events();
                kernel.take_effect_log();
            }
        }
        let cost = median(REPS, || step(&mut kernel));
        processes += REPS;
        rows.push(Row {
            level,
            cost,
            resident: resident_megabytes(),
            note: format!("{} idle processes", processes),
        });
        kernel.take_trace_events();
        kernel.take_effect_log();
    }
    report(
        "live processes",
        "one continuation through two epochs",
        &rows,
    );
}

/// Cancelling a process, as continuations pile up in the scheduler's bins.
///
/// `Scheduler::remove` walks both epoch buffers of every bin per continuation
/// removed, and cancellation removes every continuation a process owns, so the
/// suspicion is that cancelling k continuations with q pending costs k·q.
fn pending_continuations() {
    let mut rows = Vec::new();
    for level in LEVELS {
        // A fresh kernel per level: the point is the depth of the bins, and
        // reusing one kernel would drain them as epochs run.
        let mut kernel = Kernel::new();
        let bystander = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        for _ in 0..level {
            kernel
                .create_continuation(
                    bystander,
                    bystander,
                    ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
                )
                .unwrap();
        }

        // Each victim owns a handful of continuations, so cancelling it is a
        // realistic removal rather than a single one.
        let victims: Vec<Ref64> = (0..REPS)
            .map(|_| {
                let victim = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
                for _ in 0..4 {
                    kernel
                        .create_continuation(
                            victim,
                            victim,
                            ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
                        )
                        .unwrap();
                }
                victim
            })
            .collect();

        let pending = kernel.total_pending();
        let mut remaining = victims.into_iter();
        let cost = median(REPS, || {
            let victim = remaining.next().expect("one victim per repetition");
            kernel.cancel_process(SYSTEM_PRINCIPAL, victim).unwrap();
        });
        rows.push(Row {
            level,
            cost,
            resident: resident_megabytes(),
            note: format!("{pending} pending continuations"),
        });
    }
    report(
        "pending continuations",
        "cancelling one process owning four continuations",
        &rows,
    );
}

/// Publishing one more batch, with the logs retained rather than drained.
///
/// `Retain` is the default and is what every whole-run check needs, so the
/// question is whether keeping the logs costs anything per operation beyond
/// the memory. Appending is O(1); this is here to confirm nothing reads them.
fn retained_logs() {
    let program = synthetic_program(831, 2, 8);
    let stride = program.stride();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = vec![5u8; 8 * stride as usize];
    let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, input).unwrap();

    let publish = |kernel: &mut Kernel,
                       accelerator: &mut CpuReferenceBackend,
                       cpu: &mut CpuReferenceBackend| {
        let (collective, _) = kernel
            .create_batch_evaluate_for(owner, program.id(), input, 8, stride)
            .unwrap();
        execute_with_spill(
            kernel,
            owner,
            collective,
            u32::MAX,
            accelerator,
            cpu,
            &mut PlacementStats::default(),
        )
        .unwrap();
    };

    let mut rows = Vec::new();
    let mut published = 0usize;
    // Stops earlier than the other probes: nothing is drained here, so the
    // trace is the memory limit rather than the objects.
    for level in LEVELS.iter().take_while(|level| **level <= 256_000) {
        while published < *level {
            publish(&mut kernel, &mut accelerator, &mut cpu);
            published += 1;
        }
        let cost = median(REPS, || publish(&mut kernel, &mut accelerator, &mut cpu));
        published += REPS;
        rows.push(Row {
            level: *level,
            cost,
            resident: resident_megabytes(),
            note: format!("{} trace rows", kernel.trace_events().len()),
        });
    }
    report(
        "retained logs",
        "one more execute_with_spill, nothing drained",
        &rows,
    );
}
