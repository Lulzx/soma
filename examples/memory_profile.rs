//! What a kernel is holding, and on whose behalf.
//!
//! `examples/growth_sweep` made every operation's *time* flat while the state
//! it walks kept growing. That leaves the other half: a machine meant to run
//! indefinitely cannot hold every object, continuation, and capability it has
//! ever made. Time being constant per operation just means it now reaches the
//! memory limit faster.
//!
//! This reports resident bytes per unit of work alongside the counts, so the
//! per-unit cost is attributable rather than a single number that only says
//! "large". Two workloads, because they leak differently: publishing batches
//! accumulates objects and capabilities, while a process that runs to
//! completion should be able to give its private state back and does not.
//!
//!     cargo run --release --example memory_profile

use soma::abi::{ObjectKind, ProcessMode, StateAccess};
use soma::executives::batch::{execute_with_spill, CpuReferenceBackend, PlacementStats};
use soma::experiments::backend_bench::synthetic_program;
use soma::kernel::ownership::freeze;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};

const LEVELS: [usize; 5] = [0, 50_000, 100_000, 200_000, 400_000];

fn resident_kilobytes() -> u64 {
    let pid = std::process::id();
    let Ok(out) = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    else {
        return 0;
    };
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
}

fn header(title: &str, unit: &str) {
    println!("\n=== {title} ===");
    println!(
        "{:>10} {:>10} {:>12} {:>11} {:>11} {:>11} {:>11}",
        unit, "resident", "bytes each", "objects", "capabils.", "continuat.", "processes"
    );
}

fn row(level: usize, base: u64, kernel: &Kernel) {
    let resident = resident_kilobytes();
    let each = if level == 0 {
        0.0
    } else {
        (resident.saturating_sub(base) as f64 * 1024.0) / level as f64
    };
    println!(
        "{:>10} {:>8}MB {:>11.0}B {:>11} {:>11} {:>11} {:>11}",
        level,
        resident / 1024,
        each,
        kernel.object_count(),
        kernel.capability_count(),
        kernel.continuation_count(),
        kernel.process_count(),
    );
}

fn main() {
    if cfg!(debug_assertions) {
        println!("warning: debug build, re-run with --release");
    }
    // One section per run when asked, because resident memory is
    // process-wide: an allocator that has already served four hundred
    // thousand processes does not hand the pages back before the next
    // section starts, and the comparison would read as a saving that
    // reclamation did not make.
    //
    //     cargo run --release --example memory_profile -- reclaiming
    match std::env::args().nth(1).as_deref() {
        Some("batches") => published_batches(),
        Some("processes") => finished_processes(),
        Some("reclaiming") => reclaiming_processes(),
        Some("released") => released_batches(),
        _ => {
            published_batches();
            finished_processes();
            reclaiming_processes();
            released_batches();
            println!(
                "\nrun one section at a time for comparable resident figures: \
                 `-- batches`, `-- processes`, `-- reclaiming`, `-- released`"
            );
        }
    }
}

/// Publishing accumulates output objects and the capabilities over them. Every
/// one is genuinely still reachable — the collective that produced it holds a
/// reference — so this is the cost of never collecting, not a leak.
fn published_batches() {
    let program = synthetic_program(840, 2, 8);
    let stride = program.stride();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = vec![5u8; 8 * stride as usize];
    let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, input).unwrap();

    header("published batches", "batches");
    let base = resident_kilobytes();
    let mut published = 0usize;
    for level in LEVELS {
        while published < level {
            let (collective, _) = kernel
                .create_batch_evaluate_for(owner, program.id(), input, 8, stride)
                .unwrap();
            execute_with_spill(
                &mut kernel,
                owner,
                collective,
                u32::MAX,
                &mut accelerator,
                &mut cpu,
                &mut PlacementStats::default(),
            )
            .unwrap();
            published += 1;
        }
        // Logs have their own retention policy; drain them so this measures
        // the state, not the trace.
        kernel.take_trace_events();
        kernel.take_effect_log();
        row(level, base, &kernel);
    }
    println!(
        "  payload bytes alone: {}MB of {} objects",
        (kernel.object_count() * 32) / (1024 * 1024),
        kernel.object_count()
    );
}

/// A process that runs its continuation to completion and terminates keeps
/// everything: its descriptor, its state object, its mailbox, its capability
/// space, its continuations and their frames. Nothing can reach any of it.
fn finished_processes() {
    header("processes run to termination", "processes");
    let mut kernel = Kernel::new();
    let base = resident_kilobytes();
    let mut done = 0usize;
    for level in LEVELS {
        while done < level {
            let worker = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            kernel
                .create_continuation(
                    worker,
                    worker,
                    ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
                )
                .unwrap();
            kernel.run_epoch();
            kernel.run_epoch();
            done += 1;
            if done.is_multiple_of(50_000) {
                kernel.take_trace_events();
                kernel.take_effect_log();
            }
        }
        kernel.take_trace_events();
        kernel.take_effect_log();
        row(level, base, &kernel);
    }
    let terminated = kernel.terminated_process_count();
    println!(
        "  of {} processes, {terminated} are terminated and still resident",
        kernel.process_count()
    );
}

/// The same workload, reclaiming each finished process before starting the
/// next. Bounded rather than merely slower to grow.
fn reclaiming_processes() {
    header("processes run to termination, reclaiming", "processes");
    let mut kernel = Kernel::new();
    let base = resident_kilobytes();
    let mut done = 0usize;
    for level in LEVELS {
        while done < level {
            let worker = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            kernel
                .create_continuation(
                    worker,
                    worker,
                    ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
                )
                .unwrap();
            kernel.run_epoch();
            kernel.run_epoch();
            kernel.reclaim_finished_processes();
            done += 1;
            if done.is_multiple_of(50_000) {
                kernel.take_trace_events();
                kernel.take_effect_log();
            }
        }
        kernel.take_trace_events();
        kernel.take_effect_log();
        row(level, base, &kernel);
    }
    println!(
        "  {} processes still resident at the end",
        kernel.process_count()
    );
}

/// The publishing workload again, with the owner letting go of each round and
/// a reachability pass collecting what nothing can name.
///
/// The logs are drained every round rather than every fifty thousand, because
/// with the state bounded they are the only thing left growing and leaving
/// them in would report their size as the kernel's. That is not a dodge: the
/// logs already have a retention policy (`kernel::retention`), and
/// `examples/growth_sweep` shows retaining them costs no time — only memory,
/// which is the caller's to spend.
fn released_batches() {
    let program = synthetic_program(841, 2, 8);
    let stride = program.stride();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    header("published batches, released", "batches");
    let base = resident_kilobytes();
    let mut published = 0usize;
    for level in LEVELS {
        while published < level {
            let bytes = vec![5u8; 8 * stride as usize];
            let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
            freeze(&mut kernel, owner, input).unwrap();
            let (collective, completion) = kernel
                .create_batch_evaluate_for(owner, program.id(), input, 8, stride)
                .unwrap();
            let output = execute_with_spill(
                &mut kernel,
                owner,
                collective,
                u32::MAX,
                &mut accelerator,
                &mut cpu,
                &mut PlacementStats::default(),
            )
            .unwrap();
            for held in [output, input, collective, completion] {
                kernel.release_authority(owner, held).unwrap();
            }
            kernel.reclaim_unreachable();
            // Every round, not every fifty thousand: with the state bounded,
            // whatever is left growing is either the logs or the tables, and
            // draining here separates them.
            kernel.take_trace_events();
            kernel.take_effect_log();
            published += 1;
        }
        kernel.take_trace_events();
        kernel.take_effect_log();
        row(level, base, &kernel);
    }
    println!(
        "  {} collectives and {} futures still resident",
        kernel.collective_count(),
        kernel.future_count()
    );
}
