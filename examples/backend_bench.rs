//! What the batch backends actually cost, in wall-clock time.
//!
//! Run with `--release`; the CPU reference backend is a tree-walking
//! interpreter and an unoptimized build measures the interpreter's debug
//! bookkeeping rather than the work. With `--features metal` on macOS the
//! Metal backend is measured beside it and the crossover between them is
//! fitted rather than assumed.
//!
//!     cargo run --release --example backend_bench
//!     cargo run --release --features metal --example backend_bench

use soma::compiler::body::EvaluatorProgram;
use soma::executives::batch::{BatchBackend, CpuReferenceBackend};
use soma::experiments::backend_bench::{
    crossover, measured_crossover, render, time_epoch, time_evaluate, time_published_path, Sweep,
};

/// Fields per element, arithmetic ops per element, and a name for the regime.
/// The three rows are the memory-bound / compute-bound / in-between split: at
/// 2 ops the body is a copy with a multiply in it, at 512 it is arithmetic
/// with a copy attached.
const REGIMES: [(u32, u32, &str); 4] = [
    (2, 2, "memory-bound"),
    (2, 32, "light"),
    (2, 128, "compute-heavy"),
    (8, 128, "compute-heavy, 32B element"),
];

const SIZES: [u32; 8] = [32, 128, 1_024, 8_192, 65_536, 262_144, 1_048_576, 4_194_304];

/// Interpreted element-steps one sweep may spend at a given op count, so the
/// compute-heavy regimes do not run the CPU interpreter for minutes at four
/// million elements to re-establish that it is linear.
const CPU_WORK_BUDGET: u64 = 64_000_000;

fn sizes_for(alu_ops: u32) -> Vec<u32> {
    let limit = (CPU_WORK_BUDGET / alu_ops.max(1) as u64).min(u32::MAX as u64) as u32;
    SIZES.iter().copied().filter(|n| *n <= limit).collect()
}

fn sweep(
    name: &str,
    backend: &mut dyn BatchBackend,
    program: &EvaluatorProgram,
    sizes: &[u32],
) -> Option<Sweep> {
    let mut timings = Vec::new();
    for &count in sizes {
        match time_evaluate(backend, program, count) {
            Ok(timing) => timings.push(timing),
            Err(error) => {
                println!("  {name} refused {count} elements: {error:?}");
                return None;
            }
        }
    }
    Some(Sweep {
        backend: name.to_string(),
        timings,
    })
}

fn main() {
    if cfg!(debug_assertions) {
        println!(
            "warning: debug build. Numbers below measure the interpreter's \
             bookkeeping, not the backends. Re-run with --release.\n"
        );
    }

    for (fields, alu_ops, label) in REGIMES {
        let program =
            soma::experiments::backend_bench::synthetic_program(700 + alu_ops, fields, alu_ops);
        let sizes = sizes_for(alu_ops);
        println!(
            "\n=== {label}: {} fields, stride {}B, {} instructions ===",
            fields,
            program.stride(),
            program.ops().len(),
        );

        let mut cpu = CpuReferenceBackend::with(&[&program]);
        let Some(cpu_sweep) = sweep("cpu", &mut cpu, &program, &sizes) else {
            continue;
        };

        let mut sweeps = vec![cpu_sweep];

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match soma::executives::metal::MetalBatchBackend::with(&[&program]) {
                Ok(mut metal) => {
                    if let Some(metal_sweep) = sweep("metal", &mut metal, &program, &sizes) {
                        sweeps.push(metal_sweep);
                    }
                }
                Err(error) => println!("  metal unavailable: {error:?}"),
            }
        }

        print!("{}", render("backend evaluate() alone", &sweeps));

        if sweeps.len() > 1 {
            let name = &sweeps[1].backend;
            match measured_crossover(&sweeps[0], &sweeps[1], 0.15) {
                Some(n) => println!("  {name} first wins by 15% at {n} elements (measured)"),
                None => println!("  {name} never won by 15% at any size measured"),
            }
            let (cpu_model, accel_model) = (sweeps[0].model(), sweeps[1].model());
            if let (Some(cpu_model), Some(accel_model)) = (cpu_model, accel_model) {
                match crossover(&cpu_model, &accel_model, 0.15) {
                    Some(n) => println!("  {n} elements (extrapolated from the fitted lines)"),
                    None => println!(
                        "  the fit says never: {name}'s per-element cost is not below the CPU's",
                    ),
                }
            }
        }
    }

    published_path();
    epochs();
}

/// What `execute_with_spill` adds on top of the backend: the `to_vec` of the
/// frozen input, and freezing the output into a published object.
///
/// Measured against both backends, because the fraction is what matters and
/// the denominators differ by two orders of magnitude. Against the CPU
/// interpreter the copies vanish into the arithmetic; against Metal they are
/// competing with a backend that costs nanoseconds per element, which is the
/// case the unified-memory argument is actually about.
fn published_path() {
    println!("\n=== publication overhead: execute_with_spill vs evaluate() alone ===");
    let program = soma::experiments::backend_bench::synthetic_program(799, 2, 32);
    let sizes = [1_024u32, 65_536, 1_048_576];

    let mut cpu = CpuReferenceBackend::with(&[&program]);
    let mut spill = CpuReferenceBackend::with(&[&program]);

    println!(
        "{:>10} | {:>9} {:>12} {:>12} {:>9}",
        "elements", "backend", "evaluate", "full path", "overhead"
    );
    for count in sizes {
        let bare = time_evaluate(&mut cpu, &program, count).unwrap();
        // `u32::MAX` as the threshold sends everything to the CPU arm, so the
        // difference is publication cost and not a change of backend.
        let full = time_published_path(&program, count, u32::MAX, &mut cpu, &mut spill).unwrap();
        report_overhead(count, "cpu", &bare, &full);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let Ok(mut metal) = soma::executives::metal::MetalBatchBackend::with(&[&program]) else {
            return;
        };
        for count in sizes {
            let bare = time_evaluate(&mut metal, &program, count).unwrap();
            // Threshold 1 sends every batch to the accelerator arm.
            let full = time_published_path(&program, count, 1, &mut metal, &mut spill).unwrap();
            report_overhead(count, "metal", &bare, &full);
        }
    }
}

fn report_overhead(
    count: u32,
    backend: &str,
    bare: &soma::experiments::backend_bench::Timing,
    full: &soma::experiments::backend_bench::Timing,
) {
    let bare_ms = bare.median().as_secs_f64() * 1e3;
    let full_ms = full.median().as_secs_f64() * 1e3;
    println!(
        "{count:>10} | {backend:>9} {bare_ms:>10.3}ms {full_ms:>10.3}ms {:>8.1}%",
        (full_ms - bare_ms) / bare_ms * 100.0
    );
}

/// An epoch of ready cohorts, submitted together or one at a time.
///
/// Both arms go through the kernel and pay the same authorization,
/// publication, and freezing per cohort; the only difference is whether the
/// backend was handed the requests together and could submit them as one unit.
fn epochs() {
    println!("\n=== an epoch of 8192-element cohorts, through the kernel ===");
    let program = soma::experiments::backend_bench::synthetic_program(801, 2, 32);

    println!(
        "{:>8} | {:>9} {:>12} {:>12} {:>9}",
        "cohorts", "backend", "one by one", "as an epoch", "speedup"
    );

    let run = |name: &str, accelerator: &mut dyn BatchBackend, threshold: u32| {
        for cohorts in [1u32, 4, 16, 64] {
            let mut cpu = CpuReferenceBackend::with(&[&program]);
            let single = time_epoch(
                &program,
                cohorts,
                8_192,
                threshold,
                accelerator,
                &mut cpu,
                false,
            )
            .unwrap();
            let batched = time_epoch(
                &program,
                cohorts,
                8_192,
                threshold,
                accelerator,
                &mut cpu,
                true,
            )
            .unwrap();
            let single_ms = single.median().as_secs_f64() * 1e3;
            let batched_ms = batched.median().as_secs_f64() * 1e3;
            println!(
                "{cohorts:>8} | {name:>9} {single_ms:>10.3}ms {batched_ms:>10.3}ms {:>8.2}x",
                single_ms / batched_ms
            );
        }
    };

    let mut cpu_accelerator = CpuReferenceBackend::with(&[&program]);
    run("cpu", &mut cpu_accelerator, u32::MAX);

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Ok(mut metal) = soma::executives::metal::MetalBatchBackend::with(&[&program]) {
            run("metal", &mut metal, 1);
        }
    }
}
