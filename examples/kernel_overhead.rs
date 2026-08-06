//! What a published cohort costs outside the backend.
//!
//! With submission batched, an epoch of sixty-four cohorts spends about 4.2ms
//! of its 4.5ms in the kernel rather than on the GPU. That is now the whole
//! cost, and it is spread across the operations every cohort performs
//! regardless of where it ran: creating the output object, freezing it,
//! minting and checking capabilities, completing the collective, and tracing
//! all of it.
//!
//! Each stage is timed on its own here, at two batch sizes, because the two
//! answers are different questions. At 64KB the per-cohort cost may be
//! dominated by bytes; at 64 bytes nothing is left but bookkeeping, and
//! whatever remains there is what an epoch of many small cohorts pays.
//!
//!     cargo run --release --example kernel_overhead

use std::time::{Duration, Instant};

use soma::abi::{ObjectKind, ProcessMode, Ref64};
use soma::executives::batch::{
    execute_epoch_with_spill, BatchBackend, CpuReferenceBackend, PlacementStats,
};
use soma::experiments::backend_bench::{synthetic_inputs, synthetic_program};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

fn median(reps: usize, mut body: impl FnMut()) -> (Duration, Duration) {
    for _ in 0..8 {
        body();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed());
    }
    samples.sort();
    (samples[samples.len() / 2], samples[0])
}

fn line(name: &str, (median, min): (Duration, Duration)) {
    println!(
        "{name:>38} {:>9.2}µs {:>9.2}µs",
        median.as_secs_f64() * 1e6,
        min.as_secs_f64() * 1e6,
    );
}

fn main() {
    if cfg!(debug_assertions) {
        println!("warning: debug build, re-run with --release\n");
    }

    // 8192 elements is the cohort size the epoch benchmark uses; 8 elements is
    // the same work with the bytes taken away.
    for elements in [8_192u32, 8] {
        let program = synthetic_program(810, 2, 32);
        let stride = program.stride();
        let bytes = synthetic_inputs(elements, stride);
        println!(
            "\n=== one cohort of {elements} elements ({} bytes) ===\n{:>38} {:>11} {:>11}",
            bytes.len(),
            "stage",
            "median",
            "fastest"
        );

        // The allocation alone, so the object-creation row below can be read
        // as bookkeeping rather than as malloc.
        line(
            "vec![] of the payload",
            median(200, || {
                std::hint::black_box(bytes.clone());
            }),
        );

        let mut kernel = Kernel::new();
        let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

        line(
            "create_object (frozen array)",
            median(200, || {
                let object = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
                std::hint::black_box(object);
            }),
        );

        let mut unfrozen: Vec<Ref64> = (0..240)
            .map(|_| kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone()))
            .collect();
        line(
            "freeze",
            median(200, || {
                let object = unfrozen.pop().expect("enough objects staged");
                freeze(&mut kernel, owner, object).unwrap();
            }),
        );

        let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
        freeze(&mut kernel, owner, input).unwrap();

        line(
            "create_batch_evaluate_for",
            median(200, || {
                let created = kernel
                    .create_batch_evaluate_for(owner, program.id(), input, elements, stride)
                    .unwrap();
                std::hint::black_box(created);
            }),
        );

        line(
            "object_bytes_many (one object)",
            median(200, || {
                let read = kernel.object_bytes_many(owner, &[input]).unwrap();
                std::hint::black_box(read.len());
            }),
        );

        // The whole path for one cohort, so the rows above can be checked
        // against the total they are meant to explain.
        let mut accelerator = CpuReferenceBackend::with(&[&program]);
        let mut cpu = CpuReferenceBackend::with(&[&program]);
        line(
            "execute_epoch_with_spill (1 cohort)",
            median(200, || {
                let (collective, _) = kernel
                    .create_batch_evaluate_for(owner, program.id(), input, elements, stride)
                    .unwrap();
                let published = execute_epoch_with_spill(
                    &mut kernel,
                    owner,
                    &[collective],
                    u32::MAX,
                    &mut accelerator,
                    &mut cpu,
                    &mut PlacementStats::default(),
                )
                .unwrap();
                std::hint::black_box(published);
            }),
        );

        // And the backend's share of that total, measured the same way.
        line(
            "  of which: the CPU backend",
            median(200, || {
                let outputs = cpu
                    .evaluate(program.id(), &bytes, elements, stride)
                    .unwrap();
                std::hint::black_box(outputs);
            }),
        );

        println!("  trace rows so far: {}", kernel.trace_snapshot().len());
    }

    growth();
}

/// Whether a cohort costs more when the kernel already holds more.
///
/// The stage timings above are taken on a nearly empty kernel. A long run is
/// not: every published batch adds an object, its capabilities, and its trace
/// rows, and every subsequent authorization has to find a capability among
/// however many now exist. If that search is linear, per-cohort cost grows
/// with the age of the run, which no single-shot measurement would show and
/// which would make a long epoch sequence quietly quadratic.
fn growth() {
    println!("\n=== does a cohort cost more as the kernel fills up? ===");
    let program = synthetic_program(811, 2, 32);
    let stride = program.stride();
    let elements = 8u32;
    let bytes = synthetic_inputs(elements, stride);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
    freeze(&mut kernel, owner, input).unwrap();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    println!(
        "{:>38} {:>11} {:>11}",
        "published cohorts so far", "median", "fastest"
    );
    let mut published = 0u32;
    for target in [0u32, 1_000, 4_000, 16_000] {
        while published < target {
            let (collective, _) = kernel
                .create_batch_evaluate_for(owner, program.id(), input, elements, stride)
                .unwrap();
            execute_epoch_with_spill(
                &mut kernel,
                owner,
                &[collective],
                u32::MAX,
                &mut accelerator,
                &mut cpu,
                &mut PlacementStats::default(),
            )
            .unwrap();
            published += 1;
        }
        println!(
            "  capabilities held: {}, of them naming the one input: {}",
            kernel.capability_count(),
            kernel.capabilities_naming(input),
        );
        line(
            &format!("{target}: whole cohort"),
            median(200, || {
                let (collective, _) = kernel
                    .create_batch_evaluate_for(owner, program.id(), input, elements, stride)
                    .unwrap();
                let out = execute_epoch_with_spill(
                    &mut kernel,
                    owner,
                    &[collective],
                    u32::MAX,
                    &mut accelerator,
                    &mut cpu,
                    &mut PlacementStats::default(),
                )
                .unwrap();
                std::hint::black_box(out);
            }),
        );
        // Which stage is the one that grows.
        line(
            &format!("{target}:   create_batch_evaluate_for"),
            median(200, || {
                let created = kernel
                    .create_batch_evaluate_for(owner, program.id(), input, elements, stride)
                    .unwrap();
                std::hint::black_box(created);
            }),
        );
        line(
            &format!("{target}:   object_bytes_many"),
            median(200, || {
                let read = kernel.object_bytes_many(owner, &[input]).unwrap();
                std::hint::black_box(read.len());
            }),
        );
        line(
            &format!("{target}:   create_object alone"),
            median(200, || {
                let object = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
                std::hint::black_box(object);
            }),
        );
        let mut staged: Vec<Ref64> = (0..240)
            .map(|_| kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone()))
            .collect();
        line(
            &format!("{target}:   freeze alone"),
            median(200, || {
                let object = staged.pop().expect("enough staged");
                freeze(&mut kernel, owner, object).unwrap();
            }),
        );
        published += 200;
    }
}
