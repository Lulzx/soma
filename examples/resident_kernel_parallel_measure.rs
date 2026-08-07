//! Physical-device measurement of canonical owned-`Kernel` resident execution.
//!
//! Run with:
//! `cargo run --release --example resident_kernel_parallel_measure --features metal,resident-sync-measurement`

#[cfg(not(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
)))]
fn main() {
    eprintln!(
        "resident_kernel_parallel_measure requires macOS and --features metal,resident-sync-measurement"
    );
    std::process::exit(1);
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
use soma::abi::{ProcessMode, Ref64, StateAccess};
#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
use soma::executives::resident_sync::{
    HANDLER_COMPLETE, HANDLER_STORE_IMMEDIATE_U64, HANDLER_YIELD,
};
#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
use soma::kernel::resident_sync::{
    measurement, KernelResidentInstruction, KernelResidentMetalExecutor, KernelResidentProgram,
    KernelResidentSyncPlan,
};
#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
use soma::kernel::{ContinuationSpec, Kernel};
#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
use std::time::Instant;

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
const SAMPLE_COUNT: usize = 9;
#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
const MIN_QUALIFYING_NS: u128 = 20_000_000;

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
#[derive(Clone)]
struct Row {
    sample: usize,
    order: &'static str,
    backend: &'static str,
    elapsed_ns: u128,
    hash: [u8; 32],
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn workload() -> (Kernel, KernelResidentSyncPlan) {
    let mut kernel = Kernel::new();
    let process = kernel.create_process(Ref64::NULL, ProcessMode::Pure);
    for epoch in 0..16_u32 {
        let run_class = 2000 + epoch;
        let mut instructions = Vec::with_capacity(256);
        for value in 0..255_u64 {
            instructions.push(KernelResidentInstruction::plain(
                HANDLER_STORE_IMMEDIATE_U64,
                0,
                (u64::from(epoch) << 32) | value,
            ));
        }
        instructions.push(KernelResidentInstruction::plain(
            if epoch == 15 {
                HANDLER_COMPLETE
            } else {
                HANDLER_YIELD
            },
            if epoch == 15 { 0 } else { run_class + 1 },
            0,
        ));
        assert_eq!(instructions.len(), 256);
        kernel
            .install_resident_sync_program(KernelResidentProgram {
                run_class,
                instructions,
            })
            .unwrap();
    }
    for _ in 0..4096 {
        kernel
            .create_continuation(
                process,
                process,
                ContinuationSpec::new(StateAccess::ReadOnly, 2000, 0, vec![0; 8], 16),
            )
            .unwrap();
    }
    let plan = kernel.plan_resident_sync(16, 1, 8, 32).unwrap();
    (kernel, plan)
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn run_cpu(base: &Kernel, plan: &KernelResidentSyncPlan) -> (Row, Kernel, u32) {
    let mut kernel = base.clone();
    let owned_plan = plan.clone();
    let start = Instant::now();
    let epochs = measurement::execute_cpu_reference(&mut kernel, owned_plan).unwrap();
    let elapsed_ns = start.elapsed().as_nanos();
    let hash = measurement::deterministic_state_hash(&kernel);
    (
        Row {
            sample: 0,
            order: "",
            backend: "cpu_reference",
            elapsed_ns,
            hash,
        },
        kernel,
        epochs,
    )
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn run_metal(
    executor: &KernelResidentMetalExecutor,
    base: &Kernel,
    plan: &KernelResidentSyncPlan,
) -> (Row, Kernel, u32) {
    let mut kernel = base.clone();
    let owned_plan = plan.clone();
    let start = Instant::now();
    let epochs = executor.execute(&mut kernel, owned_plan).unwrap();
    let elapsed_ns = start.elapsed().as_nanos();
    let hash = measurement::deterministic_state_hash(&kernel);
    (
        Row {
            sample: 0,
            order: "",
            backend: "metal",
            elapsed_ns,
            hash,
        },
        kernel,
        epochs,
    )
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn bootstrap_ci(cpu: &[u128], metal: &[u128]) -> (f64, f64) {
    let mut seed = 0x6d_61_62_30_30_37_u64;
    let mut estimates = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let mut resampled_cpu = Vec::with_capacity(SAMPLE_COUNT);
        let mut resampled_metal = Vec::with_capacity(SAMPLE_COUNT);
        for _ in 0..SAMPLE_COUNT {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let index = (seed as usize) % SAMPLE_COUNT;
            resampled_cpu.push(cpu[index]);
            resampled_metal.push(metal[index]);
        }
        estimates.push(median(&mut resampled_cpu) as f64 / median(&mut resampled_metal) as f64);
    }
    estimates.sort_by(f64::total_cmp);
    (estimates[249], estimates[9749])
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn hex(hash: [u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn csv_field(value: &str) -> String {
    value.replace([',', '\n', '\r'], "_")
}

#[cfg(all(
    feature = "metal",
    feature = "resident-sync-measurement",
    target_os = "macos"
))]
fn main() {
    let (base, plan) = workload();
    let executor = KernelResidentMetalExecutor::new().unwrap();

    // Exactly one untimed warm-up of each canonical backend.
    let (_, warm_cpu, warm_cpu_epochs) = run_cpu(&base, &plan);
    let (_, warm_metal, warm_metal_epochs) = run_metal(&executor, &base, &plan);
    assert_eq!(warm_cpu_epochs, 16);
    assert_eq!(warm_metal_epochs, 16);
    assert!(soma::semantics::invariants::check(&warm_cpu).is_empty());
    assert!(soma::semantics::invariants::check(&warm_metal).is_empty());
    assert!(soma::semantics::order::placement_neutral(&[&warm_cpu, &warm_metal]).is_empty());
    assert_eq!(
        measurement::deterministic_state_hash(&warm_cpu),
        measurement::deterministic_state_hash(&warm_metal)
    );

    let mut rows = Vec::with_capacity(SAMPLE_COUNT * 2);
    let mut cpu_times = Vec::with_capacity(SAMPLE_COUNT);
    let mut metal_times = Vec::with_capacity(SAMPLE_COUNT);
    let mut expected_hash = None;
    for sample in 1..=SAMPLE_COUNT {
        let cpu_first = sample % 2 == 1;
        let order = if cpu_first { "AB" } else { "BA" };
        let (first, second) = if cpu_first {
            let cpu = run_cpu(&base, &plan);
            let metal = run_metal(&executor, &base, &plan);
            (cpu, metal)
        } else {
            let metal = run_metal(&executor, &base, &plan);
            let cpu = run_cpu(&base, &plan);
            (metal, cpu)
        };
        let mut pair = [first, second];
        let cpu_index = usize::from(!cpu_first);
        let metal_index = usize::from(cpu_first);
        assert_eq!(pair[cpu_index].2, 16);
        assert_eq!(pair[metal_index].2, 16);
        assert!(soma::semantics::invariants::check(&pair[cpu_index].1).is_empty());
        assert!(soma::semantics::invariants::check(&pair[metal_index].1).is_empty());
        assert!(soma::semantics::order::placement_neutral(&[
            &pair[cpu_index].1,
            &pair[metal_index].1,
        ])
        .is_empty());
        assert_eq!(pair[cpu_index].0.hash, pair[metal_index].0.hash);
        if let Some(hash) = expected_hash {
            assert_eq!(pair[cpu_index].0.hash, hash);
        } else {
            expected_hash = Some(pair[cpu_index].0.hash);
        }
        cpu_times.push(pair[cpu_index].0.elapsed_ns);
        metal_times.push(pair[metal_index].0.elapsed_ns);
        for item in &mut pair {
            item.0.sample = sample;
            item.0.order = order;
            rows.push(item.0.clone());
        }
    }

    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let device = metal::Device::system_default()
        .map(|device| device.name().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let config = "n4096_classes2000-2015_instr256_epochs16_effects1_frame8_width32_steps16";
    println!("commit,device,config,sample,order,backend,elapsed_ns,hash");
    for row in &rows {
        println!(
            "{},{},{},{},{},{},{},{}",
            csv_field(&commit),
            csv_field(&device),
            config,
            row.sample,
            row.order,
            row.backend,
            row.elapsed_ns,
            hex(row.hash)
        );
    }
    let mut cpu_for_median = cpu_times.clone();
    let mut metal_for_median = metal_times.clone();
    let median_cpu = median(&mut cpu_for_median);
    let median_metal = median(&mut metal_for_median);
    let speedup = median_cpu as f64 / median_metal as f64;
    let (ci_low, ci_high) = bootstrap_ci(&cpu_times, &metal_times);
    let qualifying = rows.iter().all(|row| row.elapsed_ns >= MIN_QUALIFYING_NS);
    println!(
        "# summary,samples={},median_cpu_ns={},median_metal_ns={},median_speedup={:.6},bootstrap95_low={:.6},bootstrap95_high={:.6},qualifying_20ms={}",
        SAMPLE_COUNT, median_cpu, median_metal, speedup, ci_low, ci_high, qualifying
    );
}
