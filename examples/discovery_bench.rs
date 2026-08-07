//! Repeated release benchmark for the complete synthetic Discovery regime map.
//!
//!     cargo run --release --example discovery_bench
//!     cargo run --release --features metal --example discovery_bench
//!
//! `SOMA_DISCOVERY_WARMUPS` and `SOMA_DISCOVERY_REPETITIONS` override the
//! default protocol. Output is TSV with raw nanosecond samples retained in the
//! last two columns.

use std::time::Duration;

use soma::executives::batch::CpuReferenceBackend;
use soma::experiments::discovery_bench::{
    benchmark_sweep, policy_crossovers, BenchmarkProtocol, DiscoveryBenchmarkSweep,
    MeasuredCrossover,
};
use soma::experiments::discovery_search::{evaluator_programs, DiscoveryKnobs};

#[cfg(all(feature = "metal", target_os = "macos"))]
use soma::experiments::discovery_bench::backend_crossovers;

const CROSSOVER_MARGIN: f64 = 0.15;

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("warning: use --release; debug timings are not benchmark results");
    }
    let protocol = BenchmarkProtocol {
        warmups: env_u32("SOMA_DISCOVERY_WARMUPS", 2),
        repetitions: env_u32("SOMA_DISCOVERY_REPETITIONS", 9),
    };
    let base = DiscoveryKnobs {
        branching_factor: 2,
        depth: 3,
        elements_per_experiment: 1,
        ..Default::default()
    };
    let programs = evaluator_programs(16);
    let refs: Vec<_> = programs.iter().collect();

    let mut cpu_naive = CpuReferenceBackend::with(&refs);
    let mut cpu_optimized = CpuReferenceBackend::with(&refs);
    let cpu = benchmark_sweep(
        "cpu-reference",
        base,
        protocol,
        &mut cpu_naive,
        &mut cpu_optimized,
    )
    .expect("CPU Discovery benchmark should complete with D1-D7 holding");

    println!("# soma-discovery-benchmark-v1");
    println!(
        "# warmups={}\trepetitions={}\tbranching_factor={}\tdepth={}\tcrossover_margin={:.2}",
        protocol.warmups, protocol.repetitions, base.branching_factor, base.depth, CROSSOVER_MARGIN,
    );
    print_sweep(&cpu);
    print_crossovers(
        "optimized-vs-literal",
        &cpu.backend,
        policy_crossovers(&cpu, CROSSOVER_MARGIN),
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use soma::executives::metal::MetalBatchBackend;

        match (
            MetalBatchBackend::with(&refs),
            MetalBatchBackend::with(&refs),
        ) {
            (Ok(mut metal_naive), Ok(mut metal_optimized)) => {
                let metal = benchmark_sweep(
                    "metal",
                    base,
                    protocol,
                    &mut metal_naive,
                    &mut metal_optimized,
                )
                .expect("Metal Discovery benchmark should complete with D1-D7 holding");
                print_sweep(&metal);
                print_crossovers(
                    "optimized-vs-literal",
                    &metal.backend,
                    policy_crossovers(&metal, CROSSOVER_MARGIN),
                );
                let crossovers = backend_crossovers(&cpu, &metal, CROSSOVER_MARGIN)
                    .expect("CPU and Metal sweeps use the same grid");
                print_crossovers("metal-vs-cpu-optimized", "metal", crossovers);
            }
            (Err(error), _) | (_, Err(error)) => eprintln!("Metal unavailable: {error:?}"),
        }
    }
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn print_sweep(sweep: &DiscoveryBenchmarkSweep) {
    println!("# points\tbackend={}", sweep.backend);
    println!(
        "kind\tbackend\tduplicate\tpruning\tclasses\telements\trepetitions\tnaive_p10_ms\tnaive_median_ms\tnaive_p90_ms\toptimized_p10_ms\toptimized_median_ms\toptimized_p90_ms\tmedian_speedup\tcompute_compression\telimination\tbatch_compression\td1_d7\tnaive_raw_ns\toptimized_raw_ns"
    );
    for point in &sweep.points {
        println!(
            "point\t{}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}",
            sweep.backend,
            point.knobs.duplicate_rate,
            point.knobs.rejection_rate,
            point.knobs.evaluator_classes,
            point.knobs.elements_per_experiment,
            sweep.protocol.repetitions,
            millis(point.naive.p10()),
            millis(point.naive.median()),
            millis(point.naive.p90()),
            millis(point.optimized.p10()),
            millis(point.optimized.median()),
            millis(point.optimized.p90()),
            point.median_speedup(),
            point.compute_compression,
            point.elimination_rate,
            point.batch_compression,
            point.invariants.all_hold(),
            raw(&point.naive.samples),
            raw(&point.optimized.samples),
        );
    }
}

fn print_crossovers(kind: &str, candidate: &str, crossovers: Vec<MeasuredCrossover>) {
    println!("# crossovers\tkind={kind}\tcandidate={candidate}");
    println!(
        "record\tcomparison\tcandidate\tduplicate\tpruning\tclasses\tminimum_margin\tfirst_measured_elements"
    );
    for crossover in crossovers {
        let elements = crossover
            .elements_per_experiment
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not-observed".to_string());
        println!(
            "crossover\t{kind}\t{candidate}\t{:.2}\t{:.2}\t{}\t{:.2}\t{elements}",
            crossover.duplicate_rate,
            crossover.rejection_rate,
            crossover.evaluator_classes,
            crossover.minimum_margin,
        );
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn raw(samples: &[Duration]) -> String {
    samples
        .iter()
        .map(|sample| sample.as_nanos().to_string())
        .collect::<Vec<_>>()
        .join(",")
}
