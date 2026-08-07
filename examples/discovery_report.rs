use soma::experiments::discovery_search::{self, DiscoveryKnobs, DiscoveryReport};

fn main() {
    let knobs = DiscoveryKnobs::default();
    let report = run(&knobs).expect("discovery replay should complete");
    print_report(&report);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run(knobs: &DiscoveryKnobs) -> Result<DiscoveryReport, soma::discovery::DiscoveryError> {
    use soma::executives::metal::MetalBatchBackend;

    let programs = discovery_search::evaluator_programs(knobs.evaluator_classes);
    let refs: Vec<_> = programs.iter().collect();
    let mut naive =
        MetalBatchBackend::with(&refs).map_err(soma::discovery::DiscoveryError::Backend)?;
    let mut optimized =
        MetalBatchBackend::with(&refs).map_err(soma::discovery::DiscoveryError::Backend)?;
    discovery_search::run_with_backend(knobs, &mut naive, &mut optimized)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run(knobs: &DiscoveryKnobs) -> Result<DiscoveryReport, soma::discovery::DiscoveryError> {
    discovery_search::run_cpu(knobs)
}

fn print_report(report: &DiscoveryReport) {
    let naive = &report.naive.metrics;
    let optimized = &report.optimized.metrics;
    println!("SOMA discovery replay");
    println!(
        "logical experiment requests  {:>10}",
        optimized.logical_requests
    );
    println!(
        "unique deterministic nodes   {:>10}",
        optimized.unique_deterministic_nodes
    );
    println!("cache hits                   {:>10}", optimized.cache_hits);
    println!(
        "pending-request joins        {:>10}",
        optimized.pending_request_joins
    );
    println!(
        "cancelled before execution   {:>10}",
        optimized.cancelled_before_execution
    );
    println!(
        "physical executions          {:>10}",
        optimized.physical_evaluator_executions
    );
    println!(
        "physical dispatches          {:>10}",
        optimized.physical_dispatches
    );
    println!(
        "command buffers              {:>10}",
        optimized.command_buffers
    );
    println!("input bytes                  {:>10}", optimized.input_bytes);
    println!(
        "output bytes                 {:>10}",
        optimized.output_bytes
    );
    println!(
        "bytes transferred            {:>10}",
        optimized.bytes_transferred
    );
    println!(
        "peak pending bytes           {:>10}",
        optimized.peak_pending_bytes
    );
    println!(
        "compute compression          {:>10.3}x",
        report.compute_compression()
    );
    println!(
        "elimination rate             {:>9.2}%",
        report.elimination_rate() * 100.0
    );
    println!(
        "batch compression            {:>10.3}x",
        optimized.batch_compression()
    );
    println!(
        "naive wall time              {:>10.3}ms",
        naive.wall_time.as_secs_f64() * 1e3
    );
    println!(
        "optimized wall time          {:>10.3}ms",
        optimized.wall_time.as_secs_f64() * 1e3
    );
    println!(
        "optimized CPU backend time   {:>10.3}ms",
        optimized.cpu_time.as_secs_f64() * 1e3
    );
    println!(
        "optimized GPU backend time   {:>10.3}ms",
        optimized.gpu_time.as_secs_f64() * 1e3
    );
    println!(
        "D1-D7 hold                   {:>10}",
        report.invariants.all_hold()
    );
}
