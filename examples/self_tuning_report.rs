use soma::experiments::self_tuning::{self, SelfTuningReport, TuningStudy};

fn main() {
    let study = TuningStudy::default();
    let parallel = std::thread::available_parallelism()
        .map(|count| count.get().min(8))
        .unwrap_or(1);
    let threads = if parallel == 1 {
        vec![1]
    } else {
        vec![1, parallel]
    };
    let report = run(&study, &threads).expect("self-tuning study should complete");
    print_report(&report);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run(
    study: &TuningStudy,
    threads: &[usize],
) -> Result<SelfTuningReport, soma::discovery::DiscoveryError> {
    self_tuning::run_metal(study, threads)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run(
    study: &TuningStudy,
    threads: &[usize],
) -> Result<SelfTuningReport, soma::discovery::DiscoveryError> {
    self_tuning::run_cpu(study, threads)
}

fn print_report(report: &SelfTuningReport) {
    println!("SOMA self-tuning discovery study");
    println!("trials per configuration: {}", report.captured.study.trials);
    println!("configurations: {}", report.captured.configs.len());
    println!("observations: {}", report.captured.observations.len());
    println!("D1-D7 all hold: {}", report.invariants.all_hold());
    println!(
        "scientific states equal: {}",
        report.naive.scientific == report.optimized.scientific
    );
    println!(
        "deterministic preparation: {} logical, {} physical",
        report.captured.study.workloads.len() * report.captured.configs.len() * 2,
        report.optimized.metrics.deterministic_physical_executions
    );
    println!(
        "independent observations executed: {}",
        report.optimized.metrics.physical_evaluator_executions
            - report.optimized.metrics.deterministic_physical_executions
    );

    for ranking in &report.rankings {
        let workload = report
            .captured
            .study
            .workloads
            .iter()
            .find(|workload| workload.id == ranking.workload_id)
            .unwrap();
        println!(
            "\nworkload {}: {} ALU ops, {} x {} elements",
            workload.id, workload.alu_ops, workload.cohorts, workload.elements_per_cohort
        );
        let baseline = ranking
            .configs
            .iter()
            .find(|score| score.config_id == 1)
            .map(|score| score.median_nanos)
            .unwrap_or(ranking.configs[0].median_nanos);
        for score in &ranking.configs {
            let config = report
                .captured
                .configs
                .iter()
                .find(|config| config.id == score.config_id)
                .unwrap();
            println!(
                "  {:<36} {:>10.3} ms  {:>6.2}x vs cpu/1t/single",
                config.label(),
                score.median_nanos as f64 / 1e6,
                baseline as f64 / score.median_nanos.max(1) as f64,
            );
        }
    }
}
