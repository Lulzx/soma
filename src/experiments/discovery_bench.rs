//! Repeated wall-clock benchmarking for the synthetic Discovery workload.
//!
//! The structural regime map in `discovery_search` is deterministic, but a
//! single wall-clock observation is not.  This module repeats every cell,
//! retains the raw samples, and derives percentiles and observed crossovers
//! from medians.  It deliberately does not fit or extrapolate a crossover.

use std::time::Duration;

use crate::discovery::invariants::{verify_pair, DiscoveryInvariantReport};
use crate::discovery::{
    execute_naive, execute_optimized, DiscoveryError, DiscoveryResult, DiscoveryTrace,
};
use crate::executives::batch::BatchBackend;

use super::discovery_search::{
    generate_trace, DiscoveryKnobs, DiscoveryReport, BATCH_SIZES, DUPLICATION_RATES,
    EVALUATOR_CLASSES, PRUNING_RATES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkProtocol {
    pub warmups: u32,
    pub repetitions: u32,
}

impl Default for BenchmarkProtocol {
    fn default() -> Self {
        Self {
            warmups: 2,
            repetitions: 9,
        }
    }
}

impl BenchmarkProtocol {
    pub fn validate(self) -> Result<Self, DiscoveryError> {
        if self.repetitions == 0 {
            return Err(DiscoveryError::InvalidTrace(
                "a discovery benchmark needs at least one repetition",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimingDistribution {
    /// Raw samples in acquisition order. These are the source of every
    /// reported percentile and are intentionally retained for reanalysis.
    pub samples: Vec<Duration>,
}

impl TimingDistribution {
    pub fn percentile(&self, percentile: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let p = percentile.clamp(0.0, 1.0);
        let index = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[index]
    }

    pub fn p10(&self) -> Duration {
        self.percentile(0.10)
    }

    pub fn median(&self) -> Duration {
        self.percentile(0.50)
    }

    pub fn p90(&self) -> Duration {
        self.percentile(0.90)
    }
}

#[derive(Clone, Debug)]
pub struct DiscoveryBenchmarkPoint {
    pub knobs: DiscoveryKnobs,
    pub naive: TimingDistribution,
    pub optimized: TimingDistribution,
    /// The checked invariant report from the final repetition. Every warmup
    /// and measured repetition is also checked and causes the sweep to fail
    /// immediately if any D1-D7 invariant does not hold.
    pub invariants: DiscoveryInvariantReport,
    pub compute_compression: f64,
    pub elimination_rate: f64,
    pub batch_compression: f64,
}

impl DiscoveryBenchmarkPoint {
    pub fn median_speedup(&self) -> f64 {
        let optimized = self.optimized.median().as_secs_f64();
        if optimized == 0.0 {
            f64::INFINITY
        } else {
            self.naive.median().as_secs_f64() / optimized
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiscoveryBenchmarkSweep {
    pub backend: String,
    pub protocol: BenchmarkProtocol,
    pub points: Vec<DiscoveryBenchmarkPoint>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredCrossover {
    pub duplicate_rate: f32,
    pub rejection_rate: f32,
    pub evaluator_classes: u32,
    pub minimum_margin: f64,
    /// `None` means the candidate never cleared the requested margin at a
    /// batch size in the measured grid. It does not mean it can never cross.
    pub elements_per_experiment: Option<u32>,
}

/// Measure every existing Discovery regime using long-lived backend instances.
/// Alternating literal-first and optimized-first acquisition limits systematic
/// order bias without pretending to control temperature or other system load.
pub fn benchmark_sweep(
    backend: impl Into<String>,
    base: DiscoveryKnobs,
    protocol: BenchmarkProtocol,
    naive_backend: &mut dyn BatchBackend,
    optimized_backend: &mut dyn BatchBackend,
) -> Result<DiscoveryBenchmarkSweep, DiscoveryError> {
    let protocol = protocol.validate()?;
    let mut points = Vec::new();
    for duplicate_rate in DUPLICATION_RATES {
        for rejection_rate in PRUNING_RATES {
            for evaluator_classes in EVALUATOR_CLASSES {
                for elements_per_experiment in BATCH_SIZES {
                    let knobs = DiscoveryKnobs {
                        duplicate_rate,
                        rejection_rate,
                        evaluator_classes,
                        elements_per_experiment,
                        ..base
                    };
                    points.push(benchmark_point(
                        knobs,
                        protocol,
                        naive_backend,
                        optimized_backend,
                    )?);
                }
            }
        }
    }
    Ok(DiscoveryBenchmarkSweep {
        backend: backend.into(),
        protocol,
        points,
    })
}

fn benchmark_point(
    knobs: DiscoveryKnobs,
    protocol: BenchmarkProtocol,
    naive_backend: &mut dyn BatchBackend,
    optimized_backend: &mut dyn BatchBackend,
) -> Result<DiscoveryBenchmarkPoint, DiscoveryError> {
    let trace = generate_trace(&knobs);
    for trial in 0..protocol.warmups {
        let _ = execute_checked_pair(&trace, naive_backend, optimized_backend, trial % 2 == 1)?;
    }

    let mut naive_samples = Vec::with_capacity(protocol.repetitions as usize);
    let mut optimized_samples = Vec::with_capacity(protocol.repetitions as usize);
    let mut final_report = None;
    for trial in 0..protocol.repetitions {
        let report = execute_checked_pair(
            &trace,
            naive_backend,
            optimized_backend,
            (trial + protocol.warmups) % 2 == 1,
        )?;
        naive_samples.push(report.naive.metrics.wall_time);
        optimized_samples.push(report.optimized.metrics.wall_time);
        final_report = Some(report);
    }
    let report = final_report.expect("a validated protocol has repetitions");
    let compute_compression = report.compute_compression();
    let elimination_rate = report.elimination_rate();
    let batch_compression = report.optimized.metrics.batch_compression();
    Ok(DiscoveryBenchmarkPoint {
        knobs,
        naive: TimingDistribution {
            samples: naive_samples,
        },
        optimized: TimingDistribution {
            samples: optimized_samples,
        },
        invariants: report.invariants,
        compute_compression,
        elimination_rate,
        batch_compression,
    })
}

fn execute_checked_pair(
    trace: &DiscoveryTrace,
    naive_backend: &mut dyn BatchBackend,
    optimized_backend: &mut dyn BatchBackend,
    optimized_first: bool,
) -> Result<DiscoveryReport, DiscoveryError> {
    let (naive, optimized) = if optimized_first {
        let optimized = execute_optimized(trace, optimized_backend)?;
        let naive = execute_naive(trace, naive_backend)?;
        (naive, optimized)
    } else {
        let naive = execute_naive(trace, naive_backend)?;
        let optimized = execute_optimized(trace, optimized_backend)?;
        (naive, optimized)
    };
    checked_report(trace, naive, optimized)
}

fn checked_report(
    trace: &DiscoveryTrace,
    naive: DiscoveryResult,
    optimized: DiscoveryResult,
) -> Result<DiscoveryReport, DiscoveryError> {
    let invariants = verify_pair(trace, &naive, &optimized);
    if !invariants.all_hold() {
        return Err(DiscoveryError::InvalidTrace(
            "D1-D7 failed during discovery benchmark",
        ));
    }
    Ok(DiscoveryReport {
        naive,
        optimized,
        invariants,
    })
}

/// First measured batch size where optimized replay beats literal replay by
/// at least `minimum_margin`, grouped across the other three regime axes.
pub fn policy_crossovers(
    sweep: &DiscoveryBenchmarkSweep,
    minimum_margin: f64,
) -> Vec<MeasuredCrossover> {
    crossover_by(
        &sweep.points,
        &sweep.points,
        minimum_margin,
        |point| point.naive.median(),
        |point| point.optimized.median(),
    )
}

/// First measured batch size where the candidate backend's optimized replay
/// beats the baseline backend's optimized replay by at least `minimum_margin`.
pub fn backend_crossovers(
    baseline: &DiscoveryBenchmarkSweep,
    candidate: &DiscoveryBenchmarkSweep,
    minimum_margin: f64,
) -> Result<Vec<MeasuredCrossover>, DiscoveryError> {
    if baseline.points.len() != candidate.points.len()
        || baseline
            .points
            .iter()
            .zip(&candidate.points)
            .any(|(a, b)| !same_regime(&a.knobs, &b.knobs))
    {
        return Err(DiscoveryError::InvalidTrace(
            "backend crossover sweeps use different regime grids",
        ));
    }
    Ok(crossover_by(
        &baseline.points,
        &candidate.points,
        minimum_margin,
        |point| point.optimized.median(),
        |point| point.optimized.median(),
    ))
}

fn crossover_by(
    baseline: &[DiscoveryBenchmarkPoint],
    candidate: &[DiscoveryBenchmarkPoint],
    minimum_margin: f64,
    baseline_time: impl Fn(&DiscoveryBenchmarkPoint) -> Duration,
    candidate_time: impl Fn(&DiscoveryBenchmarkPoint) -> Duration,
) -> Vec<MeasuredCrossover> {
    let margin = minimum_margin.clamp(0.0, 1.0);
    let mut out = Vec::new();
    for duplicate_rate in DUPLICATION_RATES {
        for rejection_rate in PRUNING_RATES {
            for evaluator_classes in EVALUATOR_CLASSES {
                let crossing = baseline
                    .iter()
                    .zip(candidate)
                    .filter(|(a, _)| {
                        a.knobs.duplicate_rate == duplicate_rate
                            && a.knobs.rejection_rate == rejection_rate
                            && a.knobs.evaluator_classes == evaluator_classes
                    })
                    .find(|(a, b)| {
                        candidate_time(b).as_secs_f64()
                            <= baseline_time(a).as_secs_f64() * (1.0 - margin)
                    })
                    .map(|(point, _)| point.knobs.elements_per_experiment);
                out.push(MeasuredCrossover {
                    duplicate_rate,
                    rejection_rate,
                    evaluator_classes,
                    minimum_margin: margin,
                    elements_per_experiment: crossing,
                });
            }
        }
    }
    out
}

fn same_regime(a: &DiscoveryKnobs, b: &DiscoveryKnobs) -> bool {
    a.branching_factor == b.branching_factor
        && a.depth == b.depth
        && a.duplicate_rate == b.duplicate_rate
        && a.shared_prefix_rate == b.shared_prefix_rate
        && a.rejection_rate == b.rejection_rate
        && a.rejection_depth == b.rejection_depth
        && a.evaluator_classes == b.evaluator_classes
        && a.elements_per_experiment == b.elements_per_experiment
        && a.arrival_skew == b.arrival_skew
        && a.observation_rate == b.observation_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executives::batch::CpuReferenceBackend;
    use crate::experiments::discovery_search::evaluator_programs;

    fn point(elements: u32, naive: u64, optimized: u64) -> DiscoveryBenchmarkPoint {
        DiscoveryBenchmarkPoint {
            knobs: DiscoveryKnobs {
                duplicate_rate: 0.0,
                rejection_rate: 0.0,
                evaluator_classes: 1,
                elements_per_experiment: elements,
                ..Default::default()
            },
            naive: TimingDistribution {
                samples: vec![Duration::from_nanos(naive)],
            },
            optimized: TimingDistribution {
                samples: vec![Duration::from_nanos(optimized)],
            },
            invariants: DiscoveryInvariantReport::default(),
            compute_compression: 0.0,
            elimination_rate: 0.0,
            batch_compression: 0.0,
        }
    }

    #[test]
    fn percentile_selects_an_observed_sample() {
        let distribution = TimingDistribution {
            samples: [9, 1, 7, 3, 5]
                .into_iter()
                .map(Duration::from_nanos)
                .collect(),
        };
        assert_eq!(distribution.p10(), Duration::from_nanos(1));
        assert_eq!(distribution.median(), Duration::from_nanos(5));
        assert_eq!(distribution.p90(), Duration::from_nanos(9));
    }

    #[test]
    fn crossover_is_a_measured_size_and_respects_the_margin() {
        let mut points = vec![point(1, 100, 95), point(64, 100, 84), point(1_024, 100, 70)];
        // Supply the remaining groups as empty-by-filter; only inspect the
        // first group returned by the fixed public grid.
        points.sort_by_key(|point| point.knobs.elements_per_experiment);
        let sweep = DiscoveryBenchmarkSweep {
            backend: "test".into(),
            protocol: BenchmarkProtocol {
                warmups: 0,
                repetitions: 1,
            },
            points,
        };
        assert_eq!(
            policy_crossovers(&sweep, 0.15)[0].elements_per_experiment,
            Some(64)
        );
    }

    #[test]
    fn every_measured_sample_keeps_d1_through_d7() {
        let knobs = DiscoveryKnobs {
            branching_factor: 1,
            depth: 1,
            evaluator_classes: 1,
            elements_per_experiment: 1,
            ..Default::default()
        };
        let programs = evaluator_programs(1);
        let refs: Vec<_> = programs.iter().collect();
        let mut naive = CpuReferenceBackend::with(&refs);
        let mut optimized = CpuReferenceBackend::with(&refs);
        let point = benchmark_point(
            knobs,
            BenchmarkProtocol {
                warmups: 1,
                repetitions: 3,
            },
            &mut naive,
            &mut optimized,
        )
        .unwrap();
        assert_eq!(point.naive.samples.len(), 3);
        assert!(point.invariants.all_hold());
    }
}
