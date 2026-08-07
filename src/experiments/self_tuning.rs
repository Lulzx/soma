//! SOMA as its own first real discovery target.
//!
//! Timing acquisition and Discovery replay are separate phases. Acquisition
//! runs each physical configuration and records every independent wall-clock
//! sample exactly once. Replay then presents those captured samples as
//! `Observation` nodes while evaluator generation and input preparation are
//! deterministic nodes shared across configurations. This prevents executor
//! overhead or a second benchmark run from changing the evidence being
//! compared.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::abi::{ObjectKind, ProcessMode, Ref64};
use crate::compiler::body::{ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};
use crate::discovery::invariants::{verify_pair, DiscoveryInvariantReport};
use crate::discovery::{
    execute_naive, execute_optimized, DiscoveryError, DiscoveryEvent, DiscoveryNode,
    DiscoveryResult, DiscoveryTrace, EvaluationSpec, FusionClass, ModuleDigest,
};
use crate::executives::batch::{
    execute_epoch_with_spill, execute_with_spill, BackendError, BatchBackend, CpuReferenceBackend,
    PlacementStats,
};
use crate::experiments::backend_bench::{synthetic_inputs, synthetic_program};
use crate::kernel::ownership::freeze;
use crate::kernel::{Kernel, SYSTEM_PRINCIPAL};

const REPLAY_EVALUATOR: u32 = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Placement {
    Cpu,
    Metal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionConfig {
    pub id: u32,
    pub placement: Placement,
    pub batched_epoch: bool,
    pub cpu_threads: usize,
    pub threadgroup_width: Option<u64>,
    pub reuse_scratch_buffers: bool,
}

impl ExecutionConfig {
    pub fn label(&self) -> String {
        match self.placement {
            Placement::Cpu => format!(
                "cpu/{}t/{}",
                self.cpu_threads,
                if self.batched_epoch {
                    "epoch"
                } else {
                    "single"
                }
            ),
            Placement::Metal => format!(
                "metal/tg-{}/{}-buffers/{}",
                self.threadgroup_width
                    .map_or_else(|| "auto".to_string(), |width| width.to_string()),
                if self.reuse_scratch_buffers {
                    "warm"
                } else {
                    "fresh"
                },
                if self.batched_epoch {
                    "epoch"
                } else {
                    "single"
                }
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuningWorkload {
    pub id: u32,
    pub alu_ops: u32,
    pub cohorts: u32,
    pub elements_per_cohort: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningStudy {
    pub workloads: Vec<TuningWorkload>,
    pub trials: u32,
}

impl Default for TuningStudy {
    fn default() -> Self {
        Self {
            workloads: vec![
                TuningWorkload {
                    id: 1,
                    alu_ops: 8,
                    cohorts: 1,
                    elements_per_cohort: 1_024,
                },
                TuningWorkload {
                    id: 2,
                    alu_ops: 256,
                    cohorts: 16,
                    elements_per_cohort: 8_192,
                },
                TuningWorkload {
                    id: 3,
                    alu_ops: 2_048,
                    cohorts: 8,
                    elements_per_cohort: 131_072,
                },
            ],
            trials: 9,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimingObservation {
    pub workload_id: u32,
    pub config_id: u32,
    pub trial: u32,
    pub elapsed_nanos: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedStudy {
    pub study: TuningStudy,
    pub configs: Vec<ExecutionConfig>,
    pub observations: Vec<TimingObservation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigScore {
    pub config_id: u32,
    pub median_nanos: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkloadRanking {
    pub workload_id: u32,
    pub configs: Vec<ConfigScore>,
}

#[derive(Clone, Debug)]
pub struct SelfTuningReport {
    pub captured: CapturedStudy,
    pub naive: DiscoveryResult,
    pub optimized: DiscoveryResult,
    pub invariants: DiscoveryInvariantReport,
    pub rankings: Vec<WorkloadRanking>,
}

pub fn cpu_configurations(thread_counts: &[usize]) -> Vec<ExecutionConfig> {
    let mut configs = Vec::new();
    for &threads in thread_counts {
        for batched_epoch in [false, true] {
            configs.push(ExecutionConfig {
                id: configs.len() as u32 + 1,
                placement: Placement::Cpu,
                batched_epoch,
                cpu_threads: threads.max(1),
                threadgroup_width: None,
                reuse_scratch_buffers: true,
            });
        }
    }
    configs
}

/// Capture samples in a deterministic rotating order. Every candidate occupies
/// every relative position over a full cycle, limiting bias from thermal drift
/// without making the evidence schedule nondeterministic.
pub fn capture_with(
    study: &TuningStudy,
    configs: &[ExecutionConfig],
    mut measure: impl FnMut(&TuningWorkload, &ExecutionConfig) -> Result<u64, BackendError>,
) -> Result<CapturedStudy, BackendError> {
    if configs.is_empty() || study.workloads.is_empty() || study.trials == 0 {
        return Err(BackendError::InvalidInput);
    }
    let mut observations =
        Vec::with_capacity(study.workloads.len() * configs.len() * study.trials as usize);
    for workload in &study.workloads {
        for config in configs {
            std::hint::black_box(measure(workload, config)?);
        }
        for trial in 0..study.trials {
            let offset = trial as usize % configs.len();
            for position in 0..configs.len() {
                let config = &configs[(position + offset) % configs.len()];
                observations.push(TimingObservation {
                    workload_id: workload.id,
                    config_id: config.id,
                    trial,
                    elapsed_nanos: measure(workload, config)?,
                });
            }
        }
    }
    Ok(CapturedStudy {
        study: study.clone(),
        configs: configs.to_vec(),
        observations,
    })
}

pub fn capture_cpu(
    study: &TuningStudy,
    thread_counts: &[usize],
) -> Result<CapturedStudy, BackendError> {
    let configs = cpu_configurations(thread_counts);
    let programs: Vec<_> = study
        .workloads
        .iter()
        .map(|workload| synthetic_program(21_000 + workload.id, 2, workload.alu_ops))
        .collect();
    let refs: Vec<_> = programs.iter().collect();
    let mut backends: BTreeMap<usize, CpuReferenceBackend> = thread_counts
        .iter()
        .copied()
        .map(|threads| {
            (
                threads.max(1),
                CpuReferenceBackend::with(&refs).with_threads(threads),
            )
        })
        .collect();
    let mut unused_accelerator = CpuReferenceBackend::with(&refs);

    capture_with(study, &configs, |workload, config| {
        let program = programs
            .iter()
            .find(|program| program.id() == 21_000 + workload.id)
            .ok_or(BackendError::InvalidInput)?;
        let cpu = backends
            .get_mut(&config.cpu_threads)
            .ok_or(BackendError::InvalidInput)?;
        measure_epoch_once(
            program,
            workload,
            u32::MAX,
            &mut unused_accelerator,
            cpu,
            config.batched_epoch,
        )
    })
}

pub fn replay(captured: CapturedStudy) -> Result<SelfTuningReport, DiscoveryError> {
    replay_with_preparation_sharing(captured, true)
}

/// Replay control used to isolate the effect of deterministic DAG sharing.
pub fn replay_with_preparation_sharing(
    captured: CapturedStudy,
    share_preparation: bool,
) -> Result<SelfTuningReport, DiscoveryError> {
    validate_capture(&captured)?;
    let trace = trace_from_capture(&captured, share_preparation);
    let program = replay_program();
    let mut naive_backend = CpuReferenceBackend::with(&[&program]);
    let mut optimized_backend = CpuReferenceBackend::with(&[&program]);
    let naive = execute_naive(&trace, &mut naive_backend)?;
    let optimized = execute_optimized(&trace, &mut optimized_backend)?;
    let invariants = verify_pair(&trace, &naive, &optimized);
    let rankings = rankings(&captured);
    Ok(SelfTuningReport {
        captured,
        naive,
        optimized,
        invariants,
        rankings,
    })
}

pub fn run_cpu(
    study: &TuningStudy,
    thread_counts: &[usize],
) -> Result<SelfTuningReport, DiscoveryError> {
    replay(capture_cpu(study, thread_counts).map_err(DiscoveryError::Backend)?)
}

pub fn trace_from_capture(captured: &CapturedStudy, share_preparation: bool) -> DiscoveryTrace {
    let mut trace = DiscoveryTrace::default();
    for config in &captured.configs {
        trace.push(DiscoveryEvent::HypothesisCreated {
            id: u64::from(config.id),
            parent: None,
        });
    }

    let module = ModuleDigest::of(b"soma.self-tuning.replay.v1");
    let mut request = 1u64;
    let mut preparation = BTreeMap::new();
    for workload in &captured.study.workloads {
        for config in &captured.configs {
            let mut generated = replay_spec(
                "generate-evaluator",
                module,
                [u64::from(workload.id), u64::from(workload.alu_ops), 0, 0],
            );
            let mut prepared = replay_spec(
                "prepare-input",
                module,
                [
                    u64::from(workload.id),
                    u64::from(workload.cohorts),
                    u64::from(workload.elements_per_cohort),
                    8,
                ],
            );
            if !share_preparation {
                generated.contract = config.id.to_le_bytes().to_vec();
                prepared.contract = config.id.to_le_bytes().to_vec();
            }
            let generated_request = request;
            trace.push(DiscoveryEvent::NodeRequested {
                request,
                hypothesis: u64::from(config.id),
                node: DiscoveryNode::Derivation(generated),
            });
            request += 1;
            let prepared_request = request;
            trace.push(DiscoveryEvent::NodeRequested {
                request,
                hypothesis: u64::from(config.id),
                node: DiscoveryNode::Derivation(prepared),
            });
            request += 1;
            preparation.insert(
                (workload.id, config.id),
                (generated_request, prepared_request),
            );
        }

        for observation in captured
            .observations
            .iter()
            .filter(|observation| observation.workload_id == workload.id)
        {
            let config = captured
                .configs
                .iter()
                .find(|config| config.id == observation.config_id)
                .expect("capture validation guarantees the config exists");
            let mut evaluation = replay_spec(
                "wall-clock-observation",
                module,
                [
                    u64::from(observation.workload_id),
                    u64::from(observation.config_id),
                    u64::from(observation.trial),
                    observation.elapsed_nanos,
                ],
            );
            evaluation.contract = config.label().into_bytes();
            let observation_request = request;
            trace.push(DiscoveryEvent::NodeRequested {
                request,
                hypothesis: u64::from(config.id),
                node: DiscoveryNode::Observation {
                    sample: request,
                    evaluation,
                },
            });
            let (generated, prepared) = preparation[&(workload.id, config.id)];
            trace.push(DiscoveryEvent::DependencyAdded {
                node: observation_request,
                depends_on: generated,
            });
            trace.push(DiscoveryEvent::DependencyAdded {
                node: observation_request,
                depends_on: prepared,
            });
            request += 1;
        }
        trace.push(DiscoveryEvent::EvidencePublished);
    }
    for config in &captured.configs {
        trace.push(DiscoveryEvent::HypothesisAccepted {
            id: u64::from(config.id),
        });
    }
    trace
}

pub fn rankings(captured: &CapturedStudy) -> Vec<WorkloadRanking> {
    captured
        .study
        .workloads
        .iter()
        .map(|workload| {
            let mut configs: Vec<_> = captured
                .configs
                .iter()
                .map(|config| {
                    let mut samples: Vec<_> = captured
                        .observations
                        .iter()
                        .filter(|sample| {
                            sample.workload_id == workload.id && sample.config_id == config.id
                        })
                        .map(|sample| sample.elapsed_nanos)
                        .collect();
                    samples.sort_unstable();
                    ConfigScore {
                        config_id: config.id,
                        median_nanos: samples[samples.len() / 2],
                    }
                })
                .collect();
            configs.sort_by_key(|score| (score.median_nanos, score.config_id));
            WorkloadRanking {
                workload_id: workload.id,
                configs,
            }
        })
        .collect()
}

fn validate_capture(captured: &CapturedStudy) -> Result<(), DiscoveryError> {
    if captured.configs.is_empty()
        || captured.study.workloads.is_empty()
        || captured.study.trials == 0
    {
        return Err(DiscoveryError::InvalidTrace("empty self-tuning study"));
    }
    let expected =
        captured.configs.len() * captured.study.workloads.len() * captured.study.trials as usize;
    if captured.observations.len() != expected {
        return Err(DiscoveryError::InvalidTrace(
            "self-tuning observation count mismatch",
        ));
    }
    let unique_configs: std::collections::BTreeSet<_> =
        captured.configs.iter().map(|config| config.id).collect();
    let unique_workloads: std::collections::BTreeSet<_> = captured
        .study
        .workloads
        .iter()
        .map(|workload| workload.id)
        .collect();
    if unique_configs.len() != captured.configs.len()
        || unique_workloads.len() != captured.study.workloads.len()
    {
        return Err(DiscoveryError::InvalidTrace(
            "duplicate self-tuning identity",
        ));
    }
    for workload in &captured.study.workloads {
        for config in &captured.configs {
            for trial in 0..captured.study.trials {
                let count = captured
                    .observations
                    .iter()
                    .filter(|sample| {
                        sample.workload_id == workload.id
                            && sample.config_id == config.id
                            && sample.trial == trial
                    })
                    .count();
                if count != 1 {
                    return Err(DiscoveryError::InvalidTrace(
                        "self-tuning observation identity mismatch",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn replay_program() -> EvaluatorProgram {
    EvaluatorProgram::new(
        REPLAY_EVALUATOR,
        "self_tuning_replay",
        ElementLayout::new(vec![FieldWidth::U64; 4]),
        vec![Op::Load(0), Op::Load(1), Op::Load(2), Op::Load(3)],
        vec![
            Store { field: 0, value: 0 },
            Store { field: 1, value: 1 },
            Store { field: 2, value: 2 },
            Store { field: 3, value: 3 },
        ],
    )
    .expect("identity replay evaluator is valid")
}

fn replay_spec(operation: &str, module: ModuleDigest, values: [u64; 4]) -> EvaluationSpec {
    let inputs = values
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    EvaluationSpec::new(
        operation,
        module,
        REPLAY_EVALUATOR,
        inputs,
        1,
        32,
        FusionClass::Pointwise,
    )
}

fn measure_epoch_once(
    program: &EvaluatorProgram,
    workload: &TuningWorkload,
    minimum_accelerator_batch: u32,
    accelerator: &mut dyn BatchBackend,
    cpu: &mut dyn BatchBackend,
    batched: bool,
) -> Result<u64, BackendError> {
    let stride = program.stride();
    let bytes = synthetic_inputs(workload.elements_per_cohort, stride);
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs: Vec<Ref64> = (0..workload.cohorts)
        .map(|_| {
            let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
            freeze(&mut kernel, owner, input).expect("fresh benchmark input is freezable");
            input
        })
        .collect();
    let collectives: Vec<Ref64> = inputs
        .iter()
        .map(|input| {
            kernel
                .create_batch_evaluate_for(
                    owner,
                    program.id(),
                    *input,
                    workload.elements_per_cohort,
                    stride,
                )
                .map(|(collective, _)| collective)
                .map_err(BackendError::from)
        })
        .collect::<Result<_, _>>()?;

    let mut stats = PlacementStats::default();
    let started = Instant::now();
    if batched {
        let outputs = execute_epoch_with_spill(
            &mut kernel,
            owner,
            &collectives,
            minimum_accelerator_batch,
            accelerator,
            cpu,
            &mut stats,
        )?;
        std::hint::black_box(outputs);
    } else {
        for collective in collectives {
            let output = execute_with_spill(
                &mut kernel,
                owner,
                collective,
                minimum_accelerator_batch,
                accelerator,
                cpu,
                &mut stats,
            )?;
            std::hint::black_box(output);
        }
    }
    Ok(started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64)
}
