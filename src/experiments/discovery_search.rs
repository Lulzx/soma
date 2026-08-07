//! Deterministic discovery workload and regime sweep.

use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::body::{ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};
use crate::discovery::invariants::{verify_pair, DiscoveryInvariantReport};
use crate::discovery::{
    execute_naive, execute_optimized, DiscoveryError, DiscoveryEvent, DiscoveryNode,
    DiscoveryResult, DiscoveryTrace, EvaluationSpec, FusionClass, ModuleDigest,
};
use crate::executives::batch::{BatchBackend, CpuReferenceBackend};

pub const DUPLICATION_RATES: [f32; 4] = [0.0, 0.25, 0.5, 0.75];
pub const PRUNING_RATES: [f32; 3] = [0.0, 0.25, 0.5];
pub const EVALUATOR_CLASSES: [u32; 3] = [1, 4, 16];
pub const BATCH_SIZES: [u32; 3] = [1, 64, 1_024];

#[derive(Clone, Copy, Debug)]
pub struct DiscoveryKnobs {
    pub branching_factor: u32,
    pub depth: u32,
    pub duplicate_rate: f32,
    pub shared_prefix_rate: f32,
    pub rejection_rate: f32,
    pub rejection_depth: u32,
    pub evaluator_classes: u32,
    pub elements_per_experiment: u32,
    pub arrival_skew: u32,
    pub observation_rate: f32,
}

impl Default for DiscoveryKnobs {
    fn default() -> Self {
        Self {
            branching_factor: 3,
            depth: 5,
            duplicate_rate: 0.35,
            shared_prefix_rate: 0.4,
            rejection_rate: 0.25,
            rejection_depth: 2,
            evaluator_classes: 4,
            elements_per_experiment: 64,
            arrival_skew: 2,
            observation_rate: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiscoveryReport {
    pub naive: DiscoveryResult,
    pub optimized: DiscoveryResult,
    pub invariants: DiscoveryInvariantReport,
}

impl DiscoveryReport {
    pub fn compute_compression(&self) -> f64 {
        self.optimized.metrics.compute_compression()
    }

    pub fn elimination_rate(&self) -> f64 {
        self.optimized
            .metrics
            .elimination_rate_against(&self.naive.metrics)
    }
}

/// One trace, generated without an intelligence component. Every executor sees
/// precisely these events in precisely this order.
pub fn generate_trace(knobs: &DiscoveryKnobs) -> DiscoveryTrace {
    let mut trace = DiscoveryTrace::default();
    let hypotheses = hypothesis_tree(knobs.branching_factor, knobs.depth);
    for (id, parent) in &hypotheses {
        trace.push(DiscoveryEvent::HypothesisCreated {
            id: *id,
            parent: *parent,
        });
    }

    let classes = knobs.evaluator_classes.max(1);
    let elements = knobs.elements_per_experiment.max(1);
    let mut request = 1u64;
    let mut previous: BTreeMap<u64, u64> = BTreeMap::new();
    let mut rejected = BTreeSet::new();
    for level in 0..=knobs.depth {
        for (hypothesis, _) in &hypotheses {
            let entropy = splitmix64((*hypothesis << 32) ^ u64::from(level));
            let class = (entropy % u64::from(classes)) as u32;
            let shared =
                unit(entropy.rotate_left(7)) < knobs.shared_prefix_rate && level <= knobs.depth / 2;
            let duplicate = unit(entropy.rotate_left(19)) < knobs.duplicate_rate;
            let identity = if shared {
                u64::from(level)
            } else if duplicate {
                (*hypothesis / 4) ^ u64::from(level)
            } else {
                (*hypothesis << 32) ^ u64::from(level)
            };
            let mut inputs = Vec::with_capacity(elements as usize * 8);
            for element in 0..elements {
                inputs.extend_from_slice(&splitmix64(identity ^ u64::from(element)).to_le_bytes());
            }
            let module = ModuleDigest::of(format!("discovery-evaluator-{class}").as_bytes());
            let spec = EvaluationSpec::new(
                format!("derive-{class}"),
                module,
                10_000 + class,
                inputs,
                elements,
                8,
                FusionClass::Pointwise,
            );
            let node = if unit(entropy.rotate_left(31)) < knobs.observation_rate {
                DiscoveryNode::Observation {
                    sample: request,
                    evaluation: spec,
                }
            } else if level == knobs.depth {
                DiscoveryNode::Decision(spec)
            } else if level + 1 == knobs.depth {
                DiscoveryNode::Aggregate(spec)
            } else {
                DiscoveryNode::Derivation(spec)
            };
            trace.push(DiscoveryEvent::NodeRequested {
                request,
                hypothesis: *hypothesis,
                node,
            });
            if let Some(dependency) = previous.insert(*hypothesis, request) {
                trace.push(DiscoveryEvent::DependencyAdded {
                    node: request,
                    depends_on: dependency,
                });
            }
            request += 1;
        }

        if level == knobs.rejection_depth.min(knobs.depth) {
            let mut roots = Vec::new();
            for (hypothesis, _) in &hypotheses {
                let entropy = splitmix64(*hypothesis ^ 0xD15C_0A3E);
                if *hypothesis != 1
                    && !rejected.contains(hypothesis)
                    && unit(entropy) < knobs.rejection_rate
                {
                    roots.push(*hypothesis);
                }
            }
            for root in roots {
                trace.push(DiscoveryEvent::HypothesisRejected { id: root });
                for (candidate, _) in &hypotheses {
                    if *candidate == root || descends_from(*candidate, root, &hypotheses) {
                        rejected.insert(*candidate);
                    }
                }
            }
        }
        if (level + 1) % knobs.arrival_skew.max(1) == 0 {
            trace.push(DiscoveryEvent::EvidencePublished);
        }
    }
    trace.push(DiscoveryEvent::EvidencePublished);
    for (hypothesis, _) in hypotheses {
        if !rejected.contains(&hypothesis) {
            trace.push(DiscoveryEvent::HypothesisAccepted { id: hypothesis });
        }
    }
    trace
}

pub fn evaluator_programs(classes: u32) -> Vec<EvaluatorProgram> {
    (0..classes.max(1))
        .map(|class| {
            EvaluatorProgram::new(
                10_000 + class,
                format!("discovery_{class}"),
                ElementLayout::new(vec![FieldWidth::U64]),
                vec![
                    Op::Load(0),
                    Op::Const(u64::from(class) + 2),
                    Op::Mul(0, 1),
                    Op::Const(1),
                    Op::Add(2, 3),
                ],
                vec![Store { field: 0, value: 4 }],
            )
            .expect("the built-in discovery evaluator is valid")
        })
        .collect()
}

pub fn run_with_backend(
    knobs: &DiscoveryKnobs,
    naive_backend: &mut dyn BatchBackend,
    optimized_backend: &mut dyn BatchBackend,
) -> Result<DiscoveryReport, DiscoveryError> {
    let trace = generate_trace(knobs);
    let naive = execute_naive(&trace, naive_backend)?;
    let optimized = execute_optimized(&trace, optimized_backend)?;
    let invariants = verify_pair(&trace, &naive, &optimized);
    Ok(DiscoveryReport {
        naive,
        optimized,
        invariants,
    })
}

pub fn run_cpu(knobs: &DiscoveryKnobs) -> Result<DiscoveryReport, DiscoveryError> {
    let programs = evaluator_programs(knobs.evaluator_classes);
    let refs: Vec<_> = programs.iter().collect();
    let mut naive = CpuReferenceBackend::with(&refs);
    let mut optimized = CpuReferenceBackend::with(&refs);
    run_with_backend(knobs, &mut naive, &mut optimized)
}

#[derive(Clone, Debug)]
pub struct RegimePoint {
    pub duplicate_rate: f32,
    pub rejection_rate: f32,
    pub evaluator_classes: u32,
    pub elements_per_experiment: u32,
    pub compute_compression: f64,
    pub elimination_rate: f64,
    pub batch_compression: f64,
}

pub fn regime_map(base: DiscoveryKnobs) -> Result<Vec<RegimePoint>, DiscoveryError> {
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
                    let report = run_cpu(&knobs)?;
                    points.push(RegimePoint {
                        duplicate_rate,
                        rejection_rate,
                        evaluator_classes,
                        elements_per_experiment,
                        compute_compression: report.compute_compression(),
                        elimination_rate: report.elimination_rate(),
                        batch_compression: report.optimized.metrics.batch_compression(),
                    });
                }
            }
        }
    }
    Ok(points)
}

fn hypothesis_tree(branching: u32, depth: u32) -> Vec<(u64, Option<u64>)> {
    let branching = branching.max(1);
    let mut out = vec![(1, None)];
    let mut frontier = vec![1u64];
    let mut next = 2u64;
    for _ in 0..depth {
        let mut children = Vec::new();
        for parent in frontier {
            for _ in 0..branching {
                out.push((next, Some(parent)));
                children.push(next);
                next += 1;
            }
        }
        frontier = children;
    }
    out
}

fn descends_from(candidate: u64, ancestor: u64, hypotheses: &[(u64, Option<u64>)]) -> bool {
    let mut current = candidate;
    while let Some(parent) = hypotheses
        .iter()
        .find_map(|(id, parent)| (*id == current).then_some(*parent))
        .flatten()
    {
        if parent == ancestor {
            return true;
        }
        current = parent;
    }
    false
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn unit(value: u64) -> f32 {
    (value >> 40) as f32 / (1u32 << 24) as f32
}
