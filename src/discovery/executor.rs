//! Literal and optimized replay of one logical discovery trace.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use crate::executives::batch::{BackendError, BatchBackend};

use super::batcher::{execute_jobs, PhysicalJob};
use super::graph::{DiscoveryGraph, HypothesisStatus};
use super::interest;
use super::key::{ExperimentKey, NodeDigest, ObjectDigest};
use super::metrics::DiscoveryMetrics;
use super::node::{HypothesisId, RequestId};
use super::registry::{Entry, Registry, RequestDisposition};
use super::trace::{DiscoveryEvent, DiscoveryTrace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub node: NodeDigest,
    pub output: ObjectDigest,
    pub dependencies: Vec<RequestId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScientificState {
    pub hypotheses: BTreeMap<HypothesisId, HypothesisStatus>,
    pub evidence: BTreeMap<(HypothesisId, RequestId), EvidenceRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryResult {
    pub scientific: ScientificState,
    pub outputs: BTreeMap<RequestId, Vec<u8>>,
    pub metrics: DiscoveryMetrics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Executed,
    CacheHit,
    Joined,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscoveryError {
    InvalidTrace(&'static str),
    Backend(BackendError),
    Accounting,
}

impl From<BackendError> for DiscoveryError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

pub fn execute_naive(
    trace: &DiscoveryTrace,
    backend: &mut dyn BatchBackend,
) -> Result<DiscoveryResult, DiscoveryError> {
    let started = Instant::now();
    let mut graph = DiscoveryGraph::default();
    let mut pending = Vec::new();
    let mut outputs = BTreeMap::new();
    let mut outcomes = HashMap::new();
    let mut seen_keys = BTreeSet::new();
    let mut metrics = DiscoveryMetrics::default();

    for event in &trace.events {
        apply_graph_event(&mut graph, event)?;
        match event {
            DiscoveryEvent::NodeRequested { request, node, .. } => {
                metrics.logical_requests += 1;
                if node.cacheable() {
                    seen_keys.insert(node.key());
                }
                pending.push((*request, node.clone()));
            }
            DiscoveryEvent::EvidencePublished => flush_naive(
                backend,
                &mut pending,
                &mut outputs,
                &mut outcomes,
                &mut metrics,
            )?,
            _ => {}
        }
    }
    flush_naive(
        backend,
        &mut pending,
        &mut outputs,
        &mut outcomes,
        &mut metrics,
    )?;
    metrics.unique_deterministic_nodes = seen_keys.len() as u64;
    finish(graph, outputs, outcomes, metrics, started)
}

pub fn execute_optimized(
    trace: &DiscoveryTrace,
    backend: &mut dyn BatchBackend,
) -> Result<DiscoveryResult, DiscoveryError> {
    let started = Instant::now();
    let mut graph = DiscoveryGraph::default();
    let mut registry = Registry::default();
    let mut observations = Vec::new();
    let mut outputs = BTreeMap::new();
    let mut outcomes = HashMap::new();
    let mut seen_keys = BTreeSet::new();
    let mut metrics = DiscoveryMetrics::default();

    for event in &trace.events {
        apply_graph_event(&mut graph, event)?;
        match event {
            DiscoveryEvent::NodeRequested { request, node, .. } => {
                metrics.logical_requests += 1;
                if node.cacheable() {
                    seen_keys.insert(node.key());
                }
                if !graph.is_interesting(*request) {
                    outcomes.insert(*request, Outcome::Cancelled);
                    continue;
                }
                if node.cacheable() {
                    match registry.request(*request, node) {
                        RequestDisposition::Started => {}
                        RequestDisposition::Joined => {}
                        RequestDisposition::Ready(output) => {
                            outputs.insert(*request, output);
                            outcomes.insert(*request, Outcome::CacheHit);
                        }
                    }
                } else {
                    observations.push((*request, node.clone()));
                }
            }
            DiscoveryEvent::EvidencePublished => flush_optimized(
                backend,
                &graph,
                &mut registry,
                &mut observations,
                &mut outputs,
                &mut outcomes,
                &mut metrics,
            )?,
            DiscoveryEvent::DependencyAdded { depends_on, .. }
                if graph.is_interesting(*depends_on)
                    && outcomes.get(depends_on) == Some(&Outcome::Cancelled) =>
            {
                outcomes.remove(depends_on);
                let node = graph
                    .requests
                    .get(depends_on)
                    .expect("dependency was validated")
                    .node
                    .clone();
                if node.cacheable() {
                    let _ = registry.request(*depends_on, &node);
                } else {
                    observations.push((*depends_on, node));
                }
            }
            _ => {}
        }
    }
    flush_optimized(
        backend,
        &graph,
        &mut registry,
        &mut observations,
        &mut outputs,
        &mut outcomes,
        &mut metrics,
    )?;
    metrics.unique_deterministic_nodes = seen_keys.len() as u64;
    finish(graph, outputs, outcomes, metrics, started)
}

fn apply_graph_event(
    graph: &mut DiscoveryGraph,
    event: &DiscoveryEvent,
) -> Result<(), DiscoveryError> {
    let result = match event {
        DiscoveryEvent::HypothesisCreated { id, parent } => graph.create_hypothesis(*id, *parent),
        DiscoveryEvent::NodeRequested {
            request,
            hypothesis,
            node,
        } => node
            .evaluation()
            .validate()
            .and_then(|()| graph.request(*request, *hypothesis, node.clone())),
        DiscoveryEvent::DependencyAdded { node, depends_on } => {
            graph.add_dependency(*node, *depends_on)
        }
        DiscoveryEvent::HypothesisAccepted { id } => interest::accept(graph, *id),
        DiscoveryEvent::HypothesisRejected { id } => interest::reject(graph, *id),
        DiscoveryEvent::InterestDropped {
            hypothesis,
            request,
        } => interest::drop_interest(graph, *hypothesis, *request),
        DiscoveryEvent::EvidencePublished => Ok(()),
    };
    result.map_err(DiscoveryError::InvalidTrace)
}

fn flush_naive(
    backend: &mut dyn BatchBackend,
    pending: &mut Vec<(RequestId, super::node::DiscoveryNode)>,
    outputs: &mut BTreeMap<RequestId, Vec<u8>>,
    outcomes: &mut HashMap<RequestId, Outcome>,
    metrics: &mut DiscoveryMetrics,
) -> Result<(), DiscoveryError> {
    let jobs: Vec<_> = pending
        .iter()
        .enumerate()
        .map(|(id, (_, node))| PhysicalJob {
            id,
            node: node.clone(),
        })
        .collect();
    metrics.deterministic_physical_executions +=
        pending.iter().filter(|(_, node)| node.cacheable()).count() as u64;
    metrics.physical_evaluator_executions += jobs.len() as u64;
    let results = execute_jobs(backend, jobs, false, metrics)?;
    for (id, output) in results {
        let request = pending[id].0;
        outputs.insert(request, output);
        outcomes.insert(request, Outcome::Executed);
    }
    pending.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_optimized(
    backend: &mut dyn BatchBackend,
    graph: &DiscoveryGraph,
    registry: &mut Registry,
    observations: &mut Vec<(RequestId, super::node::DiscoveryNode)>,
    outputs: &mut BTreeMap<RequestId, Vec<u8>>,
    outcomes: &mut HashMap<RequestId, Outcome>,
    metrics: &mut DiscoveryMetrics,
) -> Result<(), DiscoveryError> {
    struct PendingResult {
        key: Option<ExperimentKey>,
        requests: Vec<RequestId>,
    }
    let mut jobs = Vec::new();
    let mut result_map = Vec::new();
    let keys: Vec<_> = registry
        .entries
        .iter()
        .filter_map(|(key, entry)| matches!(entry, Entry::Pending { .. }).then_some(*key))
        .collect();
    for key in keys {
        let Entry::Pending { node, requests } =
            registry.entries.get(&key).expect("key collected").clone()
        else {
            unreachable!()
        };
        let active: Vec<_> = requests
            .iter()
            .copied()
            .filter(|r| graph.is_interesting(*r))
            .collect();
        for request in requests.iter().filter(|r| !graph.is_interesting(**r)) {
            outcomes.insert(*request, Outcome::Cancelled);
        }
        if active.is_empty() {
            registry.entries.remove(&key);
            continue;
        }
        let id = jobs.len();
        jobs.push(PhysicalJob { id, node });
        result_map.push(PendingResult {
            key: Some(key),
            requests: active,
        });
    }
    metrics.deterministic_physical_executions += jobs.len() as u64;
    for (request, node) in observations.drain(..) {
        if graph.is_interesting(request) {
            let id = jobs.len();
            jobs.push(PhysicalJob { id, node });
            result_map.push(PendingResult {
                key: None,
                requests: vec![request],
            });
        } else {
            outcomes.insert(request, Outcome::Cancelled);
        }
    }

    metrics.physical_evaluator_executions += jobs.len() as u64;
    let results = execute_jobs(backend, jobs, true, metrics)?;
    for (id, output) in results {
        let pending = &result_map[id];
        for (index, request) in pending.requests.iter().enumerate() {
            outputs.insert(*request, output.clone());
            outcomes.insert(
                *request,
                if index == 0 {
                    Outcome::Executed
                } else {
                    Outcome::Joined
                },
            );
        }
        if let Some(key) = pending.key {
            registry.entries.insert(key, Entry::Ready { output });
        }
    }
    Ok(())
}

fn finish(
    graph: DiscoveryGraph,
    outputs: BTreeMap<RequestId, Vec<u8>>,
    outcomes: HashMap<RequestId, Outcome>,
    mut metrics: DiscoveryMetrics,
    started: Instant,
) -> Result<DiscoveryResult, DiscoveryError> {
    for outcome in outcomes.values() {
        match outcome {
            Outcome::Executed => {}
            Outcome::CacheHit => metrics.cache_hits += 1,
            Outcome::Joined => metrics.pending_request_joins += 1,
            Outcome::Cancelled => metrics.cancelled_before_execution += 1,
        }
    }
    if outcomes.len() as u64 != metrics.logical_requests {
        return Err(DiscoveryError::Accounting);
    }
    metrics.wall_time = started.elapsed();

    let hypotheses = graph
        .hypotheses
        .iter()
        .map(|(id, state)| (*id, state.status))
        .collect();
    let mut evidence = BTreeMap::new();
    for (request, state) in &graph.requests {
        let Some(output) = outputs.get(request) else {
            continue;
        };
        for consumer in &state.consumers {
            evidence.insert(
                (*consumer, *request),
                EvidenceRecord {
                    node: state.node.digest(),
                    output: ObjectDigest::of(output),
                    dependencies: state.dependencies.iter().copied().collect(),
                },
            );
        }
    }
    Ok(DiscoveryResult {
        scientific: ScientificState {
            hypotheses,
            evidence,
        },
        outputs,
        metrics,
    })
}
