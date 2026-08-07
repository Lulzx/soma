//! Executable D1-D7 checks for discovery replay.

use std::collections::BTreeMap;

use super::executor::DiscoveryResult;
use super::graph::{DiscoveryGraph, HypothesisStatus};
use super::key::ExperimentKey;
use super::node::RequestId;
use super::trace::{DiscoveryEvent, DiscoveryTrace};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiscoveryInvariantReport {
    pub d1_semantic_key_soundness: bool,
    pub d2_observation_multiplicity: bool,
    pub d3_single_physical_realization: bool,
    pub d4_interest_preserving_cancellation: bool,
    pub d5_fused_execution_equivalence: bool,
    pub d6_scientific_equivalence: bool,
    pub d7_accounting_conservation: bool,
}

impl DiscoveryInvariantReport {
    pub fn all_hold(&self) -> bool {
        self.d1_semantic_key_soundness
            && self.d2_observation_multiplicity
            && self.d3_single_physical_realization
            && self.d4_interest_preserving_cancellation
            && self.d5_fused_execution_equivalence
            && self.d6_scientific_equivalence
            && self.d7_accounting_conservation
    }
}

pub fn verify_pair(
    trace: &DiscoveryTrace,
    naive: &DiscoveryResult,
    optimized: &DiscoveryResult,
) -> DiscoveryInvariantReport {
    let mut by_key: BTreeMap<ExperimentKey, Vec<RequestId>> = BTreeMap::new();
    let mut observation_outputs = 0u64;
    let mut graph = DiscoveryGraph::default();
    for event in &trace.events {
        match event {
            DiscoveryEvent::HypothesisCreated { id, parent } => {
                let _ = graph.create_hypothesis(*id, *parent);
            }
            DiscoveryEvent::NodeRequested {
                request,
                hypothesis,
                node,
            } => {
                let _ = graph.request(*request, *hypothesis, node.clone());
            }
            DiscoveryEvent::DependencyAdded { node, depends_on } => {
                let _ = graph.add_dependency(*node, *depends_on);
            }
            DiscoveryEvent::HypothesisAccepted { id } => {
                let _ = graph.set_status(*id, HypothesisStatus::Accepted);
            }
            DiscoveryEvent::HypothesisRejected { id } => {
                let _ = graph.set_status(*id, HypothesisStatus::Rejected);
            }
            DiscoveryEvent::InterestDropped {
                hypothesis,
                request,
            } => {
                let _ = graph.withdraw_request(*hypothesis, *request);
            }
            DiscoveryEvent::EvidencePublished => {}
        }
        if let DiscoveryEvent::NodeRequested { request, node, .. } = event {
            if node.cacheable() {
                by_key.entry(node.key()).or_default().push(*request);
            } else if optimized.outputs.contains_key(request) {
                observation_outputs += 1;
            }
        }
    }

    let d1 = by_key.values().all(|requests| {
        let mut outputs = requests
            .iter()
            .filter_map(|request| optimized.outputs.get(request));
        let Some(first) = outputs.next() else {
            return true;
        };
        outputs.all(|output| output == first)
    });
    let missing_requests: Vec<_> = graph
        .requests
        .keys()
        .filter(|request| !optimized.outputs.contains_key(request))
        .collect();
    let accounted = |result: &DiscoveryResult| {
        result.metrics.physical_evaluator_executions
            + result.metrics.cache_hits
            + result.metrics.pending_request_joins
            + result.metrics.cancelled_before_execution
            == result.metrics.logical_requests
    };

    DiscoveryInvariantReport {
        d1_semantic_key_soundness: d1,
        // Fusion may share a dispatch, never a logical physical realization.
        d2_observation_multiplicity: optimized.metrics.physical_evaluator_executions
            - optimized.metrics.deterministic_physical_executions
            == observation_outputs,
        // One registry entry owns each semantic key. A second request is
        // necessarily a join or hit; this accounting equality detects an
        // accidental second owner in the public result.
        d3_single_physical_realization: optimized.metrics.deterministic_physical_executions
            <= optimized.metrics.unique_deterministic_nodes,
        d4_interest_preserving_cancellation: missing_requests.len() as u64
            == optimized.metrics.cancelled_before_execution
            && missing_requests
                .iter()
                .all(|request| !graph.is_interesting(**request)),
        d5_fused_execution_equivalence: naive.scientific.evidence == optimized.scientific.evidence,
        d6_scientific_equivalence: naive.scientific == optimized.scientific,
        d7_accounting_conservation: accounted(naive) && accounted(optimized),
    }
}
