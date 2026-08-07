//! Scientific-interest operations over the discovery DAG.

use super::graph::{DiscoveryGraph, HypothesisStatus};
use super::node::{HypothesisId, RequestId};

pub fn reject(graph: &mut DiscoveryGraph, hypothesis: HypothesisId) -> Result<(), &'static str> {
    graph.set_status(hypothesis, HypothesisStatus::Rejected)
}

pub fn accept(graph: &mut DiscoveryGraph, hypothesis: HypothesisId) -> Result<(), &'static str> {
    graph.set_status(hypothesis, HypothesisStatus::Accepted)
}

pub fn drop_interest(
    graph: &mut DiscoveryGraph,
    hypothesis: HypothesisId,
    request: RequestId,
) -> Result<(), &'static str> {
    graph.withdraw_request(hypothesis, request)
}
