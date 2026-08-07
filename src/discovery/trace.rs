//! Deterministic logical discovery traces.

use std::collections::BTreeSet;

use super::node::{DiscoveryNode, HypothesisId, RequestId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryEvent {
    HypothesisCreated {
        id: HypothesisId,
        parent: Option<HypothesisId>,
    },
    NodeRequested {
        request: RequestId,
        hypothesis: HypothesisId,
        node: DiscoveryNode,
    },
    DependencyAdded {
        node: RequestId,
        depends_on: RequestId,
    },
    /// Complete the currently pending physical work. Explicit barriers make
    /// cache hits versus pending joins reproducible across executors.
    EvidencePublished,
    HypothesisAccepted {
        id: HypothesisId,
    },
    HypothesisRejected {
        id: HypothesisId,
    },
    InterestDropped {
        hypothesis: HypothesisId,
        request: RequestId,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiscoveryTrace {
    pub events: Vec<DiscoveryEvent>,
}

impl DiscoveryTrace {
    pub fn push(&mut self, event: DiscoveryEvent) {
        self.events.push(event);
    }

    pub fn hypotheses(&self) -> BTreeSet<HypothesisId> {
        self.events
            .iter()
            .filter_map(|event| match event {
                DiscoveryEvent::HypothesisCreated { id, .. } => Some(*id),
                _ => None,
            })
            .collect()
    }
}
