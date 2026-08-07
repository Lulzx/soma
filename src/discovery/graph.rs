//! The logical hypothesis and experiment DAG reconstructed from a trace.

use std::collections::{BTreeMap, BTreeSet};

use super::node::{DiscoveryNode, HypothesisId, RequestId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypothesisStatus {
    Live,
    Accepted,
    Rejected,
}

#[derive(Clone, Debug)]
pub struct HypothesisState {
    pub parent: Option<HypothesisId>,
    pub status: HypothesisStatus,
}

#[derive(Clone, Debug)]
pub struct RequestState {
    pub hypothesis: HypothesisId,
    pub node: DiscoveryNode,
    pub dependencies: BTreeSet<RequestId>,
    pub direct_consumers: BTreeSet<HypothesisId>,
    pub consumers: BTreeSet<HypothesisId>,
}

#[derive(Clone, Debug, Default)]
pub struct DiscoveryGraph {
    pub hypotheses: BTreeMap<HypothesisId, HypothesisState>,
    pub requests: BTreeMap<RequestId, RequestState>,
}

impl DiscoveryGraph {
    pub fn create_hypothesis(
        &mut self,
        id: HypothesisId,
        parent: Option<HypothesisId>,
    ) -> Result<(), &'static str> {
        if self.hypotheses.contains_key(&id) {
            return Err("duplicate hypothesis id");
        }
        if parent.is_some_and(|p| !self.hypotheses.contains_key(&p)) {
            return Err("unknown parent hypothesis");
        }
        self.hypotheses.insert(
            id,
            HypothesisState {
                parent,
                status: HypothesisStatus::Live,
            },
        );
        Ok(())
    }

    pub fn request(
        &mut self,
        request: RequestId,
        hypothesis: HypothesisId,
        node: DiscoveryNode,
    ) -> Result<(), &'static str> {
        if self.requests.contains_key(&request) {
            return Err("duplicate request id");
        }
        let Some(h) = self.hypotheses.get(&hypothesis) else {
            return Err("request for unknown hypothesis");
        };
        let mut consumers = BTreeSet::new();
        if h.status != HypothesisStatus::Rejected {
            consumers.insert(hypothesis);
        }
        self.requests.insert(
            request,
            RequestState {
                hypothesis,
                node,
                dependencies: BTreeSet::new(),
                direct_consumers: consumers.clone(),
                consumers,
            },
        );
        Ok(())
    }

    pub fn add_dependency(
        &mut self,
        node: RequestId,
        depends_on: RequestId,
    ) -> Result<(), &'static str> {
        if node == depends_on {
            return Err("self dependency");
        }
        let consumers = self
            .requests
            .get(&node)
            .ok_or("unknown dependent request")?
            .consumers
            .clone();
        if !self.requests.contains_key(&depends_on) {
            return Err("unknown dependency request");
        }
        if self.reaches(depends_on, node) {
            return Err("dependency cycle");
        }
        self.requests
            .get_mut(&node)
            .expect("checked")
            .dependencies
            .insert(depends_on);
        for consumer in consumers {
            self.add_consumer_recursive(depends_on, consumer);
        }
        Ok(())
    }

    fn reaches(&self, from: RequestId, target: RequestId) -> bool {
        if from == target {
            return true;
        }
        self.requests.get(&from).is_some_and(|request| {
            request
                .dependencies
                .iter()
                .any(|dependency| self.reaches(*dependency, target))
        })
    }

    fn add_consumer_recursive(&mut self, request: RequestId, consumer: HypothesisId) {
        let dependencies = {
            let state = self.requests.get_mut(&request).expect("dependency checked");
            if !state.consumers.insert(consumer) {
                return;
            }
            state.dependencies.iter().copied().collect::<Vec<_>>()
        };
        for dependency in dependencies {
            self.add_consumer_recursive(dependency, consumer);
        }
    }

    pub fn set_status(
        &mut self,
        id: HypothesisId,
        status: HypothesisStatus,
    ) -> Result<(), &'static str> {
        if !self.hypotheses.contains_key(&id) {
            return Err("unknown hypothesis");
        }
        if status != HypothesisStatus::Rejected {
            self.hypotheses.get_mut(&id).expect("checked").status = status;
            return Ok(());
        }

        // A rejected research branch includes every hypothesis below it.
        // Descendants may have requested disjoint work, so merely withdrawing
        // the parent's own consumer id would leave precisely that work alive.
        let mut rejected = vec![id];
        let mut cursor = 0;
        while cursor < rejected.len() {
            let parent = rejected[cursor];
            rejected.extend(
                self.hypotheses
                    .iter()
                    .filter_map(|(child, state)| (state.parent == Some(parent)).then_some(*child)),
            );
            cursor += 1;
        }
        for hypothesis in rejected {
            self.hypotheses
                .get_mut(&hypothesis)
                .expect("descendant exists")
                .status = HypothesisStatus::Rejected;
            self.withdraw_hypothesis(hypothesis);
        }
        Ok(())
    }

    pub fn withdraw_request(
        &mut self,
        hypothesis: HypothesisId,
        request: RequestId,
    ) -> Result<(), &'static str> {
        if !self.hypotheses.contains_key(&hypothesis) {
            return Err("unknown hypothesis");
        }
        let state = self.requests.get_mut(&request).ok_or("unknown request")?;
        state.direct_consumers.remove(&hypothesis);
        self.recompute_consumers();
        Ok(())
    }

    pub fn withdraw_hypothesis(&mut self, hypothesis: HypothesisId) {
        for request in self.requests.values_mut() {
            request.direct_consumers.remove(&hypothesis);
            request.consumers.remove(&hypothesis);
        }
    }

    pub fn is_interesting(&self, request: RequestId) -> bool {
        self.requests
            .get(&request)
            .is_some_and(|state| !state.consumers.is_empty())
    }

    fn recompute_consumers(&mut self) {
        for request in self.requests.values_mut() {
            request.consumers = request.direct_consumers.clone();
        }
        loop {
            let edges: Vec<_> = self
                .requests
                .iter()
                .flat_map(|(_, state)| {
                    let consumers = state.consumers.clone();
                    state
                        .dependencies
                        .iter()
                        .map(move |dependency| (*dependency, consumers.clone()))
                })
                .collect();
            let mut changed = false;
            for (dependency, consumers) in edges {
                let state = self
                    .requests
                    .get_mut(&dependency)
                    .expect("dependency validated");
                let before = state.consumers.len();
                state.consumers.extend(consumers);
                changed |= state.consumers.len() != before;
            }
            if !changed {
                break;
            }
        }
    }
}
