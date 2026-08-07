//! Pending/ready registry for deterministic discovery nodes.

use std::collections::BTreeMap;

use super::key::ExperimentKey;
use super::node::{DiscoveryNode, RequestId};

#[derive(Clone, Debug)]
pub enum Entry {
    Pending {
        node: DiscoveryNode,
        requests: Vec<RequestId>,
    },
    Ready {
        output: Vec<u8>,
    },
    Failed,
}

#[derive(Clone, Debug, Default)]
pub struct Registry {
    pub entries: BTreeMap<ExperimentKey, Entry>,
}

pub enum RequestDisposition {
    Started,
    Joined,
    Ready(Vec<u8>),
}

impl Registry {
    pub fn request(&mut self, request: RequestId, node: &DiscoveryNode) -> RequestDisposition {
        let key = node.key();
        match self.entries.get_mut(&key) {
            Some(Entry::Pending { requests, .. }) => {
                requests.push(request);
                RequestDisposition::Joined
            }
            Some(Entry::Ready { output }) => RequestDisposition::Ready(output.clone()),
            Some(entry @ Entry::Failed) => {
                *entry = Entry::Pending {
                    node: node.clone(),
                    requests: vec![request],
                };
                RequestDisposition::Started
            }
            None => {
                self.entries.insert(
                    key,
                    Entry::Pending {
                        node: node.clone(),
                        requests: vec![request],
                    },
                );
                RequestDisposition::Started
            }
        }
    }
}
