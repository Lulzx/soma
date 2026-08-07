//! Work and performance accounting for discovery replay.

use std::time::Duration;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DiscoveryMetrics {
    pub logical_requests: u64,
    pub unique_deterministic_nodes: u64,
    pub cache_hits: u64,
    pub pending_request_joins: u64,
    pub cancelled_before_execution: u64,
    pub physical_evaluator_executions: u64,
    pub deterministic_physical_executions: u64,
    pub physical_dispatches: u64,
    pub command_buffers: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub bytes_transferred: u64,
    pub peak_pending_bytes: u64,
    pub wall_time: Duration,
    pub cpu_time: Duration,
    pub gpu_time: Duration,
}

impl DiscoveryMetrics {
    pub fn compute_compression(&self) -> f64 {
        ratio(self.logical_requests, self.physical_evaluator_executions)
    }

    pub fn elimination_rate_against(&self, naive: &DiscoveryMetrics) -> f64 {
        if naive.physical_evaluator_executions == 0 {
            return 0.0;
        }
        1.0 - self.physical_evaluator_executions as f64 / naive.physical_evaluator_executions as f64
    }

    pub fn batch_compression(&self) -> f64 {
        ratio(self.physical_evaluator_executions, self.physical_dispatches)
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
