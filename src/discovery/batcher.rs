//! Safe fusion of compatible physical evaluations.

use std::time::Instant;

use crate::executives::batch::{AuxArray, BackendError, BackendKind, BatchBackend, BatchRequest};

use super::metrics::DiscoveryMetrics;
use super::node::{DiscoveryNode, FusionClass};

#[derive(Clone, Debug)]
pub(crate) struct PhysicalJob {
    pub id: usize,
    pub node: DiscoveryNode,
}

#[derive(Clone, Debug)]
struct Member {
    id: usize,
    byte_start: usize,
    byte_len: usize,
}

#[derive(Clone, Debug)]
struct Group {
    exemplar: DiscoveryNode,
    inputs: Vec<u8>,
    members: Vec<Member>,
}

pub(crate) fn execute_jobs(
    backend: &mut dyn BatchBackend,
    jobs: Vec<PhysicalJob>,
    fuse: bool,
    metrics: &mut DiscoveryMetrics,
) -> Result<Vec<(usize, Vec<u8>)>, BackendError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let mut groups: Vec<Group> = Vec::new();
    for job in jobs {
        let spec = job.node.evaluation();
        let fusible = fuse && spec.fusion == FusionClass::Pointwise && spec.aux_inputs.is_empty();
        let existing = if fusible {
            groups
                .iter_mut()
                .find(|group| compatible(&group.exemplar, &job.node))
        } else {
            None
        };
        if let Some(group) = existing {
            let start = group.inputs.len();
            group.inputs.extend_from_slice(&spec.inputs);
            group.members.push(Member {
                id: job.id,
                byte_start: start,
                byte_len: spec.inputs.len(),
            });
        } else {
            let inputs = spec.inputs.clone();
            let byte_len = inputs.len();
            groups.push(Group {
                exemplar: job.node,
                inputs,
                members: vec![Member {
                    id: job.id,
                    byte_start: 0,
                    byte_len,
                }],
            });
        }
    }

    metrics.physical_dispatches += groups.len() as u64;
    let pending_bytes: u64 = groups.iter().map(|group| group.inputs.len() as u64).sum();
    metrics.peak_pending_bytes = metrics.peak_pending_bytes.max(pending_bytes);
    metrics.input_bytes += pending_bytes;

    let requests: Vec<BatchRequest<'_>> = groups
        .iter()
        .map(|group| {
            let spec = group.exemplar.evaluation();
            BatchRequest {
                evaluator_id: spec.evaluator_id,
                inputs: &group.inputs,
                aux: AuxArray::new(
                    &spec.aux_inputs,
                    spec.aux_element_count,
                    spec.aux_element_stride,
                ),
                element_count: (group.inputs.len() / spec.element_stride as usize) as u32,
                element_stride: spec.element_stride,
            }
        })
        .collect();
    let kind = backend.kind();
    let backend_started = Instant::now();
    let payloads = backend.evaluate_epoch(&requests)?;
    let backend_elapsed = backend_started.elapsed();
    match kind {
        BackendKind::Cpu => metrics.cpu_time += backend_elapsed,
        BackendKind::Accelerator => metrics.gpu_time += backend_elapsed,
        BackendKind::Remote => metrics.remote_time += backend_elapsed,
    }
    if matches!(kind, BackendKind::Accelerator | BackendKind::Remote) {
        metrics.command_buffers += 1;
    }

    if payloads.len() != groups.len() {
        return Err(BackendError::ExecutionFailed);
    }
    let mut outputs = Vec::new();
    for (group, payload) in groups.iter().zip(payloads) {
        if payload.len() != group.inputs.len() {
            return Err(BackendError::ExecutionFailed);
        }
        metrics.output_bytes += payload.len() as u64;
        for member in &group.members {
            let end = member.byte_start + member.byte_len;
            outputs.push((
                member.id,
                payload.as_slice()[member.byte_start..end].to_vec(),
            ));
        }
    }
    if matches!(kind, BackendKind::Accelerator | BackendKind::Remote) {
        metrics.bytes_transferred = metrics.input_bytes + metrics.output_bytes;
    }
    Ok(outputs)
}

fn compatible(a: &DiscoveryNode, b: &DiscoveryNode) -> bool {
    let a = a.evaluation();
    let b = b.evaluation();
    a.fusion == FusionClass::Pointwise
        && b.fusion == FusionClass::Pointwise
        && a.module == b.module
        && a.evaluator_id == b.evaluator_id
        && a.element_stride == b.element_stride
        && a.parameters == b.parameters
        && a.contract == b.contract
        && a.aux_inputs.is_empty()
        && b.aux_inputs.is_empty()
}
