//! Conflict metadata for the speculative CPU epoch executive.
//!
//! A worker runs against an isolated kernel snapshot. `LaneView` records the
//! semantic resources touched by the handler; the epoch validator then accepts
//! only histories whose read/write sets commute. Everything else is replayed
//! by the reference executive in plan order.

use std::collections::HashSet;

use crate::abi::{MessageDescriptor, ObjectKind, ProcessMode, Ref64};
use crate::kernel::{AwaitOutcome, ContinuationSpec, RuntimeError};
use crate::scheduler::device::{DeviceLaneAccess, DEVICE_ACCESS_READ, DEVICE_ACCESS_WRITE};
use crate::scheduler::device_ops::{
    await_result, encode_spec, message_result, observe_result, ref_result, unit_result,
    DeviceLaneOperation, DeviceOperationJournal, OP_AWAIT_FUTURE, OP_CREATE_CONTINUATION,
    OP_CREATE_FUTURE, OP_CREATE_OBJECT, OP_CREATE_PROCESS, OP_ENQUEUE_MESSAGE, OP_OBSERVE_FUTURE,
    OP_READ_OBJECT, OP_RECEIVE_MESSAGE, OP_RESOLVE_FUTURE, OP_WRITE_OBJECT, RESULT_OK,
};

/// How Phase F executes admitted lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EpochExecutive {
    /// The normative, plan-order interpreter.
    #[default]
    Reference,
    /// Execute at most `max_lanes` isolated snapshots concurrently, validate
    /// their recorded accesses, then commit in canonical lane order.
    Speculative { max_lanes: usize },
}

/// Cumulative measurements for speculative epochs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculationStats {
    pub attempted_epochs: u64,
    pub committed_epochs: u64,
    pub fallback_epochs: u64,
    pub speculative_lanes: u64,
    pub committed_lanes: u64,
    pub conflict_fallbacks: u64,
    pub unsupported_fallbacks: u64,
    /// Bit `opcode - 1` for every internal lane operation successfully lowered
    /// to the fixed-width device ABI, including epochs later rejected for a
    /// semantic conflict.
    pub device_operation_kinds: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Resource {
    Object(Ref64),
    Future(Ref64),
    Process(Ref64),
    Mailbox(Ref64),
    Domain(Ref64),
    Allocation(u8),
}

/// Concrete kernel calls made by one handler. Accepted speculative executions
/// replay this list against the real kernel; results are retained so a debug
/// build can detect an incomplete conflict declaration at the commit boundary.
#[derive(Clone, Debug)]
pub(crate) enum LaneOperation {
    ObserveFuture {
        actor: Ref64,
        future: Ref64,
        result: Result<Option<Ref64>, RuntimeError>,
    },
    ReadObject {
        actor: Ref64,
        object: Ref64,
    },
    CreateProcess {
        actor: Ref64,
        mode: ProcessMode,
        result: Result<Ref64, RuntimeError>,
    },
    CreateContinuation {
        actor: Ref64,
        process: Ref64,
        spec: ContinuationSpec,
        result: Result<Ref64, RuntimeError>,
    },
    CreateFuture {
        actor: Ref64,
        result: Ref64,
    },
    CreateObject {
        actor: Ref64,
        kind: ObjectKind,
        bytes: Vec<u8>,
        result: Ref64,
    },
    WriteObject {
        actor: Ref64,
        object: Ref64,
        growable: bool,
    },
    EnqueueMessage {
        actor: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_continuation: Ref64,
        result: Result<(), RuntimeError>,
    },
    ReceiveMessage {
        actor: Ref64,
        continuation: Ref64,
        result: Result<Option<MessageDescriptor>, RuntimeError>,
    },
    ResolveFuture {
        actor: Ref64,
        future: Ref64,
        value: Ref64,
        result: Result<(), RuntimeError>,
    },
    AwaitFuture {
        actor: Ref64,
        continuation: Ref64,
        future: Ref64,
        next_run_class: u32,
        result: Result<AwaitOutcome, RuntimeError>,
    },
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LaneJournal {
    pub(crate) reads: HashSet<Resource>,
    pub(crate) writes: HashSet<Resource>,
    pub(crate) mutated_objects: HashSet<Ref64>,
    pub(crate) operations: Vec<LaneOperation>,
    pub(crate) unsupported: bool,
}

impl LaneJournal {
    pub(crate) fn read(&mut self, resource: Resource) {
        self.reads.insert(resource);
    }

    pub(crate) fn write(&mut self, resource: Resource) {
        self.writes.insert(resource);
    }

    pub(crate) fn push(&mut self, operation: LaneOperation) {
        self.operations.push(operation);
    }

    pub(crate) fn conflicts_with(&self, other: &Self) -> bool {
        self.writes
            .iter()
            .any(|resource| other.writes.contains(resource) || other.reads.contains(resource))
            || other
                .writes
                .iter()
                .any(|resource| self.reads.contains(resource))
    }

    pub(crate) fn device_accesses(&self, lane: u32) -> Vec<DeviceLaneAccess> {
        let mut accesses: Vec<_> = self
            .reads
            .iter()
            .map(|resource| (resource.device_key(), DEVICE_ACCESS_READ))
            .chain(
                self.writes
                    .iter()
                    .map(|resource| (resource.device_key(), DEVICE_ACCESS_WRITE)),
            )
            .collect();
        accesses.sort_unstable();
        accesses
            .into_iter()
            .enumerate()
            .map(|(ordinal, ((resource_kind, resource), mode))| {
                DeviceLaneAccess::new(lane, resource_kind, resource, mode, ordinal as u32)
            })
            .collect()
    }

    pub(crate) fn device_operations(&self, lane: u32) -> Option<DeviceOperationJournal> {
        let mut journal = DeviceOperationJournal::default();
        for (ordinal, operation) in self.operations.iter().enumerate() {
            let mut record = DeviceLaneOperation {
                lane,
                ordinal: ordinal.try_into().ok()?,
                ..DeviceLaneOperation::default()
            };
            let payload = match operation {
                LaneOperation::ObserveFuture {
                    actor,
                    future,
                    result,
                } => {
                    record.opcode = OP_OBSERVE_FUTURE;
                    record.actor = actor.to_u64();
                    record.target = future.to_u64();
                    (record.result_code, record.result_ref) = observe_result(result);
                    None
                }
                LaneOperation::ReadObject { actor, object } => {
                    record.opcode = OP_READ_OBJECT;
                    record.actor = actor.to_u64();
                    record.target = object.to_u64();
                    None
                }
                LaneOperation::CreateProcess {
                    actor,
                    mode,
                    result,
                } => {
                    record.opcode = OP_CREATE_PROCESS;
                    record.actor = actor.to_u64();
                    record.flags = *mode as u32;
                    (record.result_code, record.result_ref) = ref_result(result);
                    None
                }
                LaneOperation::CreateContinuation {
                    actor,
                    process,
                    spec,
                    result,
                } => {
                    record.opcode = OP_CREATE_CONTINUATION;
                    record.actor = actor.to_u64();
                    record.target = process.to_u64();
                    (record.result_code, record.result_ref) = ref_result(result);
                    Some(encode_spec(spec))
                }
                LaneOperation::CreateFuture { actor, result } => {
                    record.opcode = OP_CREATE_FUTURE;
                    record.actor = actor.to_u64();
                    record.result_code = RESULT_OK;
                    record.result_ref = result.to_u64();
                    None
                }
                LaneOperation::CreateObject {
                    actor,
                    kind,
                    bytes,
                    result,
                } => {
                    record.opcode = OP_CREATE_OBJECT;
                    record.actor = actor.to_u64();
                    record.flags = *kind as u32;
                    record.result_code = RESULT_OK;
                    record.result_ref = result.to_u64();
                    Some(bytes.clone())
                }
                LaneOperation::WriteObject {
                    actor,
                    object,
                    growable,
                } => {
                    record.opcode = OP_WRITE_OBJECT;
                    record.actor = actor.to_u64();
                    record.target = object.to_u64();
                    record.flags = u32::from(*growable);
                    None
                }
                LaneOperation::EnqueueMessage {
                    actor,
                    receiver,
                    payload,
                    sender_continuation,
                    result,
                } => {
                    record.opcode = OP_ENQUEUE_MESSAGE;
                    record.actor = actor.to_u64();
                    record.target = receiver.to_u64();
                    record.value = payload.to_u64();
                    record.auxiliary = sender_continuation.to_u64();
                    record.result_code = unit_result(result);
                    None
                }
                LaneOperation::ReceiveMessage {
                    actor,
                    continuation,
                    result,
                } => {
                    record.opcode = OP_RECEIVE_MESSAGE;
                    record.actor = actor.to_u64();
                    record.target = continuation.to_u64();
                    let (code, bytes) = message_result(result);
                    record.result_code = code;
                    (!bytes.is_empty()).then_some(bytes)
                }
                LaneOperation::ResolveFuture {
                    actor,
                    future,
                    value,
                    result,
                } => {
                    record.opcode = OP_RESOLVE_FUTURE;
                    record.actor = actor.to_u64();
                    record.target = future.to_u64();
                    record.value = value.to_u64();
                    record.result_code = unit_result(result);
                    None
                }
                LaneOperation::AwaitFuture {
                    actor,
                    continuation,
                    future,
                    next_run_class,
                    result,
                } => {
                    record.opcode = OP_AWAIT_FUTURE;
                    record.actor = actor.to_u64();
                    record.target = continuation.to_u64();
                    record.value = future.to_u64();
                    record.flags = *next_run_class;
                    (record.result_code, record.result_aux) = await_result(result);
                    None
                }
            };
            if let Some(payload) = payload {
                (record.payload_offset, record.payload_len) = journal.append_payload(&payload)?;
            }
            journal.operations.push(record);
        }
        Some(journal)
    }
}

impl Resource {
    fn device_key(self) -> (u32, u64) {
        match self {
            Self::Object(reference) => (1, reference.to_u64()),
            Self::Future(reference) => (2, reference.to_u64()),
            Self::Process(reference) => (3, reference.to_u64()),
            Self::Mailbox(reference) => (4, reference.to_u64()),
            Self::Domain(reference) => (5, reference.to_u64()),
            Self::Allocation(partition) => (6, u64::from(partition)),
        }
    }
}
