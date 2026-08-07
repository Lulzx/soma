//! Fixed-width operation journal emitted by a concurrent/device lane.
//!
//! Scalar snapshot execution and future device execution meet at this ABI.
//! Variable frame, object, and message data lives in one bounded byte arena;
//! operation records contain offsets only, never Rust pointers.

use crate::abi::{AbiError, MessageDescriptor, ObjectKind, ProcessMode, Ref64, StateAccess};
use crate::kernel::{AwaitOutcome, ContinuationSpec, RuntimeError};

pub const OP_OBSERVE_FUTURE: u32 = 1;
pub const OP_READ_OBJECT: u32 = 2;
pub const OP_CREATE_PROCESS: u32 = 3;
pub const OP_CREATE_CONTINUATION: u32 = 4;
pub const OP_CREATE_FUTURE: u32 = 5;
pub const OP_CREATE_OBJECT: u32 = 6;
pub const OP_WRITE_OBJECT: u32 = 7;
pub const OP_ENQUEUE_MESSAGE: u32 = 8;
pub const OP_RECEIVE_MESSAGE: u32 = 9;
pub const OP_RESOLVE_FUTURE: u32 = 10;
pub const OP_AWAIT_FUTURE: u32 = 11;
pub const ALL_OPERATION_KINDS: u32 = (1 << OP_AWAIT_FUTURE) - 1;

pub const RESULT_OK: u32 = 0;
pub const RESULT_NONE: u32 = 1;
pub const RESULT_SOME: u32 = 2;
pub const RESULT_REGISTERED: u32 = 3;
pub const RESULT_SETTLED: u32 = 4;
const RESULT_ERROR_BASE: u32 = 0x100;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLaneOperation {
    pub lane: u32,
    pub ordinal: u32,
    pub opcode: u32,
    pub flags: u32,
    pub actor: u64,
    pub target: u64,
    pub value: u64,
    pub auxiliary: u64,
    pub result_ref: u64,
    pub payload_offset: u32,
    pub payload_len: u32,
    pub result_code: u32,
    pub result_aux: u32,
}

const _: () = assert!(std::mem::size_of::<DeviceLaneOperation>() == 72);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceOperationJournal {
    pub operations: Vec<DeviceLaneOperation>,
    pub payload: Vec<u8>,
}

impl DeviceOperationJournal {
    pub fn payload(&self, operation: DeviceLaneOperation) -> Option<&[u8]> {
        let start = operation.payload_offset as usize;
        let end = start.checked_add(operation.payload_len as usize)?;
        self.payload.get(start..end)
    }

    pub(crate) fn append_payload(&mut self, bytes: &[u8]) -> Option<(u32, u32)> {
        let offset: u32 = self.payload.len().try_into().ok()?;
        let len: u32 = bytes.len().try_into().ok()?;
        self.payload.extend_from_slice(bytes);
        Some((offset, len))
    }
}

pub(crate) fn unit_result(result: &Result<(), RuntimeError>) -> u32 {
    match result {
        Ok(()) => RESULT_OK,
        Err(error) => runtime_error_code(*error),
    }
}

pub(crate) fn ref_result(result: &Result<Ref64, RuntimeError>) -> (u32, u64) {
    match result {
        Ok(reference) => (RESULT_OK, reference.to_u64()),
        Err(error) => (runtime_error_code(*error), 0),
    }
}

pub(crate) fn observe_result(result: &Result<Option<Ref64>, RuntimeError>) -> (u32, u64) {
    match result {
        Ok(None) => (RESULT_NONE, 0),
        Ok(Some(reference)) => (RESULT_SOME, reference.to_u64()),
        Err(error) => (runtime_error_code(*error), 0),
    }
}

pub(crate) fn await_result(result: &Result<AwaitOutcome, RuntimeError>) -> (u32, u32) {
    match result {
        Ok(AwaitOutcome::Registered) => (RESULT_REGISTERED, 0),
        Ok(AwaitOutcome::AlreadySettled(state)) => (RESULT_SETTLED, *state as u32),
        Err(error) => (runtime_error_code(*error), 0),
    }
}

pub(crate) fn message_result(
    result: &Result<Option<MessageDescriptor>, RuntimeError>,
) -> (u32, Vec<u8>) {
    match result {
        Ok(None) => (RESULT_NONE, Vec::new()),
        Ok(Some(message)) => (RESULT_SOME, encode_message(message)),
        Err(error) => (runtime_error_code(*error), Vec::new()),
    }
}

fn runtime_error_code(error: RuntimeError) -> u32 {
    RESULT_ERROR_BASE
        + match error {
            RuntimeError::Abi(error) => match error {
                AbiError::BadSlot => 1,
                AbiError::KindMismatch => 2,
                AbiError::StaleReference => 3,
                AbiError::NoAuthority => 4,
                AbiError::BoundsViolation => 5,
                AbiError::UnknownKind => 6,
            },
            RuntimeError::MissingPayload => 10,
            RuntimeError::MissingMailbox => 11,
            RuntimeError::MailboxFull => 12,
            RuntimeError::ProcessUnavailable => 13,
            RuntimeError::ChannelClosed => 14,
            RuntimeError::InvalidCollective => 15,
            RuntimeError::InvalidStateAccess => 16,
            RuntimeError::AlreadyResolved => 17,
            RuntimeError::NotResolved => 18,
            RuntimeError::MissingCapabilitySpace => 19,
            RuntimeError::InvalidCapabilityDerivation => 20,
            RuntimeError::InvalidSupervisionPolicy => 21,
            RuntimeError::InvalidMultiInput => 22,
            RuntimeError::DomainQuotaExceeded => 23,
            RuntimeError::InvalidContract => 24,
            RuntimeError::InvalidModule => 25,
            RuntimeError::AuthorityDenied => 26,
            RuntimeError::NodeUnavailable => 27,
        }
}

pub(crate) fn encode_spec(spec: &ContinuationSpec) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(20 + spec.frame_bytes.len());
    put_u32(&mut bytes, spec.state_access as u32);
    put_u32(&mut bytes, spec.run_class);
    put_u32(&mut bytes, spec.resume_point);
    put_u32(&mut bytes, spec.max_steps);
    put_u32(&mut bytes, spec.frame_bytes.len() as u32);
    bytes.extend_from_slice(&spec.frame_bytes);
    bytes
}

pub(crate) fn decode_spec(bytes: &[u8]) -> Option<ContinuationSpec> {
    let mut cursor = Cursor::new(bytes);
    let state_access = match cursor.u32()? {
        1 => StateAccess::ReadOnly,
        2 => StateAccess::Mutable,
        _ => return None,
    };
    let run_class = cursor.u32()?;
    let resume_point = cursor.u32()?;
    let max_steps = cursor.u32()?;
    let frame_len = cursor.u32()? as usize;
    let frame_bytes = cursor.take(frame_len)?.to_vec();
    cursor.is_empty().then_some(ContinuationSpec::new(
        state_access,
        run_class,
        resume_point,
        frame_bytes,
        max_steps,
    ))
}

pub(crate) fn encode_message(message: &MessageDescriptor) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(68);
    put_u32(&mut bytes, message.type_id);
    put_u32(&mut bytes, message.flags);
    for value in [
        message.sender.to_u64(),
        message.receiver.to_u64(),
        message.sender_sequence,
        message.logical_timestamp,
        message.payload.to_u64(),
        message.transferred_capability.to_u64(),
        message.completion_future.to_u64(),
    ] {
        put_u64(&mut bytes, value);
    }
    bytes
}

pub(crate) fn process_mode(code: u32) -> Option<ProcessMode> {
    match code {
        1 => Some(ProcessMode::Serial),
        2 => Some(ProcessMode::Pure),
        3 => Some(ProcessMode::System),
        _ => None,
    }
}

pub(crate) fn object_kind(code: u32) -> Option<ObjectKind> {
    match code {
        1 => Some(ObjectKind::RawBytes),
        2 => Some(ObjectKind::ProcessState),
        3 => Some(ObjectKind::ContinuationFrame),
        4 => Some(ObjectKind::MessagePayload),
        5 => Some(ObjectKind::FrozenArray),
        6 => Some(ObjectKind::FutureValue),
        7 => Some(ObjectKind::TraceBuffer),
        _ => None,
    }
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(count)?;
        let bytes = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(bytes)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }
}
