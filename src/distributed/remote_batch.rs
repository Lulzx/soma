//! Authenticated TCP backend for evaluator batches.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::Rights;
use crate::compiler::body::EvaluatorProgram;
use crate::executives::batch::{
    AuxArray, BackendError, BackendKind, BatchBackend, BatchRequest, CpuReferenceBackend,
};
use crate::kernel::payload::Payload;

const MAGIC: u32 = 0x534F_4D41;
const WIRE_VERSION: u16 = 1;
const MAX_FRAME: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
enum Status {
    Ok = 0,
    AuthorityDenied = 1,
    InvalidRequest = 2,
    UnsupportedEvaluator = 3,
    ExecutionFailed = 4,
}

impl Status {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::AuthorityDenied),
            2 => Some(Self::InvalidRequest),
            3 => Some(Self::UnsupportedEvaluator),
            4 => Some(Self::ExecutionFailed),
            _ => None,
        }
    }
}

struct WireRequest {
    id: RequestId,
    epoch: u32,
    evaluator_id: u32,
    element_count: u32,
    element_stride: u32,
    aux_count: u32,
    aux_stride: u32,
    grant: RemoteGrant,
    inputs: Vec<u8>,
    aux: Vec<u8>,
}

impl WireRequest {
    fn new(
        epoch: u32,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
        grant: RemoteGrant,
    ) -> Self {
        let mut request = Self {
            id: RequestId([0; 32]),
            epoch,
            evaluator_id,
            element_count,
            element_stride,
            aux_count: aux.element_count,
            aux_stride: aux.element_stride,
            grant,
            inputs: inputs.to_vec(),
            aux: aux.bytes.to_vec(),
        };
        request.id = request.computed_id();
        request
    }

    fn computed_id(&self) -> RequestId {
        let mut hash = Sha256::new();
        hash.update(self.identity_bytes());
        RequestId(hash.finalize().into())
    }

    fn identity_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(138 + self.inputs.len() + self.aux.len());
        put_u32(&mut out, self.epoch);
        put_u32(&mut out, self.evaluator_id);
        put_u32(&mut out, self.element_count);
        put_u32(&mut out, self.element_stride);
        put_u32(&mut out, self.aux_count);
        put_u32(&mut out, self.aux_stride);
        put_u64(&mut out, self.inputs.len() as u64);
        put_u64(&mut out, self.aux.len() as u64);
        out.extend_from_slice(&self.grant.encode());
        out.extend_from_slice(&self.inputs);
        out.extend_from_slice(&self.aux);
        out
    }

    fn encode(&self) -> Vec<u8> {
        let identity = self.identity_bytes();
        let mut out = Vec::with_capacity(40 + identity.len());
        put_u32(&mut out, MAGIC);
        put_u16(&mut out, WIRE_VERSION);
        put_u16(&mut out, 1);
        out.extend_from_slice(&self.id.0);
        out.extend_from_slice(&identity);
        out
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != WIRE_VERSION || cursor.u16()? != 1 {
            return None;
        }
        let id = RequestId(cursor.array()?);
        let epoch = cursor.u32()?;
        let evaluator_id = cursor.u32()?;
        let element_count = cursor.u32()?;
        let element_stride = cursor.u32()?;
        let aux_count = cursor.u32()?;
        let aux_stride = cursor.u32()?;
        let input_len: usize = cursor.u64()?.try_into().ok()?;
        let aux_len: usize = cursor.u64()?.try_into().ok()?;
        if input_len > MAX_FRAME
            || aux_len > MAX_FRAME
            || input_len.checked_add(aux_len)? > MAX_FRAME
        {
            return None;
        }
        let grant = RemoteGrant::decode(cursor.take(RemoteGrant::ENCODED_LEN)?)?;
        let inputs = cursor.take(input_len)?.to_vec();
        let aux = cursor.take(aux_len)?.to_vec();
        if !cursor.is_empty() {
            return None;
        }
        let request = Self {
            id,
            epoch,
            evaluator_id,
            element_count,
            element_stride,
            aux_count,
            aux_stride,
            grant,
            inputs,
            aux,
        };
        (request.computed_id() == id).then_some(request)
    }
}

#[derive(Clone)]
struct WireResponse {
    id: RequestId,
    status: Status,
    bytes: Vec<u8>,
}

impl WireResponse {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(48 + self.bytes.len());
        put_u32(&mut out, MAGIC);
        put_u16(&mut out, WIRE_VERSION);
        put_u16(&mut out, self.status as u16);
        out.extend_from_slice(&self.id.0);
        put_u64(&mut out, self.bytes.len() as u64);
        out.extend_from_slice(&self.bytes);
        out
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != WIRE_VERSION {
            return None;
        }
        let status = Status::from_u16(cursor.u16()?)?;
        let id = RequestId(cursor.array()?);
        let len: usize = cursor.u64()?.try_into().ok()?;
        if len > MAX_FRAME {
            return None;
        }
        let payload = cursor.take(len)?.to_vec();
        cursor.is_empty().then_some(Self {
            id,
            status,
            bytes: payload,
        })
    }
}

pub struct RemoteBatchService {
    node: NodeId,
    target: RemoteRef,
    object_version: u32,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    backend: CpuReferenceBackend,
    ledger: HashMap<RequestId, WireResponse>,
    applied_requests: u64,
}

impl RemoteBatchService {
    pub fn with(
        node: NodeId,
        target: RemoteRef,
        object_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
        programs: &[&EvaluatorProgram],
    ) -> Self {
        Self {
            node,
            target,
            object_version,
            authority,
            backend: CpuReferenceBackend::with(programs),
            ledger: HashMap::new(),
            applied_requests: 0,
        }
    }

    pub fn applied_requests(&self) -> u64 {
        self.applied_requests
    }

    fn handle(&mut self, frame: &[u8]) -> WireResponse {
        let Some(request) = WireRequest::decode(frame) else {
            return WireResponse {
                id: RequestId([0; 32]),
                status: Status::InvalidRequest,
                bytes: Vec::new(),
            };
        };
        let authorized = self.authority.lock().ok().is_some_and(|authority| {
            authority
                .authorize(
                    &request.grant,
                    self.node,
                    self.target,
                    Rights::READ,
                    self.object_version,
                    request.epoch,
                )
                .is_ok()
        });
        if !authorized {
            return WireResponse {
                id: request.id,
                status: Status::AuthorityDenied,
                bytes: Vec::new(),
            };
        }
        if let Some(cached) = self.ledger.get(&request.id) {
            return cached.clone();
        }
        let aux = AuxArray::new(&request.aux, request.aux_count, request.aux_stride);
        let (status, bytes) = match self.backend.evaluate_with_aux(
            request.evaluator_id,
            &request.inputs,
            request.element_count,
            request.element_stride,
            aux,
        ) {
            Ok(bytes) => {
                self.applied_requests += 1;
                (Status::Ok, bytes)
            }
            Err(BackendError::UnsupportedEvaluator) => (Status::UnsupportedEvaluator, Vec::new()),
            Err(BackendError::InvalidInput) => (Status::InvalidRequest, Vec::new()),
            Err(_) => (Status::ExecutionFailed, Vec::new()),
        };
        let response = WireResponse {
            id: request.id,
            status,
            bytes,
        };
        self.ledger.insert(request.id, response.clone());
        response
    }
}

pub struct RemoteBatchServer;

impl RemoteBatchServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteBatchService>>,
        requests: usize,
    ) -> std::io::Result<()> {
        let mut served = 0usize;
        while served < requests {
            let (mut stream, _) = listener.accept()?;
            while served < requests {
                let frame = match read_frame(&mut stream) {
                    Ok(frame) => frame,
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset
                        ) =>
                    {
                        break
                    }
                    Err(error) => return Err(error),
                };
                let response = service
                    .lock()
                    .map_err(|_| std::io::Error::other("remote service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &response)?;
                served += 1;
            }
        }
        Ok(())
    }
}

pub struct RemoteBatchBackend {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
    programs: HashMap<u32, (u32, u32)>,
}

impl RemoteBatchBackend {
    pub fn with(
        endpoint: SocketAddr,
        grant: RemoteGrant,
        epoch: u32,
        programs: &[&EvaluatorProgram],
    ) -> Result<Self, BackendError> {
        let mut backend = Self {
            endpoint,
            timeout: Duration::from_secs(5),
            grant,
            epoch,
            programs: HashMap::new(),
        };
        for program in programs {
            backend.install(program)?;
        }
        Ok(backend)
    }

    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    fn round_trip(&self, request: WireRequest) -> Result<Vec<u8>, BackendError> {
        let mut stream = self.connect()?;
        self.round_trip_on(&mut stream, request)
    }

    fn connect(&self) -> Result<TcpStream, BackendError> {
        let stream = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| BackendError::NodeUnavailable)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|_| BackendError::NodeUnavailable)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|_| BackendError::NodeUnavailable)?;
        Ok(stream)
    }

    fn round_trip_on(
        &self,
        stream: &mut TcpStream,
        request: WireRequest,
    ) -> Result<Vec<u8>, BackendError> {
        write_frame(stream, &request.encode()).map_err(|_| BackendError::NodeLost)?;
        let response = read_frame(stream).map_err(|_| BackendError::NodeLost)?;
        let response = WireResponse::decode(&response).ok_or(BackendError::ProtocolError)?;
        if response.id != request.id {
            return Err(BackendError::ProtocolError);
        }
        match response.status {
            Status::Ok => Ok(response.bytes),
            Status::AuthorityDenied => Err(BackendError::AuthorityDenied),
            Status::InvalidRequest => Err(BackendError::InvalidInput),
            Status::UnsupportedEvaluator => Err(BackendError::UnsupportedEvaluator),
            Status::ExecutionFailed => Err(BackendError::ExecutionFailed),
        }
    }
}

impl BatchBackend for RemoteBatchBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Remote
    }

    fn install(&mut self, program: &EvaluatorProgram) -> Result<(), BackendError> {
        self.programs
            .insert(program.id(), (program.stride(), program.aux_stride()));
        Ok(())
    }

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        self.evaluate_with_aux(
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            AuxArray::NONE,
        )
    }

    fn evaluate_with_aux(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Vec<u8>, BackendError> {
        let Some((stride, aux_stride)) = self.programs.get(&evaluator_id) else {
            return Err(BackendError::UnsupportedEvaluator);
        };
        if *stride != element_stride || *aux_stride != aux.element_stride {
            return Err(BackendError::InvalidInput);
        }
        self.round_trip(WireRequest::new(
            self.epoch,
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            aux,
            self.grant,
        ))
    }

    fn evaluate_epoch(
        &mut self,
        requests: &[BatchRequest<'_>],
    ) -> Result<Vec<Payload>, BackendError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let mut wire = Vec::with_capacity(requests.len());
        for request in requests {
            let Some((stride, aux_stride)) = self.programs.get(&request.evaluator_id) else {
                return Err(BackendError::UnsupportedEvaluator);
            };
            if *stride != request.element_stride || *aux_stride != request.aux.element_stride {
                return Err(BackendError::InvalidInput);
            }
            wire.push(WireRequest::new(
                self.epoch,
                request.evaluator_id,
                request.inputs,
                request.element_count,
                request.element_stride,
                request.aux,
                self.grant,
            ));
        }
        let mut stream = self.connect()?;
        wire.into_iter()
            .map(|request| self.round_trip_on(&mut stream, request).map(Payload::from))
            .collect()
    }
}

fn write_frame(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "remote frame too large",
        ));
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)
}

fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut size = [0u8; 4];
    stream.read_exact(&mut size)?;
    let size = u32::from_le_bytes(size) as usize;
    if size > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "remote frame too large",
        ));
    }
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
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

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(len)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.array()?))
    }

    fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }
}
