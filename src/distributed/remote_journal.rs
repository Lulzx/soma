//! Authenticated remote validation for stateful lane-access journals.
//!
//! Journals remain speculative data: the worker returns only a conflict
//! decision, and the coordinator is the sole authority that may replay their
//! operations at canonical commit. Exact retries are content-addressed and
//! served from a ledger, but authorization is rechecked before that lookup.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::Rights;
use crate::scheduler::device::{
    reference_lane_conflicts, DeviceLaneAccess, DeviceLaneConflict, LaneConflictValidator,
    LaneValidationError, DEVICE_ACCESS_READ, DEVICE_ACCESS_WRITE,
};

const MAGIC: u32 = 0x534A_4E4C;
const VERSION: u16 = 1;
const REQUEST: u16 = 1;
const MAX_FRAME: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RequestId([u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
enum Status {
    Ok = 0,
    AuthorityDenied = 1,
    InvalidRequest = 2,
    ExecutionFailed = 3,
}

impl Status {
    fn decode(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::AuthorityDenied),
            2 => Some(Self::InvalidRequest),
            3 => Some(Self::ExecutionFailed),
            _ => None,
        }
    }
}

struct WireRequest {
    id: RequestId,
    epoch: u32,
    lane_count: u32,
    grant: RemoteGrant,
    accesses: Vec<DeviceLaneAccess>,
}

impl WireRequest {
    fn new(epoch: u32, lane_count: u32, grant: RemoteGrant, accesses: &[DeviceLaneAccess]) -> Self {
        let mut request = Self {
            id: RequestId([0; 32]),
            epoch,
            lane_count,
            grant,
            accesses: accesses.to_vec(),
        };
        request.id = request.computed_id();
        request
    }

    fn identity(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            12 + RemoteGrant::ENCODED_LEN
                + self.accesses.len() * std::mem::size_of::<DeviceLaneAccess>(),
        );
        put_u32(&mut bytes, self.epoch);
        put_u32(&mut bytes, self.lane_count);
        put_u32(&mut bytes, self.accesses.len() as u32);
        bytes.extend_from_slice(&self.grant.encode());
        for access in &self.accesses {
            put_u64(&mut bytes, access.resource);
            put_u32(&mut bytes, access.lane);
            put_u32(&mut bytes, access.resource_kind);
            put_u32(&mut bytes, access.mode);
            put_u32(&mut bytes, access.ordinal);
        }
        bytes
    }

    fn computed_id(&self) -> RequestId {
        RequestId(Sha256::digest(self.identity()).into())
    }

    fn encode(&self) -> Vec<u8> {
        let identity = self.identity();
        let mut bytes = Vec::with_capacity(40 + identity.len());
        put_u32(&mut bytes, MAGIC);
        put_u16(&mut bytes, VERSION);
        put_u16(&mut bytes, REQUEST);
        bytes.extend_from_slice(&self.id.0);
        bytes.extend_from_slice(&identity);
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != VERSION || cursor.u16()? != REQUEST {
            return None;
        }
        let id = RequestId(cursor.array()?);
        let epoch = cursor.u32()?;
        let lane_count = cursor.u32()?;
        let count = cursor.u32()? as usize;
        if count > MAX_FRAME / std::mem::size_of::<DeviceLaneAccess>() {
            return None;
        }
        let grant = RemoteGrant::decode(cursor.take(RemoteGrant::ENCODED_LEN)?)?;
        let mut accesses = Vec::with_capacity(count);
        for _ in 0..count {
            let resource = cursor.u64()?;
            let lane = cursor.u32()?;
            let resource_kind = cursor.u32()?;
            let mode = cursor.u32()?;
            let ordinal = cursor.u32()?;
            accesses.push(DeviceLaneAccess::new(
                lane,
                resource_kind,
                resource,
                mode,
                ordinal,
            ));
        }
        if !cursor.is_empty() {
            return None;
        }
        let request = Self {
            id,
            epoch,
            lane_count,
            grant,
            accesses,
        };
        (request.computed_id() == id).then_some(request)
    }
}

#[derive(Clone)]
struct WireResponse {
    id: RequestId,
    status: Status,
    conflicts: Vec<DeviceLaneConflict>,
}

impl WireResponse {
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(44 + self.conflicts.len() * 16);
        put_u32(&mut bytes, MAGIC);
        put_u16(&mut bytes, VERSION);
        put_u16(&mut bytes, self.status as u16);
        bytes.extend_from_slice(&self.id.0);
        put_u32(&mut bytes, self.conflicts.len() as u32);
        for conflict in &self.conflicts {
            put_u32(&mut bytes, conflict.lane);
            put_u32(&mut bytes, conflict.conflicts);
            put_u32(&mut bytes, conflict.first_other_lane);
            put_u32(&mut bytes, conflict.reserved);
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != VERSION {
            return None;
        }
        let status = Status::decode(cursor.u16()?)?;
        let id = RequestId(cursor.array()?);
        let count = cursor.u32()? as usize;
        if count > MAX_FRAME / 16 {
            return None;
        }
        let mut conflicts = Vec::with_capacity(count);
        for _ in 0..count {
            conflicts.push(DeviceLaneConflict {
                lane: cursor.u32()?,
                conflicts: cursor.u32()?,
                first_other_lane: cursor.u32()?,
                reserved: cursor.u32()?,
            });
        }
        cursor.is_empty().then_some(Self {
            id,
            status,
            conflicts,
        })
    }
}

pub struct RemoteJournalService {
    node: NodeId,
    target: RemoteRef,
    object_version: u32,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    ledger: HashMap<RequestId, WireResponse>,
    applied_requests: u64,
}

impl RemoteJournalService {
    pub fn new(
        node: NodeId,
        target: RemoteRef,
        object_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Self {
        Self {
            node,
            target,
            object_version,
            authority,
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
                conflicts: Vec::new(),
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
                conflicts: Vec::new(),
            };
        }
        if let Some(cached) = self.ledger.get(&request.id) {
            return cached.clone();
        }
        let valid = request.accesses.iter().all(|access| {
            access.lane < request.lane_count
                && matches!(access.mode, DEVICE_ACCESS_READ | DEVICE_ACCESS_WRITE)
        });
        let response = if valid {
            self.applied_requests += 1;
            WireResponse {
                id: request.id,
                status: Status::Ok,
                conflicts: reference_lane_conflicts(&request.accesses, request.lane_count),
            }
        } else {
            WireResponse {
                id: request.id,
                status: Status::InvalidRequest,
                conflicts: Vec::new(),
            }
        };
        self.ledger.insert(request.id, response.clone());
        response
    }
}

pub struct RemoteJournalServer;

impl RemoteJournalServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteJournalService>>,
        requests: usize,
    ) -> std::io::Result<()> {
        let mut served = 0;
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
                    .map_err(|_| std::io::Error::other("remote journal service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &response)?;
                served += 1;
            }
        }
        Ok(())
    }
}

pub struct RemoteJournalValidator {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
}

impl RemoteJournalValidator {
    pub fn new(endpoint: SocketAddr, grant: RemoteGrant, epoch: u32) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_secs(5),
            grant,
            epoch,
        }
    }

    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
}

impl LaneConflictValidator for RemoteJournalValidator {
    fn validate_lane_journals(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError> {
        if accesses.len() > u32::MAX as usize
            || accesses.iter().any(|access| access.lane >= lane_count)
        {
            return Err(LaneValidationError::InvalidInput);
        }
        let request = WireRequest::new(self.epoch, lane_count, self.grant, accesses);
        let mut stream = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| LaneValidationError::Unavailable)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|_| LaneValidationError::Unavailable)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|_| LaneValidationError::Unavailable)?;
        write_frame(&mut stream, &request.encode()).map_err(|_| LaneValidationError::NodeLost)?;
        let bytes = read_frame(&mut stream).map_err(|_| LaneValidationError::NodeLost)?;
        let response = WireResponse::decode(&bytes).ok_or(LaneValidationError::ProtocolError)?;
        if response.id != request.id {
            return Err(LaneValidationError::ProtocolError);
        }
        match response.status {
            Status::Ok if response.conflicts.len() == lane_count as usize => Ok(response.conflicts),
            Status::Ok => Err(LaneValidationError::ProtocolError),
            Status::AuthorityDenied => Err(LaneValidationError::AuthorityDenied),
            Status::InvalidRequest => Err(LaneValidationError::InvalidInput),
            Status::ExecutionFailed => Err(LaneValidationError::ExecutionFailed),
        }
    }
}

fn write_frame(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "remote journal frame too large",
        ));
    }
    stream.write_all(&(bytes.len() as u32).to_le_bytes())?;
    stream.write_all(bytes)
}

fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut size = [0; 4];
    stream.read_exact(&mut size)?;
    let size = u32::from_le_bytes(size) as usize;
    if size > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "remote journal frame too large",
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

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(count)?;
        let bytes = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(bytes)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }
}
