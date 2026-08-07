//! Authoritative growable byte objects owned by a remote node.
//!
//! The client stores only an endpoint and a grant: canonical bytes and their
//! optimistic version exist solely in `RemoteObjectService`. Mutating requests
//! are content-addressed and successful results are replayed from an apply-once
//! ledger, after checking the grant against the live revocation registry.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::Rights;

const MAGIC: u32 = 0x534f_424a;
const VERSION: u16 = 1;
const READ: u16 = 1;
const WRITE: u16 = 2;
const APPEND: u16 = 3;
/// Largest accepted request or response body (not including the length prefix).
pub const MAX_REMOTE_OBJECT_FRAME: usize = 1024 * 1024;
const REQUEST_FIXED: usize = 4 + 2 + 2 + 32 + 4 + RemoteGrant::ENCODED_LEN + 8 + 8 + 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RemoteObjectRequestId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteObjectRead {
    pub version: u64,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteObjectWriteOutcome {
    pub version: u64,
    pub byte_length: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteObjectError {
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    AuthorityDenied,
    StaleVersion { expected: u64, actual: u64 },
    InvalidRequest,
    FrameTooLarge,
}

#[derive(Clone)]
struct WireRequest {
    id: RemoteObjectRequestId,
    opcode: u16,
    epoch: u32,
    grant: RemoteGrant,
    expected_version: u64,
    offset: u64,
    payload: Vec<u8>,
}
impl WireRequest {
    fn new(
        opcode: u16,
        epoch: u32,
        grant: RemoteGrant,
        expected_version: u64,
        offset: u64,
        payload: Vec<u8>,
    ) -> Self {
        let mut out = Self {
            id: RemoteObjectRequestId([0; 32]),
            opcode,
            epoch,
            grant,
            expected_version,
            offset,
            payload,
        };
        out.id = RemoteObjectRequestId(Sha256::digest(out.identity()).into());
        out
    }
    fn identity(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(2 + 4 + RemoteGrant::ENCODED_LEN + 20 + self.payload.len());
        out.extend_from_slice(&self.opcode.to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.grant.encode());
        out.extend_from_slice(&self.expected_version.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
    fn encode(&self) -> Option<Vec<u8>> {
        let size = REQUEST_FIXED.checked_add(self.payload.len())?;
        if size > MAX_REMOTE_OBJECT_FRAME || self.payload.len() > u32::MAX as usize {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.opcode.to_le_bytes());
        out.extend_from_slice(&self.id.0);
        out.extend_from_slice(&self.identity()[2..]);
        Some(out)
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_REMOTE_OBJECT_FRAME {
            return None;
        }
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let opcode = c.u16()?;
        let id = RemoteObjectRequestId(c.array()?);
        let epoch = c.u32()?;
        let grant = RemoteGrant::decode(c.take(RemoteGrant::ENCODED_LEN)?)?;
        let expected_version = c.u64()?;
        let offset = c.u64()?;
        let len = c.u32()? as usize;
        let payload = c.take(len)?.to_vec();
        if !c.is_empty() {
            return None;
        }
        let req = Self {
            id,
            opcode,
            epoch,
            grant,
            expected_version,
            offset,
            payload,
        };
        (RemoteObjectRequestId(Sha256::digest(req.identity()).into()) == id).then_some(req)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
enum Status {
    Ok = 0,
    AuthorityDenied = 1,
    StaleVersion = 2,
    InvalidRequest = 3,
}
impl Status {
    fn decode(v: u16) -> Option<Self> {
        Some(match v {
            0 => Self::Ok,
            1 => Self::AuthorityDenied,
            2 => Self::StaleVersion,
            3 => Self::InvalidRequest,
            _ => return None,
        })
    }
}
#[derive(Clone)]
struct WireResponse {
    id: RemoteObjectRequestId,
    status: Status,
    version: u64,
    byte_length: u64,
    payload: Vec<u8>,
}
impl WireResponse {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(60 + self.payload.len());
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.status as u16).to_le_bytes());
        out.extend_from_slice(&self.id.0);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.byte_length.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_REMOTE_OBJECT_FRAME {
            return None;
        }
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let status = Status::decode(c.u16()?)?;
        let id = RemoteObjectRequestId(c.array()?);
        let version = c.u64()?;
        let byte_length = c.u64()?;
        let n = c.u32()? as usize;
        let payload = c.take(n)?.to_vec();
        c.is_empty().then_some(Self {
            id,
            status,
            version,
            byte_length,
            payload,
        })
    }
}
fn response(
    id: RemoteObjectRequestId,
    status: Status,
    version: u64,
    byte_length: usize,
    payload: Vec<u8>,
) -> WireResponse {
    WireResponse {
        id,
        status,
        version,
        byte_length: byte_length as u64,
        payload,
    }
}

/// The sole canonical storage for one remote growable object.
pub struct RemoteObjectService {
    node: NodeId,
    target: RemoteRef,
    authority_object_version: u32,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    bytes: Vec<u8>,
    version: u64,
    ledger: HashMap<RemoteObjectRequestId, WireResponse>,
    applied_writes: u64,
}
impl RemoteObjectService {
    pub fn new(
        node: NodeId,
        target: RemoteRef,
        authority_object_version: u32,
        initial_bytes: Vec<u8>,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Self {
        assert_eq!(
            target.node, node,
            "remote object must be owned by the serving node"
        );
        Self {
            node,
            target,
            authority_object_version,
            authority,
            bytes: initial_bytes,
            version: 0,
            ledger: HashMap::new(),
            applied_writes: 0,
        }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn applied_writes(&self) -> u64 {
        self.applied_writes
    }
    fn handle(&mut self, frame: &[u8]) -> WireResponse {
        let Some(req) = WireRequest::decode(frame) else {
            return response(
                RemoteObjectRequestId([0; 32]),
                Status::InvalidRequest,
                self.version,
                self.bytes.len(),
                vec![],
            );
        };
        let required = match req.opcode {
            READ => Rights::READ,
            WRITE | APPEND => Rights::WRITE,
            _ => {
                return response(
                    req.id,
                    Status::InvalidRequest,
                    self.version,
                    self.bytes.len(),
                    vec![],
                )
            }
        };
        let authorized = self.authority.lock().ok().is_some_and(|a| {
            a.authorize(
                &req.grant,
                self.node,
                self.target,
                required,
                self.authority_object_version,
                req.epoch,
            )
            .is_ok()
        });
        if !authorized {
            return response(
                req.id,
                Status::AuthorityDenied,
                self.version,
                self.bytes.len(),
                vec![],
            );
        }
        // Revocation is intentionally checked before replay.
        if matches!(req.opcode, WRITE | APPEND) {
            if let Some(cached) = self.ledger.get(&req.id) {
                return cached.clone();
            }
        }
        match req.opcode {
            READ => {
                if !req.payload.is_empty() {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                }
                let Ok(offset) = usize::try_from(req.offset) else {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                };
                let Ok(length) = usize::try_from(req.expected_version) else {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                };
                let Some(end) = offset.checked_add(length) else {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                };
                if end > self.bytes.len()
                    || 60usize.saturating_add(length) > MAX_REMOTE_OBJECT_FRAME
                {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                }
                response(
                    req.id,
                    Status::Ok,
                    self.version,
                    self.bytes.len(),
                    self.bytes[offset..end].to_vec(),
                )
            }
            WRITE | APPEND => {
                if req.expected_version != self.version {
                    return response(
                        req.id,
                        Status::StaleVersion,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                }
                let Ok(mut offset) = usize::try_from(req.offset) else {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                };
                if req.opcode == APPEND {
                    offset = self.bytes.len();
                }
                let Some(end) = offset.checked_add(req.payload.len()) else {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                };
                // Writes may replace or grow, but cannot create an implicit sparse hole.
                if offset > self.bytes.len() {
                    return response(
                        req.id,
                        Status::InvalidRequest,
                        self.version,
                        self.bytes.len(),
                        vec![],
                    );
                }
                if end > self.bytes.len() {
                    self.bytes.resize(end, 0);
                }
                self.bytes[offset..end].copy_from_slice(&req.payload);
                self.version = self.version.wrapping_add(1);
                self.applied_writes += 1;
                let out = response(req.id, Status::Ok, self.version, self.bytes.len(), vec![]);
                self.ledger.insert(req.id, out.clone());
                out
            }
            _ => unreachable!(),
        }
    }
}

pub struct RemoteObjectServer;
impl RemoteObjectServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteObjectService>>,
        requests: usize,
    ) -> std::io::Result<()> {
        let mut served = 0;
        while served < requests {
            let (mut stream, _) = listener.accept()?;
            while served < requests {
                let frame = match read_frame(&mut stream) {
                    Ok(v) => v,
                    Err(e)
                        if matches!(
                            e.kind(),
                            std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset
                        ) =>
                    {
                        break
                    }
                    Err(e) => return Err(e),
                };
                let out = service
                    .lock()
                    .map_err(|_| std::io::Error::other("remote object service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &out)?;
                served += 1;
            }
        }
        Ok(())
    }
}

/// A capability-bearing transport proxy. It deliberately contains no bytes or
/// local object descriptor that could become a shadow copy of canonical state.
pub struct RemoteObjectClient {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
}
impl RemoteObjectClient {
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
    pub fn read(&self, offset: u64, length: usize) -> Result<RemoteObjectRead, RemoteObjectError> {
        let length = u64::try_from(length).map_err(|_| RemoteObjectError::FrameTooLarge)?;
        let r = self.round_trip(WireRequest::new(
            READ,
            self.epoch,
            self.grant,
            length,
            offset,
            vec![],
        ))?;
        match r.status {
            Status::Ok => Ok(RemoteObjectRead {
                version: r.version,
                bytes: r.payload,
            }),
            _ => Err(map_status(r.status, length, r.version)),
        }
    }
    pub fn write(
        &self,
        expected_version: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<RemoteObjectWriteOutcome, RemoteObjectError> {
        if REQUEST_FIXED
            .checked_add(bytes.len())
            .is_none_or(|n| n > MAX_REMOTE_OBJECT_FRAME)
        {
            return Err(RemoteObjectError::FrameTooLarge);
        }
        let r = self.round_trip(WireRequest::new(
            WRITE,
            self.epoch,
            self.grant,
            expected_version,
            offset,
            bytes.to_vec(),
        ))?;
        match r.status {
            Status::Ok => Ok(RemoteObjectWriteOutcome {
                version: r.version,
                byte_length: r.byte_length,
            }),
            _ => Err(map_status(r.status, expected_version, r.version)),
        }
    }
    pub fn append(
        &self,
        expected_version: u64,
        bytes: &[u8],
    ) -> Result<RemoteObjectWriteOutcome, RemoteObjectError> {
        if REQUEST_FIXED
            .checked_add(bytes.len())
            .is_none_or(|n| n > MAX_REMOTE_OBJECT_FRAME)
        {
            return Err(RemoteObjectError::FrameTooLarge);
        }
        let r = self.round_trip(WireRequest::new(
            APPEND,
            self.epoch,
            self.grant,
            expected_version,
            0,
            bytes.to_vec(),
        ))?;
        match r.status {
            Status::Ok => Ok(RemoteObjectWriteOutcome {
                version: r.version,
                byte_length: r.byte_length,
            }),
            _ => Err(map_status(r.status, expected_version, r.version)),
        }
    }
    fn round_trip(&self, req: WireRequest) -> Result<WireResponse, RemoteObjectError> {
        let body = req.encode().ok_or(RemoteObjectError::FrameTooLarge)?;
        let mut stream = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| RemoteObjectError::NodeUnavailable)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|_| RemoteObjectError::NodeUnavailable)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|_| RemoteObjectError::NodeUnavailable)?;
        write_frame(&mut stream, &body).map_err(|_| RemoteObjectError::NodeLost)?;
        let bytes = read_frame(&mut stream).map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidData {
                RemoteObjectError::ProtocolError
            } else {
                RemoteObjectError::NodeLost
            }
        })?;
        let response = WireResponse::decode(&bytes).ok_or(RemoteObjectError::ProtocolError)?;
        if response.id != req.id {
            return Err(RemoteObjectError::ProtocolError);
        }
        Ok(response)
    }
}
fn map_status(status: Status, expected: u64, actual: u64) -> RemoteObjectError {
    match status {
        Status::AuthorityDenied => RemoteObjectError::AuthorityDenied,
        Status::StaleVersion => RemoteObjectError::StaleVersion { expected, actual },
        Status::InvalidRequest => RemoteObjectError::InvalidRequest,
        Status::Ok => RemoteObjectError::ProtocolError,
    }
}
fn write_frame(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    if bytes.len() > MAX_REMOTE_OBJECT_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "oversized frame",
        ));
    }
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}
fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0; 8];
    stream.read_exact(&mut len)?;
    let n = usize::try_from(u64::from_le_bytes(len))
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "oversized frame"))?;
    if n > MAX_REMOTE_OBJECT_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "oversized frame",
        ));
    }
    let mut out = vec![0; n];
    stream.read_exact(&mut out)?;
    Ok(out)
}
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(n)?;
        let out = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(out)
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
        self.offset == self.bytes.len()
    }
}
