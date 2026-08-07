//! Authoritative remote single-assignment futures.
//!
//! Unlike speculative operation journals, this service stores the canonical
//! future state on the node named by `RemoteRef`. Content-addressed retries are
//! idempotent, while a distinct second resolution is rejected.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::{Ref64, Rights};

const MAGIC: u32 = 0x5346_5554;
const VERSION: u16 = 1;
const POLL: u16 = 1;
const RESOLVE: u16 = 2;
const FRAME_LEN: usize = 4 + 2 + 2 + 32 + 4 + RemoteGrant::ENCODED_LEN + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RequestId([u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteFutureState {
    Pending,
    Resolved { value: Ref64, resolved_epoch: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteFutureError {
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    AuthorityDenied,
    AlreadyResolved,
    InvalidRequest,
}

#[derive(Clone, Copy)]
struct WireRequest {
    id: RequestId,
    opcode: u16,
    epoch: u32,
    grant: RemoteGrant,
    value: Ref64,
}

impl WireRequest {
    fn new(opcode: u16, epoch: u32, grant: RemoteGrant, value: Ref64) -> Self {
        let mut request = Self {
            id: RequestId([0; 32]),
            opcode,
            epoch,
            grant,
            value,
        };
        request.id = RequestId(Sha256::digest(request.identity()).into());
        request
    }

    fn identity(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(2 + 4 + RemoteGrant::ENCODED_LEN + 8);
        bytes.extend_from_slice(&self.opcode.to_le_bytes());
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.grant.encode());
        bytes.extend_from_slice(&self.value.to_u64().to_le_bytes());
        bytes
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FRAME_LEN);
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.opcode.to_le_bytes());
        bytes.extend_from_slice(&self.id.0);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.grant.encode());
        bytes.extend_from_slice(&self.value.to_u64().to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != FRAME_LEN {
            return None;
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != VERSION {
            return None;
        }
        let opcode = cursor.u16()?;
        let id = RequestId(cursor.array()?);
        let epoch = cursor.u32()?;
        let grant = RemoteGrant::decode(cursor.take(RemoteGrant::ENCODED_LEN)?)?;
        let value = Ref64::from_u64(cursor.u64()?);
        let request = Self {
            id,
            opcode,
            epoch,
            grant,
            value,
        };
        (RequestId(Sha256::digest(request.identity()).into()) == id).then_some(request)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
enum Status {
    Pending = 0,
    Resolved = 1,
    AuthorityDenied = 2,
    AlreadyResolved = 3,
    InvalidRequest = 4,
}

impl Status {
    fn decode(value: u16) -> Option<Self> {
        Some(match value {
            0 => Self::Pending,
            1 => Self::Resolved,
            2 => Self::AuthorityDenied,
            3 => Self::AlreadyResolved,
            4 => Self::InvalidRequest,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct WireResponse {
    id: RequestId,
    status: Status,
    value: Ref64,
    resolved_epoch: u32,
}

impl WireResponse {
    fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(52);
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.status as u16).to_le_bytes());
        bytes.extend_from_slice(&self.id.0);
        bytes.extend_from_slice(&self.value.to_u64().to_le_bytes());
        bytes.extend_from_slice(&self.resolved_epoch.to_le_bytes());
        bytes
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let status = Status::decode(c.u16()?)?;
        let id = RequestId(c.array()?);
        let value = Ref64::from_u64(c.u64()?);
        let resolved_epoch = c.u32()?;
        c.is_empty().then_some(Self {
            id,
            status,
            value,
            resolved_epoch,
        })
    }
}

/// Canonical state for one future owned by `node`.
pub struct RemoteFutureService {
    node: NodeId,
    target: RemoteRef,
    object_version: u32,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    state: RemoteFutureState,
    ledger: HashMap<RequestId, WireResponse>,
    applied_resolutions: u64,
}

impl RemoteFutureService {
    pub fn new(
        node: NodeId,
        target: RemoteRef,
        object_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Self {
        assert_eq!(
            target.node, node,
            "remote future must be owned by the serving node"
        );
        Self {
            node,
            target,
            object_version,
            authority,
            state: RemoteFutureState::Pending,
            ledger: HashMap::new(),
            applied_resolutions: 0,
        }
    }
    pub fn state(&self) -> RemoteFutureState {
        self.state
    }
    pub fn applied_resolutions(&self) -> u64 {
        self.applied_resolutions
    }

    fn handle(&mut self, frame: &[u8]) -> WireResponse {
        let Some(request) = WireRequest::decode(frame) else {
            return invalid_response();
        };
        let required = match request.opcode {
            POLL => Rights::AWAIT,
            RESOLVE => Rights::RESOLVE,
            _ => {
                return WireResponse {
                    id: request.id,
                    ..invalid_response()
                }
            }
        };
        let authorized = self.authority.lock().ok().is_some_and(|store| {
            store
                .authorize(
                    &request.grant,
                    self.node,
                    self.target,
                    required,
                    self.object_version,
                    request.epoch,
                )
                .is_ok()
        });
        if !authorized {
            return WireResponse {
                id: request.id,
                status: Status::AuthorityDenied,
                value: Ref64::NULL,
                resolved_epoch: 0,
            };
        }
        // Authorization deliberately precedes replay lookup, so revocation also
        // denies an otherwise idempotent retry. Polls are observations rather
        // than mutations and must not cache a stale Pending state.
        if request.opcode == RESOLVE {
            if let Some(response) = self.ledger.get(&request.id) {
                return *response;
            }
        }
        let response = match request.opcode {
            POLL => response_for_state(request.id, self.state),
            RESOLVE => match self.state {
                RemoteFutureState::Pending => {
                    self.state = RemoteFutureState::Resolved {
                        value: request.value,
                        resolved_epoch: request.epoch,
                    };
                    self.applied_resolutions += 1;
                    response_for_state(request.id, self.state)
                }
                RemoteFutureState::Resolved {
                    value,
                    resolved_epoch,
                } => WireResponse {
                    id: request.id,
                    status: Status::AlreadyResolved,
                    value,
                    resolved_epoch,
                },
            },
            _ => unreachable!(),
        };
        if request.opcode == RESOLVE {
            self.ledger.insert(request.id, response);
        }
        response
    }
}

fn response_for_state(id: RequestId, state: RemoteFutureState) -> WireResponse {
    match state {
        RemoteFutureState::Pending => WireResponse {
            id,
            status: Status::Pending,
            value: Ref64::NULL,
            resolved_epoch: 0,
        },
        RemoteFutureState::Resolved {
            value,
            resolved_epoch,
        } => WireResponse {
            id,
            status: Status::Resolved,
            value,
            resolved_epoch,
        },
    }
}
fn invalid_response() -> WireResponse {
    WireResponse {
        id: RequestId([0; 32]),
        status: Status::InvalidRequest,
        value: Ref64::NULL,
        resolved_epoch: 0,
    }
}

pub struct RemoteFutureServer;
impl RemoteFutureServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteFutureService>>,
        requests: usize,
    ) -> std::io::Result<()> {
        let mut served = 0;
        while served < requests {
            let (mut stream, _) = listener.accept()?;
            while served < requests {
                let frame = match read_frame(&mut stream) {
                    Ok(frame) => frame,
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
                let response = service
                    .lock()
                    .map_err(|_| std::io::Error::other("remote future service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &response)?;
                served += 1;
            }
        }
        Ok(())
    }
}

pub struct RemoteFutureClient {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
}
impl RemoteFutureClient {
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
    pub fn poll(&self) -> Result<RemoteFutureState, RemoteFutureError> {
        self.round_trip(WireRequest::new(POLL, self.epoch, self.grant, Ref64::NULL))
    }
    pub fn resolve(&self, value: Ref64) -> Result<RemoteFutureState, RemoteFutureError> {
        self.round_trip(WireRequest::new(RESOLVE, self.epoch, self.grant, value))
    }
    fn round_trip(&self, request: WireRequest) -> Result<RemoteFutureState, RemoteFutureError> {
        let mut stream = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| RemoteFutureError::NodeUnavailable)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|_| RemoteFutureError::NodeUnavailable)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|_| RemoteFutureError::NodeUnavailable)?;
        write_frame(&mut stream, &request.encode()).map_err(|_| RemoteFutureError::NodeLost)?;
        let bytes = read_frame(&mut stream).map_err(|_| RemoteFutureError::NodeLost)?;
        let response = WireResponse::decode(&bytes).ok_or(RemoteFutureError::ProtocolError)?;
        if response.id != request.id {
            return Err(RemoteFutureError::ProtocolError);
        }
        match response.status {
            Status::Pending => Ok(RemoteFutureState::Pending),
            Status::Resolved => Ok(RemoteFutureState::Resolved {
                value: response.value,
                resolved_epoch: response.resolved_epoch,
            }),
            Status::AuthorityDenied => Err(RemoteFutureError::AuthorityDenied),
            Status::AlreadyResolved => Err(RemoteFutureError::AlreadyResolved),
            Status::InvalidRequest => Err(RemoteFutureError::InvalidRequest),
        }
    }
}

/// Result of coupling an authoritative remote future observation to a local
/// SOMA continuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteAwaitOutcome {
    Registered,
    AlreadySettled { value: Ref64, resolved_epoch: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteFutureBridgeError {
    Kernel(crate::kernel::RuntimeError),
    Remote(RemoteFutureError),
}

/// Small scheduling bridge for one remotely-owned future.
///
/// The bridge never creates a local future descriptor and never caches
/// canonical state across epoch boundaries. It only retains local waiter ids
/// plus the observation made for the current boundary. Thus resolution and its
/// epoch remain authoritative at `RemoteFutureService`, while wakeups enter the
/// ordinary SOMA runnable bins.
pub struct RemoteFutureBridge {
    target: RemoteRef,
    client: RemoteFutureClient,
    waiters: Vec<Ref64>,
    boundary_observation: Option<(u32, Result<RemoteFutureState, RemoteFutureError>)>,
    contacted_owner: bool,
}

impl RemoteFutureBridge {
    pub fn new(target: RemoteRef, client: RemoteFutureClient) -> Self {
        Self {
            target,
            client,
            waiters: Vec::new(),
            boundary_observation: None,
            contacted_owner: false,
        }
    }

    pub fn waiter_count(&self) -> usize {
        self.waiters.len()
    }

    fn observe_boundary(&mut self, epoch: u32) -> Result<RemoteFutureState, RemoteFutureError> {
        if let Some((observed_epoch, result)) = self.boundary_observation {
            if observed_epoch == epoch {
                return result;
            }
        }
        self.client.set_epoch(epoch);
        let mut result = self.client.poll();
        if result.is_ok() {
            self.contacted_owner = true;
        } else if result == Err(RemoteFutureError::NodeUnavailable) && self.contacted_owner {
            // A connection failure before any successful contact is an
            // unavailable node. Losing a node which already accepted this
            // bridge's waiter is a distinct, stronger condition.
            result = Err(RemoteFutureError::NodeLost);
        }
        self.boundary_observation = Some((epoch, result));
        result
    }

    /// Observe and, if still pending, park `continuation`. All registrations in
    /// one epoch share the same observation, making the boundary independent of
    /// intra-epoch network timing.
    pub fn await_at_epoch_boundary(
        &mut self,
        kernel: &mut crate::kernel::Kernel,
        continuation: Ref64,
        next_run_class: u32,
    ) -> Result<RemoteAwaitOutcome, RemoteFutureBridgeError> {
        if self.waiters.contains(&continuation) {
            return Ok(RemoteAwaitOutcome::Registered);
        }
        match self
            .observe_boundary(kernel.current_epoch())
            .map_err(RemoteFutureBridgeError::Remote)?
        {
            RemoteFutureState::Pending => {
                kernel
                    .register_remote_future_waiter(continuation, self.target.entity, next_run_class)
                    .map_err(RemoteFutureBridgeError::Kernel)?;
                self.waiters.push(continuation);
                Ok(RemoteAwaitOutcome::Registered)
            }
            RemoteFutureState::Resolved {
                value,
                resolved_epoch,
            } => Ok(RemoteAwaitOutcome::AlreadySettled {
                value,
                resolved_epoch,
            }),
        }
    }

    /// Poll once at this kernel epoch boundary and wake all registered local
    /// continuations if the owner reports resolution. Repeated calls in the
    /// same epoch are idempotent and perform no extra network request.
    pub fn sync_epoch_boundary(
        &mut self,
        kernel: &mut crate::kernel::Kernel,
    ) -> Result<RemoteFutureState, RemoteFutureBridgeError> {
        let state = self
            .observe_boundary(kernel.current_epoch())
            .map_err(RemoteFutureBridgeError::Remote)?;
        if matches!(state, RemoteFutureState::Resolved { .. }) {
            for waiter in self.waiters.drain(..) {
                kernel.wake_remote_future_waiter(waiter, self.target.entity);
            }
        }
        Ok(state)
    }
}

fn write_frame(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}
fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0; 8];
    stream.read_exact(&mut len)?;
    let len: usize = u64::from_le_bytes(len)
        .try_into()
        .map_err(|_| std::io::Error::other("oversized frame"))?;
    if len > 1024 * 1024 {
        return Err(std::io::Error::other("oversized frame"));
    }
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
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
