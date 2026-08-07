//! Authoritative cross-node process terminal notices.
//!
//! The child-owner stores the single canonical terminal outcome. Supervisors
//! poll it only at epoch boundaries and retain no shadow process descriptor.
//! Signed grants are checked on every request (including retries), so issuer
//! revocation remains effective after publication.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::{ExitReason, Ref64, Rights, SupervisionPolicy};

const MAGIC: u32 = 0x5355_5056;
const VERSION: u16 = 1;
const POLL: u16 = 1;
const PUBLISH: u16 = 2;
const FRAME_LEN: usize = 4 + 2 + 2 + 32 + 4 + RemoteGrant::ENCODED_LEN + 1 + 3 + 4 + 4 + 8 + 8 + 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RequestId([u8; 32]);

/// Owner-authored immutable process outcome. `restart_of` and
/// `restart_attempt` preserve lineage without manufacturing a supervisor-side
/// process descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteTerminalNotice {
    pub child: RemoteRef,
    pub reason: ExitReason,
    pub failure_count: u32,
    pub owner_epoch: u32,
    pub restart_of: RemoteRef,
    pub restart_attempt: u32,
}

impl RemoteTerminalNotice {
    pub fn new(child: RemoteRef, reason: ExitReason, failure_count: u32, owner_epoch: u32) -> Self {
        Self {
            child,
            reason,
            failure_count,
            owner_epoch,
            restart_of: RemoteRef {
                node: child.node,
                entity: Ref64::NULL,
            },
            restart_attempt: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteSupervisionState {
    Running,
    Terminal(RemoteTerminalNotice),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteSupervisionError {
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    AuthorityDenied,
    AlreadyPublished,
    InvalidRequest,
    UnsupportedPolicy,
}

#[derive(Clone, Copy)]
struct WireRequest {
    id: RequestId,
    opcode: u16,
    epoch: u32,
    grant: RemoteGrant,
    reason: u8,
    failure_count: u32,
    owner_epoch: u32,
    restart_node: u64,
    restart_entity: u64,
    restart_attempt: u32,
}

impl WireRequest {
    fn poll(epoch: u32, grant: RemoteGrant) -> Self {
        Self::new(POLL, epoch, grant, None)
    }
    fn publish(epoch: u32, grant: RemoteGrant, n: RemoteTerminalNotice) -> Self {
        Self::new(PUBLISH, epoch, grant, Some(n))
    }
    fn new(
        opcode: u16,
        epoch: u32,
        grant: RemoteGrant,
        notice: Option<RemoteTerminalNotice>,
    ) -> Self {
        let mut r = Self {
            id: RequestId([0; 32]),
            opcode,
            epoch,
            grant,
            reason: notice.map(|n| n.reason as u8).unwrap_or(0),
            failure_count: notice.map(|n| n.failure_count).unwrap_or(0),
            owner_epoch: notice.map(|n| n.owner_epoch).unwrap_or(0),
            restart_node: notice.map(|n| n.restart_of.node.0).unwrap_or(0),
            restart_entity: notice.map(|n| n.restart_of.entity.to_u64()).unwrap_or(0),
            restart_attempt: notice.map(|n| n.restart_attempt).unwrap_or(0),
        };
        r.id = RequestId(Sha256::digest(r.identity()).into());
        r
    }
    fn identity(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&self.opcode.to_le_bytes());
        b.extend_from_slice(&self.epoch.to_le_bytes());
        b.extend_from_slice(&self.grant.encode());
        b.push(self.reason);
        b.extend_from_slice(&[0; 3]);
        b.extend_from_slice(&self.failure_count.to_le_bytes());
        b.extend_from_slice(&self.owner_epoch.to_le_bytes());
        b.extend_from_slice(&self.restart_node.to_le_bytes());
        b.extend_from_slice(&self.restart_entity.to_le_bytes());
        b.extend_from_slice(&self.restart_attempt.to_le_bytes());
        b
    }
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(FRAME_LEN);
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.opcode.to_le_bytes());
        b.extend_from_slice(&self.id.0);
        b.extend_from_slice(&self.epoch.to_le_bytes());
        b.extend_from_slice(&self.grant.encode());
        b.push(self.reason);
        b.extend_from_slice(&[0; 3]);
        b.extend_from_slice(&self.failure_count.to_le_bytes());
        b.extend_from_slice(&self.owner_epoch.to_le_bytes());
        b.extend_from_slice(&self.restart_node.to_le_bytes());
        b.extend_from_slice(&self.restart_entity.to_le_bytes());
        b.extend_from_slice(&self.restart_attempt.to_le_bytes());
        b
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != FRAME_LEN {
            return None;
        }
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let opcode = c.u16()?;
        let id = RequestId(c.array()?);
        let epoch = c.u32()?;
        let grant = RemoteGrant::decode(c.take(RemoteGrant::ENCODED_LEN)?)?;
        let reason = c.u8()?;
        c.take(3)?;
        let failure_count = c.u32()?;
        let owner_epoch = c.u32()?;
        let restart_node = c.u64()?;
        let restart_entity = c.u64()?;
        let restart_attempt = c.u32()?;
        let r = Self {
            id,
            opcode,
            epoch,
            grant,
            reason,
            failure_count,
            owner_epoch,
            restart_node,
            restart_entity,
            restart_attempt,
        };
        (RequestId(Sha256::digest(r.identity()).into()) == id).then_some(r)
    }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Running = 0,
    Terminal = 1,
    AuthorityDenied = 2,
    AlreadyPublished = 3,
    InvalidRequest = 4,
}
impl Status {
    fn decode(v: u16) -> Option<Self> {
        Some(match v {
            0 => Self::Running,
            1 => Self::Terminal,
            2 => Self::AuthorityDenied,
            3 => Self::AlreadyPublished,
            4 => Self::InvalidRequest,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct WireResponse {
    id: RequestId,
    status: Status,
    reason: u8,
    failure_count: u32,
    owner_epoch: u32,
    restart_node: u64,
    restart_entity: u64,
    restart_attempt: u32,
}
impl WireResponse {
    fn encode(self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&(self.status as u16).to_le_bytes());
        b.extend_from_slice(&self.id.0);
        b.push(self.reason);
        b.extend_from_slice(&[0; 3]);
        b.extend_from_slice(&self.failure_count.to_le_bytes());
        b.extend_from_slice(&self.owner_epoch.to_le_bytes());
        b.extend_from_slice(&self.restart_node.to_le_bytes());
        b.extend_from_slice(&self.restart_entity.to_le_bytes());
        b.extend_from_slice(&self.restart_attempt.to_le_bytes());
        b
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let status = Status::decode(c.u16()?)?;
        let id = RequestId(c.array()?);
        let reason = c.u8()?;
        c.take(3)?;
        let failure_count = c.u32()?;
        let owner_epoch = c.u32()?;
        let restart_node = c.u64()?;
        let restart_entity = c.u64()?;
        let restart_attempt = c.u32()?;
        c.is_empty().then_some(Self {
            id,
            status,
            reason,
            failure_count,
            owner_epoch,
            restart_node,
            restart_entity,
            restart_attempt,
        })
    }
}

pub struct RemoteSupervisionService {
    node: NodeId,
    target: RemoteRef,
    object_version: u32,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    terminal: Option<RemoteTerminalNotice>,
    ledger: HashMap<RequestId, WireResponse>,
    applied_publications: u64,
}
impl RemoteSupervisionService {
    pub fn new(
        node: NodeId,
        target: RemoteRef,
        object_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Self {
        assert_eq!(node, target.node);
        Self {
            node,
            target,
            object_version,
            authority,
            terminal: None,
            ledger: HashMap::new(),
            applied_publications: 0,
        }
    }
    pub fn state(&self) -> RemoteSupervisionState {
        self.terminal
            .map(RemoteSupervisionState::Terminal)
            .unwrap_or(RemoteSupervisionState::Running)
    }
    pub fn applied_publications(&self) -> u64 {
        self.applied_publications
    }
    fn handle(&mut self, frame: &[u8]) -> WireResponse {
        let Some(r) = WireRequest::decode(frame) else {
            return invalid();
        };
        let rights = match r.opcode {
            POLL => Rights::AWAIT,
            PUBLISH => Rights::WRITE,
            _ => {
                return WireResponse {
                    id: r.id,
                    ..invalid()
                }
            }
        };
        let authorized = self.authority.lock().ok().is_some_and(|s| {
            s.authorize(
                &r.grant,
                self.node,
                self.target,
                rights,
                self.object_version,
                r.epoch,
            )
            .is_ok()
        });
        if !authorized {
            return WireResponse {
                id: r.id,
                status: Status::AuthorityDenied,
                ..invalid()
            };
        }
        if r.opcode == PUBLISH {
            if let Some(x) = self.ledger.get(&r.id) {
                return *x;
            }
        }
        let response = match r.opcode {
            POLL => response(r.id, self.terminal),
            PUBLISH => {
                let Some(reason) = decode_reason(r.reason) else {
                    return WireResponse {
                        id: r.id,
                        ..invalid()
                    };
                };
                let n = RemoteTerminalNotice {
                    child: self.target,
                    reason,
                    failure_count: r.failure_count,
                    owner_epoch: r.owner_epoch,
                    restart_of: RemoteRef {
                        node: NodeId(r.restart_node),
                        entity: Ref64::from_u64(r.restart_entity),
                    },
                    restart_attempt: r.restart_attempt,
                };
                match self.terminal {
                    None => {
                        self.terminal = Some(n);
                        self.applied_publications += 1;
                        response(r.id, Some(n))
                    }
                    Some(existing) if existing == n => response(r.id, Some(n)),
                    Some(_) => WireResponse {
                        id: r.id,
                        status: Status::AlreadyPublished,
                        ..invalid()
                    },
                }
            }
            _ => unreachable!(),
        };
        if r.opcode == PUBLISH {
            self.ledger.insert(r.id, response);
        }
        response
    }
}
fn response(id: RequestId, n: Option<RemoteTerminalNotice>) -> WireResponse {
    match n {
        None => WireResponse {
            id,
            status: Status::Running,
            ..invalid()
        },
        Some(n) => WireResponse {
            id,
            status: Status::Terminal,
            reason: n.reason as u8,
            failure_count: n.failure_count,
            owner_epoch: n.owner_epoch,
            restart_node: n.restart_of.node.0,
            restart_entity: n.restart_of.entity.to_u64(),
            restart_attempt: n.restart_attempt,
        },
    }
}
fn invalid() -> WireResponse {
    WireResponse {
        id: RequestId([0; 32]),
        status: Status::InvalidRequest,
        reason: 0,
        failure_count: 0,
        owner_epoch: 0,
        restart_node: 0,
        restart_entity: 0,
        restart_attempt: 0,
    }
}
fn decode_reason(v: u8) -> Option<ExitReason> {
    Some(match v {
        1 => ExitReason::Completed,
        2 => ExitReason::Failed,
        3 => ExitReason::Cancelled,
        4 => ExitReason::NodeLost,
        _ => return None,
    })
}

pub struct RemoteSupervisionServer;
impl RemoteSupervisionServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteSupervisionService>>,
        requests: usize,
    ) -> std::io::Result<()> {
        let mut served = 0;
        while served < requests {
            let (mut stream, _) = listener.accept()?;
            while served < requests {
                let frame = match read_frame(&mut stream) {
                    Ok(f) => f,
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
                    .map_err(|_| std::io::Error::other("remote supervision service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &out)?;
                served += 1;
            }
        }
        Ok(())
    }
}

pub struct RemoteSupervisionClient {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
    target: RemoteRef,
}
impl RemoteSupervisionClient {
    pub fn new(endpoint: SocketAddr, grant: RemoteGrant, epoch: u32) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_secs(5),
            target: grant.target,
            grant,
            epoch,
        }
    }
    pub fn set_epoch(&mut self, e: u32) {
        self.epoch = e
    }
    pub fn set_timeout(&mut self, d: Duration) {
        self.timeout = d
    }
    pub fn poll(&self) -> Result<RemoteSupervisionState, RemoteSupervisionError> {
        self.round_trip(WireRequest::poll(self.epoch, self.grant))
    }
    pub fn publish(
        &self,
        n: RemoteTerminalNotice,
    ) -> Result<RemoteSupervisionState, RemoteSupervisionError> {
        if n.child != self.target {
            return Err(RemoteSupervisionError::InvalidRequest);
        }
        self.round_trip(WireRequest::publish(self.epoch, self.grant, n))
    }
    fn round_trip(&self, r: WireRequest) -> Result<RemoteSupervisionState, RemoteSupervisionError> {
        let mut s = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| RemoteSupervisionError::NodeUnavailable)?;
        s.set_read_timeout(Some(self.timeout))
            .map_err(|_| RemoteSupervisionError::NodeUnavailable)?;
        s.set_write_timeout(Some(self.timeout))
            .map_err(|_| RemoteSupervisionError::NodeUnavailable)?;
        write_frame(&mut s, &r.encode()).map_err(|_| RemoteSupervisionError::NodeLost)?;
        let b = read_frame(&mut s).map_err(|_| RemoteSupervisionError::NodeLost)?;
        let x = WireResponse::decode(&b).ok_or(RemoteSupervisionError::ProtocolError)?;
        if x.id != r.id {
            return Err(RemoteSupervisionError::ProtocolError);
        }
        match x.status {
            Status::Running => Ok(RemoteSupervisionState::Running),
            Status::Terminal => {
                let reason =
                    decode_reason(x.reason).ok_or(RemoteSupervisionError::ProtocolError)?;
                Ok(RemoteSupervisionState::Terminal(RemoteTerminalNotice {
                    child: self.target,
                    reason,
                    failure_count: x.failure_count,
                    owner_epoch: x.owner_epoch,
                    restart_of: RemoteRef {
                        node: NodeId(x.restart_node),
                        entity: Ref64::from_u64(x.restart_entity),
                    },
                    restart_attempt: x.restart_attempt,
                }))
            }
            Status::AuthorityDenied => Err(RemoteSupervisionError::AuthorityDenied),
            Status::AlreadyPublished => Err(RemoteSupervisionError::AlreadyPublished),
            Status::InvalidRequest => Err(RemoteSupervisionError::InvalidRequest),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteSupervisionBridgeError {
    Kernel(crate::kernel::RuntimeError),
    Remote(RemoteSupervisionError),
}
/// Epoch-boundary-only observer. Notify and Escalate are supported. Distributed
/// Restart intentionally remains owner-orchestrated: the supervisor cannot
/// synthesize canonical child identity or state.
pub struct RemoteSupervisionBridge {
    supervisor: Ref64,
    policy: SupervisionPolicy,
    client: RemoteSupervisionClient,
    boundary: Option<(u32, Result<RemoteSupervisionState, RemoteSupervisionError>)>,
    contacted: bool,
    applied: bool,
    /// Unconsumed immutable event receipt. It is bridge-owned specifically so
    /// a foreign child reference never enters a local ABI descriptor queue.
    receipt: Option<RemoteTerminalNotice>,
}
impl RemoteSupervisionBridge {
    pub fn new(
        supervisor: Ref64,
        policy: SupervisionPolicy,
        client: RemoteSupervisionClient,
    ) -> Result<Self, RemoteSupervisionError> {
        if policy == SupervisionPolicy::Restart {
            return Err(RemoteSupervisionError::UnsupportedPolicy);
        }
        Ok(Self {
            supervisor,
            policy,
            client,
            boundary: None,
            contacted: false,
            applied: false,
            receipt: None,
        })
    }
    fn observe(&mut self, epoch: u32) -> Result<RemoteSupervisionState, RemoteSupervisionError> {
        // A terminal notice is immutable. Once its scheduling consequence was
        // applied, retain the event receipt rather than requiring the owner to
        // remain online; this is not a shadow process descriptor.
        if self.applied {
            if let Some((_, Ok(RemoteSupervisionState::Terminal(notice)))) = self.boundary {
                return Ok(RemoteSupervisionState::Terminal(notice));
            }
        }
        if let Some((e, r)) = self.boundary {
            if e == epoch {
                return r;
            }
        }
        self.client.set_epoch(epoch);
        let mut r = self.client.poll();
        if r.is_ok() {
            self.contacted = true
        } else if r == Err(RemoteSupervisionError::NodeUnavailable) && self.contacted {
            r = Err(RemoteSupervisionError::NodeLost)
        }
        self.boundary = Some((epoch, r));
        r
    }
    pub fn sync_epoch_boundary(
        &mut self,
        kernel: &mut crate::kernel::Kernel,
    ) -> Result<RemoteSupervisionState, RemoteSupervisionBridgeError> {
        let state = self
            .observe(kernel.current_epoch())
            .map_err(RemoteSupervisionBridgeError::Remote)?;
        if let RemoteSupervisionState::Terminal(n) = state {
            if !self.applied {
                kernel
                    .wake_remote_supervision(self.supervisor, n.child.entity, n.reason, self.policy)
                    .map_err(RemoteSupervisionBridgeError::Kernel)?;
                self.receipt = Some(n);
                self.applied = true;
            }
        }
        Ok(state)
    }
    pub fn applied(&self) -> bool {
        self.applied
    }

    /// Consume the immutable remote terminal receipt. Unlike
    /// `Kernel::receive_supervision`, this carries a node-qualified child and
    /// is therefore never confused with a local ABI process reference.
    pub fn receive_terminal(&mut self) -> Option<RemoteTerminalNotice> {
        self.receipt.take()
    }

    pub fn has_terminal_receipt(&self) -> bool {
        self.receipt.is_some()
    }
}

fn write_frame(s: &mut TcpStream, b: &[u8]) -> std::io::Result<()> {
    s.write_all(&(b.len() as u64).to_le_bytes())?;
    s.write_all(b)?;
    s.flush()
}
fn read_frame(s: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut l = [0; 8];
    s.read_exact(&mut l)?;
    let n: usize = u64::from_le_bytes(l)
        .try_into()
        .map_err(|_| std::io::Error::other("oversized frame"))?;
    if n > 1024 * 1024 {
        return Err(std::io::Error::other("oversized frame"));
    }
    let mut b = vec![0; n];
    s.read_exact(&mut b)?;
    Ok(b)
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
        let e = self.offset.checked_add(n)?;
        let x = self.bytes.get(self.offset..e)?;
        self.offset = e;
        Some(x)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(*self.take(1)?.first()?)
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
