//! Signed TCP ingress for a mailbox owned by a real local `Kernel`.
//!
//! The socket thread authenticates and stages immutable requests.  It never
//! owns a mailbox and never mutates the kernel.  At an epoch boundary the
//! `RemoteNodeRuntime` drains the stage on its owner thread and calls the one
//! canonical kernel enqueue path.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::{Ref64, Rights};
use crate::kernel::{Kernel, RuntimeError};

const MAGIC: u32 = 0x534d_4249;
const VERSION: u16 = 1;
const SEND: u16 = 1;
pub const MAX_REMOTE_MAILBOX_FRAME: usize = 1024 * 1024;
const RESPONSE_LEN: usize = 4 + 2 + 2 + 32 + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RemoteMailboxRequestId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteMailboxSendOutcome {
    Staged(RemoteMailboxRequestId),
    Duplicate(RemoteMailboxRequestId),
    Applied(RemoteMailboxRequestId),
    Backpressured(RemoteMailboxRequestId),
    ProcessUnavailable(RemoteMailboxRequestId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteMailboxApplyStatus {
    Applied,
    Backpressured,
    ProcessUnavailable,
    AuthorityDenied,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteMailboxApplyOutcome {
    pub request_id: RemoteMailboxRequestId,
    pub actor_node: NodeId,
    pub actor: Ref64,
    pub target: RemoteRef,
    pub sender_sequence: u64,
    pub status: RemoteMailboxApplyStatus,
}

/// Receiver-visible, immutable identity/capability envelope. Foreign actor and
/// capability references remain node-qualified bytes and are never installed
/// as local kernel descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteMailboxReceipt {
    pub actor_node: NodeId,
    pub actor: Ref64,
    pub sender_sequence: u64,
    pub capability: Option<RemoteGrant>,
}

impl RemoteMailboxReceipt {
    pub fn encode(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(25 + RemoteGrant::ENCODED_LEN);
        out.extend_from_slice(&self.actor_node.0.to_le_bytes());
        out.extend_from_slice(&self.actor.to_u64().to_le_bytes());
        out.extend_from_slice(&self.sender_sequence.to_le_bytes());
        out.push(u8::from(self.capability.is_some()));
        if let Some(capability) = self.capability {
            out.extend_from_slice(&capability.encode());
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 25 && bytes.len() != 25 + RemoteGrant::ENCODED_LEN {
            return None;
        }
        let actor_node = NodeId(u64::from_le_bytes(bytes[0..8].try_into().ok()?));
        let actor = Ref64::from_u64(u64::from_le_bytes(bytes[8..16].try_into().ok()?));
        let sender_sequence = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        let capability = match bytes[24] {
            0 if bytes.len() == 25 => None,
            1 if bytes.len() == 25 + RemoteGrant::ENCODED_LEN => {
                Some(RemoteGrant::decode(&bytes[25..])?)
            }
            _ => return None,
        };
        Some(Self {
            actor_node,
            actor,
            sender_sequence,
            capability,
        })
    }
}

/// The single immutable local message payload. Its transferred capability
/// names this same object, while all foreign identity/authority remains opaque,
/// node-qualified receipt bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteMailboxEnvelope {
    pub receipt: RemoteMailboxReceipt,
    pub value: Vec<u8>,
}

impl RemoteMailboxEnvelope {
    pub fn encode(&self) -> Vec<u8> {
        let receipt = self.receipt.encode();
        let mut out = Vec::with_capacity(8 + receipt.len() + self.value.len());
        out.extend_from_slice(&(receipt.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.value.len() as u32).to_le_bytes());
        out.extend_from_slice(&receipt);
        out.extend_from_slice(&self.value);
        out
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let receipt_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let value_len = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let receipt_end = 8usize.checked_add(receipt_len)?;
        let value_end = receipt_end.checked_add(value_len)?;
        if value_end != bytes.len() {
            return None;
        }
        Some(Self {
            receipt: RemoteMailboxReceipt::decode(&bytes[8..receipt_end])?,
            value: bytes[receipt_end..value_end].to_vec(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteMailboxError {
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    AuthorityDenied,
    InvalidSequence,
    InvalidRequest,
    FrameTooLarge,
}

#[derive(Clone)]
struct WireRequest {
    id: RemoteMailboxRequestId,
    epoch: u32,
    grant: RemoteGrant,
    sender_sequence: u64,
    urgent: bool,
    capability: Option<RemoteGrant>,
    value: Vec<u8>,
}

impl WireRequest {
    fn new(
        epoch: u32,
        grant: RemoteGrant,
        sender_sequence: u64,
        value: Vec<u8>,
        capability: Option<RemoteGrant>,
        urgent: bool,
    ) -> Self {
        let mut request = Self {
            id: RemoteMailboxRequestId([0; 32]),
            epoch,
            grant,
            sender_sequence,
            urgent,
            capability,
            value,
        };
        request.id = RemoteMailboxRequestId(Sha256::digest(request.identity()).into());
        request
    }

    fn identity(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + 4 + RemoteGrant::ENCODED_LEN + 14 + self.value.len());
        out.extend_from_slice(&SEND.to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.grant.encode());
        out.extend_from_slice(&self.sender_sequence.to_le_bytes());
        out.push(u8::from(self.urgent));
        out.push(u8::from(self.capability.is_some()));
        if let Some(grant) = self.capability {
            out.extend_from_slice(&grant.encode());
        }
        out.extend_from_slice(&(self.value.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.value);
        out
    }

    fn encode(&self) -> Option<Vec<u8>> {
        let identity = self.identity();
        let size = 4usize
            .checked_add(2)?
            .checked_add(32)?
            .checked_add(identity.len())?;
        if size > MAX_REMOTE_MAILBOX_FRAME || self.value.len() > u32::MAX as usize {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.id.0);
        out.extend_from_slice(&identity);
        Some(out)
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_REMOTE_MAILBOX_FRAME {
            return None;
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != VERSION {
            return None;
        }
        let id = RemoteMailboxRequestId(cursor.array()?);
        if cursor.u16()? != SEND {
            return None;
        }
        let epoch = cursor.u32()?;
        let grant = RemoteGrant::decode(cursor.take(RemoteGrant::ENCODED_LEN)?)?;
        let sender_sequence = cursor.u64()?;
        let urgent = match cursor.u8()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        let capability = match cursor.u8()? {
            0 => None,
            1 => Some(RemoteGrant::decode(cursor.take(RemoteGrant::ENCODED_LEN)?)?),
            _ => return None,
        };
        let length = cursor.u32()? as usize;
        let value = cursor.take(length)?.to_vec();
        if !cursor.is_empty() {
            return None;
        }
        let request = Self {
            id,
            epoch,
            grant,
            sender_sequence,
            urgent,
            capability,
            value,
        };
        (RemoteMailboxRequestId(Sha256::digest(request.identity()).into()) == id).then_some(request)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
enum Status {
    Staged = 0,
    Duplicate = 1,
    AuthorityDenied = 2,
    InvalidSequence = 3,
    InvalidRequest = 4,
    Applied = 5,
    Backpressured = 6,
    ProcessUnavailable = 7,
}
impl Status {
    fn decode(value: u16) -> Option<Self> {
        Some(match value {
            0 => Self::Staged,
            1 => Self::Duplicate,
            2 => Self::AuthorityDenied,
            3 => Self::InvalidSequence,
            4 => Self::InvalidRequest,
            5 => Self::Applied,
            6 => Self::Backpressured,
            7 => Self::ProcessUnavailable,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct WireResponse {
    id: RemoteMailboxRequestId,
    status: Status,
    next_sequence: u64,
}
impl WireResponse {
    fn encode(self) -> [u8; RESPONSE_LEN] {
        let mut out = [0; RESPONSE_LEN];
        out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&VERSION.to_le_bytes());
        out[6..8].copy_from_slice(&(self.status as u16).to_le_bytes());
        out[8..40].copy_from_slice(&self.id.0);
        out[40..48].copy_from_slice(&self.next_sequence.to_le_bytes());
        out
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != RESPONSE_LEN {
            return None;
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.u32()? != MAGIC || cursor.u16()? != VERSION {
            return None;
        }
        Some(Self {
            status: Status::decode(cursor.u16()?)?,
            id: RemoteMailboxRequestId(cursor.array()?),
            next_sequence: cursor.u64()?,
        })
    }
}

/// Network-side validation and staging state. It contains no mailbox.
pub struct RemoteMailboxIngress {
    node: NodeId,
    target: RemoteRef,
    object_version: u32,
    authorities: Vec<Arc<Mutex<RemoteAuthorityStore>>>,
    next_sequence: HashMap<(NodeId, Ref64), u64>,
    staged_ids: HashSet<RemoteMailboxRequestId>,
    staged_status: HashMap<RemoteMailboxRequestId, RemoteMailboxApplyStatus>,
    pending: VecDeque<WireRequest>,
    ledger: HashMap<RemoteMailboxRequestId, RemoteMailboxApplyStatus>,
    applied: u64,
}

impl RemoteMailboxIngress {
    pub fn new(
        node: NodeId,
        target: RemoteRef,
        object_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Self {
        assert_eq!(
            target.node, node,
            "mailbox ingress must be owned by its serving node"
        );
        Self {
            node,
            target,
            object_version,
            authorities: vec![authority],
            next_sequence: HashMap::new(),
            staged_ids: HashSet::new(),
            staged_status: HashMap::new(),
            pending: VecDeque::new(),
            ledger: HashMap::new(),
            applied: 0,
        }
    }

    /// Add another trusted issuer. Each issuer retains an independent actor
    /// sequence namespace.
    pub fn trust_authority(&mut self, authority: Arc<Mutex<RemoteAuthorityStore>>) {
        self.authorities.push(authority);
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
    pub fn applied_count(&self) -> u64 {
        self.applied
    }
    pub fn outcome(&self, id: RemoteMailboxRequestId) -> Option<RemoteMailboxApplyStatus> {
        self.ledger.get(&id).copied()
    }

    fn authorize(&self, request: &WireRequest, epoch: u32) -> bool {
        let send_authorized = self.authorities.iter().any(|store| {
            store.lock().ok().is_some_and(|authority| {
                authority
                    .authorize(
                        &request.grant,
                        self.node,
                        self.target,
                        Rights::SEND,
                        self.object_version,
                        epoch,
                    )
                    .is_ok()
            })
        });
        if !send_authorized {
            return false;
        }
        let Some(capability) = request.capability else {
            return true;
        };
        capability.actor == request.grant.actor
            && capability.audience == self.node
            && self.authorities.iter().any(|store| {
                store.lock().ok().is_some_and(|authority| {
                    authority
                        .authorize(
                            &capability,
                            self.node,
                            capability.target,
                            Rights::TRANSFER,
                            capability.object_version,
                            epoch,
                        )
                        .is_ok()
                })
            })
    }

    fn handle(&mut self, frame: &[u8]) -> WireResponse {
        let Some(request) = WireRequest::decode(frame) else {
            return WireResponse {
                id: RemoteMailboxRequestId([0; 32]),
                status: Status::InvalidRequest,
                next_sequence: 0,
            };
        };
        let actor_key = (request.grant.issuer, request.grant.actor);
        let next = *self.next_sequence.get(&actor_key).unwrap_or(&0);
        // Live authorization deliberately precedes both staged and applied replay.
        if !self.authorize(&request, request.epoch) {
            return WireResponse {
                id: request.id,
                status: Status::AuthorityDenied,
                next_sequence: next,
            };
        }
        if let Some(status) = self.ledger.get(&request.id) {
            let status = match status {
                RemoteMailboxApplyStatus::Applied => Status::Applied,
                RemoteMailboxApplyStatus::ProcessUnavailable => Status::ProcessUnavailable,
                RemoteMailboxApplyStatus::Backpressured => Status::Backpressured,
                RemoteMailboxApplyStatus::AuthorityDenied => Status::AuthorityDenied,
            };
            return WireResponse {
                id: request.id,
                status,
                next_sequence: next,
            };
        }
        if self.staged_ids.contains(&request.id) {
            let status = if self.staged_status.get(&request.id)
                == Some(&RemoteMailboxApplyStatus::Backpressured)
            {
                Status::Backpressured
            } else {
                Status::Duplicate
            };
            return WireResponse {
                id: request.id,
                status,
                next_sequence: next,
            };
        }
        if request.sender_sequence != next {
            return WireResponse {
                id: request.id,
                status: Status::InvalidSequence,
                next_sequence: next,
            };
        }
        self.next_sequence.insert(actor_key, next.wrapping_add(1));
        self.staged_ids.insert(request.id);
        self.pending.push_back(request.clone());
        WireResponse {
            id: request.id,
            status: Status::Staged,
            next_sequence: next.wrapping_add(1),
        }
    }

    /// Apply the boundary snapshot in a network-arrival-independent order.
    /// Backpressured requests remain staged, with their bytes and transfer
    /// proof held in escrow for the next boundary.
    pub(crate) fn apply_boundary(
        &mut self,
        kernel: &mut Kernel,
        epoch: u32,
    ) -> Vec<RemoteMailboxApplyOutcome> {
        let mut requests: Vec<_> = self.pending.drain(..).collect();
        requests.sort_by_key(|request| {
            (
                request.grant.issuer.0,
                request.grant.actor.to_u64(),
                request.sender_sequence,
                request.id,
            )
        });
        let mut retained = VecDeque::new();
        let mut outcomes = Vec::with_capacity(requests.len());
        for request in requests {
            let status = if !self.authorize(&request, epoch) {
                RemoteMailboxApplyStatus::AuthorityDenied
            } else {
                let envelope = RemoteMailboxEnvelope {
                    receipt: RemoteMailboxReceipt {
                        actor_node: request.grant.issuer,
                        actor: request.grant.actor,
                        sender_sequence: request.sender_sequence,
                        capability: request.capability,
                    },
                    value: request.value.clone(),
                }
                .encode();
                match kernel.ingest_remote_message(self.target.entity, envelope, request.urgent) {
                    Ok(()) => RemoteMailboxApplyStatus::Applied,
                    Err(RuntimeError::MailboxFull) => RemoteMailboxApplyStatus::Backpressured,
                    Err(RuntimeError::ProcessUnavailable) | Err(RuntimeError::Abi(_)) => {
                        RemoteMailboxApplyStatus::ProcessUnavailable
                    }
                    Err(_) => RemoteMailboxApplyStatus::ProcessUnavailable,
                }
            };
            outcomes.push(RemoteMailboxApplyOutcome {
                request_id: request.id,
                actor_node: request.grant.issuer,
                actor: request.grant.actor,
                target: self.target,
                sender_sequence: request.sender_sequence,
                status,
            });
            if status == RemoteMailboxApplyStatus::Backpressured {
                self.staged_status.insert(request.id, status);
                retained.push_back(request);
            } else {
                self.staged_ids.remove(&request.id);
                self.staged_status.remove(&request.id);
                self.ledger.insert(request.id, status);
                if status == RemoteMailboxApplyStatus::Applied {
                    self.applied += 1;
                }
            }
        }
        self.pending = retained;
        outcomes
    }
}

pub struct RemoteMailboxServer;
impl RemoteMailboxServer {
    pub fn serve_until(
        listener: TcpListener,
        ingress: Arc<Mutex<RemoteMailboxIngress>>,
        shutdown: Arc<std::sync::atomic::AtomicBool>,
    ) -> std::io::Result<()> {
        use std::sync::atomic::Ordering;
        listener.set_nonblocking(true)?;
        while !shutdown.load(Ordering::Acquire) {
            let (mut stream, _) = match listener.accept() {
                Ok(pair) => pair,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(error) => return Err(error),
            };
            stream.set_nonblocking(false)?;
            stream.set_read_timeout(Some(Duration::from_millis(20)))?;
            while !shutdown.load(Ordering::Acquire) {
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
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                    {
                        continue
                    }
                    Err(error) => return Err(error),
                };
                let response = ingress
                    .lock()
                    .map_err(|_| std::io::Error::other("remote mailbox ingress poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &response)?;
            }
        }
        Ok(())
    }
}

pub struct RemoteMailboxClient {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
    contacted_owner: std::sync::atomic::AtomicBool,
}
impl RemoteMailboxClient {
    pub fn new(endpoint: SocketAddr, grant: RemoteGrant, epoch: u32) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_secs(5),
            grant,
            epoch,
            contacted_owner: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
    pub fn target(&self) -> RemoteRef {
        self.grant.target
    }
    pub fn send(
        &self,
        sender_sequence: u64,
        value: Vec<u8>,
        capability: Option<RemoteGrant>,
        urgent: bool,
    ) -> Result<RemoteMailboxSendOutcome, RemoteMailboxError> {
        let request = WireRequest::new(
            self.epoch,
            self.grant,
            sender_sequence,
            value,
            capability,
            urgent,
        );
        let bytes = request.encode().ok_or(RemoteMailboxError::FrameTooLarge)?;
        let mut stream =
            TcpStream::connect_timeout(&self.endpoint, self.timeout).map_err(|_| {
                if self
                    .contacted_owner
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    RemoteMailboxError::NodeLost
                } else {
                    RemoteMailboxError::NodeUnavailable
                }
            })?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|_| RemoteMailboxError::NodeUnavailable)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(|_| RemoteMailboxError::NodeUnavailable)?;
        write_frame(&mut stream, &bytes).map_err(|_| RemoteMailboxError::NodeLost)?;
        let response = WireResponse::decode(
            &read_frame(&mut stream).map_err(|_| RemoteMailboxError::NodeLost)?,
        )
        .ok_or(RemoteMailboxError::ProtocolError)?;
        self.contacted_owner
            .store(true, std::sync::atomic::Ordering::Release);
        if response.id != request.id {
            return Err(RemoteMailboxError::ProtocolError);
        }
        match response.status {
            Status::Staged => Ok(RemoteMailboxSendOutcome::Staged(request.id)),
            Status::Duplicate => Ok(RemoteMailboxSendOutcome::Duplicate(request.id)),
            Status::Applied => Ok(RemoteMailboxSendOutcome::Applied(request.id)),
            Status::Backpressured => Ok(RemoteMailboxSendOutcome::Backpressured(request.id)),
            Status::ProcessUnavailable => {
                Ok(RemoteMailboxSendOutcome::ProcessUnavailable(request.id))
            }
            Status::AuthorityDenied => Err(RemoteMailboxError::AuthorityDenied),
            Status::InvalidSequence => Err(RemoteMailboxError::InvalidSequence),
            Status::InvalidRequest => Err(RemoteMailboxError::InvalidRequest),
        }
    }
}

pub type RemoteMailboxIngressClient = RemoteMailboxClient;
pub type RemoteMailboxIngressService = RemoteMailboxIngress;

fn write_frame(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}
fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut length = [0; 8];
    stream.read_exact(&mut length)?;
    let length: usize = u64::from_le_bytes(length)
        .try_into()
        .map_err(|_| std::io::Error::other("oversized frame"))?;
    if length > MAX_REMOTE_MAILBOX_FRAME {
        return Err(std::io::Error::other("oversized frame"));
    }
    let mut bytes = vec![0; length];
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
    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(length)?;
        let out = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(out)
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
