//! Authoritative bounded channels whose queue is owned by another node.
//!
//! Requests are content addressed. Successful queue mutations are recorded and
//! replayed exactly once; live authority is nevertheless checked before the
//! replay ledger, so revocation takes effect immediately.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::{Ref64, Rights};

const MAGIC: u32 = 0x5343_484e;
const VERSION: u16 = 1;
const SEND: u16 = 1;
const RECEIVE: u16 = 2;
const CLOSE: u16 = 3;
const PROBE_SEND: u16 = 4;
const PROBE_RECEIVE: u16 = 5;
const FRAME_LEN: usize = 4 + 2 + 2 + 32 + 4 + RemoteGrant::ENCODED_LEN + 8 + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RemoteChannelRequestId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteChannelEntry {
    pub value: Ref64,
    pub sender_sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteChannelState {
    pub capacity: usize,
    pub len: usize,
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteSendOutcome {
    Sent { sender_sequence: u64, len: usize },
    Full,
    Closed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteReceiveOutcome {
    Received(RemoteChannelEntry),
    Empty,
    Closed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteCloseOutcome {
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteChannelError {
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    AuthorityDenied,
    InvalidSequence,
    InvalidRequest,
}

#[derive(Clone, Copy)]
struct WireRequest {
    id: RemoteChannelRequestId,
    opcode: u16,
    epoch: u32,
    grant: RemoteGrant,
    operation_sequence: u64,
    value: Ref64,
}
impl WireRequest {
    fn new(
        opcode: u16,
        epoch: u32,
        grant: RemoteGrant,
        operation_sequence: u64,
        value: Ref64,
    ) -> Self {
        let mut request = Self {
            id: RemoteChannelRequestId([0; 32]),
            opcode,
            epoch,
            grant,
            operation_sequence,
            value,
        };
        request.id = RemoteChannelRequestId(Sha256::digest(request.identity()).into());
        request
    }
    fn identity(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FRAME_LEN - 40);
        out.extend_from_slice(&self.opcode.to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.grant.encode());
        out.extend_from_slice(&self.operation_sequence.to_le_bytes());
        out.extend_from_slice(&self.value.to_u64().to_le_bytes());
        out
    }
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FRAME_LEN);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.opcode.to_le_bytes());
        out.extend_from_slice(&self.id.0);
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.grant.encode());
        out.extend_from_slice(&self.operation_sequence.to_le_bytes());
        out.extend_from_slice(&self.value.to_u64().to_le_bytes());
        out
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != FRAME_LEN {
            return None;
        }
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let request = Self {
            opcode: c.u16()?,
            id: RemoteChannelRequestId(c.array()?),
            epoch: c.u32()?,
            grant: RemoteGrant::decode(c.take(RemoteGrant::ENCODED_LEN)?)?,
            operation_sequence: c.u64()?,
            value: Ref64::from_u64(c.u64()?),
        };
        (RemoteChannelRequestId(Sha256::digest(request.identity()).into()) == request.id)
            .then_some(request)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
enum Status {
    Sent = 0,
    Received = 1,
    Full = 2,
    Empty = 3,
    Closed = 4,
    Ready = 5,
    AuthorityDenied = 6,
    InvalidSequence = 7,
    InvalidRequest = 8,
}
impl Status {
    fn decode(v: u16) -> Option<Self> {
        Some(match v {
            0 => Self::Sent,
            1 => Self::Received,
            2 => Self::Full,
            3 => Self::Empty,
            4 => Self::Closed,
            5 => Self::Ready,
            6 => Self::AuthorityDenied,
            7 => Self::InvalidSequence,
            8 => Self::InvalidRequest,
            _ => return None,
        })
    }
}
#[derive(Clone, Copy)]
struct WireResponse {
    id: RemoteChannelRequestId,
    status: Status,
    value: Ref64,
    sequence: u64,
    len: u64,
}
impl WireResponse {
    fn encode(self) -> Vec<u8> {
        let mut o = Vec::with_capacity(60);
        o.extend_from_slice(&MAGIC.to_le_bytes());
        o.extend_from_slice(&VERSION.to_le_bytes());
        o.extend_from_slice(&(self.status as u16).to_le_bytes());
        o.extend_from_slice(&self.id.0);
        o.extend_from_slice(&self.value.to_u64().to_le_bytes());
        o.extend_from_slice(&self.sequence.to_le_bytes());
        o.extend_from_slice(&self.len.to_le_bytes());
        o
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        if c.u32()? != MAGIC || c.u16()? != VERSION {
            return None;
        }
        let out = Self {
            status: Status::decode(c.u16()?)?,
            id: RemoteChannelRequestId(c.array()?),
            value: Ref64::from_u64(c.u64()?),
            sequence: c.u64()?,
            len: c.u64()?,
        };
        c.is_empty().then_some(out)
    }
}
fn response(
    id: RemoteChannelRequestId,
    status: Status,
    value: Ref64,
    sequence: u64,
    len: usize,
) -> WireResponse {
    WireResponse {
        id,
        status,
        value,
        sequence,
        len: len as u64,
    }
}
fn invalid() -> WireResponse {
    response(
        RemoteChannelRequestId([0; 32]),
        Status::InvalidRequest,
        Ref64::NULL,
        0,
        0,
    )
}

/// Canonical FIFO and closed bit for one remotely-owned channel.
pub struct RemoteChannelService {
    node: NodeId,
    target: RemoteRef,
    object_version: u32,
    capacity: usize,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    queue: VecDeque<RemoteChannelEntry>,
    closed: bool,
    next_send_sequence: HashMap<Ref64, u64>,
    next_receive_sequence: HashMap<Ref64, u64>,
    ledger: HashMap<RemoteChannelRequestId, WireResponse>,
    applied_sends: u64,
    applied_receives: u64,
    applied_closes: u64,
}
impl RemoteChannelService {
    pub fn new(
        node: NodeId,
        target: RemoteRef,
        object_version: u32,
        capacity: usize,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Self {
        assert_eq!(
            target.node, node,
            "remote channel must be owned by the serving node"
        );
        assert!(capacity > 0, "remote channel capacity must be nonzero");
        Self {
            node,
            target,
            object_version,
            capacity,
            authority,
            queue: VecDeque::new(),
            closed: false,
            next_send_sequence: HashMap::new(),
            next_receive_sequence: HashMap::new(),
            ledger: HashMap::new(),
            applied_sends: 0,
            applied_receives: 0,
            applied_closes: 0,
        }
    }
    pub fn state(&self) -> RemoteChannelState {
        RemoteChannelState {
            capacity: self.capacity,
            len: self.queue.len(),
            closed: self.closed,
        }
    }
    pub fn entries(&self) -> Vec<RemoteChannelEntry> {
        self.queue.iter().copied().collect()
    }
    pub fn applied_sends(&self) -> u64 {
        self.applied_sends
    }
    pub fn applied_receives(&self) -> u64 {
        self.applied_receives
    }
    pub fn applied_closes(&self) -> u64 {
        self.applied_closes
    }
    fn handle(&mut self, frame: &[u8]) -> WireResponse {
        let Some(req) = WireRequest::decode(frame) else {
            return invalid();
        };
        let required = match req.opcode {
            SEND | PROBE_SEND => Rights::SEND,
            RECEIVE | PROBE_RECEIVE => Rights::RECEIVE,
            CLOSE => Rights::DESTROY,
            _ => {
                return WireResponse {
                    id: req.id,
                    ..invalid()
                }
            }
        };
        let authorized = self.authority.lock().ok().is_some_and(|a| {
            a.authorize(
                &req.grant,
                self.node,
                self.target,
                required,
                self.object_version,
                req.epoch,
            )
            .is_ok()
        });
        if !authorized {
            return response(
                req.id,
                Status::AuthorityDenied,
                Ref64::NULL,
                0,
                self.queue.len(),
            );
        }
        // Deliberately after live authorization: a revoked grant cannot replay.
        if matches!(req.opcode, SEND | RECEIVE | CLOSE) {
            if let Some(r) = self.ledger.get(&req.id) {
                return *r;
            }
        }
        let mut mutated = false;
        let out = match req.opcode {
            SEND => {
                if self.closed {
                    response(
                        req.id,
                        Status::Closed,
                        Ref64::NULL,
                        req.operation_sequence,
                        self.queue.len(),
                    )
                } else if self.queue.len() >= self.capacity {
                    response(
                        req.id,
                        Status::Full,
                        Ref64::NULL,
                        req.operation_sequence,
                        self.queue.len(),
                    )
                } else {
                    let next = self.next_send_sequence.entry(req.grant.actor).or_insert(0);
                    if req.operation_sequence != *next {
                        response(
                            req.id,
                            Status::InvalidSequence,
                            Ref64::NULL,
                            *next,
                            self.queue.len(),
                        )
                    } else {
                        self.queue.push_back(RemoteChannelEntry {
                            value: req.value,
                            sender_sequence: req.operation_sequence,
                        });
                        *next = next.wrapping_add(1);
                        self.applied_sends += 1;
                        mutated = true;
                        response(
                            req.id,
                            Status::Sent,
                            Ref64::NULL,
                            req.operation_sequence,
                            self.queue.len(),
                        )
                    }
                }
            }
            RECEIVE => {
                let next = *self
                    .next_receive_sequence
                    .entry(req.grant.actor)
                    .or_insert(0);
                if req.operation_sequence != next {
                    response(
                        req.id,
                        Status::InvalidSequence,
                        Ref64::NULL,
                        next,
                        self.queue.len(),
                    )
                } else if let Some(entry) = self.queue.pop_front() {
                    *self
                        .next_receive_sequence
                        .get_mut(&req.grant.actor)
                        .unwrap() = next.wrapping_add(1);
                    self.applied_receives += 1;
                    mutated = true;
                    response(
                        req.id,
                        Status::Received,
                        entry.value,
                        entry.sender_sequence,
                        self.queue.len(),
                    )
                } else if self.closed {
                    response(req.id, Status::Closed, Ref64::NULL, 0, 0)
                } else {
                    response(req.id, Status::Empty, Ref64::NULL, 0, 0)
                }
            }
            CLOSE => {
                if !self.closed {
                    self.closed = true;
                    self.applied_closes += 1;
                    mutated = true
                }
                response(req.id, Status::Closed, Ref64::NULL, 0, self.queue.len())
            }
            PROBE_SEND => {
                if self.closed {
                    response(req.id, Status::Closed, Ref64::NULL, 0, self.queue.len())
                } else if self.queue.len() >= self.capacity {
                    response(req.id, Status::Full, Ref64::NULL, 0, self.queue.len())
                } else {
                    response(req.id, Status::Ready, Ref64::NULL, 0, self.queue.len())
                }
            }
            PROBE_RECEIVE => {
                if !self.queue.is_empty() {
                    response(req.id, Status::Ready, Ref64::NULL, 0, self.queue.len())
                } else if self.closed {
                    response(req.id, Status::Closed, Ref64::NULL, 0, 0)
                } else {
                    response(req.id, Status::Empty, Ref64::NULL, 0, 0)
                }
            }
            _ => unreachable!(),
        };
        if mutated {
            self.ledger.insert(req.id, out);
        }
        out
    }
}

pub struct RemoteChannelServer;
impl RemoteChannelServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteChannelService>>,
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
                    .map_err(|_| std::io::Error::other("remote channel service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &out)?;
                served += 1
            }
        }
        Ok(())
    }

    /// Serve until the owning node runtime requests shutdown, without relying
    /// on an exact request budget.
    pub fn serve_until(
        listener: TcpListener,
        service: Arc<Mutex<RemoteChannelService>>,
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
                let response = service
                    .lock()
                    .map_err(|_| std::io::Error::other("remote channel service poisoned"))?
                    .handle(&frame)
                    .encode();
                write_frame(&mut stream, &response)?;
            }
        }
        Ok(())
    }
}

pub struct RemoteChannelClient {
    endpoint: SocketAddr,
    timeout: Duration,
    grant: RemoteGrant,
    epoch: u32,
}
impl RemoteChannelClient {
    pub fn new(endpoint: SocketAddr, grant: RemoteGrant, epoch: u32) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_secs(5),
            grant,
            epoch,
        }
    }
    /// Rebind the transport proxy to the exact grant carried by a multiplexed
    /// lane effect; endpoint and timeout are transport configuration only.
    pub fn rebound(&self, grant: RemoteGrant, epoch: u32) -> Self {
        Self {
            endpoint: self.endpoint,
            timeout: self.timeout,
            grant,
            epoch,
        }
    }
    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch
    }
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout
    }
    pub fn target(&self) -> RemoteRef {
        self.grant.target
    }

    pub fn send(
        &self,
        sender_sequence: u64,
        value: Ref64,
    ) -> Result<RemoteSendOutcome, RemoteChannelError> {
        let r = self.round_trip(WireRequest::new(
            SEND,
            self.epoch,
            self.grant,
            sender_sequence,
            value,
        ))?;
        match r.status {
            Status::Sent => Ok(RemoteSendOutcome::Sent {
                sender_sequence: r.sequence,
                len: r.len as usize,
            }),
            Status::Full => Ok(RemoteSendOutcome::Full),
            Status::Closed => Ok(RemoteSendOutcome::Closed),
            _ => Err(map_status(r.status)),
        }
    }
    pub fn receive(
        &self,
        receive_sequence: u64,
    ) -> Result<RemoteReceiveOutcome, RemoteChannelError> {
        let r = self.round_trip(WireRequest::new(
            RECEIVE,
            self.epoch,
            self.grant,
            receive_sequence,
            Ref64::NULL,
        ))?;
        match r.status {
            Status::Received => Ok(RemoteReceiveOutcome::Received(RemoteChannelEntry {
                value: r.value,
                sender_sequence: r.sequence,
            })),
            Status::Empty => Ok(RemoteReceiveOutcome::Empty),
            Status::Closed => Ok(RemoteReceiveOutcome::Closed),
            _ => Err(map_status(r.status)),
        }
    }
    pub fn close(&self) -> Result<RemoteCloseOutcome, RemoteChannelError> {
        let r = self.round_trip(WireRequest::new(
            CLOSE,
            self.epoch,
            self.grant,
            0,
            Ref64::NULL,
        ))?;
        match r.status {
            Status::Closed => Ok(RemoteCloseOutcome::Closed),
            _ => Err(map_status(r.status)),
        }
    }
    fn probe_send(&self) -> Result<Status, RemoteChannelError> {
        self.round_trip(WireRequest::new(
            PROBE_SEND,
            self.epoch,
            self.grant,
            0,
            Ref64::NULL,
        ))
        .map(|r| r.status)
    }
    fn probe_receive(&self) -> Result<Status, RemoteChannelError> {
        self.round_trip(WireRequest::new(
            PROBE_RECEIVE,
            self.epoch,
            self.grant,
            0,
            Ref64::NULL,
        ))
        .map(|r| r.status)
    }
    fn round_trip(&self, req: WireRequest) -> Result<WireResponse, RemoteChannelError> {
        let mut s = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| RemoteChannelError::NodeUnavailable)?;
        s.set_read_timeout(Some(self.timeout))
            .map_err(|_| RemoteChannelError::NodeUnavailable)?;
        s.set_write_timeout(Some(self.timeout))
            .map_err(|_| RemoteChannelError::NodeUnavailable)?;
        write_frame(&mut s, &req.encode()).map_err(|_| RemoteChannelError::NodeLost)?;
        let b = read_frame(&mut s).map_err(|_| RemoteChannelError::NodeLost)?;
        let r = WireResponse::decode(&b).ok_or(RemoteChannelError::ProtocolError)?;
        if r.id != req.id {
            return Err(RemoteChannelError::ProtocolError);
        };
        if matches!(
            r.status,
            Status::AuthorityDenied | Status::InvalidSequence | Status::InvalidRequest
        ) {
            return Err(map_status(r.status));
        }
        Ok(r)
    }
}
fn map_status(s: Status) -> RemoteChannelError {
    match s {
        Status::AuthorityDenied => RemoteChannelError::AuthorityDenied,
        Status::InvalidSequence => RemoteChannelError::InvalidSequence,
        Status::InvalidRequest => RemoteChannelError::InvalidRequest,
        _ => RemoteChannelError::ProtocolError,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteChannelWaitKind {
    Send,
    Receive,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteChannelBridgeError {
    Kernel(crate::kernel::RuntimeError),
    Remote(RemoteChannelError),
}
/// Epoch-boundary-only bridge. It stores local scheduling dependencies, never
/// a shadow queue, and probes with the operation-specific grant.
pub struct RemoteChannelBridge {
    target: RemoteRef,
    send_client: RemoteChannelClient,
    receive_client: RemoteChannelClient,
    send_waiters: Vec<Ref64>,
    receive_waiters: Vec<Ref64>,
    last_epoch: Option<u32>,
    contacted_owner: bool,
}
impl RemoteChannelBridge {
    pub fn new(
        target: RemoteRef,
        send_client: RemoteChannelClient,
        receive_client: RemoteChannelClient,
    ) -> Self {
        Self {
            target,
            send_client,
            receive_client,
            send_waiters: Vec::new(),
            receive_waiters: Vec::new(),
            last_epoch: None,
            contacted_owner: false,
        }
    }
    pub fn register(
        &mut self,
        kernel: &mut crate::kernel::Kernel,
        kind: RemoteChannelWaitKind,
        continuation: Ref64,
        next_run_class: u32,
    ) -> Result<(), RemoteChannelBridgeError> {
        let waiters = match kind {
            RemoteChannelWaitKind::Send => &mut self.send_waiters,
            RemoteChannelWaitKind::Receive => &mut self.receive_waiters,
        };
        if waiters.contains(&continuation) {
            return Ok(());
        }
        kernel
            .register_remote_channel_waiter(
                continuation,
                self.target.node.0,
                self.target.entity,
                next_run_class,
            )
            .map_err(RemoteChannelBridgeError::Kernel)?;
        waiters.push(continuation);
        Ok(())
    }
    pub fn sync_epoch_boundary(
        &mut self,
        kernel: &mut crate::kernel::Kernel,
    ) -> Result<(), RemoteChannelBridgeError> {
        let epoch = kernel.current_epoch();
        if self.last_epoch == Some(epoch) {
            return Ok(());
        }
        self.send_client.set_epoch(epoch);
        self.receive_client.set_epoch(epoch);
        let send = self.send_client.probe_send().map_err(|error| {
            RemoteChannelBridgeError::Remote(
                if self.contacted_owner && error == RemoteChannelError::NodeUnavailable {
                    RemoteChannelError::NodeLost
                } else {
                    error
                },
            )
        })?;
        self.contacted_owner = true;
        let receive = self.receive_client.probe_receive().map_err(|error| {
            RemoteChannelBridgeError::Remote(
                if self.contacted_owner && error == RemoteChannelError::NodeUnavailable {
                    RemoteChannelError::NodeLost
                } else {
                    error
                },
            )
        })?;
        if matches!(send, Status::Ready | Status::Closed) {
            for w in self.send_waiters.drain(..) {
                kernel.wake_remote_channel_waiter(w, self.target.node.0, self.target.entity)
            }
        }
        if matches!(receive, Status::Ready | Status::Closed) {
            for w in self.receive_waiters.drain(..) {
                kernel.wake_remote_channel_waiter(w, self.target.node.0, self.target.entity)
            }
        }
        self.last_epoch = Some(epoch);
        Ok(())
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
        let o = self.bytes.get(self.offset..e)?;
        self.offset = e;
        Some(o)
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
