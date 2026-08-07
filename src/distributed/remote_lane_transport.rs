//! Authenticated, bounded TCP transport for the narrow remote-lane journal.
//!
//! This transports the existing validated special-dispatch protocol.  It does
//! not make that protocol a general `LaneView` implementation. The configured
//! symmetric session key authenticates the two configured peers; this is not a
//! TLS identity or a discovery/PKI mechanism.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::RemoteAuthorityError;
use super::remote_lane_effect::{
    RemoteLaneApply, RemoteLaneClientRouter, RemoteLaneEffectBatch, RemoteLaneEffectService,
    RemoteLaneError, RemoteLaneOutcome, RemoteLaneRequestId, RemoteLaneValue,
    MAX_REMOTE_LANE_EFFECTS, MAX_REMOTE_LANE_PAYLOAD,
};
use super::{NodeId, RemoteRef};
use crate::abi::Ref64;

const MAGIC: u32 = 0x534c_5452;
const VERSION: u16 = 1;
const REQUEST: u16 = 1;
const RESPONSE: u16 = 2;
pub const MAX_REMOTE_LANE_TRANSPORT_BATCHES: usize = 16;
pub const MAX_REMOTE_LANE_TRANSPORT_FRAME: usize = 2 * 1024 * 1024;
pub const REMOTE_LANE_TRANSPORT_TIMEOUT: Duration = Duration::from_secs(2);
pub const MAX_REMOTE_LANE_TRANSPORT_REPLAY_ENTRIES: usize = 4096;
pub const MAX_REMOTE_LANE_TRANSPORT_REPLAY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteLaneTransportError {
    TemporaryUnavailable,
    Timeout,
    FrameTooLarge,
    Protocol,
    Authentication,
    WrongSession,
    WrongIssuer,
    WrongOwner,
    Replay,
    Collision,
    Late,
    Capacity,
    Lane(RemoteLaneError),
}

#[derive(Clone)]
/// A session is process-incarnation scoped. Callers must provision a fresh
/// unpredictable `session_id` after either peer restarts; transport replay
/// tables are bounded in-memory state, not durable exactly-once storage.
pub struct RemoteLaneClientSession {
    pub session_id: [u8; 16],
    pub issuer: NodeId,
    pub owner: NodeId,
    session_key: [u8; 32],
}
impl RemoteLaneClientSession {
    pub fn new(session_id: [u8; 16], issuer: NodeId, owner: NodeId, session_key: [u8; 32]) -> Self {
        Self {
            session_id,
            issuer,
            owner,
            session_key,
        }
    }
}

/// In-memory replay state is scoped to this owner process. After either peer
/// restarts, provision a fresh unpredictable `session_id` (and preferably key);
/// reusing a session across restart is unsupported and provides no durable replay.
pub struct RemoteLaneOwnerSession {
    pub session_id: [u8; 16],
    pub issuer: NodeId,
    pub owner: NodeId,
    session_key: [u8; 32],
}
impl RemoteLaneOwnerSession {
    pub fn new(session_id: [u8; 16], issuer: NodeId, owner: NodeId, session_key: [u8; 32]) -> Self {
        Self {
            session_id,
            issuer,
            owner,
            session_key,
        }
    }
}

struct Request {
    nonce: u64,
    boundary: u32,
    frames: Vec<Vec<u8>>,
    digest: [u8; 32],
}

fn request_unsigned(
    session: &RemoteLaneClientSession,
    nonce: u64,
    boundary: u32,
    frames: &[Vec<u8>],
) -> Result<Vec<u8>, RemoteLaneTransportError> {
    if frames.is_empty() || frames.len() > MAX_REMOTE_LANE_TRANSPORT_BATCHES {
        return Err(RemoteLaneTransportError::FrameTooLarge);
    }
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, REQUEST);
    out.extend_from_slice(&session.session_id);
    put_u64(&mut out, session.issuer.0);
    put_u64(&mut out, session.owner.0);
    put_u64(&mut out, nonce);
    put_u32(&mut out, boundary);
    put_u16(&mut out, frames.len() as u16);
    put_u16(&mut out, 0);
    for frame in frames {
        if frame.len() > MAX_REMOTE_LANE_PAYLOAD + MAX_REMOTE_LANE_EFFECTS * 256 {
            return Err(RemoteLaneTransportError::FrameTooLarge);
        }
        put_u32(&mut out, frame.len() as u32);
        out.extend_from_slice(frame);
    }
    if out.len() + 32 > MAX_REMOTE_LANE_TRANSPORT_FRAME {
        return Err(RemoteLaneTransportError::FrameTooLarge);
    }
    Ok(out)
}

fn decode_request(
    bytes: &[u8],
    session: &RemoteLaneOwnerSession,
) -> Result<Request, RemoteLaneTransportError> {
    if bytes.len() > MAX_REMOTE_LANE_TRANSPORT_FRAME || bytes.len() < 84 {
        return Err(RemoteLaneTransportError::FrameTooLarge);
    }
    let signed_len = bytes
        .len()
        .checked_sub(32)
        .ok_or(RemoteLaneTransportError::Protocol)?;
    let expected = hmac_sha256(
        &session.session_key,
        b"soma.remote-lane.request.v1",
        &bytes[..signed_len],
    );
    if !constant_time_eq(&expected, &bytes[signed_len..]) {
        return Err(RemoteLaneTransportError::Authentication);
    }
    let mut c = Cursor::new(&bytes[..signed_len]);
    if c.u32() != Some(MAGIC) || c.u16() != Some(VERSION) || c.u16() != Some(REQUEST) {
        return Err(RemoteLaneTransportError::Protocol);
    }
    if c.array::<16>() != Some(session.session_id) {
        return Err(RemoteLaneTransportError::WrongSession);
    }
    if c.u64() != Some(session.issuer.0) {
        return Err(RemoteLaneTransportError::WrongIssuer);
    }
    if c.u64() != Some(session.owner.0) {
        return Err(RemoteLaneTransportError::WrongOwner);
    }
    let nonce = c.u64().ok_or(RemoteLaneTransportError::Protocol)?;
    let boundary = c.u32().ok_or(RemoteLaneTransportError::Protocol)?;
    let count = c.u16().ok_or(RemoteLaneTransportError::Protocol)? as usize;
    if c.u16() != Some(0) || count == 0 || count > MAX_REMOTE_LANE_TRANSPORT_BATCHES {
        return Err(RemoteLaneTransportError::Protocol);
    }
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let n = c.u32().ok_or(RemoteLaneTransportError::Protocol)? as usize;
        frames.push(
            c.take(n)
                .ok_or(RemoteLaneTransportError::Protocol)?
                .to_vec(),
        );
    }
    if !c.empty() {
        return Err(RemoteLaneTransportError::Protocol);
    }
    Ok(Request {
        nonce,
        boundary,
        frames,
        digest: Sha256::digest(bytes).into(),
    })
}

#[derive(Clone, Debug)]
pub struct AuthenticatedRemoteLaneResponse {
    wire: Vec<u8>,
}
impl AuthenticatedRemoteLaneResponse {
    pub fn wire_bytes(&self) -> &[u8] {
        &self.wire
    }
    pub fn from_wire(wire: Vec<u8>) -> Self {
        Self { wire }
    }
}

fn encode_response(
    session: &RemoteLaneOwnerSession,
    request: &Request,
    response_ordinal: u64,
    outcomes: &[RemoteLaneOutcome],
) -> Result<Vec<u8>, RemoteLaneTransportError> {
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, RESPONSE);
    out.extend_from_slice(&session.session_id);
    put_u64(&mut out, session.issuer.0);
    put_u64(&mut out, session.owner.0);
    put_u64(&mut out, request.nonce);
    out.extend_from_slice(&request.digest);
    put_u32(&mut out, request.boundary);
    put_u64(&mut out, response_ordinal);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u32(&mut out, outcomes.len() as u32);
    for o in outcomes {
        encode_outcome(&mut out, o)?;
    }
    if out.len() + 32 > MAX_REMOTE_LANE_TRANSPORT_FRAME {
        return Err(RemoteLaneTransportError::FrameTooLarge);
    }
    let mac = hmac_sha256(&session.session_key, b"soma.remote-lane.response.v1", &out);
    out.extend_from_slice(&mac);
    Ok(out)
}

/// Outcomes which passed peer authentication and exact outstanding-request checks.
/// Fields are intentionally private so callers cannot fabricate a Kernel receipt.
#[derive(Debug, PartialEq, Eq)]
pub struct VerifiedRemoteLaneOutcomes {
    nonce: u64,
    session_id: [u8; 16],
    issuer: NodeId,
    owner: NodeId,
    outcomes: Vec<RemoteLaneOutcome>,
}
impl VerifiedRemoteLaneOutcomes {
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
    pub fn outcomes(&self) -> &[RemoteLaneOutcome] {
        &self.outcomes
    }
    pub(crate) fn routing_binding(&self) -> ([u8; 16], NodeId, NodeId) {
        (self.session_id, self.issuer, self.owner)
    }
    pub(crate) fn into_outcomes(self) -> Vec<RemoteLaneOutcome> {
        self.outcomes
    }
}

struct PendingRequest {
    digest: [u8; 32],
    boundary: u32,
    last_response_ordinal: u64,
    expected: Vec<(RemoteLaneRequestId, RemoteRef)>,
    wire: Vec<u8>,
}

pub struct RemoteLaneTransportClient {
    endpoint: SocketAddr,
    session: RemoteLaneClientSession,
    next_nonce: u64,
    pending: HashMap<u64, PendingRequest>,
    consumed: HashMap<u64, [u8; 32]>,
    pending_bytes: usize,
    timeout: Duration,
}
impl RemoteLaneTransportClient {
    pub fn new(endpoint: SocketAddr, session: RemoteLaneClientSession) -> Self {
        Self {
            endpoint,
            session,
            next_nonce: 1,
            pending: HashMap::new(),
            consumed: HashMap::new(),
            pending_bytes: 0,
            timeout: REMOTE_LANE_TRANSPORT_TIMEOUT,
        }
    }
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
    /// Nonces retained after ambiguous transport loss or a temporary result.
    pub fn pending_nonces(&self) -> Vec<u64> {
        let mut n: Vec<_> = self.pending.keys().copied().collect();
        n.sort_unstable();
        n
    }
    pub fn exchange(
        &mut self,
        boundary: u32,
        batches: &[RemoteLaneEffectBatch],
    ) -> Result<VerifiedRemoteLaneOutcomes, RemoteLaneTransportError> {
        if self
            .pending
            .len()
            .checked_add(self.consumed.len())
            .is_none_or(|n| n >= MAX_REMOTE_LANE_TRANSPORT_REPLAY_ENTRIES)
        {
            return Err(RemoteLaneTransportError::Capacity);
        }
        let nonce = self.next_nonce;
        let next_nonce = nonce
            .checked_add(1)
            .ok_or(RemoteLaneTransportError::Protocol)?;
        let frames: Vec<_> = batches.iter().map(RemoteLaneEffectBatch::encode).collect();
        let unsigned = request_unsigned(&self.session, nonce, boundary, &frames)?;
        let mut wire = unsigned;
        let mac = hmac_sha256(
            &self.session.session_key,
            b"soma.remote-lane.request.v1",
            &wire,
        );
        wire.extend_from_slice(&mac);
        if self
            .pending_bytes
            .checked_add(wire.len())
            .is_none_or(|n| n > MAX_REMOTE_LANE_TRANSPORT_REPLAY_BYTES)
        {
            return Err(RemoteLaneTransportError::Capacity);
        }
        let digest: [u8; 32] = Sha256::digest(&wire).into();
        let mut ordered: Vec<_> = batches.iter().flat_map(|b| b.effects().iter()).collect();
        ordered.sort_by_key(|e| (e.epoch, e.lane, e.ordinal, e.request_id));
        let expected = ordered
            .into_iter()
            .map(|e| (e.request_id, e.target))
            .collect();
        self.next_nonce = next_nonce;
        self.pending_bytes += wire.len();
        self.pending.insert(
            nonce,
            PendingRequest {
                digest,
                boundary,
                last_response_ordinal: 0,
                expected,
                wire: wire.clone(),
            },
        );
        let response = match round_trip(self.endpoint, &wire, self.timeout) {
            Ok(response) => response,
            Err(error) => {
                if !matches!(
                    error,
                    RemoteLaneTransportError::TemporaryUnavailable
                        | RemoteLaneTransportError::Timeout
                ) {
                    if let Some(p) = self.pending.remove(&nonce) {
                        self.pending_bytes -= p.wire.len();
                    }
                }
                return Err(error);
            }
        };
        self.accept(AuthenticatedRemoteLaneResponse { wire: response })
    }
    /// Send the exact retained frame without waiting for its response. This
    /// models an ambiguous connection loss; `retry` resends the same nonce and bytes.
    pub fn send_without_receiving(&self, nonce: u64) -> Result<(), RemoteLaneTransportError> {
        let wire = &self
            .pending
            .get(&nonce)
            .ok_or(RemoteLaneTransportError::Replay)?
            .wire;
        let mut stream =
            TcpStream::connect_timeout(&self.endpoint, self.timeout).map_err(map_io)?;
        configure(&stream, self.timeout).map_err(map_io)?;
        write_frame(&mut stream, wire).map_err(map_io)?;
        stream.shutdown(Shutdown::Both).map_err(map_io)
    }

    /// Retry the byte-identical request after a temporary or would-block result.
    pub fn retry(
        &mut self,
        nonce: u64,
    ) -> Result<VerifiedRemoteLaneOutcomes, RemoteLaneTransportError> {
        let wire = self
            .pending
            .get(&nonce)
            .ok_or(RemoteLaneTransportError::Replay)?
            .wire
            .clone();
        let response = match round_trip(self.endpoint, &wire, self.timeout) {
            Ok(response) => response,
            Err(error) => {
                if !matches!(
                    error,
                    RemoteLaneTransportError::TemporaryUnavailable
                        | RemoteLaneTransportError::Timeout
                ) {
                    if let Some(p) = self.pending.remove(&nonce) {
                        self.pending_bytes -= p.wire.len();
                    }
                }
                return Err(error);
            }
        };
        self.accept(AuthenticatedRemoteLaneResponse { wire: response })
    }
    pub fn accept(
        &mut self,
        response: AuthenticatedRemoteLaneResponse,
    ) -> Result<VerifiedRemoteLaneOutcomes, RemoteLaneTransportError> {
        let bytes = response.wire;
        if bytes.len() > MAX_REMOTE_LANE_TRANSPORT_FRAME || bytes.len() < 112 {
            return Err(RemoteLaneTransportError::FrameTooLarge);
        }
        let n = bytes.len() - 32;
        let expected_mac = hmac_sha256(
            &self.session.session_key,
            b"soma.remote-lane.response.v1",
            &bytes[..n],
        );
        if !constant_time_eq(&expected_mac, &bytes[n..]) {
            return Err(RemoteLaneTransportError::Authentication);
        }
        let mut c = Cursor::new(&bytes[..n]);
        if c.u32() != Some(MAGIC) || c.u16() != Some(VERSION) || c.u16() != Some(RESPONSE) {
            return Err(RemoteLaneTransportError::Protocol);
        }
        if c.array::<16>() != Some(self.session.session_id) {
            return Err(RemoteLaneTransportError::WrongSession);
        }
        if c.u64() != Some(self.session.issuer.0) {
            return Err(RemoteLaneTransportError::WrongIssuer);
        }
        if c.u64() != Some(self.session.owner.0) {
            return Err(RemoteLaneTransportError::WrongOwner);
        }
        let nonce = c.u64().ok_or(RemoteLaneTransportError::Protocol)?;
        let digest = c.array::<32>().ok_or(RemoteLaneTransportError::Protocol)?;
        if self.consumed.get(&nonce).is_some_and(|d| *d == digest) {
            return Err(RemoteLaneTransportError::Replay);
        }
        let boundary = c.u32().ok_or(RemoteLaneTransportError::Protocol)?;
        let ordinal = c.u64().ok_or(RemoteLaneTransportError::Protocol)?;
        let pending = self
            .pending
            .get(&nonce)
            .ok_or(RemoteLaneTransportError::Replay)?;
        if pending.digest != digest {
            return Err(RemoteLaneTransportError::Collision);
        }
        if pending.boundary != boundary || ordinal <= pending.last_response_ordinal {
            return Err(RemoteLaneTransportError::Replay);
        }
        let expected = &pending.expected;
        let status = c.u16().ok_or(RemoteLaneTransportError::Protocol)?;
        let detail_len = c.u16().ok_or(RemoteLaneTransportError::Protocol)? as usize;
        let detail = c
            .take(detail_len)
            .ok_or(RemoteLaneTransportError::Protocol)?;
        if status != 0 {
            if c.u32() != Some(0) || !c.empty() {
                return Err(RemoteLaneTransportError::Protocol);
            }
            let error = decode_transport_error(detail);
            if retryable_transport_error(error) {
                if let Some(p) = self.pending.get_mut(&nonce) {
                    p.last_response_ordinal = ordinal;
                }
            } else if let Some(p) = self.pending.remove(&nonce) {
                self.pending_bytes -= p.wire.len();
            }
            return Err(error);
        }
        if !detail.is_empty() {
            return Err(RemoteLaneTransportError::Protocol);
        }
        let count = c.u32().ok_or(RemoteLaneTransportError::Protocol)? as usize;
        if count > MAX_REMOTE_LANE_EFFECTS * MAX_REMOTE_LANE_TRANSPORT_BATCHES {
            return Err(RemoteLaneTransportError::FrameTooLarge);
        }
        let mut outcomes = Vec::with_capacity(count);
        for _ in 0..count {
            outcomes.push(decode_outcome(&mut c)?)
        }
        if !c.empty() {
            return Err(RemoteLaneTransportError::Protocol);
        }
        if outcomes.len() != expected.len()
            || outcomes
                .iter()
                .zip(expected)
                .any(|(o, (id, target))| o.request_id != *id || o.target != *target)
        {
            return Err(RemoteLaneTransportError::Protocol);
        }
        let terminal = outcomes.iter().all(|o| {
            !matches!(
                o.result,
                Ok(RemoteLaneApply::WouldBlock) | Err(RemoteLaneError::NodeUnavailable)
            )
        });
        if terminal {
            if let Some(pending) = self.pending.remove(&nonce) {
                self.pending_bytes -= pending.wire.len();
            }
            self.consumed.insert(nonce, digest);
        } else if let Some(p) = self.pending.get_mut(&nonce) {
            p.last_response_ordinal = ordinal;
        }
        Ok(VerifiedRemoteLaneOutcomes {
            nonce,
            session_id: self.session.session_id,
            issuer: self.session.issuer,
            owner: self.session.owner,
            outcomes,
        })
    }
}

pub struct RemoteLaneTransportServer;
impl RemoteLaneTransportServer {
    pub fn serve_n(
        listener: TcpListener,
        service: Arc<Mutex<RemoteLaneEffectService>>,
        router: Arc<Mutex<RemoteLaneClientRouter>>,
        session: RemoteLaneOwnerSession,
        requests: usize,
    ) -> std::io::Result<()> {
        let mut positions: HashMap<u64, [u8; 32]> = HashMap::new();
        let mut terminal: HashMap<u64, Vec<u8>> = HashMap::new();
        let mut outcome_cache: HashMap<RemoteLaneRequestId, RemoteLaneOutcome> = HashMap::new();
        let mut reserved_response_bytes = 0usize;
        let mut response_reservations: HashMap<u64, usize> = HashMap::new();
        let mut closed = None;
        let mut response_ordinal = 0u64;
        for stream in listener.incoming().take(requests) {
            let mut stream = stream?;
            configure(&stream, REMOTE_LANE_TRANSPORT_TIMEOUT)?;
            let wire = read_frame(&mut stream)?;
            let result: Result<Vec<u8>, RemoteLaneTransportError> = (|| {
                let req = decode_request(&wire, &session)?;
                let processed: Result<Vec<u8>, RemoteLaneTransportError> = (|| {
                    if !positions.contains_key(&req.nonce)
                        && positions.len() >= MAX_REMOTE_LANE_TRANSPORT_REPLAY_ENTRIES
                    {
                        return Err(RemoteLaneTransportError::Capacity);
                    }
                    let exact_retry = if let Some(d) = positions.get(&req.nonce) {
                        if *d != req.digest {
                            return Err(RemoteLaneTransportError::Collision);
                        }
                        true
                    } else {
                        false
                    };
                    let batches: Vec<_> = req
                        .frames
                        .iter()
                        .map(|f| {
                            RemoteLaneEffectBatch::decode(f).map_err(RemoteLaneTransportError::Lane)
                        })
                        .collect::<Result<_, _>>()?;
                    if batches
                        .iter()
                        .flat_map(|b| b.effects())
                        .any(|e| e.target.node != session.owner || e.actor_node != session.issuer)
                    {
                        return Err(RemoteLaneTransportError::WrongOwner);
                    }
                    if exact_retry {
                        // A retry may mix a now-revoked effect with other effects which are
                        // still staged.  Retire the authority failures and retry every live
                        // effect while holding the service/router transaction; cached
                        // WouldBlock outcomes must not strand the live work.
                        let mut exec = router
                            .lock()
                            .map_err(|_| RemoteLaneTransportError::Protocol)?;
                        let mut svc = service
                            .lock()
                            .map_err(|_| RemoteLaneTransportError::Protocol)?;
                        let authority_outcomes: Vec<_> = batches
                            .iter()
                            .flat_map(|b| b.effects())
                            .filter_map(|effect| {
                                svc.authorization_error(effect)
                                    .map(|error| RemoteLaneOutcome {
                                        request_id: effect.request_id,
                                        target: effect.target,
                                        result: Err(error),
                                    })
                            })
                            .collect();
                        if !authority_outcomes.is_empty() {
                            svc.finalize_authority_outcomes(&authority_outcomes);
                            let applied = svc.apply_epoch(req.boundary, &mut *exec);
                            drop(svc);
                            drop(exec);
                            for outcome in applied.into_iter().chain(authority_outcomes) {
                                outcome_cache.insert(outcome.request_id, outcome);
                            }
                            let mut requested: Vec<_> =
                                batches.iter().flat_map(|b| b.effects().iter()).collect();
                            requested.sort_by_key(|e| (e.epoch, e.lane, e.ordinal, e.request_id));
                            let outcomes: Vec<_> = requested
                                .into_iter()
                                .map(|effect| {
                                    outcome_cache
                                        .get(&effect.request_id)
                                        .cloned()
                                        .ok_or(RemoteLaneTransportError::Protocol)
                                })
                                .collect::<Result<_, _>>()?;
                            response_ordinal = response_ordinal
                                .checked_add(1)
                                .ok_or(RemoteLaneTransportError::Protocol)?;
                            let response =
                                encode_response(&session, &req, response_ordinal, &outcomes)?;
                            if outcomes.iter().all(|outcome| {
                                !matches!(
                                    outcome.result,
                                    Ok(RemoteLaneApply::WouldBlock)
                                        | Err(RemoteLaneError::NodeUnavailable)
                                )
                            }) {
                                terminal.insert(req.nonce, response.clone());
                            }
                            return Ok(response);
                        }
                        drop(svc);
                        drop(exec);
                        if let Some(cached) = terminal.get(&req.nonce) {
                            return Ok(cached.clone());
                        }
                    }
                    if !exact_retry
                        && closed.is_some_and(|epoch| {
                            batches
                                .iter()
                                .flat_map(|b| b.effects())
                                .any(|e| e.epoch <= epoch)
                        })
                    {
                        return Err(RemoteLaneTransportError::Late);
                    }
                    if batches
                        .iter()
                        .flat_map(|b| b.effects())
                        .any(|e| e.epoch > req.boundary)
                    {
                        return Err(RemoteLaneTransportError::Lane(
                            RemoteLaneError::InvalidEnvelope,
                        ));
                    }
                    let reservation = batches
                        .iter()
                        .flat_map(|b| b.effects())
                        .try_fold(128usize, |sum, e| {
                            sum.checked_add(match &e.operation {
                                super::remote_lane_effect::RemoteLaneOperation::ObjectRead {
                                    length,
                                    ..
                                } => 128 + *length as usize,
                                _ => 256,
                            })
                        })
                        .ok_or(RemoteLaneTransportError::Capacity)?;
                    if reservation > MAX_REMOTE_LANE_TRANSPORT_FRAME
                        || (!response_reservations.contains_key(&req.nonce)
                            && reserved_response_bytes
                                .checked_add(reservation)
                                .is_none_or(|n| n > MAX_REMOTE_LANE_TRANSPORT_REPLAY_BYTES))
                    {
                        return Err(RemoteLaneTransportError::Capacity);
                    }
                    // Validate the complete request against a shadow service.  Only after
                    // every frame passes do we consume the nonce/reservation and publish it.
                    let exec = router
                        .lock()
                        .map_err(|_| RemoteLaneTransportError::Protocol)?;
                    let mut svc = service
                        .lock()
                        .map_err(|_| RemoteLaneTransportError::Protocol)?;
                    let mut shadow = svc.clone();
                    let frame_refs: Vec<&[u8]> = req.frames.iter().map(Vec::as_slice).collect();
                    shadow
                        .stage_many(&frame_refs, req.boundary, &*exec)
                        .map_err(RemoteLaneTransportError::Lane)?;
                    drop(exec);
                    if !exact_retry {
                        positions.insert(req.nonce, req.digest);
                        reserved_response_bytes += reservation;
                        response_reservations.insert(req.nonce, reservation);
                    }
                    *svc = shadow;
                    drop(svc);
                    let applied = {
                        let mut exec = router
                            .lock()
                            .map_err(|_| RemoteLaneTransportError::Protocol)?;
                        service
                            .lock()
                            .map_err(|_| RemoteLaneTransportError::Protocol)?
                            .apply_epoch(req.boundary, &mut *exec)
                    };
                    for outcome in applied {
                        outcome_cache.insert(outcome.request_id, outcome);
                    }
                    let mut requested: Vec<_> =
                        batches.iter().flat_map(|b| b.effects().iter()).collect();
                    requested.sort_by_key(|e| (e.epoch, e.lane, e.ordinal, e.request_id));
                    let outcomes: Vec<_> = requested
                        .into_iter()
                        .map(|effect| {
                            outcome_cache
                                .get(&effect.request_id)
                                .cloned()
                                .ok_or(RemoteLaneTransportError::Protocol)
                        })
                        .collect::<Result<_, _>>()?;
                    closed = Some(closed.map_or(req.boundary, |v: u32| v.max(req.boundary)));
                    response_ordinal = response_ordinal
                        .checked_add(1)
                        .ok_or(RemoteLaneTransportError::Protocol)?;
                    let response = encode_response(&session, &req, response_ordinal, &outcomes)?;
                    if outcomes.iter().all(|o| {
                        !matches!(
                            o.result,
                            Ok(RemoteLaneApply::WouldBlock) | Err(RemoteLaneError::NodeUnavailable)
                        )
                    }) {
                        if response.len()
                            > *response_reservations
                                .get(&req.nonce)
                                .ok_or(RemoteLaneTransportError::Protocol)?
                        {
                            return Err(RemoteLaneTransportError::Protocol);
                        }
                        terminal.insert(req.nonce, response.clone());
                    }
                    Ok(response)
                })();
                match processed {
                    Ok(response) => Ok(response),
                    Err(error) => {
                        response_ordinal = response_ordinal
                            .checked_add(1)
                            .ok_or(RemoteLaneTransportError::Protocol)?;
                        let response =
                            encode_error_response(&session, &req, response_ordinal, error)?;
                        if positions.get(&req.nonce) == Some(&req.digest)
                            && !retryable_transport_error(error)
                        {
                            terminal.insert(req.nonce, response.clone());
                        }
                        Ok(response)
                    }
                }
            })();
            match result {
                Ok(response) => {
                    let _ = write_frame(&mut stream, &response);
                }
                Err(_) => {
                    let _ = stream.write_all(&0u32.to_le_bytes());
                    let _ = stream.flush();
                }
            }
        }
        Ok(())
    }
}

fn encode_error_response(
    session: &RemoteLaneOwnerSession,
    request: &Request,
    response_ordinal: u64,
    error: RemoteLaneTransportError,
) -> Result<Vec<u8>, RemoteLaneTransportError> {
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    put_u16(&mut out, VERSION);
    put_u16(&mut out, RESPONSE);
    out.extend_from_slice(&session.session_id);
    put_u64(&mut out, session.issuer.0);
    put_u64(&mut out, session.owner.0);
    put_u64(&mut out, request.nonce);
    out.extend_from_slice(&request.digest);
    put_u32(&mut out, request.boundary);
    put_u64(&mut out, response_ordinal);
    put_u16(&mut out, 1);
    let detail = encode_transport_error(error);
    put_u16(&mut out, detail.len() as u16);
    out.extend_from_slice(&detail);
    put_u32(&mut out, 0);
    let mac = hmac_sha256(&session.session_key, b"soma.remote-lane.response.v1", &out);
    out.extend_from_slice(&mac);
    Ok(out)
}
fn encode_transport_error(error: RemoteLaneTransportError) -> Vec<u8> {
    let mut out = Vec::new();
    let tag = match error {
        RemoteLaneTransportError::FrameTooLarge => 1,
        RemoteLaneTransportError::Protocol => 2,
        RemoteLaneTransportError::Authentication => 3,
        RemoteLaneTransportError::WrongSession => 4,
        RemoteLaneTransportError::WrongIssuer => 5,
        RemoteLaneTransportError::WrongOwner => 6,
        RemoteLaneTransportError::Replay => 7,
        RemoteLaneTransportError::Collision => 8,
        RemoteLaneTransportError::Late => 9,
        RemoteLaneTransportError::Capacity => 10,
        RemoteLaneTransportError::Lane(error) => {
            put_u16(&mut out, 11);
            encode_error(&mut out, error);
            return out;
        }
        RemoteLaneTransportError::TemporaryUnavailable => 12,
        RemoteLaneTransportError::Timeout => 13,
    };
    put_u16(&mut out, tag);
    out
}
fn decode_transport_error(bytes: &[u8]) -> RemoteLaneTransportError {
    let mut c = Cursor::new(bytes);
    let Some(tag) = c.u16() else {
        return RemoteLaneTransportError::Protocol;
    };
    let error = match tag {
        1 => RemoteLaneTransportError::FrameTooLarge,
        2 => RemoteLaneTransportError::Protocol,
        3 => RemoteLaneTransportError::Authentication,
        4 => RemoteLaneTransportError::WrongSession,
        5 => RemoteLaneTransportError::WrongIssuer,
        6 => RemoteLaneTransportError::WrongOwner,
        7 => RemoteLaneTransportError::Replay,
        8 => RemoteLaneTransportError::Collision,
        9 => RemoteLaneTransportError::Late,
        10 => RemoteLaneTransportError::Capacity,
        11 => match decode_error(&mut c) {
            Ok(error) => RemoteLaneTransportError::Lane(error),
            Err(_) => RemoteLaneTransportError::Protocol,
        },
        12 => RemoteLaneTransportError::TemporaryUnavailable,
        13 => RemoteLaneTransportError::Timeout,
        _ => RemoteLaneTransportError::Protocol,
    };
    if !c.empty() {
        RemoteLaneTransportError::Protocol
    } else {
        error
    }
}

fn retryable_transport_error(error: RemoteLaneTransportError) -> bool {
    matches!(
        error,
        RemoteLaneTransportError::TemporaryUnavailable
            | RemoteLaneTransportError::Timeout
            | RemoteLaneTransportError::Lane(RemoteLaneError::NodeUnavailable)
    )
}

fn round_trip(
    endpoint: SocketAddr,
    wire: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>, RemoteLaneTransportError> {
    let mut s = TcpStream::connect_timeout(&endpoint, timeout).map_err(map_io)?;
    configure(&s, timeout).map_err(|_| RemoteLaneTransportError::TemporaryUnavailable)?;
    write_frame(&mut s, wire).map_err(map_io)?;
    read_frame(&mut s).map_err(map_io)
}
fn configure(s: &TcpStream, d: Duration) -> std::io::Result<()> {
    s.set_read_timeout(Some(d))?;
    s.set_write_timeout(Some(d))
}
fn map_io(e: std::io::Error) -> RemoteLaneTransportError {
    if matches!(
        e.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        RemoteLaneTransportError::Timeout
    } else if e.kind() == std::io::ErrorKind::InvalidData {
        RemoteLaneTransportError::Protocol
    } else {
        RemoteLaneTransportError::TemporaryUnavailable
    }
}
fn write_frame(s: &mut TcpStream, b: &[u8]) -> std::io::Result<()> {
    if b.len() > MAX_REMOTE_LANE_TRANSPORT_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame",
        ));
    }
    s.write_all(&(b.len() as u32).to_le_bytes())?;
    s.write_all(b)?;
    s.flush()
}
fn read_frame(s: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut n = [0; 4];
    s.read_exact(&mut n)?;
    let n = u32::from_le_bytes(n) as usize;
    if n == 0 || n > MAX_REMOTE_LANE_TRANSPORT_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame",
        ));
    }
    let mut b = vec![0; n];
    s.read_exact(&mut b)?;
    Ok(b)
}

fn encode_outcome(o: &mut Vec<u8>, x: &RemoteLaneOutcome) -> Result<(), RemoteLaneTransportError> {
    o.extend_from_slice(&x.request_id.0);
    put_u64(o, x.target.node.0);
    put_u64(o, x.target.entity.to_u64());
    match &x.result {
        Ok(a) => {
            o.push(0);
            encode_apply(o, a)?
        }
        Err(e) => {
            o.push(1);
            encode_error(o, *e)
        }
    }
    Ok(())
}
fn decode_outcome(c: &mut Cursor<'_>) -> Result<RemoteLaneOutcome, RemoteLaneTransportError> {
    let request_id = RemoteLaneRequestId(c.array().ok_or(RemoteLaneTransportError::Protocol)?);
    let target = RemoteRef {
        node: NodeId(c.u64().ok_or(RemoteLaneTransportError::Protocol)?),
        entity: Ref64::from_u64(c.u64().ok_or(RemoteLaneTransportError::Protocol)?),
    };
    let result = match c.u8() {
        Some(0) => Ok(decode_apply(c)?),
        Some(1) => Err(decode_error(c)?),
        _ => return Err(RemoteLaneTransportError::Protocol),
    };
    Ok(RemoteLaneOutcome {
        request_id,
        target,
        result,
    })
}
fn encode_apply(o: &mut Vec<u8>, a: &RemoteLaneApply) -> Result<(), RemoteLaneTransportError> {
    match a {
        RemoteLaneApply::Applied(v) => {
            o.push(0);
            encode_value(o, v)?
        }
        RemoteLaneApply::WouldBlock => o.push(1),
        RemoteLaneApply::Closed => o.push(2),
        RemoteLaneApply::Lost => o.push(3),
    }
    Ok(())
}
fn decode_apply(c: &mut Cursor<'_>) -> Result<RemoteLaneApply, RemoteLaneTransportError> {
    Ok(match c.u8() {
        Some(0) => RemoteLaneApply::Applied(decode_value(c)?),
        Some(1) => RemoteLaneApply::WouldBlock,
        Some(2) => RemoteLaneApply::Closed,
        Some(3) => RemoteLaneApply::Lost,
        _ => return Err(RemoteLaneTransportError::Protocol),
    })
}
fn encode_value(o: &mut Vec<u8>, v: &RemoteLaneValue) -> Result<(), RemoteLaneTransportError> {
    match v {
        RemoteLaneValue::Unit => o.push(0),
        RemoteLaneValue::Pending => o.push(1),
        RemoteLaneValue::Ref(r) => {
            o.push(2);
            put_u64(o, r.node.0);
            put_u64(o, r.entity.to_u64())
        }
        RemoteLaneValue::Bytes { version, bytes } => {
            if bytes.len() > MAX_REMOTE_LANE_PAYLOAD {
                return Err(RemoteLaneTransportError::FrameTooLarge);
            }
            o.push(3);
            put_u64(o, *version);
            put_u32(o, bytes.len() as u32);
            o.extend_from_slice(bytes)
        }
        RemoteLaneValue::Version {
            version,
            byte_length,
        } => {
            o.push(4);
            put_u64(o, *version);
            put_u64(o, *byte_length)
        }
        RemoteLaneValue::Terminal { status } => {
            o.push(5);
            put_u32(o, *status)
        }
    }
    Ok(())
}
fn decode_value(c: &mut Cursor<'_>) -> Result<RemoteLaneValue, RemoteLaneTransportError> {
    Ok(match c.u8() {
        Some(0) => RemoteLaneValue::Unit,
        Some(1) => RemoteLaneValue::Pending,
        Some(2) => RemoteLaneValue::Ref(RemoteRef {
            node: NodeId(c.u64().ok_or(RemoteLaneTransportError::Protocol)?),
            entity: Ref64::from_u64(c.u64().ok_or(RemoteLaneTransportError::Protocol)?),
        }),
        Some(3) => {
            let version = c.u64().ok_or(RemoteLaneTransportError::Protocol)?;
            let n = c.u32().ok_or(RemoteLaneTransportError::Protocol)? as usize;
            if n > MAX_REMOTE_LANE_PAYLOAD {
                return Err(RemoteLaneTransportError::FrameTooLarge);
            }
            RemoteLaneValue::Bytes {
                version,
                bytes: c
                    .take(n)
                    .ok_or(RemoteLaneTransportError::Protocol)?
                    .to_vec(),
            }
        }
        Some(4) => RemoteLaneValue::Version {
            version: c.u64().ok_or(RemoteLaneTransportError::Protocol)?,
            byte_length: c.u64().ok_or(RemoteLaneTransportError::Protocol)?,
        },
        Some(5) => RemoteLaneValue::Terminal {
            status: c.u32().ok_or(RemoteLaneTransportError::Protocol)?,
        },
        _ => return Err(RemoteLaneTransportError::Protocol),
    })
}
fn encode_error(o: &mut Vec<u8>, e: RemoteLaneError) {
    let (tag, detail) = match e {
        RemoteLaneError::JournalFull => (0, 0),
        RemoteLaneError::PayloadTooLarge => (1, 0),
        RemoteLaneError::InvalidEnvelope => (2, 0),
        RemoteLaneError::WrongOwner => (3, 0),
        RemoteLaneError::Authority(a) => (4, authority_tag(a)),
        RemoteLaneError::Unsupported => (5, 0),
        RemoteLaneError::Protocol => (6, 0),
        RemoteLaneError::NodeUnavailable => (7, 0),
        RemoteLaneError::NodeLost => (8, 0),
        RemoteLaneError::AuthorityDenied => (9, 0),
        RemoteLaneError::StaleVersion { expected, actual } => {
            o.push(10);
            put_u64(o, expected);
            put_u64(o, actual);
            return;
        }
        RemoteLaneError::InvalidSequence => (11, 0),
        RemoteLaneError::InvalidProgram => (12, 0),
        RemoteLaneError::ApplyFailed => (13, 0),
    };
    o.push(tag);
    if tag == 4 {
        o.push(detail)
    }
}
fn decode_error(c: &mut Cursor<'_>) -> Result<RemoteLaneError, RemoteLaneTransportError> {
    Ok(match c.u8() {
        Some(0) => RemoteLaneError::JournalFull,
        Some(1) => RemoteLaneError::PayloadTooLarge,
        Some(2) => RemoteLaneError::InvalidEnvelope,
        Some(3) => RemoteLaneError::WrongOwner,
        Some(4) => RemoteLaneError::Authority(decode_authority(
            c.u8().ok_or(RemoteLaneTransportError::Protocol)?,
        )?),
        Some(5) => RemoteLaneError::Unsupported,
        Some(6) => RemoteLaneError::Protocol,
        Some(7) => RemoteLaneError::NodeUnavailable,
        Some(8) => RemoteLaneError::NodeLost,
        Some(9) => RemoteLaneError::AuthorityDenied,
        Some(10) => RemoteLaneError::StaleVersion {
            expected: c.u64().ok_or(RemoteLaneTransportError::Protocol)?,
            actual: c.u64().ok_or(RemoteLaneTransportError::Protocol)?,
        },
        Some(11) => RemoteLaneError::InvalidSequence,
        Some(12) => RemoteLaneError::InvalidProgram,
        Some(13) => RemoteLaneError::ApplyFailed,
        _ => return Err(RemoteLaneTransportError::Protocol),
    })
}
fn authority_tag(a: RemoteAuthorityError) -> u8 {
    match a {
        RemoteAuthorityError::UnsupportedVersion => 0,
        RemoteAuthorityError::InvalidSignature => 1,
        RemoteAuthorityError::WrongIssuer => 2,
        RemoteAuthorityError::WrongAudience => 3,
        RemoteAuthorityError::WrongTarget => 4,
        RemoteAuthorityError::ObjectVersionMismatch => 5,
        RemoteAuthorityError::InsufficientRights => 6,
        RemoteAuthorityError::NotYetValid => 7,
        RemoteAuthorityError::Expired => 8,
        RemoteAuthorityError::Revoked => 9,
        RemoteAuthorityError::UnknownGrant => 10,
    }
}
fn decode_authority(v: u8) -> Result<RemoteAuthorityError, RemoteLaneTransportError> {
    Ok(match v {
        0 => RemoteAuthorityError::UnsupportedVersion,
        1 => RemoteAuthorityError::InvalidSignature,
        2 => RemoteAuthorityError::WrongIssuer,
        3 => RemoteAuthorityError::WrongAudience,
        4 => RemoteAuthorityError::WrongTarget,
        5 => RemoteAuthorityError::ObjectVersionMismatch,
        6 => RemoteAuthorityError::InsufficientRights,
        7 => RemoteAuthorityError::NotYetValid,
        8 => RemoteAuthorityError::Expired,
        9 => RemoteAuthorityError::Revoked,
        10 => RemoteAuthorityError::UnknownGrant,
        _ => return Err(RemoteLaneTransportError::Protocol),
    })
}
fn put_u16(o: &mut Vec<u8>, v: u16) {
    o.extend_from_slice(&v.to_le_bytes())
}
fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes())
}
fn put_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes())
}
struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let e = self.p.checked_add(n)?;
        let x = self.b.get(self.p..e)?;
        self.p = e;
        Some(x)
    }
    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }
    fn u8(&mut self) -> Option<u8> {
        Some(*self.take(1)?.first()?)
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
    fn empty(&self) -> bool {
        self.p == self.b.len()
    }
}

fn hmac_sha256(key: &[u8; 32], domain: &[u8], message: &[u8]) -> [u8; 32] {
    let mut inner_key = [0x36u8; 64];
    let mut outer_key = [0x5cu8; 64];
    for (i, b) in key.iter().enumerate() {
        inner_key[i] ^= *b;
        outer_key[i] ^= *b;
    }
    let mut inner = Sha256::new();
    inner.update(inner_key);
    inner.update((domain.len() as u64).to_le_bytes());
    inner.update(domain);
    inner.update((message.len() as u64).to_le_bytes());
    inner.update(message);
    let digest = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_key);
    outer.update(digest);
    outer.finalize().into()
}
fn constant_time_eq(left: &[u8; 32], right: &[u8]) -> bool {
    right.len() == 32 && left.iter().zip(right).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Kind;

    fn target(slot: u32) -> RemoteRef {
        RemoteRef {
            node: NodeId(8),
            entity: Ref64::new(slot, 1, Kind::Future),
        }
    }
    fn outcome(byte: u8, slot: u32) -> RemoteLaneOutcome {
        RemoteLaneOutcome {
            request_id: RemoteLaneRequestId([byte; 32]),
            target: target(slot),
            result: Ok(RemoteLaneApply::Applied(RemoteLaneValue::Unit)),
        }
    }
    fn pair(
        expected: Vec<(RemoteLaneRequestId, RemoteRef)>,
    ) -> (RemoteLaneTransportClient, RemoteLaneOwnerSession, Request) {
        let key = [7; 32];
        let client_session = RemoteLaneClientSession::new([4; 16], NodeId(7), NodeId(8), key);
        let owner = RemoteLaneOwnerSession::new([4; 16], NodeId(7), NodeId(8), key);
        let mut client =
            RemoteLaneTransportClient::new("127.0.0.1:1".parse().unwrap(), client_session);
        let digest = [9; 32];
        client.pending.insert(
            3,
            PendingRequest {
                digest,
                boundary: 11,
                last_response_ordinal: 0,
                expected,
                wire: vec![1],
            },
        );
        client.pending_bytes = 1;
        (
            client,
            owner,
            Request {
                nonce: 3,
                boundary: 11,
                frames: vec![],
                digest,
            },
        )
    }

    #[test]
    fn authenticated_response_refuses_tamper_wrong_binding_shape_and_replay() {
        let first = outcome(1, 1);
        let second = outcome(2, 2);
        let expected = vec![
            (first.request_id, first.target),
            (second.request_id, second.target),
        ];
        let (mut client, owner, request) = pair(expected.clone());
        let valid = encode_response(&owner, &request, 1, &[first.clone(), second.clone()]).unwrap();

        let mut tampered = valid.clone();
        tampered[80] ^= 1;
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(tampered)),
            Err(RemoteLaneTransportError::Authentication)
        ));

        let wrong_owner = RemoteLaneOwnerSession::new([4; 16], NodeId(7), NodeId(99), [7; 32]);
        let wrong =
            encode_response(&wrong_owner, &request, 1, &[first.clone(), second.clone()]).unwrap();
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(wrong)),
            Err(RemoteLaneTransportError::WrongOwner)
        ));

        let mut wrong_request = Request {
            nonce: 3,
            boundary: 11,
            frames: vec![],
            digest: [8; 32],
        };
        let collision =
            encode_response(&owner, &wrong_request, 1, &[first.clone(), second.clone()]).unwrap();
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(collision)),
            Err(RemoteLaneTransportError::Collision)
        ));
        wrong_request.digest = [9; 32];

        let missing = encode_response(&owner, &request, 1, std::slice::from_ref(&first)).unwrap();
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(missing)),
            Err(RemoteLaneTransportError::Protocol)
        ));
        let reordered =
            encode_response(&owner, &request, 1, &[second.clone(), first.clone()]).unwrap();
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(reordered)),
            Err(RemoteLaneTransportError::Protocol)
        ));

        let accepted = client
            .accept(AuthenticatedRemoteLaneResponse::from_wire(valid.clone()))
            .unwrap();
        assert_eq!(accepted.outcomes(), &[first, second]);
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(valid)),
            Err(RemoteLaneTransportError::Replay)
        ));
    }
    #[test]
    fn signed_lane_error_preserves_exact_class_and_tampering_fails_authentication() {
        let first = outcome(1, 1);
        let expected = vec![(first.request_id, first.target)];
        let (mut client, owner, request) = pair(expected.clone());
        let wire = encode_error_response(
            &owner,
            &request,
            1,
            RemoteLaneTransportError::Lane(RemoteLaneError::Authority(
                RemoteAuthorityError::Revoked,
            )),
        )
        .unwrap();
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(wire)),
            Err(RemoteLaneTransportError::Lane(RemoteLaneError::Authority(
                RemoteAuthorityError::Revoked
            )))
        ));
        let (mut client, owner, request) = pair(expected);
        let mut wire = encode_error_response(
            &owner,
            &request,
            1,
            RemoteLaneTransportError::Lane(RemoteLaneError::InvalidSequence),
        )
        .unwrap();
        let index = wire.len() - 33;
        wire[index] ^= 1;
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(wire)),
            Err(RemoteLaneTransportError::Authentication)
        ));
    }

    #[test]
    fn tcp_collision_and_prereservation_capacity_refuse_before_owner_mutation() {
        use crate::abi::Rights;
        use crate::distributed::authority::{GrantSpec, RemoteAuthorityStore};
        use crate::distributed::remote_lane_effect::{RemoteLaneApi, RemoteLaneOperation};
        let issuer = NodeId(7);
        let owner_node = NodeId(8);
        let key = [6; 32];
        let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [2; 32])));
        let service = Arc::new(Mutex::new(RemoteLaneEffectService::new(
            owner_node,
            authority.clone(),
        )));
        let router = Arc::new(Mutex::new(RemoteLaneClientRouter::default()));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server_service = service.clone();
        let server = std::thread::spawn(move || {
            RemoteLaneTransportServer::serve_n(
                listener,
                server_service,
                router,
                RemoteLaneOwnerSession::new([6; 16], issuer, owner_node, key),
                3,
            )
        });
        let session = RemoteLaneClientSession::new([6; 16], issuer, owner_node, key);
        let frame = RemoteLaneEffectBatch::default().encode();
        let sign = |nonce, boundary, frames: &[Vec<u8>]| {
            let mut wire = request_unsigned(&session, nonce, boundary, frames).unwrap();
            let mac = hmac_sha256(&key, b"soma.remote-lane.request.v1", &wire);
            wire.extend_from_slice(&mac);
            wire
        };
        let first = sign(1, 0, std::slice::from_ref(&frame));
        let _ = round_trip(endpoint, &first, REMOTE_LANE_TRANSPORT_TIMEOUT).unwrap();
        let changed = sign(1, 1, std::slice::from_ref(&frame));
        let response = round_trip(endpoint, &changed, REMOTE_LANE_TRANSPORT_TIMEOUT).unwrap();
        let digest: [u8; 32] = Sha256::digest(&changed).into();
        let mut client = RemoteLaneTransportClient::new(endpoint, session.clone());
        client.pending_bytes = changed.len();
        client.pending.insert(
            1,
            PendingRequest {
                digest,
                boundary: 1,
                last_response_ordinal: 0,
                expected: vec![],
                wire: changed,
            },
        );
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(response)),
            Err(RemoteLaneTransportError::Collision)
        ));

        let actor = Ref64::new(9, 1, Kind::Process);
        let target = RemoteRef {
            node: owner_node,
            entity: Ref64::new(10, 1, Kind::Object),
        };
        let grant = authority.lock().unwrap().issue(GrantSpec {
            audience: owner_node,
            actor,
            target,
            rights: Rights::READ,
            object_version: 1,
            valid_from_epoch: 0,
            valid_until_epoch: 4,
        });
        let mut lane = RemoteLaneApi::new(1, 0, actor);
        lane.emit(
            target,
            grant,
            RemoteLaneOperation::ObjectRead {
                offset: 0,
                length: (MAX_REMOTE_LANE_TRANSPORT_FRAME as u32) + 1,
            },
        )
        .unwrap();
        let batch = lane.finish();
        let oversized = sign(2, 1, &[batch.encode()]);
        let response = round_trip(endpoint, &oversized, REMOTE_LANE_TRANSPORT_TIMEOUT).unwrap();
        let digest: [u8; 32] = Sha256::digest(&oversized).into();
        let expected = vec![(batch.effects()[0].request_id, target)];
        let mut client = RemoteLaneTransportClient::new(endpoint, session);
        client.pending_bytes = oversized.len();
        client.pending.insert(
            2,
            PendingRequest {
                digest,
                boundary: 1,
                last_response_ordinal: 0,
                expected,
                wire: oversized,
            },
        );
        assert!(matches!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(response)),
            Err(RemoteLaneTransportError::Capacity)
        ));
        server.join().unwrap().unwrap();
        assert_eq!(service.lock().unwrap().applied_len(), 0);
        assert_eq!(service.lock().unwrap().pending_len(), 0);
    }
    #[test]
    fn signed_retryable_errors_retain_exact_wire_and_advance_ordinal() {
        let first = outcome(1, 1);
        let (mut client, owner, request) = pair(vec![(first.request_id, first.target)]);
        let original = client.pending.get(&3).unwrap().wire.clone();
        for (ordinal, error) in [
            (1, RemoteLaneTransportError::TemporaryUnavailable),
            (2, RemoteLaneTransportError::Timeout),
            (
                3,
                RemoteLaneTransportError::Lane(RemoteLaneError::NodeUnavailable),
            ),
        ] {
            let response = encode_error_response(&owner, &request, ordinal, error).unwrap();
            assert_eq!(
                client.accept(AuthenticatedRemoteLaneResponse::from_wire(response)),
                Err(error)
            );
            let pending = client.pending.get(&3).unwrap();
            assert_eq!(pending.wire, original);
            assert_eq!(pending.last_response_ordinal, ordinal);
        }
        let terminal =
            encode_error_response(&owner, &request, 4, RemoteLaneTransportError::Protocol).unwrap();
        assert_eq!(
            client.accept(AuthenticatedRemoteLaneResponse::from_wire(terminal)),
            Err(RemoteLaneTransportError::Protocol)
        );
        assert!(!client.pending.contains_key(&3));
    }
}
