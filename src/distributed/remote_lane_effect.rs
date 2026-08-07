//! One bounded, authenticated, multiplexed journal for remote lane effects.
//!
//! The journal is transport-neutral.  A lane emits node-qualified operations;
//! an owner stages a whole bounded frame and applies it only at an epoch
//! boundary.  Request identities cover every semantic byte (including the
//! grant), so retries are exact.  Live authority is checked both at staging and
//! immediately before the apply-once ledger is consulted.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityError, RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::{Ref64, Rights};

pub const MAX_REMOTE_LANE_EFFECTS: usize = 256;
pub const MAX_REMOTE_LANE_AUTHORITIES: usize = 8;
/// Session-lifetime apply-once ledger bounds. This is bounded refusal, not a
/// persistence or durability claim; entries are never evicted and reused.
pub const MAX_REMOTE_LANE_LEDGER_ENTRIES: usize = 4096;
pub const MAX_REMOTE_LANE_LEDGER_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_REMOTE_LANE_OUTBOX_ENTRIES: usize = 256;
pub const MAX_REMOTE_LANE_OUTBOX_BYTES: usize = 1024 * 1024;
/// Conservative encoded size reservation for the single-instruction v1 program.
pub const MAX_REMOTE_LANE_PROGRAM_FRAME_BYTES: usize = 512;
pub const MAX_REMOTE_LANE_PAYLOAD: usize = 1024 * 1024;
const MAGIC: u32 = 0x534c_4546;
const VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RemoteLaneRequestId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteLaneOperation {
    FutureAwait,
    FutureResolve {
        value: Ref64,
    },
    ChannelSend {
        sequence: u64,
        value: Ref64,
    },
    ChannelReceive {
        sequence: u64,
    },
    ObjectRead {
        offset: u64,
        length: u32,
    },
    ObjectWrite {
        expected_version: u64,
        offset: u64,
        bytes: Vec<u8>,
    },
    MailboxSend {
        sender_sequence: u64,
        bytes: Vec<u8>,
    },
    ObserveTerminal,
}
impl RemoteLaneOperation {
    pub fn required_rights(&self) -> u32 {
        match self {
            Self::FutureAwait => Rights::AWAIT,
            Self::FutureResolve { .. } => Rights::RESOLVE,
            Self::ChannelSend { .. } | Self::MailboxSend { .. } => Rights::SEND,
            Self::ChannelReceive { .. } => Rights::RECEIVE,
            Self::ObjectRead { .. } | Self::ObserveTerminal => Rights::READ,
            Self::ObjectWrite { .. } => Rights::WRITE,
        }
    }
    fn tag(&self) -> u16 {
        match self {
            Self::FutureAwait => 1,
            Self::FutureResolve { .. } => 2,
            Self::ChannelSend { .. } => 3,
            Self::ChannelReceive { .. } => 4,
            Self::ObjectRead { .. } => 5,
            Self::ObjectWrite { .. } => 6,
            Self::MailboxSend { .. } => 7,
            Self::ObserveTerminal => 8,
        }
    }
    fn encode_body(&self, out: &mut Vec<u8>) {
        match self {
            Self::FutureAwait | Self::ObserveTerminal => {}
            Self::FutureResolve { value } => put_u64(out, value.to_u64()),
            Self::ChannelSend { sequence, value } => {
                put_u64(out, *sequence);
                put_u64(out, value.to_u64());
            }
            Self::ChannelReceive { sequence } => put_u64(out, *sequence),
            Self::ObjectRead { offset, length } => {
                put_u64(out, *offset);
                put_u32(out, *length);
            }
            Self::ObjectWrite {
                expected_version,
                offset,
                bytes,
            } => {
                put_u64(out, *expected_version);
                put_u64(out, *offset);
                put_bytes(out, bytes);
            }
            Self::MailboxSend {
                sender_sequence,
                bytes,
            } => {
                put_u64(out, *sender_sequence);
                put_bytes(out, bytes);
            }
        }
    }
    fn decode(tag: u16, c: &mut Cursor<'_>) -> Option<Self> {
        Some(match tag {
            1 => Self::FutureAwait,
            2 => Self::FutureResolve {
                value: Ref64::from_u64(c.u64()?),
            },
            3 => Self::ChannelSend {
                sequence: c.u64()?,
                value: Ref64::from_u64(c.u64()?),
            },
            4 => Self::ChannelReceive { sequence: c.u64()? },
            5 => Self::ObjectRead {
                offset: c.u64()?,
                length: c.u32()?,
            },
            6 => Self::ObjectWrite {
                expected_version: c.u64()?,
                offset: c.u64()?,
                bytes: c.bytes()?.to_vec(),
            },
            7 => Self::MailboxSend {
                sender_sequence: c.u64()?,
                bytes: c.bytes()?.to_vec(),
            },
            8 => Self::ObserveTerminal,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteLaneEffect {
    pub request_id: RemoteLaneRequestId,
    pub epoch: u32,
    pub lane: u32,
    pub ordinal: u32,
    pub actor_node: NodeId,
    pub actor: Ref64,
    pub target: RemoteRef,
    pub grant: RemoteGrant,
    pub operation: RemoteLaneOperation,
}
impl RemoteLaneEffect {
    pub fn new(
        epoch: u32,
        lane: u32,
        ordinal: u32,
        actor: Ref64,
        target: RemoteRef,
        grant: RemoteGrant,
        operation: RemoteLaneOperation,
    ) -> Self {
        let mut e = Self {
            request_id: RemoteLaneRequestId([0; 32]),
            epoch,
            lane,
            ordinal,
            actor_node: grant.issuer,
            actor,
            target,
            grant,
            operation,
        };
        e.request_id = e.computed_id();
        e
    }
    fn identity(&self) -> Vec<u8> {
        let mut o = Vec::new();
        put_u32(&mut o, self.epoch);
        put_u32(&mut o, self.lane);
        put_u32(&mut o, self.ordinal);
        put_u64(&mut o, self.actor_node.0);
        put_u64(&mut o, self.actor.to_u64());
        put_u64(&mut o, self.target.node.0);
        put_u64(&mut o, self.target.entity.to_u64());
        o.extend_from_slice(&self.grant.encode());
        put_u16(&mut o, self.operation.tag());
        self.operation.encode_body(&mut o);
        o
    }
    pub fn computed_id(&self) -> RemoteLaneRequestId {
        RemoteLaneRequestId(Sha256::digest(self.identity()).into())
    }
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.request_id.0);
        let b = self.identity();
        put_u32(out, b.len() as u32);
        out.extend_from_slice(&b)
    }
    fn decode(c: &mut Cursor<'_>) -> Option<Self> {
        let request_id = RemoteLaneRequestId(c.array()?);
        let n = c.u32()? as usize;
        let mut i = Cursor::new(c.take(n)?);
        let epoch = i.u32()?;
        let lane = i.u32()?;
        let ordinal = i.u32()?;
        let actor_node = NodeId(i.u64()?);
        let actor = Ref64::from_u64(i.u64()?);
        let target = RemoteRef {
            node: NodeId(i.u64()?),
            entity: Ref64::from_u64(i.u64()?),
        };
        let grant = RemoteGrant::decode(i.take(RemoteGrant::ENCODED_LEN)?)?;
        let operation = RemoteLaneOperation::decode(i.u16()?, &mut i)?;
        if !i.empty() {
            return None;
        }
        let e = Self {
            request_id,
            epoch,
            lane,
            ordinal,
            actor_node,
            actor,
            target,
            grant,
            operation,
        };
        (actor_node == grant.issuer && e.computed_id() == request_id).then_some(e)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RemoteLaneEffectBatch {
    effects: Vec<RemoteLaneEffect>,
    payload_bytes: usize,
}
impl RemoteLaneEffectBatch {
    pub fn effects(&self) -> &[RemoteLaneEffect] {
        &self.effects
    }
    pub fn push(&mut self, effect: RemoteLaneEffect) -> Result<(), RemoteLaneError> {
        if effect.target != effect.grant.target
            || effect.actor != effect.grant.actor
            || effect.actor_node != effect.grant.issuer
        {
            return Err(RemoteLaneError::InvalidEnvelope);
        }
        if self.effects.len() >= MAX_REMOTE_LANE_EFFECTS {
            return Err(RemoteLaneError::JournalFull);
        }
        let n = operation_payload_len(&effect.operation);
        if self
            .payload_bytes
            .checked_add(n)
            .is_none_or(|v| v > MAX_REMOTE_LANE_PAYLOAD)
        {
            return Err(RemoteLaneError::PayloadTooLarge);
        }
        self.payload_bytes += n;
        self.effects.push(effect);
        Ok(())
    }
    pub fn canonicalize(&mut self) {
        self.effects
            .sort_by_key(|e| (e.epoch, e.lane, e.ordinal, e.request_id));
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut o = Vec::new();
        put_u32(&mut o, MAGIC);
        put_u16(&mut o, VERSION);
        put_u16(&mut o, 0);
        put_u32(&mut o, self.effects.len() as u32);
        for e in &self.effects {
            e.encode(&mut o)
        }
        o
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, RemoteLaneError> {
        if bytes.len() > MAX_REMOTE_LANE_PAYLOAD + MAX_REMOTE_LANE_EFFECTS * 256 {
            return Err(RemoteLaneError::PayloadTooLarge);
        }
        let mut c = Cursor::new(bytes);
        if c.u32() != Some(MAGIC) || c.u16() != Some(VERSION) || c.u16() != Some(0) {
            return Err(RemoteLaneError::Protocol);
        }
        let n = c.u32().ok_or(RemoteLaneError::Protocol)? as usize;
        if n > MAX_REMOTE_LANE_EFFECTS {
            return Err(RemoteLaneError::JournalFull);
        }
        let mut b = Self::default();
        for _ in 0..n {
            b.push(RemoteLaneEffect::decode(&mut c).ok_or(RemoteLaneError::Protocol)?)?
        }
        if !c.empty() {
            return Err(RemoteLaneError::Protocol);
        }
        Ok(b)
    }
}

/// Bounded journal builder for a future validated lane bridge. It is not itself
/// a Kernel continuation handler and does not assign canonical lane positions.
pub struct RemoteLaneApi {
    epoch: u32,
    lane: u32,
    actor: Ref64,
    next_ordinal: u32,
    batch: RemoteLaneEffectBatch,
}
impl RemoteLaneApi {
    pub fn new(epoch: u32, lane: u32, actor: Ref64) -> Self {
        Self {
            epoch,
            lane,
            actor,
            next_ordinal: 0,
            batch: RemoteLaneEffectBatch::default(),
        }
    }
    pub fn emit(
        &mut self,
        target: RemoteRef,
        grant: RemoteGrant,
        operation: RemoteLaneOperation,
    ) -> Result<RemoteLaneRequestId, RemoteLaneError> {
        let e = RemoteLaneEffect::new(
            self.epoch,
            self.lane,
            self.next_ordinal,
            self.actor,
            target,
            grant,
            operation,
        );
        let id = e.request_id;
        self.batch.push(e)?;
        self.next_ordinal = self
            .next_ordinal
            .checked_add(1)
            .ok_or(RemoteLaneError::JournalFull)?;
        Ok(id)
    }
    pub fn finish(self) -> RemoteLaneEffectBatch {
        self.batch
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteLaneValue {
    Unit,
    Pending,
    Ref(RemoteRef),
    Bytes { version: u64, bytes: Vec<u8> },
    Version { version: u64, byte_length: u64 },
    Terminal { status: u32 },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteLaneApply {
    Applied(RemoteLaneValue),
    WouldBlock,
    Closed,
    Lost,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteLaneOutcome {
    pub request_id: RemoteLaneRequestId,
    pub target: RemoteRef,
    pub result: Result<RemoteLaneApply, RemoteLaneError>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteLaneError {
    JournalFull,
    PayloadTooLarge,
    InvalidEnvelope,
    WrongOwner,
    Authority(RemoteAuthorityError),
    Unsupported,
    Protocol,
    NodeUnavailable,
    NodeLost,
    AuthorityDenied,
    StaleVersion { expected: u64, actual: u64 },
    InvalidSequence,
    InvalidProgram,
    ApplyFailed,
}

pub trait RemoteLaneExecutor {
    fn supports(&self, operation: &RemoteLaneOperation) -> bool;
    fn apply(&mut self, effect: &RemoteLaneEffect) -> Result<RemoteLaneApply, RemoteLaneError>;
}

/// Owner-side boundary stage. It contains no resource descriptors: authoritative
/// resource implementations remain behind `RemoteLaneExecutor`.
#[derive(Clone)]
pub struct RemoteLaneEffectService {
    node: NodeId,
    authorities: Vec<Arc<Mutex<RemoteAuthorityStore>>>,
    object_versions: HashMap<RemoteRef, u32>,
    staged: Vec<RemoteLaneEffect>,
    ledger: HashMap<RemoteLaneRequestId, RemoteLaneOutcome>,
    positions: HashMap<(u32, u32, u32), RemoteLaneRequestId>,
    last_closed_epoch: Option<u32>,
    reserved_ledger_bytes: usize,
}
impl RemoteLaneEffectService {
    pub fn new(node: NodeId, authority: Arc<Mutex<RemoteAuthorityStore>>) -> Self {
        Self {
            node,
            authorities: vec![authority],
            object_versions: HashMap::new(),
            staged: vec![],
            ledger: HashMap::new(),
            positions: HashMap::new(),
            last_closed_epoch: None,
            reserved_ledger_bytes: 0,
        }
    }
    pub fn trust_authority(
        &mut self,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Result<(), RemoteLaneError> {
        if self.authorities.len() >= MAX_REMOTE_LANE_AUTHORITIES {
            return Err(RemoteLaneError::JournalFull);
        }
        self.authorities.push(authority);
        Ok(())
    }
    pub fn register_target(
        &mut self,
        target: RemoteRef,
        object_version: u32,
    ) -> Result<(), RemoteLaneError> {
        if target.node != self.node {
            return Err(RemoteLaneError::WrongOwner);
        }
        self.object_versions.insert(target, object_version);
        Ok(())
    }
    fn authorize(&self, e: &RemoteLaneEffect) -> Result<(), RemoteLaneError> {
        if e.target.node != self.node {
            return Err(RemoteLaneError::WrongOwner);
        }
        let version = *self
            .object_versions
            .get(&e.target)
            .ok_or(RemoteLaneError::Unsupported)?;
        let mut last = RemoteAuthorityError::UnknownGrant;
        for authority in &self.authorities {
            let guard = authority.lock().map_err(|_| RemoteLaneError::ApplyFailed)?;
            match guard.authorize(
                &e.grant,
                self.node,
                e.target,
                e.operation.required_rights(),
                version,
                e.epoch,
            ) {
                Ok(()) => return Ok(()),
                Err(error) => last = error,
            }
        }
        Err(RemoteLaneError::Authority(last))
    }
    pub(crate) fn authorization_error(&self, effect: &RemoteLaneEffect) -> Option<RemoteLaneError> {
        self.authorize(effect).err()
    }
    /// Rejects unknown operation kinds and bad grants before anything is staged.
    pub fn stage(
        &mut self,
        frame: &[u8],
        executor: &dyn RemoteLaneExecutor,
    ) -> Result<(), RemoteLaneError> {
        self.stage_many(std::slice::from_ref(&frame), u32::MAX, executor)
    }

    /// Atomically validates and stages every frame in one bounded request.
    /// No position, reservation, or effect is published unless all frames pass,
    /// and an effect may never be staged ahead of the request boundary.
    pub fn stage_many(
        &mut self,
        frames: &[&[u8]],
        boundary: u32,
        executor: &dyn RemoteLaneExecutor,
    ) -> Result<(), RemoteLaneError> {
        if frames.is_empty() {
            return Err(RemoteLaneError::InvalidEnvelope);
        }
        let mut effects = Vec::new();
        for frame in frames {
            let mut batch = RemoteLaneEffectBatch::decode(frame)?;
            batch.canonicalize();
            effects.extend(batch.effects);
        }
        if effects.len() > MAX_REMOTE_LANE_EFFECTS {
            return Err(RemoteLaneError::JournalFull);
        }
        effects.sort_by_key(|e| (e.epoch, e.lane, e.ordinal, e.request_id));

        let mut previous: Option<(u32, u32, u32)> = None;
        let mut proposed_positions = HashMap::new();
        let mut proposed_ids: HashMap<RemoteLaneRequestId, RemoteLaneEffect> = HashMap::new();
        for e in &effects {
            if e.epoch > boundary || !executor.supports(&e.operation) {
                return Err(if e.epoch > boundary {
                    RemoteLaneError::InvalidEnvelope
                } else {
                    RemoteLaneError::Unsupported
                });
            }
            self.authorize(e)?;
            if let Some((epoch, lane, ordinal)) = previous {
                if epoch == e.epoch && lane == e.lane && e.ordinal != ordinal + 1 {
                    return Err(RemoteLaneError::InvalidEnvelope);
                }
                if (epoch != e.epoch || lane != e.lane) && e.ordinal != 0 {
                    return Err(RemoteLaneError::InvalidEnvelope);
                }
            } else if e.ordinal != 0 {
                return Err(RemoteLaneError::InvalidEnvelope);
            }
            previous = Some((e.epoch, e.lane, e.ordinal));
            let position = (e.epoch, e.lane, e.ordinal);
            if self
                .positions
                .get(&position)
                .or_else(|| proposed_positions.get(&position))
                .is_some_and(|id| *id != e.request_id)
            {
                return Err(RemoteLaneError::InvalidEnvelope);
            }
            proposed_positions.insert(position, e.request_id);
            if self
                .last_closed_epoch
                .is_some_and(|closed| e.epoch <= closed)
                && !self.ledger.contains_key(&e.request_id)
                && !self
                    .staged
                    .iter()
                    .any(|pending| pending.request_id == e.request_id)
            {
                return Err(RemoteLaneError::InvalidEnvelope);
            }
            if let Some(existing) = proposed_ids.insert(e.request_id, e.clone()) {
                if existing != *e {
                    return Err(RemoteLaneError::InvalidEnvelope);
                }
            }
        }
        let new_effects: Vec<_> = proposed_ids
            .into_values()
            .filter(|e| {
                !self.staged.iter().any(|p| p.request_id == e.request_id)
                    && !self.ledger.contains_key(&e.request_id)
            })
            .collect();
        if self
            .staged
            .len()
            .checked_add(new_effects.len())
            .is_none_or(|n| n > MAX_REMOTE_LANE_EFFECTS)
            || self
                .positions
                .len()
                .checked_add(new_effects.len())
                .is_none_or(|n| n > MAX_REMOTE_LANE_LEDGER_ENTRIES)
        {
            return Err(RemoteLaneError::JournalFull);
        }
        let reserve = new_effects
            .iter()
            .try_fold(0usize, |sum, e| {
                sum.checked_add(outcome_reservation(&e.operation))
            })
            .ok_or(RemoteLaneError::JournalFull)?;
        if self
            .reserved_ledger_bytes
            .checked_add(reserve)
            .is_none_or(|n| n > MAX_REMOTE_LANE_LEDGER_BYTES)
        {
            return Err(RemoteLaneError::JournalFull);
        }

        self.reserved_ledger_bytes += reserve;
        for e in &new_effects {
            self.positions
                .insert((e.epoch, e.lane, e.ordinal), e.request_id);
        }
        self.staged.extend(new_effects);
        Ok(())
    }
    /// Applies canonical order once. Would-block records remain staged for the
    /// next boundary; revocation is checked on every attempt and before replay.
    pub fn apply_epoch(
        &mut self,
        boundary: u32,
        executor: &mut dyn RemoteLaneExecutor,
    ) -> Vec<RemoteLaneOutcome> {
        self.last_closed_epoch = Some(
            self.last_closed_epoch
                .map_or(boundary, |closed| closed.max(boundary)),
        );
        self.staged
            .sort_by_key(|e| (e.epoch, e.lane, e.ordinal, e.request_id));
        let mut keep = Vec::new();
        let mut out = Vec::new();
        for e in std::mem::take(&mut self.staged) {
            if e.epoch > boundary {
                keep.push(e);
                continue;
            }
            if let Err(error) = self.authorize(&e) {
                out.push(RemoteLaneOutcome {
                    request_id: e.request_id,
                    target: e.target,
                    result: Err(error),
                });
                continue;
            }
            if let Some(cached) = self.ledger.get(&e.request_id) {
                out.push(cached.clone());
                continue;
            }
            let result = executor.apply(&e);
            let outcome = RemoteLaneOutcome {
                request_id: e.request_id,
                target: e.target,
                result,
            };
            if matches!(
                outcome.result,
                Ok(RemoteLaneApply::WouldBlock) | Err(RemoteLaneError::NodeUnavailable)
            ) {
                keep.push(e);
            } else {
                self.ledger.insert(outcome.request_id, outcome.clone());
            }
            out.push(outcome)
        }
        self.staged = keep;
        out
    }
    pub(crate) fn finalize_authority_outcomes(&mut self, outcomes: &[RemoteLaneOutcome]) {
        let terminal: std::collections::HashSet<_> = outcomes
            .iter()
            .filter_map(|outcome| {
                matches!(outcome.result, Err(RemoteLaneError::Authority(_)))
                    .then_some(outcome.request_id)
            })
            .collect();
        let released = self
            .staged
            .iter()
            .filter(|effect| terminal.contains(&effect.request_id))
            .map(|effect| outcome_reservation(&effect.operation))
            .sum::<usize>();
        self.staged
            .retain(|effect| !terminal.contains(&effect.request_id));
        self.reserved_ledger_bytes = self.reserved_ledger_bytes.saturating_sub(released);
    }
    pub fn pending_len(&self) -> usize {
        self.staged.len()
    }
    pub fn applied_len(&self) -> usize {
        self.ledger.len()
    }
}

fn outcome_reservation(o: &RemoteLaneOperation) -> usize {
    match o {
        RemoteLaneOperation::ObjectRead { length, .. } => 128 + *length as usize,
        _ => 256,
    }
}
fn operation_payload_len(o: &RemoteLaneOperation) -> usize {
    match o {
        RemoteLaneOperation::ObjectWrite { bytes, .. }
        | RemoteLaneOperation::MailboxSend { bytes, .. } => bytes.len(),
        _ => 0,
    }
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
fn put_bytes(o: &mut Vec<u8>, b: &[u8]) {
    put_u32(o, b.len() as u32);
    o.extend_from_slice(b)
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
        let s = self.b.get(self.p..e)?;
        self.p = e;
        Some(s)
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
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn empty(&self) -> bool {
        self.p == self.b.len()
    }
}

pub const PROGRAM_FUTURE_AWAIT: u16 = 1;
pub const PROGRAM_FUTURE_RESOLVE: u16 = 2;
pub const PROGRAM_CHANNEL_SEND: u16 = 3;
pub const PROGRAM_CHANNEL_RECEIVE: u16 = 4;
pub const PROGRAM_OBJECT_READ: u16 = 5;
pub const PROGRAM_OBJECT_WRITE: u16 = 6;
/// Pointer-free fixed-width instruction. Variable object bytes live in the
/// separately bounded program payload and are addressed only by offset/length.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteLaneInstruction {
    pub opcode: u16,
    pub reserved: u16,
    pub target_node: u64,
    pub target_entity: u64,
    pub grant: [u8; RemoteGrant::ENCODED_LEN],
    pub argument0: u64,
    pub argument1: u64,
    pub value: u64,
    pub payload_offset: u32,
    pub payload_len: u32,
}
#[derive(Clone, Debug)]
pub struct RemoteLaneProgram {
    instructions: Vec<RemoteLaneInstruction>,
    payload: Vec<u8>,
}
impl RemoteLaneProgram {
    pub fn validate(
        instructions: Vec<RemoteLaneInstruction>,
        payload: Vec<u8>,
    ) -> Result<Self, RemoteLaneError> {
        // v1 Kernel dispatch intentionally accepts exactly one operation that
        // can park. Multi-op/group result commit and frame result slots are a
        // later version; rejecting here is transactional, before dispatch.
        if instructions.len() != 1
            || !payload.is_empty()
            || !matches!(
                instructions[0].opcode,
                PROGRAM_FUTURE_AWAIT | PROGRAM_CHANNEL_SEND | PROGRAM_CHANNEL_RECEIVE
            )
        {
            return Err(RemoteLaneError::InvalidProgram);
        }
        for i in &instructions {
            if i.reserved != 0 || !(PROGRAM_FUTURE_AWAIT..=PROGRAM_OBJECT_WRITE).contains(&i.opcode)
            {
                return Err(RemoteLaneError::InvalidProgram);
            }
            let grant = RemoteGrant::decode(&i.grant).ok_or(RemoteLaneError::InvalidProgram)?;
            let target = RemoteRef {
                node: NodeId(i.target_node),
                entity: Ref64::from_u64(i.target_entity),
            };
            if grant.target != target {
                return Err(RemoteLaneError::InvalidProgram);
            }
            let end = (i.payload_offset as usize)
                .checked_add(i.payload_len as usize)
                .ok_or(RemoteLaneError::InvalidProgram)?;
            if end > payload.len() || (i.opcode != PROGRAM_OBJECT_WRITE && i.payload_len != 0) {
                return Err(RemoteLaneError::InvalidProgram);
            }
        }
        Ok(Self {
            instructions,
            payload,
        })
    }
    pub fn instructions(&self) -> &[RemoteLaneInstruction] {
        &self.instructions
    }
    pub(crate) fn instantiate<F>(
        &self,
        epoch: u32,
        lane: u32,
        actor: Ref64,
        mut sequence: F,
    ) -> Result<RemoteLaneEffectBatch, RemoteLaneError>
    where
        F: FnMut(RemoteRef, bool) -> u64,
    {
        // Validate the signed actor and every shape before sequence allocation.
        for i in &self.instructions {
            let g = RemoteGrant::decode(&i.grant).ok_or(RemoteLaneError::InvalidProgram)?;
            if g.actor != actor {
                return Err(RemoteLaneError::InvalidEnvelope);
            }
        }
        let mut api = RemoteLaneApi::new(epoch, lane, actor);
        for i in &self.instructions {
            let grant = RemoteGrant::decode(&i.grant).ok_or(RemoteLaneError::InvalidProgram)?;
            let target = grant.target;
            let operation = match i.opcode {
                PROGRAM_FUTURE_AWAIT => RemoteLaneOperation::FutureAwait,
                PROGRAM_FUTURE_RESOLVE => RemoteLaneOperation::FutureResolve {
                    value: Ref64::from_u64(i.value),
                },
                PROGRAM_CHANNEL_SEND => RemoteLaneOperation::ChannelSend {
                    sequence: sequence(target, true),
                    value: Ref64::from_u64(i.value),
                },
                PROGRAM_CHANNEL_RECEIVE => RemoteLaneOperation::ChannelReceive {
                    sequence: sequence(target, false),
                },
                PROGRAM_OBJECT_READ => RemoteLaneOperation::ObjectRead {
                    offset: i.argument0,
                    length: i
                        .argument1
                        .try_into()
                        .map_err(|_| RemoteLaneError::InvalidProgram)?,
                },
                PROGRAM_OBJECT_WRITE => {
                    let start = i.payload_offset as usize;
                    let end = start + i.payload_len as usize;
                    RemoteLaneOperation::ObjectWrite {
                        expected_version: i.argument0,
                        offset: i.argument1,
                        bytes: self.payload[start..end].to_vec(),
                    }
                }
                _ => return Err(RemoteLaneError::InvalidProgram),
            };
            api.emit(target, grant, operation)?;
        }
        Ok(api.finish())
    }
}
#[derive(Clone, Debug)]
pub struct KernelRemoteLaneEmission {
    pub continuation: Ref64,
    pub process: Ref64,
    pub run_class: u32,
    pub batch: RemoteLaneEffectBatch,
}

use super::remote_channel::{
    RemoteChannelClient, RemoteChannelError, RemoteReceiveOutcome, RemoteSendOutcome,
};
use super::remote_future::{RemoteFutureClient, RemoteFutureError, RemoteFutureState};
use super::remote_object::{RemoteObjectClient, RemoteObjectError};

enum RoutedResource {
    Future(RemoteFutureClient),
    Channel(RemoteChannelClient),
    Object(RemoteObjectClient),
}
/// Multiplexes the six implemented operations through one executor. Clients are
/// transport proxies only; registering one never allocates a local ABI ref.
#[derive(Default)]
pub struct RemoteLaneClientRouter {
    resources: HashMap<RemoteRef, RoutedResource>,
}
impl RemoteLaneClientRouter {
    pub fn register_future(
        &mut self,
        target: RemoteRef,
        client: RemoteFutureClient,
    ) -> Result<(), RemoteLaneError> {
        self.insert(target, RoutedResource::Future(client))
    }
    pub fn register_channel(
        &mut self,
        target: RemoteRef,
        client: RemoteChannelClient,
    ) -> Result<(), RemoteLaneError> {
        self.insert(target, RoutedResource::Channel(client))
    }
    pub fn register_object(
        &mut self,
        target: RemoteRef,
        client: RemoteObjectClient,
    ) -> Result<(), RemoteLaneError> {
        self.insert(target, RoutedResource::Object(client))
    }
    fn insert(&mut self, target: RemoteRef, r: RoutedResource) -> Result<(), RemoteLaneError> {
        if self.resources.insert(target, r).is_some() {
            return Err(RemoteLaneError::InvalidEnvelope);
        }
        Ok(())
    }
}
impl RemoteLaneExecutor for RemoteLaneClientRouter {
    fn supports(&self, o: &RemoteLaneOperation) -> bool {
        matches!(
            o,
            RemoteLaneOperation::FutureAwait
                | RemoteLaneOperation::FutureResolve { .. }
                | RemoteLaneOperation::ChannelSend { .. }
                | RemoteLaneOperation::ChannelReceive { .. }
                | RemoteLaneOperation::ObjectRead { .. }
                | RemoteLaneOperation::ObjectWrite { .. }
        )
    }
    fn apply(&mut self, e: &RemoteLaneEffect) -> Result<RemoteLaneApply, RemoteLaneError> {
        let resource = self
            .resources
            .get_mut(&e.target)
            .ok_or(RemoteLaneError::Unsupported)?;
        match (resource, &e.operation) {
            (RoutedResource::Future(c), RemoteLaneOperation::FutureAwait) => {
                let c = c.rebound(e.grant, e.epoch);
                match c.poll().map_err(map_future)? {
                    RemoteFutureState::Pending => Ok(RemoteLaneApply::WouldBlock),
                    RemoteFutureState::Resolved { value, .. } => {
                        Ok(RemoteLaneApply::Applied(RemoteLaneValue::Ref(RemoteRef {
                            node: e.target.node,
                            entity: value,
                        })))
                    }
                }
            }
            (RoutedResource::Future(c), RemoteLaneOperation::FutureResolve { value }) => {
                let c = c.rebound(e.grant, e.epoch);
                c.resolve(*value).map_err(map_future)?;
                Ok(RemoteLaneApply::Applied(RemoteLaneValue::Unit))
            }
            (RoutedResource::Channel(c), RemoteLaneOperation::ChannelSend { sequence, value }) => {
                let c = c.rebound(e.grant, e.epoch);
                match c.send(*sequence, *value).map_err(map_channel)? {
                    RemoteSendOutcome::Sent { .. } => {
                        Ok(RemoteLaneApply::Applied(RemoteLaneValue::Unit))
                    }
                    RemoteSendOutcome::Full => Ok(RemoteLaneApply::WouldBlock),
                    RemoteSendOutcome::Closed => Ok(RemoteLaneApply::Closed),
                }
            }
            (RoutedResource::Channel(c), RemoteLaneOperation::ChannelReceive { sequence }) => {
                let c = c.rebound(e.grant, e.epoch);
                match c.receive(*sequence).map_err(map_channel)? {
                    RemoteReceiveOutcome::Received(v) => {
                        Ok(RemoteLaneApply::Applied(RemoteLaneValue::Ref(RemoteRef {
                            node: e.target.node,
                            entity: v.value,
                        })))
                    }
                    RemoteReceiveOutcome::Empty => Ok(RemoteLaneApply::WouldBlock),
                    RemoteReceiveOutcome::Closed => Ok(RemoteLaneApply::Closed),
                }
            }
            (RoutedResource::Object(c), RemoteLaneOperation::ObjectRead { offset, length }) => {
                let c = c.rebound(e.grant, e.epoch);
                let r = c.read(*offset, *length as usize).map_err(map_object)?;
                Ok(RemoteLaneApply::Applied(RemoteLaneValue::Bytes {
                    version: r.version,
                    bytes: r.bytes,
                }))
            }
            (
                RoutedResource::Object(c),
                RemoteLaneOperation::ObjectWrite {
                    expected_version,
                    offset,
                    bytes,
                },
            ) => {
                let c = c.rebound(e.grant, e.epoch);
                let r = c
                    .write(*expected_version, *offset, bytes)
                    .map_err(map_object)?;
                Ok(RemoteLaneApply::Applied(RemoteLaneValue::Version {
                    version: r.version,
                    byte_length: r.byte_length,
                }))
            }
            _ => Err(RemoteLaneError::Unsupported),
        }
    }
}
fn map_future(e: RemoteFutureError) -> RemoteLaneError {
    match e {
        RemoteFutureError::NodeUnavailable => RemoteLaneError::NodeUnavailable,
        RemoteFutureError::NodeLost => RemoteLaneError::NodeLost,
        RemoteFutureError::ProtocolError | RemoteFutureError::InvalidRequest => {
            RemoteLaneError::Protocol
        }
        RemoteFutureError::AuthorityDenied => RemoteLaneError::AuthorityDenied,
        RemoteFutureError::AlreadyResolved => RemoteLaneError::ApplyFailed,
    }
}
fn map_channel(e: RemoteChannelError) -> RemoteLaneError {
    match e {
        RemoteChannelError::NodeUnavailable => RemoteLaneError::NodeUnavailable,
        RemoteChannelError::NodeLost => RemoteLaneError::NodeLost,
        RemoteChannelError::ProtocolError | RemoteChannelError::InvalidRequest => {
            RemoteLaneError::Protocol
        }
        RemoteChannelError::AuthorityDenied => RemoteLaneError::AuthorityDenied,
        RemoteChannelError::InvalidSequence => RemoteLaneError::InvalidSequence,
    }
}
fn map_object(e: RemoteObjectError) -> RemoteLaneError {
    match e {
        RemoteObjectError::NodeUnavailable => RemoteLaneError::NodeUnavailable,
        RemoteObjectError::NodeLost => RemoteLaneError::NodeLost,
        RemoteObjectError::ProtocolError
        | RemoteObjectError::InvalidRequest
        | RemoteObjectError::FrameTooLarge => RemoteLaneError::Protocol,
        RemoteObjectError::AuthorityDenied => RemoteLaneError::AuthorityDenied,
        RemoteObjectError::StaleVersion { expected, actual } => {
            RemoteLaneError::StaleVersion { expected, actual }
        }
    }
}
