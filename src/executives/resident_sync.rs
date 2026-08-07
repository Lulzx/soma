//! Standalone reference implementation of the resident synchronization ABI.
//!
//! This is deliberately not a search or ping-pong kernel.  Installed handlers
//! are arbitrary bounded state machines: the executive gives a handler its
//! private frame and the results of its previous effects, and the handler emits
//! a bounded list of synchronization effects plus a scheduling disposition.
//! Future, mailbox, and bounded object storage is owned by the executive and is
//! only mutated by the canonical lane-order applier.  The module is the CPU oracle for a Metal
//! implementation; it does not claim `Kernel` integration.

use crate::scheduler::device::{DeviceLaneAccess, DEVICE_ACCESS_WRITE};
use crate::scheduler::device_ops::{
    DeviceLaneOperation, DeviceOperationJournal, OP_AWAIT_FUTURE, OP_ENQUEUE_MESSAGE,
    OP_READ_OBJECT, OP_RECEIVE_MESSAGE, OP_RESOLVE_FUTURE, OP_WRITE_OBJECT,
};
use std::collections::{BTreeMap, VecDeque};

pub const RESOURCE_OBJECT: u32 = 1;
pub const RESOURCE_FUTURE: u32 = 2;
pub const RESOURCE_MAILBOX: u32 = 3;
pub const RIGHT_READ: u32 = 1;
pub const RIGHT_WRITE: u32 = 2;
/// Hard device-plan bounds, checked before CPU cloning or Metal arena allocation.
pub const MAX_RESIDENT_OBJECTS: usize = 4_096;
pub const MAX_RESIDENT_OBJECT_CAPABILITIES: usize = 16_384;
pub const MAX_RESIDENT_OBJECT_ARENA_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentEffect {
    /// Exact bounded 8-byte little-endian object range read.
    ObjectRead {
        target: u32,
        offset: u32,
    },
    /// Exact bounded 8-byte little-endian in-place object range mutation.
    ObjectWrite {
        target: u32,
        offset: u32,
        value: u64,
    },
    /// Non-blocking governed LaneView::future_value observation.
    FutureObserve {
        target: u32,
    },
    FutureAwait {
        target: u32,
    },
    FutureResolve {
        target: u32,
        value: u64,
    },
    MailboxSend {
        target: u32,
        value: u64,
    },
    MailboxReceive {
        target: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentOutcome {
    ObjectRead(u64),
    ObjectWritten,
    Resolved(u64),
    /// A non-blocking future observation found the cell pending.
    Pending,
    Registered,
    Sent,
    Received {
        value: u64,
        sender: u64,
    },
    CapabilityDenied,
    InvalidTarget,
    Full,
    Empty,
    DoubleResolve,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentDisposition {
    Yield(u32),
    Complete,
}

/// Fixed-width, pointer-free resident handler instruction. Programs are
/// validated before any continuation is admitted; the device never executes
/// host function pointers or arbitrary Rust closures.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentInstruction {
    pub opcode: u32,
    /// Frame offset/skip/run class, or byte offset for object effects.
    pub argument: u32,
    /// Dense resource index for effects.
    pub target: u32,
    pub reserved: u32,
    pub value: u64,
}

pub const HANDLER_EFFECT_FUTURE_AWAIT: u32 = 1;
pub const HANDLER_EFFECT_FUTURE_RESOLVE: u32 = 2;
pub const HANDLER_EFFECT_MAILBOX_SEND: u32 = 3;
pub const HANDLER_EFFECT_MAILBOX_RECEIVE: u32 = 4;
pub const HANDLER_STORE_IMMEDIATE_U64: u32 = 5;
pub const HANDLER_STORE_PREVIOUS_VALUE_U64: u32 = 6;
/// Skip `argument` instructions when the most recent value-bearing previous
/// outcome is absent or is not equal to `value`.
pub const HANDLER_IF_PREVIOUS_VALUE_NE_SKIP: u32 = 7;
pub const HANDLER_YIELD: u32 = 8;
pub const HANDLER_COMPLETE: u32 = 9;
/// Emit the explicit non-blocking `OP_OBSERVE_FUTURE` ABI record.
pub const HANDLER_EFFECT_FUTURE_OBSERVE: u32 = 10;
pub const HANDLER_EFFECT_OBJECT_READ: u32 = 11;
pub const HANDLER_EFFECT_OBJECT_WRITE: u32 = 12;
/// Wrapping little-endian `u64` add in the continuation's private frame.
pub const HANDLER_ADD_FRAME_IMMEDIATE_U64: u32 = 13;
/// Complete when the little-endian frame word at `argument` equals `value`.
pub const HANDLER_COMPLETE_IF_FRAME_U64_EQ: u32 = 14;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentHandlerProgram {
    pub run_class: u32,
    pub instructions: Vec<ResidentInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandlerOutput {
    pub frame: Vec<u8>,
    pub effects: Vec<ResidentEffect>,
    pub disposition: ResidentDisposition,
}

fn previous_value(previous: &[ResidentOutcome]) -> Option<u64> {
    previous.iter().rev().find_map(|outcome| match *outcome {
        ResidentOutcome::Resolved(value)
        | ResidentOutcome::ObjectRead(value)
        | ResidentOutcome::Received { value, .. } => Some(value),
        _ => None,
    })
}

pub(crate) fn validate_handler_program(
    program: &ResidentHandlerProgram,
    max_effects: u32,
    max_frame_bytes: u32,
) -> bool {
    if program.instructions.is_empty() || program.instructions.len() > 256 {
        return false;
    }
    let blocking_effects = program
        .instructions
        .iter()
        .filter(|instruction| {
            matches!(
                instruction.opcode,
                HANDLER_EFFECT_FUTURE_AWAIT
                    | HANDLER_EFFECT_MAILBOX_SEND
                    | HANDLER_EFFECT_MAILBOX_RECEIVE
            )
        })
        .count();
    if blocking_effects > 1 {
        return false;
    }
    let mut pending = vec![(0usize, 0u32)];
    let mut seen = std::collections::BTreeSet::new();
    while let Some((pc, effects)) = pending.pop() {
        if !seen.insert((pc, effects)) || pc >= program.instructions.len() {
            if pc >= program.instructions.len() {
                return false;
            }
            continue;
        }
        let instruction = program.instructions[pc];
        match instruction.opcode {
            HANDLER_EFFECT_OBJECT_READ
            | HANDLER_EFFECT_OBJECT_WRITE
            | HANDLER_EFFECT_FUTURE_OBSERVE
            | HANDLER_EFFECT_FUTURE_AWAIT
            | HANDLER_EFFECT_FUTURE_RESOLVE
            | HANDLER_EFFECT_MAILBOX_SEND
            | HANDLER_EFFECT_MAILBOX_RECEIVE => {
                let next_effects = effects.saturating_add(1);
                if next_effects > max_effects {
                    return false;
                }
                pending.push((pc + 1, next_effects));
            }
            HANDLER_STORE_IMMEDIATE_U64
            | HANDLER_STORE_PREVIOUS_VALUE_U64
            | HANDLER_ADD_FRAME_IMMEDIATE_U64
            | HANDLER_COMPLETE_IF_FRAME_U64_EQ => {
                if (instruction.argument as usize)
                    .checked_add(8)
                    .is_none_or(|end| end > max_frame_bytes as usize)
                {
                    return false;
                }
                pending.push((pc + 1, effects));
            }
            HANDLER_IF_PREVIOUS_VALUE_NE_SKIP => {
                let skipped = pc
                    .saturating_add(1)
                    .saturating_add(instruction.argument as usize);
                if skipped > program.instructions.len() {
                    return false;
                }
                pending.push((pc + 1, effects));
                pending.push((skipped, effects));
            }
            HANDLER_YIELD | HANDLER_COMPLETE => {}
            _ => return false,
        }
    }
    true
}

fn execute_program(
    program: &ResidentHandlerProgram,
    frame: &[u8],
    previous: &[ResidentOutcome],
    max_effects: usize,
    max_frame_bytes: usize,
) -> Option<HandlerOutput> {
    if program.instructions.is_empty() || program.instructions.len() > 256 {
        return None;
    }
    let mut frame = frame.to_vec();
    let mut effects = Vec::new();
    let mut disposition = None;
    let mut pc = 0usize;
    while pc < program.instructions.len() {
        let instruction = program.instructions[pc];
        pc += 1;
        match instruction.opcode {
            HANDLER_EFFECT_OBJECT_READ => effects.push(ResidentEffect::ObjectRead {
                target: instruction.target,
                offset: instruction.argument,
            }),
            HANDLER_EFFECT_OBJECT_WRITE => effects.push(ResidentEffect::ObjectWrite {
                target: instruction.target,
                offset: instruction.argument,
                value: instruction.value,
            }),
            HANDLER_EFFECT_FUTURE_OBSERVE => effects.push(ResidentEffect::FutureObserve {
                target: instruction.argument,
            }),
            HANDLER_EFFECT_FUTURE_AWAIT => effects.push(ResidentEffect::FutureAwait {
                target: instruction.argument,
            }),
            HANDLER_EFFECT_FUTURE_RESOLVE => effects.push(ResidentEffect::FutureResolve {
                target: instruction.argument,
                value: instruction.value,
            }),
            HANDLER_EFFECT_MAILBOX_SEND => effects.push(ResidentEffect::MailboxSend {
                target: instruction.argument,
                value: instruction.value,
            }),
            HANDLER_EFFECT_MAILBOX_RECEIVE => effects.push(ResidentEffect::MailboxReceive {
                target: instruction.argument,
            }),
            HANDLER_STORE_IMMEDIATE_U64 | HANDLER_STORE_PREVIOUS_VALUE_U64 => {
                let at = instruction.argument as usize;
                let end = at.checked_add(8)?;
                if end > frame.len() || end > max_frame_bytes {
                    return None;
                }
                let value = if instruction.opcode == HANDLER_STORE_IMMEDIATE_U64 {
                    instruction.value
                } else {
                    previous_value(previous)?
                };
                frame[at..end].copy_from_slice(&value.to_le_bytes());
            }
            HANDLER_ADD_FRAME_IMMEDIATE_U64 => {
                let at = instruction.argument as usize;
                let end = at.checked_add(8)?;
                let word: [u8; 8] = frame.get(at..end)?.try_into().ok()?;
                let value = u64::from_le_bytes(word).wrapping_add(instruction.value);
                frame[at..end].copy_from_slice(&value.to_le_bytes());
            }
            HANDLER_COMPLETE_IF_FRAME_U64_EQ => {
                let at = instruction.argument as usize;
                let end = at.checked_add(8)?;
                let word: [u8; 8] = frame.get(at..end)?.try_into().ok()?;
                if u64::from_le_bytes(word) == instruction.value {
                    disposition = Some(ResidentDisposition::Complete);
                    break;
                }
            }
            HANDLER_IF_PREVIOUS_VALUE_NE_SKIP => {
                if previous_value(previous) != Some(instruction.value) {
                    pc = pc.checked_add(instruction.argument as usize)?;
                    if pc > program.instructions.len() {
                        return None;
                    }
                }
            }
            HANDLER_YIELD => {
                disposition = Some(ResidentDisposition::Yield(instruction.argument));
                break;
            }
            HANDLER_COMPLETE => {
                disposition = Some(ResidentDisposition::Complete);
                break;
            }
            _ => return None,
        }
        if effects.len() > max_effects {
            return None;
        }
    }
    Some(HandlerOutput {
        frame,
        effects,
        disposition: disposition?,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentContinuation {
    pub id: u64,
    pub actor: u64,
    pub run_class: u32,
    /// Whether this continuation claims mutable access to its actor process.
    pub mutable_access: bool,
    /// Absolute epoch used by deterministic longest-waiting admission.
    pub waiting_since: u32,
    pub frame: Vec<u8>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentCapability {
    pub actor: u64,
    pub resource_kind: u32,
    pub target: u32,
    pub rights: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialFuture {
    Pending,
    Resolved(u64),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentObject {
    pub version: u32,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentObjectCapability {
    pub actor: u64,
    pub target: u32,
    pub offset: u64,
    pub length: u64,
    pub rights: u32,
    pub object_version: u32,
    pub valid_until_epoch: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentSyncConfig {
    pub max_epochs: u32,
    pub max_effects_per_step: u32,
    /// Fixed frame-stride bound shared by the CPU oracle and Metal ABI.
    pub max_frame_bytes: u32,
    /// Bounds all continuation, waiter, and wake tables.
    pub max_continuations: u32,
    pub cohort_width: u32,
    pub initial_epoch: u32,
    pub objects: Vec<ResidentObject>,
    pub object_capabilities: Vec<ResidentObjectCapability>,
    pub futures: Vec<InitialFuture>,
    pub mailbox_capacities: Vec<u32>,
    /// Canonical FIFO `(sender, value)` entries present at the resident boundary.
    /// The outer vector is exactly parallel to `mailbox_capacities`.
    pub mailbox_messages: Vec<Vec<(u64, u64)>>,
    pub capabilities: Vec<ResidentCapability>,
}

pub(crate) fn object_storage_is_bounded(config: &ResidentSyncConfig) -> bool {
    if config.objects.len() > MAX_RESIDENT_OBJECTS
        || config.object_capabilities.len() > MAX_RESIDENT_OBJECT_CAPABILITIES
    {
        return false;
    }
    let actual_bytes = config.objects.iter().try_fold(0usize, |total, object| {
        total.checked_add(object.bytes.len())
    });
    let arena_bytes = config
        .objects
        .len()
        .checked_mul(config.max_frame_bytes as usize);
    actual_bytes.is_some_and(|bytes| bytes <= MAX_RESIDENT_OBJECT_ARENA_BYTES)
        && arena_bytes.is_some_and(|bytes| bytes <= MAX_RESIDENT_OBJECT_ARENA_BYTES)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentSyncTrace {
    pub epoch: u32,
    pub lane: u32,
    pub continuation: u64,
    pub run_class: u32,
    pub event: u32,
    pub word: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentEffectRecord {
    pub epoch: u32,
    pub lane: u32,
    pub ordinal: u32,
    pub continuation: u64,
    pub effect: ResidentEffect,
    pub outcome: ResidentOutcome,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentFinalContinuation {
    pub id: u64,
    pub run_class: u32,
    pub completed: bool,
    pub pending: Option<ResidentEffect>,
    /// Monotonic registration ticket used to reconstruct mailbox FIFO waiters.
    pub waiter_order: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentInvocationRecord {
    pub epoch: u32,
    pub lane: u32,
    pub continuation: u64,
    pub run_class: u32,
    /// 1 = Yield, 2 = Complete, 3 = Parked.
    pub disposition: u32,
    pub next_run_class: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentWakeRecord {
    pub epoch: u32,
    pub lane: u32,
    pub cause_opcode: u32,
    pub target: u32,
    pub cause_continuation: u64,
    pub continuation: u64,
    pub run_class: u32,
    pub ticket: u32,
    pub ordinal: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidentEpochRecord {
    pub epoch: u32,
    pub invocations: u32,
    pub runnable_after: u32,
    pub completed_after: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentSyncResult {
    pub frames: BTreeMap<u64, Vec<u8>>,
    pub effects: Vec<ResidentEffectRecord>,
    pub accesses: Vec<DeviceLaneAccess>,
    pub operations: Vec<DeviceOperationJournal>,
    pub trace: Vec<ResidentSyncTrace>,
    pub epochs: u32,
    pub quiescent: bool,
    pub completed: Vec<u64>,
    pub object_values: Vec<Vec<u8>>,
    pub future_values: Vec<Option<u64>>,
    pub mailboxes: Vec<Vec<(u64, u64)>>,
    pub final_continuations: Vec<ResidentFinalContinuation>,
    pub invocations: Vec<ResidentInvocationRecord>,
    pub wakes: Vec<ResidentWakeRecord>,
    pub epoch_records: Vec<ResidentEpochRecord>,
}

#[derive(Clone)]
struct Cont {
    spec: ResidentContinuation,
    previous: Vec<ResidentOutcome>,
    runnable: bool,
    completed: bool,
    pending: Option<ResidentEffect>,
    waiter_order: u32,
}
struct FutureCell {
    value: Option<u64>,
    waiters: Vec<u64>,
}
struct Mail {
    capacity: usize,
    queue: VecDeque<(u64, u64)>,
    receivers: Vec<u64>,
    senders: Vec<u64>,
}

fn opcode(e: ResidentEffect) -> u32 {
    match e {
        ResidentEffect::ObjectRead { .. } => OP_READ_OBJECT,
        ResidentEffect::ObjectWrite { .. } => OP_WRITE_OBJECT,
        ResidentEffect::FutureObserve { .. } => crate::scheduler::device_ops::OP_OBSERVE_FUTURE,
        ResidentEffect::FutureAwait { .. } => OP_AWAIT_FUTURE,
        ResidentEffect::FutureResolve { .. } => OP_RESOLVE_FUTURE,
        ResidentEffect::MailboxSend { .. } => OP_ENQUEUE_MESSAGE,
        ResidentEffect::MailboxReceive { .. } => OP_RECEIVE_MESSAGE,
    }
}
fn target(e: ResidentEffect) -> u32 {
    match e {
        ResidentEffect::ObjectRead { target, .. }
        | ResidentEffect::ObjectWrite { target, .. }
        | ResidentEffect::FutureObserve { target }
        | ResidentEffect::FutureAwait { target }
        | ResidentEffect::FutureResolve { target, .. }
        | ResidentEffect::MailboxSend { target, .. }
        | ResidentEffect::MailboxReceive { target } => target,
    }
}
fn resource(e: ResidentEffect) -> u32 {
    match e {
        ResidentEffect::ObjectRead { .. } | ResidentEffect::ObjectWrite { .. } => RESOURCE_OBJECT,
        ResidentEffect::FutureObserve { .. }
        | ResidentEffect::FutureAwait { .. }
        | ResidentEffect::FutureResolve { .. } => RESOURCE_FUTURE,
        _ => RESOURCE_MAILBOX,
    }
}
fn required_right(e: ResidentEffect) -> u32 {
    match e {
        ResidentEffect::ObjectRead { .. } => RIGHT_READ,
        ResidentEffect::ObjectWrite { .. } => RIGHT_WRITE,
        ResidentEffect::FutureObserve { .. }
        | ResidentEffect::FutureAwait { .. }
        | ResidentEffect::MailboxReceive { .. } => RIGHT_READ,
        _ => RIGHT_WRITE,
    }
}
fn outcome_code(o: ResidentOutcome) -> u32 {
    match o {
        ResidentOutcome::ObjectRead(_) => 2,
        ResidentOutcome::ObjectWritten => 0,
        ResidentOutcome::Resolved(_) => 2,
        ResidentOutcome::Pending => 1,
        ResidentOutcome::Registered => 3,
        ResidentOutcome::Sent => 0,
        ResidentOutcome::Received { .. } => 2,
        ResidentOutcome::CapabilityDenied => 0x104,
        ResidentOutcome::InvalidTarget => 0x101,
        ResidentOutcome::Full => 0x10c,
        ResidentOutcome::Empty => 1,
        ResidentOutcome::DoubleResolve => 0x111,
    }
}
fn hash(bytes: &[u8]) -> u32 {
    bytes.iter().fold(2_166_136_261, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(16_777_619)
    })
}

/// Run installed handlers and canonically apply their effects. `cohort_width`
/// affects packing only; changing it cannot change any returned semantic field.
pub fn run_resident_sync(
    config: &ResidentSyncConfig,
    continuations: Vec<ResidentContinuation>,
    handlers: &BTreeMap<u32, ResidentHandlerProgram>,
) -> Option<ResidentSyncResult> {
    let unique_ids = continuations
        .iter()
        .map(|continuation| continuation.id)
        .collect::<std::collections::BTreeSet<_>>();
    if unique_ids.len() != continuations.len()
        || config.max_epochs == 0
        || config.cohort_width == 0
        || config.max_effects_per_step == 0
        || config.max_frame_bytes == 0
        || config.max_continuations == 0
        || config
            .initial_epoch
            .checked_add(config.max_epochs)
            .is_none()
        || !object_storage_is_bounded(config)
        || config
            .objects
            .iter()
            .any(|object| object.bytes.len() > config.max_frame_bytes as usize)
        || config.object_capabilities.iter().any(|cap| {
            let Some(object) = config.objects.get(cap.target as usize) else {
                return true;
            };
            cap.object_version != object.version
                || cap
                    .offset
                    .checked_add(cap.length)
                    .is_none_or(|end| end > object.bytes.len() as u64)
        })
        || config.mailbox_messages.len() != config.mailbox_capacities.len()
        || config
            .mailbox_messages
            .iter()
            .zip(&config.mailbox_capacities)
            .any(|(messages, capacity)| messages.len() > *capacity as usize)
        || continuations.len() > config.max_continuations as usize
        || continuations.iter().any(|continuation| {
            continuation.frame.len() > config.max_frame_bytes as usize
                || !handlers.contains_key(&continuation.run_class)
        })
        || handlers.iter().any(|(run_class, program)| {
            *run_class != program.run_class
                || !validate_handler_program(
                    program,
                    config.max_effects_per_step,
                    config.max_frame_bytes,
                )
        })
    {
        return None;
    }
    let mut cs: BTreeMap<u64, Cont> = continuations
        .into_iter()
        .map(|s| {
            (
                s.id,
                Cont {
                    spec: s,
                    previous: vec![],
                    runnable: true,
                    completed: false,
                    pending: None,
                    waiter_order: 0,
                },
            )
        })
        .collect();
    let mut objects: Vec<Vec<u8>> = config.objects.iter().map(|o| o.bytes.clone()).collect();
    let mut fs: Vec<FutureCell> = config
        .futures
        .iter()
        .map(|s| FutureCell {
            value: match s {
                InitialFuture::Pending => None,
                InitialFuture::Resolved(v) => Some(*v),
            },
            waiters: vec![],
        })
        .collect();
    let mut ms: Vec<Mail> = config
        .mailbox_capacities
        .iter()
        .zip(&config.mailbox_messages)
        .map(|(capacity, messages)| Mail {
            capacity: *capacity as usize,
            queue: messages.iter().copied().collect(),
            receivers: vec![],
            senders: vec![],
        })
        .collect();
    let allowed = |actor: u64, kind: u32, t: u32, right: u32| {
        config.capabilities.iter().any(|c| {
            c.actor == actor && c.resource_kind == kind && c.target == t && (c.rights & right) != 0
        })
    };
    let object_allowed = |actor: u64, target: u32, offset: u32, right: u32, epoch: u32| {
        let Some(object) = config.objects.get(target as usize) else {
            return false;
        };
        let end = u64::from(offset).checked_add(8);
        config.object_capabilities.iter().any(|c| {
            c.actor == actor
                && c.target == target
                && (c.rights & right) != 0
                && c.object_version == object.version
                && c.valid_until_epoch >= epoch
                && u64::from(offset) >= c.offset
                && end.is_some_and(|end| end <= c.offset.saturating_add(c.length))
        })
    };
    let mut result = ResidentSyncResult::default();
    let mut waiter_order = 0u32;
    for epoch in 0..config.max_epochs {
        let mut mutable_winners: BTreeMap<u64, (u32, u64)> = BTreeMap::new();
        for continuation in cs
            .values()
            .filter(|c| c.runnable && !c.completed && c.spec.mutable_access)
        {
            let key = (continuation.spec.waiting_since, continuation.spec.id);
            mutable_winners
                .entry(continuation.spec.actor)
                .and_modify(|winner| {
                    if key < *winner {
                        *winner = key;
                    }
                })
                .or_insert(key);
        }
        let mut lanes: Vec<u64> = cs
            .values()
            .filter(|c| {
                c.runnable
                    && !c.completed
                    && (!c.spec.mutable_access
                        || mutable_winners
                            .get(&c.spec.actor)
                            .is_some_and(|winner| winner.1 == c.spec.id))
            })
            .map(|c| c.spec.id)
            .collect();
        if lanes.is_empty() {
            result.quiescent = true;
            break;
        }
        lanes.sort_by_key(|id| (cs[id].spec.run_class, *id));
        let mut emitted = Vec::new();
        for (li, id) in lanes.iter().enumerate() {
            let c = cs.get_mut(id).unwrap();
            c.runnable = false;
            c.spec.waiting_since = config.initial_epoch.saturating_add(epoch);
            let effects;
            let disposition;
            if let Some(retry) = c.pending.take() {
                effects = vec![retry];
                disposition = ResidentDisposition::Yield(c.spec.run_class);
            } else {
                let h = handlers.get(&c.spec.run_class)?;
                if h.run_class != c.spec.run_class {
                    return None;
                }
                let out = execute_program(
                    h,
                    &c.spec.frame,
                    &c.previous,
                    config.max_effects_per_step as usize,
                    config.max_frame_bytes as usize,
                )?;
                c.spec.frame = out.frame;
                effects = out.effects;
                disposition = out.disposition;
                c.previous.clear();
            }
            result.trace.push(ResidentSyncTrace {
                epoch,
                lane: li as u32,
                continuation: *id,
                run_class: c.spec.run_class,
                event: 0,
                word: hash(&c.spec.frame),
            });
            result.invocations.push(ResidentInvocationRecord {
                epoch,
                lane: li as u32,
                continuation: *id,
                run_class: c.spec.run_class,
                disposition: match disposition {
                    ResidentDisposition::Yield(_) => 1,
                    ResidentDisposition::Complete => 2,
                },
                next_run_class: match disposition {
                    ResidentDisposition::Yield(rc) => rc,
                    ResidentDisposition::Complete => 0,
                },
            });
            emitted.push((*id, li as u32, effects, disposition));
        }
        // This loop is the canonical applier. No handler runs while it mutates tables.
        for (id, lane, effects, disp) in emitted {
            let actor = cs[&id].spec.actor;
            let mut parked = false;
            let mut journal = DeviceOperationJournal::default();
            for (ord, e) in effects.into_iter().enumerate() {
                let t = target(e);
                let kind = resource(e);
                let right = required_right(e);
                let authorized = match e {
                    ResidentEffect::ObjectRead { offset, .. }
                    | ResidentEffect::ObjectWrite { offset, .. } => object_allowed(
                        actor,
                        t,
                        offset,
                        right,
                        config.initial_epoch.saturating_add(epoch),
                    ),
                    _ => allowed(actor, kind, t, right),
                };
                let outcome = if !authorized {
                    ResidentOutcome::CapabilityDenied
                } else {
                    match e {
                        ResidentEffect::ObjectRead { target, offset } => {
                            let start = offset as usize;
                            objects
                                .get(target as usize)
                                .and_then(|bytes| bytes.get(start..start.checked_add(8)?))
                                .map_or(ResidentOutcome::InvalidTarget, |bytes| {
                                    ResidentOutcome::ObjectRead(u64::from_le_bytes(
                                        bytes.try_into().unwrap(),
                                    ))
                                })
                        }
                        ResidentEffect::ObjectWrite {
                            target,
                            offset,
                            value,
                        } => {
                            let start = offset as usize;
                            match objects
                                .get_mut(target as usize)
                                .and_then(|bytes| bytes.get_mut(start..start.checked_add(8)?))
                            {
                                Some(bytes) => {
                                    bytes.copy_from_slice(&value.to_le_bytes());
                                    ResidentOutcome::ObjectWritten
                                }
                                None => ResidentOutcome::InvalidTarget,
                            }
                        }
                        ResidentEffect::FutureObserve { target } => match fs.get(target as usize) {
                            None => ResidentOutcome::InvalidTarget,
                            Some(f) => f
                                .value
                                .map_or(ResidentOutcome::Pending, ResidentOutcome::Resolved),
                        },
                        ResidentEffect::FutureAwait { target } => match fs.get_mut(target as usize)
                        {
                            None => ResidentOutcome::InvalidTarget,
                            Some(f) => {
                                if let Some(v) = f.value {
                                    ResidentOutcome::Resolved(v)
                                } else {
                                    if !f.waiters.contains(&id) {
                                        f.waiters.push(id)
                                    }
                                    cs.get_mut(&id).unwrap().pending = Some(e);
                                    parked = true;
                                    waiter_order = waiter_order.saturating_add(1);
                                    cs.get_mut(&id).unwrap().waiter_order = waiter_order;
                                    ResidentOutcome::Registered
                                }
                            }
                        },
                        ResidentEffect::FutureResolve { target, value } => {
                            match fs.get_mut(target as usize) {
                                None => ResidentOutcome::InvalidTarget,
                                Some(f) => {
                                    if f.value.is_some() {
                                        ResidentOutcome::DoubleResolve
                                    } else {
                                        f.value = Some(value);
                                        for w in std::mem::take(&mut f.waiters) {
                                            if let Some(c) = cs.get_mut(&w) {
                                                result.wakes.push(ResidentWakeRecord {
                                                    epoch,
                                                    lane,
                                                    cause_opcode: OP_RESOLVE_FUTURE,
                                                    target,
                                                    continuation: w,
                                                    run_class: c.spec.run_class,
                                                    ticket: c.waiter_order,
                                                    ordinal: ord as u32,
                                                    cause_continuation: id,
                                                    reserved: 0,
                                                });
                                                c.runnable = true;
                                                c.pending = None;
                                                c.previous.push(ResidentOutcome::Resolved(value));
                                            }
                                        }
                                        ResidentOutcome::Resolved(value)
                                    }
                                }
                            }
                        }
                        ResidentEffect::MailboxSend { target, value } => {
                            match ms.get_mut(target as usize) {
                                None => ResidentOutcome::InvalidTarget,
                                Some(m) => {
                                    if m.queue.len() >= m.capacity {
                                        if !m.senders.contains(&id) {
                                            m.senders.push(id)
                                        }
                                        cs.get_mut(&id).unwrap().pending = Some(e);
                                        parked = true;
                                        waiter_order = waiter_order.saturating_add(1);
                                        cs.get_mut(&id).unwrap().waiter_order = waiter_order;
                                        ResidentOutcome::Full
                                    } else {
                                        m.queue.push_back((actor, value));
                                        if let Some(w) = m.receivers.first().copied() {
                                            m.receivers.remove(0);
                                            let c = cs.get_mut(&w).unwrap();
                                            result.wakes.push(ResidentWakeRecord {
                                                epoch,
                                                lane,
                                                cause_opcode: OP_ENQUEUE_MESSAGE,
                                                target,
                                                continuation: w,
                                                run_class: c.spec.run_class,
                                                ticket: c.waiter_order,
                                                ordinal: ord as u32,
                                                cause_continuation: id,
                                                reserved: 0,
                                            });
                                            c.runnable = true;
                                        }
                                        ResidentOutcome::Sent
                                    }
                                }
                            }
                        }
                        ResidentEffect::MailboxReceive { target } => {
                            match ms.get_mut(target as usize) {
                                None => ResidentOutcome::InvalidTarget,
                                Some(m) => {
                                    if let Some((sender, value)) = m.queue.pop_front() {
                                        if let Some(w) = m.senders.first().copied() {
                                            m.senders.remove(0);
                                            let c = cs.get_mut(&w).unwrap();
                                            result.wakes.push(ResidentWakeRecord {
                                                epoch,
                                                lane,
                                                cause_opcode: OP_RECEIVE_MESSAGE,
                                                target,
                                                continuation: w,
                                                run_class: c.spec.run_class,
                                                ticket: c.waiter_order,
                                                ordinal: ord as u32,
                                                cause_continuation: id,
                                                reserved: 0,
                                            });
                                            c.runnable = true;
                                        }
                                        ResidentOutcome::Received { value, sender }
                                    } else {
                                        if !m.receivers.contains(&id) {
                                            m.receivers.push(id)
                                        }
                                        cs.get_mut(&id).unwrap().pending = Some(e);
                                        parked = true;
                                        waiter_order = waiter_order.saturating_add(1);
                                        cs.get_mut(&id).unwrap().waiter_order = waiter_order;
                                        ResidentOutcome::Empty
                                    }
                                }
                            }
                        }
                    }
                };
                result.accesses.push(DeviceLaneAccess::new(
                    lane,
                    kind,
                    u64::from(t),
                    if matches!(
                        e,
                        ResidentEffect::FutureObserve { .. } | ResidentEffect::ObjectRead { .. }
                    ) {
                        crate::scheduler::device::DEVICE_ACCESS_READ
                    } else {
                        DEVICE_ACCESS_WRITE
                    },
                    ord as u32,
                ));
                let (value, aux) = match e {
                    ResidentEffect::ObjectWrite { offset, value, .. } => (value, u64::from(offset)),
                    ResidentEffect::ObjectRead { offset, .. } => (0, u64::from(offset)),
                    ResidentEffect::FutureResolve { value, .. }
                    | ResidentEffect::MailboxSend { value, .. } => (value, 0),
                    _ => (0, 0),
                };
                let payload = match outcome {
                    ResidentOutcome::ObjectRead(value) => value.to_le_bytes().to_vec(),
                    ResidentOutcome::ObjectWritten => value.to_le_bytes().to_vec(),
                    _ => Vec::new(),
                };
                let (payload_offset, payload_len) = journal.append_payload(&payload)?;
                journal.operations.push(DeviceLaneOperation {
                    lane,
                    ordinal: ord as u32,
                    opcode: opcode(e),
                    actor,
                    target: u64::from(t),
                    value,
                    auxiliary: aux,
                    result_code: outcome_code(outcome),
                    payload_offset,
                    payload_len,
                    result_ref: match outcome {
                        ResidentOutcome::ObjectRead(v) | ResidentOutcome::Resolved(v) => v,
                        ResidentOutcome::Received { value, .. } => value,
                        _ => 0,
                    },
                    ..Default::default()
                });
                result.effects.push(ResidentEffectRecord {
                    epoch,
                    lane,
                    ordinal: ord as u32,
                    continuation: id,
                    effect: e,
                    outcome,
                });
                result.trace.push(ResidentSyncTrace {
                    epoch,
                    lane,
                    continuation: id,
                    run_class: cs[&id].spec.run_class,
                    event: opcode(e),
                    word: outcome_code(outcome),
                });
                if !matches!(
                    outcome,
                    ResidentOutcome::Registered | ResidentOutcome::Full | ResidentOutcome::Empty
                ) {
                    cs.get_mut(&id).unwrap().previous.push(outcome)
                }
            }
            result.operations.push(journal);
            if parked {
                if let Some(invocation) = result
                    .invocations
                    .iter_mut()
                    .rev()
                    .find(|invocation| invocation.epoch == epoch && invocation.lane == lane)
                {
                    invocation.disposition = 3;
                    invocation.next_run_class = cs[&id].spec.run_class;
                }
            }
            let c = cs.get_mut(&id).unwrap();
            if !parked {
                match disp {
                    ResidentDisposition::Yield(rc) => {
                        c.spec.run_class = rc;
                        c.runnable = true
                    }
                    ResidentDisposition::Complete => {
                        c.completed = true;
                        result.completed.push(id)
                    }
                }
            }
        }
        result.epochs = epoch + 1;
        result.epoch_records.push(ResidentEpochRecord {
            epoch,
            invocations: lanes.len() as u32,
            runnable_after: cs.values().filter(|c| c.runnable && !c.completed).count() as u32,
            completed_after: cs.values().filter(|c| c.completed).count() as u32,
        });
    }
    result.quiescent = cs.values().all(|c| !c.runnable || c.completed);
    result.final_continuations = cs
        .values()
        .map(|c| ResidentFinalContinuation {
            id: c.spec.id,
            run_class: c.spec.run_class,
            completed: c.completed,
            pending: c.pending,
            waiter_order: c.pending.map_or(0, |_| c.waiter_order),
        })
        .collect();
    result.frames = cs.into_iter().map(|(id, c)| (id, c.spec.frame)).collect();
    result.object_values = objects;
    result.future_values = fs.into_iter().map(|f| f.value).collect();
    result.mailboxes = ms
        .into_iter()
        .map(|m| m.queue.into_iter().collect())
        .collect();
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cap(actor: u64, kind: u32, target: u32, rights: u32) -> ResidentCapability {
        ResidentCapability {
            actor,
            resource_kind: kind,
            target,
            rights,
        }
    }
    fn future_case(width: u32) -> ResidentSyncResult {
        let cfg = ResidentSyncConfig {
            max_epochs: 8,
            max_effects_per_step: 2,
            max_frame_bytes: 8,
            max_continuations: 2,
            cohort_width: width,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![
                cap(10, RESOURCE_FUTURE, 0, RIGHT_READ),
                cap(20, RESOURCE_FUTURE, 0, RIGHT_WRITE),
            ],
        };
        let mut hs: BTreeMap<u32, ResidentHandlerProgram> = BTreeMap::new();
        hs.insert(
            100,
            ResidentHandlerProgram {
                run_class: 100,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        target: 0,
                        reserved: 0,
                        value: 77,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_AWAIT,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 100,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        hs.insert(
            200,
            ResidentHandlerProgram {
                run_class: 200,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 77,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        run_resident_sync(
            &cfg,
            vec![
                ResidentContinuation {
                    id: 1,
                    actor: 10,
                    run_class: 100,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![0; 8],
                },
                ResidentContinuation {
                    id: 2,
                    actor: 20,
                    run_class: 200,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
            ],
            &hs,
        )
        .unwrap()
    }
    #[test]
    fn width_1_and_32_are_i19_identical() {
        let a = future_case(1);
        let b = future_case(32);
        assert_eq!(a, b);
        assert_eq!(a.future_values, vec![Some(77)]);
        assert_eq!(a.completed, vec![2, 1]);
        assert_eq!(
            a.effects.iter().map(|e| e.outcome).collect::<Vec<_>>(),
            vec![ResidentOutcome::Registered, ResidentOutcome::Resolved(77)]
        );
    }
    #[test]
    fn successful_mailbox_send_wakes_and_delivers_to_parked_receiver() {
        let cfg = ResidentSyncConfig {
            max_epochs: 8,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 2,
            cohort_width: 32,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![],
            mailbox_capacities: vec![1],
            mailbox_messages: vec![vec![]],
            capabilities: vec![
                cap(10, RESOURCE_MAILBOX, 0, RIGHT_READ),
                cap(20, RESOURCE_MAILBOX, 0, RIGHT_WRITE),
            ],
        };
        let mut programs = BTreeMap::new();
        programs.insert(
            100,
            ResidentHandlerProgram {
                run_class: 100,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        target: 0,
                        reserved: 0,
                        value: 55,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_RECEIVE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 100,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            200,
            ResidentHandlerProgram {
                run_class: 200,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 55,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        let result = run_resident_sync(
            &cfg,
            vec![
                ResidentContinuation {
                    id: 1,
                    actor: 10,
                    run_class: 100,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![0; 8],
                },
                ResidentContinuation {
                    id: 2,
                    actor: 20,
                    run_class: 200,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
            ],
            &programs,
        )
        .unwrap();
        assert_eq!(result.completed, vec![2, 1]);
        assert_eq!(result.frames[&1], 55u64.to_le_bytes());
        assert!(result.mailboxes[0].is_empty());
        assert_eq!(
            result
                .effects
                .iter()
                .map(|record| record.outcome)
                .collect::<Vec<_>>(),
            vec![
                ResidentOutcome::Empty,
                ResidentOutcome::Sent,
                ResidentOutcome::Received {
                    value: 55,
                    sender: 20
                },
            ]
        );
    }

    #[test]
    fn handler_previous_is_immutable_during_step_and_cleared_after_consumption() {
        let cfg = ResidentSyncConfig {
            max_epochs: 6,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 1,
            cohort_width: 1,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![cap(7, RESOURCE_FUTURE, 0, RIGHT_WRITE)],
        };
        let mut programs = BTreeMap::new();
        // The resolve emitted here must not make the following IF true in the
        // same invocation. Its outcome only becomes class 2's next input.
        programs.insert(
            1,
            ResidentHandlerProgram {
                run_class: 1,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 9,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        target: 0,
                        reserved: 0,
                        value: 9,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_IMMEDIATE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 99,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 2,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        // Consume value 9 once, then yield. The following invocation must see
        // an empty previous list and take the COMPLETE branch.
        programs.insert(
            2,
            ResidentHandlerProgram {
                run_class: 2,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        target: 0,
                        reserved: 0,
                        value: 9,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_IMMEDIATE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 9,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 2,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        let result = run_resident_sync(
            &cfg,
            vec![ResidentContinuation {
                id: 1,
                actor: 7,
                run_class: 1,
                mutable_access: false,
                waiting_since: 0,
                frame: vec![0; 8],
            }],
            &programs,
        )
        .unwrap();
        assert_eq!(result.epochs, 3);
        assert_eq!(result.completed, vec![1]);
        assert_eq!(result.frames[&1], 9u64.to_le_bytes());
        assert_eq!(result.effects.len(), 1);
    }

    #[test]
    fn rejects_programs_with_multiple_potential_parks() {
        let program = ResidentHandlerProgram {
            run_class: 1,
            instructions: vec![
                ResidentInstruction {
                    opcode: HANDLER_EFFECT_FUTURE_AWAIT,
                    argument: 0,
                    target: 0,
                    reserved: 0,
                    value: 0,
                },
                ResidentInstruction {
                    opcode: HANDLER_EFFECT_MAILBOX_RECEIVE,
                    argument: 0,
                    target: 0,
                    reserved: 0,
                    value: 0,
                },
                ResidentInstruction {
                    opcode: HANDLER_COMPLETE,
                    argument: 0,
                    target: 0,
                    reserved: 0,
                    value: 0,
                },
            ],
        };
        assert!(!validate_handler_program(&program, 2, 8));
    }

    fn observe_case(width: u32) -> ResidentSyncResult {
        let config = ResidentSyncConfig {
            max_epochs: 1,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 4,
            cohort_width: width,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending, InitialFuture::Resolved(44)],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![
                cap(7, RESOURCE_FUTURE, 0, RIGHT_READ),
                cap(7, RESOURCE_FUTURE, 1, RIGHT_READ),
                // Authority is checked before bounds, making target 9 an
                // explicit governed invalid-target outcome rather than denial.
                cap(7, RESOURCE_FUTURE, 9, RIGHT_READ),
            ],
        };
        let mut programs = BTreeMap::new();
        for (run_class, target) in [(10, 0), (11, 1), (12, 0), (13, 9)] {
            programs.insert(
                run_class,
                ResidentHandlerProgram {
                    run_class,
                    instructions: vec![
                        ResidentInstruction {
                            opcode: HANDLER_EFFECT_FUTURE_OBSERVE,
                            argument: target,
                            target: 0,
                            reserved: 0,
                            value: 0,
                        },
                        ResidentInstruction {
                            opcode: HANDLER_COMPLETE,
                            argument: 0,
                            target: 0,
                            reserved: 0,
                            value: 0,
                        },
                    ],
                },
            );
        }
        run_resident_sync(
            &config,
            vec![
                ResidentContinuation {
                    id: 1,
                    actor: 7,
                    run_class: 10,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
                ResidentContinuation {
                    id: 2,
                    actor: 7,
                    run_class: 11,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
                ResidentContinuation {
                    id: 3,
                    actor: 8,
                    run_class: 12,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
                ResidentContinuation {
                    id: 4,
                    actor: 7,
                    run_class: 13,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
            ],
            &programs,
        )
        .unwrap()
    }

    #[test]
    fn observe_is_bounded_governed_and_has_exact_read_journals_width_1_32() {
        let narrow = observe_case(1);
        let wide = observe_case(32);
        assert_eq!(narrow, wide);
        assert_eq!(
            narrow
                .effects
                .iter()
                .map(|effect| effect.outcome)
                .collect::<Vec<_>>(),
            vec![
                ResidentOutcome::Pending,
                ResidentOutcome::Resolved(44),
                ResidentOutcome::CapabilityDenied,
                ResidentOutcome::InvalidTarget,
            ]
        );
        assert!(narrow
            .effects
            .iter()
            .all(|effect| { matches!(effect.effect, ResidentEffect::FutureObserve { .. }) }));
        assert!(narrow.accesses.iter().all(|access| {
            access.resource_kind == RESOURCE_FUTURE
                && access.mode == crate::scheduler::device::DEVICE_ACCESS_READ
        }));
        let operations = narrow
            .operations
            .iter()
            .flat_map(|journal| &journal.operations)
            .collect::<Vec<_>>();
        assert_eq!(operations.len(), 4);
        assert!(operations.iter().all(|operation| {
            operation.opcode == crate::scheduler::device_ops::OP_OBSERVE_FUTURE
        }));
        assert_eq!(
            operations
                .iter()
                .map(|operation| operation.result_code)
                .collect::<Vec<_>>(),
            vec![1, 2, 0x104, 0x101]
        );
        assert_eq!(
            narrow
                .trace
                .iter()
                .map(|entry| (entry.event, entry.word))
                .collect::<Vec<_>>(),
            vec![
                (0, 2_166_136_261),
                (0, 2_166_136_261),
                (0, 2_166_136_261),
                (0, 2_166_136_261),
                (crate::scheduler::device_ops::OP_OBSERVE_FUTURE, 1),
                (crate::scheduler::device_ops::OP_OBSERVE_FUTURE, 2),
                (crate::scheduler::device_ops::OP_OBSERVE_FUTURE, 0x104),
                (crate::scheduler::device_ops::OP_OBSERVE_FUTURE, 0x101),
            ]
        );
        assert!(narrow
            .final_continuations
            .iter()
            .all(|continuation| { continuation.completed && continuation.pending.is_none() }));
    }

    #[test]
    fn independently_expected_error_outcomes_and_exact_journals() {
        let cfg = ResidentSyncConfig {
            max_epochs: 2,
            max_effects_per_step: 3,
            max_frame_bytes: 8,
            max_continuations: 3,
            cohort_width: 1,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Resolved(1)],
            mailbox_capacities: vec![0],
            mailbox_messages: vec![vec![]],
            capabilities: vec![
                cap(7, RESOURCE_FUTURE, 0, RIGHT_WRITE),
                cap(7, RESOURCE_FUTURE, 9, RIGHT_WRITE),
                cap(7, RESOURCE_MAILBOX, 0, RIGHT_READ | RIGHT_WRITE),
            ],
        };
        let mut programs = BTreeMap::new();
        programs.insert(
            9,
            ResidentHandlerProgram {
                run_class: 9,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 2,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 9,
                        target: 0,
                        reserved: 0,
                        value: 2,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_AWAIT,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            10,
            ResidentHandlerProgram {
                run_class: 10,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_RECEIVE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            11,
            ResidentHandlerProgram {
                run_class: 11,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 3,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        let result = run_resident_sync(
            &cfg,
            vec![
                ResidentContinuation {
                    id: 4,
                    actor: 7,
                    run_class: 9,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![1],
                },
                ResidentContinuation {
                    id: 5,
                    actor: 7,
                    run_class: 10,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
                ResidentContinuation {
                    id: 6,
                    actor: 7,
                    run_class: 11,
                    mutable_access: false,
                    waiting_since: 0,
                    frame: vec![],
                },
            ],
            &programs,
        )
        .unwrap();
        assert_eq!(
            result.effects.iter().map(|e| e.outcome).collect::<Vec<_>>(),
            vec![
                ResidentOutcome::DoubleResolve,
                ResidentOutcome::InvalidTarget,
                ResidentOutcome::CapabilityDenied,
                ResidentOutcome::Empty,
                ResidentOutcome::Full,
            ]
        );
        assert_eq!(
            result
                .operations
                .iter()
                .flat_map(|j| j.operations.iter().map(|o| o.opcode))
                .collect::<Vec<_>>(),
            vec![
                OP_RESOLVE_FUTURE,
                OP_RESOLVE_FUTURE,
                OP_AWAIT_FUTURE,
                OP_RECEIVE_MESSAGE,
                OP_ENQUEUE_MESSAGE,
            ]
        );
        assert_eq!(
            result
                .accesses
                .iter()
                .map(|access| access.ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 0, 0]
        );
        assert_eq!(
            result
                .trace
                .iter()
                .take(3)
                .map(|entry| entry.event)
                .collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
    }
}
