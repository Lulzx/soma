//! Kernel: the tables and operations that make SOMA's processes, continuations,
//! messages, and futures real (§7, §8, §11, §12). Phase 1 is single-threaded and
//! deterministic; every table is generation-checked (§4).

pub mod commit;
pub mod epochs;
pub mod ownership;

use std::collections::{HashMap, VecDeque};

use crate::abi::capabilities::CapabilityEntry;
use crate::abi::{
    AbiError, EventKind, FutureDescriptor, FutureState, Kind, MessageDescriptor, ObjectDescriptor,
    ObjectKind, ProcessDescriptor, ProcessMode, ProcessState, Ref64, TraceEvent,
};
use crate::scheduler::runnable_bins::Scheduler;
use crate::table::GenTable;

/// Runtime errors raised by kernel operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeError {
    Abi(AbiError),
    MissingPayload,
    MissingMailbox,
    MailboxFull,
    AlreadyResolved,
    NotResolved,
}

impl From<AbiError> for RuntimeError {
    fn from(e: AbiError) -> Self {
        RuntimeError::Abi(e)
    }
}

/// A bounded, ordered mailbox belonging to one process (§11).
#[derive(Debug, Default)]
pub struct Mailbox {
    /// In-flight messages, oldest first. Ordered per sender–receiver pair via
    /// `sender_sequence`.
    pub entries: VecDeque<MessageDescriptor>,
    pub capacity: usize,
    /// Continuations waiting for a message to arrive (FIFO).
    pub recv_waiters: VecDeque<Ref64>,
    /// Continuations waiting for capacity (a full mailbox), oldest first. One is
    /// woken per message received, since a receive frees exactly one slot.
    /// This must be a queue, not a single slot: with `Option`, a second blocked
    /// sender would never be registered and would park forever (§11).
    pub full_waiters: VecDeque<Ref64>,
}

impl Mailbox {
    pub fn new(capacity: usize) -> Mailbox {
        Mailbox {
            entries: VecDeque::new(),
            capacity,
            recv_waiters: VecDeque::new(),
            full_waiters: VecDeque::new(),
        }
    }
}

/// Outcome of registering a continuation on a future (§12).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AwaitOutcome {
    /// The future was pending; the continuation is now a waiter and will be
    /// woken by `resolve_future`. The caller should return `StepResult::await_on`.
    Registered,
    /// The future was already resolved, so there is nothing left to wait for.
    /// The caller should return `StepResult::yield_next` instead — registering
    /// would park the continuation forever, since `resolve_future` drains its
    /// waiter list exactly once.
    AlreadyResolved,
}

/// One comparable row of a trace snapshot (§21). Two runs are identical exactly
/// when their snapshot vectors are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceSnapshotRow {
    pub logical_time: u64,
    pub epoch: u32,
    pub event_kind: EventKind,
    pub engine: u16,
    pub process: u64,
    pub continuation: u64,
    pub run_class: u32,
    pub auxiliary: u32,
}

/// The whole kernel state: every table, the object payloads, mailboxes, the
/// scheduler, and the deterministic event trace (§21).
#[derive(Debug)]
pub struct Kernel {
    pub epoch: u32,
    pub logical_time: u64,
    pub trace: Vec<TraceEvent>,
    /// Total runnable continuations at the end of each epoch, for accounting.
    pub epoch_runnable: Vec<usize>,

    pub processes: GenTable<ProcessDescriptor>,
    pub objects: GenTable<ObjectDescriptor>,
    pub capabilities: GenTable<CapabilityEntry>,
    pub continuations: crate::table::GenTable<crate::abi::continuations::ContinuationDescriptor>,
    pub futures: GenTable<FutureDescriptor>,

    /// Object payload bytes, keyed by object slot. Kernel-private (§6: user
    /// programs cannot inspect or construct the physical mapping).
    pub object_payloads: HashMap<u32, Vec<u8>>,
    /// Mailboxes keyed by process slot.
    pub mailboxes: HashMap<u32, Mailbox>,
    /// Future waiters keyed by future slot.
    pub future_waiters: HashMap<u32, Vec<Ref64>>,
    /// Per (sender, receiver) pair, the next `sender_sequence` value (§11).
    pub send_sequences: HashMap<(u32, u32), u64>,

    pub scheduler: Scheduler,
}

impl Kernel {
    pub fn new() -> Kernel {
        Kernel {
            epoch: 0,
            logical_time: 0,
            trace: Vec::new(),
            epoch_runnable: Vec::new(),
            processes: GenTable::new(Kind::Process),
            objects: GenTable::new(Kind::Object),
            capabilities: GenTable::new(Kind::Capability),
            continuations: GenTable::new(Kind::Continuation),
            futures: GenTable::new(Kind::Future),
            object_payloads: HashMap::new(),
            mailboxes: HashMap::new(),
            future_waiters: HashMap::new(),
            send_sequences: HashMap::new(),
            scheduler: Scheduler::default(),
        }
    }

    // ---- trace -----------------------------------------------------------

    /// Append a trace event, advancing the logical clock deterministically.
    pub fn trace(
        &mut self,
        event_kind: EventKind,
        process: Ref64,
        continuation: Ref64,
        run_class: u32,
        auxiliary: u32,
    ) {
        self.logical_time = self.logical_time.wrapping_add(1);
        self.trace.push(TraceEvent::new(
            self.logical_time,
            self.epoch,
            event_kind,
            process,
            continuation,
            run_class,
        ));
        if let Some(last) = self.trace.last_mut() {
            last.auxiliary = auxiliary;
        }
    }

    /// Compact, comparable snapshot of the trace for determinism tests.
    ///
    /// Every field that can diverge between two runs is included — notably
    /// `epoch` and `auxiliary`, so a run that produces the same events in the
    /// same order but on different epoch boundaries is still detected as
    /// divergent (§21).
    pub fn trace_snapshot(&self) -> Vec<TraceSnapshotRow> {
        self.trace
            .iter()
            .map(|t| TraceSnapshotRow {
                logical_time: t.logical_time,
                epoch: t.epoch,
                event_kind: t.event_kind,
                engine: t.engine,
                process: t.process.to_u64(),
                continuation: t.continuation.to_u64(),
                run_class: t.run_class,
                auxiliary: t.auxiliary,
            })
            .collect()
    }

    // ---- objects ---------------------------------------------------------

    pub fn create_object(&mut self, kind: ObjectKind, bytes: Vec<u8>) -> Ref64 {
        let r = self.objects.alloc(ObjectDescriptor::new(kind, bytes.len() as u64));
        self.object_payloads.insert(r.slot, bytes);
        {
            let o = self.objects.get_mut(r).expect("fresh object");
            o.id = r;
        }
        r
    }

    pub fn object_bytes(&self, obj: Ref64) -> Result<&[u8], RuntimeError> {
        let _ = self.objects.get(obj)?;
        self.object_payloads
            .get(&obj.slot)
            .map(|v| v.as_slice())
            .ok_or(RuntimeError::MissingPayload)
    }

    pub fn object_bytes_mut(&mut self, obj: Ref64) -> Result<&mut Vec<u8>, RuntimeError> {
        let _ = self.objects.get(obj)?;
        self.object_payloads
            .get_mut(&obj.slot)
            .ok_or(RuntimeError::MissingPayload)
    }

    /// Read the first 8 bytes of an object's payload as a little-endian u64.
    pub fn read_u64_object(&self, obj: Ref64) -> Option<u64> {
        let b = self.object_bytes(obj).ok()?;
        if b.len() < 8 {
            return None;
        }
        let arr: [u8; 8] = b[..8].try_into().ok()?;
        Some(u64::from_le_bytes(arr))
    }

    // ---- processes -------------------------------------------------------

    pub fn create_process(&mut self, mode: ProcessMode) -> Ref64 {
        let r = self.processes.alloc(ProcessDescriptor::new(mode));
        let state_obj = self.create_object(ObjectKind::ProcessState, Vec::new());
        self.mailboxes.insert(r.slot, Mailbox::new(8));
        {
            let p = self.processes.get_mut(r).expect("fresh process");
            p.id = r;
            p.domain = r;
            p.state = state_obj;
            p.inbox = r;
            p.status = ProcessState::Created as u32;
        }
        self.trace(EventKind::ProcessCreated, r, Ref64::NULL, 0, mode as u32);
        r
    }

    pub fn process_state(&self, p: Ref64) -> Result<ProcessState, RuntimeError> {
        let pd = self.processes.get(p)?;
        Ok(match pd.status {
            1 => ProcessState::Created,
            2 => ProcessState::Runnable,
            3 => ProcessState::Running,
            4 => ProcessState::Waiting,
            5 => ProcessState::CancelPending,
            6 => ProcessState::Failed,
            7 => ProcessState::Terminated,
            _ => ProcessState::Created,
        })
    }

    // ---- continuations ---------------------------------------------------

    /// Allocate a continuation with a fresh frame object and make it runnable in
    /// its run-class bin. The child is scheduled for the next epoch.
    pub fn create_continuation(
        &mut self,
        process: Ref64,
        run_class: u32,
        resume_point: u32,
        frame_bytes: Vec<u8>,
        max_steps: u32,
    ) -> Ref64 {
        let frame_obj = self.create_object(ObjectKind::ContinuationFrame, frame_bytes);
        let r = self.continuations.alloc(
            crate::abi::continuations::ContinuationDescriptor::new(process, run_class, resume_point),
        );
        {
            let c = self.continuations.get_mut(r).expect("fresh continuation");
            c.id = r;
            c.frame = frame_obj;
            c.remaining_steps = max_steps;
            c.status = crate::abi::continuations::ContinuationState::Runnable;
            c.created_epoch = self.epoch;
        }
        self.scheduler.enqueue(run_class, r);
        self.trace(EventKind::ContinuationReady, process, r, run_class, 0);
        r
    }

    pub fn continuation_state(
        &self,
        c: Ref64,
    ) -> Result<crate::abi::continuations::ContinuationState, RuntimeError> {
        let cd = self.continuations.get(c)?;
        Ok(cd.status)
    }

    // ---- futures ---------------------------------------------------------

    pub fn create_future(&mut self) -> Ref64 {
        let r = self.futures.alloc(FutureDescriptor::new());
        {
            let f = self.futures.get_mut(r).expect("fresh future");
            f.id = r;
            f.owner_domain = r;
        }
        r
    }

    /// Register `cont` as a waiter on `future`, moving it to run class
    /// `next_run_class` and the WAITING state. It is woken by `resolve_future`.
    ///
    /// If the future is *already* resolved, nothing is registered and
    /// `AlreadyResolved` is returned: the caller must yield rather than await,
    /// because `resolve_future` has already drained the waiter list and will
    /// never revisit it.
    pub fn await_future(
        &mut self,
        cont: Ref64,
        future: Ref64,
        next_run_class: u32,
    ) -> Result<AwaitOutcome, RuntimeError> {
        let resolved = self.futures.get(future)?.state != FutureState::Pending;
        {
            let c = self.continuations.get_mut(cont)?;
            c.run_class = next_run_class;
            c.dependency = future;
            if !resolved {
                c.status = crate::abi::continuations::ContinuationState::Waiting;
            }
        }
        if resolved {
            return Ok(AwaitOutcome::AlreadyResolved);
        }
        self.future_waiters.entry(future.slot).or_default().push(cont);
        // The `ContinuationWaiting` trace is emitted once, by the commit phase
        // (§18 Phase G), so every await path produces exactly one event.
        Ok(AwaitOutcome::Registered)
    }

    /// Single-assignment resolution of a future: publish the value, then wake
    /// every waiter into its (next) run-class bin (§12, §19).
    pub fn resolve_future(&mut self, future: Ref64, value: Ref64) -> Result<(), RuntimeError> {
        {
            let f = self.futures.get_mut(future)?;
            if f.state != FutureState::Pending {
                return Err(RuntimeError::AlreadyResolved);
            }
            f.state = FutureState::Resolved;
            f.value = value;
            f.resolved_epoch = self.epoch;
        }
        let waiters = self.future_waiters.remove(&future.slot).unwrap_or_default();
        let owner = self.futures.get(future)?.owner_domain;
        for w in waiters {
            let (process, rc) = {
                let c = self.continuations.get(w).ok();
                match c {
                    Some(c) => (c.process, c.run_class),
                    None => continue,
                }
            };
            {
                let c = self.continuations.get_mut(w).unwrap();
                c.status = crate::abi::continuations::ContinuationState::Runnable;
            }
            self.scheduler.enqueue(rc, w);
            self.trace(EventKind::ContinuationReady, process, w, rc, 0);
        }
        self.trace(EventKind::FutureResolved, owner, Ref64::NULL, 0, value.slot);
        Ok(())
    }

    /// Read a resolved future's value (an object ref), or `None` if unresolved.
    pub fn future_value(&self, future: Ref64) -> Option<Ref64> {
        match self.futures.get(future) {
            Ok(f) if f.state == FutureState::Resolved => Some(f.value),
            _ => None,
        }
    }

    // ---- messages --------------------------------------------------------

    /// Send a message from a continuation into `receiver`'s mailbox with
    /// ordered per-pair sequencing (§11), emitting a `MessageSent` trace on
    /// committed send. If the mailbox is full, registers `sender_cont` as the
    /// `full_waiter` and returns `MailboxFull` (the sender does not spin, §11).
    pub fn enqueue_message(
        &mut self,
        sender: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_cont: Ref64,
    ) -> Result<(), RuntimeError> {
        self.push_message(sender, receiver, payload, sender_cont, true)
    }

    /// Deliver an externally-ingested message (§18 Phase A: external messages)
    /// without a `MessageSent` trace — it is an input, not a send.
    pub fn ingest_message(
        &mut self,
        sender: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_cont: Ref64,
    ) -> Result<(), RuntimeError> {
        self.push_message(sender, receiver, payload, sender_cont, false)
    }

    fn push_message(
        &mut self,
        sender: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_cont: Ref64,
        trace_send: bool,
    ) -> Result<(), RuntimeError> {
        let seq = {
            let key = (sender.slot, receiver.slot);
            let s = self.send_sequences.entry(key).or_insert(0);
            let v = *s;
            *s = s.wrapping_add(1);
            v
        };
        let mut msg = MessageDescriptor::new(sender, receiver, payload);
        msg.sender_sequence = seq;
        msg.logical_timestamp = self.logical_time;

        // Commit the send inside a scoped block so the mailbox borrow ends
        // before we touch other tables for tracing / waking.
        let waiter = {
            let mailbox = self
                .mailboxes
                .get_mut(&receiver.slot)
                .ok_or(RuntimeError::MissingMailbox)?;
            if mailbox.entries.len() >= mailbox.capacity {
                // Every blocked sender is registered, not just the first, so no
                // sender can park on a full mailbox and never be woken (§11).
                // A retrying sender is registered only once.
                if !sender_cont.is_null() && !mailbox.full_waiters.contains(&sender_cont) {
                    mailbox.full_waiters.push_back(sender_cont);
                }
                return Err(RuntimeError::MailboxFull);
            }
            mailbox.entries.push_back(msg);
            mailbox.recv_waiters.pop_front()
        };

        if trace_send {
            self.trace(EventKind::MessageSent, sender, sender_cont, 0, seq as u32);
        }

        if let Some(waiter) = waiter {
            let rc = self.continuations.get(waiter)?.run_class;
            self.scheduler.enqueue(rc, waiter);
            {
                let c = self.continuations.get_mut(waiter).unwrap();
                c.status = crate::abi::continuations::ContinuationState::Runnable;
            }
            self.trace(EventKind::MessageReceived, receiver, waiter, rc, 0);
        }
        Ok(())
    }

    /// Pop the next message for `cont`'s process, waking a `full_waiter` if
    /// capacity frees. Returns `Ok(None)` after registering `cont` as a receiver
    /// waiter (the caller should await).
    pub fn receive_message(&mut self, cont: Ref64) -> Result<Option<MessageDescriptor>, RuntimeError> {
        let process = self.continuations.get(cont)?.process;
        let mailbox = self
            .mailboxes
            .get_mut(&process.slot)
            .ok_or(RuntimeError::MissingMailbox)?;
        if let Some(msg) = mailbox.entries.pop_front() {
            // One receive frees exactly one slot, so wake exactly one sender.
            if let Some(w) = mailbox.full_waiters.pop_front() {
                let rc = self.continuations.get(w)?.run_class;
                self.scheduler.enqueue(rc, w);
                {
                    let c = self.continuations.get_mut(w).unwrap();
                    c.status = crate::abi::continuations::ContinuationState::Runnable;
                }
                self.trace(EventKind::ContinuationReady, process, w, rc, 0);
            }
            self.trace(
                EventKind::MessageReceived,
                process,
                cont,
                0,
                msg.sender_sequence as u32,
            );
            return Ok(Some(msg));
        }
        mailbox.recv_waiters.push_back(cont);
        Ok(None)
    }

    /// Number of undelivered messages in a process's mailbox.
    pub fn mailbox_len(&self, p: Ref64) -> usize {
        self.mailboxes
            .get(&p.slot)
            .map(|m| m.entries.len())
            .unwrap_or(0)
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}
