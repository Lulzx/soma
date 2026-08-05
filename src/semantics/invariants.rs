//! Executable well-formedness invariants for the SOMA abstract machine.
//!
//! These are the machine-checked half of the semantic specification in
//! `docs/SOMA-v0.2.md`. Every invariant here has an identifier (`I1`, `I2`, …)
//! that matches a numbered clause in that document, so the prose and the code
//! cannot drift apart without a test failing.
//!
//! The checker is a predicate over a whole machine state, not over a
//! transition. It answers "is this a legal state", which is what makes it
//! usable as a postcondition after *any* transition: run it after every epoch
//! and any rule that can produce an illegal state gets caught, without having
//! to anticipate which rule.
//!
//! Capability safety is split into structural attenuation/integrity checks and
//! a trace-level effect check. The latter rejects every governed effect that is
//! not immediately paired with the matching successful authority decision.

use crate::abi::continuations::ContinuationState;
use crate::abi::{CollectiveState, FutureState, Kind, ProcessState, Ref64, StateAccess};
use crate::kernel::Kernel;

/// Which clause of the specification a violation belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Invariant {
    /// I1. Every reference held in a live descriptor resolves.
    ReferenceIntegrity,
    /// I2. No continuation is left mid-execution at a quiescent state.
    NoContinuationLeftRunning,
    /// I3. A continuation's process is live; a terminated process has no
    /// schedulable continuations.
    ProcessContinuationConsistency,
    /// I4. Futures are single-assignment, and nothing waits on a settled one.
    FutureSingleAssignment,
    /// I5. Mailboxes respect their declared bound.
    MailboxBound,
    /// I6. Messages from one sender to one receiver stay in send order.
    MessageOrdering,
    /// I7. Everything in a runnable bin is live, runnable, and correctly binned.
    SchedulerWellFormed,
    /// I8. No two continuations share a frame object.
    FrameExclusivity,
    /// I10a. Derived capabilities never amplify rights or byte range.
    CapabilityAttenuation,
    /// I10b. Capability targets and parent links resolve with valid rights.
    CapabilityIntegrity,
    /// I10c. Every governed effect immediately follows matching authority.
    NoUnauthorizedEffect,
    /// I11. The trace is a strictly increasing logical clock.
    TraceMonotonicity,
    /// I12. Accounting counters are mutually consistent.
    AccountingConsistency,
    /// I13. At most one mutable process-state continuation starts per epoch.
    SerialProcessExecution,
}

/// A specific way in which a state was illegal.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Violation {
    pub invariant: Invariant,
    pub detail: String,
}

impl Violation {
    fn new(invariant: Invariant, detail: impl Into<String>) -> Violation {
        Violation {
            invariant,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.invariant, self.detail)
    }
}

/// Check every invariant, returning all violations rather than the first, so a
/// broken transition reports its full damage in one pass.
pub fn check(kernel: &Kernel) -> Vec<Violation> {
    let mut v = Vec::new();
    reference_integrity(kernel, &mut v);
    no_continuation_left_running(kernel, &mut v);
    process_continuation_consistency(kernel, &mut v);
    future_single_assignment(kernel, &mut v);
    mailbox_bound(kernel, &mut v);
    message_ordering(kernel, &mut v);
    scheduler_well_formed(kernel, &mut v);
    frame_exclusivity(kernel, &mut v);
    capability_attenuation(kernel, &mut v);
    capability_integrity(kernel, &mut v);
    no_unauthorized_effect(kernel, &mut v);
    trace_monotonicity(kernel, &mut v);
    accounting_consistency(kernel, &mut v);
    serial_process_execution(kernel, &mut v);
    v.sort();
    v
}

/// Panic with every violation, for use as a test postcondition.
pub fn assert_legal(kernel: &Kernel) {
    let violations = check(kernel);
    assert!(
        violations.is_empty(),
        "illegal machine state at epoch {}:\n{}",
        kernel.epoch_number(),
        violations
            .iter()
            .map(|v| format!("  {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ---- I1 ------------------------------------------------------------------

fn live(kernel: &Kernel, r: Ref64) -> bool {
    match r.kind {
        Kind::Process => kernel.processes().get(r).is_ok(),
        Kind::Object => kernel.objects().get(r).is_ok(),
        Kind::Continuation => kernel.continuations().get(r).is_ok(),
        Kind::Future => kernel.futures().get(r).is_ok(),
        // Capability references are actor-relative and are checked by I10b in
        // the space that owns them; there is no meaningful global lookup.
        Kind::Capability => true,
        Kind::Channel => kernel.channels().get(r).is_ok(),
        Kind::Collective => kernel.collectives().get(r).is_ok(),
        // Domains, contracts and modules have no table in the reference model;
        // a reference to one cannot yet be validated.
        _ => true,
    }
}

fn reference_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, p) in kernel.processes().iter() {
        if !p.id.is_null() && p.id != r {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("process {} carries id {}", r.slot, p.id.slot),
            ));
        }
        if !p.state.is_null() && !live(kernel, p.state) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("process {} has a dangling state object", r.slot),
            ));
        }
    }

    for (r, c) in kernel.continuations().iter() {
        if !live(kernel, c.process) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} references a dead process", r.slot),
            ));
        }
        if !c.frame.is_null() && !live(kernel, c.frame) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} has a dangling frame", r.slot),
            ));
        }
        if !c.dependency.is_null() && !live(kernel, c.dependency) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("continuation {} depends on a dead entity", r.slot),
            ));
        }
    }

    for (r, f) in kernel.futures().iter() {
        if !f.owner_process.is_null() && !live(kernel, f.owner_process) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("future {} references a dead owner process", r.slot),
            ));
        }
        if f.state == FutureState::Resolved && !f.value.is_null() && !live(kernel, f.value) {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("future {} resolved to a dead object", r.slot),
            ));
        }
    }

    for (slot, mailbox) in kernel.mailboxes() {
        for m in &mailbox.entries {
            if !m.payload.is_null() && !live(kernel, m.payload) {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!("mailbox {slot} holds a message with a dead payload"),
                ));
            }
        }
    }

    for (r, channel) in kernel.channels().iter() {
        if channel.id != r || channel.closed > 1 {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!("channel {} contains an invalid id or closed state", r.slot),
            ));
        }
    }

    for queue in kernel.channel_queue_snapshots() {
        for (payload, _, escrow_target) in queue.entries {
            if !live(kernel, payload) || escrow_target != payload {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!(
                        "channel {} holds an invalid escrowed payload",
                        queue.channel.slot
                    ),
                ));
            }
        }
        for waiter in queue.send_waiters.into_iter().chain(queue.receive_waiters) {
            if !live(kernel, waiter) {
                out.push(Violation::new(
                    Invariant::ReferenceIntegrity,
                    format!("channel {} holds a dead waiter", queue.channel.slot),
                ));
            }
        }
    }

    for (r, collective) in kernel.collectives().iter() {
        if collective.id != r
            || (!collective.owner_process.is_null() && !live(kernel, collective.owner_process))
            || !live(kernel, collective.inputs)
            || !live(kernel, collective.completion_future)
            || (!collective.outputs.is_null() && !live(kernel, collective.outputs))
        {
            out.push(Violation::new(
                Invariant::ReferenceIntegrity,
                format!(
                    "collective {} contains a dangling or inconsistent reference",
                    r.slot
                ),
            ));
        }
    }
}

// ---- I2 ------------------------------------------------------------------

fn no_continuation_left_running(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, c) in kernel.continuations().iter() {
        if c.status == ContinuationState::Running {
            out.push(Violation::new(
                Invariant::NoContinuationLeftRunning,
                format!("continuation {} is still RUNNING between epochs", r.slot),
            ));
        }
    }
}

// ---- I3 ------------------------------------------------------------------

fn process_continuation_consistency(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, c) in kernel.continuations().iter() {
        let process = match kernel.processes().get(c.process) {
            Ok(p) => p,
            // Already reported by I1.
            Err(_) => continue,
        };
        let schedulable = matches!(
            c.status,
            ContinuationState::Runnable | ContinuationState::Waiting
        );
        let terminal = process.status == ProcessState::Failed as u32
            || process.status == ProcessState::Terminated as u32
            || process.status == ProcessState::Cancelled as u32;
        if schedulable && terminal {
            out.push(Violation::new(
                Invariant::ProcessContinuationConsistency,
                format!(
                    "continuation {} is {:?} but its process {} is terminal",
                    r.slot, c.status, c.process.slot
                ),
            ));
        }
    }
}

// ---- I4 ------------------------------------------------------------------

fn future_single_assignment(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (r, f) in kernel.futures().iter() {
        match f.state {
            FutureState::Pending => {
                if !f.value.is_null() {
                    out.push(Violation::new(
                        Invariant::FutureSingleAssignment,
                        format!("future {} is pending but carries a value", r.slot),
                    ));
                }
            }
            FutureState::Resolved | FutureState::Failed | FutureState::Cancelled => {
                if f.resolved_epoch > kernel.epoch_number() {
                    out.push(Violation::new(
                        Invariant::FutureSingleAssignment,
                        format!("future {} resolved in a future epoch", r.slot),
                    ));
                }
                // A settled future's waiter list has already been drained, so a
                // continuation registered on it would never wake.
                if let Some(waiters) = kernel.future_waiters().get(&r.slot) {
                    if !waiters.is_empty() {
                        out.push(Violation::new(
                            Invariant::FutureSingleAssignment,
                            format!(
                                "{} continuations wait on already-settled future {}",
                                waiters.len(),
                                r.slot
                            ),
                        ));
                    }
                }
            }
        }
    }

    for (r, collective) in kernel.collectives().iter() {
        let Ok(completion) = kernel.futures().get(collective.completion_future) else {
            continue;
        };
        let consistent = match collective.state {
            CollectiveState::Pending => {
                completion.state == FutureState::Pending && collective.outputs.is_null()
            }
            CollectiveState::Completed => {
                completion.state == FutureState::Resolved
                    && !collective.outputs.is_null()
                    && completion.value == collective.outputs
            }
            CollectiveState::Failed => completion.state == FutureState::Failed,
            CollectiveState::Cancelled => completion.state == FutureState::Cancelled,
        };
        if !consistent {
            out.push(Violation::new(
                Invariant::FutureSingleAssignment,
                format!("collective {} disagrees with its completion future", r.slot),
            ));
        }
    }
}

// ---- I5 ------------------------------------------------------------------

fn mailbox_bound(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (slot, mailbox) in kernel.mailboxes() {
        if mailbox.entries.len() > mailbox.capacity {
            out.push(Violation::new(
                Invariant::MailboxBound,
                format!(
                    "mailbox {slot} holds {} messages over a capacity of {}",
                    mailbox.entries.len(),
                    mailbox.capacity
                ),
            ));
        }
    }
    for queue in kernel.channel_queue_snapshots() {
        let Ok(descriptor) = kernel.channels().get(queue.channel) else {
            continue;
        };
        if queue.entries.len() > descriptor.capacity as usize {
            out.push(Violation::new(
                Invariant::MailboxBound,
                format!(
                    "channel {} holds {} messages over a capacity of {}",
                    queue.channel.slot,
                    queue.entries.len(),
                    descriptor.capacity
                ),
            ));
        }
    }
}

// ---- I6 ------------------------------------------------------------------

fn message_ordering(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (slot, mailbox) in kernel.mailboxes() {
        let mut last: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        for m in &mailbox.entries {
            if let Some(previous) = last.get(&m.sender.slot) {
                if m.sender_sequence <= *previous {
                    out.push(Violation::new(
                        Invariant::MessageOrdering,
                        format!(
                            "mailbox {slot}: sender {} delivered sequence {} after {}",
                            m.sender.slot, m.sender_sequence, previous
                        ),
                    ));
                }
            }
            last.insert(m.sender.slot, m.sender_sequence);
        }
    }
    for queue in kernel.channel_queue_snapshots() {
        for pair in queue.entries.windows(2) {
            if pair[1].1 <= pair[0].1 {
                out.push(Violation::new(
                    Invariant::MessageOrdering,
                    format!(
                        "channel {} sequence {} follows {}",
                        queue.channel.slot, pair[1].1, pair[0].1
                    ),
                ));
            }
        }
    }
}

// ---- I7 ------------------------------------------------------------------

fn scheduler_well_formed(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (bin, cont) in kernel.scheduler().pending_entries() {
        let c = match kernel.continuations().get(cont) {
            Ok(c) => c,
            Err(_) => {
                out.push(Violation::new(
                    Invariant::SchedulerWellFormed,
                    format!("bin {bin} holds dead continuation {}", cont.slot),
                ));
                continue;
            }
        };
        if c.status != ContinuationState::Runnable {
            out.push(Violation::new(
                Invariant::SchedulerWellFormed,
                format!(
                    "bin {bin} holds continuation {} in state {:?}",
                    cont.slot, c.status
                ),
            ));
        }
        let expected = kernel.scheduler().bin_of(c.run_class);
        if expected != bin {
            out.push(Violation::new(
                Invariant::SchedulerWellFormed,
                format!(
                    "continuation {} of run class {} sits in bin {bin}, not {expected}",
                    cont.slot, c.run_class
                ),
            ));
        }
    }
}

// ---- I8 ------------------------------------------------------------------

fn frame_exclusivity(kernel: &Kernel, out: &mut Vec<Violation>) {
    let mut owner: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (r, c) in kernel.continuations().iter() {
        if c.frame.is_null() {
            continue;
        }
        if let Some(previous) = owner.insert(c.frame.slot, r.slot) {
            out.push(Violation::new(
                Invariant::FrameExclusivity,
                format!(
                    "frame {} is shared by continuations {} and {}",
                    c.frame.slot, previous, r.slot
                ),
            ));
        }
    }
}

// ---- I10 -----------------------------------------------------------------

fn capability_attenuation(kernel: &Kernel, out: &mut Vec<Violation>) {
    for (holder, space) in kernel.capability_spaces() {
        for (r, cap) in space.iter() {
            if cap.parent_capability.is_null() {
                continue;
            }
            let Ok(parent) = space.get(cap.parent_capability) else {
                // I10b reports the broken link.
                continue;
            };
            let cap_end = cap.offset.checked_add(cap.length);
            let parent_end = parent.offset.checked_add(parent.length);
            if cap.rights & !parent.rights != 0
                || cap.offset < parent.offset
                || cap_end.is_none()
                || parent_end.is_none()
                || cap_end > parent_end
            {
                out.push(Violation::new(
                    Invariant::CapabilityAttenuation,
                    format!(
                        "capability {} in space {holder} amplifies its parent",
                        r.slot
                    ),
                ));
            }
        }
    }
}

fn capability_integrity(kernel: &Kernel, out: &mut Vec<Violation>) {
    use crate::abi::Rights;

    for (holder, space) in kernel.capability_spaces() {
        for (r, cap) in space.iter() {
            let target_live = match cap.target.kind {
                Kind::Process => kernel.processes().get(cap.target).is_ok(),
                Kind::Object => kernel.objects().get(cap.target).is_ok(),
                Kind::Continuation => kernel.continuations().get(cap.target).is_ok(),
                Kind::Future => kernel.futures().get(cap.target).is_ok(),
                Kind::Channel => kernel.channels().get(cap.target).is_ok(),
                Kind::Collective => kernel.collectives().get(cap.target).is_ok(),
                Kind::Capability => space.get(cap.target).is_ok(),
                _ => false,
            };
            if !target_live {
                out.push(Violation::new(
                    Invariant::CapabilityIntegrity,
                    format!(
                        "capability {} in space {holder} has a dead or unsupported target",
                        r.slot
                    ),
                ));
            }
            if cap.rights & !Rights::for_target(cap.target.kind) != 0 {
                out.push(Violation::new(
                    Invariant::CapabilityIntegrity,
                    format!(
                        "capability {} in space {holder} has rights invalid for {:?}",
                        r.slot, cap.target.kind
                    ),
                ));
            }
            if !cap.parent_capability.is_null() && space.get(cap.parent_capability).is_err() {
                out.push(Violation::new(
                    Invariant::CapabilityIntegrity,
                    format!("capability {} in space {holder} has a dead parent", r.slot),
                ));
            }
        }
    }

    for (object, _) in kernel.objects().iter() {
        let writers = kernel.authority_holder_count(object, Rights::WRITE);
        if writers > 1 {
            out.push(Violation::new(
                Invariant::CapabilityIntegrity,
                format!(
                    "object {} has {writers} mutable authority holders",
                    object.slot
                ),
            ));
        }
    }
}

fn no_unauthorized_effect(kernel: &Kernel, out: &mut Vec<Violation>) {
    use crate::abi::EventKind;

    for (index, effect) in kernel.trace_events().iter().enumerate() {
        if effect.event_kind != EventKind::AuthorityEffect {
            continue;
        }
        let authorized = index.checked_sub(1).and_then(|previous| {
            let decision = &kernel.trace_events()[previous];
            (decision.event_kind == EventKind::AuthorityGranted
                && decision.process == effect.process
                && decision.continuation == effect.continuation
                && decision.run_class == effect.run_class)
                .then_some(())
        });
        if authorized.is_none() {
            out.push(Violation::new(
                Invariant::NoUnauthorizedEffect,
                format!(
                    "trace event {index} applies right {} by actor {} to target {} without an adjacent grant",
                    effect.run_class, effect.process.slot, effect.continuation.slot
                ),
            ));
        }
    }
}

// ---- I11 -----------------------------------------------------------------

fn trace_monotonicity(kernel: &Kernel, out: &mut Vec<Violation>) {
    let mut previous_time = 0u64;
    let mut previous_epoch = 0u32;
    for (i, e) in kernel.trace_events().iter().enumerate() {
        if e.logical_time <= previous_time && i > 0 {
            out.push(Violation::new(
                Invariant::TraceMonotonicity,
                format!(
                    "trace event {i} has logical time {} after {}",
                    e.logical_time, previous_time
                ),
            ));
        }
        if e.epoch < previous_epoch {
            out.push(Violation::new(
                Invariant::TraceMonotonicity,
                format!("trace event {i} moves backward to epoch {}", e.epoch),
            ));
        }
        previous_time = e.logical_time;
        previous_epoch = e.epoch;
    }
}

// ---- I12 -----------------------------------------------------------------

fn accounting_consistency(kernel: &Kernel, out: &mut Vec<Violation>) {
    let a = kernel.accounting();
    if a.useful_lane_slots > a.lane_slots {
        out.push(Violation::new(
            Invariant::AccountingConsistency,
            format!(
                "useful lane slots {} exceed issued lane slots {}",
                a.useful_lane_slots, a.lane_slots
            ),
        ));
    }
    if a.full_cohorts > a.cohorts {
        out.push(Violation::new(
            Invariant::AccountingConsistency,
            format!(
                "full cohorts {} exceed total cohorts {}",
                a.full_cohorts, a.cohorts
            ),
        ));
    }
    if a.lane_slots != a.useful_lane_slots + a.idle_lane_slots {
        out.push(Violation::new(
            Invariant::AccountingConsistency,
            format!(
                "lane slots {} do not split into {} useful and {} idle",
                a.lane_slots, a.useful_lane_slots, a.idle_lane_slots
            ),
        ));
    }
}

// ---- I13 -----------------------------------------------------------------

fn serial_process_execution(kernel: &Kernel, out: &mut Vec<Violation>) {
    use crate::abi::EventKind;

    let mut claimed = std::collections::HashSet::new();
    for (index, event) in kernel.trace_events().iter().enumerate() {
        if event.event_kind != EventKind::ContinuationStarted {
            continue;
        }
        let Ok(continuation) = kernel.continuations().get(event.continuation) else {
            continue;
        };
        if continuation.state_access != StateAccess::Mutable {
            continue;
        }
        let key = (event.epoch, continuation.process.slot);
        if !claimed.insert(key) {
            out.push(Violation::new(
                Invariant::SerialProcessExecution,
                format!(
                    "trace event {index} starts a second mutable continuation for process {} in epoch {}",
                    continuation.process.slot, event.epoch
                ),
            ));
        }
    }
}

/// Whether a continuation declares mutable access to its process state.
pub fn mutates_process_state(kernel: &Kernel, continuation: Ref64) -> bool {
    kernel
        .continuations()
        .get(continuation)
        .map(|c| c.state_access == StateAccess::Mutable)
        .unwrap_or(false)
}
