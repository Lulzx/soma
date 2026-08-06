//! Giving back what a finished process was holding.
//!
//! The machine allocates and never releases. Every published object, every
//! continuation, every capability minted since the run started is still in its
//! table, and `examples/growth_sweep` made each operation's *time* independent
//! of that — which only means the memory limit arrives sooner and no slower.
//! `examples/memory_profile` puts it at about 1.3KB per published batch
//! against 32 bytes of payload, and at four hundred thousand processes run to
//! completion, all four hundred thousand still resident.
//!
//! A terminated process is the unambiguous case. Its state object, its
//! continuations and their frames, its mailbox, its supervision queue, and its
//! capability space are private to it: nothing else can name them, and it will
//! never run again. That is what this reclaims.
//!
//! **It is explicit, and it is not called from anywhere.** Reclamation is a
//! policy — when a supervisor may still ask about a child, whether a run wants
//! its history inspectable afterwards — and this file does not decide that.
//! It provides the operation and the conditions under which it is safe; a
//! caller decides when. Nothing in the existing semantics changes unless it is
//! called.
//!
//! Slots are recycled with a generation bump (`table::GenTable::delete`), so a
//! reference held to something reclaimed resolves as `StaleReference` rather
//! than as whatever later occupies the slot. That is the property that makes
//! this safe to do at all, and it is why reclaiming a process nothing can
//! reach differs from reclaiming one somebody still holds a reference to: the
//! second is caught, not corrupted.

use crate::abi::{Kind, ObjectKind, ProcessState, Ref64};
use crate::kernel::{Kernel, RuntimeError};

/// What a reachability pass found, for callers that want to see the shape of
/// what is being held before deciding to release it.
#[derive(Clone, Debug, Default)]
pub struct Unreachable {
    pub objects: Vec<Ref64>,
    pub collectives: Vec<Ref64>,
    pub futures: Vec<Ref64>,
}

impl Unreachable {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.collectives.is_empty() && self.futures.is_empty()
    }

    pub fn len(&self) -> usize {
        self.objects.len() + self.collectives.len() + self.futures.len()
    }
}

/// What one reclamation pass gave back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reclaimed {
    pub processes: usize,
    pub continuations: usize,
    pub objects: usize,
    pub capabilities: usize,
    pub collectives: usize,
    pub futures: usize,
}

impl Reclaimed {
    pub fn is_empty(&self) -> bool {
        *self == Reclaimed::default()
    }
}

/// Why a terminated process was left alone.
///
/// Returned rather than silently skipped, because "nothing was reclaimed" and
/// "nothing was reclaimable" are different answers and a caller watching its
/// memory needs to tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retained {
    /// A supervisor has not yet taken the notice about it.
    SupervisionPending,
    /// It supervises a process that has not finished.
    SupervisesLive,
    /// A restart blueprint could still bring it back.
    Restartable,
}

impl Kernel {
    /// Processes that have finished and hold nothing anyone can still reach.
    pub fn reclaimable_processes(&self) -> Vec<Ref64> {
        self.processes
            .iter()
            .map(|(process, _)| process)
            .filter(|process| self.retention_reason(*process).is_none())
            .filter(|process| self.is_finished(*process))
            .collect()
    }

    /// Terminated processes this pass would keep, and why.
    pub fn retained_processes(&self) -> Vec<(Ref64, Retained)> {
        self.processes
            .iter()
            .map(|(process, _)| process)
            .filter(|process| self.is_finished(*process))
            .filter_map(|process| Some((process, self.retention_reason(process)?)))
            .collect()
    }

    fn is_finished(&self, process: Ref64) -> bool {
        matches!(
            self.process_state(process),
            Ok(ProcessState::Terminated | ProcessState::Failed | ProcessState::Cancelled)
        )
    }

    fn retention_reason(&self, process: Ref64) -> Option<Retained> {
        // A notice the supervisor has not taken is the supervisor's to read,
        // and it names this process.
        let observed = self.supervision_queues.values().any(|queue| {
            queue
                .notices
                .iter()
                .any(|notice| notice.child == process || notice.replacement == process)
        });
        if observed {
            return Some(Retained::SupervisionPending);
        }
        if self.restart_blueprints.contains_key(&process.key()) {
            return Some(Retained::Restartable);
        }
        let supervises_live = self.processes.iter().any(|(other, descriptor)| {
            descriptor.supervisor == process && other != process && !self.is_finished(other)
        });
        if supervises_live {
            return Some(Retained::SupervisesLive);
        }
        None
    }

    /// Release the private state of every finished process that nothing can
    /// still reach.
    ///
    /// Frees exactly what `allocate_process` allocated — the capability space,
    /// the mailbox, the supervision queue, the state object — plus the
    /// continuations the process created and the frame object each one holds.
    /// It does not touch objects the process published: a frozen array handed
    /// to a collective outlives its producer, and deciding when *that* is dead
    /// is a different question from this one.
    pub fn reclaim_finished_processes(&mut self) -> Reclaimed {
        let mut reclaimed = Reclaimed::default();
        let dead = self.reclaimable_processes();
        if dead.is_empty() {
            return reclaimed;
        }

        for process in &dead {
            let continuations = self
                .continuations_by_process
                .remove(&process.key())
                .unwrap_or_default();

            for continuation in &continuations {
                let frame = self
                    .continuations
                    .get(*continuation)
                    .ok()
                    .map(|descriptor| descriptor.frame);
                // The bins are cleared first: a reclaimed continuation must not
                // stay queued behind a reference that no longer resolves.
                self.scheduler.remove(*continuation);
                if self.continuations.delete(*continuation).is_ok() {
                    reclaimed.continuations += 1;
                }
                reclaimed.capabilities += self.purge_capabilities_naming(*continuation);
                if let Some(frame) = frame {
                    if !frame.is_null() && self.release_object(frame) {
                        reclaimed.objects += 1;
                        reclaimed.capabilities += self.purge_capabilities_naming(frame);
                    }
                }
            }

            // Anything still listed as waiting on a future is gone.
            let gone: Vec<Ref64> = continuations;
            for waiters in self.future_waiters.values_mut() {
                waiters.retain(|waiter| !gone.contains(waiter));
            }

            let state = self
                .processes
                .get(*process)
                .map(|p| p.state)
                .unwrap_or(Ref64::NULL);
            if !state.is_null() && self.release_object(state) {
                reclaimed.objects += 1;
            }

            if let Some(space) = self.capability_spaces.remove(&process.key()) {
                reclaimed.capabilities += space.len();
            }
            // Its own space is not the only one naming it. Whoever created it
            // holds a capability over it, and a capability whose target no
            // longer resolves is exactly what `CapabilityIntegrity` rejects —
            // so reclaiming without this produces an illegal machine rather
            // than a smaller one.
            reclaimed.capabilities += self.purge_capabilities_naming(*process);
            self.mailboxes.remove(&process.key());
            self.supervision_queues.remove(&process.key());
            self.restart_blueprints.remove(&process.key());

            // `DomainContractIntegrity` checks this against the number of
            // live processes in the domain, so despite its name it is a
            // population and not a total, and reclaiming has to give one back.
            let domain = self
                .processes
                .get(*process)
                .map(|descriptor| descriptor.domain)
                .unwrap_or(Ref64::NULL);
            if self.processes.delete(*process).is_ok() {
                reclaimed.processes += 1;
                if let Ok(descriptor) = self.domains.get_mut(domain) {
                    descriptor.processes_created = descriptor.processes_created.saturating_sub(1);
                }
            }
        }

        // Sequence numbers are keyed by sender and receiver together, so they
        // are purged once for the whole pass rather than scanned per process.
        let reclaimed_keys: std::collections::HashSet<u64> =
            dead.iter().map(|process| process.key()).collect();
        self.send_sequences.retain(|(sender, receiver), _| {
            !reclaimed_keys.contains(sender) && !reclaimed_keys.contains(receiver)
        });

        reclaimed
    }

    /// Delete every capability naming `target`, in every space.
    ///
    /// Cheap because a space indexes its capabilities by target; before that
    /// index this would have been a scan of every capability in the kernel per
    /// reclaimed entity, which is the shape `kernel::capability_space`
    /// describes.
    fn purge_capabilities_naming(&mut self, target: Ref64) -> usize {
        let mut purged = 0;
        for space in self.capability_spaces.values_mut() {
            let naming: Vec<Ref64> = space
                .for_target(target)
                .into_iter()
                .map(|(capability, _)| capability)
                .collect();
            for capability in naming {
                if space.delete(capability).is_ok() {
                    purged += 1;
                }
            }
        }
        purged
    }

    /// Delete an object and its payload. Reports whether anything was there.
    fn release_object(&mut self, object: Ref64) -> bool {
        if self.objects.get(object).is_err() {
            return false;
        }
        // Process state is deleted here as part of its process; nothing else
        // reclaims one, so the kind is not a guard, only a note.
        let _ = ObjectKind::ProcessState;
        self.object_payloads.remove(&object.key());
        self.objects.delete(object).is_ok()
    }
}

/// Reachability, in the sense a capability machine already has one.
///
/// A finished process is easy: it is named by its own descriptor and nothing
/// else needs it. A *published batch* is not. Its object outlives the process
/// that produced it, its collective and completion future outlive both, and
/// `examples/memory_profile` prices the three together at about 1.3KB per
/// batch that nothing ever released — against 32 bytes of payload.
///
/// There is no reference counting here and adding one would be a change to the
/// ABI. But the machine already says what "reachable" means: an entity is
/// reachable when something can *name* it. So this marks from the roots that
/// can name anything — every capability's target, every live process,
/// continuation, queue and waiter list — follows the references those hold,
/// and reports what nothing arrived at.
///
/// The subtlety that makes it a closure rather than a scan: a collective names
/// its input and output objects, so an object is not garbage merely because no
/// capability names it. But if the collective *itself* is unreachable, its
/// objects must not be kept alive by it. Marking therefore starts from the
/// roots and propagates, rather than treating every descriptor as a root.
impl Kernel {
    /// Give up the authority `actor` holds over `target`.
    ///
    /// The counterpart to reachability: nothing can become unreachable while
    /// its owner still holds a capability naming it, so a run that wants its
    /// finished work collected needs a way to say it is finished with it.
    /// This is that, and it is the only new *semantic* operation here —
    /// everything else in this module is bookkeeping over state nothing can
    /// name.
    ///
    /// It drops only this actor's authority. A frozen array several processes
    /// read stays readable by the others; it becomes unreachable when the last
    /// of them lets go.
    pub fn release_authority(&mut self, actor: Ref64, target: Ref64) -> Result<(), RuntimeError> {
        let space = self
            .capability_spaces
            .get_mut(&actor.key())
            .ok_or(RuntimeError::Abi(crate::abi::AbiError::NoAuthority))?;
        let held: Vec<Ref64> = space
            .for_target(target)
            .into_iter()
            .map(|(capability, _)| capability)
            .collect();
        if held.is_empty() {
            return Err(RuntimeError::Abi(crate::abi::AbiError::NoAuthority));
        }
        for capability in held {
            Self::revoke_capability_tree(space, capability);
        }
        self.trace_authority_released(actor, target);
        Ok(())
    }

    /// Everything nothing can name any more.
    ///
    /// Pure inspection: it deletes nothing, so a caller can look at what a
    /// release would take before taking it.
    pub fn unreachable(&self) -> Unreachable {
        let mut marked: std::collections::HashSet<Ref64> = std::collections::HashSet::new();
        let mut worklist: Vec<Ref64> = Vec::new();

        let root = |reference: Ref64,
                    marked: &mut std::collections::HashSet<Ref64>,
                    worklist: &mut Vec<Ref64>| {
            if !reference.is_null() && marked.insert(reference) {
                worklist.push(reference);
            }
        };

        // Anything a capability names can be reached by whoever holds it.
        for space in self.capability_spaces.values() {
            for (_, capability) in space.iter() {
                root(capability.target, &mut marked, &mut worklist);
            }
        }
        // A live process, and everything a live process's descriptor names.
        for (process, descriptor) in self.processes.iter() {
            root(process, &mut marked, &mut worklist);
            root(descriptor.state, &mut marked, &mut worklist);
            root(descriptor.inbox, &mut marked, &mut worklist);
            root(descriptor.urgent_inbox, &mut marked, &mut worklist);
        }
        // A live continuation, whether or not anyone holds a capability to it.
        for (continuation, descriptor) in self.continuations.iter() {
            root(continuation, &mut marked, &mut worklist);
            root(descriptor.frame, &mut marked, &mut worklist);
        }
        // In-flight messages, and everything waiting on something.
        for mailbox in self.mailboxes.values() {
            for entry in &mailbox.entries {
                root(entry.payload, &mut marked, &mut worklist);
                root(entry.transferred_capability, &mut marked, &mut worklist);
                root(entry.completion_future, &mut marked, &mut worklist);
            }
        }
        for queue in self.channel_queues.values() {
            for entry in &queue.entries {
                root(entry.descriptor.payload, &mut marked, &mut worklist);
                root(
                    entry.descriptor.completion_future,
                    &mut marked,
                    &mut worklist,
                );
                root(entry.payload_authority.target, &mut marked, &mut worklist);
            }
        }
        for queue in self.supervision_queues.values() {
            for notice in &queue.notices {
                root(notice.child, &mut marked, &mut worklist);
                root(notice.replacement, &mut marked, &mut worklist);
            }
        }

        // Follow what the marked entities name.
        while let Some(reference) = worklist.pop() {
            let named: Vec<Ref64> = match reference.kind {
                Kind::Collective => self
                    .collectives
                    .get(reference)
                    .map(|descriptor| {
                        vec![
                            descriptor.inputs,
                            descriptor.outputs,
                            descriptor.completion_future,
                        ]
                    })
                    .unwrap_or_default(),
                Kind::Future => self
                    .futures
                    .get(reference)
                    .map(|descriptor| vec![descriptor.value, descriptor.failure])
                    .unwrap_or_default(),
                Kind::Continuation => self
                    .continuations
                    .get(reference)
                    .map(|descriptor| vec![descriptor.frame])
                    .unwrap_or_default(),
                Kind::Process => self
                    .processes
                    .get(reference)
                    .map(|descriptor| vec![descriptor.state])
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            for next in named {
                if !next.is_null() && marked.insert(next) {
                    worklist.push(next);
                }
            }
        }

        Unreachable {
            objects: self
                .objects
                .iter()
                .map(|(object, _)| object)
                .filter(|object| !marked.contains(object))
                .collect(),
            collectives: self
                .collectives
                .iter()
                .map(|(collective, _)| collective)
                .filter(|collective| !marked.contains(collective))
                .collect(),
            futures: self
                .futures
                .iter()
                .map(|(future, _)| future)
                .filter(|future| !marked.contains(future))
                .collect(),
        }
    }

    /// Release everything nothing can name.
    ///
    /// Like `reclaim_finished_processes`, explicit and called from nowhere: a
    /// run that wants its published history inspectable afterwards is not
    /// wrong, and this cannot tell the difference between that and a leak.
    pub fn reclaim_unreachable(&mut self) -> Reclaimed {
        let unreachable = self.unreachable();
        let mut reclaimed = Reclaimed::default();
        for object in unreachable.objects {
            if self.release_object(object) {
                reclaimed.objects += 1;
            }
        }
        for collective in unreachable.collectives {
            if self.collectives.delete(collective).is_ok() {
                reclaimed.collectives += 1;
            }
        }
        for future in unreachable.futures {
            self.future_waiters.remove(&future.key());
            if self.futures.delete(future).is_ok() {
                reclaimed.futures += 1;
            }
        }
        reclaimed
    }
}
