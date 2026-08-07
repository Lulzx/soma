//! Double-buffered append-only runnable bins (§13).
//!
//! SOMA-P1 does not need a general lock-free work queue. Each run class has two
//! buffers — `current` and `next`. During an epoch, workers consume only from
//! `current`; new runnable continuations append only to `next`; the buffers swap
//! at the next scheduling boundary. This gives deterministic epoch boundaries,
//! cheap grouping by run class, and no concurrent pop / ABA problems.
//!
//! This grouping *is* the simplest form of continuation cohorting (§9): every
//! yielded continuation already knows the exact queue it belongs to via
//! `next_run_class`.

use std::collections::{HashMap, HashSet};

use crate::abi::Ref64;
use crate::kernel::effects::Committing;

/// One double-buffered bin for a single run class.
///
/// Entries are `Option<Ref64>` so that a cancelled continuation can be removed
/// without moving its neighbours. Both buffers are only ever appended to and
/// then drained whole, so an entry's index is stable for as long as it is
/// queued, and `placed` can record where each one sits. Removing by scanning
/// the buffers instead made cancellation cost the depth of the queue:
/// `examples/growth_sweep` measured cancelling one four-continuation process
/// at 2.88µs against an empty scheduler and 1.96ms against a million pending.
///
/// `placed` stores the swap sequence an entry was enqueued at rather than
/// which buffer holds it, because the buffers trade places every epoch and
/// rewriting a tag per entry per swap would cost exactly what this avoids. An
/// entry enqueued at sequence `s` is in `next` while `seq == s` and in
/// `current` while `seq == s + 1`; after that it has been drained.
#[derive(Clone, Debug, Default)]
pub struct DoubleBin {
    /// Consumed this epoch.
    current: Vec<Option<Ref64>>,
    /// Appended this epoch; swapped to `current` at the boundary.
    next: Vec<Option<Ref64>>,
    /// Where each queued continuation sits: the swap sequence it arrived at,
    /// and its index in whichever buffer that puts it in.
    placed: HashMap<Ref64, (u64, usize)>,
    /// Entries still present, so a length is not a scan for `Some`.
    live_current: usize,
    live_next: usize,
    seq: u64,
    pub capacity: u32,
}

impl DoubleBin {
    fn new(capacity: u32) -> DoubleBin {
        DoubleBin {
            current: Vec::new(),
            next: Vec::new(),
            placed: HashMap::new(),
            live_current: 0,
            live_next: 0,
            seq: 0,
            capacity,
        }
    }

    /// Append a runnable continuation to the next-epoch buffer.
    pub fn enqueue(&mut self, cont: Ref64) {
        self.next.push(Some(cont));
        self.placed.insert(cont, (self.seq, self.next.len() - 1));
        self.live_next += 1;
    }

    /// Drain the current-epoch buffer for execution.
    pub fn drain_current(&mut self) -> Vec<Ref64> {
        let drained: Vec<Ref64> = std::mem::take(&mut self.current)
            .into_iter()
            .flatten()
            .collect();
        for cont in &drained {
            self.placed.remove(cont);
        }
        self.live_current = 0;
        drained
    }

    pub fn current_len(&self) -> usize {
        self.live_current
    }

    pub fn next_len(&self) -> usize {
        self.live_next
    }

    /// Remove one continuation, if this bin holds it, without touching the
    /// entries around it.
    fn remove_one(&mut self, continuation: Ref64) {
        let Some((sequence, index)) = self.placed.remove(&continuation) else {
            return;
        };
        let (buffer, live): (&mut Vec<Option<Ref64>>, &mut usize) = if sequence == self.seq {
            (&mut self.next, &mut self.live_next)
        } else if sequence + 1 == self.seq {
            (&mut self.current, &mut self.live_current)
        } else {
            // Already drained; `placed` had a stale entry only if a drain
            // missed it, which it cannot.
            return;
        };
        if let Some(slot) = buffer.get_mut(index) {
            if slot.take().is_some() {
                *live -= 1;
            }
        }
    }

    /// Every queued continuation in this bin, current buffer first.
    fn entries(&self) -> impl Iterator<Item = Ref64> + '_ {
        self.current
            .iter()
            .chain(self.next.iter())
            .flatten()
            .copied()
    }

    /// Swap `next` into `current` at the epoch boundary.
    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        self.next.clear();
        self.live_current = self.live_next;
        self.live_next = 0;
        self.seq += 1;
    }
}

/// How runnable continuations are assigned to bins — the single variable the
/// cohorting experiment turns (§26).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SchedulingMode {
    /// One bin per run class, so a bin's contents are uniform by construction.
    /// This is the mechanism under test.
    #[default]
    RunClassBins,
    /// A single persistent queue in arrival order, ignoring run class. The
    /// baseline that isolates launch elimination from cohorting: work is still
    /// resident and still dispatched in lane groups, but those groups mix run
    /// classes exactly as they arrive.
    PersistentFifo,
}

/// The bin every continuation lands in under `PersistentFifo`.
pub const FIFO_BIN: u32 = 0;

/// All runnable bins, keyed by run-class id. Run-class grouping is implicit:
/// each bin belongs to exactly one run class.
#[derive(Clone, Debug, Default)]
pub struct Scheduler {
    bins: std::collections::HashMap<u32, DoubleBin>,
    mode: SchedulingMode,
    /// Every bin entry ever made, mediated or not. I24 compares it against the
    /// effect log, so a write that reached a bin without producing an effect
    /// shows up as a count the log cannot account for.
    admissions: u64,
}

impl Scheduler {
    pub(crate) fn canonical_fingerprint_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(match self.mode {
            SchedulingMode::RunClassBins => 1,
            SchedulingMode::PersistentFifo => 2,
        });
        out.extend_from_slice(&self.admissions.to_le_bytes());
        let mut bins: Vec<_> = self.bins.iter().collect();
        bins.sort_by_key(|(run_class, _)| **run_class);
        for (run_class, bin) in bins {
            out.extend_from_slice(&run_class.to_le_bytes());
            out.extend_from_slice(&bin.capacity.to_le_bytes());
            out.extend_from_slice(&bin.seq.to_le_bytes());
            for buffer in [&bin.current, &bin.next] {
                out.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
                for entry in buffer {
                    out.push(entry.is_some() as u8);
                    if let Some(reference) = entry {
                        out.extend_from_slice(&reference.to_u64().to_le_bytes());
                    }
                }
            }
        }
        out
    }

    /// A scheduler that bins by the given mode.
    pub fn with_mode(mode: SchedulingMode) -> Scheduler {
        Scheduler {
            bins: std::collections::HashMap::new(),
            mode,
            admissions: 0,
        }
    }

    pub fn mode(&self) -> SchedulingMode {
        self.mode
    }

    /// The bin a continuation of `run_class` belongs to under the current mode.
    pub fn bin_of(&self, run_class: u32) -> u32 {
        match self.mode {
            SchedulingMode::RunClassBins => run_class,
            SchedulingMode::PersistentFifo => FIFO_BIN,
        }
    }

    /// Register a run class with a bin capacity.
    pub fn register_run_class(&mut self, run_class: u32, capacity: u32) {
        self.bins
            .entry(run_class)
            .or_insert_with(|| DoubleBin::new(capacity));
    }

    /// Append a runnable continuation to the next-epoch bin its run class maps
    /// to. Under `PersistentFifo` every run class maps to the same bin, which is
    /// what makes lane groups divergent downstream.
    ///
    /// The [`Committing`] token is the point: a bin entry is the one piece of
    /// state every lane of an epoch writes, so a step that writes it as it runs
    /// cannot be run concurrently with another. Only `kernel::effects` can build
    /// the token, so producing an effect is the only way to reach here
    /// (v0.3 §4.4). `admissions` counts what actually landed, so an
    /// implementation that got in another way is visible in the count without
    /// being visible in the effect log — which is I24 clause 3.
    pub fn enqueue(&mut self, run_class: u32, cont: Ref64, _applying: &Committing) {
        self.enqueue_unmediated(run_class, cont);
    }

    /// The bin write itself, with no proof required. `kernel::raw` is the only
    /// caller that is not the effect applier, and it exists so I24 clause 3 has
    /// a failing case.
    pub(crate) fn enqueue_unmediated(&mut self, run_class: u32, cont: Ref64) {
        let bin = self.bin_of(run_class);
        self.bins
            .entry(bin)
            .or_insert_with(|| DoubleBin::new(u32::MAX))
            .enqueue(cont);
        self.admissions = self.admissions.saturating_add(1);
    }

    /// How many continuations have entered a bin over the scheduler's life.
    pub fn admissions(&self) -> u64 {
        self.admissions
    }

    /// Remove one continuation from whichever bin holds it.
    pub fn remove(&mut self, continuation: Ref64) {
        for bin in self.bins.values_mut() {
            bin.remove_one(continuation);
        }
    }

    /// Remove several continuations, which is what cancelling a process does.
    ///
    /// Costs the number removed times the number of bins, and nothing in the
    /// depth of the queues.
    pub fn remove_all(&mut self, continuations: &HashSet<Ref64>) {
        for continuation in continuations {
            self.remove(*continuation);
        }
    }

    /// Total runnable continuations in current-epoch buffers (all classes).
    pub fn total_runnable(&self) -> usize {
        self.bins.values().map(|b| b.current_len()).sum()
    }

    /// Total continuations still outstanding anywhere: current + next buffers.
    /// Used to detect quiescence, since produced work lands in `next` until the
    /// next epoch-boundary swap (§13).
    pub fn total_pending(&self) -> usize {
        self.bins
            .values()
            .map(|b| b.current_len() + b.next_len())
            .sum()
    }

    /// Per-class counts of all pending (current + next) work, in class order.
    pub fn pending_counts(&self) -> Vec<(u32, usize)> {
        let mut v: Vec<(u32, usize)> = self
            .bins
            .iter()
            .map(|(k, b)| (*k, b.current_len() + b.next_len()))
            .filter(|(_, n)| *n > 0)
            .collect();
        v.sort();
        v
    }

    /// Snapshot of current-epoch runnable counts per bin, for tracing /
    /// cohorting statistics. Sorted, so iteration order is deterministic
    /// regardless of the underlying map.
    pub fn runnable_counts(&self) -> Vec<(u32, usize)> {
        let mut v: Vec<(u32, usize)> = self
            .bins
            .iter()
            .map(|(k, b)| (*k, b.current_len()))
            .filter(|(_, n)| *n > 0)
            .collect();
        v.sort();
        v
    }

    /// Every queued continuation with the bin it sits in, current buffer first,
    /// in deterministic bin order. Used by the semantic invariant checker to
    /// verify that nothing unrunnable is sitting in a runnable bin.
    pub fn pending_entries(&self) -> Vec<(u32, Ref64)> {
        let mut bins: Vec<(&u32, &DoubleBin)> = self.bins.iter().collect();
        bins.sort_by_key(|(k, _)| **k);
        let mut out = Vec::new();
        for (bin, b) in bins {
            for c in b.entries() {
                out.push((*bin, c));
            }
        }
        out
    }

    /// Drain one run class's current-epoch bin for execution.
    pub fn drain(&mut self, run_class: u32) -> Vec<Ref64> {
        match self.bins.get_mut(&run_class) {
            Some(b) => b.drain_current(),
            None => Vec::new(),
        }
    }

    /// Move every `next` buffer into `current` (epoch boundary). Deterministic:
    /// within a run class, order is preserved from insertion.
    pub fn swap_all(&mut self) {
        for b in self.bins.values_mut() {
            b.swap();
        }
    }
}
