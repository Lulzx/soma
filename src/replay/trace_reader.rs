//! Deterministic replay support (§21).
//!
//! Deterministic replay consumes initial object state, external inputs, message
//! order, placement decisions, cohort membership, and random seeds. Phase 1
//! records a full `TraceEvent` stream; the functions here make it inspectable
//! and comparable so two identically-seeded runs can be proven identical.

use crate::abi::{EventKind, TraceEvent};
use crate::kernel::Kernel;

/// Summary counts per event kind (process/continuation-level view).
pub fn summarize(kernel: &Kernel) -> std::collections::BTreeMap<EventKind, usize> {
    let mut m = std::collections::BTreeMap::new();
    for t in kernel.trace_events() {
        *m.entry(t.event_kind).or_insert(0) += 1;
    }
    m
}

/// Total number of trace events recorded.
pub fn event_count(kernel: &Kernel) -> usize {
    kernel.trace_events().len()
}

/// All events of a given kind, in order.
pub fn events_of<'a>(
    kernel: &'a Kernel,
    kind: EventKind,
) -> impl Iterator<Item = &'a TraceEvent> + 'a {
    kernel
        .trace_events()
        .iter()
        .filter(move |t| t.event_kind == kind)
}

/// Whether two runs produced identical traces (used to assert determinism).
pub fn same_trace(a: &Kernel, b: &Kernel) -> bool {
    a.trace_snapshot() == b.trace_snapshot()
}
