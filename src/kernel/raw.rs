//! Deliberate unmediated state access for invariant fault-injection tests.
//!
//! Normal code must use kernel operations. Keeping this unsafe escape hatch in
//! one explicitly named module makes every bypass both explicit and greppable.

use std::collections::HashMap;

use crate::abi::continuations::ContinuationDescriptor;
use crate::abi::{
    ChannelDescriptor, CollectiveDescriptor, DomainDescriptor, ExecutionContract, FutureDescriptor,
    ModuleDescriptor, ObjectDescriptor, ProcessDescriptor, Ref64, TraceEvent,
};
use crate::kernel::accounting::Accounting;
use crate::kernel::effects::EffectRecord;
use crate::kernel::{Kernel, Mailbox, SupervisionQueue};
use crate::scheduler::admission::AdmissionRecord;
use crate::scheduler::runnable_bins::Scheduler;
use crate::table::GenTable;

/// Raw mutable views used only to construct illegal states in tests.
pub struct State<'a> {
    pub epoch: &'a mut u32,
    pub trace: &'a mut Vec<TraceEvent>,
    pub processes: &'a mut GenTable<ProcessDescriptor>,
    pub domains: &'a mut GenTable<DomainDescriptor>,
    pub contracts: &'a mut GenTable<ExecutionContract>,
    pub modules: &'a mut GenTable<ModuleDescriptor>,
    pub objects: &'a mut GenTable<ObjectDescriptor>,
    pub capability_spaces: &'a mut HashMap<u64, crate::kernel::capability_space::CapabilitySpace>,
    pub continuations: &'a mut GenTable<ContinuationDescriptor>,
    pub futures: &'a mut GenTable<FutureDescriptor>,
    pub channels: &'a mut GenTable<ChannelDescriptor>,
    pub collectives: &'a mut GenTable<CollectiveDescriptor>,
    pub mailboxes: &'a mut HashMap<u64, Mailbox>,
    pub future_waiters: &'a mut HashMap<u64, Vec<Ref64>>,
    pub supervision_queues: &'a mut HashMap<u64, SupervisionQueue>,
    pub scheduler: &'a mut Scheduler,
    pub accounting: &'a mut Accounting,
    pub admission_log: &'a mut Vec<AdmissionRecord>,
    pub effect_log: &'a mut Vec<EffectRecord>,
}

/// Borrow all raw kernel storage.
///
/// # Safety
///
/// Mutating these tables can violate every machine invariant. The caller must
/// use this only for deliberate fault injection and must not resume execution
/// from a corrupted state.
pub unsafe fn state(kernel: &mut Kernel) -> State<'_> {
    State {
        epoch: &mut kernel.epoch,
        trace: &mut kernel.trace,
        processes: &mut kernel.processes,
        domains: &mut kernel.domains,
        contracts: &mut kernel.contracts,
        modules: &mut kernel.modules,
        objects: &mut kernel.objects,
        capability_spaces: &mut kernel.capability_spaces,
        continuations: &mut kernel.continuations,
        futures: &mut kernel.futures,
        channels: &mut kernel.channels,
        collectives: &mut kernel.collectives,
        mailboxes: &mut kernel.mailboxes,
        future_waiters: &mut kernel.future_waiters,
        supervision_queues: &mut kernel.supervision_queues,
        scheduler: &mut kernel.scheduler,
        accounting: &mut kernel.accounting,
        admission_log: &mut kernel.admission_log,
        effect_log: &mut kernel.effect_log,
    }
}

/// Put a continuation in a runnable bin without producing an effect for it.
///
/// This is the shape I24 clause 3 forbids: a bin entry the effect log cannot
/// account for. `Scheduler::enqueue` demands a `Committing` token that only the
/// effect applier can build, so this module is the only way to write a bin
/// unmediated — which is what gives the clause a failing case.
///
/// # Safety
///
/// The continuation is binned with no status change and no record. The caller
/// must be constructing an illegal state deliberately.
pub unsafe fn enqueue_unmediated(kernel: &mut Kernel, run_class: u32, cont: Ref64) {
    kernel.scheduler.enqueue_unmediated(run_class, cont);
}
