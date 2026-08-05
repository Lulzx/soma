//! Deliberate unmediated state access for invariant fault-injection tests.
//!
//! Normal code must use kernel operations. Keeping this unsafe escape hatch in
//! one explicitly named module makes every bypass both explicit and greppable.

use std::collections::HashMap;

use crate::abi::capabilities::CapabilityEntry;
use crate::abi::continuations::ContinuationDescriptor;
use crate::abi::{
    ChannelDescriptor, CollectiveDescriptor, DomainDescriptor, ExecutionContract, FutureDescriptor,
    ModuleDescriptor, ObjectDescriptor, ProcessDescriptor, Ref64, TraceEvent,
};
use crate::kernel::accounting::Accounting;
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
    pub capability_spaces: &'a mut HashMap<u64, GenTable<CapabilityEntry>>,
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
    }
}
