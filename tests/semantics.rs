//! Conformance of the reference model to the semantic specification.
//!
//! Two obligations, and the second matters more than the first. The reference
//! model must satisfy the invariants — but an invariant checker that cannot
//! fail proves nothing, so each clause is also shown catching a state that
//! violates it. A green suite where the checker is silently broken would be
//! worse than no checker, because it would read as evidence.

use soma::abi::continuations::ContinuationState;
use soma::abi::objects::OwnershipState;
use soma::abi::{EventKind, Kind, ObjectKind, ProcessMode, ProcessState, Ref64, Rights, TraceEvent};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};
use soma::kernel::raw;
use soma::scheduler::runnable_bins::SchedulingMode;
use soma::semantics::invariants::{assert_legal, check, Invariant};

fn leaf(kernel: &mut Kernel, process: Ref64) -> Ref64 {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    kernel
        .create_continuation(process, process, SEARCH_BRANCH, 0, bytes, DEFAULT_MAX_STEPS)
        .unwrap()
}

fn violated(kernel: &Kernel, invariant: Invariant) -> bool {
    check(kernel).iter().any(|v| v.invariant == invariant)
}

// ---- the reference model conforms ----------------------------------------

#[test]
fn a_fresh_machine_is_legal() {
    assert_legal(&Kernel::new());
}

#[test]
fn the_expand_workload_is_legal_at_every_epoch() {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 7);
    assert_legal(&kernel);

    // The invariants are state predicates, so checking after every epoch
    // catches any transition that can produce an illegal state without having
    // to know which transition did it.
    for _ in 0..20 {
        kernel.run_epoch();
        assert_legal(&kernel);
    }
}

#[test]
fn the_search_workload_is_legal_at_every_epoch() {
    for class_count in [1u32, 4] {
        let knobs = ControlKnobs {
            class_count,
            depth: 3,
            branching_factor: 3,
            process_count: 2,
            ..ControlKnobs::default()
        };
        let mut kernel = build(&knobs);
        while kernel.total_pending() > 0 {
            kernel.run_epoch();
            assert_legal(&kernel);
        }
    }
}

#[test]
fn cohorting_and_fifo_modes_are_both_legal() {
    for mode in [SchedulingMode::RunClassBins, SchedulingMode::PersistentFifo] {
        let knobs = ControlKnobs {
            class_count: 4,
            depth: 3,
            ..ControlKnobs::default()
        };
        let mut kernel = Kernel::with_mode(mode);
        kernel.configure_cohorts(8, Default::default());
        let mut kernel = soma::experiments::dynamic_search::build_in(kernel, &knobs);
        while kernel.total_pending() > 0 {
            kernel.run_epoch();
            assert_legal(&kernel);
        }
    }
}

// ---- every clause can actually fail --------------------------------------

#[test]
fn i1_catches_a_dangling_reference() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::ReferenceIntegrity));

    // Delete the frame out from under the continuation.
    let state = unsafe { raw::state(&mut kernel) };
    let frame = state.continuations.get(cont).unwrap().frame;
    state.objects.delete(frame).unwrap();
    assert!(violated(&kernel, Invariant::ReferenceIntegrity));
}

#[test]
fn i2_catches_a_continuation_left_running() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::NoContinuationLeftRunning));

    unsafe { raw::state(&mut kernel) }.continuations.get_mut(cont).unwrap().status =
        ContinuationState::Running;
    assert!(violated(&kernel, Invariant::NoContinuationLeftRunning));
}

#[test]
fn i3_catches_schedulable_work_on_a_terminated_process() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    leaf(&mut kernel, p);
    assert!(!violated(
        &kernel,
        Invariant::ProcessContinuationConsistency
    ));

    unsafe { raw::state(&mut kernel) }.processes.get_mut(p).unwrap().status =
        ProcessState::Terminated as u32;
    assert!(violated(&kernel, Invariant::ProcessContinuationConsistency));
}

#[test]
fn i4_catches_a_waiter_on_a_settled_future() {
    // This is the liveness bug the kernel had: registering on a future whose
    // waiter list has already been drained parks the continuation forever.
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    let future = kernel.create_future(p);
    let value = kernel.create_object(p, ObjectKind::FutureValue, vec![0; 8]);
    kernel.resolve_future(p, future, value).unwrap();
    assert!(!violated(&kernel, Invariant::FutureSingleAssignment));

    unsafe { raw::state(&mut kernel) }
        .future_waiters
        .entry(future.slot)
        .or_default()
        .push(cont);
    assert!(violated(&kernel, Invariant::FutureSingleAssignment));
}

#[test]
fn i5_catches_an_overfull_mailbox() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    assert!(!violated(&kernel, Invariant::MailboxBound));

    let mailbox = unsafe { raw::state(&mut kernel) }.mailboxes.get_mut(&p.slot).unwrap();
    mailbox.capacity = 0;
    let filler = kernel.create_object(p, ObjectKind::MessagePayload, vec![0; 8]);
    let mailbox = unsafe { raw::state(&mut kernel) }.mailboxes.get_mut(&p.slot).unwrap();
    mailbox
        .entries
        .push_back(soma::abi::MessageDescriptor::new(p, p, filler));
    assert!(violated(&kernel, Invariant::MailboxBound));
}

#[test]
fn i6_catches_messages_delivered_out_of_send_order() {
    let mut kernel = Kernel::new();
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let payload = kernel.create_object(sender, ObjectKind::MessagePayload, vec![0; 8]);

    for _ in 0..2 {
        kernel
            .ingest_message(SYSTEM_PRINCIPAL, sender, receiver, payload, Ref64::NULL)
            .unwrap();
    }
    assert!(!violated(&kernel, Invariant::MessageOrdering));

    // Swap two messages from the same sender.
    let mailbox = unsafe { raw::state(&mut kernel) }.mailboxes.get_mut(&receiver.slot).unwrap();
    mailbox.entries.make_contiguous().swap(0, 1);
    assert!(violated(&kernel, Invariant::MessageOrdering));
}

#[test]
fn i7_catches_unrunnable_work_sitting_in_a_bin() {
    // The other kernel bug: a faulted continuation left live in a runnable bin
    // would be dispatched a second time.
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::SchedulerWellFormed));

    unsafe { raw::state(&mut kernel) }.continuations.get_mut(cont).unwrap().status =
        ContinuationState::Faulted;
    assert!(violated(&kernel, Invariant::SchedulerWellFormed));
}

#[test]
fn i7_catches_a_continuation_in_the_wrong_bin() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::SchedulerWellFormed));

    // Change the run class without moving the continuation between bins.
    unsafe { raw::state(&mut kernel) }.continuations.get_mut(cont).unwrap().run_class =
        SEARCH_BRANCH + 1;
    assert!(violated(&kernel, Invariant::SchedulerWellFormed));
}

#[test]
fn i8_catches_two_continuations_sharing_a_frame() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let a = leaf(&mut kernel, p);
    let b = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::FrameExclusivity));

    let state = unsafe { raw::state(&mut kernel) };
    let frame = state.continuations.get(a).unwrap().frame;
    state.continuations.get_mut(b).unwrap().frame = frame;
    assert!(violated(&kernel, Invariant::FrameExclusivity));
}

#[test]
fn i9_catches_an_unpublished_frozen_object() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::RawBytes, vec![1, 2, 3]);
    soma::kernel::ownership::freeze(&mut kernel, SYSTEM_PRINCIPAL, object).unwrap();
    assert!(!violated(&kernel, Invariant::OwnershipMonotonicity));

    unsafe { raw::state(&mut kernel) }.objects.get_mut(object).unwrap().reader_count = 0;
    assert!(violated(&kernel, Invariant::OwnershipMonotonicity));
}

#[test]
fn i10a_catches_capability_amplification() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::RawBytes, vec![1, 2, 3, 4]);
    let parent = kernel
        .find_capability(Ref64::NULL, object, Rights::READ)
        .unwrap();
    let child = kernel
        .derive_capability(Ref64::NULL, parent, Rights::READ, 1, 2)
        .unwrap();
    assert!(!violated(&kernel, Invariant::CapabilityAttenuation));

    unsafe { raw::state(&mut kernel) }
        .capability_spaces
        .get_mut(&0)
        .unwrap()
        .get_mut(child)
        .unwrap()
        .rights |= Rights::SEND;
    assert!(violated(&kernel, Invariant::CapabilityAttenuation));
}

#[test]
fn i10b_catches_a_dead_capability_parent() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::RawBytes, vec![1, 2, 3, 4]);
    let parent = kernel
        .find_capability(Ref64::NULL, object, Rights::READ)
        .unwrap();
    kernel
        .derive_capability(Ref64::NULL, parent, Rights::READ, 0, 4)
        .unwrap();
    assert!(!violated(&kernel, Invariant::CapabilityIntegrity));

    unsafe { raw::state(&mut kernel) }
        .capability_spaces
        .get_mut(&0)
        .unwrap()
        .delete(parent)
        .unwrap();
    assert!(violated(&kernel, Invariant::CapabilityIntegrity));
}

#[test]
fn i11_catches_a_non_monotonic_clock() {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 3);
    kernel.run_epoch();
    assert!(!violated(&kernel, Invariant::TraceMonotonicity));

    unsafe { raw::state(&mut kernel) }.trace[1].logical_time = 0;
    assert!(violated(&kernel, Invariant::TraceMonotonicity));
}

#[test]
fn i12_catches_inconsistent_accounting() {
    let mut kernel = Kernel::new();
    assert!(!violated(&kernel, Invariant::AccountingConsistency));

    let accounting = unsafe { raw::state(&mut kernel) }.accounting;
    accounting.useful_lane_slots = 10;
    accounting.lane_slots = 4;
    assert!(violated(&kernel, Invariant::AccountingConsistency));
}

// ---- I10c ---------------------------------------------------------------

#[test]
fn i10c_records_grants_denials_and_authorized_effects() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::RawBytes, vec![9]);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    // A process that holds no capability over `object` can neither read,
    // mutate, freeze, nor transfer it.
    assert!(matches!(
        kernel.object_bytes(stranger, object),
        Err(soma::kernel::RuntimeError::AuthorityDenied)
    ));
    assert!(matches!(
        kernel.object_bytes_mut(stranger, object),
        Err(soma::kernel::RuntimeError::AuthorityDenied)
    ));
    assert_eq!(
        kernel.trace_events().last().unwrap().event_kind,
        EventKind::AuthorityDenied
    );
    assert_eq!(kernel.object_bytes(SYSTEM_PRINCIPAL, object).unwrap(), &[9]);
    assert_eq!(
        kernel.find_capability(stranger, object, Rights::WRITE),
        None,
        "the stranger has no authority over this object"
    );

    assert!(matches!(
        soma::kernel::ownership::freeze(&mut kernel, stranger, object),
        Err(soma::kernel::RuntimeError::AuthorityDenied)
    ));
    assert!(matches!(
        soma::kernel::ownership::transfer_unique(&mut kernel, stranger, object, stranger),
        Err(soma::kernel::RuntimeError::AuthorityDenied)
    ));
    assert_eq!(
        soma::kernel::ownership::ownership_state(&kernel, object).unwrap(),
        OwnershipState::UniqueMutable
    );

    kernel.object_bytes_mut(SYSTEM_PRINCIPAL, object).unwrap()[0] = 10;
    let pair = &kernel.trace_events()[kernel.trace_events().len() - 2..];
    assert_eq!(pair[0].event_kind, EventKind::AuthorityGranted);
    assert_eq!(pair[1].event_kind, EventKind::AuthorityEffect);
    assert_eq!(pair[0].process, pair[1].process);
    assert_eq!(pair[0].continuation, pair[1].continuation);
    assert_eq!(pair[0].run_class, pair[1].run_class);

    assert_legal(&kernel);
    assert_eq!(Kind::Capability, kernel.capability_table_kind());
}

#[test]
fn i10c_catches_an_effect_without_an_adjacent_grant() {
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let object = kernel.create_object(actor, ObjectKind::RawBytes, vec![0]);
    assert!(!violated(&kernel, Invariant::NoUnauthorizedEffect));

    let trace = unsafe { raw::state(&mut kernel) }.trace;
    let logical_time = trace.last().map(|event| event.logical_time + 1).unwrap_or(1);
    trace.push(TraceEvent::new(
        logical_time,
        0,
        EventKind::AuthorityEffect,
        actor,
        object,
        Rights::WRITE,
    ));

    assert!(violated(&kernel, Invariant::NoUnauthorizedEffect));
}
