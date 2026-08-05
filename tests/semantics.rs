//! Conformance of the reference model to the semantic specification.
//!
//! Two obligations, and the second matters more than the first. The reference
//! model must satisfy the invariants — but an invariant checker that cannot
//! fail proves nothing, so each clause is also shown catching a state that
//! violates it. A green suite where the checker is silently broken would be
//! worse than no checker, because it would read as evidence.

use soma::abi::continuations::ContinuationState;
use soma::abi::objects::OwnershipState;
use soma::abi::{Kind, ObjectKind, ProcessMode, ProcessState, Ref64};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::{create_expand, SearchFrame};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::Kernel;
use soma::scheduler::runnable_bins::SchedulingMode;
use soma::semantics::invariants::{assert_legal, check, Invariant};

fn leaf(kernel: &mut Kernel, process: Ref64) -> Ref64 {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    kernel.create_continuation(process, SEARCH_BRANCH, 0, bytes, DEFAULT_MAX_STEPS)
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
        kernel.cohort_width = 8;
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
    let p = kernel.create_process(ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::ReferenceIntegrity));

    // Delete the frame out from under the continuation.
    let frame = kernel.continuations.get(cont).unwrap().frame;
    kernel.objects.delete(frame).unwrap();
    assert!(violated(&kernel, Invariant::ReferenceIntegrity));
}

#[test]
fn i2_catches_a_continuation_left_running() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::NoContinuationLeftRunning));

    kernel.continuations.get_mut(cont).unwrap().status = ContinuationState::Running;
    assert!(violated(&kernel, Invariant::NoContinuationLeftRunning));
}

#[test]
fn i3_catches_schedulable_work_on_a_terminated_process() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    leaf(&mut kernel, p);
    assert!(!violated(
        &kernel,
        Invariant::ProcessContinuationConsistency
    ));

    kernel.processes.get_mut(p).unwrap().status = ProcessState::Terminated as u32;
    assert!(violated(&kernel, Invariant::ProcessContinuationConsistency));
}

#[test]
fn i4_catches_a_waiter_on_a_settled_future() {
    // This is the liveness bug the kernel had: registering on a future whose
    // waiter list has already been drained parks the continuation forever.
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    let future = kernel.create_future();
    let value = kernel.create_object(ObjectKind::FutureValue, vec![0; 8]);
    kernel.resolve_future(future, value).unwrap();
    assert!(!violated(&kernel, Invariant::FutureSingleAssignment));

    kernel
        .future_waiters
        .entry(future.slot)
        .or_default()
        .push(cont);
    assert!(violated(&kernel, Invariant::FutureSingleAssignment));
}

#[test]
fn i5_catches_an_overfull_mailbox() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    assert!(!violated(&kernel, Invariant::MailboxBound));

    let mailbox = kernel.mailboxes.get_mut(&p.slot).unwrap();
    mailbox.capacity = 0;
    let filler = kernel.create_object(ObjectKind::MessagePayload, vec![0; 8]);
    let mailbox = kernel.mailboxes.get_mut(&p.slot).unwrap();
    mailbox
        .entries
        .push_back(soma::abi::MessageDescriptor::new(p, p, filler));
    assert!(violated(&kernel, Invariant::MailboxBound));
}

#[test]
fn i6_catches_messages_delivered_out_of_send_order() {
    let mut kernel = Kernel::new();
    let sender = kernel.create_process(ProcessMode::Serial);
    let receiver = kernel.create_process(ProcessMode::Serial);
    let payload = kernel.create_object(ObjectKind::MessagePayload, vec![0; 8]);

    for _ in 0..2 {
        kernel
            .ingest_message(sender, receiver, payload, Ref64::NULL)
            .unwrap();
    }
    assert!(!violated(&kernel, Invariant::MessageOrdering));

    // Swap two messages from the same sender.
    let mailbox = kernel.mailboxes.get_mut(&receiver.slot).unwrap();
    mailbox.entries.make_contiguous().swap(0, 1);
    assert!(violated(&kernel, Invariant::MessageOrdering));
}

#[test]
fn i7_catches_unrunnable_work_sitting_in_a_bin() {
    // The other kernel bug: a faulted continuation left live in a runnable bin
    // would be dispatched a second time.
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::SchedulerWellFormed));

    kernel.continuations.get_mut(cont).unwrap().status = ContinuationState::Faulted;
    assert!(violated(&kernel, Invariant::SchedulerWellFormed));
}

#[test]
fn i7_catches_a_continuation_in_the_wrong_bin() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    let cont = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::SchedulerWellFormed));

    // Change the run class without moving the continuation between bins.
    kernel.continuations.get_mut(cont).unwrap().run_class = SEARCH_BRANCH + 1;
    assert!(violated(&kernel, Invariant::SchedulerWellFormed));
}

#[test]
fn i8_catches_two_continuations_sharing_a_frame() {
    let mut kernel = Kernel::new();
    let p = kernel.create_process(ProcessMode::Serial);
    let a = leaf(&mut kernel, p);
    let b = leaf(&mut kernel, p);
    assert!(!violated(&kernel, Invariant::FrameExclusivity));

    let frame = kernel.continuations.get(a).unwrap().frame;
    kernel.continuations.get_mut(b).unwrap().frame = frame;
    assert!(violated(&kernel, Invariant::FrameExclusivity));
}

#[test]
fn i9_catches_an_unpublished_frozen_object() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(ObjectKind::RawBytes, vec![1, 2, 3]);
    soma::kernel::ownership::freeze(&mut kernel, object).unwrap();
    assert!(!violated(&kernel, Invariant::OwnershipMonotonicity));

    kernel.objects.get_mut(object).unwrap().reader_count = 0;
    assert!(violated(&kernel, Invariant::OwnershipMonotonicity));
}

#[test]
fn i11_catches_a_non_monotonic_clock() {
    let mut kernel = Kernel::new();
    create_expand(&mut kernel, 3);
    kernel.run_epoch();
    assert!(!violated(&kernel, Invariant::TraceMonotonicity));

    kernel.trace[1].logical_time = 0;
    assert!(violated(&kernel, Invariant::TraceMonotonicity));
}

#[test]
fn i12_catches_inconsistent_accounting() {
    let mut kernel = Kernel::new();
    assert!(!violated(&kernel, Invariant::AccountingConsistency));

    kernel.accounting.useful_lane_slots = 10;
    kernel.accounting.lane_slots = 4;
    assert!(violated(&kernel, Invariant::AccountingConsistency));
}

// ---- the honest gap ------------------------------------------------------

#[test]
fn capability_authority_is_unenforced_and_the_spec_says_so() {
    // §I10 has no checker because there is nothing to check: the capability
    // table is allocated and never consulted. This test exists to make that
    // gap visible and to fail the moment enforcement is added without the
    // corresponding invariant, rather than letting the spec quietly overstate
    // what the machine guarantees.
    let mut kernel = Kernel::new();
    let object = kernel.create_object(ObjectKind::RawBytes, vec![9]);
    let stranger = kernel.create_process(ProcessMode::Serial);

    // A process that holds no capability over `object` can still mutate it.
    kernel.object_bytes_mut(object).unwrap().push(10);
    assert_eq!(kernel.object_bytes(object).unwrap(), &[9, 10]);
    assert_eq!(
        kernel.capabilities.len(),
        0,
        "no capability was ever needed, granted, or checked"
    );

    // Ownership is likewise advisory: `unique_owner` records intent and stops
    // nothing.
    soma::kernel::ownership::transfer_unique(&mut kernel, object, stranger).unwrap();
    kernel.object_bytes_mut(object).unwrap().push(11);
    assert_eq!(
        soma::kernel::ownership::ownership_state(&kernel, object).unwrap(),
        OwnershipState::UniqueMutable
    );

    // And the machine still reports itself legal, because the invariants make
    // no claim about authority.
    assert_legal(&kernel);
    assert_eq!(Kind::Capability, kernel.capabilities.kind());
}
