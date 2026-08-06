//! Reclaiming a finished process gives back memory and changes nothing else.
//!
//! The kernel allocates and never releases, so a long run's memory is the sum
//! of everything it has ever done. `kernel::reclaim` releases the private
//! state of a process that has finished — and the reason it needs tests rather
//! than a comment is that "private" is a claim about reachability, and getting
//! it wrong means either holding memory forever or deleting something another
//! process can still name.
//!
//! Both directions are checked: that the same workload publishes the same
//! bytes and the same trace whether or not it reclaims, and that a reference
//! into reclaimed state reports itself stale rather than resolving to whatever
//! took the slot.

use soma::abi::{ObjectKind, ProcessMode, ProcessState, Ref64, StateAccess};
use soma::executives::batch::{execute_with_spill, CpuReferenceBackend, PlacementStats};
use soma::experiments::backend_bench::synthetic_program;
use soma::kernel::ownership::freeze;
use soma::kernel::reclaim::Retained;
use soma::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};

/// Run one worker to completion: a process, one continuation, two epochs.
fn run_a_worker(kernel: &mut Kernel) -> Ref64 {
    let worker = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    kernel
        .create_continuation(
            worker,
            worker,
            ContinuationSpec::new(StateAccess::ReadOnly, 0, 0, Vec::new(), 4),
        )
        .unwrap();
    kernel.run_epoch();
    kernel.run_epoch();
    worker
}

#[test]
fn a_finished_process_gives_back_everything_it_was_holding() {
    let mut kernel = Kernel::new();
    let before = (
        kernel.process_count(),
        kernel.continuation_count(),
        kernel.object_count(),
        kernel.capability_count(),
    );

    for _ in 0..50 {
        run_a_worker(&mut kernel);
    }
    assert_eq!(kernel.terminated_process_count(), 50);

    let reclaimed = kernel.reclaim_finished_processes();
    assert_eq!(reclaimed.processes, 50);
    assert_eq!(reclaimed.continuations, 50);
    // A state object and a frame object each.
    assert_eq!(reclaimed.objects, 100);
    // Terminating already revokes much of a process's own authority, so the
    // count here is dominated by the capabilities *other* spaces held over it.
    assert!(
        reclaimed.capabilities >= 50,
        "reclaiming fifty processes released only {} capabilities",
        reclaimed.capabilities
    );

    let after = (
        kernel.process_count(),
        kernel.continuation_count(),
        kernel.object_count(),
        kernel.capability_count(),
    );
    assert_eq!(
        after, before,
        "reclaiming fifty finished processes did not return the tables to where they started"
    );
    soma::semantics::invariants::assert_legal(&kernel);
}

#[test]
fn reclaiming_repeatedly_keeps_a_long_run_flat() {
    // The point of the whole exercise: a run that reclaims as it goes holds a
    // bounded amount regardless of how much work it has done.
    let mut kernel = Kernel::new();
    for _ in 0..200 {
        run_a_worker(&mut kernel);
    }
    kernel.reclaim_finished_processes();
    let settled = (
        kernel.process_count(),
        kernel.continuation_count(),
        kernel.object_count(),
    );

    for _ in 0..2_000 {
        run_a_worker(&mut kernel);
        kernel.reclaim_finished_processes();
    }
    assert_eq!(
        (
            kernel.process_count(),
            kernel.continuation_count(),
            kernel.object_count()
        ),
        settled,
        "ten times the work left more behind"
    );
}

#[test]
fn a_reference_into_reclaimed_state_is_stale_rather_than_wrong() {
    let mut kernel = Kernel::new();
    let worker = run_a_worker(&mut kernel);
    // A worker whose continuation runs out of frame is Failed rather than
    // Terminated; both are finished, which is what reclamation looks at.
    assert!(matches!(
        kernel.process_state(worker),
        Ok(ProcessState::Terminated | ProcessState::Failed | ProcessState::Cancelled)
    ));

    kernel.reclaim_finished_processes();

    // The slot may be occupied again by now; what must not happen is the old
    // reference resolving to its new occupant.
    for _ in 0..8 {
        run_a_worker(&mut kernel);
    }
    assert!(
        matches!(
            kernel.process_state(worker),
            Err(RuntimeError::Abi(soma::abi::AbiError::StaleReference))
        ),
        "a reference to a reclaimed process resolved to something: {:?}",
        kernel.process_state(worker)
    );
}

#[test]
fn a_process_whose_supervisor_has_not_looked_is_kept() {
    // Reclaiming a child before its supervisor takes the notice would lose the
    // notice, so the pass leaves it alone and says why.
    let mut kernel = Kernel::new();
    let supervisor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let child = kernel
        .create_supervised_process(SYSTEM_PRINCIPAL, supervisor, ProcessMode::Serial)
        .unwrap();
    kernel.cancel_process(SYSTEM_PRINCIPAL, child).unwrap();

    let retained = kernel.retained_processes();
    if retained.iter().any(|(process, _)| *process == child) {
        assert!(
            retained
                .iter()
                .any(|(process, why)| *process == child && *why == Retained::SupervisionPending),
            "a cancelled child was retained for the wrong reason: {retained:?}"
        );
        let reclaimed = kernel.reclaim_finished_processes();
        assert_eq!(
            reclaimed.processes, 0,
            "a child with an unread supervision notice was reclaimed anyway"
        );
    }
}

#[test]
fn published_batches_are_not_reclaimed_with_their_producer() {
    // An output object outlives the process that produced it, and reclaiming
    // the producer must not take the batch with it.
    let program = synthetic_program(850, 2, 8);
    let stride = program.stride();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    let mut kernel = Kernel::new();
    let holder = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let producer = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = vec![9u8; 8 * stride as usize];
    let input = kernel.create_object(producer, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, producer, input).unwrap();
    let (collective, _) = kernel
        .create_batch_evaluate_for(producer, program.id(), input, 8, stride)
        .unwrap();
    let output = execute_with_spill(
        &mut kernel,
        producer,
        collective,
        u32::MAX,
        &mut accelerator,
        &mut cpu,
        &mut PlacementStats::default(),
    )
    .unwrap();
    let published = kernel.object_bytes(producer, output).unwrap().to_vec();

    kernel.cancel_process(SYSTEM_PRINCIPAL, producer).unwrap();
    kernel.reclaim_finished_processes();

    assert_eq!(
        kernel.object_bytes(SYSTEM_PRINCIPAL, output).unwrap(),
        published,
        "reclaiming the producer took its published batch with it"
    );
    let _ = holder;
}

#[test]
fn a_run_that_reclaims_computes_what_a_run_that_does_not_computes() {
    // The claim that matters: reclamation is a memory policy and not a
    // semantic one. Two identical workloads, one reclaiming after every
    // worker, must agree on what they produced.
    fn workload(reclaiming: bool) -> Vec<Vec<u8>> {
        let program = synthetic_program(851, 2, 8);
        let stride = program.stride();
        let mut accelerator = CpuReferenceBackend::with(&[&program]);
        let mut cpu = CpuReferenceBackend::with(&[&program]);

        let mut kernel = Kernel::new();
        let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let mut published = Vec::new();
        for round in 0..12u8 {
            let bytes = vec![round; 8 * stride as usize];
            let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
            freeze(&mut kernel, owner, input).unwrap();
            let (collective, _) = kernel
                .create_batch_evaluate_for(owner, program.id(), input, 8, stride)
                .unwrap();
            let output = execute_with_spill(
                &mut kernel,
                owner,
                collective,
                u32::MAX,
                &mut accelerator,
                &mut cpu,
                &mut PlacementStats::default(),
            )
            .unwrap();
            published.push(kernel.object_bytes(owner, output).unwrap().to_vec());

            run_a_worker(&mut kernel);
            if reclaiming {
                kernel.reclaim_finished_processes();
            }
        }
        soma::semantics::invariants::assert_legal(&kernel);
        published
    }

    assert_eq!(
        workload(true),
        workload(false),
        "reclaiming changed what the run computed"
    );
}
