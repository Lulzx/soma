//! Publishing a batch must cost the same in an old kernel as in a new one.
//!
//! This is a performance guard rather than an invariant, and it is here
//! because the thing it guards is not visible as one. Authorization used to
//! answer "may this actor do X to object Y?" by scanning the actor's whole
//! capability space, and revocation found a capability's children the same
//! way. Both are correct at any size and both are linear in what the kernel
//! already holds, so a run publishing n batches did O(n²) work while every
//! test passed: nothing about the result changes, only how long it takes.
//!
//! `examples/kernel_overhead` measured the cost of publishing one more cohort
//! at 6µs into an empty kernel and 485µs after sixteen thousand. With the
//! target and parent indexes it is flat at about 2.4µs.
//!
//! The bound below is deliberately loose. What it has to separate is "roughly
//! constant" from "linear in a kernel eighty times larger", which is two
//! orders of magnitude apart, so a factor that a loaded machine cannot trip
//! still catches the regression that matters.

use std::time::{Duration, Instant};

use soma::abi::{ObjectKind, ProcessMode, Ref64};
use soma::executives::batch::{execute_with_spill, CpuReferenceBackend, PlacementStats};
use soma::experiments::backend_bench::synthetic_program;
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

/// Small elements, so the measurement is bookkeeping rather than bytes.
const ELEMENTS: u32 = 8;
const SAMPLE: usize = 200;
const AGED: usize = 8_000;

/// What one run of the loop needs, gathered so the helper takes a request
/// rather than eight positional arguments.
struct Workload {
    owner: Ref64,
    input: Ref64,
    program_id: u32,
    stride: u32,
}

fn publish_batches(
    kernel: &mut Kernel,
    workload: &Workload,
    accelerator: &mut CpuReferenceBackend,
    cpu: &mut CpuReferenceBackend,
    count: usize,
) -> Duration {
    let Workload {
        owner,
        input,
        program_id,
        stride,
    } = *workload;
    let start = Instant::now();
    for _ in 0..count {
        let (collective, _) = kernel
            .create_batch_evaluate_for(owner, program_id, input, ELEMENTS, stride)
            .unwrap();
        execute_with_spill(
            kernel,
            owner,
            collective,
            u32::MAX,
            accelerator,
            cpu,
            &mut PlacementStats::default(),
        )
        .unwrap();
    }
    start.elapsed()
}

#[test]
fn publishing_a_batch_does_not_get_slower_as_the_kernel_fills_up() {
    let program = synthetic_program(820, 2, 8);
    let stride = program.stride();
    let mut accelerator = CpuReferenceBackend::with(&[&program]);
    let mut cpu = CpuReferenceBackend::with(&[&program]);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = vec![3u8; (ELEMENTS * stride) as usize];
    let input = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, input).unwrap();

    let workload = Workload {
        owner,
        input,
        program_id: program.id(),
        stride,
    };
    let mut publish = |kernel: &mut Kernel, count: usize| {
        publish_batches(kernel, &workload, &mut accelerator, &mut cpu, count)
    };

    // Warm the allocator and the code paths before the first measurement, so
    // the comparison is not first-run against steady-state.
    publish(&mut kernel, SAMPLE);
    let fresh = publish(&mut kernel, SAMPLE);

    publish(&mut kernel, AGED);
    let aged = publish(&mut kernel, SAMPLE);

    let capabilities = kernel.capability_count();
    assert!(
        capabilities > 3 * AGED,
        "the kernel did not actually accumulate capabilities ({capabilities}), \
         so this test is not measuring what it claims"
    );
    assert!(
        aged.as_secs_f64() < fresh.as_secs_f64() * 8.0,
        "publishing got {:.1}x slower after {AGED} more batches ({:?} -> {:?}); \
         something on the publication path is scanning the whole kernel again",
        aged.as_secs_f64() / fresh.as_secs_f64(),
        fresh,
        aged,
    );
}
