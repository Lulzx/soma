#![cfg(all(feature = "metal", target_os = "macos"))]

//! I20 (backend agreement) against real Apple GPU hardware.
//!
//! Before v0.3 this file compared a hand-written MSL kernel computing
//! `2*x + 1` with a hand-written Rust function computing `2*x + 1`. They
//! agreed, which was evidence that two people had transcribed one constant
//! correctly and no evidence at all about a compiler. Both sides are now
//! generated from `compiler::examples::SOURCE`, so agreement is a statement
//! about the lowering.

use soma::abi::{ObjectKind, ProcessMode, Ref64};
use soma::compiler::examples;
use soma::executives::batch::{
    execute_with_spill, BatchBackend, CpuReferenceBackend, PlacementStats,
};
use soma::executives::metal::MetalBatchBackend;
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

/// Inputs chosen to reach the edges the body language defines away:
/// wraparound on multiply, both arms of a `select`, and a shift whose result
/// leaves the field width.
fn sample_elements() -> Vec<u8> {
    let pairs: [(u32, u32); 7] = [
        (0, 0),
        (1, 2),
        (2, 1),
        (17, 17),
        (u32::MAX, 1),
        (1, u32::MAX),
        (0x8000_0001, 0x7FFF_FFFF),
    ];
    let mut bytes = Vec::new();
    for (left, right) in pairs {
        bytes.extend_from_slice(&left.to_le_bytes());
        bytes.extend_from_slice(&right.to_le_bytes());
    }
    bytes
}

fn frozen_input(kernel: &mut Kernel, owner: Ref64, bytes: Vec<u8>) -> Ref64 {
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(kernel, owner, inputs).unwrap();
    inputs
}

/// Run one evaluator end to end on the given backend pair and return the
/// published bytes plus the semantic trace.
fn run(
    evaluator: u32,
    stride: u32,
    minimum_accelerator_batch: u32,
    accelerator: &mut dyn BatchBackend,
    cpu: &mut dyn BatchBackend,
) -> (Vec<u8>, Vec<soma::kernel::TraceSnapshotRow>, PlacementStats) {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = sample_elements();
    let count = bytes.len() as u32 / stride;
    let inputs = frozen_input(&mut kernel, owner, bytes);
    let (collective, _) = kernel
        .create_batch_evaluate_for(owner, evaluator, inputs, count, stride)
        .unwrap();
    let mut stats = PlacementStats::default();
    let output = execute_with_spill(
        &mut kernel,
        owner,
        collective,
        minimum_accelerator_batch,
        accelerator,
        cpu,
        &mut stats,
    )
    .unwrap();
    soma::semantics::invariants::assert_legal(&kernel);
    let published = kernel.object_bytes(owner, output).unwrap().to_vec();
    (published, kernel.trace_snapshot(), stats)
}

#[test]
fn every_example_body_agrees_between_metal_and_the_cpu_interpreter() {
    let module = examples::module();
    let programs = module.programs();
    let mut metal = MetalBatchBackend::with(&programs).unwrap();
    let mut cpu = CpuReferenceBackend::with(&programs);

    // The single-field body has a different stride, so it is covered by
    // `tests/evaluator_bodies.rs`; the GPU path here uses the 8-byte elements.
    for evaluator in [
        examples::DOUBLE_PLUS_ONE_TAGGED,
        examples::MIN_AND_XOR,
        examples::BITMIX,
    ] {
        // Batch large enough to reach the accelerator.
        let (gpu_bytes, gpu_trace, gpu_stats) = run(evaluator, 8, 1, &mut metal, &mut cpu);
        assert_eq!(gpu_stats.accelerator_executions, 1);
        assert_eq!(gpu_stats.cpu_executions, 0);

        // Batch too small for the accelerator, so the same work spills to CPU.
        let (cpu_bytes, cpu_trace, cpu_stats) = run(evaluator, 8, u32::MAX, &mut metal, &mut cpu);
        assert_eq!(cpu_stats.cpu_executions, 1);

        assert_eq!(
            gpu_bytes, cpu_bytes,
            "evaluator {evaluator} disagreed between Metal and the CPU interpreter"
        );
        assert_eq!(
            gpu_trace, cpu_trace,
            "evaluator {evaluator} published differently depending on placement"
        );
    }
}

#[test]
fn metal_rejects_an_evaluator_it_was_never_given() {
    // I20's other half: a backend that cannot realize a body must say so
    // rather than apply whatever it last compiled.
    let module = examples::module();
    let one = module.program(examples::DOUBLE_PLUS_ONE_TAGGED).unwrap();
    let mut metal = MetalBatchBackend::with(&[one]).unwrap();
    let mut cpu = CpuReferenceBackend::with(&module.programs());

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = frozen_input(&mut kernel, owner, sample_elements());
    let (collective, _) = kernel
        .create_batch_evaluate_for(owner, examples::MIN_AND_XOR, inputs, 7, 8)
        .unwrap();

    let result = execute_with_spill(
        &mut kernel,
        owner,
        collective,
        1,
        &mut metal,
        &mut cpu,
        &mut PlacementStats::default(),
    );
    assert!(
        matches!(
            result,
            Err(soma::executives::batch::BackendError::UnsupportedEvaluator)
        ),
        "an uninstalled evaluator was silently evaluated: {result:?}"
    );
}

#[test]
fn a_reused_buffer_does_not_leak_the_previous_batch_into_a_smaller_one() {
    // The backend grows one input/output pair to the largest batch it has
    // seen and reuses it. That makes batch order observable in a way it was
    // not when every call allocated: a short batch runs against buffers still
    // holding a long batch's bytes, and reading one element too many would
    // publish the previous collective's results as this one's.
    let module = examples::module();
    let programs = module.programs();
    let mut metal = MetalBatchBackend::with(&programs).unwrap();
    let mut cpu = CpuReferenceBackend::with(&programs);

    let stride = 8u32;
    let long: Vec<u8> = (0..64u32)
        .flat_map(|value| {
            let mut element = value.to_le_bytes().to_vec();
            element.extend_from_slice(&(value ^ 0xFFFF).to_le_bytes());
            element
        })
        .collect();
    let short = long[..(3 * stride as usize)].to_vec();

    for evaluator in [
        examples::DOUBLE_PLUS_ONE_TAGGED,
        examples::MIN_AND_XOR,
        examples::BITMIX,
    ] {
        metal
            .evaluate(evaluator, &long, 64, stride)
            .expect("long batch runs");
        let after_long = metal
            .evaluate(evaluator, &short, 3, stride)
            .expect("short batch runs");

        let expected = cpu.evaluate(evaluator, &short, 3, stride).unwrap();
        assert_eq!(
            after_long.len(),
            expected.len(),
            "evaluator {evaluator} returned the reused buffer's capacity, not its batch"
        );
        assert_eq!(
            after_long, expected,
            "evaluator {evaluator} leaked the previous batch through a reused buffer"
        );
    }
}
