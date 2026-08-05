//! Movement scoring as a batch evaluator (`experiments::ant_scoring`).
//!
//! The body is generated, so the first thing worth checking is that it computes
//! the argmax the reference computes — a generated body that is wrong is wrong
//! on *both* backends, and backend agreement would happily confirm it. So the
//! CPU backend is checked against an independent reference first, and only then
//! is Metal checked against the CPU.

use soma::abi::{ObjectKind, ProcessMode, Ref64};
use soma::executives::batch::{
    execute_with_spill, BatchBackend, CpuReferenceBackend, PlacementStats,
};
use soma::experiments::ant_scoring::{
    module, pack, program, sample_batch, unpack, ANT_MOVEMENT_SCORE, STRIDE,
};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

/// Run the scoring collective on one backend and return the chosen directions.
fn score(accelerator: &mut dyn BatchBackend, minimum_batch: u32) -> (Vec<u32>, PlacementStats) {
    let batch = sample_batch();
    let bytes = pack(&batch);
    let count = batch.len() as u32;

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, inputs).expect("the owner may freeze its own input array");

    let (collective, _) = kernel
        .create_batch_evaluate_for(owner, ANT_MOVEMENT_SCORE, inputs, count, STRIDE)
        .expect("a batch evaluate over a frozen array");

    let scoring = program();
    let mut cpu = CpuReferenceBackend::with(&[&scoring]);
    let mut stats = PlacementStats::default();
    let output: Ref64 = execute_with_spill(
        &mut kernel,
        owner,
        collective,
        minimum_batch,
        accelerator,
        &mut cpu,
        &mut stats,
    )
    .expect("the scoring body must run");

    soma::semantics::invariants::assert_legal(&kernel);
    let published = kernel
        .object_bytes(owner, output)
        .expect("the owner may read the published output")
        .to_vec();
    (unpack(&published, batch.len()), stats)
}

#[test]
fn the_generated_module_parses() {
    let module = module();
    let evaluator = module
        .evaluators()
        .iter()
        .find(|e| e.id == ANT_MOVEMENT_SCORE)
        .expect("the module declares the scoring evaluator");
    assert_eq!(evaluator.schema.element_stride, STRIDE);
    assert!(
        evaluator.body.is_some(),
        "an evaluator without a body cannot be realized by any backend"
    );
}

/// The body against an independent reference. This is the check that would
/// catch a generator that emitted a consistent but wrong fold.
#[test]
fn the_body_computes_the_argmax() {
    let batch = sample_batch();
    let mut cpu = CpuReferenceBackend::with(&[&program()]);
    let (chosen, _) = score(&mut cpu, u32::MAX);

    assert_eq!(chosen.len(), batch.len());
    for (index, (got, item)) in chosen.iter().zip(batch.iter()).enumerate() {
        assert_eq!(
            *got,
            item.expected(),
            "element {index} with readings {:?}",
            item.readings
        );
    }
}

/// Ties resolve to the first maximum, because `CmpLt` is strict. Pinned
/// explicitly: a body that used `cmple` would pass the argmax test on almost
/// every input and disagree here.
#[test]
fn a_tie_resolves_to_the_first_maximum() {
    let batch = sample_batch();
    let tied = batch
        .iter()
        .position(|item| item.readings.iter().all(|r| *r == batch[1].readings[0]))
        .expect("the sample includes a tie");
    let mut cpu = CpuReferenceBackend::with(&[&program()]);
    let (chosen, _) = score(&mut cpu, u32::MAX);
    assert_eq!(chosen[tied], 0);
}

/// A batch below the accelerator's minimum spills to the CPU rather than being
/// dispatched. The gather boundary is a CPU/accelerator boundary, so the spill
/// path is part of what this workload actually uses.
#[test]
fn a_small_batch_spills_to_the_cpu() {
    let mut cpu = CpuReferenceBackend::with(&[&program()]);
    let (_, stats) = score(&mut cpu, u32::MAX);
    assert!(
        stats.cpu_spills > 0,
        "a batch under the minimum must spill, stats={stats:?}"
    );
}

/// I20 for this body, on real hardware. Both lowerings come from one source, so
/// agreement is a statement about the compiler rather than about two people
/// transcribing the same fold.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_agrees_with_the_cpu_reference() {
    use soma::executives::metal::MetalBatchBackend;

    let scoring = program();
    let mut metal = match MetalBatchBackend::with(&[&scoring]) {
        Ok(backend) => backend,
        // No device on this machine: the claim is untestable here, and
        // pretending otherwise would be worse than skipping.
        Err(_) => return,
    };
    let mut cpu = CpuReferenceBackend::with(&[&scoring]);

    let (on_cpu, _) = score(&mut cpu, u32::MAX);
    let (on_metal, stats) = score(&mut metal, 1);

    assert_eq!(
        on_cpu, on_metal,
        "the two backends must decide identically for every element"
    );
    assert!(
        stats.cpu_spills == 0,
        "the batch should have reached the accelerator, stats={stats:?}"
    );
}
