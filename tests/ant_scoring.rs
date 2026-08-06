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

// ---- sensing ---------------------------------------------------------------

use soma::experiments::ant_colony::{write_trail, TRAIL_FOOD, TRAIL_HOME};
use soma::experiments::ant_scoring::{
    expected_direction, pack_sensors, sensing_program, unpack_sensors, Sensor, ANT_SENSE_AND_SCORE,
    SENSE_STRIDE,
};
use soma::kernel::AuxBinding;

const GRID_W: u32 = 9;
const GRID_H: u32 = 7;

/// A trail grid with a deterministic, uneven distribution, so the argmax has a
/// different answer in different places rather than one hot cell.
fn grid() -> Vec<u8> {
    let cells = (GRID_W * GRID_H) as usize;
    // Two `u16` channels, laid out exactly as `ant_colony::field_offset` says.
    let mut bytes = vec![0u8; cells * 2 * 2];
    for cell in 0..cells {
        let x = (cell % GRID_W as usize) as u32;
        let y = (cell / GRID_W as usize) as u32;
        write_trail(
            &mut bytes,
            cells,
            TRAIL_FOOD,
            cell,
            ((x * 7 + y * 13) % 251) as u16,
        );
        write_trail(
            &mut bytes,
            cells,
            TRAIL_HOME,
            cell,
            ((x * 29 + y * 5) % 241) as u16,
        );
    }
    bytes
}

/// Every cell of the grid, on both channels — so the batch includes every edge
/// and both corners, which is where the bounds test earns its place.
fn sensors() -> Vec<Sensor> {
    let mut out = Vec::new();
    for channel in [TRAIL_FOOD as u32, TRAIL_HOME as u32] {
        for y in 0..GRID_H {
            for x in 0..GRID_W {
                out.push(Sensor {
                    x,
                    y,
                    width: GRID_W,
                    height: GRID_H,
                    channel,
                });
            }
        }
    }
    out
}

/// Run the sensing collective, with the trail grid bound as the second array.
fn sense(accelerator: &mut dyn BatchBackend, minimum_batch: u32) -> (Vec<u32>, PlacementStats) {
    let batch = sensors();
    let grid_bytes = grid();
    let count = batch.len() as u32;
    let cells = GRID_W * GRID_H;

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, pack_sensors(&batch));
    freeze(&mut kernel, owner, inputs).expect("the owner may freeze its own input array");
    let trail = kernel.create_object(owner, ObjectKind::FrozenArray, grid_bytes);
    freeze(&mut kernel, owner, trail).expect("the owner may freeze the trail grid");

    let (collective, _) = kernel
        .create_batch_evaluate_bound(
            owner,
            ANT_SENSE_AND_SCORE,
            inputs,
            count,
            SENSE_STRIDE,
            // Two `u16` channels of `cells` each, as one flat array of `u16`
            // elements. The channel offset is arithmetic the body does.
            AuxBinding::new(trail, cells * 2, 2),
        )
        .expect("a batch evaluate over two frozen arrays");

    let sensing = sensing_program();
    let mut cpu = CpuReferenceBackend::with(&[&sensing]);
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
    .expect("the sensing body must run");

    let published = kernel.object_bytes(owner, output).unwrap().to_vec();
    (unpack_sensors(&published, batch.len()), stats)
}

#[test]
fn the_body_gathers_the_neighbourhood_the_host_would_have_gathered() {
    // The point of the whole change: the eight reads that `executives::ant_colony`
    // does on the CPU before packing an element are now the body's own, against
    // a second bound array. The reference is computed from the host-side sensing
    // code, so a body with a transposed direction or an off-by-one bound fails
    // here rather than agreeing with itself on two backends.
    let mut cpu = CpuReferenceBackend::with(&[&sensing_program()]);
    let (chosen, _) = sense(&mut cpu, u32::MAX);
    let grid_bytes = grid();

    for (sensor, actual) in sensors().iter().zip(&chosen) {
        assert_eq!(
            *actual,
            expected_direction(sensor, &grid_bytes),
            "wrong direction at ({}, {}) on channel {}",
            sensor.x,
            sensor.y,
            sensor.channel
        );
    }
}

#[test]
fn the_null_is_that_the_grid_actually_decides_something() {
    // A body returning a constant would pass the comparison above if the
    // reference happened to be that constant everywhere. It is not: the batch
    // covers every cell of the grid on both channels, and the answers differ.
    let mut cpu = CpuReferenceBackend::with(&[&sensing_program()]);
    let (chosen, _) = sense(&mut cpu, u32::MAX);
    let distinct: std::collections::BTreeSet<u32> = chosen.iter().copied().collect();
    assert!(
        distinct.len() > 4,
        "the sensing body chose only {distinct:?}, so the grid is not deciding"
    );
}

#[test]
fn a_body_that_names_a_second_array_and_is_given_none_is_refused() {
    // The binding is checked in both directions at the backend boundary. A
    // backend that evaluated this body against the input array alone would
    // return plausible bytes — every reading zero, direction zero — and no
    // other invariant in the machine would notice.
    use soma::executives::batch::{AuxArray, BackendError};

    let sensing = sensing_program();
    let mut cpu = CpuReferenceBackend::with(&[&sensing]);
    let inputs = pack_sensors(&sensors());
    assert_eq!(
        cpu.evaluate(
            ANT_SENSE_AND_SCORE,
            &inputs,
            sensors().len() as u32,
            SENSE_STRIDE
        ),
        Err(BackendError::InvalidInput)
    );

    // And the reverse: an array bound to a body with no name for it means the
    // caller froze something for nothing, which it would never find out.
    let scoring = program();
    let mut scoring_cpu = CpuReferenceBackend::with(&[&scoring]);
    let scored = pack(&sample_batch());
    let junk = vec![0u8; 64];
    assert_eq!(
        scoring_cpu.evaluate_with_aux(
            ANT_MOVEMENT_SCORE,
            &scored,
            sample_batch().len() as u32,
            STRIDE,
            AuxArray::new(&junk, 32, 2),
        ),
        Err(BackendError::InvalidInput)
    );
}

#[test]
fn a_sensing_batch_spills_to_the_cpu_like_any_other() {
    // The second binding travels the spill path too, which is the part most
    // likely to be forgotten: an accelerator that declines has to hand the
    // whole request over, not just the first array.
    let mut cpu = CpuReferenceBackend::with(&[&sensing_program()]);
    let (chosen, stats) = sense(&mut cpu, u32::MAX);
    assert_eq!(stats.cpu_executions, 1);
    assert_eq!(chosen.len(), sensors().len());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_senses_the_trail_grid_and_agrees_with_the_cpu() {
    // This is the claim the module's header could not make. The gather is no
    // longer a host loop feeding a dispatch; it is the dispatch. The grid is a
    // second buffer bound to the same kernel, and the eight reads per ant
    // happen on the GPU.
    use soma::executives::metal::MetalBatchBackend;

    let sensing = sensing_program();
    let mut metal = match MetalBatchBackend::with(&[&sensing]) {
        Ok(backend) => backend,
        Err(_) => return,
    };
    let mut cpu = CpuReferenceBackend::with(&[&sensing]);

    let (on_cpu, _) = sense(&mut cpu, u32::MAX);
    let (on_metal, stats) = sense(&mut metal, 1);

    assert_eq!(
        on_cpu, on_metal,
        "the two backends must sense and decide identically for every ant"
    );
    // And it really ran there, rather than spilling and agreeing trivially.
    assert_eq!(stats.accelerator_executions, 1);
    assert_eq!(stats.cpu_executions, 0);

    // The independent reference too, so agreement is not two lowerings of one
    // mistake.
    let grid_bytes = grid();
    for (sensor, actual) in sensors().iter().zip(&on_metal) {
        assert_eq!(*actual, expected_direction(sensor, &grid_bytes));
    }
}
