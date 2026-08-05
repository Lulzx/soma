use soma::abi::{ObjectKind, ProcessMode};
use soma::executives::batch::{
    execute_with_spill, BackendError, BackendKind, BatchBackend, CpuReferenceBackend,
    PlacementStats,
};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

#[derive(Default)]
struct ReferenceAccelerator(CpuReferenceBackend);

impl BatchBackend for ReferenceAccelerator {
    fn kind(&self) -> BackendKind {
        BackendKind::Accelerator
    }

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        self.0
            .evaluate(evaluator_id, inputs, element_count, element_stride)
    }
}

struct UnavailableAccelerator;

impl BatchBackend for UnavailableAccelerator {
    fn kind(&self) -> BackendKind {
        BackendKind::Accelerator
    }

    fn evaluate(
        &mut self,
        _evaluator_id: u32,
        _inputs: &[u8],
        _element_count: u32,
        _element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        Err(BackendError::Unavailable)
    }
}

fn batch(kernel: &mut Kernel, owner: soma::abi::Ref64, values: &[u32]) -> soma::abi::Ref64 {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(kernel, owner, inputs).unwrap();
    kernel
        .create_batch_evaluate_for(owner, 7, inputs, values.len() as u32, 4)
        .unwrap()
        .0
}

fn words(kernel: &mut Kernel, owner: soma::abi::Ref64, object: soma::abi::Ref64) -> Vec<u32> {
    kernel
        .object_bytes(owner, object)
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn accelerator_and_cpu_publish_identical_frozen_results() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let collective = batch(&mut kernel, owner, &[1, 2, 3, 4]);
    let mut accelerator = ReferenceAccelerator::default();
    let mut cpu = CpuReferenceBackend;
    let mut stats = PlacementStats::default();

    let output = execute_with_spill(
        &mut kernel,
        owner,
        collective,
        4,
        &mut accelerator,
        &mut cpu,
        &mut stats,
    )
    .unwrap();

    assert_eq!(words(&mut kernel, owner, output), vec![3, 5, 7, 9]);
    assert_eq!(stats.accelerator_executions, 1);
    assert_eq!(stats.cpu_spills, 0);
    soma::semantics::invariants::assert_legal(&kernel);
}

#[test]
fn unavailable_or_underfilled_accelerator_work_spills_to_cpu() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let unavailable = batch(&mut kernel, owner, &[10, 20, 30, 40]);
    let small = batch(&mut kernel, owner, &[5]);
    let mut accelerator = UnavailableAccelerator;
    let mut cpu = CpuReferenceBackend;
    let mut stats = PlacementStats::default();

    let first = execute_with_spill(
        &mut kernel,
        owner,
        unavailable,
        4,
        &mut accelerator,
        &mut cpu,
        &mut stats,
    )
    .unwrap();
    assert_eq!(words(&mut kernel, owner, first), vec![21, 41, 61, 81]);

    let second = execute_with_spill(
        &mut kernel,
        owner,
        small,
        4,
        &mut ReferenceAccelerator::default(),
        &mut cpu,
        &mut stats,
    )
    .unwrap();
    assert_eq!(words(&mut kernel, owner, second), vec![11]);
    assert_eq!(stats.cpu_executions, 2);
    assert_eq!(stats.cpu_spills, 2);
}

#[test]
fn switching_backend_at_a_collective_boundary_counts_as_migration() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let large = batch(&mut kernel, owner, &[1, 2, 3, 4]);
    let small = batch(&mut kernel, owner, &[5]);
    let mut accelerator = ReferenceAccelerator::default();
    let mut cpu = CpuReferenceBackend;
    let mut stats = PlacementStats::default();

    execute_with_spill(
        &mut kernel,
        owner,
        large,
        4,
        &mut accelerator,
        &mut cpu,
        &mut stats,
    )
    .unwrap();
    execute_with_spill(
        &mut kernel,
        owner,
        small,
        4,
        &mut accelerator,
        &mut cpu,
        &mut stats,
    )
    .unwrap();

    assert_eq!(stats.accelerator_executions, 1);
    assert_eq!(stats.cpu_executions, 1);
    assert_eq!(stats.migrations, 1);
}
