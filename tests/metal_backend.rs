#![cfg(all(feature = "metal", target_os = "macos"))]

use soma::abi::{ObjectKind, ProcessMode};
use soma::executives::batch::{execute_with_spill, CpuReferenceBackend, PlacementStats};
use soma::executives::metal::MetalBatchBackend;
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

#[test]
fn real_metal_backend_executes_and_publishes_a_batch() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let values = [0u32, 1, 2, 17, u32::MAX];
    let mut bytes = Vec::new();
    for (index, value) in values.iter().enumerate() {
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes.extend_from_slice(&(index as u32 + 100).to_le_bytes());
    }
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, inputs).unwrap();
    let (collective, _) = kernel
        .create_batch_evaluate_for(owner, 7, inputs, values.len() as u32, 8)
        .unwrap();
    let mut metal = MetalBatchBackend::new().unwrap();
    let mut cpu = CpuReferenceBackend;
    let mut stats = PlacementStats::default();

    let output = execute_with_spill(
        &mut kernel,
        owner,
        collective,
        1,
        &mut metal,
        &mut cpu,
        &mut stats,
    )
    .unwrap();
    let actual = kernel.object_bytes(owner, output).unwrap().to_vec();
    for (index, element) in actual.chunks_exact(8).enumerate() {
        assert_eq!(
            u32::from_le_bytes(element[..4].try_into().unwrap()),
            values[index].wrapping_mul(2).wrapping_add(1)
        );
        assert_eq!(
            u32::from_le_bytes(element[4..].try_into().unwrap()),
            index as u32 + 100
        );
    }
    assert_eq!(stats.accelerator_executions, 1);
    assert_eq!(stats.cpu_executions, 0);
    soma::semantics::invariants::assert_legal(&kernel);

    let gpu_bytes = actual;
    let gpu_trace = kernel.trace_snapshot();
    let mut cpu_kernel = Kernel::new();
    let cpu_owner = cpu_kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mut cpu_inputs_bytes = Vec::new();
    for (index, value) in values.iter().enumerate() {
        cpu_inputs_bytes.extend_from_slice(&value.to_le_bytes());
        cpu_inputs_bytes.extend_from_slice(&(index as u32 + 100).to_le_bytes());
    }
    let cpu_inputs = cpu_kernel.create_object(cpu_owner, ObjectKind::FrozenArray, cpu_inputs_bytes);
    freeze(&mut cpu_kernel, cpu_owner, cpu_inputs).unwrap();
    let (cpu_collective, _) = cpu_kernel
        .create_batch_evaluate_for(cpu_owner, 7, cpu_inputs, values.len() as u32, 8)
        .unwrap();
    let cpu_output = execute_with_spill(
        &mut cpu_kernel,
        cpu_owner,
        cpu_collective,
        u32::MAX,
        &mut metal,
        &mut cpu,
        &mut PlacementStats::default(),
    )
    .unwrap();
    assert_eq!(
        cpu_kernel.object_bytes(cpu_owner, cpu_output).unwrap(),
        gpu_bytes
    );
    assert_eq!(cpu_kernel.trace_snapshot(), gpu_trace);
}
