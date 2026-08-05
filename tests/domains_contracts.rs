use soma::abi::{
    DeterminismPolicy, ExecutionContract, PlacementPolicy, ProcessMode, Shape, StateAccess,
};
use soma::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::semantics::invariants::assert_legal;

#[test]
fn domains_enforce_membership_authority_and_creation_quota() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let domain = kernel
        .create_domain(owner, kernel.root_domain(), 1)
        .unwrap();
    let child = kernel
        .create_process_in_domain(owner, domain, ProcessMode::Serial)
        .unwrap();

    assert_eq!(kernel.process_domain(child).unwrap(), domain);
    assert_eq!(kernel.domain_processes_created(domain).unwrap(), 1);
    assert_eq!(
        kernel.create_process_in_domain(owner, domain, ProcessMode::Serial),
        Err(RuntimeError::DomainQuotaExceeded)
    );
    assert_eq!(
        kernel.create_process_in_domain(stranger, domain, ProcessMode::Serial),
        Err(RuntimeError::AuthorityDenied)
    );
    assert_legal(&kernel);
}

#[test]
fn contracts_bound_steps_and_frame_bytes() {
    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mut descriptor = ExecutionContract::new(Shape::Scalar, PlacementPolicy::Any);
    descriptor.maximum_steps = 4;
    descriptor.local_memory_bytes = 8;
    let contract = kernel
        .create_execution_contract(process, descriptor)
        .unwrap();

    let continuation = kernel
        .create_contracted_continuation(
            process,
            process,
            contract,
            ContinuationSpec::new(StateAccess::ReadOnly, 700, 0, vec![0; 8], 4),
        )
        .unwrap();
    assert_eq!(
        kernel.continuation_contract(continuation).unwrap(),
        contract
    );
    assert_eq!(
        kernel.create_contracted_continuation(
            process,
            process,
            contract,
            ContinuationSpec::new(StateAccess::ReadOnly, 701, 0, vec![0; 8], 5),
        ),
        Err(RuntimeError::InvalidContract)
    );
    assert_eq!(
        kernel.create_contracted_continuation(
            process,
            process,
            contract,
            ContinuationSpec::new(StateAccess::ReadOnly, 702, 0, vec![0; 9], 4),
        ),
        Err(RuntimeError::InvalidContract)
    );
    assert_legal(&kernel);
}

#[test]
fn hardware_and_wall_clock_requirements_are_rejected() {
    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    let mut hardware = ExecutionContract::new(Shape::Scalar, PlacementPolicy::RequireGpu);
    assert_eq!(
        kernel.create_execution_contract(process, hardware.clone()),
        Err(RuntimeError::InvalidContract)
    );
    hardware.placement_policy = PlacementPolicy::Any;
    hardware.shape = Shape::Lanes;
    assert_eq!(
        kernel.create_execution_contract(process, hardware.clone()),
        Err(RuntimeError::InvalidContract)
    );
    hardware.shape = Shape::Scalar;
    hardware.deadline_ns = 1;
    assert_eq!(
        kernel.create_execution_contract(process, hardware.clone()),
        Err(RuntimeError::InvalidContract)
    );
    hardware.deadline_ns = 0;
    hardware.determinism_policy = DeterminismPolicy::Relaxed;
    assert_eq!(
        kernel.create_execution_contract(process, hardware),
        Err(RuntimeError::InvalidContract)
    );
    assert_legal(&kernel);
}

#[test]
fn a_contract_is_actor_relative_authority() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let contract = kernel
        .create_execution_contract(
            owner,
            ExecutionContract::new(Shape::Scalar, PlacementPolicy::Any),
        )
        .unwrap();
    assert_eq!(
        kernel.create_contracted_continuation(
            stranger,
            stranger,
            contract,
            ContinuationSpec::new(StateAccess::ReadOnly, 703, 0, vec![], 1),
        ),
        Err(RuntimeError::AuthorityDenied)
    );
    assert_legal(&kernel);
}
