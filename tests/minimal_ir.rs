use soma::abi::{ObjectKind, ProcessMode, StateAccess};
use soma::compiler::ir::{BatchEvaluator, FrozenArraySchema, IrError, Module, ResumePoint};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

fn evaluator(id: u32, name: &str, resume_base: u32) -> BatchEvaluator {
    BatchEvaluator {
        id,
        name: name.into(),
        schema: FrozenArraySchema { element_stride: 8 },
        entry: ResumePoint {
            id: resume_base,
            run_class: resume_base + 100,
            state_access: StateAccess::ReadOnly,
        },
        completion: ResumePoint {
            id: resume_base + 1,
            run_class: resume_base + 101,
            state_access: StateAccess::ReadOnly,
        },
    }
}

#[test]
fn module_rejects_ambiguous_evaluator_and_resume_identities() {
    assert_eq!(
        Module::new(vec![evaluator(1, "score", 1), evaluator(1, "other", 3)]),
        Err(IrError::DuplicateEvaluator)
    );
    assert_eq!(
        Module::new(vec![evaluator(1, "score", 1), evaluator(2, "other", 1)]),
        Err(IrError::DuplicateResumePoint)
    );
}

#[test]
fn module_rejects_empty_shapes_and_names() {
    let mut invalid = evaluator(1, "", 1);
    assert_eq!(Module::new(vec![invalid.clone()]), Err(IrError::EmptyName));
    invalid.name = "score".into();
    invalid.schema.element_stride = 0;
    assert_eq!(Module::new(vec![invalid]), Err(IrError::ZeroStride));
}

#[test]
fn evaluator_ir_instantiates_the_named_batch_collective() {
    let module = Module::new(vec![evaluator(7, "theoretical-score", 1)]).unwrap();
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, vec![0; 24]);
    freeze(&mut kernel, owner, inputs).unwrap();

    let (collective, _) = module
        .instantiate_batch(&mut kernel, owner, 7, inputs, 3)
        .unwrap();
    assert_eq!(kernel.collective_evaluator(collective).unwrap(), 7);
    assert_eq!(
        module.instantiate_batch(&mut kernel, owner, 99, inputs, 3),
        Err(IrError::UnknownEvaluator)
    );
}
