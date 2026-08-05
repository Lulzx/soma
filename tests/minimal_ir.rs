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
        // These cases are about identity and stride validation, which is what
        // I17 covers and which applies to bodyless evaluators too.
        body: None,
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
    assert_eq!(Module::new(vec![]), Err(IrError::EmptyModule));
}

#[test]
fn textual_module_loads_and_links_its_collective() {
    let module = Module::parse(
        "module scoring\n\
         evaluator 7 theoretical-score 8 1 101 ro 2 102 ro\n",
    )
    .unwrap();
    assert_eq!(module.name(), "scoring");

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let loaded = module.load(&mut kernel, owner).unwrap();
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, vec![0; 16]);
    freeze(&mut kernel, owner, inputs).unwrap();
    let (collective, _) = module
        .instantiate_loaded_batch(&mut kernel, owner, loaded, 7, inputs, 2)
        .unwrap();

    assert_eq!(kernel.collective_module(collective).unwrap(), loaded);
    assert_eq!(kernel.collective_evaluator(collective).unwrap(), 7);
    soma::semantics::invariants::assert_legal(&kernel);
}

#[test]
fn textual_module_rejects_ambiguous_surface_syntax() {
    assert_eq!(Module::parse("evaluator 1 x"), Err(IrError::Syntax));
    assert_eq!(
        Module::parse("module x\nevaluator 1 score 8 1 101 maybe 2 102 ro"),
        Err(IrError::InvalidAccess)
    );
}

#[test]
fn loaded_instantiation_rejects_a_different_module_with_reused_ids() {
    let expected = Module::named("expected", vec![evaluator(7, "score", 1)]).unwrap();
    let mut other_evaluator = evaluator(7, "score", 3);
    other_evaluator.schema.element_stride = 4;
    let other = Module::named("other", vec![other_evaluator]).unwrap();
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let loaded_other = other.load(&mut kernel, owner).unwrap();
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, vec![0; 16]);
    freeze(&mut kernel, owner, inputs).unwrap();

    assert_eq!(
        expected.instantiate_loaded_batch(&mut kernel, owner, loaded_other, 7, inputs, 2),
        Err(IrError::ModuleMismatch)
    );
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
