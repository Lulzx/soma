//! The evaluator body language and I20 (backend agreement).
//!
//! Two obligations, as everywhere else in this suite. The bodies compute what
//! they say they compute, *and* the agreement check catches a backend that
//! does not. A backend-agreement check that cannot fail would restore exactly
//! the hole this workstream existed to close: before v0.3 nothing anywhere
//! verified that a backend applied the function its module named.

use soma::abi::{ObjectKind, ProcessMode};
use soma::compiler::body::{
    BodyError, ElementLayout, EvaluatorProgram, FieldWidth, Op, Store,
};
use soma::compiler::examples;
use soma::compiler::ir::{IrError, Module};
use soma::executives::batch::{
    check_agreement, execute_with_spill, BackendError, BackendKind, BatchBackend,
    CpuReferenceBackend, PlacementStats,
};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

// ---- the language computes what it says ----------------------------------

fn evaluate(program: &EvaluatorProgram, element: &[u8]) -> Vec<u8> {
    let mut output = element.to_vec();
    program.evaluate_element(element, &mut output);
    output
}

fn pair(left: u32, right: u32) -> Vec<u8> {
    let mut bytes = left.to_le_bytes().to_vec();
    bytes.extend_from_slice(&right.to_le_bytes());
    bytes
}

fn word(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(bytes[index * 4..(index + 1) * 4].try_into().unwrap())
}

#[test]
fn the_example_module_parses_and_derives_its_strides() {
    let module = examples::module();
    assert_eq!(module.programs().len(), 4);
    assert_eq!(
        module.program(examples::DOUBLE_PLUS_ONE).unwrap().stride(),
        4
    );
    assert_eq!(module.program(examples::MIN_AND_XOR).unwrap().stride(), 8);
}

#[test]
fn double_plus_one_computes_two_x_plus_one_and_preserves_other_fields() {
    let module = examples::module();
    let program = module.program(examples::DOUBLE_PLUS_ONE_TAGGED).unwrap();
    for (x, tag) in [(0u32, 100u32), (1, 101), (17, 102), (u32::MAX, 103)] {
        let out = evaluate(program, &pair(x, tag));
        assert_eq!(word(&out, 0), x.wrapping_mul(2).wrapping_add(1));
        assert_eq!(word(&out, 1), tag, "a field with no store was modified");
    }
}

#[test]
fn min_and_xor_takes_both_branches_and_writes_both_fields() {
    // The body that is not expressible as a*x + b. Both arms of the select
    // must be reached or the test would pass for a body that ignored the
    // comparison entirely.
    let module = examples::module();
    let program = module.program(examples::MIN_AND_XOR).unwrap();
    for (x, y) in [(1u32, 2u32), (2, 1), (5, 5), (0, u32::MAX)] {
        let out = evaluate(program, &pair(x, y));
        assert_eq!(word(&out, 0), x.min(y));
        assert_eq!(word(&out, 1), x ^ y);
    }
    assert_eq!(program.loaded_fields().len(), 2);
}

#[test]
fn bitmix_masks_its_shift_amounts_to_six_bits() {
    let module = examples::module();
    let program = module.program(examples::BITMIX).unwrap();
    for (x, y) in [(1u32, 0u32), (0x8000_0001, 7), (u32::MAX, u32::MAX)] {
        let out = evaluate(program, &pair(x, y));
        let expected = ((((x as u64) << 13) ^ ((x as u64) >> 7)).wrapping_add(y as u64)) as u32;
        assert_eq!(word(&out, 0), expected);
    }
}

#[test]
fn a_body_is_deterministic() {
    let module = examples::module();
    let program = module.program(examples::BITMIX).unwrap();
    let element = pair(0xDEAD_BEEF, 0x1234_5678);
    let first = evaluate(program, &element);
    for _ in 0..100 {
        assert_eq!(evaluate(program, &element), first);
    }
}

// ---- validation rejects what it should -----------------------------------

#[test]
fn a_body_cannot_read_outside_its_declared_element() {
    // "Reading outside the element is a validation error, not a runtime
    // fault" — so an invalid body can never reach a backend at all.
    let layout = ElementLayout::new(vec![FieldWidth::U32]);
    let result = EvaluatorProgram::new(
        1,
        "out_of_range",
        layout,
        vec![Op::Load(1)],
        vec![Store { field: 0, value: 0 }],
    );
    assert_eq!(result.unwrap_err(), BodyError::FieldOutOfRange);
}

#[test]
fn a_body_cannot_write_outside_its_declared_element() {
    let layout = ElementLayout::new(vec![FieldWidth::U32]);
    let result = EvaluatorProgram::new(
        1,
        "bad_store",
        layout,
        vec![Op::Load(0)],
        vec![Store { field: 4, value: 0 }],
    );
    assert_eq!(result.unwrap_err(), BodyError::FieldOutOfRange);
}

#[test]
fn a_body_cannot_reference_an_instruction_that_has_not_run() {
    let layout = ElementLayout::new(vec![FieldWidth::U32]);
    let result = EvaluatorProgram::new(
        1,
        "forward",
        layout,
        vec![Op::Load(0), Op::Add(0, 5)],
        vec![Store { field: 0, value: 1 }],
    );
    assert_eq!(result.unwrap_err(), BodyError::ForwardReference);
}

#[test]
fn a_body_that_stores_nothing_is_rejected() {
    let layout = ElementLayout::new(vec![FieldWidth::U32]);
    let result = EvaluatorProgram::new(1, "silent", layout, vec![Op::Load(0)], Vec::new());
    assert_eq!(result.unwrap_err(), BodyError::NoStore);
}

#[test]
fn a_layout_whose_stride_contradicts_the_evaluator_is_rejected() {
    // A backend striding differently from its collective reads across element
    // boundaries and returns plausible garbage, so the disagreement has to be
    // caught at module load.
    let source = "\
module mismatch
evaluator 1 wrong 8 10 11 ro 12 13 ro
  field u32
  op 0 load 0
  store 0 0
";
    assert_eq!(
        Module::parse(source).unwrap_err(),
        IrError::Body(BodyError::StrideMismatch)
    );
}

#[test]
fn body_lines_without_an_evaluator_are_rejected() {
    let source = "\
module orphan
  field u32
";
    assert_eq!(Module::parse(source).unwrap_err(), IrError::Syntax);
}

#[test]
fn an_evaluator_may_still_be_declared_without_a_body() {
    // The pre-v0.3 form still loads: naming an evaluator without describing it
    // is legal, it simply means no backend can realize it.
    let source = "\
module bodyless
evaluator 1 opaque 4 10 11 ro 12 13 ro
";
    let module = Module::parse(source).unwrap();
    assert!(module.programs().is_empty());
    assert!(module.program(1).is_none());
}

// ---- I20: backend agreement ----------------------------------------------

/// A backend that computes something plausible but wrong.
struct DivergentBackend;

impl BatchBackend for DivergentBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Accelerator
    }

    fn install(&mut self, _program: &EvaluatorProgram) -> Result<(), BackendError> {
        Ok(())
    }

    fn evaluate(
        &mut self,
        _evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        // `2*x + 2`, off by one from the body it claims to implement. This is
        // exactly the class of defect the old hardcoded backends could not
        // have detected in each other.
        let required = (element_count * element_stride) as usize;
        let mut out = inputs[..required].to_vec();
        for element in out.chunks_exact_mut(element_stride as usize) {
            let value = u32::from_le_bytes(element[..4].try_into().unwrap());
            element[..4].copy_from_slice(&value.wrapping_mul(2).wrapping_add(2).to_le_bytes());
        }
        Ok(out)
    }
}

/// A backend that declines everything.
struct AbstainingBackend;

impl BatchBackend for AbstainingBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Accelerator
    }

    fn install(&mut self, _program: &EvaluatorProgram) -> Result<(), BackendError> {
        Ok(())
    }

    fn evaluate(
        &mut self,
        _evaluator_id: u32,
        _inputs: &[u8],
        _element_count: u32,
        _element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        Err(BackendError::UnsupportedEvaluator)
    }
}

fn inputs_for(count: u32) -> Vec<u8> {
    (0..count)
        .flat_map(|index| pair(index.wrapping_mul(37).wrapping_add(1), index + 100))
        .collect()
}

#[test]
fn agreeing_backends_pass_the_check() {
    let module = examples::module();
    let programs = module.programs();
    let mut first = CpuReferenceBackend::with(&programs);
    let mut second = CpuReferenceBackend::with(&programs);
    let program = module.program(examples::DOUBLE_PLUS_ONE_TAGGED).unwrap();

    let violations = check_agreement(
        program,
        &inputs_for(8),
        8,
        &mut [&mut first as &mut dyn BatchBackend, &mut second],
    );
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn a_divergent_backend_is_caught() {
    // Fault injection for I20. Without this the check could be vacuous and
    // look identical to a working one.
    let module = examples::module();
    let programs = module.programs();
    let mut cpu = CpuReferenceBackend::with(&programs);
    let mut divergent = DivergentBackend;
    let program = module.program(examples::DOUBLE_PLUS_ONE_TAGGED).unwrap();

    let violations = check_agreement(
        program,
        &inputs_for(8),
        8,
        &mut [&mut cpu as &mut dyn BatchBackend, &mut divergent],
    );
    assert_eq!(violations.len(), 1, "a wrong backend was accepted");
    assert_eq!(violations[0].evaluator, examples::DOUBLE_PLUS_ONE_TAGGED);
}

#[test]
fn declining_a_body_is_allowed_but_answering_wrongly_is_not() {
    // The two halves of I20 pull in opposite directions, so both are tested:
    // an honest abstention must pass where a wrong answer fails.
    let module = examples::module();
    let programs = module.programs();
    let mut cpu = CpuReferenceBackend::with(&programs);
    let mut abstaining = AbstainingBackend;
    let program = module.program(examples::MIN_AND_XOR).unwrap();

    let violations = check_agreement(
        program,
        &inputs_for(8),
        8,
        &mut [&mut cpu as &mut dyn BatchBackend, &mut abstaining],
    );
    assert!(violations.is_empty(), "abstention was treated as a defect");
}

#[test]
fn every_example_body_agrees_across_two_independent_interpreters() {
    let module = examples::module();
    let programs = module.programs();
    for program in &programs {
        let mut first = CpuReferenceBackend::with(&programs);
        let mut second = CpuReferenceBackend::with(&programs);
        let count = 16;
        let bytes: Vec<u8> = (0..count)
            .flat_map(|index: u32| {
                let mut element = Vec::new();
                for field in 0..program.layout().fields().len() {
                    element.extend_from_slice(
                        &index
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(field as u32)
                            .to_le_bytes(),
                    );
                }
                element.truncate(program.stride() as usize);
                element
            })
            .collect();
        let violations = check_agreement(
            program,
            &bytes,
            count,
            &mut [&mut first as &mut dyn BatchBackend, &mut second],
        );
        assert!(violations.is_empty(), "{violations:?}");
    }
}

// ---- the collective path uses the body -----------------------------------

#[test]
fn a_collective_publishes_what_its_body_computes() {
    let module = examples::module();
    let programs = module.programs();
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);

    let bytes = inputs_for(4);
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
    freeze(&mut kernel, owner, inputs).unwrap();
    let (collective, _) = kernel
        .create_batch_evaluate_for(owner, examples::MIN_AND_XOR, inputs, 4, 8)
        .unwrap();

    let mut cpu = CpuReferenceBackend::with(&programs);
    let mut accelerator = AbstainingBackend;
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

    let published = kernel.object_bytes(owner, output).unwrap().to_vec();
    for index in 0..4usize {
        let x = word(&bytes, index * 2);
        let y = word(&bytes, index * 2 + 1);
        assert_eq!(word(&published, index * 2), x.min(y));
        assert_eq!(word(&published, index * 2 + 1), x ^ y);
    }
    soma::semantics::invariants::assert_legal(&kernel);
}

#[test]
fn a_collective_whose_evaluator_no_backend_installed_is_refused() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, inputs_for(4));
    freeze(&mut kernel, owner, inputs).unwrap();
    let (collective, _) = kernel
        .create_batch_evaluate_for(owner, 999, inputs, 4, 8)
        .unwrap();

    let mut cpu = CpuReferenceBackend::default();
    let mut accelerator = AbstainingBackend;
    let result = execute_with_spill(
        &mut kernel,
        owner,
        collective,
        u32::MAX,
        &mut accelerator,
        &mut cpu,
        &mut PlacementStats::default(),
    );
    assert!(matches!(result, Err(BackendError::UnsupportedEvaluator)));
}
