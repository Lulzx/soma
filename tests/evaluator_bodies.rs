//! The evaluator body language and I20 (backend agreement).
//!
//! Two obligations, as everywhere else in this suite. The bodies compute what
//! they say they compute, *and* the agreement check catches a backend that
//! does not. A backend-agreement check that cannot fail would restore exactly
//! the hole this workstream existed to close: before v0.3 nothing anywhere
//! verified that a backend applied the function its module named.

use soma::abi::{ObjectKind, ProcessMode};
use soma::compiler::body::{BodyError, ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};
use soma::compiler::examples;
use soma::compiler::ir::{IrError, Module};
use soma::executives::batch::{
    check_agreement, execute_with_spill, BackendError, BackendKind, BatchBackend,
    CpuReferenceBackend, PlacementStats,
};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};

// ---- the language computes what it says ----------------------------------

/// Evaluate a whole array through the reference backend.
///
/// Deliberately the production path rather than a loop written here: a test
/// harness with its own element loop can pass while the loop a backend
/// actually runs is wrong, which is the same shape of hole as two backends
/// hardcoding one constant and agreeing.
fn evaluate_array(program: &EvaluatorProgram, inputs: &[u8]) -> Vec<u8> {
    let stride = program.stride();
    let count = inputs.len() as u32 / stride;
    CpuReferenceBackend::with(&[program])
        .evaluate(program.id(), inputs, count, stride)
        .expect("the reference backend evaluates a body it was given")
}

/// Evaluate a one-element array.
fn evaluate(program: &EvaluatorProgram, element: &[u8]) -> Vec<u8> {
    evaluate_array(program, element)
}

/// Pack `(field0, field1)` pairs into a two-`u32` array.
fn array(pairs: &[(u32, u32)]) -> Vec<u8> {
    pairs.iter().flat_map(|(a, b)| pair(*a, *b)).collect()
}

/// Field `field` of element `element` in a two-`u32` array.
fn at(bytes: &[u8], element: usize, field: usize) -> u32 {
    word(bytes, element * 2 + field)
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
    assert_eq!(module.programs().len(), 8);
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

// ---- gather: reading an element other than your own -----------------------

#[test]
fn neighbour_max_reads_both_neighbours() {
    let module = examples::module();
    let program = module.program(examples::NEIGHBOUR_MAX).unwrap();
    let input = array(&[(3, 0), (1, 0), (9, 0), (2, 0), (5, 0)]);
    let out = evaluate_array(program, &input);

    // Interior elements see both sides. Element 1 has neighbours 3 and 9.
    assert_eq!(at(&out, 1, 1), 9);
    assert_eq!(at(&out, 2, 1), 9);
    assert_eq!(at(&out, 3, 1), 9);
    // A body with no gather could not produce any of those: every one of them
    // exceeds the element's own field 0.
    assert!(at(&out, 1, 1) > at(&out, 1, 0));
}

#[test]
fn the_edges_clamp_instead_of_wrapping_or_faulting() {
    // The left edge is the interesting one: `0 - 1` wraps to u64::MAX, which
    // would clamp to the *last* element and silently make the array a ring.
    // The body substitutes its own index there, so element 0 sees only itself
    // and its right neighbour.
    let module = examples::module();
    let program = module.program(examples::NEIGHBOUR_MAX).unwrap();
    let input = array(&[(3, 0), (1, 0), (9, 0), (2, 0), (5, 0)]);
    let out = evaluate_array(program, &input);

    // Element 0: max(3, 1) = 3. If the left index had wrapped to the end it
    // would have seen 5 and answered 5.
    assert_eq!(at(&out, 0, 1), 3);
    // Element 4: max(2, 5) = 5, with the right index clamping back to itself.
    assert_eq!(at(&out, 4, 1), 5);
}

#[test]
fn a_single_element_array_gathers_only_itself() {
    let module = examples::module();
    let program = module.program(examples::NEIGHBOUR_MAX).unwrap();
    let out = evaluate_array(program, &array(&[(7, 0)]));
    assert_eq!(at(&out, 0, 1), 7);
}

#[test]
fn a_gather_reads_the_input_and_never_the_output() {
    // Every element overwrites the field its neighbours are reading. If a
    // gather saw the output array, the answer would depend on the order the
    // elements ran in — and the whole point of the frozen input is that it
    // cannot. A reversal is the sharpest case: element 0 reads element 4's
    // payload while element 4 is overwriting its own.
    let module = examples::module();
    let program = module.program(examples::PERMUTE).unwrap();
    let input = array(&[(4, 10), (3, 20), (2, 30), (1, 40), (0, 50)]);
    let out = evaluate_array(program, &input);
    assert_eq!(
        (0..5).map(|i| at(&out, i, 1)).collect::<Vec<_>>(),
        vec![50, 40, 30, 20, 10]
    );

    // Same body, elements evaluated back to front. Identical bytes, which is
    // the property that lets a cohort run these lanes in any order at all.
    let stride = program.stride() as usize;
    let mut reversed = input.clone();
    for index in (0..5u32).rev() {
        let start = index as usize * stride;
        program.evaluate_at(&input, 5, index, &mut reversed[start..start + stride]);
    }
    assert_eq!(reversed, out);
}

#[test]
fn an_out_of_range_gather_index_clamps_to_the_last_element() {
    // Totality under a computed index: an index past the end is not a fault.
    let module = examples::module();
    let program = module.program(examples::PERMUTE).unwrap();
    let out = evaluate_array(program, &array(&[(99, 10), (0, 20), (u32::MAX, 30)]));
    assert_eq!(at(&out, 0, 1), 30, "clamped to the last element");
    assert_eq!(at(&out, 1, 1), 10);
    assert_eq!(at(&out, 2, 1), 30);
}

#[test]
fn index_gives_each_element_its_own_position() {
    // The one op whose value differs across the lanes of a cohort.
    let source = "\
module positions
evaluator 1 position 4 10 11 ro 12 13 ro
  field u32
  op 0 index
  store 0 0
";
    let module = Module::parse(source).unwrap();
    let program = module.program(1).unwrap();
    let out = evaluate_array(program, &[0u8; 16]);
    assert_eq!(
        (0..4).map(|i| word(&out, i)).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn a_gather_counts_as_observing_the_field_it_reads() {
    let module = examples::module();
    let program = module.program(examples::PERMUTE).unwrap();
    // Field 0 is loaded, field 1 is only ever gathered. Both are observed.
    assert_eq!(
        program.loaded_fields().iter().copied().collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(program.gathers());
    assert!(!module.program(examples::BITMIX).unwrap().gathers());
}

// ---- validation rejects what it should -----------------------------------

#[test]
fn a_gather_cannot_read_a_field_outside_the_declared_element() {
    // The gathered field is static, so it is checked exactly as a load's is.
    // Only the element index is dynamic, and that clamps.
    let layout = ElementLayout::new(vec![FieldWidth::U32]);
    let result = EvaluatorProgram::new(
        1,
        "gather_out_of_range",
        layout,
        vec![Op::Index, Op::Gather(0, 1)],
        vec![Store { field: 0, value: 1 }],
    );
    assert_eq!(result.unwrap_err(), BodyError::FieldOutOfRange);
}

#[test]
fn a_gather_cannot_take_its_index_from_an_instruction_that_has_not_run() {
    let layout = ElementLayout::new(vec![FieldWidth::U32]);
    let result = EvaluatorProgram::new(
        1,
        "gather_forward",
        layout,
        vec![Op::Gather(3, 0)],
        vec![Store { field: 0, value: 0 }],
    );
    assert_eq!(result.unwrap_err(), BodyError::ForwardReference);
}

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

// ---- loops ----------------------------------------------------------------

/// A body's declared and evaluated results, for the loop examples.
fn loop_program(id: u32) -> EvaluatorProgram {
    examples::module()
        .program(id)
        .expect("the example module declares the loop bodies")
        .clone()
}

#[test]
fn a_loop_accumulates_across_iterations() {
    let program = loop_program(examples::WINDOW_SUM);
    // Ten elements, field 0 counting up. The window is eight wide and clamps
    // at the end, so the last elements sum a run that repeats the final value
    // — which is the same clamp rule a gather already had, now visible eight
    // times in one body.
    let inputs = array(&[
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (10, 0),
    ]);
    let out = evaluate_array(&program, &inputs);

    let values: Vec<u32> = (1..=10).collect();
    for element in 0..10usize {
        let expected: u32 = (0..8)
            .map(|step| values[(element + step).min(values.len() - 1)])
            .sum();
        assert_eq!(
            at(&out, element, 1),
            expected,
            "element {element} summed its window wrongly"
        );
    }
    // The null: without it, a body that returned its own field eight times
    // would pass the loop above for the first element by coincidence.
    assert_ne!(at(&out, 0, 1), at(&out, 1, 1));
}

#[test]
fn a_loop_carries_nothing_but_its_locals() {
    // Field 0 is untouched by the body, so the loop wrote only through its
    // locals and the store. A body that leaked an iteration's value into the
    // element would show up here.
    let program = loop_program(examples::WINDOW_SUM);
    let inputs = array(&[(3, 77), (4, 88)]);
    let out = evaluate_array(&program, &inputs);
    assert_eq!(at(&out, 0, 0), 3);
    assert_eq!(at(&out, 1, 0), 4);
}

#[test]
fn an_early_exit_leaves_on_a_different_iteration_per_element() {
    let program = loop_program(examples::RUN_LENGTH);
    // Runs of non-zero values of length 3, then 0, then 2, then 0.
    let inputs = array(&[
        (5, 0),
        (6, 0),
        (7, 0),
        (0, 0),
        (9, 0),
        (8, 0),
        (0, 0),
        (0, 0),
    ]);
    let out = evaluate_array(&program, &inputs);

    assert_eq!(at(&out, 0, 1), 3, "element 0 counts its own run");
    assert_eq!(at(&out, 1, 1), 2);
    assert_eq!(at(&out, 2, 1), 1);
    assert_eq!(
        at(&out, 3, 1),
        0,
        "an element that is itself zero counts none"
    );
    assert_eq!(at(&out, 4, 1), 2);
    assert_eq!(at(&out, 5, 1), 1);
    assert_eq!(at(&out, 6, 1), 0);

    // This is the property the body exists to demonstrate: the elements did
    // not all run the same number of iterations. Without it the body is just
    // an expensive constant.
    let counts: std::collections::BTreeSet<u32> =
        (0..7).map(|element| at(&out, element, 1)).collect();
    assert!(
        counts.len() > 2,
        "every element left on the same iteration, so nothing diverged: {counts:?}"
    );
}

#[test]
fn divergence_is_a_property_of_a_body_and_the_examples_disagree_about_it() {
    let module = examples::module();
    // The pre-loop bodies and the counted loop are lockstep across a cohort.
    for id in [
        examples::DOUBLE_PLUS_ONE,
        examples::MIN_AND_XOR,
        examples::NEIGHBOUR_MAX,
        examples::WINDOW_SUM,
    ] {
        assert!(
            module.program(id).unwrap().is_uniform(),
            "{id} should be uniform"
        );
    }
    // A static trip count is not divergence; leaving early is.
    assert!(!module.program(examples::RUN_LENGTH).unwrap().is_uniform());
}

#[test]
fn a_loops_worst_case_length_is_known_before_it_runs() {
    let module = examples::module();
    let straight = module.program(examples::MIN_AND_XOR).unwrap();
    assert_eq!(straight.step_bound(), straight.ops().len() as u64);

    // Eight iterations of a nine-instruction body plus the four instructions
    // outside it: the bound multiplies rather than counts.
    let looping = module.program(examples::WINDOW_SUM).unwrap();
    assert!(
        looping.step_bound() > looping.ops().len() as u64,
        "the bound did not multiply out the loop"
    );
    assert!(looping.step_bound() <= soma::compiler::body::MAX_STEPS);
}

// ---- what the loop rules reject ------------------------------------------

fn build(locals: u32, ops: Vec<Op>, stores: Vec<Store>) -> Result<EvaluatorProgram, BodyError> {
    EvaluatorProgram::with_locals(
        1,
        "candidate",
        ElementLayout::new(vec![FieldWidth::U32]),
        locals,
        ops,
        stores,
    )
}

#[test]
fn a_loop_that_is_never_closed_is_rejected() {
    assert_eq!(
        build(
            0,
            vec![Op::Const(1), Op::Repeat(2), Op::Const(3)],
            vec![Store { field: 0, value: 0 }],
        ),
        Err(BodyError::UnbalancedLoop)
    );
}

#[test]
fn closing_a_loop_that_was_never_opened_is_rejected() {
    assert_eq!(
        build(
            0,
            vec![Op::Const(1), Op::EndRepeat],
            vec![Store { field: 0, value: 0 }],
        ),
        Err(BodyError::UnbalancedLoop)
    );
}

#[test]
fn leaving_a_loop_from_outside_every_loop_is_rejected() {
    // There is nothing for it to leave. The language has no `return`, so this
    // is a typo rather than a shorter way to write one.
    assert_eq!(
        build(
            0,
            vec![Op::Const(1), Op::BreakIf(0)],
            vec![Store { field: 0, value: 0 }],
        ),
        Err(BodyError::UnbalancedLoop)
    );
}

#[test]
fn a_value_computed_inside_a_loop_cannot_be_read_after_it() {
    // This is the rule that replaces phi nodes. Op 2's value belongs to
    // whichever iteration produced it, so op 4 must not be able to name it —
    // and neither may a store.
    assert_eq!(
        build(
            0,
            vec![
                Op::Const(1),
                Op::Repeat(3),
                Op::Const(7),
                Op::EndRepeat,
                Op::Add(0, 2),
            ],
            vec![Store { field: 0, value: 4 }],
        ),
        Err(BodyError::EscapingValue)
    );
    assert_eq!(
        build(
            0,
            vec![Op::Const(1), Op::Repeat(3), Op::Const(7), Op::EndRepeat],
            vec![Store { field: 0, value: 2 }],
        ),
        Err(BodyError::EscapingValue)
    );
}

#[test]
fn the_null_is_that_a_local_carries_the_same_value_out_legally() {
    // Otherwise the rejection above would read as "a loop cannot produce
    // anything", which is not what it says.
    let program = build(
        1,
        vec![
            Op::Const(7),
            Op::Repeat(3),
            Op::Const(7),
            Op::Set(0, 2),
            Op::EndRepeat,
            Op::Get(0),
        ],
        vec![Store { field: 0, value: 5 }],
    )
    .expect("a local may carry a value out of a loop");
    let out = evaluate(&program, &7u32.to_le_bytes());
    assert_eq!(word(&out, 0), 7);
}

#[test]
fn reading_an_enclosing_scopes_value_from_inside_a_loop_is_allowed() {
    // The escape rule is a prefix test, not an equality test: a value computed
    // before the loop does not change while the loop runs, so reading it in is
    // fine. Equality would have made loops nearly useless.
    let program = build(
        1,
        vec![
            Op::Const(5),
            Op::Repeat(3),
            Op::Get(0),
            Op::Add(2, 0),
            Op::Set(0, 3),
            Op::EndRepeat,
            Op::Get(0),
        ],
        vec![Store { field: 0, value: 6 }],
    )
    .expect("a loop body may read a value from outside the loop");
    let out = evaluate(&program, &0u32.to_le_bytes());
    assert_eq!(word(&out, 0), 15);
}

#[test]
fn a_local_outside_the_declared_set_is_rejected() {
    assert_eq!(
        build(
            1,
            vec![Op::Const(1), Op::Set(4, 0)],
            vec![Store { field: 0, value: 0 }],
        ),
        Err(BodyError::LocalOutOfRange)
    );
}

#[test]
fn a_loop_with_no_iterations_is_rejected() {
    assert_eq!(
        build(
            0,
            vec![Op::Const(1), Op::Repeat(0), Op::Const(2), Op::EndRepeat],
            vec![Store { field: 0, value: 0 }],
        ),
        Err(BodyError::EmptyLoop)
    );
}

#[test]
fn a_body_whose_unrolled_length_exceeds_the_bound_is_rejected() {
    // Totality is the reason the trip count is static, so the check that it
    // buys something has to exist. Three nested loops of 2048 multiply past
    // MAX_STEPS; the same body with one fewer level is accepted, which is the
    // null showing the rejection is about the bound and not about nesting.
    let ops = |levels: usize| {
        let mut ops = vec![Op::Const(1)];
        for _ in 0..levels {
            ops.push(Op::Repeat(2048));
        }
        ops.push(Op::Const(2));
        for _ in 0..levels {
            ops.push(Op::EndRepeat);
        }
        ops
    };
    assert_eq!(
        build(0, ops(3), vec![Store { field: 0, value: 0 }]),
        Err(BodyError::Unbounded)
    );
    assert!(build(0, ops(1), vec![Store { field: 0, value: 0 }]).is_ok());
}

#[test]
fn a_nested_loop_runs_the_product_of_its_trip_counts() {
    let program = build(
        1,
        vec![
            Op::Const(1),
            Op::Repeat(3),
            Op::Repeat(4),
            Op::Get(0),
            Op::Add(3, 0),
            Op::Set(0, 4),
            Op::EndRepeat,
            Op::EndRepeat,
            Op::Get(0),
        ],
        vec![Store { field: 0, value: 8 }],
    )
    .expect("nested loops validate");
    let out = evaluate(&program, &0u32.to_le_bytes());
    assert_eq!(word(&out, 0), 12);
}

#[test]
fn the_generated_metal_declares_its_values_before_its_loops() {
    // A `ulong v7` declared inside a C loop body stops existing at the closing
    // brace, which would make a local's whole purpose unreachable. The check
    // is textual because the alternative is a GPU, and the GPU check exists in
    // `tests/metal_backend.rs` behind a feature.
    let source = examples::module()
        .program(examples::WINDOW_SUM)
        .unwrap()
        .metal_source();
    let first_declaration = source.find("ulong v0").expect("values are declared");
    let first_loop = source.find("for (uint t").expect("the loop is emitted");
    assert!(
        first_declaration < first_loop,
        "a value was declared inside the loop that uses it"
    );
    assert!(source.contains("l0 = "), "locals are assigned");
    assert!(
        source.matches('{').count() == source.matches('}').count(),
        "the generated kernel's braces do not balance:\n{source}"
    );
}

// ---- the second binding ---------------------------------------------------

#[test]
fn gathering_from_an_array_the_body_never_declared_is_rejected() {
    assert_eq!(
        EvaluatorProgram::bound(
            1,
            "no_aux_declared",
            ElementLayout::new(vec![FieldWidth::U32]),
            None,
            0,
            vec![Op::Const(0), Op::GatherAux(0, 0)],
            vec![Store { field: 0, value: 1 }],
        ),
        Err(BodyError::AuxMismatch)
    );
}

#[test]
fn declaring_an_array_the_body_never_reads_is_rejected() {
    // The other direction, and the one that would otherwise survive: the
    // collective is well-formed, the caller freezes an array for the duration,
    // and nothing ever reads it.
    assert_eq!(
        EvaluatorProgram::bound(
            1,
            "aux_unread",
            ElementLayout::new(vec![FieldWidth::U32]),
            Some(ElementLayout::new(vec![FieldWidth::U16])),
            0,
            vec![Op::Load(0)],
            vec![Store { field: 0, value: 0 }],
        ),
        Err(BodyError::AuxMismatch)
    );
}

#[test]
fn an_aux_field_outside_the_aux_layout_is_rejected() {
    // Checked against the *aux* layout, not the element layout. The element
    // here has four fields and the aux has one, so field 2 is in range for the
    // wrong one — which is the mistake this catches.
    assert_eq!(
        EvaluatorProgram::bound(
            1,
            "aux_field",
            ElementLayout::new(vec![
                FieldWidth::U32,
                FieldWidth::U32,
                FieldWidth::U32,
                FieldWidth::U32
            ]),
            Some(ElementLayout::new(vec![FieldWidth::U16])),
            0,
            vec![Op::Const(0), Op::GatherAux(0, 2)],
            vec![Store { field: 0, value: 1 }],
        ),
        Err(BodyError::FieldOutOfRange)
    );
}

#[test]
fn an_aux_gather_past_the_end_clamps_to_the_last_element() {
    // The same totality rule `Gather` obeys, against the other array. A
    // runtime fault here would break the step budget the same way.
    let program = EvaluatorProgram::bound(
        1,
        "aux_clamp",
        ElementLayout::new(vec![FieldWidth::U32]),
        Some(ElementLayout::new(vec![FieldWidth::U16])),
        0,
        vec![Op::Const(9_999), Op::GatherAux(0, 0)],
        vec![Store { field: 0, value: 1 }],
    )
    .expect("an aux gather validates");

    let aux: Vec<u8> = [7u16, 8, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out = vec![0u8; 4];
    program.evaluate_bound(
        soma::compiler::body::Arrays::of(&0u32.to_le_bytes(), 1).with_aux(&aux, 3),
        0,
        &mut out,
    );
    assert_eq!(word(&out, 0), 9, "an out-of-range aux index reads the last");
}

// ---- threaded evaluation --------------------------------------------------

/// Every example body, evaluated on one thread and on several, must produce
/// identical bytes.
///
/// This is I20 with the two backends being one backend at two thread counts,
/// and it is the clause that says the body language's rules were worth having.
/// A body reads the frozen input and never the output and writes only its own
/// element — stated for I19, because a body that could observe another
/// element's store would make the result depend on the schedule — so an
/// element's output is a function of the input and its index and chunking
/// cannot move a byte.
#[test]
fn threading_the_elements_does_not_change_a_byte() {
    let module = examples::module();
    for program in module.programs() {
        let stride = program.stride() as usize;
        // Enough elements that every thread count below splits them unevenly,
        // so a chunk-boundary defect has somewhere to happen.
        let count = 37u32;
        let inputs: Vec<u8> = (0..count)
            .flat_map(|value| {
                let mut element = Vec::new();
                for field in 0..(stride / 4) {
                    element.extend_from_slice(
                        &(value.wrapping_mul(2_654_435_761).rotate_left(field as u32 * 7))
                            .to_le_bytes(),
                    );
                }
                element.resize(stride, 0);
                element
            })
            .collect();

        let expected = CpuReferenceBackend::with(&[program])
            .evaluate(program.id(), &inputs, count, program.stride())
            .expect("the single-threaded reference evaluates");

        for threads in [2usize, 3, 8, 64] {
            let actual = CpuReferenceBackend::with(&[program])
                .with_threads(threads)
                .evaluate(program.id(), &inputs, count, program.stride())
                .expect("the threaded backend evaluates");
            assert_eq!(
                actual,
                expected,
                "body {} disagreed with itself at {threads} threads",
                program.name()
            );
        }
    }
}

#[test]
fn a_gathering_body_still_reads_the_whole_array_from_every_thread() {
    // The sharpest case for chunking: `permute` has every element read a field
    // its neighbours are overwriting. A thread that gathered from its own
    // chunk's output — or from a chunk-local view of the input — would return
    // an answer that depends on how the work was split, which is exactly the
    // I19 failure the "reads the input, never the output" rule exists to stop.
    let module = examples::module();
    let program = module.program(examples::PERMUTE).unwrap();
    let inputs = array(&[(4, 40), (3, 30), (2, 20), (1, 10), (0, 0)]);

    let expected = CpuReferenceBackend::with(&[program])
        .evaluate(program.id(), &inputs, 5, program.stride())
        .unwrap();
    // Field 1 becomes the field 1 of the element field 0 names: a reversal.
    assert_eq!(
        (0..5).map(|e| at(&expected, e, 1)).collect::<Vec<_>>(),
        vec![0, 10, 20, 30, 40]
    );

    for threads in [2usize, 5] {
        let actual = CpuReferenceBackend::with(&[program])
            .with_threads(threads)
            .evaluate(program.id(), &inputs, 5, program.stride())
            .unwrap();
        assert_eq!(actual, expected, "permute differed at {threads} threads");
    }
}

#[test]
fn the_null_is_that_the_thread_count_is_real() {
    // Without this, everything above passes for a `with_threads` that ignores
    // its argument. A count above the element count must still cover every
    // element exactly once, which is where an off-by-one in the chunking shows
    // up as a short or duplicated output rather than as a wrong value.
    let module = examples::module();
    let program = module.program(examples::DOUBLE_PLUS_ONE_TAGGED).unwrap();
    assert_eq!(
        CpuReferenceBackend::with(&[program])
            .with_threads(9)
            .threads(),
        9
    );
    assert_eq!(
        CpuReferenceBackend::with(&[program])
            .with_threads(0)
            .threads(),
        1,
        "a zero thread count must fall back to one rather than doing nothing"
    );

    let inputs = array(&[(1, 0), (2, 0), (3, 0)]);
    let out = CpuReferenceBackend::with(&[program])
        .with_threads(64)
        .evaluate(program.id(), &inputs, 3, program.stride())
        .unwrap();
    assert_eq!(out.len(), inputs.len(), "the threaded run lost elements");
    assert_eq!(
        (0..3).map(|e| at(&out, e, 0)).collect::<Vec<_>>(),
        vec![3, 5, 7]
    );
}

// ---- threading an epoch's collectives -------------------------------------

/// One epoch's worth of independent requests over the same body.
fn epoch_requests<'a>(
    program: &EvaluatorProgram,
    inputs: &'a [Vec<u8>],
) -> Vec<soma::executives::batch::BatchRequest<'a>> {
    inputs
        .iter()
        .map(|bytes| soma::executives::batch::BatchRequest {
            evaluator_id: program.id(),
            inputs: bytes,
            aux: soma::executives::batch::AuxArray::NONE,
            element_count: (bytes.len() / program.stride() as usize) as u32,
            element_stride: program.stride(),
        })
        .collect()
}

#[test]
fn running_an_epochs_collectives_in_parallel_publishes_the_same_bytes() {
    // The parallelism that element threading cannot supply. An epoch of many
    // small cohorts has too few elements per cohort to fill a thread, and the
    // cohorts are independent — each names its own frozen input and its own
    // output — so the level worth splitting is the epoch, not the element.
    let module = examples::module();
    let program = module.program(examples::MIN_AND_XOR).unwrap();

    // Deliberately uneven: 17 requests across 4 threads leaves a short group,
    // and each request is small enough that element threading would not help.
    let inputs: Vec<Vec<u8>> = (0..17u32)
        .map(|seed| {
            array(&[
                (seed, seed.wrapping_mul(7)),
                (seed + 1, seed.wrapping_mul(13)),
                (seed + 2, seed.wrapping_mul(29)),
            ])
        })
        .collect();
    let requests = epoch_requests(program, &inputs);

    let sequential = CpuReferenceBackend::with(&[program])
        .evaluate_epoch(&requests)
        .expect("the sequential epoch runs");

    for threads in [2usize, 4, 32] {
        let parallel = CpuReferenceBackend::with(&[program])
            .with_threads(threads)
            .evaluate_epoch(&requests)
            .expect("the threaded epoch runs");
        assert_eq!(
            parallel.len(),
            sequential.len(),
            "{threads} threads returned a different number of results"
        );
        for (index, (a, b)) in parallel.iter().zip(&sequential).enumerate() {
            assert_eq!(
                a.as_slice(),
                b.as_slice(),
                "request {index} differed at {threads} threads"
            );
        }
    }
}

#[test]
fn a_threaded_epoch_keeps_its_results_in_request_order() {
    // Results are indexed by request: the caller publishes result `i` into
    // collective `i`. A grouping that returned results in completion order
    // would publish every collective's output into the wrong object, and every
    // byte would still look plausible.
    let module = examples::module();
    let program = module.program(examples::DOUBLE_PLUS_ONE_TAGGED).unwrap();

    // Each request has a distinguishable answer, so an order swap is visible.
    let inputs: Vec<Vec<u8>> = (0..12u32).map(|seed| array(&[(seed, 0)])).collect();
    let requests = epoch_requests(program, &inputs);

    let results = CpuReferenceBackend::with(&[program])
        .with_threads(5)
        .evaluate_epoch(&requests)
        .unwrap();

    for (seed, payload) in results.iter().enumerate() {
        assert_eq!(
            at(payload.as_slice(), 0, 0),
            (seed as u32) * 2 + 1,
            "request {seed} came back in the wrong slot"
        );
    }
}

#[test]
fn one_failed_request_fails_the_whole_threaded_epoch() {
    // The contract the sequential path already has. A partial epoch leaves the
    // caller holding some published outputs and some unstarted collectives with
    // no way to say which, so the threaded path must not weaken it into
    // per-request results.
    let module = examples::module();
    let program = module.program(examples::MIN_AND_XOR).unwrap();
    let good = array(&[(1, 2)]);

    let mut requests = epoch_requests(program, std::slice::from_ref(&good));
    requests.extend(epoch_requests(program, std::slice::from_ref(&good)));
    // An evaluator the backend was never given.
    requests[1].evaluator_id = 9_999;
    // Enough requests that the bad one lands in a different group.
    while requests.len() < 8 {
        requests.push(requests[0]);
    }

    assert_eq!(
        CpuReferenceBackend::with(&[program])
            .with_threads(4)
            .evaluate_epoch(&requests)
            .err(),
        Some(BackendError::UnsupportedEvaluator)
    );
}
