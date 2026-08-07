//! Deterministic bounded binary32 evaluator subset and I20 agreement.

use soma::compiler::body::BodyError;
use soma::compiler::surface::{compile_evaluator, SurfaceErrorKind};
use soma::executives::batch::{BatchBackend, CpuReferenceBackend};

const SOURCE: &str = r#"
field f32
field f32
field f32
field f32
let x = load 0
let y = load 1
let sum = fadd x y
let product = fmul x y
store 2 sum
store 3 product
"#;

fn program() -> soma::compiler::body::EvaluatorProgram {
    compile_evaluator(40_001, "f32_pair", SOURCE).unwrap()
}

fn inputs() -> Vec<u8> {
    [
        (1.5f32.to_bits(), 2.25f32.to_bits()),
        (f32::MAX.to_bits(), 2.0f32.to_bits()),
        (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits()),
        (f32::INFINITY.to_bits(), 0.0f32.to_bits()),
        (0x8000_0000, 0),
        (0x7fa1_2345, 3.0f32.to_bits()),
        (0x0000_0001, 0x0000_0001),
        (0x0080_0000, 0x3f00_0000),
    ]
    .into_iter()
    .flat_map(|(x, y)| [x, y, 0, 0].into_iter().flat_map(u32::to_le_bytes))
    .collect()
}

fn canonical(bits: u32) -> u32 {
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        0x7fc0_0000
    } else if magnitude < 0x0080_0000 {
        0
    } else {
        bits
    }
}

#[test]
fn reference_f32_is_exact_for_finite_and_special_values() {
    let program = program();
    let input = inputs();
    let mut reference = CpuReferenceBackend::with(&[&program]);
    let output = reference
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    for (lane, bytes) in output.chunks_exact(16).enumerate() {
        let x = f32::from_bits(u32::from_le_bytes(bytes[0..4].try_into().unwrap()));
        let y = f32::from_bits(u32::from_le_bytes(bytes[4..8].try_into().unwrap()));
        let sum = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let product = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let x = f32::from_bits(canonical(x.to_bits()));
        let y = f32::from_bits(canonical(y.to_bits()));
        assert_eq!(sum, canonical((x + y).to_bits()), "sum lane {lane}");
        assert_eq!(product, canonical((x * y).to_bits()), "product lane {lane}");
    }
    assert_eq!(
        u32::from_le_bytes(output[2 * 16 + 8..2 * 16 + 12].try_into().unwrap()),
        0x7fc0_0000
    );
    assert_eq!(
        u32::from_le_bytes(output[4 * 16 + 8..4 * 16 + 12].try_into().unwrap()),
        0
    );
}

#[test]
fn validation_rejects_integer_float_mixing_and_wrong_store_fields() {
    let mixed = compile_evaluator(
        40_002,
        "mixed",
        "field f32\nlet x = load 0\nlet one = const 1\nlet bad = fadd x one\nstore 0 bad\n",
    )
    .unwrap_err();
    assert_eq!(mixed.kind, SurfaceErrorKind::Body(BodyError::TypeMismatch));
    let wrong_store = compile_evaluator(
        40_003,
        "store",
        "field u32\nlet x = fconst 1.0\nstore 0 x\n",
    )
    .unwrap_err();
    assert_eq!(
        wrong_store.kind,
        SurfaceErrorKind::Body(BodyError::TypeMismatch)
    );
    let float_condition = compile_evaluator(
        40_004,
        "float_condition",
        "field f32\nlet x = load 0\nlet bad = fselect x x x\nstore 0 bad\n",
    )
    .unwrap_err();
    assert_eq!(
        float_condition.kind,
        SurfaceErrorKind::Body(BodyError::TypeMismatch)
    );
    let integer_arms = compile_evaluator(
        40_005,
        "integer_arms",
        "field f32\nlet x = load 0\nlet one = const 1\nlet bad = fselect one one one\nstore 0 bad\n",
    )
    .unwrap_err();
    assert_eq!(
        integer_arms.kind,
        SurfaceErrorKind::Body(BodyError::TypeMismatch)
    );
    let wrong_float_local = compile_evaluator(
        40_006,
        "wrong_float_local",
        "field f32\nlocal f32 total\nlet one = const 1\nset total one\nlet result = get total\nstore 0 result\n",
    )
    .unwrap_err();
    assert_eq!(
        wrong_float_local.kind,
        SurfaceErrorKind::Body(BodyError::TypeMismatch)
    );
    let wrong_integer_local = compile_evaluator(
        40_007,
        "wrong_integer_local",
        "field u32\nlocal total\nlet one = fconst 1.0\nset total one\nlet result = get total\nstore 0 result\n",
    )
    .unwrap_err();
    assert_eq!(
        wrong_integer_local.kind,
        SurfaceErrorKind::Body(BodyError::TypeMismatch)
    );
}

#[cfg(feature = "native")]
#[test]
fn native_f32_is_byte_identical_to_reference() {
    use soma::executives::native::NativeCpuBackend;
    let program = program();
    let input = inputs();
    let expected = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    let actual = NativeCpuBackend::with(&[&program])
        .unwrap()
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    assert_eq!(actual, expected);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_f32_is_byte_identical_to_reference() {
    use soma::executives::metal::MetalBatchBackend;
    let program = program();
    let input = inputs();
    let expected = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    let actual = MetalBatchBackend::with(&[&program])
        .unwrap()
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    assert_eq!(actual, expected);
}

const COMPLETE_SOURCE: &str = r#"
field f32
field f32
field f32
field f32
field u32
field u32
field f32
let x = load 0
let y = load 1
let difference = fsub x y
let quotient = fdiv x y
let equal = feq x y
let less = flt x y
let chosen = fselect less x y
store 2 difference
store 3 quotient
store 4 equal
store 5 less
store 6 chosen
"#;

fn complete_program() -> soma::compiler::body::EvaluatorProgram {
    compile_evaluator(40_010, "complete_f32", COMPLETE_SOURCE).unwrap()
}

fn complete_inputs() -> Vec<u8> {
    [
        (1.0f32.to_bits(), 0.0f32.to_bits()),
        (0.0f32.to_bits(), 0.0f32.to_bits()),
        ((-1.0f32).to_bits(), 0.0f32.to_bits()),
        (f32::INFINITY.to_bits(), f32::INFINITY.to_bits()),
        (0x7fa1_2345, 2.0f32.to_bits()),
        (0x8000_0000, 0.0f32.to_bits()),
        (1.0f32.to_bits(), 2.0f32.to_bits()),
        (0x0000_0001, 1.0f32.to_bits()),
    ]
    .into_iter()
    .flat_map(|(x, y)| [x, y, 0, 0, 0, 0, 0].into_iter().flat_map(u32::to_le_bytes))
    .collect()
}

#[test]
fn reference_sub_div_comparison_and_float_select_have_exact_special_semantics() {
    let program = complete_program();
    let input = complete_inputs();
    let output = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    for (lane, bytes) in output.chunks_exact(28).enumerate() {
        let raw_x = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let raw_y = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let x = f32::from_bits(canonical(raw_x));
        let y = f32::from_bits(canonical(raw_y));
        let difference = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let quotient = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let equal = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let less = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let selected = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert_eq!(difference, canonical((x - y).to_bits()), "sub lane {lane}");
        assert_eq!(quotient, canonical((x / y).to_bits()), "div lane {lane}");
        assert_eq!(equal, u32::from(x == y), "eq lane {lane}");
        assert_eq!(less, u32::from(x < y), "lt lane {lane}");
        assert_eq!(
            selected,
            canonical(if x < y { raw_x } else { raw_y }),
            "select lane {lane}"
        );
    }
    assert_eq!(
        u32::from_le_bytes(output[12..16].try_into().unwrap()),
        f32::INFINITY.to_bits()
    );
    assert_eq!(
        u32::from_le_bytes(output[28 + 12..28 + 16].try_into().unwrap()),
        0x7fc0_0000
    );
    assert_eq!(
        u32::from_le_bytes(output[2 * 28 + 12..2 * 28 + 16].try_into().unwrap()),
        f32::NEG_INFINITY.to_bits()
    );
}

#[cfg(feature = "native")]
#[test]
fn complete_native_f32_surface_is_byte_identical() {
    use soma::executives::native::NativeCpuBackend;
    let program = complete_program();
    let input = complete_inputs();
    let expected = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    let actual = NativeCpuBackend::with(&[&program])
        .unwrap()
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    assert_eq!(actual, expected);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn complete_metal_f32_surface_is_byte_identical() {
    use soma::executives::metal::MetalBatchBackend;
    let program = complete_program();
    let input = complete_inputs();
    let expected = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    let actual = MetalBatchBackend::with(&[&program])
        .unwrap()
        .evaluate(program.id(), &input, 8, program.stride())
        .unwrap();
    assert_eq!(actual, expected);
}

const FLOAT_LOCAL_SOURCE: &str = r#"
field f32
local f32 sum
let sample = load 0
repeat 4
  let current = get sum
  let next = fadd current sample
  set sum next
end
let result = get sum
store 0 result
"#;

fn float_local_program() -> soma::compiler::body::EvaluatorProgram {
    compile_evaluator(40_020, "float_local", FLOAT_LOCAL_SOURCE).unwrap()
}

fn float_local_inputs() -> Vec<u8> {
    [
        1.5f32.to_bits(),
        (-2.25f32).to_bits(),
        0x8000_0000,
        0x7fa1_2345,
        0x0000_0001,
        f32::MAX.to_bits(),
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect()
}

#[test]
fn typed_float_local_starts_at_canonical_positive_zero_and_accumulates() {
    use soma::compiler::body::LocalKind;
    let program = float_local_program();
    assert_eq!(program.local_kinds(), &[LocalKind::F32]);
    let output = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &float_local_inputs(), 6, program.stride())
        .unwrap();
    for (input, actual) in float_local_inputs()
        .chunks_exact(4)
        .zip(output.chunks_exact(4))
    {
        let mut expected = 0.0f32;
        let sample = f32::from_bits(canonical(u32::from_le_bytes(input.try_into().unwrap())));
        for _ in 0..4 {
            expected = f32::from_bits(canonical((expected + sample).to_bits()));
        }
        assert_eq!(
            u32::from_le_bytes(actual.try_into().unwrap()),
            expected.to_bits()
        );
    }
}

#[cfg(feature = "native")]
#[test]
fn native_float_local_loop_is_byte_identical() {
    use soma::executives::native::NativeCpuBackend;
    let program = float_local_program();
    let input = float_local_inputs();
    let expected = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 6, program.stride())
        .unwrap();
    let actual = NativeCpuBackend::with(&[&program])
        .unwrap()
        .evaluate(program.id(), &input, 6, program.stride())
        .unwrap();
    assert_eq!(actual, expected);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_float_local_loop_is_byte_identical() {
    use soma::executives::metal::MetalBatchBackend;
    let program = float_local_program();
    let input = float_local_inputs();
    let expected = CpuReferenceBackend::with(&[&program])
        .evaluate(program.id(), &input, 6, program.stride())
        .unwrap();
    let actual = MetalBatchBackend::with(&[&program])
        .unwrap()
        .evaluate(program.id(), &input, 6, program.stride())
        .unwrap();
    assert_eq!(actual, expected);
}
