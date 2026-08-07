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
