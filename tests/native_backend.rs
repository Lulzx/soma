#![cfg(feature = "native")]

use soma::compiler::body::{ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};
use soma::compiler::examples;
use soma::executives::batch::{BackendError, BatchBackend, CpuReferenceBackend};
use soma::executives::native::NativeCpuBackend;

fn inputs() -> Vec<u8> {
    (0..257u32)
        .flat_map(|value| {
            let mut bytes = value.to_le_bytes().to_vec();
            bytes.extend_from_slice(&value.rotate_left(13).to_le_bytes());
            bytes
        })
        .collect()
}

#[test]
fn native_code_agrees_with_the_reference_on_every_supported_example() {
    let module = examples::module();
    let programs: Vec<_> = [
        examples::DOUBLE_PLUS_ONE_TAGGED,
        examples::MIN_AND_XOR,
        examples::BITMIX,
    ]
    .iter()
    .map(|id| module.program(*id).unwrap())
    .collect();
    let refs: Vec<_> = programs.to_vec();
    let mut reference = CpuReferenceBackend::with(&refs);

    for threads in [1, 4] {
        let mut native = NativeCpuBackend::with(&refs).unwrap().with_threads(threads);
        assert_eq!(native.threads(), threads);
        for program in &programs {
            let expected = reference.evaluate(program.id(), &inputs(), 257, 8).unwrap();
            let actual = native.evaluate(program.id(), &inputs(), 257, 8).unwrap();
            assert_eq!(actual, expected, "native mismatch for {}", program.name());
        }
    }
}

#[test]
fn native_lowering_covers_the_straight_line_integer_language() {
    let program = EvaluatorProgram::with_locals(
        30_000,
        "native_integer_surface",
        ElementLayout::new(vec![
            FieldWidth::U8,
            FieldWidth::U16,
            FieldWidth::U32,
            FieldWidth::U64,
        ]),
        1,
        vec![
            Op::Load(0),
            Op::Load(1),
            Op::Load(2),
            Op::Load(3),
            Op::Index,
            Op::Const(u64::MAX),
            Op::Add(0, 1),
            Op::Sub(3, 2),
            Op::Mul(6, 4),
            Op::And(7, 5),
            Op::Or(8, 0),
            Op::Xor(9, 1),
            Op::Shl(10, 0),
            Op::Shr(11, 1),
            Op::CmpEq(12, 13),
            Op::CmpLt(12, 13),
            Op::Select(15, 12, 13),
            Op::Set(0, 16),
            Op::Get(0),
        ],
        vec![
            Store {
                field: 0,
                value: 14,
            },
            Store {
                field: 1,
                value: 15,
            },
            Store {
                field: 2,
                value: 17,
            },
            Store {
                field: 3,
                value: 18,
            },
        ],
    )
    .unwrap();
    let stride = program.stride();
    let bytes = soma::experiments::backend_bench::synthetic_inputs(513, stride);
    let mut reference = CpuReferenceBackend::with(&[&program]);
    let mut native = NativeCpuBackend::with(&[&program]).unwrap().with_threads(6);
    assert_eq!(
        native.evaluate(program.id(), &bytes, 513, stride).unwrap(),
        reference
            .evaluate(program.id(), &bytes, 513, stride)
            .unwrap()
    );
}

#[test]
fn unsupported_control_flow_and_gathers_are_declined() {
    let module = examples::module();
    for id in [
        examples::NEIGHBOUR_MAX,
        examples::PERMUTE,
        examples::WINDOW_SUM,
        examples::RUN_LENGTH,
    ] {
        let program = module.program(id).unwrap();
        let mut native = NativeCpuBackend::new().unwrap();
        assert_eq!(
            native.install(program),
            Err(BackendError::UnsupportedEvaluator),
            "unsupported body {} was partially compiled",
            program.name()
        );
    }
}

#[test]
fn native_backend_rejects_shape_and_installation_mismatches() {
    let module = examples::module();
    let program = module.program(examples::BITMIX).unwrap();
    let mut native = NativeCpuBackend::with(&[program]).unwrap();
    assert_eq!(
        native.evaluate(program.id(), &inputs(), 257, 4),
        Err(BackendError::InvalidInput)
    );
    assert_eq!(
        native.evaluate(999_999, &inputs(), 257, 8),
        Err(BackendError::UnsupportedEvaluator)
    );
}
