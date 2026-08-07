use soma::compiler::examples;
use soma::compiler::surface::{compile_evaluator, SurfaceError, SurfaceErrorKind};
use soma::executives::batch::{BatchBackend, CpuReferenceBackend};

const WINDOW_SOURCE: &str = r#"
field u32
field u32
local sum
local cursor

let at = index
set cursor at
let one = const 1
let zero = const 0
repeat 8
  let position = get cursor
  let sample = gather position 0
  let current = get sum
  let next = add current sample
  set sum next
  let old_cursor = get cursor
  let advanced = add old_cursor one
  set cursor advanced
end
let result = get sum
store 1 result
"#;

const RUN_LENGTH_SOURCE: &str = r#"
field u32
field u32
local length
local cursor
let at = index
set cursor at
let one = const 1
let zero = const 0
repeat 8
  let position = get cursor
  let sample = gather position 0
  let stopped = eq sample zero
  break_if stopped
  let old_length = get length
  let longer = add old_length one
  set length longer
  let old_cursor = get cursor
  let advanced = add old_cursor one
  set cursor advanced
end
let result = get length
store 1 result
"#;

fn input() -> Vec<u8> {
    (0..257u32)
        .flat_map(|value| {
            let mut bytes = (value % 11).to_le_bytes().to_vec();
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes
        })
        .collect()
}

#[test]
fn named_source_lowers_to_the_existing_loop_bodies() {
    let module = examples::module();
    for (source, expected_id) in [
        (WINDOW_SOURCE, examples::WINDOW_SUM),
        (RUN_LENGTH_SOURCE, examples::RUN_LENGTH),
    ] {
        let compiled = compile_evaluator(expected_id, "named", source).unwrap();
        let expected = module.program(expected_id).unwrap();
        assert_eq!(compiled.layout(), expected.layout());
        assert_eq!(compiled.locals(), expected.locals());
        assert_eq!(compiled.ops(), expected.ops());
        assert_eq!(compiled.stores(), expected.stores());
    }
}

#[test]
fn named_source_runs_on_the_reference_backend() {
    let program = compile_evaluator(31_000, "window", WINDOW_SOURCE).unwrap();
    let mut backend = CpuReferenceBackend::with(&[&program]);
    let output = backend
        .evaluate(program.id(), &input(), 257, program.stride())
        .unwrap();
    assert_eq!(output.len(), input().len());
    assert_ne!(output, input());
}

#[cfg(feature = "native")]
#[test]
fn named_source_runs_as_native_machine_code() {
    use soma::executives::native::NativeCpuBackend;

    let program = compile_evaluator(31_001, "run_length", RUN_LENGTH_SOURCE).unwrap();
    let mut reference = CpuReferenceBackend::with(&[&program]);
    let mut native = NativeCpuBackend::with(&[&program]).unwrap().with_threads(4);
    assert_eq!(
        native
            .evaluate(program.id(), &input(), 257, program.stride())
            .unwrap(),
        reference
            .evaluate(program.id(), &input(), 257, program.stride())
            .unwrap()
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn named_source_runs_on_real_metal() {
    use soma::executives::metal::MetalBatchBackend;

    let program = compile_evaluator(31_002, "window", WINDOW_SOURCE).unwrap();
    let mut reference = CpuReferenceBackend::with(&[&program]);
    let mut metal = MetalBatchBackend::with(&[&program]).unwrap();
    assert_eq!(
        metal
            .evaluate(program.id(), &input(), 257, program.stride())
            .unwrap(),
        reference
            .evaluate(program.id(), &input(), 257, program.stride())
            .unwrap()
    );
}

#[test]
fn diagnostics_name_the_source_line_and_failure_kind() {
    let source = "field u32\nlet x = add missing missing\nstore 0 x\n";
    assert_eq!(
        compile_evaluator(31_003, "broken", source),
        Err(SurfaceError {
            line: 2,
            kind: SurfaceErrorKind::UnknownValue,
        })
    );

    let duplicate = "field u32\nlet x = const 0x10\nlocal x\nstore 0 x\n";
    assert_eq!(
        compile_evaluator(31_004, "duplicate", duplicate),
        Err(SurfaceError {
            line: 3,
            kind: SurfaceErrorKind::DuplicateName,
        })
    );
}
