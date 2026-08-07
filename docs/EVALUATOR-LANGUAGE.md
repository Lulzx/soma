# SOMA evaluator language

SOMA evaluator source is a small, general-purpose language for pure, total
array evaluators. One source program lowers to the validated evaluator IR and
can then run in the scalar reference interpreter, as native CPU machine code
through Cranelift, or as generated Metal Shading Language.

The language is deliberately bounded by the evaluator contract. A lane may
read its own packed input element, gather from either frozen input array, use
integer arithmetic, a deterministic binary32 arithmetic subset, mutable
integer loop locals, and write its own output element.
It cannot allocate, perform I/O, read another lane's output, or execute an
unbounded loop. Those restrictions preserve deterministic publication,
backend byte agreement, and a validation-time worst-case step bound.

## Example

```text
# Functions are expanded at compile time; no call exists in the body IR.
fn advance value delta
  let result = add value delta
  return result
end

# The output has the same packed layout as the primary input.
field u32
field u32

local sum
local cursor

let at = index
let one = const 1
set cursor at

repeat 8
  let position = get cursor
  let sample = gather position 0
  let current = get sum
  let next = add current sample
  set sum next
  let old_cursor = get cursor
  let advanced = call advance old_cursor one
  set cursor advanced
end

let result = get sum
store 1 result
```

Compile it with `compiler::surface::compile_evaluator`. The compiler reports
the source line and a stable error kind for syntax, name, integer, width, and
validated-body errors.

## Grammar

Declarations and statements are line-oriented. `#` begins a comment.

```text
fn NAME [PARAM ...]
  let NAME = EXPRESSION
  ...
  return VALUE
end

field u8|u16|u32|u64|f32
aux u8|u16|u32|u64|f32
local NAME
local f32 NAME

let NAME = call FUNCTION [ARGUMENT ...]
let NAME = load FIELD
let NAME = index
let NAME = gather INDEX_VALUE FIELD
let NAME = gather_aux INDEX_VALUE FIELD
let NAME = const INTEGER
let NAME = fconst FLOAT_OR_0xBITS
let NAME = get LOCAL
let NAME = add|sub|mul|and|or|xor|shl|shr A B
let NAME = fadd|fsub|fmul|fdiv A B
let NAME = feq|flt A B
let NAME = fselect INTEGER_CONDITION FLOAT_YES FLOAT_NO
let NAME = eq|lt A B
let NAME = select CONDITION YES NO

set LOCAL VALUE
repeat STATIC_TRIP_COUNT
break_if CONDITION
end
store FIELD VALUE
```

Integers may be decimal or prefixed with `0x`. Values are immutable and must
be defined before use. Locals start at zero and provide the explicit mutable
state carried through counted loops. A value created inside a loop cannot
escape that loop; copy it into a local when it must survive an iteration.

Functions are reusable, pure expression sequences. Parameters and function-local
`let` values are lexical names; a function has exactly one `return`, followed
by `end`. Calls may target a definition written later and functions may call
other functions. Each call is expanded into fresh SSA instructions at compile
time. Recursive call cycles, unknown functions, and arity mismatches are compile
errors. Function bodies have no locals, stores, or control-flow statements;
state and counted loops remain visible in the evaluator body.

`gather` reads the primary frozen input and `gather_aux` reads the separately
bound frozen auxiliary array. Dynamic indices clamp to the final element.
Integer arithmetic wraps at 64 bits, integer stores truncate to the destination
field width, and shifts mask their amount to six bits. Binary32 values are
stored as four little-endian IEEE-754 bytes. `fconst` accepts a decimal value
or an exact `0x` bit pattern; `fadd`, `fsub`, `fmul`, and `fdiv` round each operation to f32,
canonicalize every NaN to `0x7fc00000`, and canonicalize both signed zeros and
all subnormal inputs/results to `+0` (the explicit flush-to-zero boundary needed
for Apple GPU equivalence). Metal fast math is disabled, and the Cranelift lowering uses distinct strict
f32 instructions. Float-producing stores repeat the canonicalization,
so loaded non-canonical NaNs cannot leak through an output field. These are
language semantics shared by every backend, not backend-specific conveniences.

Validation tracks integer versus f32 values. Integer operations and control
conditions reject floats, float operations reject integers, gather indices are
integers, and a store must match the destination field kind. Existing untyped
locals remain integer locals. The bounded f32 surface is `fconst`, arithmetic,
ordered `feq`/`flt`,
`fselect`, loads/gathers, and stores. Division by zero follows IEEE binary32 and
is total: nonzero/zero produces signed infinity and zero/zero produces the
canonical NaN. Comparisons involving NaN are false; comparison results and
`fselect` conditions are ordinary integer booleans.

Locals are explicitly typed without changing existing source: `local NAME`
remains an integer local initialized to integer zero, while `local f32 NAME` is
a binary32 local initialized to canonical `+0`. `get` produces the declared
kind and `set` must consume it, including across loop iterations. Programmatic
IR construction uses `EvaluatorProgram::bound_typed` and `LocalKind`; the
existing count-based constructors continue to create integer locals.

## Compilation and validation

The surface compiler resolves names, expands every reachable function call,
and emits the compact SSA/local body IR. There is no runtime call instruction,
stack, indirect dispatch, or recursion. Expansion itself refuses to emit more
than `MAX_STEPS` instructions, and `EvaluatorProgram::bound` then performs the
same centralized checks used for programmatically constructed bodies: field
bounds, operand dominance, loop structure, scope escape, local bounds, required
auxiliary input, and maximum expanded step count (including static loop trip
counts). Functions therefore add reuse without changing totality or the bound
that every backend receives.

Backends only accept validated `EvaluatorProgram` values. The native backend
JIT-compiles the complete IR, including gathers, nested loops, and divergent
early exits; it never falls back to interpretation. The Metal backend emits a
kernel from the same IR. Tests compare both results byte-for-byte with the
reference interpreter.
