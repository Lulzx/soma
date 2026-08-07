# SOMA evaluator language

SOMA evaluator source is a small, general-purpose language for pure, total
array evaluators. One source program lowers to the validated evaluator IR and
can then run in the scalar reference interpreter, as native CPU machine code
through Cranelift, or as generated Metal Shading Language.

The language is deliberately bounded by the evaluator contract. A lane may
read its own packed input element, gather from either frozen input array, use
integer arithmetic and mutable loop locals, and write its own output element.
It cannot allocate, perform I/O, read another lane's output, or execute an
unbounded loop. Those restrictions preserve deterministic publication,
backend byte agreement, and a validation-time worst-case step bound.

## Example

```text
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
  let advanced = add old_cursor one
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
field u8|u16|u32|u64
aux u8|u16|u32|u64
local NAME

let NAME = load FIELD
let NAME = index
let NAME = gather INDEX_VALUE FIELD
let NAME = gather_aux INDEX_VALUE FIELD
let NAME = const INTEGER
let NAME = get LOCAL
let NAME = add|sub|mul|and|or|xor|shl|shr A B
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

`gather` reads the primary frozen input and `gather_aux` reads the separately
bound frozen auxiliary array. Dynamic indices clamp to the final element.
Arithmetic wraps at 64 bits, stores truncate to the destination field width,
and shifts mask their amount to six bits. These are language semantics shared
by every backend, not backend-specific conveniences.

## Compilation and validation

The surface compiler resolves names and emits the compact SSA/local body IR.
`EvaluatorProgram::bound` then performs the same centralized checks used for
programmatically constructed bodies: field bounds, operand dominance, loop
structure, scope escape, local bounds, required auxiliary input, and maximum
expanded step count.

Backends only accept validated `EvaluatorProgram` values. The native backend
JIT-compiles the complete IR, including gathers, nested loops, and divergent
early exits; it never falls back to interpretation. The Metal backend emits a
kernel from the same IR. Tests compare both results byte-for-byte with the
reference interpreter.
