//! Example evaluator modules, in the v0.3 body language.
//!
//! `state_machine_lowering` holds the hand-lowered continuation example; this
//! holds the batch-evaluator one. Keeping them in the library rather than in a
//! test fixture means the examples are compiled by the same parser the tests
//! exercise, so a body that stops parsing breaks the build rather than one
//! test file.
//!
//! `double_plus_one` is the function both backends used to hardcode. It is
//! here to prove the migration is real: the constant now exists in exactly one
//! place, as data, and both lowerings are generated from it.
//!
//! The other three exist because a single body cannot demonstrate a compiler.
//! `min_and_xor` in particular is not expressible as `a*x + b` — it branches on
//! a comparison and writes two fields from two inputs — so a backend cannot
//! pass I20 for it by coincidence.

use super::ir::{IrError, Module};

/// Evaluator ids used by the example module.
pub const DOUBLE_PLUS_ONE: u32 = 7;
pub const DOUBLE_PLUS_ONE_TAGGED: u32 = 8;
pub const MIN_AND_XOR: u32 = 9;
pub const BITMIX: u32 = 10;
/// `ant_scoring` occupies 11; the gather examples continue from 12.
pub const NEIGHBOUR_MAX: u32 = 12;
pub const PERMUTE: u32 = 13;

pub const SOURCE: &str = r#"
module soma.examples

# The function both backends hardcoded before v0.3, over a bare u32 element.
evaluator 7 double_plus_one 4 70 71 ro 72 73 ro
  field u32
  op 0 load 0
  op 1 const 2
  op 2 mul 0 1
  op 3 const 1
  op 4 add 2 3
  store 0 4

# The same function over a two-field element. The second field is never
# stored, which is how "fields a body does not write keep their bytes" gets
# tested rather than asserted.
evaluator 8 double_plus_one_tagged 8 80 81 ro 82 83 ro
  field u32
  field u32
  op 0 load 0
  op 1 const 2
  op 2 mul 0 1
  op 3 const 1
  op 4 add 2 3
  store 0 4

# Branching and multi-field: field 0 becomes min(x, y), field 1 becomes x^y.
# Not expressible as a*x + b in either field.
evaluator 9 min_and_xor 8 90 91 ro 92 93 ro
  field u32
  field u32
  op 0 load 0
  op 1 load 1
  op 2 cmplt 0 1
  op 3 select 2 0 1
  op 4 xor 0 1
  store 0 3
  store 1 4

# Shifts, to exercise the 6-bit shift-amount masking rule on both sides.
evaluator 10 bitmix 8 100 101 ro 102 103 ro
  field u32
  field u32
  op 0 load 0
  op 1 const 13
  op 2 shl 0 1
  op 3 const 7
  op 4 shr 0 3
  op 5 xor 2 4
  op 6 load 1
  op 7 add 5 6
  store 0 7

# A three-point stencil: field 1 becomes the largest of field 0 across this
# element and its two neighbours. The first body that reads an element other
# than its own.
#
# Edge handling is the body's job and is done without branching. The right
# edge needs nothing — index `count` clamps back to the last element, which is
# this one. The left edge does: `0 - 1` wraps to a huge value and would clamp
# to the *far* end of the array, so op 5 substitutes the element's own index
# when it is at position zero.
evaluator 12 neighbour_max 8 120 121 ro 122 123 ro
  field u32
  field u32
  op 0 index
  op 1 const 0
  op 2 cmpeq 0 1
  op 3 const 1
  op 4 sub 0 3
  op 5 select 2 0 4
  op 6 add 0 3
  op 7 gather 5 0
  op 8 gather 6 0
  op 9 load 0
  op 10 cmplt 9 7
  op 11 select 10 7 9
  op 12 cmplt 11 8
  op 13 select 12 8 11
  store 1 13

# A permutation gather: field 0 names an element, and field 1 becomes that
# element's field 1. Every element both reads a payload and has its own
# overwritten, so this only produces the permutation if a gather reads the
# frozen *input*. A backend that gathered from the output it was writing would
# return an order-dependent answer, which is the failure I19 cares about.
evaluator 13 permute 8 130 131 ro 132 133 ro
  field u32
  field u32
  op 0 load 0
  op 1 gather 0 1
  store 1 1
"#;

/// Parse the example module. Panics only if the source above is malformed,
/// which is a build-time defect rather than a runtime condition.
pub fn module() -> Module {
    parse().expect("the built-in example module must parse")
}

pub fn parse() -> Result<Module, IrError> {
    Module::parse(SOURCE)
}
