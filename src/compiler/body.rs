//! Evaluator bodies: what a `BatchEvaluate` collective actually computes.
//!
//! Before this existed, `compiler::ir` named evaluators and no more. Both
//! backends took an `evaluator_id`, ignored it, and hardcoded `2*x + 1`. I17
//! checked that a collective's evaluator and stride matched its module's
//! manifest, so nothing anywhere checked that a backend applied the function
//! its module named — a backend could return arbitrary bytes and every
//! invariant still held. The CPU/Metal comparison test passed because two
//! hardcodings of one constant agree, which is not evidence about a compiler.
//!
//! A body here is deliberately small, because the point being tested is
//! placement-independent publication, not language design:
//!
//! - **Pure and total.** No allocation and no division (which would be
//!   partial). Every program terminates in a number of steps decided at
//!   validation time — `step_bound` multiplies out the loop nesting and
//!   `MAX_STEPS` is the ceiling. Two rules keep that true in the presence of
//!   things that usually break it: an out-of-range `Gather` clamps rather than
//!   faulting, and a loop's trip count is a constant rather than an
//!   expression. `BreakIf` gives back the useful half of a data-dependent
//!   loop, since leaving early can only lower the count.
//!
//!   Totality is not a preference. `kernel/epochs.rs` checks a continuation's
//!   step budget before dispatch and a collective evaluation is one step of one
//!   continuation, so a body that might not terminate makes that check
//!   meaningless.
//! - **Loops carry state in locals, not in values.** SSA and back edges need
//!   phi nodes, which is a large amount of machinery for a language this size,
//!   and "an instruction names an earlier instruction" stops meaning anything
//!   once an instruction can run twice. Instead, a value computed inside a loop
//!   is visible only within the iteration that computed it — validation rejects
//!   a reference that escapes one — and `Get`/`Set` over declared locals are
//!   how anything outlives an iteration.
//! - **Branch-free is a property of a body, not of the language.** It used to
//!   be both, because `Select` was the only conditional. `BreakIf` can put two
//!   lanes of a cohort on different iterations, so `is_uniform` is the question
//!   a scheduler asks about a body rather than an answer the language
//!   guarantees. Divergence costs occupancy and not correctness: both lowerings
//!   still agree, which is what I20 checks.
//! - **One output element per input element.** A reduction is a different
//!   collective, not a different body. `Gather` widens what a body may *read*
//!   to any element of the frozen input array, but not what it may write: a
//!   body still writes only its own element's fields.
//! - **Reads the input, never the output.** This is what keeps a gather safe
//!   to run in any order. The input array is frozen and single-assignment, so
//!   every lane sees the same bytes no matter when it runs, and no lane can
//!   observe another lane's store. A body that could read the output array
//!   would make the result depend on the schedule and take I19
//!   (placement-neutrality) with it.
//! - **Bounded deterministic f32.** Binary32 add/multiply are admitted with
//!   fast math disabled and canonical NaN/zero results, so I20 remains a byte
//!   invariant rather than weakening to a tolerance. Integer and float values
//!   are statically separated. Other float operations remain unimplemented
//!   until their exact cross-backend semantics are specified.
//! - **Typed against a declared layout.** Reading or writing outside the
//!   declared element is a validation error, not a runtime fault, so an
//!   invalid body cannot reach a backend at all.
//!
//! Arithmetic is wrapping on `u64` and truncated to the field width on store.
//! Shifts mask their amount to 6 bits. Both rules exist so that the CPU
//! interpreter and generated Metal Shading Language agree by construction
//! rather than by luck; I20 checks that they do.

use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Width of one field in a frozen-array element. Little-endian, packed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldWidth {
    U8,
    U16,
    U32,
    U64,
    /// IEEE-754 binary32 stored as four little-endian bytes.
    F32,
}

impl FieldWidth {
    pub fn bytes(self) -> u32 {
        match self {
            FieldWidth::U8 => 1,
            FieldWidth::U16 => 2,
            FieldWidth::U32 | FieldWidth::F32 => 4,
            FieldWidth::U64 => 8,
        }
    }

    pub fn mask(self) -> u64 {
        match self {
            FieldWidth::U8 => 0xFF,
            FieldWidth::U16 => 0xFFFF,
            FieldWidth::U32 | FieldWidth::F32 => 0xFFFF_FFFF,
            FieldWidth::U64 => u64::MAX,
        }
    }

    fn parse(text: &str) -> Option<FieldWidth> {
        match text {
            "u8" => Some(FieldWidth::U8),
            "u16" => Some(FieldWidth::U16),
            "u32" => Some(FieldWidth::U32),
            "u64" => Some(FieldWidth::U64),
            "f32" => Some(FieldWidth::F32),
            _ => None,
        }
    }
}

/// The element layout an evaluator reads and writes. Fields are packed in
/// declaration order with no padding, so the stride is derived rather than
/// asserted separately.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ElementLayout {
    fields: Vec<FieldWidth>,
}

impl ElementLayout {
    pub fn new(fields: Vec<FieldWidth>) -> Self {
        Self { fields }
    }

    pub fn fields(&self) -> &[FieldWidth] {
        &self.fields
    }

    pub fn stride(&self) -> u32 {
        self.fields.iter().map(|field| field.bytes()).sum()
    }

    pub fn offset(&self, index: u32) -> Option<u32> {
        let index = index as usize;
        if index >= self.fields.len() {
            return None;
        }
        Some(self.fields[..index].iter().map(|f| f.bytes()).sum())
    }

    pub fn width(&self, index: u32) -> Option<FieldWidth> {
        self.fields.get(index as usize).copied()
    }
}

/// One instruction. Operands name earlier instructions by index, so a body is
/// already in SSA form and needs no register allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Read a field of this lane's own input element.
    Load(u32),
    /// This element's own position in the array.
    ///
    /// Without it a gather index could only come from field data, so a body
    /// could chase indices it was handed but could not express "the element
    /// before me" — which is most of what a gather is for. It is also the only
    /// op whose value differs across the lanes of a cohort, and that is exactly
    /// what makes those lanes read different elements while running identical
    /// code.
    Index,
    /// Read a field of *another* input element: `Gather(index, field)` reads
    /// `field` of the element at the value of instruction `index`.
    ///
    /// The index is computed, so it cannot be range-checked at validation time
    /// the way a field can. Rather than admit a runtime fault into a language
    /// whose totality the step budget depends on, an index at or past the end
    /// of the array clamps to the last element — the clamp-to-edge rule a
    /// stencil wants anyway. Both lowerings clamp identically, which is what
    /// I20 checks.
    Gather(u32, u32),
    Const(u64),
    /// An exact binary32 bit pattern. Float values remain encoded in the low
    /// 32 bits of value slots at backend boundaries.
    FConst(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    /// Deterministic binary32 arithmetic. Each result is rounded to f32,
    /// canonicalizes every NaN to 0x7fc00000, and canonicalizes both zeros and
    /// all subnormals to +0. These rules are part of I20, not an optimization choice.
    FAdd(u32, u32),
    FMul(u32, u32),
    And(u32, u32),
    Or(u32, u32),
    Xor(u32, u32),
    /// Left shift; the amount is masked to 6 bits.
    Shl(u32, u32),
    /// Logical right shift; the amount is masked to 6 bits.
    Shr(u32, u32),
    /// 1 when equal, 0 otherwise.
    CmpEq(u32, u32),
    /// 1 when the first operand is strictly less, 0 otherwise.
    CmpLt(u32, u32),
    /// `cond != 0 ? a : b`. Branch-free, so a cohort of lanes evaluating one
    /// of these never diverges. It is no longer the only control flow the
    /// language has, but it is still the only one that costs nothing.
    Select(u32, u32, u32),

    /// Read a local. Locals start at zero and are the only values that survive
    /// an iteration of a loop.
    ///
    /// SSA and loops do not mix without phi nodes, and phi nodes are a large
    /// amount of machinery for a language this size — the operand model here
    /// is "an instruction names an earlier instruction", and a back edge makes
    /// "earlier" stop meaning what it says. Mutable locals are the other way
    /// out: values computed inside a loop body are visible only inside the
    /// iteration that computed them, and anything that has to outlive an
    /// iteration is written to a local. That keeps every operand reference
    /// meaning exactly what it meant before loops existed.
    Get(u32),
    /// Write a local, and evaluate to the value written, so a `set` can be
    /// used where a value is expected rather than needing a separate statement
    /// category.
    Set(u32, u32),
    /// Begin a loop with a trip count fixed at validation time. Matched by
    /// [`Op::EndRepeat`].
    ///
    /// The count is static and that is the whole of why totality survives:
    /// `EvaluatorProgram::step_bound` multiplies out the nesting and refuses a
    /// body whose worst case exceeds [`MAX_STEPS`], so every body still
    /// terminates in a number of steps decided before it runs. A
    /// data-dependent trip count would move that decision to runtime and take
    /// the step budget with it; `BreakIf` is what gives back the useful half
    /// of one, since exiting early can only lower the count.
    Repeat(u32),
    /// Close the innermost open [`Op::Repeat`].
    EndRepeat,
    /// Read a field of an element of the *second* bound array:
    /// `GatherAux(index, field)` reads `field` of the aux element at the value
    /// of instruction `index`.
    ///
    /// `Gather` widened what a body may read from its own element to any
    /// element of its own array. This widens it to an array that is not the
    /// one it is iterating. That is the difference between a stencil and a
    /// lookup, and it is the whole reason ant sensing could not be expressed:
    /// a colony's elements are ants and the thing being sensed is the trail
    /// grid, which is a different array with a different layout and a
    /// different length.
    ///
    /// It obeys every rule `Gather` obeys and for the same reasons. The aux
    /// array is frozen and read-only, so no lane can observe another lane's
    /// store and I19 survives. An out-of-range index clamps to the last aux
    /// element, so totality survives a computed index. The field is static and
    /// checked at validation against the *aux* layout, which is declared
    /// separately — an element of the input array and an element of the aux
    /// array have no reason to have the same shape.
    GatherAux(u32, u32),
    /// Leave the innermost enclosing loop when the operand is non-zero.
    ///
    /// This is real divergence: two lanes of a cohort can leave on different
    /// iterations. It costs occupancy on a GPU and costs nothing in
    /// correctness — the lanes still execute the same instructions in the same
    /// order, and a lane that has left simply stops contributing. I20 checks
    /// that both lowerings agree about which iteration each lane left on.
    BreakIf(u32),
}

/// The worst-case instruction count a body is allowed to have once its loops
/// are multiplied out.
///
/// Totality is not a style preference here: `kernel/epochs.rs` checks a
/// continuation's step budget before dispatch, and a collective evaluation is
/// one step of one continuation. A body that might not terminate makes that
/// check meaningless. Bounding the unrolled length at validation time keeps the
/// old guarantee — every program terminates in a fixed number of steps decided
/// before it runs — while admitting loops, which is the whole trade.
pub const MAX_STEPS: u64 = 1 << 20;

impl Op {
    fn operands(self) -> Vec<u32> {
        match self {
            Op::Load(_) | Op::Const(_) | Op::FConst(_) | Op::Index => Vec::new(),
            Op::Get(_) | Op::Repeat(_) | Op::EndRepeat => Vec::new(),
            Op::Set(_, value) => vec![value],
            Op::BreakIf(cond) => vec![cond],
            Op::Gather(index, _) | Op::GatherAux(index, _) => vec![index],
            Op::Add(a, b)
            | Op::Sub(a, b)
            | Op::Mul(a, b)
            | Op::FAdd(a, b)
            | Op::FMul(a, b)
            | Op::And(a, b)
            | Op::Or(a, b)
            | Op::Xor(a, b)
            | Op::Shl(a, b)
            | Op::Shr(a, b)
            | Op::CmpEq(a, b)
            | Op::CmpLt(a, b) => vec![a, b],
            Op::Select(c, a, b) => vec![c, a, b],
        }
    }
}

/// Write instruction `value` into output field `field`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Store {
    pub field: u32,
    pub value: u32,
}

/// A validated, executable evaluator body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluatorProgram {
    id: u32,
    name: String,
    layout: ElementLayout,
    /// The element layout of the second bound array, when the body names one.
    ///
    /// `None` is a body that binds one array, which is every body written
    /// before `GatherAux` existed. It is deliberately a separate layout rather
    /// than a reuse of `layout`: the aux array is a different array, and
    /// forcing it to share an element shape would have made the one workload
    /// this exists for -- ants indexing a trail grid -- inexpressible again.
    aux: Option<ElementLayout>,
    locals: u32,
    ops: Vec<Op>,
    stores: Vec<Store>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyError {
    EmptyLayout,
    EmptyBody,
    NoStore,
    /// An operand names an instruction that does not exist yet. Bodies are
    /// SSA and acyclic by construction, so a forward reference is a defect.
    ForwardReference,
    /// A load or store names a field outside the declared element. This is the
    /// clause that makes "cannot read outside its element" a validation error
    /// rather than a runtime fault.
    FieldOutOfRange,
    /// The layout's derived stride disagrees with the stride the module
    /// declares for this evaluator.
    StrideMismatch,
    /// A `get` or `set` names a local outside the declared set. Locals are
    /// declared like fields are, and for the same reason: an out-of-range name
    /// is a validation error rather than a runtime fault.
    LocalOutOfRange,
    /// A `repeat` was never closed, an `endrepeat` closed nothing, or a
    /// `breakif` sat outside every loop.
    UnbalancedLoop,
    /// A `gatheraux` appeared in a body that declares no aux layout, or an
    /// `aux` layout was declared and never read. Both directions are errors:
    /// the first has no array to read and the second binds an array the body
    /// cannot name, which a caller would have had to freeze for nothing.
    AuxMismatch,
    /// A `repeat` declared no iterations. Zero is not obviously wrong, but it
    /// makes the ops it encloses dead, and a body containing instructions that
    /// provably never run is more likely a mistake than an intention.
    EmptyLoop,
    /// An operand names an instruction inside a loop from outside it. Values do
    /// not escape an iteration; locals are how a loop communicates with what
    /// follows it.
    EscapingValue,
    /// The body's unrolled length exceeds [`MAX_STEPS`], so it is not
    /// obviously total and the step budget could not bound it.
    Unbounded,
    /// An integer operation consumed a float, a float operation consumed an
    /// integer, or a store's value disagreed with its field kind.
    TypeMismatch,
    Syntax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Integer,
    Float,
}

fn field_kind(width: FieldWidth) -> ValueKind {
    if width == FieldWidth::F32 {
        ValueKind::Float
    } else {
        ValueKind::Integer
    }
}

fn require_kind(kinds: &[ValueKind], index: u32, expected: ValueKind) -> Result<(), BodyError> {
    if kinds.get(index as usize).copied() == Some(expected) {
        Ok(())
    } else {
        Err(BodyError::TypeMismatch)
    }
}

impl EvaluatorProgram {
    pub fn new(
        id: u32,
        name: impl Into<String>,
        layout: ElementLayout,
        ops: Vec<Op>,
        stores: Vec<Store>,
    ) -> Result<Self, BodyError> {
        Self::with_locals(id, name, layout, 0, ops, stores)
    }

    pub fn with_locals(
        id: u32,
        name: impl Into<String>,
        layout: ElementLayout,
        locals: u32,
        ops: Vec<Op>,
        stores: Vec<Store>,
    ) -> Result<Self, BodyError> {
        Self::bound(id, name, layout, None, locals, ops, stores)
    }

    /// The full form: an element layout, an optional second array's layout,
    /// locals, instructions, and stores.
    pub fn bound(
        id: u32,
        name: impl Into<String>,
        layout: ElementLayout,
        aux: Option<ElementLayout>,
        locals: u32,
        ops: Vec<Op>,
        stores: Vec<Store>,
    ) -> Result<Self, BodyError> {
        let program = Self {
            id,
            name: name.into(),
            layout,
            aux,
            locals,
            ops,
            stores,
        };
        program.validate()?;
        Ok(program)
    }

    fn validate(&self) -> Result<(), BodyError> {
        if self.layout.fields().is_empty() {
            return Err(BodyError::EmptyLayout);
        }
        if self.ops.is_empty() {
            return Err(BodyError::EmptyBody);
        }
        if self.stores.is_empty() {
            return Err(BodyError::NoStore);
        }

        // The loop nest each instruction sits in, as the stack of enclosing
        // `Repeat` indices. Computing it once gives both remaining checks: the
        // structure is balanced exactly when this can be built, and a value
        // escapes exactly when an operand's stack is not a prefix of the
        // consumer's.
        let regions = self.regions()?;

        let field_count = self.layout.fields().len() as u32;
        for (index, op) in self.ops.iter().enumerate() {
            // The gathered *field* is static and checked here exactly as a
            // load's is. Only the element index is dynamic, and that clamps.
            if let Op::Load(field) | Op::Gather(_, field) = op {
                if *field >= field_count {
                    return Err(BodyError::FieldOutOfRange);
                }
            }
            // The aux field is static and checked against the *aux* layout,
            // exactly as a load's is checked against the element layout. Only
            // the aux element index is dynamic, and that clamps.
            if let Op::GatherAux(_, field) = op {
                let Some(aux) = &self.aux else {
                    return Err(BodyError::AuxMismatch);
                };
                if *field >= aux.fields().len() as u32 {
                    return Err(BodyError::FieldOutOfRange);
                }
            }
            if let Op::Get(local) | Op::Set(local, _) = op {
                if *local >= self.locals {
                    return Err(BodyError::LocalOutOfRange);
                }
            }
            if let Op::Repeat(trips) = op {
                if *trips == 0 {
                    return Err(BodyError::EmptyLoop);
                }
            }
            for operand in op.operands() {
                if operand as usize >= index {
                    return Err(BodyError::ForwardReference);
                }
                if !is_prefix(&regions[operand as usize], &regions[index]) {
                    return Err(BodyError::EscapingValue);
                }
            }
        }

        // Type validation is independent of storage width: all u* fields
        // produce integer values, while f32 fields and float ops produce
        // binary32 values. Integer behavior remains the original u64 algebra.
        let mut kinds = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let kind = match *op {
                Op::Load(field) => {
                    field_kind(self.layout.width(field).ok_or(BodyError::FieldOutOfRange)?)
                }
                Op::Gather(at, field) => {
                    require_kind(&kinds, at, ValueKind::Integer)?;
                    field_kind(self.layout.width(field).ok_or(BodyError::FieldOutOfRange)?)
                }
                Op::GatherAux(at, field) => {
                    require_kind(&kinds, at, ValueKind::Integer)?;
                    let layout = self.aux.as_ref().ok_or(BodyError::AuxMismatch)?;
                    field_kind(layout.width(field).ok_or(BodyError::FieldOutOfRange)?)
                }
                Op::FConst(_) => ValueKind::Float,
                Op::FAdd(a, b) | Op::FMul(a, b) => {
                    require_kind(&kinds, a, ValueKind::Float)?;
                    require_kind(&kinds, b, ValueKind::Float)?;
                    ValueKind::Float
                }
                Op::Set(_, value) => {
                    // Locals retain their historical integer-zero type. A
                    // future float-local declaration can extend this without
                    // making today's untyped `local` source ambiguous.
                    require_kind(&kinds, value, ValueKind::Integer)?;
                    ValueKind::Integer
                }
                Op::Add(a, b)
                | Op::Sub(a, b)
                | Op::Mul(a, b)
                | Op::And(a, b)
                | Op::Or(a, b)
                | Op::Xor(a, b)
                | Op::Shl(a, b)
                | Op::Shr(a, b)
                | Op::CmpEq(a, b)
                | Op::CmpLt(a, b) => {
                    require_kind(&kinds, a, ValueKind::Integer)?;
                    require_kind(&kinds, b, ValueKind::Integer)?;
                    ValueKind::Integer
                }
                Op::Select(c, a, b) => {
                    require_kind(&kinds, c, ValueKind::Integer)?;
                    require_kind(&kinds, a, ValueKind::Integer)?;
                    require_kind(&kinds, b, ValueKind::Integer)?;
                    ValueKind::Integer
                }
                Op::BreakIf(c) => {
                    require_kind(&kinds, c, ValueKind::Integer)?;
                    ValueKind::Integer
                }
                Op::Index | Op::Const(_) | Op::Get(_) | Op::Repeat(_) | Op::EndRepeat => {
                    ValueKind::Integer
                }
            };
            kinds.push(kind);
        }

        for store in &self.stores {
            if store.field >= field_count {
                return Err(BodyError::FieldOutOfRange);
            }
            if store.value as usize >= self.ops.len() {
                return Err(BodyError::ForwardReference);
            }
            // A store happens after the body, so it is outside every loop. It
            // may therefore only name a value computed outside every loop —
            // the same escape rule the operands above obey, at the one place
            // where the consumer is not an instruction.
            if !regions[store.value as usize].is_empty() {
                return Err(BodyError::EscapingValue);
            }
            let field = self
                .layout
                .width(store.field)
                .ok_or(BodyError::FieldOutOfRange)?;
            if kinds[store.value as usize] != field_kind(field) {
                return Err(BodyError::TypeMismatch);
            }
        }

        // An aux layout nobody reads means a caller has to freeze and bind an
        // array the body cannot name. The collective would be well-formed and
        // the binding would be dead, which is the kind of thing that survives
        // for a year because nothing is wrong with it.
        if self.aux.is_some() && !self.ops.iter().any(|op| matches!(op, Op::GatherAux(_, _))) {
            return Err(BodyError::AuxMismatch);
        }
        if let Some(aux) = &self.aux {
            if aux.fields().is_empty() {
                return Err(BodyError::EmptyLayout);
            }
        }

        if self.step_bound() > MAX_STEPS {
            return Err(BodyError::Unbounded);
        }
        Ok(())
    }

    /// The stack of enclosing `Repeat` indices for each instruction, and the
    /// balance check that produces it.
    ///
    /// A `BreakIf` outside every loop is rejected here rather than at runtime:
    /// there is nothing for it to leave, so it is either a typo or a `return`,
    /// and the language does not have a `return`.
    fn regions(&self) -> Result<Vec<Vec<usize>>, BodyError> {
        let mut open: Vec<usize> = Vec::new();
        let mut regions = Vec::with_capacity(self.ops.len());
        for (index, op) in self.ops.iter().enumerate() {
            match op {
                Op::Repeat(_) => {
                    // The `repeat` itself belongs to the region outside the
                    // loop it opens; its body does not.
                    regions.push(open.clone());
                    open.push(index);
                }
                Op::EndRepeat => {
                    if open.pop().is_none() {
                        return Err(BodyError::UnbalancedLoop);
                    }
                    regions.push(open.clone());
                }
                Op::BreakIf(_) => {
                    if open.is_empty() {
                        return Err(BodyError::UnbalancedLoop);
                    }
                    regions.push(open.clone());
                }
                _ => regions.push(open.clone()),
            }
        }
        if !open.is_empty() {
            return Err(BodyError::UnbalancedLoop);
        }
        Ok(regions)
    }

    /// The worst-case number of instructions this body executes, with every
    /// loop taken the full number of times.
    ///
    /// `BreakIf` is ignored, which is the conservative direction: leaving a
    /// loop early can only lower the count, so a body inside the bound stays
    /// inside it. Saturating arithmetic keeps a deeply nested body from
    /// wrapping around into a small number and passing.
    pub fn step_bound(&self) -> u64 {
        let mut total: u64 = 0;
        let mut multiplier: u64 = 1;
        let mut stack: Vec<u64> = Vec::new();
        for op in &self.ops {
            total = total.saturating_add(multiplier);
            match op {
                Op::Repeat(trips) => {
                    stack.push(multiplier);
                    multiplier = multiplier.saturating_mul(u64::from(*trips));
                }
                Op::EndRepeat => {
                    multiplier = stack.pop().unwrap_or(1);
                }
                _ => {}
            }
        }
        total
    }

    /// Whether this body is branch-free, and so evaluates in lockstep across
    /// the lanes of a cohort.
    ///
    /// Every body was in this set before loops existed, which is why nothing
    /// asked. It is a placement question rather than a correctness one — a
    /// diverging body computes the same answer on both backends, and I20
    /// checks that it does — so nothing in the machine refuses a body for
    /// failing it. It is what a scheduler would consult to decide whether a
    /// cohort's lanes are worth grouping.
    ///
    /// A `repeat` alone does not break uniformity: its trip count is static, so
    /// every lane runs the same iterations. Only `breakif` can put two lanes on
    /// different iterations, so only `breakif` is asked about.
    pub fn is_uniform(&self) -> bool {
        !self.ops.iter().any(|op| matches!(op, Op::BreakIf(_)))
    }

    pub fn locals(&self) -> u32 {
        self.locals
    }

    /// The second array's element layout, when this body names one.
    pub fn aux_layout(&self) -> Option<&ElementLayout> {
        self.aux.as_ref()
    }

    /// The second array's element stride, or zero when no second array is
    /// bound. Zero is the "not bound" signal everywhere this travels, which is
    /// why it is a stride rather than an `Option` at the backend boundary: a
    /// zero-stride array has no elements to read by construction.
    pub fn aux_stride(&self) -> u32 {
        self.aux.as_ref().map(|aux| aux.stride()).unwrap_or(0)
    }

    /// Whether this body reads a second array, and so whether a collective
    /// using it must bind one.
    pub fn binds_aux(&self) -> bool {
        self.aux.is_some()
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn layout(&self) -> &ElementLayout {
        &self.layout
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn stores(&self) -> &[Store] {
        &self.stores
    }

    pub fn stride(&self) -> u32 {
        self.layout.stride()
    }

    /// Fields this body can observe. Used by tests to show that a body really
    /// depends on more than one input, so a "distinct" body is distinct.
    ///
    /// A gathered field is observed too — from a different element, but the
    /// same field of the same layout — so it belongs here.
    pub fn loaded_fields(&self) -> BTreeSet<u32> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                Op::Load(field) | Op::Gather(_, field) => Some(*field),
                _ => None,
            })
            .collect()
    }

    /// Whether this body reads any element other than its own.
    ///
    /// Introspection, like `loaded_fields`, and used by tests for the same
    /// reason: to show that a body said to gather actually does. Nothing in the
    /// machine consults it today, because nothing splits an array — a
    /// collective is placed whole. It is the question a future chunked backend
    /// would have to ask before splitting one, since a chunk boundary changes
    /// what an index and the clamp resolve to.
    pub fn gathers(&self) -> bool {
        self.ops.iter().any(|op| matches!(op, Op::Gather(_, _)))
    }

    /// Evaluate the element at `index` of the frozen input array `inputs`.
    ///
    /// `inputs` is the *whole* array rather than one element, because `Gather`
    /// reads across it. `output` is just this element's bytes, and starts as a
    /// copy of its input, so fields the body does not store keep their
    /// incoming bytes. Nothing here can write another element, and nothing can
    /// read the output array, so calling this for the elements of one array in
    /// any order — or concurrently — produces the same bytes.
    pub fn evaluate_at(&self, inputs: &[u8], count: u32, index: u32, output: &mut [u8]) {
        self.evaluate_bound(Arrays::of(inputs, count), index, output)
    }

    /// Evaluate the element at `index`, with both arrays supplied.
    ///
    /// `evaluate_at` is this with no second array bound, which is what every
    /// body written before `GatherAux` needs and is the overwhelmingly common
    /// case. A body that gathers from an array it was not given reads zeroes
    /// rather than faulting, for the same reason an out-of-range gather clamps
    /// rather than faulting: this language is total, and a backend handed a
    /// malformed request rejects it at the boundary (`InvalidInput`) instead of
    /// the interpreter trapping in the middle of an element.
    pub fn evaluate_bound(&self, arrays: Arrays<'_>, index: u32, output: &mut [u8]) {
        let inputs = arrays.inputs;
        let count = arrays.count;
        let stride = self.stride() as usize;
        let own = (index as usize).saturating_mul(stride);
        let input = inputs.get(own..own + stride).unwrap_or(&[]);

        // Values are a fixed array indexed by instruction rather than a stack
        // that grows as the walk proceeds, because a loop re-executes an
        // instruction and would otherwise push a second slot for it. Slots for
        // instructions inside a loop hold the current iteration's value; the
        // escape rule checked at validation is what makes that the only value
        // anything can read.
        let mut values: Vec<u64> = vec![0; self.ops.len()];
        let mut locals: Vec<u64> = vec![0; self.locals as usize];
        // (index of the `Repeat`, iterations still to run after this one).
        let mut loops: Vec<(usize, u32)> = Vec::new();

        let mut pc = 0usize;
        while pc < self.ops.len() {
            let value = match self.ops[pc] {
                Op::Load(field) => self.read_field(input, field),
                Op::Index => u64::from(index),
                Op::Gather(at, field) => {
                    self.gather_field(inputs, stride, count, values[at as usize], field)
                }
                Op::GatherAux(at, field) => match &self.aux {
                    Some(layout) => read_clamped(
                        arrays.aux,
                        layout,
                        arrays.aux_count,
                        values[at as usize],
                        field,
                    ),
                    None => 0,
                },
                Op::Const(value) => value,
                Op::FConst(bits) => u64::from(canonical_f32_bits(bits)),
                Op::Add(a, b) => values[a as usize].wrapping_add(values[b as usize]),
                Op::Sub(a, b) => values[a as usize].wrapping_sub(values[b as usize]),
                Op::Mul(a, b) => values[a as usize].wrapping_mul(values[b as usize]),
                Op::FAdd(a, b) => u64::from(canonical_f32_bits(
                    (f32::from_bits(canonical_f32_bits(values[a as usize] as u32))
                        + f32::from_bits(canonical_f32_bits(values[b as usize] as u32)))
                    .to_bits(),
                )),
                Op::FMul(a, b) => u64::from(canonical_f32_bits(
                    (f32::from_bits(canonical_f32_bits(values[a as usize] as u32))
                        * f32::from_bits(canonical_f32_bits(values[b as usize] as u32)))
                    .to_bits(),
                )),
                Op::And(a, b) => values[a as usize] & values[b as usize],
                Op::Or(a, b) => values[a as usize] | values[b as usize],
                Op::Xor(a, b) => values[a as usize] ^ values[b as usize],
                // Masking the shift amount rather than saturating keeps this
                // identical to the `&63` the generated MSL performs.
                Op::Shl(a, b) => values[a as usize] << (values[b as usize] & 63),
                Op::Shr(a, b) => values[a as usize] >> (values[b as usize] & 63),
                Op::CmpEq(a, b) => u64::from(values[a as usize] == values[b as usize]),
                Op::CmpLt(a, b) => u64::from(values[a as usize] < values[b as usize]),
                Op::Select(c, a, b) => {
                    if values[c as usize] != 0 {
                        values[a as usize]
                    } else {
                        values[b as usize]
                    }
                }
                Op::Get(local) => locals[local as usize],
                Op::Set(local, value) => {
                    let value = values[value as usize];
                    locals[local as usize] = value;
                    value
                }
                Op::Repeat(trips) => {
                    // `trips` is at least 1 (validation rejects zero), so the
                    // body always runs once and the count recorded is what is
                    // left after this iteration.
                    loops.push((pc, trips - 1));
                    0
                }
                Op::EndRepeat => {
                    if let Some((start, remaining)) = loops.pop() {
                        if remaining > 0 {
                            loops.push((start, remaining - 1));
                            values[pc] = 0;
                            pc = start + 1;
                            continue;
                        }
                    }
                    0
                }
                Op::BreakIf(cond) => {
                    if values[cond as usize] != 0 {
                        if let Some((start, _)) = loops.pop() {
                            values[pc] = 0;
                            pc = self.end_of_loop(start) + 1;
                            continue;
                        }
                    }
                    0
                }
            };
            values[pc] = value;
            pc += 1;
        }

        for store in &self.stores {
            self.write_field(output, store.field, values[store.value as usize]);
        }
    }

    /// The index of the `EndRepeat` closing the `Repeat` at `start`.
    ///
    /// Validation has already established the structure is balanced, so this
    /// always finds one; the fallback is the end of the body, which makes a
    /// hypothetical unbalanced program stop rather than run off.
    fn end_of_loop(&self, start: usize) -> usize {
        let mut depth = 0usize;
        for (index, op) in self.ops.iter().enumerate().skip(start) {
            match op {
                Op::Repeat(_) => depth += 1,
                Op::EndRepeat => {
                    depth -= 1;
                    if depth == 0 {
                        return index;
                    }
                }
                _ => {}
            }
        }
        self.ops.len()
    }

    /// Read `field` of the element at `at`, clamping `at` to the last element.
    ///
    /// The clamp is written as a comparison against `count` rather than
    /// against `inputs.len() / stride` so that it depends only on values the
    /// generated MSL also has in hand. A backend that clamped against its own
    /// buffer length could disagree with this one whenever a buffer is larger
    /// than the element count it was dispatched with.
    fn gather_field(&self, inputs: &[u8], _stride: usize, count: u32, at: u64, field: u32) -> u64 {
        read_clamped(inputs, &self.layout, count, at, field)
    }

    fn read_field(&self, element: &[u8], field: u32) -> u64 {
        read_element_field(element, &self.layout, field)
    }

    fn write_field(&self, element: &mut [u8], field: u32, value: u64) {
        let (Some(offset), Some(width)) = (self.layout.offset(field), self.layout.width(field))
        else {
            return;
        };
        let value = if width == FieldWidth::F32 {
            u64::from(canonical_f32_bits(value as u32))
        } else {
            value & width.mask()
        };
        for byte in 0..width.bytes() {
            let index = (offset + byte) as usize;
            if let Some(slot) = element.get_mut(index) {
                *slot = ((value >> (8 * byte)) & 0xFF) as u8;
            }
        }
    }

    /// Parse the body lines belonging to one evaluator.
    ///
    /// ```text
    /// field u32
    /// field u8
    /// op 0 load 0
    /// op 1 const 2
    /// op 2 mul 0 1
    /// store 0 2
    /// ```
    ///
    /// `op` lines carry their own index and must be consecutive from zero, so
    /// a mistyped operand is a parse error rather than a silently different
    /// program.
    pub fn parse_lines(id: u32, name: &str, lines: &[&str]) -> Result<Self, BodyError> {
        let mut fields = Vec::new();
        let mut ops = Vec::new();
        let mut stores = Vec::new();
        let mut locals = 0u32;
        let mut aux: Vec<FieldWidth> = Vec::new();

        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.first().copied() {
                Some("field") => {
                    if parts.len() != 2 {
                        return Err(BodyError::Syntax);
                    }
                    fields.push(FieldWidth::parse(parts[1]).ok_or(BodyError::Syntax)?);
                }
                Some("aux") => {
                    if parts.len() != 2 {
                        return Err(BodyError::Syntax);
                    }
                    aux.push(FieldWidth::parse(parts[1]).ok_or(BodyError::Syntax)?);
                }
                Some("locals") => {
                    if parts.len() != 2 {
                        return Err(BodyError::Syntax);
                    }
                    locals = parts[1].parse().map_err(|_| BodyError::Syntax)?;
                }
                Some("op") => {
                    if parts.len() < 3 {
                        return Err(BodyError::Syntax);
                    }
                    let index: usize = parts[1].parse().map_err(|_| BodyError::Syntax)?;
                    if index != ops.len() {
                        return Err(BodyError::Syntax);
                    }
                    ops.push(parse_op(&parts[2..])?);
                }
                Some("store") => {
                    if parts.len() != 3 {
                        return Err(BodyError::Syntax);
                    }
                    stores.push(Store {
                        field: parts[1].parse().map_err(|_| BodyError::Syntax)?,
                        value: parts[2].parse().map_err(|_| BodyError::Syntax)?,
                    });
                }
                _ => return Err(BodyError::Syntax),
            }
        }

        let aux = (!aux.is_empty()).then(|| ElementLayout::new(aux));
        EvaluatorProgram::bound(
            id,
            name,
            ElementLayout::new(fields),
            aux,
            locals,
            ops,
            stores,
        )
    }

    /// Generate a Metal Shading Language kernel for this body.
    ///
    /// Codegen lives beside the interpreter rather than in the Metal backend
    /// so that the two lowerings of one instruction sit next to each other.
    /// When they disagree, I20 fails; keeping them adjacent is what makes the
    /// disagreement easy to see.
    pub fn metal_source(&self) -> String {
        let mut body = String::new();
        let _ = writeln!(body, "kernel void {}(", self.metal_entry_point());
        let _ = writeln!(body, "    device const uchar* input [[buffer(0)]],");
        let _ = writeln!(body, "    device uchar* output [[buffer(1)]],");
        let _ = writeln!(body, "    constant uint& count [[buffer(2)]],");
        let _ = writeln!(body, "    constant uint& stride [[buffer(3)]],");
        // The second array's three parameters exist only when the body reads
        // one. Emitting them unconditionally would mean binding a placeholder
        // buffer for every body that does not gather from an aux array, which
        // is nearly all of them; a backend knows which case it is in because it
        // has the program installed.
        if self.aux.is_some() {
            let _ = writeln!(body, "    device const uchar* aux [[buffer(4)]],");
            let _ = writeln!(body, "    constant uint& aux_count [[buffer(5)]],");
            let _ = writeln!(body, "    constant uint& aux_stride [[buffer(6)]],");
        }
        let _ = writeln!(body, "    uint gid [[thread_position_in_grid]])");
        let _ = writeln!(body, "{{");
        let _ = writeln!(body, "    if (gid >= count) return;");
        let _ = writeln!(body, "    uint base = gid * stride;");
        let _ = writeln!(
            body,
            "    for (uint b = 0; b < stride; ++b) output[base + b] = input[base + b];"
        );

        // Every value and local is declared up front rather than at the point
        // it is computed. A loop body is a C scope, so a `ulong v7` declared
        // inside one stops existing at the closing brace — and the whole reason
        // locals exist is for a value to outlive an iteration. Hoisting also
        // makes the emitted code's shape independent of the control flow around
        // an instruction, which is what keeps this lowering readable against
        // the interpreter's `values` array.
        for index in 0..self.ops.len() {
            let _ = writeln!(body, "    ulong v{index} = 0ul;");
        }
        for local in 0..self.locals {
            let _ = writeln!(body, "    ulong l{local} = 0ul;");
        }
        // A gather needs two scratch values per instruction for the same reason
        // it did before: the clamped index is used by every byte term of the
        // read, and recomputing it per byte would emit a kernel quadratic in
        // the field width.
        for (index, op) in self.ops.iter().enumerate() {
            if matches!(op, Op::Gather(_, _) | Op::GatherAux(_, _)) {
                let _ = writeln!(body, "    uint g{index} = 0u;");
                let _ = writeln!(body, "    uint gb{index} = 0u;");
            }
        }

        let mut indent = 1usize;
        for (index, op) in self.ops.iter().enumerate() {
            let pad = "    ".repeat(indent);
            match *op {
                Op::Gather(at, field) => {
                    let _ = writeln!(
                        body,
                        "{pad}g{index} = (v{at} >= (ulong)count) ? (count - 1u) : uint(v{at});"
                    );
                    let _ = writeln!(body, "{pad}gb{index} = g{index} * stride;");
                    let _ = writeln!(
                        body,
                        "{pad}v{index} = {};",
                        self.metal_read(&format!("gb{index}"), field)
                    );
                }
                Op::GatherAux(at, field) => {
                    let _ = writeln!(
                        body,
                        "{pad}g{index} = (v{at} >= (ulong)aux_count) ? (aux_count - 1u) : uint(v{at});"
                    );
                    let _ = writeln!(body, "{pad}gb{index} = g{index} * aux_stride;");
                    let _ = writeln!(
                        body,
                        "{pad}v{index} = {};",
                        self.metal_read_from(
                            "aux",
                            self.aux.as_ref(),
                            &format!("gb{index}"),
                            field
                        )
                    );
                }
                Op::Set(local, value) => {
                    let _ = writeln!(body, "{pad}l{local} = v{value};");
                    let _ = writeln!(body, "{pad}v{index} = v{value};");
                }
                Op::Repeat(trips) => {
                    let _ = writeln!(
                        body,
                        "{pad}for (uint t{index} = 0u; t{index} < {trips}u; ++t{index}) {{"
                    );
                    indent += 1;
                }
                Op::EndRepeat => {
                    // The closing brace belongs to the enclosing level, so the
                    // indent drops before it is written rather than after.
                    indent = indent.saturating_sub(1);
                    let _ = writeln!(body, "{}}}", "    ".repeat(indent));
                }
                Op::BreakIf(cond) => {
                    let _ = writeln!(body, "{pad}if (v{cond} != 0ul) break;");
                }
                _ => {
                    let expression = match *op {
                        Op::Load(field) => self.metal_read("base", field),
                        Op::Index => "ulong(gid)".to_string(),
                        Op::Const(value) => format!("{value}ul"),
                        Op::FConst(bits) => {
                            format!("ulong(soma_f32_bits(as_type<float>({bits}u)))")
                        }
                        Op::Add(a, b) => format!("v{a} + v{b}"),
                        Op::Sub(a, b) => format!("v{a} - v{b}"),
                        Op::Mul(a, b) => format!("v{a} * v{b}"),
                        Op::FAdd(a, b) => format!(
                            "ulong(soma_f32_bits(soma_f32_value(v{a}) + soma_f32_value(v{b})))"
                        ),
                        Op::FMul(a, b) => format!(
                            "ulong(soma_f32_bits(soma_f32_value(v{a}) * soma_f32_value(v{b})))"
                        ),
                        Op::And(a, b) => format!("v{a} & v{b}"),
                        Op::Or(a, b) => format!("v{a} | v{b}"),
                        Op::Xor(a, b) => format!("v{a} ^ v{b}"),
                        Op::Shl(a, b) => format!("v{a} << (v{b} & 63ul)"),
                        Op::Shr(a, b) => format!("v{a} >> (v{b} & 63ul)"),
                        Op::CmpEq(a, b) => format!("(v{a} == v{b}) ? 1ul : 0ul"),
                        Op::CmpLt(a, b) => format!("(v{a} < v{b}) ? 1ul : 0ul"),
                        Op::Select(c, a, b) => format!("(v{c} != 0ul) ? v{a} : v{b}"),
                        Op::Get(local) => format!("l{local}"),
                        // Handled above.
                        Op::Gather(_, _)
                        | Op::Set(_, _)
                        | Op::Repeat(_)
                        | Op::EndRepeat
                        | Op::GatherAux(_, _)
                        | Op::BreakIf(_) => unreachable!(),
                    };
                    let _ = writeln!(body, "{pad}v{index} = {expression};");
                }
            }
        }

        for store in &self.stores {
            let _ = writeln!(body, "{}", self.metal_store(store));
        }

        let _ = writeln!(body, "}}");

        format!("#include <metal_stdlib>\nusing namespace metal;\n\ninline uint soma_f32_bits(float x) {{\n    uint bits = as_type<uint>(x);\n    uint magnitude = bits & 0x7fffffffu;\n    return isnan(x) ? 0x7fc00000u : ((magnitude < 0x00800000u) ? 0u : bits);\n}}\ninline float soma_f32_value(ulong bits) {{ return as_type<float>(soma_f32_bits(as_type<float>(uint(bits)))); }}\n\n{body}")
    }

    pub fn metal_entry_point(&self) -> String {
        format!("soma_evaluator_{}", self.id)
    }

    /// Read one field out of the input buffer, little-endian, starting at
    /// `base`.
    ///
    /// A load and a gather differ only in that expression — `base` for this
    /// lane's own element, a clamped computed offset for a gather — so they
    /// share the byte assembly. Two copies of it would be two places for the
    /// endianness to drift away from `read_field`, and only one of them would
    /// be the one I20 happened to exercise.
    fn metal_read(&self, base: &str, field: u32) -> String {
        self.metal_read_from("input", Some(&self.layout), base, field)
    }

    /// The same byte assembly against a named buffer and layout.
    ///
    /// A load, a gather and an aux gather differ only in which buffer they
    /// index and from which offset; sharing the assembly is what stops the
    /// endianness drifting between them, and only one of three copies would be
    /// the one I20's examples happened to exercise.
    fn metal_read_from(
        &self,
        buffer: &str,
        layout: Option<&ElementLayout>,
        base: &str,
        field: u32,
    ) -> String {
        let (Some(offset), Some(width)) = (
            layout.and_then(|l| l.offset(field)),
            layout.and_then(|l| l.width(field)),
        ) else {
            return "0ul".to_string();
        };
        (0..width.bytes())
            .map(|byte| {
                format!(
                    "(ulong({buffer}[{base} + {}]) << {})",
                    offset + byte,
                    8 * byte
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    fn metal_store(&self, store: &Store) -> String {
        let (Some(offset), Some(width)) = (
            self.layout.offset(store.field),
            self.layout.width(store.field),
        ) else {
            return String::new();
        };
        let value = if width == FieldWidth::F32 {
            format!(
                "ulong(soma_f32_bits(as_type<float>(uint(v{}))))",
                store.value
            )
        } else {
            format!("v{}", store.value)
        };
        let mut lines = Vec::new();
        for byte in 0..width.bytes() {
            lines.push(format!(
                "    output[base + {}] = uchar(({} >> {}) & 0xFFul);",
                offset + byte,
                value,
                8 * byte
            ));
        }
        lines.join("\n")
    }
}

/// The arrays one evaluation reads.
///
/// A body always has an input array — the one whose elements it is iterating
/// and whose element it writes — and may have a second, read-only one it
/// gathers from. Bundling them keeps the interpreter's signature from growing
/// a parameter every time a binding is added, and keeps the "not bound" case
/// spelled one way.
#[derive(Clone, Copy, Debug, Default)]
pub struct Arrays<'a> {
    pub inputs: &'a [u8],
    pub count: u32,
    pub aux: &'a [u8],
    pub aux_count: u32,
}

impl<'a> Arrays<'a> {
    /// One array, no second binding.
    pub fn of(inputs: &'a [u8], count: u32) -> Self {
        Self {
            inputs,
            count,
            aux: &[],
            aux_count: 0,
        }
    }

    pub fn with_aux(mut self, aux: &'a [u8], aux_count: u32) -> Self {
        self.aux = aux;
        self.aux_count = aux_count;
        self
    }
}

/// Read `field` of the element at `at` in `bytes`, clamping `at` to the last
/// element.
///
/// Shared by `Gather` and `GatherAux` so the clamp cannot drift between them —
/// two copies of this rule would be two places for the two lowerings to
/// disagree, and I20 would only be exercising whichever one the examples
/// happened to use.
///
/// The clamp compares against `count` rather than against `bytes.len() /
/// stride` so that it depends only on values the generated MSL also has in
/// hand. A backend clamping against its own buffer length could disagree
/// whenever a buffer is larger than the count it was dispatched with.
/// Canonical representation used at every float-producing boundary.
/// All NaNs collapse to one quiet positive NaN and both signed zeros collapse
/// to +0, removing backend choices that would otherwise violate I20.
fn canonical_f32_bits(bits: u32) -> u32 {
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        0x7fc0_0000
    } else if magnitude < 0x0080_0000 {
        // Apple GPU arithmetic flushes binary32 subnormals. Making that
        // boundary explicit on every backend preserves byte-level I20.
        0
    } else {
        bits
    }
}

fn read_clamped(bytes: &[u8], layout: &ElementLayout, count: u32, at: u64, field: u32) -> u64 {
    if count == 0 {
        return 0;
    }
    let stride = layout.stride() as usize;
    let last = u64::from(count - 1);
    let clamped = if at > last { last } else { at } as usize;
    let base = clamped.saturating_mul(stride);
    let element = bytes.get(base..base + stride).unwrap_or(&[]);
    read_element_field(element, layout, field)
}

fn read_element_field(element: &[u8], layout: &ElementLayout, field: u32) -> u64 {
    let (Some(offset), Some(width)) = (layout.offset(field), layout.width(field)) else {
        return 0;
    };
    let mut value = 0u64;
    for byte in 0..width.bytes() {
        let index = (offset + byte) as usize;
        let byte_value = element.get(index).copied().unwrap_or(0) as u64;
        value |= byte_value << (8 * byte);
    }
    value
}

fn parse_op(parts: &[&str]) -> Result<Op, BodyError> {
    let number = |index: usize| -> Result<u32, BodyError> {
        parts
            .get(index)
            .ok_or(BodyError::Syntax)?
            .parse()
            .map_err(|_| BodyError::Syntax)
    };
    let binary =
        |make: fn(u32, u32) -> Op| -> Result<Op, BodyError> { Ok(make(number(1)?, number(2)?)) };
    match parts.first().copied().ok_or(BodyError::Syntax)? {
        "load" => Ok(Op::Load(number(1)?)),
        "index" => Ok(Op::Index),
        "gather" => Ok(Op::Gather(number(1)?, number(2)?)),
        "gatheraux" => Ok(Op::GatherAux(number(1)?, number(2)?)),
        "const" => Ok(Op::Const(
            parts
                .get(1)
                .ok_or(BodyError::Syntax)?
                .parse()
                .map_err(|_| BodyError::Syntax)?,
        )),
        "fconst" => {
            let text = parts.get(1).ok_or(BodyError::Syntax)?;
            let bits = if let Some(hex) = text.strip_prefix("0x") {
                u32::from_str_radix(hex, 16).map_err(|_| BodyError::Syntax)?
            } else {
                text.parse::<f32>()
                    .map_err(|_| BodyError::Syntax)?
                    .to_bits()
            };
            Ok(Op::FConst(bits))
        }
        "add" => binary(Op::Add),
        "sub" => binary(Op::Sub),
        "mul" => binary(Op::Mul),
        "fadd" => binary(Op::FAdd),
        "fmul" => binary(Op::FMul),
        "and" => binary(Op::And),
        "or" => binary(Op::Or),
        "xor" => binary(Op::Xor),
        "shl" => binary(Op::Shl),
        "shr" => binary(Op::Shr),
        "cmpeq" => binary(Op::CmpEq),
        "cmplt" => binary(Op::CmpLt),
        "select" => Ok(Op::Select(number(1)?, number(2)?, number(3)?)),
        "get" => Ok(Op::Get(number(1)?)),
        "set" => binary(Op::Set),
        "repeat" => Ok(Op::Repeat(number(1)?)),
        "endrepeat" => Ok(Op::EndRepeat),
        "breakif" => Ok(Op::BreakIf(number(1)?)),
        _ => Err(BodyError::Syntax),
    }
}

/// Whether `outer` is a prefix of `inner`, over loop-nesting stacks.
///
/// This is the escape rule. A stack is the chain of loops an instruction sits
/// inside, outermost first, so one instruction can read another exactly when
/// every loop enclosing the producer also encloses the consumer — otherwise the
/// producer's loop has ended and its value belongs to an iteration that is
/// over. Prefix rather than equality, because reading *out* of an enclosing
/// scope into a loop body is fine: that value does not change while the loop
/// runs.
fn is_prefix(outer: &[usize], inner: &[usize]) -> bool {
    outer.len() <= inner.len() && outer.iter().zip(inner).all(|(a, b)| a == b)
}
