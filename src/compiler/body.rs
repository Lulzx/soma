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
//! - **Pure and total.** No allocation, no division (which would be partial),
//!   no loops. Every program terminates in a fixed number of steps decided at
//!   validation time. An out-of-range `Gather` clamps rather than faulting, so
//!   totality survives a computed index.
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
//! - **Integer-only.** Admitting `f32` would force I20 to either demand
//!   bit-identical results across backends — constraining what a GPU may do —
//!   or weaken to a tolerance, which guts it as an invariant. Deferred until
//!   there is a reason to pay for it.
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
}

impl FieldWidth {
    pub fn bytes(self) -> u32 {
        match self {
            FieldWidth::U8 => 1,
            FieldWidth::U16 => 2,
            FieldWidth::U32 => 4,
            FieldWidth::U64 => 8,
        }
    }

    pub fn mask(self) -> u64 {
        match self {
            FieldWidth::U8 => 0xFF,
            FieldWidth::U16 => 0xFFFF,
            FieldWidth::U32 => 0xFFFF_FFFF,
            FieldWidth::U64 => u64::MAX,
        }
    }

    fn parse(text: &str) -> Option<FieldWidth> {
        match text {
            "u8" => Some(FieldWidth::U8),
            "u16" => Some(FieldWidth::U16),
            "u32" => Some(FieldWidth::U32),
            "u64" => Some(FieldWidth::U64),
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
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
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
    /// `cond != 0 ? a : b`. The only control flow the language has, and it is
    /// branch-free, so a cohort of lanes executing one body never diverges.
    Select(u32, u32, u32),
}

impl Op {
    fn operands(self) -> Vec<u32> {
        match self {
            Op::Load(_) | Op::Const(_) | Op::Index => Vec::new(),
            Op::Gather(index, _) => vec![index],
            Op::Add(a, b)
            | Op::Sub(a, b)
            | Op::Mul(a, b)
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
    Syntax,
}

impl EvaluatorProgram {
    pub fn new(
        id: u32,
        name: impl Into<String>,
        layout: ElementLayout,
        ops: Vec<Op>,
        stores: Vec<Store>,
    ) -> Result<Self, BodyError> {
        let program = Self {
            id,
            name: name.into(),
            layout,
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
        let field_count = self.layout.fields().len() as u32;
        for (index, op) in self.ops.iter().enumerate() {
            // The gathered *field* is static and checked here exactly as a
            // load's is. Only the element index is dynamic, and that clamps.
            if let Op::Load(field) | Op::Gather(_, field) = op {
                if *field >= field_count {
                    return Err(BodyError::FieldOutOfRange);
                }
            }
            for operand in op.operands() {
                if operand as usize >= index {
                    return Err(BodyError::ForwardReference);
                }
            }
        }
        for store in &self.stores {
            if store.field >= field_count {
                return Err(BodyError::FieldOutOfRange);
            }
            if store.value as usize >= self.ops.len() {
                return Err(BodyError::ForwardReference);
            }
        }
        Ok(())
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
        let stride = self.stride() as usize;
        let own = (index as usize).saturating_mul(stride);
        let input = inputs.get(own..own + stride).unwrap_or(&[]);
        let mut values: Vec<u64> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let value = match *op {
                Op::Load(field) => self.read_field(input, field),
                Op::Index => u64::from(index),
                Op::Gather(at, field) => {
                    self.gather_field(inputs, stride, count, values[at as usize], field)
                }
                Op::Const(value) => value,
                Op::Add(a, b) => values[a as usize].wrapping_add(values[b as usize]),
                Op::Sub(a, b) => values[a as usize].wrapping_sub(values[b as usize]),
                Op::Mul(a, b) => values[a as usize].wrapping_mul(values[b as usize]),
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
            };
            values.push(value);
        }
        for store in &self.stores {
            self.write_field(output, store.field, values[store.value as usize]);
        }
    }

    /// Read `field` of the element at `at`, clamping `at` to the last element.
    ///
    /// The clamp is written as a comparison against `count` rather than
    /// against `inputs.len() / stride` so that it depends only on values the
    /// generated MSL also has in hand. A backend that clamped against its own
    /// buffer length could disagree with this one whenever a buffer is larger
    /// than the element count it was dispatched with.
    fn gather_field(&self, inputs: &[u8], stride: usize, count: u32, at: u64, field: u32) -> u64 {
        if count == 0 {
            return 0;
        }
        let last = u64::from(count - 1);
        let clamped = if at > last { last } else { at } as usize;
        let base = clamped.saturating_mul(stride);
        let element = inputs.get(base..base + stride).unwrap_or(&[]);
        self.read_field(element, field)
    }

    fn read_field(&self, element: &[u8], field: u32) -> u64 {
        let (Some(offset), Some(width)) = (self.layout.offset(field), self.layout.width(field))
        else {
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

    fn write_field(&self, element: &mut [u8], field: u32, value: u64) {
        let (Some(offset), Some(width)) = (self.layout.offset(field), self.layout.width(field))
        else {
            return;
        };
        let value = value & width.mask();
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

        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.first().copied() {
                Some("field") => {
                    if parts.len() != 2 {
                        return Err(BodyError::Syntax);
                    }
                    fields.push(FieldWidth::parse(parts[1]).ok_or(BodyError::Syntax)?);
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

        EvaluatorProgram::new(id, name, ElementLayout::new(fields), ops, stores)
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
        let _ = writeln!(body, "    uint gid [[thread_position_in_grid]])");
        let _ = writeln!(body, "{{");
        let _ = writeln!(body, "    if (gid >= count) return;");
        let _ = writeln!(body, "    uint base = gid * stride;");
        let _ = writeln!(
            body,
            "    for (uint b = 0; b < stride; ++b) output[base + b] = input[base + b];"
        );

        for (index, op) in self.ops.iter().enumerate() {
            // A gather is the one instruction that is not a single expression:
            // the clamped index is needed by every byte term of the read, and
            // recomputing it per byte would emit a kernel quadratic in the
            // field width. It gets its own statements instead.
            if let Op::Gather(at, field) = *op {
                let _ = writeln!(
                    body,
                    "    uint g{index} = (v{at} >= (ulong)count) ? (count - 1u) : uint(v{at});"
                );
                let _ = writeln!(body, "    uint gb{index} = g{index} * stride;");
                let _ = writeln!(
                    body,
                    "    ulong v{index} = {};",
                    self.metal_read(&format!("gb{index}"), field)
                );
                continue;
            }
            let expression = match *op {
                Op::Load(field) => self.metal_read("base", field),
                Op::Index => "ulong(gid)".to_string(),
                // Emitted above.
                Op::Gather(_, _) => continue,
                Op::Const(value) => format!("{value}ul"),
                Op::Add(a, b) => format!("v{a} + v{b}"),
                Op::Sub(a, b) => format!("v{a} - v{b}"),
                Op::Mul(a, b) => format!("v{a} * v{b}"),
                Op::And(a, b) => format!("v{a} & v{b}"),
                Op::Or(a, b) => format!("v{a} | v{b}"),
                Op::Xor(a, b) => format!("v{a} ^ v{b}"),
                Op::Shl(a, b) => format!("v{a} << (v{b} & 63ul)"),
                Op::Shr(a, b) => format!("v{a} >> (v{b} & 63ul)"),
                Op::CmpEq(a, b) => format!("(v{a} == v{b}) ? 1ul : 0ul"),
                Op::CmpLt(a, b) => format!("(v{a} < v{b}) ? 1ul : 0ul"),
                Op::Select(c, a, b) => format!("(v{c} != 0ul) ? v{a} : v{b}"),
            };
            let _ = writeln!(body, "    ulong v{index} = {expression};");
        }

        for store in &self.stores {
            let _ = writeln!(body, "{}", self.metal_store(store));
        }

        let _ = writeln!(body, "}}");

        format!("#include <metal_stdlib>\nusing namespace metal;\n\n{body}")
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
        let (Some(offset), Some(width)) = (self.layout.offset(field), self.layout.width(field))
        else {
            return "0ul".to_string();
        };
        (0..width.bytes())
            .map(|byte| format!("(ulong(input[{base} + {}]) << {})", offset + byte, 8 * byte))
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
        let mut lines = Vec::new();
        for byte in 0..width.bytes() {
            lines.push(format!(
                "    output[base + {}] = uchar((v{} >> {}) & 0xFFul);",
                offset + byte,
                store.value,
                8 * byte
            ));
        }
        lines.join("\n")
    }
}

fn parse_op(parts: &[&str]) -> Result<Op, BodyError> {
    let number = |index: usize| -> Result<u32, BodyError> {
        parts
            .get(index)
            .ok_or(BodyError::Syntax)?
            .parse()
            .map_err(|_| BodyError::Syntax)
    };
    let binary = |make: fn(u32, u32) -> Op| -> Result<Op, BodyError> {
        Ok(make(number(1)?, number(2)?))
    };
    match parts.first().copied().ok_or(BodyError::Syntax)? {
        "load" => Ok(Op::Load(number(1)?)),
        "index" => Ok(Op::Index),
        "gather" => Ok(Op::Gather(number(1)?, number(2)?)),
        "const" => Ok(Op::Const(
            parts
                .get(1)
                .ok_or(BodyError::Syntax)?
                .parse()
                .map_err(|_| BodyError::Syntax)?,
        )),
        "add" => binary(Op::Add),
        "sub" => binary(Op::Sub),
        "mul" => binary(Op::Mul),
        "and" => binary(Op::And),
        "or" => binary(Op::Or),
        "xor" => binary(Op::Xor),
        "shl" => binary(Op::Shl),
        "shr" => binary(Op::Shr),
        "cmpeq" => binary(Op::CmpEq),
        "cmplt" => binary(Op::CmpLt),
        "select" => Ok(Op::Select(number(1)?, number(2)?, number(3)?)),
        _ => Err(BodyError::Syntax),
    }
}
