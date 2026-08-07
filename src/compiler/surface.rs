//! Named-value source language for evaluator bodies.
//!
//! The low-level body IR uses instruction numbers because that is the compact
//! representation every backend validates and lowers. Humans should not have
//! to renumber a whole program after inserting one expression. This surface
//! gives values and locals names, preserves structured repeat/end blocks, and
//! lowers to exactly one `EvaluatorProgram`; validation remains centralized in
//! the body IR.
//!
//! ```text
//! field u32
//! field u32
//! local sum
//! let one = const 1
//! let zero = const 0
//! set sum zero
//! repeat 8
//!   let old = get sum
//!   let next = add old one
//!   set sum next
//! end
//! let result = get sum
//! store 1 result
//! ```

use std::collections::HashMap;

use super::body::{BodyError, ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceErrorKind {
    Syntax,
    DuplicateName,
    UnknownValue,
    UnknownLocal,
    InvalidInteger,
    InvalidWidth,
    Body(BodyError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceError {
    pub line: usize,
    pub kind: SurfaceErrorKind,
}

impl SurfaceError {
    fn at(line: usize, kind: SurfaceErrorKind) -> Self {
        Self { line, kind }
    }
}

/// Compile a named evaluator body.
///
/// Declarations and statements are line-oriented. A comment begins with `#`.
/// Values are immutable; locals are the explicit state carried by loops.
pub fn compile_evaluator(
    id: u32,
    name: impl Into<String>,
    source: &str,
) -> Result<EvaluatorProgram, SurfaceError> {
    let mut fields = Vec::new();
    let mut aux_fields = Vec::new();
    let mut locals: HashMap<String, u32> = HashMap::new();
    let mut values: HashMap<String, u32> = HashMap::new();
    let mut ops = Vec::new();
    let mut stores = Vec::new();

    for (line_index, raw) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let text = raw.split('#').next().unwrap_or_default().trim();
        if text.is_empty() {
            continue;
        }
        let words: Vec<_> = text.split_whitespace().collect();
        match words.as_slice() {
            ["field", width] => fields.push(parse_width(width, line_number)?),
            ["aux", width] => aux_fields.push(parse_width(width, line_number)?),
            ["local", local] => {
                if locals.contains_key(*local) || values.contains_key(*local) {
                    return Err(SurfaceError::at(
                        line_number,
                        SurfaceErrorKind::DuplicateName,
                    ));
                }
                locals.insert((*local).to_string(), locals.len() as u32);
            }
            ["repeat", trips] => {
                ops.push(Op::Repeat(parse_u32(trips, line_number)?));
            }
            ["end"] => ops.push(Op::EndRepeat),
            ["break_if", condition] => {
                ops.push(Op::BreakIf(value(&values, condition, line_number)?));
            }
            ["set", local, input] => {
                let local = local_id(&locals, local, line_number)?;
                let input = value(&values, input, line_number)?;
                ops.push(Op::Set(local, input));
            }
            ["store", field, input] => stores.push(Store {
                field: parse_u32(field, line_number)?,
                value: value(&values, input, line_number)?,
            }),
            ["let", result, "=", rest @ ..] => {
                if locals.contains_key(*result) || values.contains_key(*result) {
                    return Err(SurfaceError::at(
                        line_number,
                        SurfaceErrorKind::DuplicateName,
                    ));
                }
                let op = parse_expression(rest, &values, &locals, line_number)?;
                let index = ops.len() as u32;
                ops.push(op);
                values.insert((*result).to_string(), index);
            }
            _ => return Err(SurfaceError::at(line_number, SurfaceErrorKind::Syntax)),
        }
    }

    let aux = (!aux_fields.is_empty()).then(|| ElementLayout::new(aux_fields));
    EvaluatorProgram::bound(
        id,
        name,
        ElementLayout::new(fields),
        aux,
        locals.len() as u32,
        ops,
        stores,
    )
    .map_err(|error| SurfaceError::at(0, SurfaceErrorKind::Body(error)))
}

fn parse_expression(
    words: &[&str],
    values: &HashMap<String, u32>,
    locals: &HashMap<String, u32>,
    line: usize,
) -> Result<Op, SurfaceError> {
    let op = match words {
        ["load", field] => Op::Load(parse_u32(field, line)?),
        ["index"] => Op::Index,
        ["gather", at, field] => Op::Gather(value(values, at, line)?, parse_u32(field, line)?),
        ["gather_aux", at, field] => {
            Op::GatherAux(value(values, at, line)?, parse_u32(field, line)?)
        }
        ["const", constant] => Op::Const(parse_u64(constant, line)?),
        ["get", local] => Op::Get(local_id(locals, local, line)?),
        ["add", a, b] => binary(values, a, b, line, Op::Add)?,
        ["sub", a, b] => binary(values, a, b, line, Op::Sub)?,
        ["mul", a, b] => binary(values, a, b, line, Op::Mul)?,
        ["and", a, b] => binary(values, a, b, line, Op::And)?,
        ["or", a, b] => binary(values, a, b, line, Op::Or)?,
        ["xor", a, b] => binary(values, a, b, line, Op::Xor)?,
        ["shl", a, b] => binary(values, a, b, line, Op::Shl)?,
        ["shr", a, b] => binary(values, a, b, line, Op::Shr)?,
        ["eq", a, b] => binary(values, a, b, line, Op::CmpEq)?,
        ["lt", a, b] => binary(values, a, b, line, Op::CmpLt)?,
        ["select", condition, yes, no] => Op::Select(
            value(values, condition, line)?,
            value(values, yes, line)?,
            value(values, no, line)?,
        ),
        _ => return Err(SurfaceError::at(line, SurfaceErrorKind::Syntax)),
    };
    Ok(op)
}

fn binary(
    values: &HashMap<String, u32>,
    a: &str,
    b: &str,
    line: usize,
    constructor: fn(u32, u32) -> Op,
) -> Result<Op, SurfaceError> {
    Ok(constructor(
        value(values, a, line)?,
        value(values, b, line)?,
    ))
}

fn value(values: &HashMap<String, u32>, name: &str, line: usize) -> Result<u32, SurfaceError> {
    values
        .get(name)
        .copied()
        .ok_or_else(|| SurfaceError::at(line, SurfaceErrorKind::UnknownValue))
}

fn local_id(locals: &HashMap<String, u32>, name: &str, line: usize) -> Result<u32, SurfaceError> {
    locals
        .get(name)
        .copied()
        .ok_or_else(|| SurfaceError::at(line, SurfaceErrorKind::UnknownLocal))
}

fn parse_width(text: &str, line: usize) -> Result<FieldWidth, SurfaceError> {
    match text {
        "u8" => Ok(FieldWidth::U8),
        "u16" => Ok(FieldWidth::U16),
        "u32" => Ok(FieldWidth::U32),
        "u64" => Ok(FieldWidth::U64),
        _ => Err(SurfaceError::at(line, SurfaceErrorKind::InvalidWidth)),
    }
}

fn parse_u32(text: &str, line: usize) -> Result<u32, SurfaceError> {
    parse_u64(text, line)?
        .try_into()
        .map_err(|_| SurfaceError::at(line, SurfaceErrorKind::InvalidInteger))
}

fn parse_u64(text: &str, line: usize) -> Result<u64, SurfaceError> {
    let parsed = if let Some(hex) = text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
    } else {
        text.parse()
    };
    parsed.map_err(|_| SurfaceError::at(line, SurfaceErrorKind::InvalidInteger))
}
