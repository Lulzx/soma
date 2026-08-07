//! Named-value source language for evaluator bodies.
//!
//! The low-level body IR uses instruction numbers because that is the compact
//! representation every backend validates and lowers. Humans should not have
//! to renumber a whole program after inserting one expression. This surface
//! gives values and locals names, preserves structured repeat/end blocks, and
//! lowers to exactly one `EvaluatorProgram`; validation remains centralized in
//! the body IR.
//!
//! Reusable functions are pure expression macros. `fn NAME PARAM...` contains
//! `let` expressions followed by one `return VALUE` and `end`; `call` expands
//! the function at compile time. No call reaches a backend, recursion is
//! rejected, and the expanded body is checked against the ordinary step bound.

use std::collections::{HashMap, HashSet};

use super::body::{BodyError, ElementLayout, EvaluatorProgram, FieldWidth, Op, Store, MAX_STEPS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceErrorKind {
    Syntax,
    DuplicateName,
    UnknownValue,
    UnknownLocal,
    UnknownFunction,
    ArityMismatch,
    RecursiveCall,
    InvalidInteger,
    InvalidFloat,
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

#[derive(Clone, Debug)]
struct SourceLine {
    number: usize,
    words: Vec<String>,
}

#[derive(Clone, Debug)]
struct Function {
    line: usize,
    params: Vec<String>,
    body: Vec<SourceLine>,
    result: String,
}

/// Compile a named evaluator body.
///
/// Declarations and statements are line-oriented. A comment begins with `#`.
/// Values are immutable; locals are the explicit state carried by loops.
/// Functions are compile-time-only, pure expression macros and may call an
/// earlier or later function, but recursive call cycles are rejected.
pub fn compile_evaluator(
    id: u32,
    name: impl Into<String>,
    source: &str,
) -> Result<EvaluatorProgram, SurfaceError> {
    let lines = source_lines(source);
    let (functions, main) = collect_functions(&lines)?;
    validate_functions(&functions)?;
    let mut fields = Vec::new();
    let mut aux_fields = Vec::new();
    let mut locals: HashMap<String, u32> = HashMap::new();
    let mut values: HashMap<String, u32> = HashMap::new();
    let mut ops = Vec::new();
    let mut stores = Vec::new();

    for source_line in main {
        let line_number = source_line.number;
        let words: Vec<_> = source_line.words.iter().map(String::as_str).collect();
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
                push_op(
                    &mut ops,
                    Op::Repeat(parse_u32(trips, line_number)?),
                    line_number,
                )?;
            }
            ["end"] => {
                push_op(&mut ops, Op::EndRepeat, line_number)?;
            }
            ["break_if", condition] => {
                let condition = value(&values, condition, line_number)?;
                push_op(&mut ops, Op::BreakIf(condition), line_number)?;
            }
            ["set", local, input] => {
                let local = local_id(&locals, local, line_number)?;
                let input = value(&values, input, line_number)?;
                push_op(&mut ops, Op::Set(local, input), line_number)?;
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
                let index = emit_expression(
                    rest,
                    &values,
                    &locals,
                    &functions,
                    &mut ops,
                    &mut Vec::new(),
                    line_number,
                )?;
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

fn source_lines(source: &str) -> Vec<SourceLine> {
    source
        .lines()
        .enumerate()
        .filter_map(|(index, raw)| {
            let text = raw.split('#').next().unwrap_or_default().trim();
            (!text.is_empty()).then(|| SourceLine {
                number: index + 1,
                words: text.split_whitespace().map(str::to_string).collect(),
            })
        })
        .collect()
}

fn collect_functions(
    lines: &[SourceLine],
) -> Result<(HashMap<String, Function>, Vec<SourceLine>), SurfaceError> {
    let mut functions = HashMap::new();
    let mut main = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = &lines[index];
        if line.words.first().map(String::as_str) != Some("fn") {
            main.push(line.clone());
            index += 1;
            continue;
        }
        if line.words.len() < 2 {
            return Err(SurfaceError::at(line.number, SurfaceErrorKind::Syntax));
        }
        let function_name = line.words[1].clone();
        if functions.contains_key(&function_name) {
            return Err(SurfaceError::at(
                line.number,
                SurfaceErrorKind::DuplicateName,
            ));
        }
        let params = line.words[2..].to_vec();
        let mut seen = HashSet::new();
        for param in &params {
            if !seen.insert(param.clone()) {
                return Err(SurfaceError::at(
                    line.number,
                    SurfaceErrorKind::DuplicateName,
                ));
            }
        }

        index += 1;
        let mut body = Vec::new();
        let mut result = None;
        let mut closed = false;
        while index < lines.len() {
            let current = &lines[index];
            match current.words.as_slice() {
                words if words == ["end"] => {
                    closed = true;
                    index += 1;
                    break;
                }
                words if words.len() == 2 && words[0] == "return" => {
                    if result.is_some() {
                        return Err(SurfaceError::at(current.number, SurfaceErrorKind::Syntax));
                    }
                    if !seen.contains(&words[1]) {
                        return Err(SurfaceError::at(
                            current.number,
                            SurfaceErrorKind::UnknownValue,
                        ));
                    }
                    result = Some(words[1].clone());
                }
                words if words.len() >= 4 && words[0] == "let" && words[2] == "=" => {
                    if result.is_some() {
                        return Err(SurfaceError::at(current.number, SurfaceErrorKind::Syntax));
                    }
                    if !seen.insert(words[1].clone()) {
                        return Err(SurfaceError::at(
                            current.number,
                            SurfaceErrorKind::DuplicateName,
                        ));
                    }
                    body.push(current.clone());
                }
                _ => return Err(SurfaceError::at(current.number, SurfaceErrorKind::Syntax)),
            }
            index += 1;
        }
        if !closed || result.is_none() {
            return Err(SurfaceError::at(line.number, SurfaceErrorKind::Syntax));
        }
        functions.insert(
            function_name,
            Function {
                line: line.number,
                params,
                body,
                result: result.unwrap(),
            },
        );
    }
    Ok((functions, main))
}

fn validate_functions(functions: &HashMap<String, Function>) -> Result<(), SurfaceError> {
    let mut ordered: Vec<_> = functions.iter().collect();
    ordered.sort_by_key(|(_, function)| function.line);
    for (name, function) in ordered {
        let mut ops = Vec::new();
        let mut values = HashMap::new();
        for param in &function.params {
            let index = push_op(&mut ops, Op::Const(0), function.line)?;
            values.insert(param.clone(), index);
        }
        let mut call = vec!["call", name.as_str()];
        call.extend(function.params.iter().map(String::as_str));
        emit_expression(
            &call,
            &values,
            &HashMap::new(),
            functions,
            &mut ops,
            &mut Vec::new(),
            function.line,
        )?;
    }
    Ok(())
}

fn emit_expression(
    words: &[&str],
    values: &HashMap<String, u32>,
    locals: &HashMap<String, u32>,
    functions: &HashMap<String, Function>,
    ops: &mut Vec<Op>,
    call_stack: &mut Vec<String>,
    line: usize,
) -> Result<u32, SurfaceError> {
    if let ["call", function_name, arguments @ ..] = words {
        let function = functions
            .get(*function_name)
            .ok_or_else(|| SurfaceError::at(line, SurfaceErrorKind::UnknownFunction))?;
        if arguments.len() != function.params.len() {
            return Err(SurfaceError::at(line, SurfaceErrorKind::ArityMismatch));
        }
        if call_stack.iter().any(|active| active == function_name) {
            return Err(SurfaceError::at(line, SurfaceErrorKind::RecursiveCall));
        }
        let mut function_values = HashMap::new();
        for (param, argument) in function.params.iter().zip(arguments) {
            function_values.insert(param.clone(), value(values, argument, line)?);
        }
        call_stack.push((*function_name).to_string());
        for body_line in &function.body {
            let body_words: Vec<_> = body_line.words.iter().map(String::as_str).collect();
            let ["let", result, "=", rest @ ..] = body_words.as_slice() else {
                call_stack.pop();
                return Err(SurfaceError::at(body_line.number, SurfaceErrorKind::Syntax));
            };
            if function_values.contains_key(*result) {
                call_stack.pop();
                return Err(SurfaceError::at(
                    body_line.number,
                    SurfaceErrorKind::DuplicateName,
                ));
            }
            let index = emit_expression(
                rest,
                &function_values,
                &HashMap::new(),
                functions,
                ops,
                call_stack,
                body_line.number,
            )?;
            function_values.insert((*result).to_string(), index);
        }
        call_stack.pop();
        return value(&function_values, &function.result, line);
    }

    let op = parse_expression(words, values, locals, line)?;
    push_op(ops, op, line)
}

fn push_op(ops: &mut Vec<Op>, op: Op, line: usize) -> Result<u32, SurfaceError> {
    if ops.len() as u64 >= MAX_STEPS {
        return Err(SurfaceError::at(
            line,
            SurfaceErrorKind::Body(BodyError::Unbounded),
        ));
    }
    let index = ops.len() as u32;
    ops.push(op);
    Ok(index)
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
        ["fconst", constant] => Op::FConst(parse_f32_bits(constant, line)?),
        ["get", local] => Op::Get(local_id(locals, local, line)?),
        ["add", a, b] => binary(values, a, b, line, Op::Add)?,
        ["sub", a, b] => binary(values, a, b, line, Op::Sub)?,
        ["mul", a, b] => binary(values, a, b, line, Op::Mul)?,
        ["fadd", a, b] => binary(values, a, b, line, Op::FAdd)?,
        ["fmul", a, b] => binary(values, a, b, line, Op::FMul)?,
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
        "f32" => Ok(FieldWidth::F32),
        _ => Err(SurfaceError::at(line, SurfaceErrorKind::InvalidWidth)),
    }
}

fn parse_f32_bits(text: &str, line: usize) -> Result<u32, SurfaceError> {
    if let Some(hex) = text.strip_prefix("0x") {
        u32::from_str_radix(hex, 16)
            .map_err(|_| SurfaceError::at(line, SurfaceErrorKind::InvalidFloat))
    } else {
        text.parse::<f32>()
            .map(f32::to_bits)
            .map_err(|_| SurfaceError::at(line, SurfaceErrorKind::InvalidFloat))
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
