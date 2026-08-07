//! Native CPU JIT backend for evaluator bodies.
//!
//! The scalar backend remains I20's definition. This backend lowers the
//! full validated evaluator language to Cranelift machine code and is checked
//! byte-for-byte against that definition. Pointwise arithmetic, gathers,
//! auxiliary arrays, structured repeats, and divergent early exits all cross
//! the same backend boundary; no instruction falls back to interpretation.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, Block, InstBuilder, MemFlags, UserFuncName, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

use super::batch::{AuxArray, BackendError, BackendKind, BatchBackend};
use crate::compiler::body::{EvaluatorProgram, FieldWidth, Op};
use crate::scheduler::device::{
    reference_lane_conflicts, DeviceEpochBackend, DeviceEvaluation, DeviceEvaluatorLane,
    DeviceEvaluatorResult, DeviceLaneAccess, DeviceLaneConflict, LaneConflictValidator,
    LaneValidationError,
};

type NativeEvaluator =
    unsafe extern "C" fn(*const u8, *mut u8, u32, u32, *const u8, u32, u32, u32, u32);

#[derive(Clone, Copy)]
struct CompiledEvaluator {
    function: NativeEvaluator,
    stride: u32,
    aux_stride: u32,
}

#[derive(Clone, Copy)]
struct ArrayValues {
    index: Value,
    input: Value,
    input_element: Value,
    count: Value,
    stride: Value,
    aux: Value,
    aux_count: Value,
    aux_stride: Value,
}

pub struct NativeCpuBackend {
    // Executable allocations belong to the module. It must outlive every
    // function pointer stored below.
    module: JITModule,
    evaluators: HashMap<u32, CompiledEvaluator>,
    threads: usize,
}

/// Cranelift-backed continuation evaluator plus the reference journal
/// validator. This makes the native lowering available at the same canonical
/// epoch boundary as Metal rather than only through batch collectives.
pub struct NativeEpochBackend {
    evaluator: NativeCpuBackend,
    handlers: HashMap<u32, (u32, u32, bool)>,
}

impl NativeEpochBackend {
    pub fn new() -> Result<Self, BackendError> {
        Ok(Self {
            evaluator: NativeCpuBackend::new()?,
            handlers: HashMap::new(),
        })
    }

    pub fn install_frame_evaluator(
        &mut self,
        run_class: u32,
        program: &EvaluatorProgram,
    ) -> Result<(), BackendError> {
        if run_class < 1024 || program.binds_aux() || program.stride() == 0 {
            return Err(BackendError::InvalidInput);
        }
        self.evaluator.install(program)?;
        let batchable = !program
            .ops()
            .iter()
            .any(|op| matches!(op, Op::Index | Op::Gather(_, _) | Op::GatherAux(_, _)));
        self.handlers
            .insert(run_class, (program.id(), program.stride(), batchable));
        Ok(())
    }
}

impl LaneConflictValidator for NativeEpochBackend {
    fn validate_lane_journals(
        &mut self,
        accesses: &[DeviceLaneAccess],
        lane_count: u32,
    ) -> Result<Vec<DeviceLaneConflict>, LaneValidationError> {
        if accesses.iter().any(|access| access.lane >= lane_count) {
            return Err(LaneValidationError::InvalidInput);
        }
        Ok(reference_lane_conflicts(accesses, lane_count))
    }
}

impl DeviceEpochBackend for NativeEpochBackend {
    fn evaluate_lanes(
        &mut self,
        lanes: &[DeviceEvaluatorLane],
        frames: &[u8],
    ) -> Result<DeviceEvaluation, LaneValidationError> {
        if lanes.is_empty() {
            return Ok(DeviceEvaluation::default());
        }
        if lanes.len() > u32::MAX as usize {
            return Err(LaneValidationError::InvalidInput);
        }
        let run_class = lanes[0].run_class;
        let (program, stride, batchable) = self
            .handlers
            .get(&run_class)
            .copied()
            .ok_or(LaneValidationError::InvalidInput)?;
        let packed = lanes.iter().enumerate().all(|(index, lane)| {
            lane.run_class == run_class
                && lane.frame_len == stride
                && lane.frame_offset as usize == index * stride as usize
        });
        if !packed || lanes.len().checked_mul(stride as usize) != Some(frames.len()) {
            return Err(LaneValidationError::InvalidInput);
        }
        let output = if batchable {
            self.evaluator
                .evaluate(program, frames, lanes.len() as u32, stride)
                .map_err(|_| LaneValidationError::ExecutionFailed)?
        } else {
            let mut output = Vec::with_capacity(frames.len());
            for lane in lanes {
                let start = lane.frame_offset as usize;
                let end = start
                    .checked_add(stride as usize)
                    .ok_or(LaneValidationError::InvalidInput)?;
                let frame = frames
                    .get(start..end)
                    .ok_or(LaneValidationError::InvalidInput)?;
                output.extend_from_slice(
                    &self
                        .evaluator
                        .evaluate(program, frame, 1, stride)
                        .map_err(|_| LaneValidationError::ExecutionFailed)?,
                );
            }
            output
        };
        let results = lanes
            .iter()
            .map(|lane| DeviceEvaluatorResult {
                lane: lane.lane,
                status: 1,
                step_kind: 1,
                consumed_steps: 1,
                frame_offset: lane.frame_offset,
                frame_len: lane.frame_len,
                ..DeviceEvaluatorResult::default()
            })
            .collect();
        Ok(DeviceEvaluation {
            results,
            frames: output,
        })
    }
}

impl NativeCpuBackend {
    pub fn new() -> Result<Self, BackendError> {
        let mut flags = settings::builder();
        flags
            .set("opt_level", "speed")
            .map_err(|_| BackendError::Unavailable)?;
        flags
            .set("use_colocated_libcalls", "false")
            .map_err(|_| BackendError::Unavailable)?;
        flags
            .set("is_pic", "false")
            .map_err(|_| BackendError::Unavailable)?;
        let isa = cranelift_native::builder()
            .map_err(|_| BackendError::Unavailable)?
            .finish(settings::Flags::new(flags))
            .map_err(|_| BackendError::Unavailable)?;
        let module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        Ok(Self {
            module,
            evaluators: HashMap::new(),
            threads: 1,
        })
    }

    pub fn with(programs: &[&EvaluatorProgram]) -> Result<Self, BackendError> {
        let mut backend = Self::new()?;
        for program in programs {
            backend.install(program)?;
        }
        Ok(backend)
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Change execution parallelism without recompiling installed bodies.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    fn compile(&mut self, program: &EvaluatorProgram) -> Result<NativeEvaluator, BackendError> {
        let pointer = self.module.target_config().pointer_type();
        let mut signature = self.module.make_signature();
        signature.params.extend([
            AbiParam::new(pointer),
            AbiParam::new(pointer),
            AbiParam::new(types::I32),
            AbiParam::new(types::I32),
            AbiParam::new(pointer),
            AbiParam::new(types::I32),
            AbiParam::new(types::I32),
            AbiParam::new(types::I32),
            AbiParam::new(types::I32),
        ]);
        let name = format!("soma_native_{}_{}", program.id(), self.evaluators.len());
        let function_id = self
            .module
            .declare_function(&name, Linkage::Local, &signature)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        let mut context = self.module.make_context();
        context.func.signature = signature;
        context.func.name = UserFuncName::user(0, function_id.as_u32());
        let mut frontend = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
            let entry = builder.create_block();
            let header = builder.create_block();
            let body = builder.create_block();
            let done = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            let params = builder.block_params(entry).to_vec();
            let input = params[0];
            let output = params[1];
            let count = params[2];
            let stride = params[3];
            let aux = params[4];
            let aux_count = params[5];
            let aux_stride = params[6];
            let start = params[7];
            let end = params[8];
            let index = builder.declare_var(types::I32);
            builder.def_var(index, start);
            builder.ins().jump(header, &[]);

            builder.switch_to_block(header);
            let at = builder.use_var(index);
            let more = builder.ins().icmp(IntCC::UnsignedLessThan, at, end);
            builder.ins().brif(more, body, &[], done, &[]);

            builder.switch_to_block(body);
            let at = builder.use_var(index);
            let at_pointer = builder.ins().uextend(pointer, at);
            let stride_pointer = builder.ins().uextend(pointer, stride);
            let element_offset = builder.ins().imul(at_pointer, stride_pointer);
            let input_element = builder.ins().iadd(input, element_offset);
            let output_element = builder.ins().iadd(output, element_offset);
            let arrays = ArrayValues {
                index: at,
                input,
                input_element,
                count,
                stride,
                aux,
                aux_count,
                aux_stride,
            };
            let mut locals = Vec::with_capacity(program.locals() as usize);
            for _ in 0..program.locals() {
                let local = builder.declare_var(types::I64);
                let zero = builder.ins().iconst(types::I64, 0);
                builder.def_var(local, zero);
                locals.push(local);
            }
            let zero = builder.ins().iconst(types::I64, 0);
            let mut values = vec![zero; program.ops().len()];
            lower_ops(
                &mut builder,
                program,
                arrays,
                0,
                program.ops().len(),
                &mut values,
                &locals,
                None,
            )?;
            for store in program.stores() {
                let offset = program
                    .layout()
                    .offset(store.field)
                    .ok_or(BackendError::UnsupportedEvaluator)?;
                let width = program
                    .layout()
                    .width(store.field)
                    .ok_or(BackendError::UnsupportedEvaluator)?;
                store_field(
                    &mut builder,
                    output_element,
                    offset,
                    width,
                    values[store.value as usize],
                );
            }
            let next = builder.ins().iadd_imm(at, 1);
            builder.def_var(index, next);
            builder.ins().jump(header, &[]);

            builder.switch_to_block(done);
            builder.ins().return_(&[]);
            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module
            .define_function(function_id, &mut context)
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        self.module.clear_context(&mut context);
        self.module
            .finalize_definitions()
            .map_err(|_| BackendError::UnsupportedEvaluator)?;
        let code = self.module.get_finalized_function(function_id);
        // Safety: `signature` exactly matches `NativeEvaluator`, and the JIT
        // module remains owned by this backend for the pointer's lifetime.
        Ok(unsafe { std::mem::transmute::<*const u8, NativeEvaluator>(code) })
    }

    fn run(
        &self,
        compiled: CompiledEvaluator,
        inputs: &[u8],
        element_count: u32,
        aux: AuxArray<'_>,
    ) -> Vec<u8> {
        let mut outputs = inputs.to_vec();
        if element_count == 0 {
            return outputs;
        }
        let workers = self.threads.min(element_count as usize).max(1);
        let per_worker = (element_count as usize).div_ceil(workers);
        let input_address = inputs.as_ptr() as usize;
        let output_address = outputs.as_mut_ptr() as usize;
        let aux_address = aux.bytes.as_ptr() as usize;
        std::thread::scope(|scope| {
            for worker in 0..workers {
                let start = worker * per_worker;
                let end = ((worker + 1) * per_worker).min(element_count as usize);
                if start == end {
                    continue;
                }
                scope.spawn(move || {
                    // Safety: input covers `element_count * stride`; output has
                    // the same length; worker ranges are disjoint; generated
                    // code writes only its own elements; and the function
                    // pointer remains live for the scoped call.
                    unsafe {
                        (compiled.function)(
                            input_address as *const u8,
                            output_address as *mut u8,
                            element_count,
                            compiled.stride,
                            aux_address as *const u8,
                            aux.element_count,
                            aux.element_stride,
                            start as u32,
                            end as u32,
                        );
                    }
                });
            }
        });
        outputs
    }
}

impl BatchBackend for NativeCpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn install(&mut self, program: &EvaluatorProgram) -> Result<(), BackendError> {
        let function = self.compile(program)?;
        self.evaluators.insert(
            program.id(),
            CompiledEvaluator {
                function,
                stride: program.stride(),
                aux_stride: program.aux_stride(),
            },
        );
        Ok(())
    }

    fn evaluate(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        self.evaluate_with_aux(
            evaluator_id,
            inputs,
            element_count,
            element_stride,
            AuxArray::NONE,
        )
    }

    fn evaluate_with_aux(
        &mut self,
        evaluator_id: u32,
        inputs: &[u8],
        element_count: u32,
        element_stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Vec<u8>, BackendError> {
        let compiled = *self
            .evaluators
            .get(&evaluator_id)
            .ok_or(BackendError::UnsupportedEvaluator)?;
        if compiled.stride != element_stride || compiled.aux_stride != aux.element_stride {
            return Err(BackendError::InvalidInput);
        }
        let required = (element_count as usize)
            .checked_mul(element_stride as usize)
            .ok_or(BackendError::InvalidInput)?;
        if inputs.len() < required {
            return Err(BackendError::InvalidInput);
        }
        let aux_required = (aux.element_count as usize)
            .checked_mul(aux.element_stride as usize)
            .ok_or(BackendError::InvalidInput)?;
        if aux.bytes.len() < aux_required {
            return Err(BackendError::InvalidInput);
        }
        Ok(self.run(
            compiled,
            &inputs[..required],
            element_count,
            AuxArray::new(
                &aux.bytes[..aux_required],
                aux.element_count,
                aux.element_stride,
            ),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_ops(
    builder: &mut FunctionBuilder<'_>,
    program: &EvaluatorProgram,
    arrays: ArrayValues,
    start: usize,
    end: usize,
    values: &mut [Value],
    locals: &[cranelift_frontend::Variable],
    break_target: Option<Block>,
) -> Result<(), BackendError> {
    let mut pc = start;
    while pc < end {
        match program.ops()[pc] {
            Op::Repeat(trips) => {
                let close =
                    matching_end(program.ops(), pc).ok_or(BackendError::UnsupportedEvaluator)?;
                if close >= end {
                    return Err(BackendError::UnsupportedEvaluator);
                }
                let zero = builder.ins().iconst(types::I64, 0);
                values[pc] = zero;
                let counter = builder.declare_var(types::I32);
                let initial = builder.ins().iconst(types::I32, i64::from(trips));
                builder.def_var(counter, initial);
                let header = builder.create_block();
                let body = builder.create_block();
                let exit = builder.create_block();
                builder.ins().jump(header, &[]);

                builder.switch_to_block(header);
                let remaining = builder.use_var(counter);
                let more = builder.ins().icmp_imm(IntCC::NotEqual, remaining, 0);
                builder.ins().brif(more, body, &[], exit, &[]);

                builder.switch_to_block(body);
                lower_ops(
                    builder,
                    program,
                    arrays,
                    pc + 1,
                    close,
                    values,
                    locals,
                    Some(exit),
                )?;
                let remaining = builder.use_var(counter);
                let next = builder.ins().iadd_imm(remaining, -1);
                builder.def_var(counter, next);
                builder.ins().jump(header, &[]);

                builder.switch_to_block(exit);
                let zero = builder.ins().iconst(types::I64, 0);
                values[close] = zero;
                pc = close + 1;
            }
            Op::BreakIf(condition) => {
                let target = break_target.ok_or(BackendError::UnsupportedEvaluator)?;
                let zero = builder.ins().iconst(types::I64, 0);
                values[pc] = zero;
                let leave = builder
                    .ins()
                    .icmp_imm(IntCC::NotEqual, values[condition as usize], 0);
                let keep_going = builder.create_block();
                builder.ins().brif(leave, target, &[], keep_going, &[]);
                builder.switch_to_block(keep_going);
                pc += 1;
            }
            Op::EndRepeat => return Err(BackendError::UnsupportedEvaluator),
            op => {
                values[pc] = lower_op(builder, program, op, arrays, values, locals)?;
                pc += 1;
            }
        }
    }
    Ok(())
}

fn matching_end(ops: &[Op], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, op) in ops.iter().enumerate().skip(start) {
        match op {
            Op::Repeat(_) => depth += 1,
            Op::EndRepeat => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn lower_op(
    builder: &mut FunctionBuilder<'_>,
    program: &EvaluatorProgram,
    op: Op,
    arrays: ArrayValues,
    values: &[Value],
    locals: &[cranelift_frontend::Variable],
) -> Result<Value, BackendError> {
    let value = match op {
        Op::Load(field) => {
            let offset = program
                .layout()
                .offset(field)
                .ok_or(BackendError::UnsupportedEvaluator)?;
            let width = program
                .layout()
                .width(field)
                .ok_or(BackendError::UnsupportedEvaluator)?;
            load_field(builder, arrays.input_element, offset, width)
        }
        Op::Index => builder.ins().uextend(types::I64, arrays.index),
        Op::Gather(at, field) => {
            let offset = program
                .layout()
                .offset(field)
                .ok_or(BackendError::UnsupportedEvaluator)?;
            let width = program
                .layout()
                .width(field)
                .ok_or(BackendError::UnsupportedEvaluator)?;
            load_clamped(
                builder,
                arrays.input,
                arrays.count,
                arrays.stride,
                values[at as usize],
                offset,
                width,
            )
        }
        Op::GatherAux(at, field) => {
            let layout = program
                .aux_layout()
                .ok_or(BackendError::UnsupportedEvaluator)?;
            let offset = layout
                .offset(field)
                .ok_or(BackendError::UnsupportedEvaluator)?;
            let width = layout
                .width(field)
                .ok_or(BackendError::UnsupportedEvaluator)?;
            load_clamped_or_zero(
                builder,
                arrays.aux,
                arrays.aux_count,
                arrays.aux_stride,
                values[at as usize],
                offset,
                width,
            )
        }
        Op::Const(value) => builder.ins().iconst(types::I64, value as i64),
        Op::FConst(bits) => {
            let raw = builder.ins().iconst(types::I64, i64::from(bits));
            canonical_f32_value(builder, raw)
        }
        Op::Add(a, b) => builder.ins().iadd(values[a as usize], values[b as usize]),
        Op::Sub(a, b) => builder.ins().isub(values[a as usize], values[b as usize]),
        Op::Mul(a, b) => builder.ins().imul(values[a as usize], values[b as usize]),
        Op::FAdd(a, b) | Op::FMul(a, b) => {
            let left_bits = canonical_f32_value(builder, values[a as usize]);
            let right_bits = canonical_f32_value(builder, values[b as usize]);
            let left32 = builder.ins().ireduce(types::I32, left_bits);
            let right32 = builder.ins().ireduce(types::I32, right_bits);
            let left = builder.ins().bitcast(types::F32, MemFlags::new(), left32);
            let right = builder.ins().bitcast(types::F32, MemFlags::new(), right32);
            let result = if matches!(op, Op::FAdd(_, _)) {
                builder.ins().fadd(left, right)
            } else {
                builder.ins().fmul(left, right)
            };
            let raw32 = builder.ins().bitcast(types::I32, MemFlags::new(), result);
            let raw = builder.ins().uextend(types::I64, raw32);
            canonical_f32_value(builder, raw)
        }
        Op::And(a, b) => builder.ins().band(values[a as usize], values[b as usize]),
        Op::Or(a, b) => builder.ins().bor(values[a as usize], values[b as usize]),
        Op::Xor(a, b) => builder.ins().bxor(values[a as usize], values[b as usize]),
        Op::Shl(a, b) | Op::Shr(a, b) => {
            let mask = builder.ins().iconst(types::I64, 63);
            let amount = builder.ins().band(values[b as usize], mask);
            if matches!(op, Op::Shl(_, _)) {
                builder.ins().ishl(values[a as usize], amount)
            } else {
                builder.ins().ushr(values[a as usize], amount)
            }
        }
        Op::CmpEq(a, b) | Op::CmpLt(a, b) => {
            let condition = if matches!(op, Op::CmpEq(_, _)) {
                IntCC::Equal
            } else {
                IntCC::UnsignedLessThan
            };
            let compared = builder
                .ins()
                .icmp(condition, values[a as usize], values[b as usize]);
            let one = builder.ins().iconst(types::I64, 1);
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().select(compared, one, zero)
        }
        Op::Select(condition, yes, no) => {
            let condition = builder
                .ins()
                .icmp_imm(IntCC::NotEqual, values[condition as usize], 0);
            builder
                .ins()
                .select(condition, values[yes as usize], values[no as usize])
        }
        Op::Get(local) => builder.use_var(locals[local as usize]),
        Op::Set(local, value) => {
            let value = values[value as usize];
            builder.def_var(locals[local as usize], value);
            value
        }
        Op::Repeat(_) | Op::EndRepeat | Op::BreakIf(_) => {
            return Err(BackendError::UnsupportedEvaluator)
        }
    };
    Ok(value)
}

fn canonical_f32_value(builder: &mut FunctionBuilder<'_>, raw: Value) -> Value {
    let magnitude_mask = builder.ins().iconst(types::I64, 0x7fff_ffff);
    let magnitude = builder.ins().band(raw, magnitude_mask);
    let infinity = builder.ins().iconst(types::I64, 0x7f80_0000);
    let is_nan = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThan, magnitude, infinity);
    let is_zero = builder
        .ins()
        .icmp_imm(IntCC::UnsignedLessThan, magnitude, 0x0080_0000);
    let zero = builder.ins().iconst(types::I64, 0);
    let canonical_nan = builder.ins().iconst(types::I64, 0x7fc0_0000);
    let zeroed = builder.ins().select(is_zero, zero, raw);
    builder.ins().select(is_nan, canonical_nan, zeroed)
}

fn load_field(
    builder: &mut FunctionBuilder<'_>,
    element: Value,
    offset: u32,
    width: FieldWidth,
) -> Value {
    let ty = field_type(width);
    let loaded = builder
        .ins()
        .load(ty, MemFlags::new(), element, offset as i32);
    if ty == types::I64 {
        loaded
    } else {
        builder.ins().uextend(types::I64, loaded)
    }
}

fn store_field(
    builder: &mut FunctionBuilder<'_>,
    element: Value,
    offset: u32,
    width: FieldWidth,
    value: Value,
) {
    let ty = field_type(width);
    let value = if width == FieldWidth::F32 {
        canonical_f32_value(builder, value)
    } else {
        value
    };
    let value = if ty == types::I64 {
        value
    } else {
        builder.ins().ireduce(ty, value)
    };
    builder
        .ins()
        .store(MemFlags::new(), value, element, offset as i32);
}

fn field_type(width: FieldWidth) -> cranelift_codegen::ir::Type {
    match width {
        FieldWidth::U8 => types::I8,
        FieldWidth::U16 => types::I16,
        FieldWidth::U32 | FieldWidth::F32 => types::I32,
        FieldWidth::U64 => types::I64,
    }
}

fn load_clamped(
    builder: &mut FunctionBuilder<'_>,
    base: Value,
    count: Value,
    stride: Value,
    requested: Value,
    field_offset: u32,
    width: FieldWidth,
) -> Value {
    let count64 = builder.ins().uextend(types::I64, count);
    let last = builder.ins().iadd_imm(count64, -1);
    let past_end = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, requested, count64);
    let at = builder.ins().select(past_end, last, requested);
    let stride64 = builder.ins().uextend(types::I64, stride);
    let byte_offset = builder.ins().imul(at, stride64);
    let element = builder.ins().iadd(base, byte_offset);
    load_field(builder, element, field_offset, width)
}

fn load_clamped_or_zero(
    builder: &mut FunctionBuilder<'_>,
    base: Value,
    count: Value,
    stride: Value,
    requested: Value,
    field_offset: u32,
    width: FieldWidth,
) -> Value {
    let zero_block = builder.create_block();
    let load_block = builder.create_block();
    let merge = builder.create_block();
    builder.append_block_param(merge, types::I64);
    let empty = builder.ins().icmp_imm(IntCC::Equal, count, 0);
    builder.ins().brif(empty, zero_block, &[], load_block, &[]);

    builder.switch_to_block(zero_block);
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().jump(merge, &[zero.into()]);

    builder.switch_to_block(load_block);
    let loaded = load_clamped(builder, base, count, stride, requested, field_offset, width);
    builder.ins().jump(merge, &[loaded.into()]);

    builder.seal_block(zero_block);
    builder.seal_block(load_block);
    builder.switch_to_block(merge);
    builder.seal_block(merge);
    builder.block_params(merge)[0]
}
