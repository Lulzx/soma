//! Wall-clock measurement of the physical batch backends.
//!
//! Every other experiment in this module measures a model: occupancy over
//! simulated ticks, advantage at a matched wait budget, cohort fill under an
//! arrival process. None of them start a clock. That was fine while the
//! question was whether SOMA's schedule is better than a bulk one, and it is
//! useless for the question of whether `MetalBatchBackend` is fast, which is a
//! question about an actual machine.
//!
//! So this file measures three things and models nothing:
//!
//! - `time_evaluate` times `BatchBackend::evaluate` alone: the bytes are
//!   already in hand and the result is a `Vec`. This is the backend's own cost.
//! - `time_published_path` times `execute_with_spill`, which additionally
//!   copies the frozen input out of the kernel, freezes the output, and
//!   completes the collective. The difference between the two is what SOMA's
//!   publication path costs on top of the backend.
//! - `PlacementModel` fits `fixed + n * per_element` to a size sweep, so
//!   "the accelerator is profitable above n elements" becomes a measured
//!   number rather than the caller-supplied constant `minimum_accelerator_batch`
//!   currently is.
//!
//! Timings are medians of repeated runs after a warmup. Median rather than
//! mean because a single scheduler preemption or a first-touch page fault
//! moves a mean and does not move a median, and neither is the quantity being
//! asked about.

use std::time::{Duration, Instant};

use crate::abi::{ObjectKind, ProcessMode, Ref64};
use crate::compiler::body::{ElementLayout, EvaluatorProgram, FieldWidth, Op, Store};
use crate::executives::batch::{
    execute_epoch_with_spill, execute_with_spill, BackendError, BatchBackend, PlacementStats,
};
use crate::kernel::ownership::freeze;
use crate::kernel::{Kernel, SYSTEM_PRINCIPAL};

/// A body with a chosen amount of arithmetic per element.
///
/// The example bodies in `compiler::examples` are all two to eight
/// instructions, which is the right size for checking a lowering and the wrong
/// size for finding out what a GPU does with SOMA work: at that op count the
/// measurement is entirely submission overhead and memory traffic, and every
/// evaluator looks identical. `alu_ops` is the knob that separates a
/// memory-bound body from a compute-bound one.
///
/// The generated chain is deliberately not a single dependent sequence. Four
/// independent accumulators are advanced round-robin, so the body has
/// instruction-level parallelism to expose; a strictly dependent chain would
/// measure ALU latency, which is not what an element-wise evaluator is
/// bottlenecked on. The accumulators are folded together at the end so that
/// none of them is dead code that either lowering may delete.
pub fn synthetic_program(id: u32, field_count: u32, alu_ops: u32) -> EvaluatorProgram {
    let field_count = field_count.max(1);
    let layout = ElementLayout::new(vec![FieldWidth::U32; field_count as usize]);

    let mut ops: Vec<Op> = (0..field_count).map(Op::Load).collect();
    // A constant with no small factors, so a `mul` chain does not collapse to
    // zero or to a shift the compiler can strength-reduce away.
    let konst = ops.len() as u32;
    ops.push(Op::Const(0x9E37_79B9));

    let mut accumulators = [0u32; 4];
    for (lane, slot) in accumulators.iter_mut().enumerate() {
        *slot = (lane as u32).min(konst);
    }
    for step in 0..alu_ops {
        let lane = (step % 4) as usize;
        let source = accumulators[lane];
        let op = match step % 3 {
            0 => Op::Mul(source, konst),
            1 => Op::Xor(source, konst),
            _ => Op::Add(source, konst),
        };
        ops.push(op);
        accumulators[lane] = ops.len() as u32 - 1;
    }

    let mut folded = accumulators[0];
    for lane in accumulators.iter().skip(1) {
        ops.push(Op::Add(folded, *lane));
        folded = ops.len() as u32 - 1;
    }

    let stores = vec![Store {
        field: 0,
        value: folded,
    }];
    EvaluatorProgram::new(
        id,
        format!("synthetic_{field_count}f_{alu_ops}alu"),
        layout,
        ops,
        stores,
    )
    .expect("synthetic body is well-formed by construction")
}

/// Deterministic input bytes. Not random: a benchmark that changes its input
/// between backends is comparing two different questions, and a benchmark that
/// changes it between runs cannot be diffed against yesterday's.
pub fn synthetic_inputs(element_count: u32, stride: u32) -> Vec<u8> {
    let len = element_count as usize * stride as usize;
    let mut bytes = Vec::with_capacity(len);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.push((state >> 33) as u8);
    }
    bytes
}

/// The result of one timed cell: several samples of the same work.
#[derive(Clone, Debug)]
pub struct Timing {
    pub element_count: u32,
    pub samples: Vec<Duration>,
}

impl Timing {
    pub fn median(&self) -> Duration {
        let mut sorted = self.samples.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    pub fn min(&self) -> Duration {
        self.samples.iter().copied().min().unwrap_or_default()
    }

    pub fn nanos_per_element(&self) -> f64 {
        self.median().as_nanos() as f64 / self.element_count.max(1) as f64
    }

    pub fn elements_per_second(&self) -> f64 {
        let seconds = self.median().as_secs_f64();
        if seconds <= 0.0 {
            return f64::INFINITY;
        }
        self.element_count as f64 / seconds
    }

    /// Spread between the fastest and the median sample, as a fraction. A
    /// large value means the cell was disturbed and its median should not be
    /// read as a property of the machine.
    pub fn jitter(&self) -> f64 {
        let min = self.min().as_secs_f64();
        if min <= 0.0 {
            return 0.0;
        }
        (self.median().as_secs_f64() - min) / min
    }
}

/// How many samples to take at a given size, so a sweep that reaches millions
/// of elements does not spend minutes on its tail while still averaging enough
/// runs at the small sizes where per-call overhead is the whole measurement.
fn repetitions_for(element_count: u32) -> u32 {
    match element_count {
        0..=1_024 => 201,
        1_025..=65_536 => 51,
        65_537..=1_048_576 => 21,
        _ => 7,
    }
}

/// Time `BatchBackend::evaluate` alone, with the input bytes already
/// materialized. Returns `Err` if the backend refuses the work — an
/// unavailable accelerator is a result to report, not a panic.
pub fn time_evaluate(
    backend: &mut dyn BatchBackend,
    program: &EvaluatorProgram,
    element_count: u32,
) -> Result<Timing, BackendError> {
    let stride = program.stride();
    let inputs = synthetic_inputs(element_count, stride);
    let repetitions = repetitions_for(element_count);

    // Warmup: the first call on a Metal backend pays lazy pipeline warmup and
    // first-touch faults on freshly allocated shared buffers, neither of which
    // is a property of steady-state execution.
    for _ in 0..3 {
        backend.evaluate(program.id(), &inputs, element_count, stride)?;
    }

    let mut samples = Vec::with_capacity(repetitions as usize);
    for _ in 0..repetitions {
        let start = Instant::now();
        let outputs = backend.evaluate(program.id(), &inputs, element_count, stride)?;
        samples.push(start.elapsed());
        // Consume the result so a future optimizer cannot decide the call is
        // dead, and so the deallocation is inside neither this sample nor a
        // later one in a way that depends on allocator state.
        std::hint::black_box(outputs);
    }
    Ok(Timing {
        element_count,
        samples,
    })
}

/// Time the full semantic path: read the frozen input out of the kernel, run a
/// backend, freeze the output, complete the collective.
///
/// This is the number that matters for SOMA, and it is strictly larger than
/// `time_evaluate` by the two copies `execute_with_spill` performs.
pub fn time_published_path(
    program: &EvaluatorProgram,
    element_count: u32,
    minimum_accelerator_batch: u32,
    accelerator: &mut dyn BatchBackend,
    cpu: &mut dyn BatchBackend,
) -> Result<Timing, BackendError> {
    let stride = program.stride();
    let repetitions = repetitions_for(element_count);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = synthetic_inputs(element_count, stride);
    let inputs = kernel.create_object(owner, ObjectKind::FrozenArray, bytes);
    freeze(&mut kernel, owner, inputs).expect("fresh object is freezable");

    let once = |kernel: &mut Kernel,
                accelerator: &mut dyn BatchBackend,
                cpu: &mut dyn BatchBackend|
     -> Result<Duration, BackendError> {
        let (collective, _) = kernel
            .create_batch_evaluate_for(owner, program.id(), inputs, element_count, stride)
            .map_err(BackendError::from)?;
        let start = Instant::now();
        let output = execute_with_spill(
            kernel,
            owner,
            collective,
            minimum_accelerator_batch,
            accelerator,
            cpu,
            &mut PlacementStats::default(),
        )?;
        let elapsed = start.elapsed();
        std::hint::black_box(output);
        Ok(elapsed)
    };

    for _ in 0..3 {
        once(&mut kernel, accelerator, cpu)?;
    }
    let mut samples = Vec::with_capacity(repetitions as usize);
    for _ in 0..repetitions {
        samples.push(once(&mut kernel, accelerator, cpu)?);
    }
    Ok(Timing {
        element_count,
        samples,
    })
}

/// Time an epoch of `cohorts` collectives, submitted either together or one
/// at a time.
///
/// This is the end-to-end version of what `examples/metal_overhead` measures
/// at the Metal API: it goes through the kernel, so it pays authorization,
/// publication, and freezing for every cohort either way, and the only
/// difference between the two arms is whether the backend was given the
/// requests together.
pub fn time_epoch(
    program: &EvaluatorProgram,
    cohorts: u32,
    elements_per_cohort: u32,
    minimum_accelerator_batch: u32,
    accelerator: &mut dyn BatchBackend,
    cpu: &mut dyn BatchBackend,
    batched: bool,
) -> Result<Timing, BackendError> {
    let stride = program.stride();
    let repetitions = repetitions_for(cohorts * elements_per_cohort).min(30);

    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let bytes = synthetic_inputs(elements_per_cohort, stride);
    let inputs: Vec<Ref64> = (0..cohorts)
        .map(|_| {
            let object = kernel.create_object(owner, ObjectKind::FrozenArray, bytes.clone());
            freeze(&mut kernel, owner, object).expect("fresh object is freezable");
            object
        })
        .collect();

    let once = |kernel: &mut Kernel,
                    accelerator: &mut dyn BatchBackend,
                    cpu: &mut dyn BatchBackend|
     -> Result<Duration, BackendError> {
        let collectives: Vec<Ref64> = inputs
            .iter()
            .map(|input| {
                kernel
                    .create_batch_evaluate_for(
                        owner,
                        program.id(),
                        *input,
                        elements_per_cohort,
                        stride,
                    )
                    .map(|(collective, _)| collective)
                    .map_err(BackendError::from)
            })
            .collect::<Result<_, _>>()?;

        let mut stats = PlacementStats::default();
        let start = Instant::now();
        if batched {
            let outputs = execute_epoch_with_spill(
                kernel,
                owner,
                &collectives,
                minimum_accelerator_batch,
                accelerator,
                cpu,
                &mut stats,
            )?;
            std::hint::black_box(outputs);
        } else {
            for collective in &collectives {
                let output = execute_with_spill(
                    kernel,
                    owner,
                    *collective,
                    minimum_accelerator_batch,
                    accelerator,
                    cpu,
                    &mut stats,
                )?;
                std::hint::black_box(output);
            }
        }
        Ok(start.elapsed())
    };

    for _ in 0..2 {
        once(&mut kernel, accelerator, cpu)?;
    }
    let mut samples = Vec::with_capacity(repetitions as usize);
    for _ in 0..repetitions {
        samples.push(once(&mut kernel, accelerator, cpu)?);
    }
    Ok(Timing {
        element_count: cohorts * elements_per_cohort,
        samples,
    })
}

/// `time(n) = fixed + n * per_element`, fitted by least squares over a size
/// sweep.
///
/// This is the shape a placement decision needs. `execute_with_spill` today
/// compares `count` against a `minimum_accelerator_batch` the caller invented;
/// with two of these it can compare two predicted times instead.
#[derive(Clone, Copy, Debug)]
pub struct PlacementModel {
    pub fixed_nanos: f64,
    pub nanos_per_element: f64,
    /// Coefficient of determination. Well below 1 means the linear shape is
    /// wrong for this backend over this range and the crossover derived from
    /// it should not be trusted.
    pub fit_quality: f64,
}

impl PlacementModel {
    pub fn fit(timings: &[Timing]) -> Option<PlacementModel> {
        if timings.len() < 2 {
            return None;
        }
        let points: Vec<(f64, f64)> = timings
            .iter()
            .map(|t| (t.element_count as f64, t.median().as_nanos() as f64))
            .collect();
        let n = points.len() as f64;
        let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n;
        let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n;
        let covariance: f64 = points
            .iter()
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let variance: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
        if variance == 0.0 {
            return None;
        }
        let slope = covariance / variance;
        let intercept = mean_y - slope * mean_x;

        let residual: f64 = points
            .iter()
            .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
            .sum();
        let total: f64 = points.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let fit_quality = if total == 0.0 {
            1.0
        } else {
            1.0 - residual / total
        };

        Some(PlacementModel {
            fixed_nanos: intercept,
            nanos_per_element: slope,
            fit_quality,
        })
    }

    pub fn predict_nanos(&self, element_count: u32) -> f64 {
        self.fixed_nanos + self.nanos_per_element * element_count as f64
    }
}

/// Smallest element count at which the accelerator is predicted to beat the
/// CPU by more than `margin` (0.15 means "at least 15% faster").
///
/// `None` when the accelerator's per-element cost is not lower than the CPU's,
/// in which case no batch size makes it profitable and the answer is not a
/// large number, it is never.
pub fn crossover(cpu: &PlacementModel, accelerator: &PlacementModel, margin: f64) -> Option<u32> {
    let scale = 1.0 - margin;
    // Solve: acc_fixed + n*acc_elem = scale * (cpu_fixed + n*cpu_elem)
    let denominator = scale * cpu.nanos_per_element - accelerator.nanos_per_element;
    if denominator <= 0.0 {
        return None;
    }
    let numerator = accelerator.fixed_nanos - scale * cpu.fixed_nanos;
    let n = numerator / denominator;
    if n <= 0.0 {
        return Some(1);
    }
    if n > u32::MAX as f64 {
        return None;
    }
    Some(n.ceil() as u32)
}

/// Smallest *measured* size at which the accelerator beat the CPU by more than
/// `margin`, rather than the smallest size at which the fitted lines say it
/// should have.
///
/// Both are reported because they disagree, and the disagreement is
/// informative. Neither backend is exactly linear over four decades — the CPU
/// leaves cache, the GPU stops being submission-bound — so a least-squares fit
/// over the whole range can put the crossing at twice the size the table
/// plainly shows. When they disagree, this one is the observation and
/// `crossover` is the extrapolation.
pub fn measured_crossover(cpu: &Sweep, accelerator: &Sweep, margin: f64) -> Option<u32> {
    let scale = 1.0 - margin;
    for (slow, fast) in cpu.timings.iter().zip(&accelerator.timings) {
        debug_assert_eq!(slow.element_count, fast.element_count);
        if fast.median().as_secs_f64() < slow.median().as_secs_f64() * scale {
            return Some(fast.element_count);
        }
    }
    None
}

/// One backend's sweep over a set of element counts.
#[derive(Clone, Debug)]
pub struct Sweep {
    pub backend: String,
    pub timings: Vec<Timing>,
}

impl Sweep {
    pub fn model(&self) -> Option<PlacementModel> {
        PlacementModel::fit(&self.timings)
    }
}

/// Render a set of sweeps over the same sizes as one table, so the backends
/// are read against each other rather than in sequence.
pub fn render(title: &str, sweeps: &[Sweep]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "{title}");
    let _ = write!(out, "{:>10}", "elements");
    for sweep in sweeps {
        let _ = write!(
            out,
            " | {:>12} {:>9} {:>6}",
            sweep.backend, "ns/elem", "jitter"
        );
    }
    let _ = writeln!(out);

    let rows = sweeps.first().map(|s| s.timings.len()).unwrap_or(0);
    for row in 0..rows {
        let count = sweeps[0].timings[row].element_count;
        let _ = write!(out, "{count:>10}");
        for sweep in sweeps {
            let timing = &sweep.timings[row];
            let _ = write!(
                out,
                " | {:>10.3}ms {:>9.2} {:>5.0}%",
                timing.median().as_secs_f64() * 1e3,
                timing.nanos_per_element(),
                timing.jitter() * 100.0,
            );
        }
        let _ = writeln!(out);
    }

    for sweep in sweeps {
        if let Some(model) = sweep.model() {
            // A negative intercept is not a negative fixed cost. It means the
            // backend's per-element cost rises with n over this range —
            // cache behaviour on the CPU side — so the line is being fitted
            // to something that is not quite a line, and any crossover read
            // off it is optimistic about small batches.
            let caveat = if model.fixed_nanos < 0.0 {
                "  [negative intercept: superlinear over this range]"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "  {:<14} time(n) = {:.0}ns + n*{:.3}ns   (fit {:.4}){caveat}",
                sweep.backend, model.fixed_nanos, model.nanos_per_element, model.fit_quality,
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executives::batch::CpuReferenceBackend;

    #[test]
    fn a_synthetic_body_computes_the_same_thing_on_both_lowerings_of_its_own_interpreter() {
        // Not an I20 check — there is one backend here. This is the weaker
        // claim that the generated body is well-formed, deterministic, and
        // actually depends on its input, so a benchmark of it is a benchmark
        // of arithmetic rather than of a body the validator would reject.
        let program = synthetic_program(900, 2, 32);
        let inputs = synthetic_inputs(4, program.stride());
        let mut backend = CpuReferenceBackend::with(&[&program]);
        let first = backend
            .evaluate(program.id(), &inputs, 4, program.stride())
            .unwrap();
        let second = backend
            .evaluate(program.id(), &inputs, 4, program.stride())
            .unwrap();
        assert_eq!(first, second);
        assert_ne!(first, inputs, "the body did not change its input");
    }

    #[test]
    fn op_count_scales_the_body() {
        let small = synthetic_program(901, 2, 8);
        let large = synthetic_program(902, 2, 512);
        assert!(large.ops().len() > small.ops().len() + 400);
        assert_eq!(small.stride(), 8);
    }

    #[test]
    fn a_fitted_model_recovers_the_line_it_was_given() {
        let timings: Vec<Timing> = [10u32, 100, 1_000, 10_000]
            .iter()
            .map(|&n| Timing {
                element_count: n,
                samples: vec![Duration::from_nanos(500 + 4 * n as u64)],
            })
            .collect();
        let model = PlacementModel::fit(&timings).unwrap();
        assert!((model.nanos_per_element - 4.0).abs() < 0.01);
        assert!((model.fixed_nanos - 500.0).abs() < 1.0);
        assert!(model.fit_quality > 0.999);
    }

    #[test]
    fn a_backend_that_is_slower_per_element_never_crosses_over() {
        let cpu = PlacementModel {
            fixed_nanos: 100.0,
            nanos_per_element: 1.0,
            fit_quality: 1.0,
        };
        let slow = PlacementModel {
            fixed_nanos: 50_000.0,
            nanos_per_element: 2.0,
            fit_quality: 1.0,
        };
        assert_eq!(crossover(&cpu, &slow, 0.15), None);

        let fast = PlacementModel {
            fixed_nanos: 50_000.0,
            nanos_per_element: 0.1,
            fit_quality: 1.0,
        };
        let n = crossover(&cpu, &fast, 0.15).unwrap();
        assert!(fast.predict_nanos(n) < cpu.predict_nanos(n) * 0.85);
        assert!(fast.predict_nanos(n - 1) >= cpu.predict_nanos(n - 1) * 0.85);
    }
}
