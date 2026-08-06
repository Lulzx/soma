//! Where the fixed cost of a `MetalBatchBackend::evaluate` call goes.
//!
//! `backend_bench` fits `time(n) = fixed + n*per_element` and finds a fixed
//! cost of roughly 400µs, which swamps everything below ten thousand elements.
//! That number is the sum of the six things `evaluate` does before it computes
//! anything, and knowing which of them it is decides what is worth changing:
//! a persistent command queue and a buffer ring are cheap to build and only
//! pay off if allocation is the cost, while asynchronous submission is a
//! change to the collective-completion path and only pays off if the
//! synchronous wait is.
//!
//!     cargo run --release --features metal --example metal_overhead

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use std::ffi::c_void;
    use std::time::{Duration, Instant};

    use metal::{Device, MTLResourceOptions, MTLSize};
    use soma::compiler::body::EvaluatorProgram;
    use soma::executives::batch::BatchBackend;
    use soma::executives::metal::MetalBatchBackend;
    use soma::experiments::backend_bench::{synthetic_inputs, synthetic_program};

    /// Median and fastest of `reps` timings of `body`, after a warmup.
    ///
    /// Both, because several of the rows below differ by less than the spread
    /// between them and a single column would invite reading a ranking into
    /// noise.
    fn median(reps: usize, mut body: impl FnMut()) -> (Duration, Duration) {
        for _ in 0..8 {
            body();
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            body();
            samples.push(start.elapsed());
        }
        samples.sort();
        (samples[samples.len() / 2], samples[0])
    }

    let device = Device::system_default().expect("no Metal device");
    println!("device: {}", device.name());

    // The batch size below which the fixed cost is the whole measurement.
    const ELEMENTS: u32 = 1_024;
    let program: EvaluatorProgram = synthetic_program(880, 2, 32);
    let stride = program.stride();
    let bytes = ELEMENTS as u64 * stride as u64;
    let inputs = synthetic_inputs(ELEMENTS, stride);

    let queue = device.new_command_queue();
    let pipeline = {
        let library = device
            .new_library_with_source(&program.metal_source(), &metal::CompileOptions::new())
            .expect("generated MSL compiles");
        let function = library
            .get_function(&program.metal_entry_point(), None)
            .unwrap();
        device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap()
    };

    println!(
        "\nfixed costs, at {ELEMENTS} elements of {stride}B ({bytes} bytes)\n{:>42} {:>11} {:>11}",
        "step", "median", "fastest"
    );
    let line = |name: &str, (median, min): (Duration, Duration)| {
        println!(
            "{name:>42} {:>9.1}µs {:>9.1}µs",
            median.as_secs_f64() * 1e6,
            min.as_secs_f64() * 1e6,
        );
    };

    line(
        "new_command_queue()",
        median(500, || {
            std::hint::black_box(device.new_command_queue());
        }),
    );

    line(
        "new_buffer_with_data(input)",
        median(500, || {
            std::hint::black_box(device.new_buffer_with_data(
                inputs.as_ptr().cast::<c_void>(),
                bytes,
                MTLResourceOptions::StorageModeShared,
            ));
        }),
    );

    line(
        "new_buffer(output)",
        median(500, || {
            std::hint::black_box(device.new_buffer(bytes, MTLResourceOptions::StorageModeShared));
        }),
    );

    line(
        "empty command buffer: commit + wait",
        median(200, || {
            let command = queue.new_command_buffer();
            command.commit();
            command.wait_until_completed();
        }),
    );

    // Allocation looked like a rounding error at 8KB, which made reusing
    // buffers look like a rounding error too. It is not a fixed cost: a fresh
    // shared buffer is fresh pages, and filling it faults every one of them.
    // Reusing one turns a per-byte cost into a warm copy, which is why the
    // fitted per-element cost moved and not just the intercept.
    println!("\nfresh allocate + fill vs. copy into a reused buffer");
    for megabytes in [1u64, 8, 32] {
        let size = megabytes * 1024 * 1024;
        let source = vec![0xA5u8; size as usize];
        let reused = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
        line(
            &format!("{megabytes}MB: new_buffer_with_data"),
            median(20, || {
                std::hint::black_box(device.new_buffer_with_data(
                    source.as_ptr().cast::<c_void>(),
                    size,
                    MTLResourceOptions::StorageModeShared,
                ));
            }),
        );
        line(
            &format!("{megabytes}MB: memcpy into reused buffer"),
            median(20, || unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    reused.contents().cast::<u8>(),
                    size as usize,
                );
            }),
        );
    }

    // The same dispatch the backend encodes, on buffers allocated once, with
    // the queue and pipeline already in hand. This is the floor a fully
    // Phase-1'd synchronous backend could reach: everything left is encoding,
    // submission, and the round trip.
    let input_buffer = device.new_buffer_with_data(
        inputs.as_ptr().cast::<c_void>(),
        bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let element_count = ELEMENTS;
    let element_stride = stride;

    let dispatch = |threads_per_group: u64| {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&element_count as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&element_stride as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: element_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group.min(element_count as u64),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
    };

    line(
        "encode + dispatch + wait (reused buffers)",
        median(200, || dispatch(pipeline.thread_execution_width())),
    );

    let mut backend = MetalBatchBackend::with(&[&program]).unwrap();
    line(
        "current evaluate() end to end",
        median(200, || {
            std::hint::black_box(
                backend
                    .evaluate(program.id(), &inputs, ELEMENTS, stride)
                    .unwrap(),
            );
        }),
    );

    // Threadgroup sizing, the other Phase 1 item. The backend currently uses
    // exactly one SIMD width.
    let simd = pipeline.thread_execution_width();
    let max = pipeline.max_total_threads_per_threadgroup();
    println!("\nthreadgroup sizing (SIMD width {simd}, max {max} per group)");
    for multiple in [1u64, 2, 4, 8, 16] {
        let size = simd * multiple;
        if size > max {
            break;
        }
        line(
            &format!("{size} threads per group"),
            median(200, || dispatch(size)),
        );
    }

    // Same sweep at a size where the GPU is doing real work, since a
    // threadgroup that is too small is invisible when the dispatch is
    // dominated by submission.
    const LARGE: u32 = 1_048_576;
    let large_bytes = LARGE as u64 * stride as u64;
    let large_inputs = synthetic_inputs(LARGE, stride);
    let large_input_buffer = device.new_buffer_with_data(
        large_inputs.as_ptr().cast::<c_void>(),
        large_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let large_output_buffer = device.new_buffer(large_bytes, MTLResourceOptions::StorageModeShared);
    let large_count = LARGE;

    let large_dispatch = |threads_per_group: u64| {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&large_input_buffer), 0);
        encoder.set_buffer(1, Some(&large_output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&large_count as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&element_stride as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: large_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group.min(large_count as u64),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
    };

    println!("\nthreadgroup sizing at {LARGE} elements");
    for multiple in [1u64, 2, 4, 8, 16] {
        let size = simd * multiple;
        if size > max {
            break;
        }
        line(
            &format!("{size} threads per group"),
            median(200, || large_dispatch(size)),
        );
    }

    // What is left of the fixed cost after a persistent queue and reused
    // buffers is encode, submit, and the round trip. Two proposals attack it
    // and they cost very different amounts to adopt, so the question is which
    // half of the floor each one removes.
    //
    // An epoch offering several ready cohorts is run three ways:
    //
    //   per-cohort wait   what the backend does now: one command buffer per
    //                     cohort, each committed and waited on before the next
    //                     is encoded. Every cohort pays a full round trip.
    //   deferred wait     the same N command buffers, all committed, waited on
    //                     once at the end. Removes the stalls and keeps the
    //                     command buffers: this is what asynchronous
    //                     submission buys, and it reaches into the
    //                     collective-completion path to get it.
    //   one command buffer  all N dispatches encoded into a single command
    //                     buffer, committed once. Removes the per-cohort
    //                     command buffer as well, and needs only a
    //                     backend-level entry point that takes several
    //                     requests.
    //
    // Each cohort gets its own slice of one buffer, so the dispatches are
    // independent work rather than N repetitions overwriting one region.
    const COHORT: u32 = 8_192;
    let chunk = COHORT as u64 * stride as u64;
    let cohort_count = COHORT;

    let encode_into = |encoder: &metal::ComputeCommandEncoderRef, slot: u64| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&large_input_buffer), slot * chunk);
        encoder.set_buffer(1, Some(&large_output_buffer), slot * chunk);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&cohort_count as *const u32).cast::<c_void>(),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&element_stride as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: COHORT as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: simd,
                height: 1,
                depth: 1,
            },
        );
    };

    println!("\nan epoch of {COHORT}-element cohorts, three submission shapes");
    for cohorts in [1u64, 4, 16, 64] {
        line(
            &format!("{cohorts} cohorts: per-cohort wait"),
            median(30, || {
                for slot in 0..cohorts {
                    let command = queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    encode_into(encoder, slot);
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                }
            }),
        );
        line(
            &format!("{cohorts} cohorts: deferred wait"),
            median(30, || {
                let mut last = None;
                for slot in 0..cohorts {
                    let command = queue.new_command_buffer().to_owned();
                    let encoder = command.new_compute_command_encoder();
                    encode_into(encoder, slot);
                    encoder.end_encoding();
                    command.commit();
                    last = Some(command);
                }
                if let Some(command) = last {
                    command.wait_until_completed();
                }
            }),
        );
        line(
            &format!("{cohorts} cohorts: one command buffer"),
            median(30, || {
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                for slot in 0..cohorts {
                    encode_into(encoder, slot);
                }
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
            }),
        );
    }
}
