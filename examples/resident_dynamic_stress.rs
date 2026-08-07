#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use soma::compiler::body::EvaluatorProgram;
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::{
        ResidentDynamicFrameConfig, ResidentDynamicTraceEvent, ResidentFrameBinding,
        ResidentFrameGraphConfig,
    };
    use std::time::{Duration, Instant};

    const A: u32 = 2_300;
    const B: u32 = 2_301;
    const GENERIC: u32 = 2_302;
    const ONE_CLASS: u32 = 2_303;
    const STEPS: u32 = 16;
    const WORK: u32 = 512;
    const LANES: usize = 49_152;
    const BATCHES: u32 = 3;
    const SAMPLES: u32 = 11;
    const WARMUPS: u32 = 2;

    fn sources() -> (String, String, String, String) {
        let prefix = "field u32\nfield u32\nfield u64\nfield u32\nfield u32\nlocal acc\nlet n = load 0\nlet minus = const 0xffffffff\nlet d = add n minus\nstore 0 d\nlet v = load 2\nset acc v\n";
        let loop_a = format!("repeat {WORK}\nlet stop_a = load 3\nbreak_if stop_a\nlet old_a = get acc\nlet three = const 3\nlet product_a = mul old_a three\nlet one_a = const 1\nlet changed_a = add product_a one_a\nset acc changed_a\nend\n");
        let loop_b = format!("repeat {WORK}\nlet stop_b = load 4\nbreak_if stop_b\nlet old_b = get acc\nlet five = const 5\nlet product_b = mul old_b five\nlet one_b = const 1\nlet changed_b = add product_b one_b\nset acc changed_b\nend\n");
        let suffix = |next: u32, skip_a: u32, skip_b: u32| {
            format!("let out = get acc\nstore 2 out\nlet next = const {next}\nstore 1 next\nlet sa = const {skip_a}\nstore 3 sa\nlet sb = const {skip_b}\nstore 4 sb\n")
        };
        let a = prefix.to_owned() + &loop_a + &suffix(B, 1, 0);
        let b = prefix.to_owned() + &loop_b + &suffix(A, 0, 1);
        // This is a competent generic worker: every lane enters both loops,
        // and break_if skips the irrelevant counted body at its head.
        let generic_tail = format!("let out = get acc\nstore 2 out\nlet current_sa = load 3\nlet current_sb = load 4\nstore 3 current_sb\nstore 4 current_sa\nlet base = const {B}\nlet delta = const 0xffffffff\nlet shift = mul current_sa delta\nlet next = add base shift\nstore 1 next\n");
        let generic = prefix.to_owned() + &loop_a + &loop_b + &generic_tail;
        let one = prefix.to_owned() + &loop_a + &suffix(A, 0, 1);
        (a, b, generic, one)
    }

    fn input(lanes: usize, heterogeneous_depth: bool, mixed_classes: bool) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(lanes * 24);
        for lane in 0..lanes {
            // Both parities occur at every depth, so every live epoch remains
            // class-mixed even as the shorter searches quiesce.
            let remaining = if heterogeneous_depth {
                STEPS - 2 * ((lane / 2) % 4) as u32
            } else {
                STEPS
            };
            let starts_a = !mixed_classes || lane % 2 == 0;
            bytes.extend_from_slice(&remaining.to_le_bytes());
            bytes.extend_from_slice(&(if starts_a { A } else { B }).to_le_bytes());
            bytes.extend_from_slice(&(lane as u64 + 1).to_le_bytes());
            bytes.extend_from_slice(&(if starts_a { 0u32 } else { 1 }).to_le_bytes());
            bytes.extend_from_slice(&(if starts_a { 1u32 } else { 0 }).to_le_bytes());
        }
        bytes
    }

    fn cpu_bucketed(
        first: &EvaluatorProgram,
        second: &EvaluatorProgram,
        input: &[u8],
        lanes: usize,
    ) -> (Vec<u8>, Vec<ResidentDynamicTraceEvent>) {
        let mut frames = input.to_vec();
        let mut classes: Vec<u32> = frames
            .chunks_exact(24)
            .map(|frame| u32::from_le_bytes(frame[4..8].try_into().unwrap()))
            .collect();
        let mut active = vec![true; lanes];
        let mut trace = Vec::new();
        for epoch in 0..STEPS {
            let frozen = frames.clone();
            let mut buckets = [Vec::new(), Vec::new()];
            for lane in 0..lanes {
                if active[lane] {
                    buckets[usize::from(classes[lane] == B)].push(lane);
                }
            }
            assert!(!buckets[0].is_empty() && !buckets[1].is_empty());
            for (bucket, program) in buckets.iter().zip([first, second]) {
                for &lane in bucket {
                    let range = lane * 24..lane * 24 + 24;
                    program.evaluate_at(
                        &frozen,
                        lanes as u32,
                        lane as u32,
                        &mut frames[range.clone()],
                    );
                    let remaining = u32::from_le_bytes(
                        frames[range.start..range.start + 4].try_into().unwrap(),
                    );
                    let run_class = classes[lane];
                    if remaining == 0 {
                        active[lane] = false;
                    } else {
                        classes[lane] = u32::from_le_bytes(
                            frames[range.start + 4..range.start + 8].try_into().unwrap(),
                        );
                    }
                    let word = frames[range].iter().fold(2_166_136_261u32, |hash, byte| {
                        (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
                    });
                    trace.push(ResidentDynamicTraceEvent {
                        epoch,
                        lane: lane as u32 + 1,
                        run_class,
                        step_kind: if remaining == 0 { 1 } else { 2 },
                        word,
                        ..Default::default()
                    });
                }
            }
        }
        trace.sort_by_key(|event| (event.epoch, event.lane));
        (frames, trace)
    }

    fn cpu_fixed(program: &EvaluatorProgram, input: &[u8], lanes: usize) -> Vec<u8> {
        let mut frames = input.to_vec();
        for _ in 0..STEPS {
            let frozen = frames.clone();
            for lane in 0..lanes {
                let range = lane * 24..lane * 24 + 24;
                program.evaluate_at(&frozen, lanes as u32, lane as u32, &mut frames[range]);
            }
        }
        frames
    }

    fn micros<T>(f: impl FnOnce() -> T) -> (u128, T) {
        let started = Instant::now();
        let value = f();
        (started.elapsed().as_micros(), value)
    }

    let compile_started = Instant::now();
    let (a_source, b_source, generic_source, one_source) = sources();
    let first = compile_evaluator(47_200, "stress-a", &a_source).unwrap();
    let second = compile_evaluator(47_201, "stress-b", &b_source).unwrap();
    let generic = compile_evaluator(47_202, "stress-generic", &generic_source).unwrap();
    let one = compile_evaluator(47_203, "stress-one", &one_source).unwrap();
    let compile_us = compile_started.elapsed().as_micros();

    let frames = input(LANES, true, true);
    let one_frames = input(LANES, false, false);
    let expected = cpu_bucketed(&first, &second, &frames, LANES);
    let expected_one = cpu_fixed(&one, &one_frames, LANES);
    for epoch in 0..STEPS {
        let events = expected.1.iter().filter(|event| event.epoch == epoch);
        let (mut saw_a, mut saw_b) = (false, false);
        for event in events {
            saw_a |= event.run_class == A;
            saw_b |= event.run_class == B;
        }
        assert!(saw_a && saw_b, "epoch {epoch} must contain both classes");
    }

    let config = ResidentDynamicFrameConfig {
        first_run_class: A,
        second_run_class: B,
        max_steps: STEPS,
        cohort_width: 32,
        remaining_offset: 0,
        next_class_offset: 4,
    };
    let bindings: Vec<_> = (0..LANES)
        .map(|lane| ResidentFrameBinding {
            continuation: lane as u64 + 1,
            process: lane as u64 + 1,
            frame: lane as u64 + 1,
            actor: lane as u64 + 1,
            target: lane as u64 + 1,
            ..Default::default()
        })
        .collect();

    eprintln!("resident_dynamic_stress: M4-class bounded divergence stress; lanes={LANES} steps={STEPS} work={WORK} batches={BATCHES} samples={SAMPLES} warmups={WARMUPS}");
    eprintln!("timings include per-call Metal allocation, one or more submit/waits, and final readback; evaluator compile_us={compile_us} is excluded");
    eprintln!("primary input alternates initial A/B layout and has four heterogeneous depths; every epoch is asserted mixed; generic uses break_if");
    eprintln!("low_arrival_us is a historical label for eight sequential chunk submissions, not asynchronous arrival");
    println!("batch,sample,first,cpu_bucket_us,grouped_us,generic_us,one_class_us,sorted_multisubmit_us,low_arrival_us,grouped_submits,generic_submits,one_submits,multisubmit_submits,arrival_submits,exact");

    for batch in 0..BATCHES {
        // Recreate all device objects per batch. Compilation/allocation caused by
        // construction and explicit warmups is outside the reported samples.
        let mut grouped = MetalResidentSearch::new().unwrap();
        grouped.install_frame_handler(A, &first).unwrap();
        grouped.install_frame_handler(B, &second).unwrap();
        let mut ungrouped = MetalResidentSearch::new().unwrap();
        ungrouped.install_frame_handler(GENERIC, &generic).unwrap();
        let mut one_class = MetalResidentSearch::new().unwrap();
        one_class.install_frame_handler(ONE_CLASS, &one).unwrap();
        let mut multisubmit = MetalResidentSearch::new().unwrap();
        multisubmit.install_frame_handler(A, &first).unwrap();
        multisubmit.install_frame_handler(B, &second).unwrap();
        let mut arrival = MetalResidentSearch::new().unwrap();
        arrival.install_frame_handler(A, &first).unwrap();
        arrival.install_frame_handler(B, &second).unwrap();

        // The CPU oracle is timed once per batch, not immediately before each
        // GPU pair: a multi-second interpreter run would thermally confound the
        // first member of every AB/BA pair.
        let (cpu_us, cpu_result) = micros(|| cpu_bucketed(&first, &second, &frames, LANES));
        assert_eq!(cpu_result, expected);

        for _ in 0..WARMUPS {
            let g = grouped
                .run_dynamic_frame_graph(config, &frames, LANES as u32)
                .unwrap();
            let u = ungrouped
                .run_dynamic_ungrouped_frame_graph(config, GENERIC, &frames, LANES as u32)
                .unwrap();
            assert_eq!((g.frames, g.trace), expected);
            assert_eq!((u.frames, u.trace), expected);
        }

        // Give every primary pair the same idle interval after warmup/control work.
        std::thread::sleep(Duration::from_millis(50));
        for sample in 0..SAMPLES {
            std::thread::sleep(Duration::from_millis(25));
            let grouped_first = (batch + sample) % 2 == 0;
            let run_grouped = |backend: &mut MetalResidentSearch| {
                micros(|| {
                    backend
                        .run_dynamic_frame_graph(config, &frames, LANES as u32)
                        .unwrap()
                })
            };
            let run_generic = |backend: &mut MetalResidentSearch| {
                micros(|| {
                    backend
                        .run_dynamic_ungrouped_frame_graph(config, GENERIC, &frames, LANES as u32)
                        .unwrap()
                })
            };
            let (grouped_timed, generic_timed) = if grouped_first {
                (run_grouped(&mut grouped), run_generic(&mut ungrouped))
            } else {
                let generic_timed = run_generic(&mut ungrouped);
                (run_grouped(&mut grouped), generic_timed)
            };
            let (grouped_us, grouped_result) = grouped_timed;
            let (generic_us, generic_result) = generic_timed;

            let (one_us, one_result) = micros(|| {
                one_class
                    .run_frame_graph(
                        ResidentFrameGraphConfig {
                            run_class: ONE_CLASS,
                            epochs: STEPS,
                            cohort_width: 32,
                        },
                        &bindings,
                        &one_frames,
                    )
                    .unwrap()
            });
            assert_eq!(one_result.frames, expected_one);

            let (multisubmit_us, multisubmit_frames) = micros(|| {
                let mut current = frames.clone();
                for _ in 0..STEPS {
                    current = multisubmit
                        .run_dynamic_frame_graph(
                            ResidentDynamicFrameConfig {
                                max_steps: 1,
                                ..config
                            },
                            &current,
                            LANES as u32,
                        )
                        .unwrap()
                        .frames;
                }
                current
            });

            let (arrival_us, arrival_frames) = micros(|| {
                let chunk_lanes = LANES / 8;
                let mut output = Vec::with_capacity(frames.len());
                for chunk in frames.chunks(chunk_lanes * 24) {
                    output.extend(
                        arrival
                            .run_dynamic_frame_graph(config, chunk, (chunk.len() / 24) as u32)
                            .unwrap()
                            .frames,
                    );
                }
                output
            });

            let exact = grouped_result.frames == expected.0
                && grouped_result.trace == expected.1
                && generic_result.frames == expected.0
                && generic_result.trace == expected.1
                && multisubmit_frames == expected.0
                && arrival_frames == expected.0;
            assert!(exact);
            println!("{batch},{sample},{},{cpu_us},{grouped_us},{generic_us},{one_us},{multisubmit_us},{arrival_us},1,1,1,{STEPS},8,{exact}", if grouped_first { "grouped" } else { "generic" });
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("resident_dynamic_stress requires macOS and --features metal");
}
