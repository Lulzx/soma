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

    const A: u32 = 2_200;
    const B: u32 = 2_201;
    const GENERIC: u32 = 2_202;
    const ONE_CLASS: u32 = 2_203;
    const STEPS: u32 = 8;
    const WORK: u32 = 32;
    const SAMPLES: u32 = 6;

    fn sources() -> (String, String, String, String) {
        let prefix = "field u32\nfield u32\nfield u64\nfield u32\nfield u32\nlocal acc\nlet n = load 0\nlet minus = const 0xffffffff\nlet d = add n minus\nstore 0 d\nlet v = load 2\nset acc v\n";
        let loop_a = format!("repeat {WORK}\nlet stop_a = load 3\nbreak_if stop_a\nlet old_a = get acc\nlet three = const 3\nlet product_a = mul old_a three\nlet one_a = const 1\nlet changed_a = add product_a one_a\nset acc changed_a\nend\n");
        let loop_b = format!("repeat {WORK}\nlet stop_b = load 4\nbreak_if stop_b\nlet old_b = get acc\nlet five = const 5\nlet product_b = mul old_b five\nlet one_b = const 1\nlet changed_b = add product_b one_b\nset acc changed_b\nend\n");
        let suffix = |next: u32, skip_a: u32, skip_b: u32| {
            format!("let out = get acc\nstore 2 out\nlet next = const {next}\nstore 1 next\nlet sa = const {skip_a}\nstore 3 sa\nlet sb = const {skip_b}\nstore 4 sb\n")
        };
        let a = prefix.to_owned() + &loop_a + &suffix(B, 1, 0);
        let b = prefix.to_owned() + &loop_b + &suffix(A, 0, 1);
        // Each lane enters both loops, but BreakIf at the head of each loop
        // makes the irrelevant class body cost one predicate, not WORK ALU rounds.
        let generic_tail = format!("let out = get acc\nstore 2 out\nlet current_sa = load 3\nlet current_sb = load 4\nstore 3 current_sb\nstore 4 current_sa\nlet base = const {B}\nlet delta = const 0xffffffff\nlet shift = mul current_sa delta\nlet next = add base shift\nstore 1 next\n");
        let generic = prefix.to_owned() + &loop_a + &loop_b + &generic_tail;
        let one = prefix.to_owned() + &loop_a + &suffix(A, 0, 1);
        (a, b, generic, one)
    }

    fn input(lanes: usize, full_depth: bool) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(lanes * 24);
        for lane in 0..lanes {
            let remaining = if full_depth {
                STEPS
            } else {
                1 + ((lane * 17 + lane / 7) % STEPS as usize) as u32
            };
            bytes.extend_from_slice(&remaining.to_le_bytes());
            bytes.extend_from_slice(&A.to_le_bytes());
            bytes.extend_from_slice(&(lane as u64 + 1).to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
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
        let mut classes = vec![A; lanes];
        let mut active = vec![true; lanes];
        let mut trace = Vec::new();
        for epoch in 0..STEPS {
            let frozen = frames.clone();
            // This is a real host class-bucket control, rather than a repeated
            // scalar pass mislabeled as sorted: build both buckets every level.
            let mut buckets = [Vec::new(), Vec::new()];
            for lane in 0..lanes {
                if active[lane] {
                    buckets[usize::from(classes[lane] == B)].push(lane);
                }
            }
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

    let (a_source, b_source, generic_source, one_source) = sources();
    let first = compile_evaluator(46_200, "bench-a", &a_source).unwrap();
    let second = compile_evaluator(46_201, "bench-b", &b_source).unwrap();
    let generic = compile_evaluator(46_202, "bench-generic", &generic_source).unwrap();
    let one = compile_evaluator(46_203, "bench-one", &one_source).unwrap();

    eprintln!("resident_dynamic_bench: bounded {STEPS}-step two-class divergent workload; {WORK} useful ALU rounds/step; release samples alternate grouped/generic order");
    eprintln!("limitation: the resident dynamic graph is one Metal submit but remains standalone; it does not perform a Kernel Phase-G final commit");
    println!("lanes,sample,first,cpu_bucket_us,grouped_us,generic_ungrouped_us,device_one_class_us,multisubmit_level_us,low_arrival_us,grouped_submits,generic_submits,level_submits,arrival_submits,exact");
    for lanes in [1_024usize, 4_096, 16_384] {
        let frames = input(lanes, false);
        let full_frames = input(lanes, true);
        let expected = cpu_bucketed(&first, &second, &frames, lanes);
        let expected_one_class = cpu_fixed(&one, &full_frames, lanes);

        let config = ResidentDynamicFrameConfig {
            first_run_class: A,
            second_run_class: B,
            max_steps: STEPS,
            cohort_width: 32,
            remaining_offset: 0,
            next_class_offset: 4,
        };
        let mut grouped = MetalResidentSearch::new().unwrap();
        grouped.install_frame_handler(A, &first).unwrap();
        grouped.install_frame_handler(B, &second).unwrap();
        let mut ungrouped = MetalResidentSearch::new().unwrap();
        ungrouped.install_frame_handler(GENERIC, &generic).unwrap();
        let mut one_class = MetalResidentSearch::new().unwrap();
        one_class.install_frame_handler(ONE_CLASS, &one).unwrap();
        let bindings: Vec<_> = (0..lanes)
            .map(|lane| ResidentFrameBinding {
                continuation: lane as u64 + 1,
                process: lane as u64 + 1,
                frame: lane as u64 + 1,
                actor: lane as u64 + 1,
                target: lane as u64 + 1,
                ..Default::default()
            })
            .collect();

        // Warm pipelines and allocations before raw samples. Results are still
        // checked; only these warm-up durations are omitted from CSV.
        let warm_g = grouped
            .run_dynamic_frame_graph(config, &frames, lanes as u32)
            .unwrap();
        let warm_u = ungrouped
            .run_dynamic_ungrouped_frame_graph(config, GENERIC, &frames, lanes as u32)
            .unwrap();
        assert_eq!((warm_g.frames, warm_g.trace), expected);
        assert_eq!((warm_u.frames, warm_u.trace), expected);

        for sample in 0..SAMPLES {
            let (cpu_us, cpu_result) = micros(|| cpu_bucketed(&first, &second, &frames, lanes));
            assert_eq!(cpu_result, expected);
            let grouped_first = sample % 2 == 0;
            let run_grouped = |backend: &mut MetalResidentSearch| {
                micros(|| {
                    backend
                        .run_dynamic_frame_graph(config, &frames, lanes as u32)
                        .unwrap()
                })
            };
            let run_generic = |backend: &mut MetalResidentSearch| {
                micros(|| {
                    backend
                        .run_dynamic_ungrouped_frame_graph(config, GENERIC, &frames, lanes as u32)
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
                        &full_frames,
                    )
                    .unwrap()
            });
            assert_eq!(one_result.frames, expected_one_class);

            let mut level = MetalResidentSearch::new().unwrap();
            level.install_frame_handler(A, &first).unwrap();
            level.install_frame_handler(B, &second).unwrap();
            let (level_us, level_frames) = micros(|| {
                let mut current = frames.clone();
                for _ in 0..STEPS {
                    current = level
                        .run_dynamic_frame_graph(
                            ResidentDynamicFrameConfig {
                                max_steps: 1,
                                ..config
                            },
                            &current,
                            lanes as u32,
                        )
                        .unwrap()
                        .frames;
                }
                current
            });

            let mut arrival = MetalResidentSearch::new().unwrap();
            arrival.install_frame_handler(A, &first).unwrap();
            arrival.install_frame_handler(B, &second).unwrap();
            let (arrival_us, arrival_frames) = micros(|| {
                let chunk_lanes = (lanes / 8).max(1);
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
                && level_frames == expected.0
                && arrival_frames == expected.0;
            assert!(exact);
            println!("{lanes},{sample},{},{cpu_us},{grouped_us},{generic_us},{one_us},{level_us},{arrival_us},1,1,{STEPS},8,{exact}", if grouped_first { "grouped" } else { "generic" });
            // Let alternating pairs begin from a less correlated host state.
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("resident_dynamic_bench requires macOS and --features metal");
}
