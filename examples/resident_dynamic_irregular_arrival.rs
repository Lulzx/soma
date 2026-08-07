#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use soma::compiler::body::EvaluatorProgram;
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::{ResidentDynamicFrameConfig, ResidentDynamicTraceEvent};
    use std::time::{Duration, Instant};

    const A: u32 = 2_300;
    const B: u32 = 2_301;
    const GENERIC: u32 = 2_302;
    const STEPS: u32 = 16;
    const WORK: u32 = 512;
    const LANES: usize = 65_536;
    const BATCHES: u32 = 2;
    const SAMPLES: u32 = 11;

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

    let compile_started = Instant::now();
    let (a_source, b_source, generic_source, one_source) = sources();
    let first = compile_evaluator(47_200, "stress-a", &a_source).unwrap();
    let second = compile_evaluator(47_201, "stress-b", &b_source).unwrap();
    let _one_source = one_source;
    let compile_us = compile_started.elapsed().as_micros();

    let frames = input(LANES, true, true);
    let expected = cpu_bucketed(&first, &second, &frames, LANES);
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
    const WAVE_SIZES: [usize; 8] = [8_192, 4_096, 12_288, 2_048, 16_384, 6_144, 10_240, 6_144];
    const DELAYS_MS: [u64; 8] = [0, 3, 11, 12, 24, 31, 33, 45];

    #[allow(clippy::too_many_arguments)]
    fn run_irregular(
        grouped: bool,
        batch: u32,
        sample: u32,
        config: ResidentDynamicFrameConfig,
        frames: &[u8],
        a_source: &str,
        b_source: &str,
        generic_source: &str,
    ) -> (u128, Vec<u8>, Vec<ResidentDynamicTraceEvent>, usize, String) {
        use std::sync::{Arc, Barrier, OnceLock};
        let ready = Arc::new(Barrier::new(WAVE_SIZES.len() + 1));
        let start = Arc::new(Barrier::new(WAVE_SIZES.len() + 1));
        let release = Arc::new(OnceLock::<Instant>::new());
        let mut offset = 0usize;
        let parts: Vec<_> = WAVE_SIZES
            .iter()
            .copied()
            .enumerate()
            .map(|(wave, lanes)| {
                let part = (wave, offset, lanes);
                offset += lanes;
                part
            })
            .collect();
        assert_eq!(offset, LANES);
        let results = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for &(wave, lane_offset, lanes) in &parts {
                let ready = Arc::clone(&ready);
                let start = Arc::clone(&start);
                let release = Arc::clone(&release);
                let input = frames[lane_offset * 24..(lane_offset + lanes) * 24].to_vec();
                handles.push(scope.spawn(move || {
                    // All evaluator and Metal pipeline setup happens before the
                    // common release barrier and is excluded from arrival time.
                    let id = 60_000 + batch * 1_000 + sample * 16 + wave as u32;
                    let first = compile_evaluator(id, "arrival-a", a_source).unwrap();
                    let second = compile_evaluator(id + 200_000, "arrival-b", b_source).unwrap();
                    let generic =
                        compile_evaluator(id + 400_000, "arrival-generic", generic_source).unwrap();
                    let mut backend = MetalResidentSearch::new().unwrap();
                    if grouped {
                        backend.install_frame_handler(A, &first).unwrap();
                        backend.install_frame_handler(B, &second).unwrap();
                    } else {
                        backend.install_frame_handler(GENERIC, &generic).unwrap();
                    }
                    ready.wait();
                    start.wait();
                    let released = *release.get().expect("main publishes release before start");
                    std::thread::sleep(Duration::from_millis(DELAYS_MS[wave]));
                    let launch_us = released.elapsed().as_micros();
                    let mut result = if grouped {
                        backend
                            .run_dynamic_frame_graph(config, &input, lanes as u32)
                            .unwrap()
                    } else {
                        backend
                            .run_dynamic_ungrouped_frame_graph(
                                config,
                                GENERIC,
                                &input,
                                lanes as u32,
                            )
                            .unwrap()
                    };
                    let finish_us = released.elapsed().as_micros();
                    for event in &mut result.trace {
                        event.lane += lane_offset as u32;
                    }
                    (wave, lane_offset, launch_us, finish_us, result)
                }));
            }
            ready.wait();
            release.set(Instant::now()).unwrap();
            start.wait();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let elapsed = release.get().unwrap().elapsed().as_micros();
        let mut output = vec![0u8; frames.len()];
        let mut trace = Vec::new();
        let mut intervals = Vec::new();
        let mut completion = results
            .iter()
            .map(|(wave, _, _, finish, _)| (*finish, *wave))
            .collect::<Vec<_>>();
        for (_, lane_offset, launch, finish, result) in results {
            let at = lane_offset * 24;
            output[at..at + result.frames.len()].copy_from_slice(&result.frames);
            trace.extend(result.trace);
            intervals.push((launch, finish));
        }
        trace.sort_by_key(|event| (event.epoch, event.lane));
        completion.sort_unstable();
        let completion_order = completion
            .into_iter()
            .map(|(_, wave)| wave.to_string())
            .collect::<Vec<_>>()
            .join(":");
        let mut points = intervals
            .iter()
            .flat_map(|&(launch, finish)| [(launch, 1i32), (finish, -1i32)])
            .collect::<Vec<_>>();
        // Finish sorts before launch at equal timestamps.
        points.sort_by_key(|&(time, delta)| (time, delta));
        let (mut inflight, mut peak) = (0i32, 0i32);
        for (_, delta) in points {
            inflight += delta;
            peak = peak.max(inflight);
        }
        (elapsed, output, trace, peak as usize, completion_order)
    }

    eprintln!("resident_dynamic_irregular_arrival: eight irregularly released concurrent frozen-chunk threads/Metal queues; delays={DELAYS_MS:?}; sizes={WAVE_SIZES:?}; compile_us={compile_us}");
    eprintln!("setup/pipeline compile excluded behind a barrier; elapsed begins at release and includes irregular waits, allocation, submit/wait/readback, and joins");
    eprintln!("control limitation: eight independent frozen chunk graphs/queues, not live ingress into one persistent resident command buffer");
    println!("batch,sample,first,grouped_arrival_us,generic_arrival_us,grouped_peak_inflight,generic_peak_inflight,grouped_completion,generic_completion,grouped_submits,generic_submits,exact");
    for batch in 0..BATCHES {
        for sample in 0..SAMPLES {
            std::thread::sleep(Duration::from_millis(25));
            let grouped_first = (batch + sample) % 2 == 0;
            let run_grouped = || {
                run_irregular(
                    true,
                    batch,
                    sample,
                    config,
                    &frames,
                    &a_source,
                    &b_source,
                    &generic_source,
                )
            };
            let run_generic = || {
                run_irregular(
                    false,
                    batch,
                    sample,
                    config,
                    &frames,
                    &a_source,
                    &b_source,
                    &generic_source,
                )
            };
            let (grouped_result, generic_result) = if grouped_first {
                (run_grouped(), run_generic())
            } else {
                let generic_result = run_generic();
                (run_grouped(), generic_result)
            };
            let (grouped_us, grouped_frames, grouped_trace, grouped_peak, grouped_order) =
                grouped_result;
            let (generic_us, generic_frames, generic_trace, generic_peak, generic_order) =
                generic_result;
            let exact = (grouped_frames, grouped_trace) == expected
                && (generic_frames, generic_trace) == expected;
            assert!(exact);
            assert!(
                grouped_peak > 1 && generic_peak > 1,
                "arrival calls must overlap"
            );
            println!("{batch},{sample},{},{grouped_us},{generic_us},{grouped_peak},{generic_peak},{grouped_order},{generic_order},8,8,{exact}", if grouped_first { "grouped" } else { "generic" });
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("resident_dynamic_irregular_arrival requires macOS and --features metal");
}
