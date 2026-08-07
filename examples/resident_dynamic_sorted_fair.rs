#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use soma::compiler::body::EvaluatorProgram;
    use soma::compiler::surface::compile_evaluator;
    use soma::executives::metal_scheduler::MetalResidentSearch;
    use soma::scheduler::device::{ResidentDynamicFrameConfig, ResidentDynamicTraceEvent};
    use std::time::{Duration, Instant};

    const A: u32 = 2_300;
    const B: u32 = 2_301;
    const STEPS: u32 = 16;
    const WORK: u32 = 512;
    const LANES: usize = 65_536;
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

    fn micros<T>(f: impl FnOnce() -> T) -> (u128, T) {
        let started = Instant::now();
        let value = f();
        (started.elapsed().as_micros(), value)
    }

    let compile_started = Instant::now();
    let (a_source, b_source, generic_source, one_source) = sources();
    let first = compile_evaluator(47_200, "stress-a", &a_source).unwrap();
    let second = compile_evaluator(47_201, "stress-b", &b_source).unwrap();
    let _generic_source = generic_source;
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
    eprintln!("resident_dynamic_sorted_fair: lanes={LANES} steps={STEPS} work={WORK} batches={BATCHES} samples={SAMPLES} warmups={WARMUPS} compile_us={compile_us}");
    eprintln!("fair AB/BA grouped-one-submit versus grouped-16-submit; equal 25 ms pre-pair cooldown; all frames and canonical traces exact");
    println!(
        "batch,sample,first,grouped_us,sorted_16_submit_us,grouped_submits,sorted_submits,exact"
    );

    for batch in 0..BATCHES {
        let mut grouped = MetalResidentSearch::new().unwrap();
        grouped.install_frame_handler(A, &first).unwrap();
        grouped.install_frame_handler(B, &second).unwrap();
        let mut sorted = MetalResidentSearch::new().unwrap();
        sorted.install_frame_handler(A, &first).unwrap();
        sorted.install_frame_handler(B, &second).unwrap();

        let run_grouped = |backend: &mut MetalResidentSearch| {
            micros(|| {
                backend
                    .run_dynamic_frame_graph(config, &frames, LANES as u32)
                    .unwrap()
            })
        };
        let run_sorted = |backend: &mut MetalResidentSearch| {
            micros(|| {
                let mut current = frames.clone();
                let mut trace = Vec::with_capacity(expected.1.len());
                for epoch in 0..STEPS {
                    let result = backend
                        .run_dynamic_frame_graph(
                            ResidentDynamicFrameConfig {
                                max_steps: 1,
                                ..config
                            },
                            &current,
                            LANES as u32,
                        )
                        .unwrap();
                    current = result.frames;
                    trace.extend(result.trace.into_iter().map(|mut event| {
                        event.epoch = epoch;
                        event
                    }));
                }
                trace.sort_by_key(|event| (event.epoch, event.lane));
                (current, trace)
            })
        };

        for _ in 0..WARMUPS {
            let grouped_result = run_grouped(&mut grouped).1;
            let sorted_result = run_sorted(&mut sorted).1;
            assert_eq!((grouped_result.frames, grouped_result.trace), expected);
            assert_eq!(sorted_result, expected);
        }
        std::thread::sleep(Duration::from_millis(50));

        for sample in 0..SAMPLES {
            std::thread::sleep(Duration::from_millis(25));
            let grouped_first = (batch + sample) % 2 == 0;
            let (grouped_timed, sorted_timed) = if grouped_first {
                (run_grouped(&mut grouped), run_sorted(&mut sorted))
            } else {
                let sorted_timed = run_sorted(&mut sorted);
                (run_grouped(&mut grouped), sorted_timed)
            };
            let (grouped_us, grouped_result) = grouped_timed;
            let (sorted_us, sorted_result) = sorted_timed;
            let exact = (grouped_result.frames, grouped_result.trace) == expected
                && sorted_result == expected;
            assert!(exact);
            println!(
                "{batch},{sample},{},{grouped_us},{sorted_us},1,{STEPS},{exact}",
                if grouped_first { "grouped" } else { "sorted" }
            );
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("resident_dynamic_sorted_fair requires macOS and --features metal");
}
