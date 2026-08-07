# SOMA self-tuning discovery

This is the first non-synthetic Discovery workload: SOMA searches physical
configurations of its own evaluator implementation.

## Question and boundary

The candidates cover:

- scalar-reference and native-compiled CPU placement at one and eight threads,
  run singly or as an epoch;
- Metal placement, singly or in one epoch command buffer;
- reused versus freshly allocated Metal scratch buffers; and
- automatic, 32, 64, 128, and 256-thread Metal threadgroups.

Every candidate runs the same generated body over the same frozen inputs. The
CPU reference interpreter remains the semantic definition. The optional
Cranelift backend supplies the native comparison and must agree with that
definition before its timing is admitted.

## Scientific DAG

Each execution configuration is one hypothesis. For each workload it requests
the same two deterministic nodes, generate-evaluator and prepare-input, then
depends on independent timing observations:

```text
generate evaluator ─┐
                    ├─ observation(config, trial) ─ aggregate/rank
prepare input ──────┘
```

Literal replay executes both preparation requests for every configuration.
Optimized replay realizes each preparation key once. Observation nodes are
never cached: equal durations still represent separate trials and execute
separately. A no-sharing control salts preparation contracts by configuration;
its pending joins fall to zero and physical preparation returns to the logical
count.

## Acquisition protocol

Timing and Discovery replay are separate phases:

1. Generate every evaluator once. Install all programs into one native CPU JIT
   and one Metal backend. Changing thread count or Metal policy does not
   recompile, so candidates share generated machine code and pipelines.
2. Warm every candidate once for each workload.
3. Acquire candidates in a deterministic rotating order. The default fifteen
   trials complete one rotation over the fifteen-candidate hardware matrix.
4. Time only collective execution, publication, and freezing. Kernel/input and
   collective construction occur before the clock.
5. Hash all published output bytes outside the timed interval. Any digest
   disagreement across configurations invalidates the capture.
6. Replay the captured observations unchanged through literal and optimized
   Discovery executors and require D1–D7 plus identical scientific state.

The report gives median and full observed range. This controls first-use and
order effects and exposes large disturbances; it does not control other system
load, temperature, or frequency. Results are a local regime map, not universal
device constants.

```text
cargo run --release --features native,metal --example self_tuning_report
```

## M4 Pro regime map

Measured 2026-08-07 on an Apple M4 Pro with 24 GiB RAM, macOS 26.6, Rust 1.92,
release profile, with Cranelift native code and real Metal dispatch. The full
fifteen-trial table is retained in
[the raw measurement](measurements/SELF-TUNING-M4-PRO-NATIVE-2026-08-07.txt).

| Workload | Winning policy | Median | Main boundary |
| --- | --- | ---: | --- |
| 8 ALU, 1 x 1,024 elements | Native CPU, 1 thread, epoch | 0.020 ms [0.017, 0.027] | Native CPU is 7.5x faster than the best Metal median; fixed submission and threading costs dominate. |
| 256 ALU, 16 x 8,192 elements | Metal, automatic width, epoch, reused buffers | 0.758 ms [0.689, 4.099] | Metal is 2.18x faster than native CPU at eight threads; native compilation is 20.6x faster than the scalar reference. |
| 2,048 ALU, 8 x 131,072 elements | Metal, 256 threads, epoch, reused buffers | 4.585 ms [2.893, 7.703] | Metal is 12.3x faster than native CPU at eight threads; native compilation is 34.4x faster than the reference. |

The interpreter ratios now measure compiler value separately. The placement
claim uses Metal against the fastest native CPU candidate: native wins the tiny
cell, while Metal wins by 2.18x and 12.3x as arithmetic and total work grow.
Within Metal, epoch grouping remains material: automatic-width epoch execution
is 6.2x faster than single submission in the medium cell and 2.19x in the heavy
cell.

The precise threadgroup winner is not stable evidence. Earlier captures chose
128 or 256 threads, and Metal ranges overlap widely. The supported result is
that an explicit width can beat automatic width in some heavy regimes. Picking
one exact width requires a quieter or longer acquisition and should not be
hard-coded from this run.

Across the final capture, 90 logical deterministic preparation requests became
6 physical realizations, while all 675 observation samples executed. Literal
and optimized terminal scientific states were identical and D1–D7 all held.

## Native compiler boundary

The current JIT compiles the full straight-line pointwise integer surface used
by this experiment: packed u8/u16/u32/u64 loads and stores, index, locals,
wrapping arithmetic, bitwise operations, masked shifts, comparisons, and
select. It emits one native element loop and can split disjoint element ranges
over OS threads.

Gather, auxiliary-array, and loop bodies currently return
UnsupportedEvaluator. They never fall back inside the native backend. Extending
native lowering to those validated operations is part of the general-purpose
evaluator compiler milestone.

## Controls

- Identical observation bytes still execute once per sample identity.
- Disabling preparation sharing restores one physical preparation per logical
  request and produces zero pending joins.
- A missing or duplicate workload/configuration/trial identity is rejected.
- A configuration output mismatch is rejected before ranking or replay.
- Extreme Metal threadgroup requests are clamped and checked against reference
  bytes on real hardware.
- Reference, native CPU, and Metal outputs share one digest per workload.
- The native compiler declines unsupported bodies rather than interpreting
  them, so native timing cannot silently include the reference backend.
