# SOMA self-tuning discovery

This is the first non-synthetic Discovery workload: SOMA searches physical
configurations of its own evaluator implementation.

## Question and boundary

The study asks which execution policy wins as evaluator shape changes. Its
candidates cover:

- CPU placement at one and eight threads, run singly or as an epoch;
- Metal placement, singly or in one epoch command buffer;
- reused versus freshly allocated Metal scratch buffers; and
- automatic, 32, 64, 128, and 256-thread Metal threadgroups.

These are physical choices only. Every candidate runs the same generated body
over the same frozen inputs. The CPU reference interpreter remains the semantic
definition; the study is not a comparison with optimized native CPU code.

## Scientific DAG

Each execution configuration is one hypothesis. For each workload it requests
the same two deterministic nodes, `generate-evaluator` and `prepare-input`, then
depends on independent timing observations:

```text
generate evaluator ─┐
                    ├─ observation(config, trial) ─ aggregate/rank
prepare input ──────┘
```

The literal replay executes both preparation requests for every configuration.
The optimized replay realizes each preparation key once. Observation nodes are
never cached: equal durations still represent separate trials and execute
separately. A no-sharing control salts preparation contracts by configuration;
its pending joins fall to zero and physical preparation returns to the logical
count.

## Acquisition protocol

Timing and Discovery replay are separate phases:

1. Generate every evaluator once and install all programs into one Metal
   backend. Changing Metal policy uses `set_tuning`, so candidates share the
   same compiled pipelines.
2. Warm every candidate once for each workload.
3. Acquire candidates in a deterministic rotating order. The default eleven
   trials complete one rotation over the eleven-candidate hardware matrix.
4. Time only collective execution, publication, and freezing. Kernel/input and
   collective construction occur before the clock.
5. Hash all published output bytes outside the timed interval. Any digest
   disagreement across configurations invalidates the capture.
6. Replay the captured observations, unchanged, through literal and optimized
   Discovery executors and require D1–D7 plus identical scientific state.

The report gives median and full observed range. This controls first-use and
order effects and exposes large disturbances; it does not control other system
load, temperature, or frequency. Results are a local regime map, not universal
device constants.

Run it with:

```text
cargo run --release --features metal --example self_tuning_report
```

## M4 Pro regime map

Measured 2026-08-07 on an Apple M4 Pro with 24 GiB RAM, macOS 26.6, Rust 1.92,
release profile, with real Metal dispatch. The table reports the final
eleven-trial capture:

The complete emitted table is retained in
[`measurements/SELF-TUNING-M4-PRO-2026-08-07.txt`](measurements/SELF-TUNING-M4-PRO-2026-08-07.txt).

| Workload | Winning policy | Median | Main boundary |
| --- | --- | ---: | --- |
| 8 ALU, 1 × 1,024 elements | CPU, 1 thread, single | 0.044 ms [0.042, 0.067] | CPU single and epoch tie; the fastest Metal median is 3.9× slower because fixed cost dominates. |
| 256 ALU, 16 × 8,192 elements | Metal, 256 threads, epoch, reused buffers | 1.300 ms [0.382, 3.442] | 26.4× over the scalar interpreter; matched automatic-width epoch submission is 3.8× faster than single Metal collectives. |
| 2,048 ALU, 8 × 131,072 elements | Metal, 256 threads, epoch, reused buffers | 3.848 ms [2.461, 7.250] | 513× over the scalar interpreter; matched automatic-width epoch submission is 2.2× faster than single Metal collectives. |

The large CPU ratios include interpretation overhead and must not be presented
as GPU-versus-native-CPU speedups. The stronger implementation finding is the
within-Metal comparison: configuration matters after placement. Epoch grouping
dominates medium cohorts, and reused buffers improve the heavy automatic-width
median by 1.29×.

The precise threadgroup winner is **not stable evidence**. A preceding
nine-trial capture selected 128 threads for both heavy cells (0.586 ms and
3.918 ms); the final capture selected 256, and its Metal ranges overlap widely.
The supported result is that an explicit width can beat automatic width on
this machine in these regimes. Choosing 128 versus 256 requires a quieter or
longer acquisition and should not be hard-coded from this run.

Across the final capture, 66 logical deterministic preparation requests became
6 physical realizations, while all 363 observation samples executed.
Literal and optimized terminal scientific states were identical and D1–D7 all
held.

## Controls

- Identical observation bytes still execute once per sample identity.
- Disabling preparation sharing produces four physical preparation executions
  instead of two in the focused control and zero pending joins.
- A missing or duplicate `(workload, configuration, trial)` is rejected.
- A configuration output mismatch is rejected before ranking or replay.
- Extreme Metal threadgroup requests are clamped and compared against CPU
  bytes by the real-hardware backend suite.
- CPU and Metal studies both run the same scientific-equivalence checks.
