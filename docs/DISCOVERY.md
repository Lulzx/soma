# SOMA Discovery

SOMA Discovery is an application-level execution model for research workloads.
It does not add hypotheses or evidence to the SOMA ABI. A deterministic
`DiscoveryTrace` is replayed through the existing evaluator backend boundary,
so the logical research process is held constant while execution policy
changes.

## Semantic model

The graph has four node kinds: deterministic `Derivation`, `Aggregate`, and
`Decision` nodes, plus independent `Observation` samples. Deterministic nodes
receive a SHA-256 `ExperimentKey` over the node kind, operation, module digest,
evaluator, content digests of both inputs, input shapes, parameters, execution
contract, and seed. Runtime references are never hashed. An observation has a
sample identity for provenance, but the registry categorically refuses to
cache it.

The optimized replay has three transformations:

1. A pending/ready registry joins concurrent requests for one semantic key and
   serves later requests from the frozen result.
2. Hypotheses hold interest in requested nodes and their transitive
   dependencies. Rejection or explicit withdrawal removes that interest;
   pending work is cancelled only when its consumer set becomes empty.
3. Compatible pointwise evaluations are concatenated into one physical
   `BatchEvaluate` request and their outputs are partitioned back by byte
   segment. A body containing `Gather` or `GatherAux` is `LocalGather` and is
   never concatenated, because concatenation would change its index space.

The literal executor evaluates every request separately and ignores rejection
for scheduling. Both executors reconstruct terminal hypotheses, evidence,
dependency provenance, and output digests. That terminal scientific state is
the equivalence boundary.

## Executable invariants

`discovery::invariants` checks D1 semantic-key soundness, D2 observation
multiplicity, D3 single physical realization, D4 interest-preserving
cancellation, D5 fused-output equivalence, D6 scientific equivalence, and D7
accounting conservation. Tests include zero-duplication, zero-rejection,
evaluator-heterogeneity and zero-sharing controls, two identical independent
observations, and a gathering body that must remain unfused.

## Running it

```text
cargo run --release --example discovery_report
cargo run --release --features metal --example discovery_report
cargo run --release --example discovery_regime_map
cargo run --release --features metal --example discovery_bench
cargo run --release --features native,metal --example self_tuning_report
```

The single replay report includes logical and physical work, cache and pending
joins, cancellation, dispatch and command-buffer counts, byte traffic, peak
pending memory, compute compression, elimination, batch compression, and a
smoke timing. Do not use that timing as a performance result.

`discovery_bench` is the performance harness. It sweeps the same 4 duplication
rates × 3 pruning rates × 3 evaluator-class counts × 3 element counts as
`discovery_regime_map`, on both the CPU reference backend and real Metal when
enabled. Each cell has two warmups and nine measured repetitions by default.
Acquisition alternates literal-first and optimized-first order, reuses warmed
backend instances, and emits p10/median/p90 plus every raw nanosecond sample as
TSV. `SOMA_DISCOVERY_WARMUPS` and `SOMA_DISCOVERY_REPETITIONS` can override the
protocol. A zero repetition count is rejected.

Every warmup and measured pair is checked immediately with D1-D7; a failure
invalidates the sweep instead of producing a timing row. Crossovers are
observations, not fitted estimates: for each duplication/pruning/class tuple,
the report gives the smallest element count in the measured `{1, 64, 1024}`
grid where the candidate median is at least 15% lower. `not-observed` means
only that no such point occurred in this finite grid.

## Local repeated-release capture

The following is a reproducible local capture from 2026-08-07, run with the
command above at revision `6009b3c` plus this benchmark change: Apple M4 Pro
(12 CPU cores, 16 GPU cores), 24 GiB RAM, macOS 26.6 build 25G5028f, Rust
1.92.0, release profile. The machine was not isolated from scheduler activity,
temperature, or frequency changes. CPU and Metal sweeps ran sequentially, so
their direct comparison is not an interleaved hardware experiment.

Selected boundary cells below show milliseconds as p10 / median / p90; the
harness prints all 216 cells and their raw samples.

| Backend and regime | Elements | Literal | Optimized | Median literal / optimized |
| --- | ---: | ---: | ---: | ---: |
| CPU, duplicate 0.00, prune 0.00, 16 classes | 1 | 0.075 / 0.075 / 0.076 | 0.131 / 0.131 / 0.134 | 0.57× |
| CPU, duplicate 0.00, prune 0.00, 16 classes | 64 | 0.258 / 0.260 / 0.614 | 0.356 / 0.361 / 0.388 | 0.72× |
| CPU, duplicate 0.00, prune 0.00, 16 classes | 1,024 | 3.046 / 3.723 / 4.377 | 3.804 / 6.305 / 8.319 | 0.59× |
| CPU, duplicate 0.75, prune 0.50, 16 classes | 1 | 0.075 / 0.076 / 0.082 | 0.113 / 0.114 / 0.123 | 0.67× |
| CPU, duplicate 0.75, prune 0.50, 16 classes | 64 | 0.245 / 0.246 / 0.259 | 0.264 / 0.265 / 0.301 | 0.93× |
| CPU, duplicate 0.75, prune 0.50, 16 classes | 1,024 | 2.764 / 2.873 / 3.030 | 2.523 / 2.668 / 2.751 | 1.08× |
| Metal, duplicate 0.00, prune 0.00, 16 classes | 1 | 0.568 / 0.592 / 1.068 | 0.513 / 0.528 / 0.701 | 1.12× |
| Metal, duplicate 0.00, prune 0.00, 16 classes | 64 | 0.619 / 0.662 / 0.688 | 0.636 / 0.687 / 1.171 | 0.96× |
| Metal, duplicate 0.00, prune 0.00, 16 classes | 1,024 | 2.031 / 2.323 / 3.056 | 2.772 / 2.998 / 3.989 | 0.77× |
| Metal, duplicate 0.75, prune 0.50, 16 classes | 1 | 0.883 / 0.951 / 0.987 | 0.585 / 0.627 / 0.867 | 1.52× |
| Metal, duplicate 0.75, prune 0.50, 16 classes | 64 | 0.647 / 0.700 / 0.950 | 0.601 / 0.674 / 0.951 | 1.04× |
| Metal, duplicate 0.75, prune 0.50, 16 classes | 1,024 | 1.665 / 1.981 / 2.575 | 2.161 / 2.226 / 2.666 | 0.89× |

All nine samples in all 216 cells passed D1-D7. At the 15% margin, optimized
CPU replay crossed literal CPU replay at 1,024 elements in 8 of the 36
duplication/pruning/class slices and did not cross in the other 28. Optimized
Metal crossed literal Metal at 1 element in 34 slices and did not cross in 2;
that early result reflects avoided Metal submissions, not greater per-element
throughput, and several larger cells reverse it. Comparing optimized backends,
Metal crossed the CPU reference at 1,024 elements in 22 slices and had no
observed crossover in 14. These counts describe this capture and this sparse
grid only. The wide ranges in several rows are direct evidence that exact
medians and individual slice assignments should not be treated as stable
device constants.

The first non-synthetic target is SOMA itself. `self_tuning_report` searches
reference/native CPU thread count and epoch grouping plus Metal placement,
command grouping, scratch-buffer reuse, and threadgroup width. Evaluator construction and input
preparation are deterministic shared nodes; each wall-clock trial is an
independent `Observation`. The acquisition phase runs once, records output
digests and timings, and the literal and optimized executors replay those exact
bytes. See [SELF-TUNING.md](SELF-TUNING.md) for the protocol and M4 Pro regime
map.

## Boundary with concurrent SOMA execution

Discovery uses the existing CPU and Metal `BatchBackend` implementations,
including Metal's one-command-buffer epoch submission. It does not weaken
SOMA's abstract-machine semantics and it does not claim arbitrary SOMA lanes
now execute concurrently. The speculative snapshot/validate/fallback executive
remains a separate runtime phase; Discovery provides its intended
mostly-independent workload and equivalence oracle.
