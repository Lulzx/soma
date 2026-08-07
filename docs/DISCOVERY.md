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
cargo run --release --features metal --example self_tuning_report
```

The report includes logical and physical work, cache and pending joins,
cancellation, dispatch and command-buffer counts, byte traffic, peak pending
memory, compute compression, elimination, batch compression, and wall time.
The regime map sweeps duplication, pruning, evaluator heterogeneity, and batch
size.

On the default trace used during implementation (2,184 logical requests), the
optimized replay performed 910 physical evaluations in 12 dispatches and all
D1-D7 checks held: 2.40× compute compression and 58.33% elimination. One local
release run completed in 11.2 ms versus 12.7 ms for literal CPU replay; real
Metal completed in 10.9 ms versus 27.5 ms and used three command buffers. These
single-run observations are smoke measurements, not a performance claim.
Crossover mapping and repeated release measurements, rather than structural
ratios alone, are what the experiment is designed to establish.

The first non-synthetic target is SOMA itself. `self_tuning_report` searches
CPU thread count and epoch grouping plus Metal placement, command grouping,
scratch-buffer reuse, and threadgroup width. Evaluator construction and input
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
