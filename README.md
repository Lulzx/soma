# SOMA

SOMA is an abstract machine for irregular concurrent programs. Processes persist,
wait on messages or futures, and resume through bounded continuations. Ready
continuations are grouped by run class so a SIMD executor can run identical code
across its lanes.

The model does not mention a GPU, SIMD width, placement, or a host. Those are
implementation choices. This repository contains the semantic specification, a
deterministic Rust interpreter, executable invariant checks, and scheduler
experiments.

Stable Rust. No dependencies.

## Run it

```sh
cargo test
cargo clippy --all-targets

cargo run --example cohort_report
cargo run --example baseline_report
cargo run --example irregular_report
cargo run --example regime_map
```

Read [docs/SOMA-v0.2.md](docs/SOMA-v0.2.md) for the current machine. The older
[Phase-1 contract](docs/SOMA-P1.md) records the broader GPU-oriented design that
preceded it.

## Machine model

A process is persistent but does not execute directly. A continuation contains
one bounded segment of work and a durable frame. When it yields, it names the
run class of its next resume point.

```text
process
   │ message or future
   ▼
continuation -> run-class bin -> cohort of W lanes -> dispatch(run_class)
   │                                                     │
   └---------------- durable byte frame <---------------┘
```

The run class is both the scheduler bin and the interpreter dispatch key. The
scheduler therefore does not inspect a continuation to decide which code it
runs. The continuation has already made that decision.

Frames are little-endian byte blobs in shared memory, not saved registers. An
executor may resume a continuation created by another executor at any
continuation boundary.

## Specification

The specification defines fourteen original invariants. Each clause has one of
three states:

- `checked`: evaluated against interpreter state after each transition
- `modelled`: enforced by transition code and tests, but not expressible as a
  state predicate
- `absent`: named by the model and not implemented

Every checked invariant has a positive test and a negative test. The negative
test constructs an illegal state and proves that the checker rejects it. A
checker that cannot reject anything proves nothing.

Capability spaces, creator genesis, derivation, attenuation, and the structural
I10a/I10b checks exist. Every right used by a reachable operation is enforced
at use, including expiry, object-version, range, and parent-chain checks.
Authority decisions and governed effects are traced, and I10c rejects any
effect without an adjacent matching grant. Channels, collectives, domains,
cancellation, and supervision are still absent.

The specification has already caught a lifetime bug. `Complete` once retired a
process when one continuation finished. A second live continuation could then
run against the dead process. `Complete` now applies to a continuation, and the
process retires only when no live continuation remains.

## Scheduler result

These numbers test a scheduler policy, not the SOMA semantics.

Grouping continuations by run class raises useful lane occupancy by 1.85 to
3.27 times over a persistent FIFO and removes 46 to 69 percent of its
dispatches.

| run classes | width | FIFO | run-class | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 0.910 | 0.910 | 1.00x |
| 2 | 32 | 0.455 | 0.843 | 1.85x |
| 4 | 32 | 0.246 | 0.758 | 3.08x |
| 8 | 32 | 0.202 | 0.615 | 3.04x |

The FIFO is a weak baseline. A bulk frontier sorted by run class reaches the
same occupancy on level-synchronous work.

| implementation | dispatches | occupancy | host launches |
| --- | ---: | ---: | ---: |
| SOMA, persistent FIFO | 185 | 0.246 | 0 |
| SOMA, run-class bins | 60 | 0.758 | 0 |
| bulk frontier, naive | 198 | 0.230 | 6 |
| bulk frontier, sorted | 60 | 0.758 | 6 |

Dynamic cohorting has no advantage when readiness arrives in complete levels.
Sorting the frontier produces the same groups.

Irregular arrival changes the comparison. A host-launched batch has one global
accumulation window. SOMA accumulates each run class independently. A full class
can dispatch while a sparse class continues waiting.

At a mean wait of one tick, SOMA reaches 0.756 occupancy while the manual batch
reaches 0.488, a ratio of 1.55. At roughly 0.945 occupancy, SOMA waits 1.2 ticks
on average and the manual batch waits 11.2. The p99 waits are 6 and 21 ticks.

The controls are useful:

- With one run class, both policies produce 1.00x.
- With no arrival irregularity, both policies produce 1.00x.
- With a zero wait budget, both policies produce 1.00x.
- At arrival rates far below cohort width, both policies starve and the ratio
  approaches 1.0.

The figures are structural bounds computed from cohort composition. They are
not timings. A lane group containing `k` run classes costs `k` masked
dispatches. Both sides use the same cost model and perform bit-identical work.
The irregular-arrival study replays one generated trace against both policies,
so ready time is an input to that experiment. GPU throughput and scheduler
overhead remain unmeasured.

## Repository

```text
src/
  abi/          references and fixed-width descriptors
  table.rs      generational slot table
  kernel/       machine state, epochs, commit, ownership, accounting
  executives/   scalar continuation interpreter
  scheduler/    run-class bins and cohort construction
  compiler/     frame encoding and hand-written state-machine lowering
  replay/       trace reader and deterministic comparison
  semantics/    executable invariants
  experiments/  scheduler studies and baselines
```

Implemented now:

- Generational references and tables
- Bounded ordered mailboxes with back-pressure
- Single-assignment futures
- Durable continuations with step budgets
- Double-buffered runnable bins
- An eight-phase epoch with committed effects and replay traces
- Four partial-cohort policies, a persistent FIFO, and two bulk-frontier
  baselines
- Trace-driven irregular-arrival and execution-territory experiments
- Actor-relative capabilities with operation checks and trace-checked effects
- Capability-derived unique/frozen object ownership
- A domain-neutral dynamic constraint-search validation workload

Not implemented:

- Channels and collectives
- Cancellation and supervision
- GPU execution, CPU/GPU migration, and spill

The current job is to finish the machine semantics before adding a surface
language or GPU backend. Open questions include process ownership, the fate of a
failed process's futures and waiters, and cancellation. Encoding those questions
in an IR would only make the ambiguity harder to remove.

## License

MIT.
