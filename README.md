# SOMA

SOMA is an abstract machine for irregular concurrent programs. Processes persist,
wait on messages or futures, and resume through bounded continuations. Ready
continuations are grouped by run class so a SIMD executor can run identical code
across its lanes.

The model does not mention a GPU, SIMD width, placement, or a host. Those are
implementation choices. This repository contains the semantic specification, a
deterministic Rust interpreter, executable invariant checks, and scheduler
experiments.

Stable Rust. The semantic core has no dependencies. The optional macOS Metal
backend is enabled with `--features metal`.

**[See it run](https://lulzx.com/soma/)** — ten thousand independent ants,
scheduled two ways.

## Run it

```sh
cargo test
cargo clippy --all-targets
cargo test --all-features                 # includes a real Metal dispatch on macOS
cargo clippy --all-targets --all-features

cargo run --example cohort_report
cargo run --example baseline_report
cargo run --example irregular_report
cargo run --example regime_map
cargo run --example streaming_report
cargo run --example supervision_report
cargo run --example multi_input_report
cargo run --example ant_colony_report      # cohorting on a population of agents
```

## The ant colony

**Live demo: [lulzx.com/soma](https://lulzx.com/soma/)**

`experiments/ant_colony.rs` is a population of persistent processes rather than a
computation: ten thousand ants across a hundred colonies, each holding its own
state, each deciding its own next behaviour, none of them synchronised with any
other. An ant that finds food yields to `ANT_CARRY_FOOD`; an ant that walks into
a rock yields to `ANT_AVOID_OBSTACLE`. Neither knows the other exists, and the
scheduler groups them anyway — because the run class an ant names *is* the bin
it lands in.

To reproduce it locally:

```sh
cargo run --release --example ant_colony_report   # the measurement
cargo run --release --example ant_colony_trace    # write viz/data/*.jsonl
cd viz && python3 -m http.server                  # then open localhost:8000
```

The page is published from `viz/` by `.github/workflows/pages.yml`, which
regenerates the traces on every deploy and runs the test suite first — so it
cannot show numbers that disagree with the code that produced them.

The same population, the same seed, the same world, binned two ways — ten
thousand ants on a 320x320 grid over 260 epochs:

| | dispatches | useful lane occupancy | full cohorts |
| --- | --- | --- | --- |
| persistent FIFO | 291,312 | 0.282 | 0 |
| run-class bins | 83,153 | 0.987 | nearly all |

3.50x occupancy at 71% fewer dispatches. The gap widens with the population:
at a few hundred ants a run class often cannot fill 32 lanes, and at ten
thousand it always can. The control that makes it mean anything
is `simulated_identical_world`: both runs deliver the same food, lay the same
trails, and leave every ant in the same place. Only the binning differs. The
width-1 null control reports 1.00x, because at one lane per dispatch binning
cannot matter.

Occupancy is a **structural** quantity here, as everywhere else in this
repository — computed from how continuations group, not measured on silicon.

Clicking *predator* fails an entire colony at epoch 150. One hundred processes
fail, one hundred terminal notices are delivered, the population settles at
9,900, and the other ninety-nine colonies are untouched -- with occupancy and
dispatch count unchanged, because containment is not a scheduling event.

`experiments/ant_scoring.rs` lifts movement into a batch evaluator that runs on
real Metal hardware under `--features metal` — including the neighbourhood
*gather*, which used to stay on the CPU. Sensing reads the trail grid, and that
took two things: `gather`/`index`, so a body can read an element other than its
own, and a second array binding, so it can name an array other than the one it
is iterating. `ant_sense_and_score` does the eight reads and the choice in one
dispatch, checked against an independent host-side reference as well as against
the CPU interpreter. The colony run itself still senses on the host; wiring the
collective into the ant step is an executive change and is not made.

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

The specification defines twenty-four numbered invariants. Each clause has one of
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
effect without an adjacent matching grant. Failure containment, direct-child
supervision, and cooperative cancellation settle sibling work, futures,
mailboxes, channels, collectives, and waiters. Supervised child exits produce
reliable typed terminal notices and wake a waiting supervisor continuation.
Relationships may also escalate a failed child by failing its direct supervisor
after notice delivery, while leaving sibling branches unaffected.
Bounded restart creates a fresh replacement identity from the registered entry
template and escalates when retries are exhausted. Atomic all-input channel
receive supports irregular joins without consuming a partial input set.
First-class bounded channels and the `BatchEvaluate` collective
are implemented. Logical domains enforce authority and process-creation quotas;
hardware-neutral execution contracts bound continuation steps and frame bytes.
A small textual module surface loads immutable evaluator manifests and links
collectives to them. I17 checks those links. A physical backend boundary can
execute a batch on CPU or Metal, spill unavailable or underfilled work to CPU,
and record collective-boundary placement changes without exposing kernel state
to either backend.

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
  table.rs      partitioned generational slot table
  kernel/       machine state, epochs, commit, ownership, accounting
  executives/   scalar interpreter and physical batch backends
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
- Explicit continuation state-access declarations with checked one-mutator
  scheduling
- Contained process failure and cooperative cancellation with terminal futures
  and waiter wakeup
- First-class bounded channels with capability-checked send, receive, close,
  back-pressure, and kernel-held payload authority
- A generic `BatchEvaluate` collective over frozen input/output arrays with a
  single-assignment completion future
- A domain-neutral dynamic constraint-search validation workload
- A generic bounded streaming-graph workload that measures back-pressure and
  proves committed FIFO data survives producer failure
- A minimal hardware-neutral evaluator IR with frozen-array schemas and resume
  points, connected to `BatchEvaluate` creation
- Gathering bodies: `index` names an element's own position and `gather` reads
  any element of the frozen input array, so stencils and permutations lower to
  both backends. Reads are confined to the frozen input, never the output, so a
  gather does not make the result depend on the order lanes run in; an
  out-of-range index clamps rather than faulting, keeping bodies total
- Direct parent/child supervision with reliable completion, failure, and
  cancellation notices, deterministic waiter wakeup, and opt-in failure
  escalation or bounded restart with replacement lineage
- Atomic multi-input channel receive and a skew/failure-controlled join workload
- Logical domains with actor-relative authority and process-creation quotas
- First-class execution contracts that enforce step and frame-byte bounds
- A minimal textual module surface, immutable loaded evaluator manifests, and
  module-linked `BatchEvaluate` creation
- A backend-neutral batch execution boundary with CPU spill and
  collective-boundary migration accounting
- An optional real Apple Metal compute backend, validated against the CPU
  result and the semantic invariant checker

Not implemented:

- A general-purpose language or compiler for arbitrary evaluator bodies. Bodies
  gather, loop, and read a second bound array now — counted `repeat`, early
  `breakif`, mutable locals, and `gatheraux` against a second frozen array, with
  a validation-time bound on the unrolled length so totality and the step budget
  survive. There are still no calls, no floating point, and no surface syntax
  above the `op` lines
- A concurrent, device-resident scheduler/executive, or a distributed backend.
  The semantics are ready for one and nothing uses it: admission is
  order-independent (I22), the trace's order is recoverable from event position
  rather than a shared clock (I23), allocation is partitioned so lanes need no
  shared allocator, and commit is canonical — an epoch runs every lane, then
  applies what they produced in plan order, so nothing an epoch commits depends
  on the order its lanes ran (I24, I25). Lanes are reorderable and the executive
  reorders them — reversed or permuted per epoch, checked to produce the same
  run and the same commit sequence — but it does so on one thread. The only
  threads in `src/` evaluate a batch's elements in parallel, which the body
  language's purity rules already paid for; the scheduler itself is sequential
- Scheduler-overhead and end-to-end migration benchmarks. The batch backend
  itself is measured on hardware — CPU against Metal from 32 to 4M elements,
  where a Metal call's fixed cost goes, what a published cohort costs off-GPU,
  and cost against accumulated state — see [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
  Whether cohorting beats a bulk schedule is not among those numbers; it is
  still the structural model above

Every entity and invariant named by the v0.2 machine is now implemented. The
remaining work extends the completed semantic core beyond the deliberately
small module surface and collective-level Metal implementation.

## License

MIT.
