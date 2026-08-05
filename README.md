# SOMA

An abstract machine for irregular concurrent computation, with an executable
specification.

> Can persistent processes, object capabilities, continuations, dataflow
> readiness, and collectives provide a coherent general programming model for
> heterogeneous computation?

SOMA is defined without reference to hardware: no SIMD width, no device, no
host, no placement. Those belong to *implementations*, of which the CPU
reference interpreter here is one. The machine is specified in
**[docs/SOMA-v0.2.md](docs/SOMA-v0.2.md)**, and most of that specification is
executable — its invariants are checked against the interpreter after every
transition, so the document cannot drift from the machine without a test
failing.

Deterministic, dependency-free Rust.

## The specification

Fourteen invariants, each marked with what it actually is:

- **checked** — machine-verified by `soma::semantics::invariants` after every
  transition. Eleven of the fourteen.
- **modelled** — implemented and tested, but a property of a transition rather
  than of a state, so not expressible as a predicate.
- **absent** — named by the model and *not implemented*. The machine does not
  have this property.

`tests/semantics.rs` asserts both directions for every checked clause: the
reference model satisfies it, **and** the checker catches a state that violates
it. A checker that cannot fail is not evidence.

The largest honest gap is capability safety (I10). The capability table is
allocated and never consulted, so any process can mutate any object it can name.
There is deliberately no checker for it — a check that cannot fail would
misrepresent the machine as safe — and a test demonstrates the hole concretely.
Channels, collectives, domains, cancellation, and supervision are vocabulary,
not machinery.

Writing the specification immediately broke something real: `Complete` had
conflated "this continuation finished" with "this process is done", so a process
with two live continuations retired when the first completed and the second went
on running against a dead process. See §6.1 of the spec.

## The mechanism

Every resume point of every state machine is a **run class**, and a run class is
simultaneously three things: the queue a ready continuation belongs to, the key
the interpreter dispatches on, and the unit a SIMD cohort is cut from. They are
the same integer. That identity is the whole idea — the scheduler never inspects
continuation metadata to decide what can run together, because a continuation
that yields already names its next bin.

```text
process (persistent, serial)
   │  message / future wakes it
   ▼
continuation ──► run-class bin ──► cohort of W lanes ──► uniform dispatch
   │                                                          │
   └──────────── durable frame (little-endian bytes) ◄─────────┘
```

The frame is a byte blob in shared memory, not register state, so a continuation
can move between executives at any continuation boundary.

## Implementation results

Everything below measures **one implementation strategy** — grouping
continuations by run class so they can be dispatched together — not a property
of the model. The specification neither requires nor mentions it. Read these as
evidence about a scheduler, not about SOMA.

Binning by run class instead of arrival order raises useful SIMD-lane occupancy
by **1.85–3.27×** over a persistent FIFO, eliminating 46–69% of dispatches.
The single-class row is the control: with nothing to diverge, cohorting buys
exactly nothing, as it must.

| run classes | width | FIFO  | run-class | ratio |
| ----------- | ----- | ----- | --------- | ----- |
| 1           | 32    | 0.910 | 0.910     | 1.00× |
| 2           | 32    | 0.455 | 0.843     | 1.85× |
| 4           | 32    | 0.246 | 0.758     | 3.08× |
| 8           | 32    | 0.202 | 0.615     | 3.04× |

**That number is against a weak baseline, and on its own it is misleading.**
A hand-written bulk frontier kernel that sorts each level by run class before
dispatching reaches the same occupancy SOMA does — 0.758 at four classes, tie,
every regime tested:

| implementation       | dispatches | occupancy | host launches |
| -------------------- | ---------- | --------- | ------------- |
| SOMA / persistent FIFO | 185      | 0.246     | 0             |
| SOMA / run-class       | 60       | 0.758     | 0             |
| bulk frontier, naive   | 198      | 0.230     | 6             |
| bulk frontier, sorted  | 60       | 0.758     | 6             |

So on **level-synchronous** work, dynamic cohorting buys nothing a competent
engineer could not get by sorting the frontier by hand.

### Where it does pay: irregular arrival

The thesis needs work whose readiness does *not* arrive in levels — roots
trickling in over time, and siblings becoming ready at different moments because
their heuristics resolve at different times. Both policies then face the same
tension, dispatch now with idle lanes or wait to accumulate, but they resolve it
at different granularities. The bulk frontier's accumulation window is **global**:
every class waits for the same host launch, so the window is set by whichever
class fills slowest. SOMA holds partial cohorts **per run class**, so a class
that has filled dispatches immediately while a sparse one keeps accumulating.

That distinction is worth a lot. At a matched mean wait of one tick, SOMA reaches
0.756 occupancy against the manual batch's 0.488 — **1.55×**. Read the other way,
both reach 0.945 occupancy eventually, but SOMA gets there after 1.2 ticks of
mean waiting and the manual batch needs 11.2 — **9× less waiting for the same
lane efficiency**, with p99 wait 6 ticks against 21.

The advantage appears as soon as there is any irregularity at all, and grows with
it. It is not a narrow corner: it holds across the sweep in
`cargo run --example regime_map`. Two limits are worth stating plainly:

- With **no** irregularity the advantage is exactly 1.00×, reproducing the tie
  above through an entirely separate code path.
- With a **zero** wait budget it is also exactly 1.00× — neither policy can
  accumulate anything. The advantage is bought with waiting; SOMA just buys more
  per tick spent.
- When arrivals per class per tick fall far below the cohort width, both policies
  starve and the advantage collapses back toward 1.0.

Reproduce with `cargo run --example cohort_report`, `--example baseline_report`,
`--example irregular_report`, and `--example regime_map`.

**These are structural bounds, not hardware measurements.** A lane group holding
`k` run classes is counted as `k` masked dispatches because a uniform-dispatch
executive cannot do better; real hardware can only do worse. Both implementations
are scored by the same model and do bit-identical arithmetic. This settles the
occupancy limb of §28.1 and the §28.3 tolerance; it says nothing about
throughput or scheduler overhead, which need the GPU executive and a real clock.

The irregular-arrival result carries one further caveat: it is a **trace-driven
policy comparison**. An arrival trace is generated once and both dispatch
policies are scored against that identical trace, which is how schedulers are
normally compared but does mean a node's ready tick is an input rather than a
consequence of when its parent ran. The SOMA side is a model of the scheduler's
binning, not the kernel executing.

## What is implemented

Steps 1–3, 6 and 7 of the contract's evidence-producing path
([§30](docs/SOMA-P1.md)):

- **ABI and generational tables.** `Ref64` (slot / generation / kind / flags) and
  every descriptor: objects, processes, continuations, run classes, cohorts,
  execution contracts, messages, futures, capabilities, traces. Deleting an
  entity bumps its slot generation, so stale references are always rejected.
- **A deterministic CPU continuation interpreter.** Uniform `dispatch(run_class)`
  under step budgets, over durable byte-blob frames.
- **The runtime.** Bounded ordered mailboxes with back-pressure,
  single-assignment futures, the serial-process invariant, double-buffered
  runnable bins, and an eight-phase epoch lifecycle that commits every side
  effect and emits a full trace for replay.
- **Cohorting and its baselines.** Cohort construction with all four
  partial-cohort policies; a persistent-FIFO scheduling mode that changes binning
  and nothing else; and a level-synchronous bulk frontier kernel in naive and
  sorted variants. All of them are scored by one shared divergence model and run
  one shared `search_step`, so no comparison rests on two implementations
  quietly doing different work.

- **The irregular-arrival experiment.** A trace generator and two dispatch
  policies scored against one identical arrival trace, with a regime map over
  arrival spread, readiness jitter, and class count.

Not yet built: the GPU SIMD executive, CPU/GPU migration and spill, and the
Sokoban workload.

## Quick start

```sh
cargo test                    # 83 tests
cargo run --example cohort_report
cargo run --example baseline_report
cargo run --example irregular_report
cargo run --example regime_map
cargo clippy --all-targets    # zero lints
```

Stable Rust, no dependencies.

## Layout

```text
src/
  abi/         refs, objects, processes, continuations, cohorts,
               contracts, messages, futures, capabilities, traces
  table.rs     generational slot table (generation bump on delete)
  kernel/      tables, epoch lifecycle, commit, ownership, accounting
  executives/  cpu_scalar: the run-class dispatch interpreter
  scheduler/   double-buffered run-class bins; cohort construction
  compiler/    frame encoding; the Expand state-machine lowering
  replay/      trace reader and determinism comparison
  semantics/   executable invariants from the specification
  experiments/ branching search; cohorting study; bulk frontier;
               irregular arrival; execution territories
```

## Status

Reference interpreter plus a draft specification, `v0.1.0`. The kernel is exercised on its negative paths as
well as its happy ones — mailbox back-pressure, step-budget exhaustion, the
serial-process invariant, re-entry after blocking — because that is where an
epoch lifecycle loses or duplicates work.

What is established: resumable continuations regrouped by run class run
deterministically; against a competent hand-written batch that regrouping is
worth nothing on level-synchronous work and a great deal — 1.55× occupancy at
matched latency, or 9× less waiting at matched occupancy — as soon as readiness
stops arriving in levels. Irregularity is the variable the mechanism trades on,
which is what the hypothesis claimed.

What is not established: any of it on hardware. Every number here is a structural
bound computed from how continuations group, and the irregular result is a policy
model rather than the kernel executing. The throughput limb of §28.1 and the
overhead budget of §28.2 remain untouched, and both need the GPU executive.

The [Phase-1 contract](docs/SOMA-P1.md) is explicit that SOMA must permit
results that falsify it (§29) — its purpose is not to protect its original
thesis. So far the cohorting strategy has survived one honest attempt, on one
synthetic workload family.

The current work is not that, though. It is
[the specification](docs/SOMA-v0.2.md): pinning down what the machine *is*
before optimising how it runs. Section 6 of that document lists the ambiguities
the executable model has already exposed and has not yet resolved — what a
process owns, what happens to a failed process's futures and waiters, and what
cancellation means. A surface syntax written before those are settled would
encode the ambiguity rather than remove it.

## License

MIT.
