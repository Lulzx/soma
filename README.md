# SOMA

**S**mall-scale SI**M**D **O**S for **m**any-**a**gent systems — a kernel prototype.

> Persistent independent processes can execute efficiently on heterogeneous SIMD
> hardware when their ready continuations are dynamically regrouped into coherent
> physical execution cohorts.

SOMA is an executable kernel contract built to test that sentence. A population of
long-lived, engine-independent processes is modelled as bounded resumable
continuations, and those continuations are regrouped by run class so they can be
dispatched as coherent SIMD cohorts. This repo is the Phase-1 prototype:
deterministic, dependency-free Rust, no GPU yet.

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

## What the prototype shows so far

Binning by run class instead of arrival order raises useful SIMD-lane occupancy
by **1.85–3.27×** on divergent work, eliminating 46–69% of dispatches.
Reproduce with `cargo run --example cohort_report`:

| run classes | width | FIFO occupancy | run-class occupancy | ratio |
| ----------- | ----- | -------------- | ------------------- | ----- |
| 1           | 32    | 0.910          | 0.910               | 1.00× |
| 2           | 32    | 0.455          | 0.843               | 1.85× |
| 4           | 32    | 0.246          | 0.758               | 3.08× |
| 8           | 32    | 0.202          | 0.615               | 3.04× |

The single-class row is the control: with nothing to diverge, cohorting buys
exactly nothing, as it must.

**This is a structural bound, not a hardware measurement.** A lane group holding
`k` run classes is counted as `k` masked dispatches because a uniform-dispatch
executive cannot do better; real hardware can only do worse. It settles the
occupancy limb of the go/no-go criterion (§28.1) and says nothing about the
throughput limb or scheduler overhead — both need the GPU executive and
wall-clock timing that Phase 1 does not have.

## What is implemented

Steps 1–3 and 6 of the contract's evidence-producing path ([§30](docs/SOMA-P1.md)):

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
- **Cohorting and its baseline.** Cohort construction with all four
  partial-cohort policies, plus a persistent-FIFO scheduling mode that changes
  binning and nothing else — which is what makes the comparison above honest.

Not yet built: the GPU SIMD executive, CPU/GPU migration and spill, the bulk
frontier baseline, and the Sokoban workload.

## Quick start

```sh
cargo test                    # 37 tests
cargo run --example cohort_report
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
  experiments/ branching search; the FIFO-vs-cohorting study
```

## Status

Working prototype, `v0.1.0`. The kernel is exercised on its negative paths as
well as its happy ones — mailbox back-pressure, step-budget exhaustion, the
serial-process invariant, re-entry after blocking — because that is where an
epoch lifecycle loses or duplicates work.

What is established: resumable continuations regrouped by run class run
deterministically, and that regrouping is worth a large occupancy factor on
divergent work. What is not: that the factor survives contact with a GPU. The
[contract](docs/SOMA-P1.md) is explicit that SOMA remains a strong but unproven
abstract machine until it does.

## License

MIT.
