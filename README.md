# SOMA

**Small-scale SIMD OS for Many-Agent systems** — Phase 1 kernel prototype.

> Persistent independent processes can execute efficiently on heterogeneous SIMD
> hardware when their ready continuations are dynamically regrouped into coherent
> physical execution cohorts.

SOMA is a minimal executable kernel contract that tests that hypothesis. It models
a population of long-lived, engine-independent processes as a set of bounded,
resumable continuations, and dynamically regroups those continuations by run class
so they can be executed as coherent SIMD cohorts. This repository is the Phase-1
prototype: a deterministic, dependency-free Rust implementation of the ABI, the
CPU scalar continuation interpreter, and the process/message/future runtime that
drives them.

## The idea in one picture

```text
processes (persistent, serial)          continuations (schedulable units)
        │                                       │
        ▼                       resumed by messages / futures ▼
   bounded state machine   ────►   grouped by run class (cohorting)
        │                                       │
        ▼                                       ▼
   durable frame (byte blob)            CPU scalar / GPU SIMD lane executive
```

Every *resume point* of every state machine becomes a **run class** — the exact
queue a continuation belongs to. The scheduler never inspects arbitrary
continuation metadata; a yielded continuation already knows its next bin. That
grouping is the seed of cohorting: continuations in one run class can be packed
into a SIMD cohort and executed through one uniform dispatch.

## What Phase 1 implements

This slice covers the first three steps of the evidence-producing path in the
[contract](docs/SOMA-P1.md) (§30):

1. **Fixed ABI references and generational tables.** `Ref64`
   (slot / generation / kind / flags), `AbiHeader`, and every descriptor struct
   (objects, processes, continuations, run classes, execution contracts,
   messages, futures, capabilities, traces). Deleting an entity bumps its slot
   generation before reuse, so stale references are always rejected.
2. **A deterministic CPU scalar continuation interpreter.** Uniform
   `dispatch(run_class)` execution with step budgets. Frames are little-endian
   byte blobs that live in shared memory, so a continuation can later move
   between executives without migrating register state.
3. **Processes, messages, futures, and double-buffered runnable bins.** Bounded,
   ordered mailboxes; single-assignment futures that awaken waiters; a
   serial-process invariant (at most one mutating continuation per process
   running at a time); and an epoch lifecycle that commits every side effect and
   emits a full `TraceEvent` stream for deterministic replay.

**Deferred to later slices:** the GPU SIMD-lane executive, real cohort
construction, CPU/GPU migration and spill, the FIFO / bulk-frontier baselines,
and the Sokoban-style workload.

## Quick start

```sh
cargo build          # clean, no warnings
cargo test           # 19 tests
cargo clippy --all-targets   # zero lints
```

Requires a stable Rust toolchain. No external dependencies.

## The `Expand` example (§22)

The source model lowers into three resume points, each its own run class:

```text
Expand.resume_0  Receive request; store in frame; spawn heuristic; await.
Expand.resume_1  Load heuristic result; generate a bounded group of moves.
Expand.resume_2  Finish child creation; send reply; complete.
```

The interpreter test drives a process through the full lifecycle — receive a
message, await a future across an epoch boundary, yield to the next resume point,
spawn children, and reply — then asserts the run is deterministic by comparing the
trace of two identical runs.

## Layout

```text
src/
  abi/        ABI structs: refs, objects, processes, continuations,
              contracts, messages, futures, capabilities, traces
  table.rs    Generational slot table (one per kind, generation-bump on delete)
  kernel/     Tables + epoch lifecycle + commit + ownership transitions
  executives/ cpu_scalar: the run-class dispatch interpreter
  scheduler/  Double-buffered runnable bins per run class
  compiler/   Frame byte-blob encoding; the Expand state-machine lowering
  replay/     Trace reader / determinism comparison
  experiments/ Synthetic branching-search workload with control knobs
tests/        ABI, Expand end-to-end, dynamic-search determinism
```

## Status

Working prototype, `v0.1.0`. Tests cover ABI reference validity, the full Expand
lifecycle, and branching-search termination + determinism. It proves, on one CPU
executive, that resumable continuations regrouped by run class run
deterministically — the mechanism cohorting depends on. The GPU executive and the
go/no-go measurements (§28) are the next slices.

## License

MIT.
