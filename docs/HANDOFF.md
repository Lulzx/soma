# SOMA — engineering handoff

Everything you need to pick this up. Read §1 and §6 before writing any code;
§6 is the part that is easy to lose and expensive to rediscover.

Repo: https://github.com/Lulzx/soma · ~7,200 lines Rust · no dependencies ·
83 tests · zero clippy lints.

```sh
cargo test
cargo clippy --all-targets
cargo run --example cohort_report      # cohorting vs a persistent FIFO
cargo run --example baseline_report    # vs a hand-written bulk frontier
cargo run --example irregular_report   # occupancy/latency frontiers
cargo run --example regime_map         # where cohorting helps and fails
cargo run --example territory_report   # distribution across territories
```

---

## 1. What the project currently is

SOMA is an **abstract machine** for irregular concurrent computation: persistent
processes, objects, continuations, dataflow readiness, capabilities,
collectives. It is deliberately hardware-neutral — no SIMD width, no device, no
host, no placement. Those belong to *implementations*.

The project changed shape recently and the git history reads misleadingly if you
don't know this. It began as a GPU operating system idea ("replace kernel
launches with persistent device-resident processes"), generalised into SOMA, and
has now been explicitly refocused: **specify the abstract machine first; a GPU OS
becomes one implementation of it later.** Performance work is paused.

Two documents, and they are not equals:

| Doc | Status |
| --- | --- |
| `docs/SOMA-v0.2.md` | **Current.** The semantic specification. Start here. |
| `docs/SOMA-P1.md` | Historical. The original broad Phase-1 contract, still referenced by `§n` markers in code comments. Useful context, but it describes a wider system than the one being built, and its framing is what the refocus moved away from. |

The directory is still named `gpu-os` and the crate `soma`. Harmless, but expect
the mismatch.

---

## 2. Architecture in one pass

```text
src/
  abi/         Fixed-width ABI structs. Ref64 = slot+generation+kind+flags.
               Descriptors for objects, processes, continuations, cohorts,
               futures, messages, capabilities, contracts, traces.
  table.rs     Generational slot table. Slot 0 reserved so NULL never resolves;
               delete bumps generation so stale refs fail.
  kernel/      The machine. mod.rs holds all state; epochs.rs is the epoch
               lifecycle; commit.rs is the only path to runnable/terminal;
               ownership.rs is freeze/transfer; accounting.rs is counters.
  executives/  cpu_scalar.rs — the interpreter. One switch on run_class.
  scheduler/   runnable_bins.rs — double-buffered per-run-class bins.
               cohorts.rs — cohort construction + the shared dispatch-cost model.
  compiler/    frame.rs — little-endian frame encode/decode.
               state_machine_lowering.rs — the Expand example, lowered by hand.
  semantics/   invariants.rs — the executable half of the specification.
  replay/      trace comparison for determinism checks.
  experiments/ measurement only; none of it is part of the machine.
```

**The central design commitment.** A *run class* is the identity of a resume
point, and it is simultaneously the interpreter's dispatch key and the
scheduler's bin key — the same integer. A continuation that yields already names
its next bin, so the scheduler never inspects continuation metadata to decide
what can run together. Most of the system follows from this.

**Continuations exist because preemption is not assumed.** A computation that
must yield ends a continuation and names the next one. Frames are byte blobs in
shared memory, not register state, so a continuation can resume on a different
executor than the one that suspended it.

---

## 3. Where the work stands

### Implemented and tested

- ABI, generational tables, reference validity.
- The deterministic CPU interpreter, step budgets, durable frames.
- Processes, bounded mailboxes with back-pressure, single-assignment futures,
  double-buffered run-class bins, an eight-phase epoch lifecycle, full trace +
  replay comparison.
- Cohort construction with all four partial-cohort policies (§14 of the old
  contract), plus a persistent-FIFO scheduling mode as a baseline.
- The semantic specification, with 11 of 14 invariants machine-checked.

### Named by the model and NOT implemented

Treat these as absent, not nearly-done:

- **Capabilities.** The table is allocated and never consulted. **Any process
  can mutate any object it can name.** `tests/semantics.rs::capability_authority_is_unenforced_and_the_spec_says_so`
  demonstrates it.
- **Ownership enforcement.** `freeze` / `transfer_unique` record intent. Nothing
  stops a write to a frozen object.
- **Channels.** Messaging is per-process mailboxes. `Kind::Channel` is a
  discriminant with no implementation.
- **Collectives, domains, execution contracts, cancellation, supervision.**
  Vocabulary only. `CancelPending` and `Cancelled` are states no transition
  reaches.

---

## 4. What the measurements actually showed

You will be tempted to quote the biggest number. Don't — it's the misleading
one, and the sequence matters more than any single figure.

1. **Cohorting beats a persistent FIFO by 1.85–3.27× useful lane occupancy.**
   True, and against a weak baseline.
2. **Against a competent hand-written bulk frontier, it ties exactly — 1.00× in
   every level-synchronous regime.** When readiness arrives in neat levels, a
   host can sort each level by run class and get the same groups. So finding 1
   on its own oversells the mechanism.
3. **On irregular arrival, cohorting wins properly:** 1.55× occupancy at matched
   latency, or ~9× less waiting at matched occupancy. The mechanism is that a
   host-launched batch has one *global* accumulation window while SOMA holds
   partial cohorts *per run class*. This is the real result.
4. **Distribution across execution territories destroys it** unless placement is
   load-proportional. Blind routing across 64 territories drops occupancy to
   0.385 (a global sort gets 0.960). Class affinity fills cohorts but idles 60
   of 64 territories. Proportional class-to-territory blocks recover both.

**Caveats that must travel with these numbers.** They are *structural bounds*
computed from how continuations group, not hardware measurements — a lane group
spanning `k` run classes is charged `k` masked dispatches because a
uniform-dispatch executive cannot do better; real hardware does worse. The
irregular-arrival result is a **trace-driven policy comparison**: an arrival
trace is generated once and both policies are scored against it, so a node's
ready tick is an input rather than a consequence of when its parent ran, and the
SOMA side is a model of the scheduler's binning rather than the kernel running.
Nothing here touches throughput or scheduler overhead.

---

## 5. Open work, in dependency order

### 5.1 Capabilities (I10) — do this first

The largest gap between what the model claims and what it does, and it
constrains everything after it.

The design decision to make first: **is authority checked at reference
resolution, or at operation?**

- *At resolution* — a `Ref64` is only dereferenceable by a holder of a matching
  capability. Uniform, and every table lookup pays for it.
- *At operation* — `send`, `freeze`, `mutate`, `spawn` each check. Cheaper,
  but the check surface is every operation and it is easy to miss one.

Whichever you choose, add the corresponding I10 checker to
`src/semantics/invariants.rs` and flip the marker in `docs/SOMA-v0.2.md` §5.
There is a test asserting capabilities are *unenforced*; it is meant to fail
when you implement this. Update it rather than deleting it — it should become
the test that enforcement works.

### 5.2 What does a process own? (spec §6.2)

Unresolved and blocking. I8 gives each continuation an exclusive frame and a
process has a state object, but nothing says how concurrent continuations of one
process may touch that shared state. I13 restricts *mutating* continuations to
one per epoch without defining which continuations mutate — the interpreter uses
`ProcessMode` as the answer, which is too coarse.

### 5.3 Failure containment and cancellation (spec §6.3, §6.4)

Both need 5.2 first. Today `Fault` marks a process failed and increments a
counter. Nothing says what happens to its other continuations, queued messages,
or unresolved futures. **A future whose resolver faults is never resolved and
its waiters wait forever** — the progress invariant does not catch this, because
those continuations are not runnable. That is a real hole, not a hypothetical.

### 5.4 Channels and collectives

Currently vocabulary. Collectives matter most: the model claims to cover
cooperative execution shapes and there is no implementation at all.

### 5.5 Validation workloads

Beyond dynamic search: streaming pipelines, actor systems, irregular dataflow.
Choose them to stress **ordering, back-pressure, and failure** — not throughput.
The existing workloads exercise none of those hard.

### 5.6 Minimal IR — deliberately deferred

Do not start until 5.2 and 5.3 are settled. A surface syntax over unresolved
ownership and failure semantics encodes the ambiguity into programs rather than
removing it.

### 5.7 Later: GPU OS as an implementation

The original thesis, parked. `src/experiments/territories.rs` is the beginning
of hardware mapping and is complete and tested as far as it goes.

---

## 6. How this project works — please keep these

These norms are why the results are trustworthy. They cost little and they are
the first thing to erode.

**A checker that cannot fail is not evidence.** Every invariant in
`semantics/invariants.rs` has two tests: the reference model satisfies it, *and*
it catches a state that violates it. Same for every regression test — three of
the kernel bug fixes were verified by reintroducing the bug and confirming the
new test failed. If you add a check, add its failing case.

**Every comparison needs a control that shows no effect.** The cohorting study
reports 1.00× for a single run class; the irregular study reports 1.00× for zero
irregularity and 1.00× for a zero wait budget. Those nulls are what distinguish
measuring the mechanism from measuring the harness.

**Steelman the baseline, then check you did.** The bulk frontier's first version
sorted the frontier and then cut it into fixed lane groups, straddling class
boundaries — SOMA came out 14% ahead. Fixing it to segment per class, which is
what a competent engineer would write, erased the advantage entirely. The near
miss is the reason `dispatch_cost` and `search_step` are *shared* by both sides:
neither can drift into different work or different scoring.

**Conservation before conclusions.** The load-bearing test in the arrival
experiments is that every policy dispatches every arrival exactly once. Any
occupancy figure from a policy that silently drops or double-counts is
meaningless.

**Mark absent things absent.** Do not write a vacuous check to make a table look
complete. The I10 gap is recorded as a hole with a test proving it, which is
more useful than a green tick.

**Determinism is load-bearing.** Every measurement depends on two runs being
comparable. No `HashMap` iteration order in any scheduling decision — sort
first. `trace_snapshot` is the equality used for run comparison.

---

## 7. Traps

- **`retire_process_if_idle`** in `commit.rs` is a linear scan of the
  continuation table per completion. Fine for a reference model, wrong for
  anything real; a production implementation wants a per-process live count.
- **16-bit generations** bound staleness detection rather than guaranteeing it:
  a slot recycled 65,536 times wraps and a stale ref can validate. Documented in
  `abi/refs.rs`. Matters if references are ever persisted.
- **`§n` comments refer to `docs/SOMA-P1.md`**, the historical contract, not to
  the v0.2 spec. The v0.2 spec uses `I1..I14` and `§6.x`.
- **`experiments/` is not the machine.** Nothing there is part of SOMA's
  semantics; it is measurement scaffolding.
- **The step budget check must stay before dispatch** in `epochs.rs`. Faulting
  after commit leaves a faulted continuation enqueued by that commit, which
  violates I7 — this was a real bug.
- **`Complete` is about a continuation, not a process.** A process retires when
  its last continuation does. Getting this wrong strands live continuations on a
  dead process; see spec §6.1.

---

## 8. Suggested first week

1. Read `docs/SOMA-v0.2.md` end to end, then `src/semantics/invariants.rs`
   alongside `tests/semantics.rs`. That pair is the fastest way into the model.
2. Run the examples. The numbers in §4 should reproduce exactly; if they don't,
   something is non-deterministic and that is a bug worth chasing before
   anything else.
3. Break something on purpose — flip a condition in `commit.rs` and confirm the
   invariant checker catches it. That tells you the safety net is real before
   you rely on it.
4. Write the capability design note (§5.1) — resolution vs operation, with the
   check surface enumerated — and settle it before implementing.
