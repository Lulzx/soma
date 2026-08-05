# SOMA v0.3 scope

**Status:** scope, not specification. Nothing here is implemented. Clauses are
proposals; the numbering (I18–I23) is reserved, not earned.

v0.2 closed the semantic core: every entity and invariant it names is
implemented and machine-checked, and `docs/SOMA-v0.2.md` §5 records no `[absent]`
clause. v0.3 is the first version where the work is not "finish the model" but
"prove the model survives contact with real execution".

Four things are named as future work by v0.2 §7 and `docs/HANDOFF.md` §8. This
document scopes them, plus the carried debts that block them.

---

## 0. The shape of v0.3

Three of the four items are implementation work that the current specification
cannot admit, because of one clause:

> **§1.2.** Two runs are **equivalent** when their traces are equal. The model
> defines no other equivalence.

A persistent device scheduler runs continuations concurrently. A multi-node
implementation runs them on machines that do not share a clock. Neither can
produce a trace equal to the reference interpreter's, because trace equality is
defined over a total order on logical time and the reference interpreter's total
order is an artifact of running one continuation at a time. Under §1.2 as
written, **every parallel implementation of SOMA is non-conforming by
construction.**

So v0.3 has exactly one specification deliverable and three implementation
deliverables that depend on it:

| | Deliverable | Kind | Depends on |
| --- | --- | --- | --- |
| **S** | Weaken §1.2 to an equivalence a concurrent implementation can satisfy | spec | — |
| **A** | General evaluator bodies and a compiler | impl + spec | — |
| **B** | Persistent device-resident scheduler | impl | S |
| **C** | Distributed / multi-node implementation | impl | S, and the ABA debt (§6.2) |
| **D** | Performance work on real hardware | measurement | A, B |

A is independent and can start immediately. B and C are both blocked on S. D is
blocked on both A and B, because there is nothing worth measuring until a real
evaluator body runs under a real scheduler.

**Recommendation:** land S + A + B as v0.3 and hold C for v0.4. C is the long
pole by a wide margin — it needs a network transport, a failure model that
distinguishes node loss from process failure, and a distributed capability space
— and none of that is derisked by the other three. Scoping it here anyway, since
the ask was for all four, but the sequencing in §7 assumes it slips.

---

## 1. S — the equivalence relation

### 1.1 What is wrong today

`trace_snapshot()` (`src/kernel/mod.rs:470`) produces a vector of rows compared
with `==` by `replay::same_trace`. Every experiment's determinism control is that
comparison. It is the right check for what it currently checks — two runs of the
sequential interpreter with the same input — and it caught real bugs. It is
simply the wrong relation for an implementation that does not serialize.

The failure is not subtle. Two cohort lanes of the same run class have no
semantic order between them. The reference interpreter emits their
`ContinuationStarted` events in lane order because a `for` loop must pick one.
A device scheduler that runs the cohort in one dispatch has no lane order to
report. The traces differ, and nothing is wrong.

### 1.2 What replaces it

**Proposed: trace equality becomes trace *refinement* modulo the happens-before
order.**

Define a partial order ≺ on trace events from the semantic dependencies already
in the model:

- events of the same continuation are ordered by logical time
- `FutureResolved(f)` ≺ every `ContinuationReady` it caused
- `MessageSent` ≺ its `MessageReceived`; `ChannelSent` ≺ its `ChannelReceived`
- `AuthorityGranted` ≺ its adjacent `AuthorityEffect` (already required by I10c)
- child terminal event ≺ its `SupervisionNotified`
- `CollectiveCreated` ≺ `CollectiveCompleted` for the same collective
- epoch boundaries order everything across them

An implementation conforms when its trace is a **linear extension of ≺ that
projects to the same per-entity event sequences** as the reference run. Trace
equality becomes the special case where ≺ happens to be total.

This is a real weakening and it costs something: two conforming implementations
may now disagree on an observable ordering. That is the price of admitting
parallelism, and it is better paid explicitly in the spec than paid silently by
an implementation that claims conformance it cannot demonstrate.

### 1.3 New clauses

**I18. Schedule conformance [proposed, checked].** An implementation's trace is
a linear extension of the happens-before order ≺, and for every entity the
projection of the trace onto that entity's events equals the reference run's
projection.

**I19. Placement neutrality [proposed, checked].** The semantic projection of a
trace is independent of placement. Running the same program with different
territory assignments, cohort widths, or backend selections yields traces that
are I18-equivalent to each other.

I19 is the control `docs/HANDOFF.md` §6 demands, applied to placement: if
changing where work runs changes what a program observes, the placement layer
has leaked into the semantics.

### 1.4 Exit criteria

- ≺ is computed from a trace by a checker in `src/semantics/`, not by hand.
- `replay::same_trace` is retained for the sequential interpreter and joined by
  `replay::conforms_to`, which checks I18.
- I18 and I19 each have a fault-injection test. For I18: reorder two events that
  ≺ actually orders, and confirm rejection. A checker that accepts every
  interleaving is not a checker.
- Every existing determinism control still passes under the weaker relation.

### 1.5 Risk

The weakened relation is easier to satisfy, so it is easier to satisfy
vacuously. If ≺ is too sparse, I18 accepts implementations that are genuinely
wrong. The fault-injection test in the exit criteria is the only thing standing
between a useful relation and a decorative one, and it should be written before
the checker, not after.

---

## 2. A — general evaluator bodies

### 2.1 What exists

`src/compiler/ir.rs` names evaluators. It does not describe them. A
`BatchEvaluator` carries an id, a name, an element stride, and two resume
points. There is no body.

Both backends prove this by ignoring the evaluator entirely:

- `src/executives/batch.rs:57` — `_evaluator_id`
- `src/executives/metal.rs:70` — `_evaluator_id`

Each hardcodes `2*x + 1`. `tests/metal_backend.rs` compares CPU and Metal
output and passes, which is meaningful — but it demonstrates that two
hardcodings of the same constant agree, not that a compiler is correct.

I17 checks that a collective's evaluator id and stride match its module's
manifest. Nothing checks that the *function* a backend applies matches the one
the module names. A backend can return arbitrary bytes and every invariant
holds.

### 2.2 Scope

An evaluator body language, and lowering to both backends from one source.

Deliberately small, because the point is placement-independent publication, not
language design:

- pure, total, element-wise over one frozen input element
- fixed-width integer and float scalars; no allocation, no control flow beyond
  bounded `select`; no loops in v0.3
- element layout declared by the module, so stride is derived rather than
  asserted
- typed: a body that reads outside its declared element is a validation error,
  not a runtime fault

The extension after v0.3 is loops and reductions. Not now — a reduction is a
different collective, not a different body.

### 2.3 New clause

**I20. Backend agreement [proposed, checked].** For a given evaluator and a
given frozen input, every backend that claims to realize that evaluator produces
identical output bytes. A backend that cannot realize a body returns
`UnsupportedEvaluator` rather than an approximation.

This closes the hole above. It also fixes the `CpuReferenceBackend` name, which
currently promises a reference semantics it does not define: under I20 the CPU
backend becomes the definition and the accelerator is checked against it.

Floating point makes I20 sharp rather than pedantic. If a body admits `f32`,
either the spec requires bit-identical results across backends — which
constrains what a GPU may do — or I20 weakens to a declared tolerance, which
makes it much less useful as an invariant. **Recommendation: integer-only bodies
in v0.3**, and defer float until there is a reason to pay for it.

### 2.4 Exit criteria

- A module source file declares a body; one compiler lowers it to the scalar
  interpreter and to MSL.
- At least three distinct bodies, one of which is not expressible as `a*x + b`,
  so the test cannot pass by coincidence.
- I20 with fault injection: a deliberately divergent backend is rejected.
- The `2*x + 1` example is expressed in the body language and the hardcoded
  paths are deleted. If they survive, the compiler is not load-bearing.

---

## 3. B — persistent device-resident scheduler

### 3.1 What exists

`src/kernel/epochs.rs` is the scheduler. It admits, cohorts, executes, and
commits on the host, one continuation at a time
(`epochs.rs:119-126`). `src/experiments/territories.rs` models placement across
territories but does not execute anything. The Metal path
(`src/executives/metal.rs`) dispatches a single collective and blocks on
`wait_until_completed` — a host-driven kernel launch, which is precisely the
thing the project's original premise wanted to remove.

### 3.2 Scope

Move the epoch loop onto the device: bins, admission, cohort construction, and
dispatch resident in device memory, with the host supplying external input and
observing the trace.

What must hold, and why each is hard:

- **Admission (I13)** currently claims a per-process mutable slot with a host
  `HashSet` (`epochs.rs:44`). On device this is a concurrent claim and must be
  deterministic, so it cannot be "whoever gets there first".
- **Commit** is the sole path to `Runnable` (v0.2 §3.4), which is what makes I7
  checkable. A device commit must preserve that exclusivity across concurrent
  writers or I7 becomes unenforceable.
- **The step-budget check must precede dispatch** (`epochs.rs:204`, and
  `HANDOFF.md` §7 records the bug from getting this wrong). Concurrency does not
  relax this.
- **Trace emission** becomes a concurrent append. Logical time must still
  satisfy I11 and now also I18.

### 3.3 What is explicitly not in B

Preemption. The model does not assume it (v0.2 §1.1) and adding it to the
scheduler before adding it to the model would be backwards.

### 3.4 Exit criteria

- An epoch runs to completion with no host round-trip inside it.
- I19 holds: the device run and the CPU run are I18-equivalent.
- Every existing test passes against the device scheduler, unchanged. If a test
  needs modification to pass, that is a semantic difference and it needs an
  explanation in the spec, not a patch to the test.
- The `retire_process_if_idle` debt (§6.1) is paid, because a linear scan per
  completion is not survivable here.

---

## 4. C — distributed, trace-equivalent

### 4.1 Scope

Multiple nodes, no shared memory, observationally equivalent to the reference
machine under I18.

The model is unusually well-positioned for this and it is worth saying why:
frames are durable position-independent byte blobs (v0.2 §1.1), so a
continuation can resume on a node that did not suspend it without migrating
register state. That is the hard part of distributed execution, and it is
already done.

What is not done:

- **Cross-node references.** §6.2 below. This is a correctness blocker, not a
  nicety.
- **Capability spaces across nodes.** Capabilities are actor-relative and
  checked at operation (`docs/SOMA-CAPABILITIES.md`). A remote operation must
  carry proof of authority, and revocation must be observable remotely. This is
  the largest single piece of C.
- **Node failure vs. process failure.** v0.2 §6.3 contains process failure
  precisely. Node loss is different: it can destroy a process that did not
  fault, and it can partition a supervisor from its child. The supervision
  model has no clause for this. Either node loss maps onto `Fault` — which
  claims the machine can always detect it — or the spec grows an explicit
  partition model. This decision should be made before any transport is
  written.
- **Escrowed channel payloads** (v0.2 §3.2) assume the kernel can hold a READ
  root on behalf of an in-flight message. Across nodes, "the kernel" is plural.

### 4.2 Exit criteria

- A two-node run of the streaming graph and the supervision tree is
  I18-equivalent to the single-node run.
- A node killed mid-epoch produces a defined, tested outcome consistent with
  whichever failure model §4.1 selects.
- No test passes by routing all work to one node. The control is a run where
  every process is remote from its supervisor.

### 4.3 Honest assessment

C is roughly the size of A and B combined. See the recommendation in §0.

---

## 5. D — performance

### 5.1 What the current numbers are

`HANDOFF.md` §4 is careful and should stay that way. The figures are
**structural bounds** computed from how continuations group, not hardware
measurements. The irregular-arrival result is a trace-driven policy comparison
where a node's ready tick is an input rather than a consequence of when its
parent ran. Nothing in the repository touches throughput or scheduler overhead.

### 5.2 Scope

Replace bounds with measurements, on the device scheduler from B, running real
bodies from A.

- Wall-clock throughput and latency against the same bulk-frontier baseline
  already used for the structural comparison, so the two are comparable.
- Scheduler overhead as a measured cost, not an assumed zero. The structural
  model charges a lane group spanning `k` run classes exactly `k` masked
  dispatches and charges nothing for binning. Real binning costs something and
  it may cost more than the divergence it avoids. **That is the result most
  likely to overturn the project's premise, so it should be measured first**,
  not last.
- The §4 nulls carried forward: 1.00× for a single run class, 1.00× for zero
  irregularity, 1.00× for a zero wait budget. A hardware measurement that
  cannot reproduce the nulls is measuring the harness.

### 5.3 Exit criteria

- Every hardware number ships with its control and its baseline in the same
  table, per the discipline in `HANDOFF.md` §6.
- `HANDOFF.md` §4 distinguishes structural bounds from measurements
  line by line. Both stay. The bounds are still the honest answer to "what could
  this mechanism do at best".
- Any claim of the form "SOMA is Nx faster" names the baseline in the same
  sentence.

---

## 6. Carried debts — the "and more"

These are not new features. They are recorded traps and gaps that the four
workstreams turn from acceptable into blocking.

### 6.1 `retire_process_if_idle` is a linear scan

`src/kernel/commit.rs`, recorded in `HANDOFF.md` §7. A scan of the continuation
table per completion. Fine for a reference model; wrong under B. Needs a
per-process live-continuation count.

**Blocks B. Cheap. Do it first** — it is the one item here that is pure win with
no design question attached.

### 6.2 16-bit generations bound staleness rather than guaranteeing it

`src/abi/refs.rs`, v0.2 §1 note. A slot recycled 2^16 times wraps and a stale
reference validates. Documented rather than solved, and the documentation is
explicit that it matters if references are ever persisted.

C persists references — across a network, with node-local slot allocators. The
ABA window stops being theoretical. Options: widen the generation field, make
slot recycling node-partitioned, or add a node id to `Ref64`. All three change
the ABI, so this decision must be made before C starts, not during.

**Blocks C. Requires an ABI change. Decide early.**

### 6.3 I14 (progress) is the only unchecked invariant

v0.2 §5 lists it as `[modelled]` — verified by test, not by a state predicate.
Everything else in the table is `[checked]`. Under B this gets harder, not
easier: a concurrent scheduler has more ways to withhold work than a `for` loop
does.

Proposal: **I21. Bounded progress [proposed, checked].** If any continuation is
runnable at an epoch boundary, at least one step executes in that epoch, and no
runnable continuation is deferred for more than a declared bound of consecutive
epochs.

The second half is new and is deliberately stronger than v0.2's I14. v0.2 §4
declines to make any fairness guarantee beyond I14 and permits one run class to
starve another. That is defensible for a sequential interpreter where starvation
is visible in a single trace. Under B, with territories and affinity-based
placement, starvation becomes a policy outcome nobody chose. The bound makes it
a checkable property instead of an emergent one.

### 6.4 Contract dimensions that are currently rejected

v0.2 §1.1 rejects hardware placement, lane shape, relaxed determinism, and
wall-clock deadlines in execution contracts, on the honest grounds that the
abstract machine cannot enforce them. B can enforce lane shape and placement.

Once it can, they should be admitted — v0.2 §7 item 2 says exactly this:
*additional contract dimensions only when an implementation can enforce them*.
Relaxed determinism and wall-clock deadlines stay rejected; the first
contradicts I18 and the second contradicts §4's "there is no wall clock".

### 6.5 `§n` comments point at the historical contract

`HANDOFF.md` §7. Code comments marked `§n` refer to `docs/SOMA-P1.md`, not to
v0.2's `I1..I17` and `§6.x`. Every file touched by v0.3 will make this worse.
Migrate the markers as files are touched rather than in one sweep.

### 6.6 Directory and crate name disagree

Directory `gpu-os`, crate `soma`. Harmless today. If v0.3 produces anything
published, rename the directory then. Not before — a rename during active work
on four workstreams is pure cost.

---

## 7. Sequencing

Nothing here is parallel across people; it is parallel across workstreams, and
the gates are real.

**Phase 0 — unblock.** §6.1 (per-process live count) and the §6.2 ABI decision.
Neither needs the spec settled. §6.1 is a day; §6.2 is an afternoon of argument
and then an ABI change.

**Phase 1 — S, and A in parallel.** S is the gate for everything else and should
start first, but A depends on nothing and can run alongside. Exit Phase 1 when
I18/I19 have passing fault-injection tests and the `2*x + 1` hardcodes are gone.

**Phase 2 — B.** Gated on S. Exit when the device run is I19-equivalent to the
CPU run and the full test suite passes unmodified.

**Phase 3 — D, first pass.** Measure scheduler overhead before anything else,
per §5.2. If binning costs more than the divergence it avoids, that result
changes what C should even be, so it is worth knowing before C starts.

**Phase 4 — C.** Or v0.4. See §0.

---

## 8. Test discipline

The rules in `HANDOFF.md` §6 hold unchanged. Three additions specific to v0.3:

**Every new invariant ships with its fault injection, before the checker.** This
already applies, but I18 makes it load-bearing in a new way: a weakened
equivalence relation that accepts everything looks exactly like one that works,
and only a rejection test tells them apart.

**Every placement-dependent result needs a placement control.** I19 is that
control promoted to an invariant. Any experiment that reports a number under one
placement reports it under two.

**A hardware measurement is not a substitute for a structural bound.** Keep
both. When they disagree, the disagreement is the finding — a structural bound
the hardware cannot reach means the model is charging for the wrong thing.

---

## 9. Not in v0.3

Named so that nothing here is quietly assumed:

- **Preemption.** §3.3.
- **Loops and reductions in evaluator bodies.** §2.2. A reduction is a new
  collective, not a new body.
- **Floating-point bodies.** §2.3, pending a reason to pay for I20 under float.
- **Relaxed-determinism and wall-clock contracts.** §6.4. Both contradict
  clauses the model relies on.
- **A general-purpose surface language.** v0.2 §7 item 1 says "language/compiler
  for arbitrary evaluator bodies". v0.3 scopes the compiler and a minimal body
  language. A programmer-facing language for whole SOMA programs is a separate
  project and should not be smuggled in under A.
