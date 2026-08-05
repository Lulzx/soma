# SOMA v0.3

**Status:** partially implemented. §1–§3 are specification and are machine-checked.
§4–§6 are scope for work that has not started.

v0.2 closed the semantic core: every entity and invariant it named was
implemented and checked. v0.3 is the first version whose work is not "finish the
model" but "prove the model survives contact with real execution".

Four extensions were named by v0.2 §7 and `docs/HANDOFF.md` §8. One
specification change blocked three of them. That change and the workstream that
did not depend on it are done; the rest is scoped below.

| | Deliverable | Status |
| --- | --- | --- |
| **0** | Carried debts on the critical path | **done** (§1) |
| **S** | An equivalence a concurrent implementation can satisfy | **done** (§2) |
| **A** | General evaluator bodies and a compiler | **done** (§3) |
| **B** | Persistent device-resident scheduler | scope (§4) |
| **C** | Distributed / multi-node implementation | scope (§5) |
| **D** | Performance work on real hardware | scope (§6) |

Four new clauses are now checked — I18, I19, I20, I21 — and v0.2's only
`[modelled]` clause is gone. The test suite went from 151 to 192.

---

## 1. Carried debts

### 1.1 Process retirement is no longer a table scan

`retire_process_if_idle` answered "does this process have a continuation left?"
by scanning the whole continuation table on every completion. Fine for a
reference model, O(n) per commit, and not survivable under B.

`ProcessDescriptor::live_continuations` now carries the count, maintained by
`Kernel::set_continuation_status` — the single path through which an existing
continuation's status changes. **I3 recomputes the count by scanning and
compares.** The fast path is derived state, so the slow path stays as the
authority on whether it is right; a status write that bypasses the helper is
reported rather than surfacing later as a stranded process.

### 1.2 The ABA window is closed, without an ABI change

v0.2 §1 documented this rather than solving it: the generation field is 16 bits,
so a slot recycled 65,536 times wraps back to a generation a long-held stale
reference could match.

The scope for this version listed three fixes, all of which changed `Ref64`:
widen the generation, partition slots by node, or add a node id. None was
needed. `GenTable::delete` now **retires a slot rather than wrapping its
generation**. Staleness detection becomes guaranteed at every generation width,
and the cost moves from correctness to space — one withdrawn slot per 65,535
recycles, reported by `GenTable::retired_slots`.

Keeping the generation at 16 bits is now a space/churn tradeoff rather than a
soundness one, and a reference still fits in 64 bits.

**Still deferred:** a node identity, so that two nodes allocating slot 7 do not
produce colliding references. That genuinely does change the reference layout,
and nothing before C can test it. It is a C obligation, not a blocker for it.

---

## 2. S — the equivalence relation

### 2.1 The problem

v0.2 §1.2 defined two runs as equivalent when their traces are **equal**. That
is the right relation for the sequential interpreter and the wrong one for
anything else. Trace equality is defined over a total order on logical time, and
the interpreter's total order is an artifact of running one continuation at a
time: two lanes of one cohort have no semantic order between them, but a `for`
loop must emit one of them first.

Under equality, **every parallel implementation of SOMA was non-conforming by
construction**, which is why B and C could not be attempted under v0.2 as
written.

### 2.2 The semantic order ≺

`semantics::order` derives a partial order from the transition rules. Two
structural relations:

- **Program order.** Events of one continuation are ordered. A continuation is
  sequential by definition (§1.1).
- **Epoch order.** Epochs never move backwards. An epoch boundary is a
  consistent cut (§3.5).

and five causal relations, each keyed on the entity the two events share: future
resolution, mailbox delivery, channel delivery, collective lifecycle, and
supervision notice. Authority grant/effect adjacency is carried over from I10c.

Deliberately **not** included: per-*process* program order. Two read-only
continuations of one process may legitimately run in the same epoch — I13
serialises only mutable ones — so their events are genuinely unordered, and an
edge between them would reject correct implementations.

**I18. Schedule conformance [checked].** A candidate trace conforms to a
reference trace when

1. it contains the same events, per epoch — same kinds, subjects, auxiliary
   data, and causal attribution;
2. every continuation's own events appear in the reference's order;
3. epochs do not move backwards; and
4. every causal edge runs forwards.

Clause 1 is what keeps the relation from being vacuous. A weakened equivalence
that only checked ordering would accept an implementation that silently dropped
work, which is strictly worse than the equality it replaced.

**I19. Placement neutrality [checked].** Every run in a set of placements is
I18-equivalent to the first. If changing where work runs changes what a program
observes, the placement layer has leaked into the semantics.

This is the control `HANDOFF.md` §6 demands, promoted to an invariant.
`tests/semantic_order.rs` runs one workload at cohort widths 1, 2, 4, and 16 and
requires them to agree — with a null showing that the widths really do produce
different raw traces, so the agreement is not vacuous.

### 2.3 What the trace ABI needed

Channel, collective, and supervision edges were already recoverable: those events
carry their entity and sequence number. Future wakes and mailbox deliveries were
not — `FutureResolved` recorded the resolved *value*, and `MessageReceived`
recorded no sender, so neither could be paired.

`TraceEvent` gains a `causal: Ref64` field naming the entity through which an
event is causally related to another. The sequential interpreter does not need
it: its total order over `logical_time` already encodes every dependency. A
distributed implementation does, because it emits events with no shared clock
and cannot infer causality from adjacency.

### 2.4 Placement events are not observable behaviour

I19 is unsatisfiable without this, and getting it wrong is the one way widening
the equivalence could hide a real defect.

Running the same program at cohort width 1 and width 16 produces a different
number of `CohortCreated` records and identical behaviour. Since v0.2 §4 already
says cohorting is a strategy the model enables rather than requires, treating
those records as observable would make §4 false. They stay in the trace —
accounting and the occupancy studies need them — and the *semantic projection*
drops them.

The rule is deliberately narrow: an event is placement-only when removing it
from every conforming implementation's trace loses no information about process,
continuation, message, future, capability, or collective state. Today that is
`CohortCreated` and `ContinuationPlaced`, and nothing else.

### 2.5 Two things writing the checker exposed

**The scope for this version had one edge backwards.** It said "`FutureResolved(f)`
≺ every `ContinuationReady` it caused". §3.3 emits the wakes *first* and the
resolution last, and the implementation agrees, so ≺ orders them wake ≺
resolution. The relation is an ordering constraint read off the transition
rules, not a claim about physical causation — orienting an edge the intuitive
way against the rule that emits it would make the reference interpreter fail its
own checker. `the_reference_run_is_a_linear_extension_of_its_own_order` is the
first test in the file for that reason.

**A forward scan makes the checker decorative.** The first implementation paired
sends with receives by consuming a "pending send" as the scan met each receive.
That finds no pair at all when the two are inverted — so it emits no edge, and
an inverted delivery is *silently accepted*. The test caught it. Every relation
now collects both sides by key and joins afterwards, so the edge exists
regardless of position and the inversion shows up as `earlier > later`.

The same trap applies to program order, which is why clause 2 of I18 compares
against the reference rather than deriving an edge: an edge derived from the
candidate's own positions can never be inverted, because the derivation simply
reads the events in whatever order it finds them.

---

## 3. A — evaluator bodies

### 3.1 The hole this closed

`compiler::ir` named evaluators and no more. Both backends took an
`evaluator_id`, ignored it, and hardcoded `2*x + 1`. I17 checked that a
collective's evaluator and stride matched its module's manifest, so **nothing
anywhere checked that a backend applied the function its module named** — a
backend could return arbitrary bytes and every invariant still held. The
CPU/Metal comparison test passed because two hardcodings of one constant agree,
which is not evidence about a compiler.

### 3.2 The language

`compiler::body` defines a deliberately small one, because the property under
test is placement-independent publication, not language design:

- **Pure and total.** No allocation, no memory beyond the element, no division
  (which would be partial), no loops. Every program terminates in a fixed
  number of steps decided at validation time.
- **Element-wise.** One input element in, one output element out. A reduction
  is a different collective, not a different body.
- **Integer-only.** Admitting `f32` would force I20 to either demand
  bit-identical results across backends — constraining what a GPU may do — or
  weaken to a tolerance, which guts it as an invariant. Deferred until there is
  a reason to pay for it.
- **Typed against a declared layout.** Reading or writing outside the declared
  element is a validation error, so an invalid body cannot reach a backend.
- **Branch-free.** `select` is the only control flow, so a cohort of lanes
  executing one body never diverges.

Arithmetic wraps on `u64` and truncates to the field width on store; shifts mask
their amount to 6 bits. Both rules exist so the CPU interpreter and the
generated Metal Shading Language agree by construction rather than by luck.

Element stride is now **derived** from the declared layout. A module whose
declared stride disagrees with its body's layout fails to load: a backend
striding differently from its collective reads across element boundaries and
returns plausible garbage.

The pre-v0.3 form still parses. An evaluator may be declared without a body,
which means no backend can realize it — `execute_with_spill` reports
`UnsupportedEvaluator` rather than applying whatever it happened to have
compiled in.

### 3.3 I20

**I20. Backend agreement [checked].** For a given evaluator and frozen input,
every backend claiming to realize that evaluator produces identical output
bytes. A backend that cannot realize a body returns `UnsupportedEvaluator`
rather than an approximation.

Both halves are tested, because they pull in opposite directions: an honest
abstention must pass where a wrong answer fails. The fault injection is a
backend computing `2*x + 2` — off by one from the body it claims to implement,
and exactly the class of defect the old hardcoded backends could not have
detected in each other.

`BatchBackend` gains `install`, so a backend answers only for bodies it has been
given. That is what makes `UnsupportedEvaluator` an honest answer rather than a
guess.

### 3.4 The hardcoded paths are gone

`compiler::examples` carries four bodies. `double_plus_one` is the function both
backends used to hardcode; the constant now exists in exactly one place, as
data, and both lowerings are generated from it. `min_and_xor` is not expressible
as `a*x + b` — it branches on a comparison and writes two fields from two inputs
— so a backend cannot pass I20 for it by coincidence. `bitmix` exercises the
shift-masking rule.

`tests/metal_backend.rs` runs all three on real Apple GPU hardware and compares
bytes and semantic trace against the CPU interpreter.

---

## 4. B — persistent device-resident scheduler

**Not started.** S was its blocker and S is done, so this is the next piece.

`kernel/epochs.rs` still admits, cohorts, executes, and commits on the host, one
continuation at a time. The Metal path dispatches a single collective and blocks
on `wait_until_completed` — a host-driven kernel launch, which is the thing the
project's original premise wanted to remove.

What must hold, and why each is hard:

- **Admission (I13)** claims a per-process mutable slot with a host `HashSet`.
  On device this is a concurrent claim and must be deterministic, so it cannot
  be "whoever gets there first".
- **Commit** is the sole path to `Runnable` (v0.2 §3.4), which is what makes I7
  checkable. A device commit must preserve that exclusivity across concurrent
  writers.
- **The step-budget check must precede dispatch** (`epochs.rs`, and
  `HANDOFF.md` §7 records the bug from getting this wrong). Concurrency does not
  relax this.
- **Trace emission** becomes a concurrent append. Logical time must still
  satisfy I11 and now also I18.

Not in B: preemption. The model does not assume it (v0.2 §1.1), and adding it to
the scheduler before adding it to the model would be backwards.

**Exit criteria.** An epoch runs with no host round-trip inside it; I19 holds
between the device run and the CPU run; and the existing suite passes
unmodified. A test that needs modification is a semantic difference and needs an
explanation in the specification, not a patch to the test.

Contract dimensions v0.2 §1.1 rejects on the honest grounds that the abstract
machine cannot enforce them — hardware placement and lane shape — should be
admitted once B can enforce them, per v0.2 §7 item 2. Relaxed determinism and
wall-clock deadlines stay rejected: the first contradicts I18 and the second
contradicts §4's "there is no wall clock".

---

## 5. C — distributed, trace-equivalent

**Not started, and still the long pole.** Roughly the size of B and A combined.

The model is unusually well-positioned for the hard part: frames are durable
position-independent byte blobs (v0.2 §1.1), so a continuation can resume on a
node that did not suspend it without migrating register state.

What is not done:

- **Node identity in references.** §1.2 above. Closing the ABA window did not
  give two nodes disjoint reference spaces.
- **Capability spaces across nodes.** Capabilities are actor-relative and
  checked at operation. A remote operation must carry proof of authority, and
  revocation must be observable remotely. This is the largest single piece.
- **Node failure vs. process failure.** v0.2 §6.3 contains process failure
  precisely. Node loss is different: it can destroy a process that did not
  fault, and it can partition a supervisor from its child. The supervision model
  has no clause for this. Either node loss maps onto `Fault` — which claims the
  machine can always detect it — or the specification grows an explicit
  partition model. **Decide before writing any transport.**
- **Escrowed channel payloads** assume the kernel can hold a `READ` root for an
  in-flight message. Across nodes, "the kernel" is plural.

**Exit criteria.** A two-node run of the streaming graph and the supervision tree
is I18-equivalent to the single-node run; a node killed mid-epoch produces a
defined, tested outcome; and no test passes by routing all work to one node —
the control is a run where every process is remote from its supervisor.

---

## 6. D — performance

**Not started.** Blocked on A and B: there was nothing worth measuring until a
real evaluator body ran under a real scheduler. A is now done.

`HANDOFF.md` §4 is careful and should stay that way. Those figures are
**structural bounds** computed from how continuations group, not hardware
measurements, and the irregular-arrival result is a trace-driven policy
comparison where a node's ready tick is an input rather than a consequence of
when its parent ran.

**Measure scheduler overhead first.** The structural model charges a lane group
spanning `k` run classes exactly `k` masked dispatches and charges nothing for
binning. Real binning costs something, and it may cost more than the divergence
it avoids. That is the result most likely to overturn the project's premise, and
knowing it changes what C should even be.

Carry the §4 nulls forward: 1.00× for a single run class, 1.00× for zero
irregularity, 1.00× for a zero wait budget. A hardware measurement that cannot
reproduce the nulls is measuring the harness.

Keep both kinds of number, line by line. When a structural bound and a hardware
measurement disagree, the disagreement is the finding — a bound the hardware
cannot reach means the model is charging for the wrong thing.

---

## 7. Clause summary

| Clause | Status | Note |
| ------ | ------ | ---- |
| I1–I13, I15–I17 | checked | unchanged from v0.2 |
| I14 progress | **superseded by I21** | was v0.2's only `[modelled]` clause |
| I18 schedule conformance | checked | replaces trace equality (§2) |
| I19 placement neutrality | checked | cohort width, with a non-vacuity null |
| I20 backend agreement | checked | CPU interpreter is the definition |
| I21 bounded progress | checked | no withholding, plus a starvation bound |

**I21** has two halves. The first — an epoch that admitted work dispatched some
of it — is a statement about a transition rather than a state, so it is counted
as it happens and checked as `stalled_epochs == 0`. The second is new and
deliberately stronger than v0.2's I14: no runnable continuation waits longer
than a declared bound. v0.2 §4 declined to promise this and permitted one run
class to starve another, which is defensible for a sequential interpreter where
starvation is visible in a single trace. Under territory placement and class
affinity it becomes a policy outcome nobody chose.

---

## 8. Not in v0.3

Named so that nothing here is quietly assumed:

- **Preemption.** §4.
- **Loops and reductions in evaluator bodies.** A reduction is a new
  collective, not a new body.
- **Floating-point bodies.** §3.2, pending a reason to pay for I20 under float.
- **Relaxed-determinism and wall-clock contracts.** Both contradict clauses the
  model relies on.
- **A general-purpose surface language.** v0.2 §7 item 1 says
  "language/compiler for arbitrary evaluator bodies". v0.3 delivers the
  compiler and a minimal body language. A programmer-facing language for whole
  SOMA programs is a separate project.
