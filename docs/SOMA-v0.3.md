# SOMA v0.3

**Status:** active implementation. §1–§3 are specified and machine-checked.
§4's semantic obligations, speculative concurrent CPU executive (§4.18), and
several real-Metal resident slices are implemented; the general no-round-trip
executive remains incomplete. §5 now has authenticated remote execution and
owner-side resource prototypes, but not a multi-kernel backend. §6 contains
real measurements, including negative end-to-end evidence. The stricter release
gates are tracked in `DREAM-COMPLETION.md`.

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
| **B** | Persistent device-resident scheduler | **in progress** (§4) — real device admission, placement, fixed graphs, evaluator execution, journals, and host canonical replay; general resident effects/quiescence not done |
| **C** | Distributed / multi-node implementation | **in progress** (§5) — authenticated remote execution and authoritative remote resource services; actual multi-kernel process execution not done |
| **D** | Performance work on real hardware | **in progress** (§6) — hardware controls and end-to-end negative evidence; qualifying application win not done |

Eight new clauses are now checked — I18 through I25 — and v0.2's only
`[modelled]` clause is gone. The suite has continued to grow beyond its original 151 tests; CI runs the
default, native, and all-feature build matrices rather than freezing a stale
count in this document.

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

0. a correspondence between the two runs' entity names exists (§2.6);
1. it contains the same events, per epoch — same kinds, subjects, auxiliary
   data, and causal attribution — read through that correspondence;
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

### 2.6 Identity is not observable either

Added while scoping §4.3, which is where the omission showed up.

§2 removed one assumption the sequential interpreter had smuggled into the
equivalence — that logical time is a total order every implementation can
reproduce. It left a second one untouched, one level down: **that two runs of a
program give their entities the same names.**

A `Ref64` is a table position. Two allocators that never coordinate — which is
what a device's lanes and a cluster's nodes have to be — hand out different
positions for the same entity. Comparing raw references makes every such
implementation non-conforming by construction, for a reason that has nothing to
do with what it did. That is the §2.1 defect exactly, and it would have been
discovered the moment §4.3 partitioned the allocator.

So I18 compares up to a correspondence between the two runs' names. The
correspondence is **forced, not chosen**: entities pair in the order they first
appear, within their kind. A checker allowed to pick the pairing could pick
whichever one made the traces agree, which would make the clause vacuous.
Positional pairing of two sequences of distinct names is a bijection, so a run
that dropped an entity or merged two into one produces sequences of different
lengths and is reported rather than renamed away — the negative cases in
`tests/identity_equivalence.rs`.

The boundary worth stating, because it looks like a defect and is not:
exchanging two entities' names *uniformly* conforms. It exchanges their
first-appearance order too, so the forced correspondence pairs them the other
way round and the swap cancels. The run did the same things under different
names. Applying the same swap to only part of a run does not conform, and that
is the case separating a renaming from a behavioural difference.

**This forced one ABI change.** Four event kinds recorded an entity in
`auxiliary` as a bare slot number: the future a continuation awaits, the value a
future resolved to, the process a restart replaced, the contract a continuation
was created under. A bare slot has no kind and no generation, so nothing can
translate it and nothing can even distinguish it from the sequence numbers
`auxiliary` also carries. `TraceEvent` gains `subject: Ref64` for the entity an
event is *about*, distinct from `causal`, which names the entity two events are
ordered *through*. `auxiliary` is now purely numeric.

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

`compiler::surface` is its named-value source front end. It adds human-readable
field, auxiliary-array, local, value, loop, store, and reusable pure-function
declarations with source-line diagnostics. Function calls are compile-time
inlined, recursive cycles are rejected, and the fully expanded program lowers
to the same step-bounded body IR used by the reference interpreter, Cranelift
native CPU backend, and Metal code generator.
It is general-purpose within the pure, total, element-wise evaluator contract;
the contract's deliberate exclusions are described in
`docs/EVALUATOR-LANGUAGE.md`.

- **Pure and total.** No allocation and no division (which would be partial).
  Every program terminates in a number of steps decided at validation time:
  `step_bound` multiplies out the loop nesting and `MAX_STEPS` is the ceiling.
  Two rules keep that true where it would usually fail — an out-of-range
  `gather` clamps to the last element rather than faulting, and a loop's trip
  count is a constant rather than an expression.

  Totality is not a preference. `kernel/epochs.rs` checks a continuation's step
  budget before dispatch and a collective evaluation is one step of one
  continuation, so a body that might not terminate makes that check
  meaningless. That is the constraint a data-dependent trip count would break,
  and `breakif` is what gives back the useful half of one: leaving a loop early
  can only lower the count, so the static bound still holds.
- **One output element per input element.** A reduction is a different
  collective, not a different body. `gather` widens what a body may *read* to
  any element of the frozen input array — `index` supplies its own position, so
  a stencil is expressible — and `gatheraux` widens it to a *second* frozen
  array the collective binds. Neither widens what a body may write: it still
  writes only its own element's fields.

  The second array is what makes a lookup expressible rather than only a
  stencil, and the distinction is not academic — it is the difference between
  an ant reading its neighbours and an ant reading the trail grid it is
  standing on, which is a different array with a different layout and a
  different length. It is declared with its own `aux` element layout for that
  reason, and bound to the collective rather than passed at the call, because
  the binding is what the capability escrow freezes. It is authorized and
  validated on the same terms as the first array: a body gathering from an
  array its actor holds no READ on would be a capability hole, and one
  gathering from an unfrozen array would make the published result depend on
  when the collective ran, which is I19.

  The binding is checked in both directions at the backend boundary. A body
  that reads a second array and was given none is `InvalidInput`, rather than
  being evaluated against the first array alone — that answer is bytes, and
  bytes are indistinguishable from a correct result to every other invariant in
  the machine. So is an array bound to a body with no name for it, because the
  caller froze something for the collective's lifetime and would otherwise
  never find out.
- **Reads the input, never the output.** This is what makes a gather safe to
  run in any order, and so what keeps I19 true of a gathering body. The input
  array is frozen and single-assignment, so every lane sees the same bytes
  whenever it runs and no lane can observe another lane's store. A body able to
  read the output array would make the published result depend on the schedule.
- **Typed integer plus bounded binary32.** Integer semantics remain unchanged.
  The bounded `f32` slice admits arithmetic, ordered comparison, and selection,
  disables Metal fast math,
  flushes subnormal inputs/results to positive zero, and canonicalizes NaNs and
  signed zero. That deliberately constrains every backend so I20 remains
  bit-identical rather than weakening to a tolerance.
- **Typed against a declared layout.** Reading or writing outside the declared
  element is a validation error, so an invalid body cannot reach a backend.
- **Loops carry state in locals, not in values.** SSA and back edges need phi
  nodes, and "an instruction names an earlier instruction" stops meaning
  anything once an instruction can run twice. So a value computed inside a loop
  is visible only within the iteration that computed it — validation rejects a
  reference that escapes one, including from a `store` — and `get`/`set` over
  declared locals are how anything outlives an iteration. The escape rule is a
  prefix test on loop nesting rather than an equality test, so reading a value
  from an enclosing scope *into* a loop stays legal; that value does not change
  while the loop runs.
- **Branch-free is a property of a body, not of the language.** It used to be
  both, because `select` was the only conditional. `breakif` can put two lanes
  of a cohort on different iterations, so `is_uniform` is a question a
  scheduler asks about a body rather than an answer the language guarantees.

  Divergence costs occupancy and not correctness: both lowerings still agree,
  and I20 checks that on hardware for a body whose lanes leave on different
  iterations (`examples::run_length`). A counted `repeat` is *not* divergence —
  every lane runs the same iterations — and neither are `index` and `gather`,
  where lanes reading different addresses still execute identical instructions
  in identical order.

  Edge handling remains the body's job. `examples::neighbour_max` substitutes
  its own index at position zero with a `select`, because `0 - 1` wraps and
  would otherwise clamp to the far end of the array.

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

**Started.** S was its blocker and S is done. All four of its obligations are
discharged and checked: admission (§4.1), trace emission (§4.2), commit (§4.4
produces the bin entries, §4.5 applies them at the epoch boundary), and the
step-budget check's position, which held already. Read visibility — §4.3's third
problem, and the one that gated the applier's move — is settled as I25 rather
than by deferring the remaining tables, for the reason §4.5 gives.

The speculative CPU executive in §4.18 runs isolated lanes on several threads,
validates their complete operation journals, and either commits them in plan
order or replays the epoch through the reference path. That establishes the
concurrent execution shape. What remains is carrying the same shape through a
device plan, device lane execution, and canonical device commit.

The first device phase is now real rather than modelled. `MetalDeviceScheduler`
runs deterministic admission and stable run-class/cohort placement concurrently
on Metal threads, with candidate and placement buffers retained across epochs
and both passes encoded in one command buffer. It agrees field-for-field with
the independent `reference_device_schedule` for all partial policies. It does
not yet carry that plan through continuation execution and canonical commit;
that remaining boundary is specified in `docs/DEVICE-SCHEDULER.md`.

`kernel/epochs.rs` still cohorts, executes, and commits on the host, one
continuation at a time. The Metal path dispatches a single collective and blocks
on `wait_until_completed` — a host-driven kernel launch, which is the thing the
project's original premise wanted to remove.

What must hold, and why each is hard:

- **Admission (I13)** claimed a per-process mutable slot with a host `HashSet`.
  On device this is a concurrent claim and must be deterministic, so it cannot
  be "whoever gets there first". **Done — §4.1.**
- **Commit** is the sole path to `Runnable` (v0.2 §3.4), which is what makes I7
  checkable. A device commit must preserve that exclusivity across concurrent
  writers. **Done — §4.4 and §4.5.** Bin entry is produced rather than
  performed, and the applier runs once at the epoch boundary in plan order.
- **The step-budget check must precede dispatch** (`epochs.rs`, and
  `HANDOFF.md` §7 records the bug from getting this wrong). Concurrency does not
  relax this.
- **Trace emission** becomes a concurrent append. Logical time must still
  satisfy I11 and now also I18. **Done — §4.2.**

### 4.1 Admission decides from state, not from position

`scheduler::admission::admit` is a pure function of the epoch's candidate set.
It resolves I13's per-process claim by taking the longest-waiting mutable
continuation, ties broken by continuation identity, having first collected every
candidate — so no candidate's fate depends on how far a scan had got when it was
reached.

The waiting term is not decoration. Identity alone would be deterministic and
unfair: a process whose lower-slot mutable continuation stays runnable would win
every epoch and the other would never run, which I21's starvation bound would
eventually report as a violation nobody chose. The scan rule got that right by
accident — a deferred continuation was re-enqueued during admission, ahead of
the appends that epoch's commits made, so it led the next epoch's bin. Once bin
order stops being trustworthy the accident is gone, so the property moves into
the key.

**I22. Admission determinism [checked].** Two halves:

1. Each epoch's decision is the one `admit` specifies for that epoch's
   candidates.
2. `admit`'s decision is invariant under permutation of those candidates.

Half 2 alone would be a statement about a function nobody is obliged to call: a
scheduler that took its own first-come claim inline — which is what the host did
and what a device is tempted to do — would leave it green. So the epoch records
the decision it took (`Kernel::admission_log`) and half 1 compares the two. This
is the same failure mode §2.5 found in the first pairing scan, caught earlier
this time because §2.5 had already named it.

Half 1 is nonetheless weak on its own, and the way that was established is worth
recording: reintroducing the first-come claim inline in `epochs.rs` broke *no
test*. The two rules agree on the reference interpreter's discovery order, so a
checker comparing one run's decision against the rule sees nothing. Only a
concurrent implementation would diverge, which is to say the clause would first
fail on the hardware it was written to protect.

So `Admission` is sealed: its fields are private and `admit` is its only
constructor, which makes an epoch that takes its own claim a compile error
rather than a test that might catch it. This is the technique
`docs/SOMA-CAPABILITIES.md` used to close the operation set, applied to the
scheduler. Half 1 stays, because the record is what lets the clause be asked of
an implementation this crate did not run — the case it is ultimately for.

The fault injection is the rule this replaced: `first_come_decision` in
`tests/admission.rs` is the pre-v0.3 claim, and half 2 rejects it. Half 1's
negative case corrupts a recorded epoch's candidates so its decision no longer
follows from them. The null is that the workload really does put two mutable
continuations of one process in one epoch — without that, the clause would be
checking a decision with nothing to decide.

**What §4.1 does not settle.** I22 constrains *which* continuations run, not the
order they run in. Lane order within a bin is still arrival order, which is
observable — two lanes receiving from one mailbox get different messages
depending on which goes first — and a device whose bins are appended
concurrently does not get that order for free. It gets it only by committing an
epoch's effects in canonical lane order, which is the second obligation above
and is not done. Folding lane order into I22 now would state a property the host
satisfies and no device could, which is exactly the defect §2 removed from the
equivalence relation.

### 4.2 The trace no longer needs a clock

`logical_time` is one counter, incremented per event. Concurrent lanes have no
counter to share, so the fourth obligation is not "make the append thread-safe"
— a mutex around the clock would do that and would serialise every lane at the
point the design is trying to parallelise.

Every event now carries the position it was emitted at: its epoch, the lane that
emitted it, and that lane's own count. Lanes are numbered from 1 in the epoch's
admitted order — a position in the plan, decided before anything runs — and
`HOST_LANE` is zero, so an epoch's own bookkeeping sorts ahead of the lanes it
set up. A lane counts locally and consults nothing shared.

**I23. Position-derived emission [checked].** Three clauses:

1. positions are unique, so the ordering they induce is total;
2. sorting the trace by position reproduces the trace as emitted; and
3. work that ran in a lane is attributed to one, and no two continuations share
   a lane within an epoch.

Clause 3 is not bookkeeping. Without it an implementation that emitted every
event from `HOST_LANE` off a single counter satisfies clauses 1 and 2 *exactly*
— positions unique, sorted order equal to emitted order — while being the
shared-clock design the clause replaces. That is not hypothetical: deleting the
`enter_lane` call leaves `sorting_by_position_reconstructs_the_run` green and
fails five other tests, which is how the hole was found.

Clause 2 is a statement about the reference, and the specification should not be
read as demanding it of an implementation. It holds for a run whose append order
*is* its emission order, which is what a sequential interpreter produces. A
concurrent implementation appends interleaved and will not satisfy it on its raw
trace; what it owes is clauses 1 and 3, and I18 after sorting by position.
Demanding clause 2 of it would re-import the assumption §2 removed from the
equivalence relation.

Two consequences worth recording. The epoch's `CohortCreated` records are now
emitted together, before Phase F, rather than one at a time as execution reached
each cohort: the plan is complete before anything runs, so a placement record
that depended on when a lane ran was recording the wrong thing. And lane and
sequence are placement information, so `semantic_projection` does not carry them
— a device that groups work differently must still be I18-equivalent.

What §4.2 does not do is make lanes reorderable. It makes the *record* of a run
reconstructible without a clock. What a lane observes still depends on when it
ran relative to other lanes. §4.4 takes the produce-then-apply half of that;
what stays is read visibility, §4.3's third problem.

### 4.3 Canonical commit is three problems, not one

Scoping this section found that the sentence below names one obstacle and there
are three. Only the first was written down.

1. **Mutation ordering.** Handlers mutate as they run, so execute and commit are
   fused. This is the one §4.3 originally named.
2. **Allocation identity.** A step allocates entities and then *uses* them
   within the same step — `expand_resume_2` creates a child process and
   immediately creates a continuation on it, and stores a freshly created
   payload object's reference **inside the encoded frame bytes**. A symbolic
   reference cannot survive into an opaque byte blob without a relocation table,
   so allocation has to be eager, so concurrent lanes need an allocator that
   hands out names without coordinating.
3. **Read visibility.** Today lane 5 sees what lane 1 wrote in the same epoch.
   Lanes reading a snapshot do not. That is a semantic change, not an
   implementation detail.

**(3) was measured rather than assumed.** Using the ≺ machinery from §2: across
the Expand workload at cohort widths 1, 2 and 16 and a four-class heuristic
search — 1025 events, 441 edges — **no ≺ edge joins two lanes of one epoch**.
That is structural rather than lucky: the wake events (`MessageReceived`,
`ContinuationReady`, `ChannelReceived`) are emitted by the *acting* lane, so a
delivery edge is either within one lane or across epochs.

It is not an invariant of the model, and the specification should not claim it
as one. The channel workloads drive their sends from outside any lane, so they
do not exercise the case; a lane-driven `channel_send` followed by a later
lane's receive in the same epoch is reachable in principle. It is a
**precondition to check per run**, not a property of the model.

§4.5 keeps that standing exactly and makes the checking continuous: I25 asks it
of every run rather than of the one workload measured here. The reason it has to
be asked rather than arranged is also settled there — deferring the remaining
tables until reads are snapshots is not reachable, because (2) above forces
allocation to stay eager.

**(2) is the same problem as C's node identity.** §1.2 deferred "a node
identity, so that two nodes allocating slot 7 do not produce colliding
references" to C on the grounds that nothing before a multi-node implementation
could test it. Lane-local allocation needs exactly that field, and it is
testable now. `Ref64.partition` is that field; it occupies the byte that was
`flags`, written as zero everywhere and read nowhere, so a reference still fits
in 64 bits. `Ref64::key` is the `(partition, slot)` pair that replaces a bare
slot wherever the kernel keys a map by entity identity.

Partitioned allocation was untestable until §2.6, because a run allocating from
a different partition fails raw-reference comparison for reasons unrelated to
its behaviour. Both are now done.

`GenTable` allocates from a partition rather than from one slot space. A lane's
partition is `(lane - 1) mod n` — a function of its position in the epoch's
plan, so it is decided before anything runs and does not depend on which worker
picks the lane up. Within a partition, allocations still happen in lane order.
That is the whole determinism argument, and it is why the partition comes from
the plan rather than from a worker id: worker assignment is dynamic, plan
position is not.

**I19 now varies allocation partitioning** alongside cohort width, at 1, 2, 4
and 8 partitions and in combination with widths 1 through 16. The nulls: a
one-partition run really does use only partition 0, an eight-partition run
really does spread, slot numbers really do repeat across partitions, and a
partitioned run really does name its entities differently in more than a quarter
of its events — which is the measurement of how badly raw-reference comparison
would have failed.

Wiring this up **found a live defect** rather than confirming a design. I8
(frame exclusivity) keyed its owner map by the frame's bare slot, so two frames
allocated as slot 7 in different partitions read as one frame shared by two
continuations. Four more checks had the same shape: I3's live-continuation
recount, I6's per-sender sequence map, and the kernel's mailbox, payload,
capability-space, supervision-queue and send-sequence maps. All of them are
keyed by `Ref64::key` now. A bare slot stopped being an identity the moment
partitions existed, and the invariant checker is what said so.

The obstacle to the first is structural rather than subtle: the
executive's handlers take `&mut Kernel` and allocate their effects as they run
(`create_process`, `resolve_future`, `enqueue_message`), so execute and commit
are fused. Canonical commit requires a step to produce an effect list that the
kernel applies afterwards, in lane order. That refactor is §4.4, and it is the
one that makes a concurrent executive — host threads first, device after —
possible at all.

§4.2 supplies the ordering key it will apply them in: a lane number is a
position in the plan, so "in lane order" is already well-defined and already
independent of when a lane ran. What is missing is a step that produces effects
instead of performing them, which needs symbolic references for the entities a
step allocates and then uses — a step that creates a future and stores it in its
frame cannot be handed a real `Ref64` before commit.

The third obligation — the step-budget check preceding dispatch — holds today
and is not weakened by anything above; it needs no work, only a guard against
being moved. `HANDOFF.md` §7 records why.

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

### 4.4 The effect log

§4.3's first problem was mutation ordering: handlers mutate as they run, so
execute and commit are fused. This section unfuses the part of it that every
lane of an epoch writes.

A step no longer writes a runnable bin. It **produces** the entries it wants
written, into a per-lane journal, and the kernel applies them afterwards. Four
shapes exist, and they are the four the kernel already had rather than a
generalisation: a commit resume, a waiter wake, a fresh continuation's first
bin, and a lane a partial-cohort policy held back. They differ in how the
continuation's status is written, and collapsing them would change which
continuations look long-waiting to §4.1's claim.

**Scope, stated plainly.** Bin entry, and the status transition that goes with
it. Nothing else. Mailboxes, futures, capability spaces and the object tables
are still written as the step runs, and allocation is still eager. The choice is
not arbitrary: v0.2 §3.4 already makes commit the sole path to `Runnable` — that
is what makes I7 checkable — and the bins are the one structure an epoch's lanes
all write. §4.3 (2) establishes that allocation *has* to stay eager, because a
step stores references it allocated into opaque frame bytes; partitioned
allocation is what makes that safe for concurrent lanes.

**The applier ran at the end of each lane** when this section was written,
which is exactly where a sequential interpreter already wrote. So no run
changed: the whole suite passed with one call site edited, and that one only
because the seal below moved a deliberate fault injection into `kernel::raw`.
This was the point at which it would have been easy to overclaim. Producing
effects is not canonical commit; it is what makes canonical commit a *change of
one line*. §4.5 makes that change.

**I24. Effect-mediated commit [checked].** Three clauses:

1. nothing is applied twice and nothing is lost — the applied indices are
   exactly `0..n`, each once;
2. sorting the log by the position an effect was produced at puts the applied
   indices in increasing order; and
3. no bin entry arrived any other way — the scheduler counts every entry it has
   ever made, and the log accounts for all of them.

Clause 2 is what "in lane order" means as a property of a record. A lane number
is a position in the plan (§4.2), so the clause is independent of when a lane
ran, which is the whole reason the plan supplies the ordering key rather than
the clock.

Clauses 1 and 2 are, on this interpreter, satisfied by construction — a
sequential applier draining one journal at a time cannot produce an out-of-order
log. That is the same standing as I23's clause 2 and I22's first half, and for
the same reason: the record is not there to catch this crate. It is there so the
clause can be asked of an implementation whose lanes append concurrently, where
the log's row order is not its application order and the question has content.
`kernel::raw` supplies the failing cases in the meantime, one per clause.

Clause 3 is the one with teeth here, and it is the runtime half of a
compile-time guarantee. `Scheduler::enqueue` demands a `Committing` token whose
only constructor is inside `kernel::effects`, so a step that writes a bin as it
runs does not compile. That is §4.1's technique for sealing `Admission`, applied
to the other end of the epoch; the count carries it to an implementation this
crate did not compile.

**One ordering rule came out of it.** Cancellation empties the bins of
everything it cancels, so it has to see the whole lane and not the part that has
landed: `cancel_process_continuations` applied the journal before it ran. No
handler in the executive creates work and then fails in the same step, so no run
exercised it — which is why it was a one-line ordering statement whose
correctness was visible by inspection, rather than a withdrawal list with
retention logic nothing tests. §4.5 is where it became load-bearing, and the
withdrawal list is what replaced it.

**What §4.4 does not do.** It does not make lanes reorderable, for the same
reason §4.2 did not: what a lane *reads* still depends on when it ran. §4.3 (3)
measured that no ≺ edge joins two lanes of one epoch across the workloads
checked, and was careful to call that a precondition to check per run rather
than an invariant. Applying an epoch's effects at the epoch boundary is what
turns that precondition into a requirement, and it is §4.5.

### 4.5 Canonical commit

`apply_lane_effects()` moved out of the lane loop. An epoch runs every lane,
then applies what they produced, sorted by the position each effect was
produced at. That is the whole diff, and §4.4 predicted it would be — what it
could not do was pay for it.

**The price is read visibility, and it is paid by checking rather than by
deferring.** This is the decision worth recording, because the obvious reading
of §4.3 (3) is that the remaining tables should be deferred too until a lane
provably reads a snapshot. They cannot be. §4.3 (2) establishes that allocation
has to stay eager, because a step stores references it allocated into opaque
frame bytes; a step will therefore keep writing tables as it runs no matter how
much of *commit* moves out of the lane. Snapshot reads are not reachable by
continuing the §4.4 refactor, and pretending otherwise would have made the next
slice an infinite one.

So the clause is stated as an obligation on the run instead:

**I25. Lane independence [checked].** No ≺ edge joins two distinct lanes of one
epoch.

The relation is §2's, unchanged, and the exclusion of `HOST_LANE` is not an
exemption: the host's part of an epoch — admission's deferrals, the cohort
records, the lanes a partial policy held back — runs strictly before or strictly
after the lanes, so an order between it and a lane is the plan's own rather than
a race between two things that could have gone either way.

What I25 asserts is exactly what §4.3 (3) measured, with its standing changed.
It was a property nothing depended on: with the applier running per lane, a run
with a cross-lane edge was merely a run whose lanes could not be reordered, and
the executive did not reorder them. With the applier at the epoch boundary such
a run is one this executive commits in an order its lanes did not observe. That
is the difference between a measurement and a requirement.

It remains a property of a *run* and not of the model, which is the standing
§4.3 gave it and the honest one. A workload driving `channel_send` from one lane
and receiving it in a later lane of the same epoch is still expressible; I25
reports it rather than the kernel refusing it. The report is the useful outcome
— it names the workload a concurrent executive cannot take, at the point the
workload does it, rather than leaving it to surface as a nondeterministic result
on hardware.

**Cancellation stopped being a one-line ordering statement.** §4.4's rule was
that `cancel_process_continuations` applies the journal before emptying the
bins, so that it sees the whole lane. Applying it now would commit the epoch's
earlier lanes in the middle of a later one, which is the ordering this section
exists to remove. It withdraws instead: entries this epoch produced and has not
applied come out of the journal, and `remove_all` still takes the ones earlier
epochs applied. The two are not alternatives — a cancelled process can easily
have both — and a withdrawn effect never reaches the log at all, which is what
distinguishes the new path from the old one in a test rather than by inspection.

§4.4 declined to write this because no handler reached it. One does: the
`CancelPending` check at the end of `apply_step_result` runs after the branch
that emits the resume, so a process cancelled while its continuation is
mid-step cancels a continuation whose bin entry is still pending. The test
constructs that precondition and lets the real path run from there, and it fails
if withdrawal is replaced by application.

**The existing suite passes unmodified**, which §4.3's exit criteria demand and
which is not a weak result here: it is §4.3 (3)'s measurement coming true. No
workload in the suite has a cross-lane intra-epoch edge, so no workload could
tell the two appliers apart. A test needing modification would have been a
semantic difference owing an explanation, and there was none to give.

**What is now unblocked, and what is not.** Lanes are reorderable: nothing an
epoch commits depends on the order its lanes ran, and I25 is the standing check
that no workload has quietly reintroduced a dependence. That is the obligation
§4.3 listed as commit's, and it is discharged. The concurrent executive itself
is not written — `src/` still contains no threads, and lanes still run one after
another in a `for` loop. What changed is that running them elsewhere is now a
question about the executive rather than about the semantics.

### 4.6 The lanes are reordered, so that "reorderable" can be wrong

§4.5 ended by saying lanes are reorderable and nothing reorders them. That is a
bad place to leave a property. Canonical commit's whole claim is that an epoch
commits the same thing whatever order its lanes ran in, and the executive ran
them in plan order every time, so the claim was about a machine that only ever
chose one order — as was I25, which is what pays for it.

`scheduler::lane_order` is the choice made explicit. `LaneOrder::Plan` is what
a sequential interpreter does. `Reverse` inverts every pair, which is the
cheapest order that is maximally wrong. `Permuted(seed)` shuffles per epoch from
a counter-based generator, because reversal misses a dependence among three
lanes that both orderings of every pair happen to satisfy, and because a fixed
shuffle is something a workload can accidentally agree with.

**A lane's number does not move.** The plan is numbered first and walked second.
A number is a position in the plan, it stamps every event and effect the lane
produces (§4.2), and it chooses the lane's allocation partition (§4.3) — so the
same continuation gets the same number, the same partition and the same position
space under every order. Only the walk changes. This is what makes a reordered
run comparable to a plan-order one rather than merely different from it.

**What a reordered run owes, and what it does not.** It gives up I23's clause 2:
its raw trace's append order is no longer its position order. That is not a
concession made here — §4.2 wrote the exemption when the clause was written, and
said what replaces it: clauses 1 and 3, and I18 *after sorting by position*. The
checker asks which executive it is looking at and skips clause 2 for the ones
that do not owe it, rather than weakening the clause to something that would
hold of anything.

`order::in_position_order` is that sort, as an explicit step rather than folded
into `conforms`. Folding it in would let an implementation that quietly appended
out of order stop failing clause 2 and start passing a silently-sorted I18, and
the clause would mean nothing. The sort needs nothing but the trace — a position
is `(epoch, lane, sequence)`, positions are unique by I23 clause 1, and all of
it is decided before the work runs — so it is a derivation any implementation
performs on its own output, not a privilege of the reference.

**What the reordering found.** Nothing, which is the outcome §4.3 (3)'s
measurement predicted and is worth recording as a result rather than as silence.
Across the search and Expand workloads at cohort widths 1 through 16, under
reversal and two permutations, every run is I18-equivalent to its plan-order
run, every run leaves a legal state, and I25 holds of all of them.

That sentence stood until §4.12 went looking for a workload it was wrong about
and found one. It is a result about the workloads run here, all of which
allocate in the unbounded root domain; a bounded domain makes the same
reordering disagree. The machinery was right and the conclusion was
under-qualified — which is the argument for keeping the reordering rather than
against it, since it is what reported the disagreement.

**What it is evidence for.** `tests/lane_order.rs` compares the *effect log's*
application sequence by production position across orders, and requires it
identical. That is canonical commit stated as an experiment: the order lanes ran
in changed, and the order the epoch committed in did not. Putting the applier
back inside the lane loop fails that test, along with two others — so §4.5 is
now load-bearing rather than merely believed.

**This is not the concurrent executive.** It is single-threaded and deliberately
so. A permutation exercises exactly what threads need — no lane observes another
within its epoch, and commit does not care who finished first — while staying
deterministic, so a defect is a reproducible failure at a fixed place rather
than an intermittent corruption. The remaining work to run lanes on threads is
mechanical rather than semantic, and it is large: handlers take `&mut Kernel`,
so it means lane-local table shards merged at commit, which is the shape
partitioned allocation was built for.

### 4.7 The first threads

`src/` had no threads at all, which made every claim in §4 a claim about a
design rather than about a program. The first ones go where parallelism needs no
argument beyond rules the machine already pays for.

`CpuReferenceBackend::with_threads` evaluates a batch's elements across OS
threads. The safety argument is entirely §3.2's: a body is pure, reads the
frozen *input* array and never the output, and writes only its own element. None
of those were stated for threading — they are what makes I19 true of a gathering
body, since a body that could observe another element's store would make the
published result depend on the schedule. Having paid for them, an element's
output is a function of the frozen input and its own index, so splitting the
elements across threads cannot move a byte. The split is `chunks_mut`, each
thread owning a disjoint run of output elements over a shared immutable input,
with no synchronisation inside the loop and none needed.

Chunking is by element and not by byte: a boundary inside an element would hand
two threads halves of one element's output, and the interpreter writes an
element as a unit.

**It is a knob and not a default.** I20 makes this backend the definition every
other backend is checked against, and a definition should be the plainest
available reading of a body. A threaded run has to agree with the single-threaded
one; a default that already threaded would leave nothing to agree with.

**What this is not.** It is not the concurrent *executive*. An epoch's lanes
still run one after another — reordered (§4.6), but sequential. Parallel element
evaluation is the part of the machine that was provably safe to thread the
moment the body language was written; the executive is the part that is not,
because handlers take `&mut Kernel` and a lane both reads and writes kernel
tables. Threading that means lane-local table shards merged at commit, which is
the shape partitioned allocation (§4.3) was built for and which is not done.

### 4.8 Lane-local allocation

The concurrent executive's obstacle is not ordering any more — §4.5 settled
that, and §4.6 checked it. It is that a lane both reads and writes kernel state
through `&mut Kernel`, and a `&mut` cannot be held by several lanes at once.

Writes divide into two kinds, and only one of them is hard. Effects are already
produced rather than performed (§4.4) and applied at the epoch boundary (§4.5).
Allocation is the other kind, and §4.3 (2) established that it cannot be
deferred the same way: a step creates an entity and stores its `Ref64` in opaque
frame bytes, so it needs a real name before commit and a symbolic one cannot
survive the byte blob.

`GenTable::shard` is the answer partitioned allocation was always for. A shard
is an allocator over slot numbers in one partition that nothing else will mint —
the partition comes from the lane's position in the epoch's plan (§4.3), and no
two lanes share one. It holds only the slots the lane mints, so opening one
leaves the table fully readable: a lane reads pre-epoch state from the table and
its own new entities from its shard, which is exactly the read-back §4.3 (2)
requires. `GenTable::merge` folds it in, and appending in shard order reproduces
the slots the shard minted, because it based its numbering on the partition's
length when it was opened and is that partition's only allocator.

Shards are taken by `&self`, which is the property that makes them usable: an
epoch opens one per lane from a single shared borrow, and the lanes then own
them independently. `two_lanes_allocate_into_their_own_partitions_at_the_same_time`
fills two of them from two threads with nothing shared and no synchronisation.

**A shard does not recycle freed slots**, and that is the one place it differs
from allocating inline. Reuse means popping the partition's free list, and two
lanes popping one list is precisely the coordination partitions exist to remove.
So a shard appends and freed slots become available again after the merge. The
cost is that an epoch does not reuse slots freed during it; the effect is on
which slot numbers a run mints and not on what it does, which is the situation
partitioned allocation was already in — I18 compares up to a correspondence
between names (§2.6) rather than by reference. It is tested rather than left as
a comment, in both directions: inline allocation recycles, a shard does not.

### 4.9 An epoch's collectives run side by side

Element threading (§4.7) splits one collective's elements. That is the wrong
axis for the shape an epoch usually has: sixty-four small cohorts have too few
elements each to fill a thread, and `examples/metal_overhead` prices exactly
that case. The requests themselves are what should run side by side.

`CpuReferenceBackend::evaluate_epoch` does. The independence argument is the
element argument one level out: each request names its own frozen input and its
own output, and a body writes only its own element, so two requests share
nothing. The two axes are alternatives rather than layers — nesting them would
oversubscribe by the product of the counts — so a request evaluated as part of a
threaded epoch runs single-threaded.

Two properties are worth stating because getting either wrong produces bytes
that look right. Results stay in **request order**, since the caller publishes
result *i* into collective *i* and a completion-ordered return would put every
output in the wrong object. And **one failed request fails the epoch**, which is
the contract the sequential path already had: a partial epoch leaves the caller
holding some published outputs and some unstarted collectives with no way to say
which.

**What is still missing for a threaded executive.** The shards are the
allocator. What has no lane-local form yet is the rest of what a step writes,
and §4.10 is where that stops being a vague quantity.

### 4.10 What a step is allowed to touch

§4.3 could name execute/commit fusion as *the* obstacle to a concurrent
executive without saying much more, because a handler took `&mut Kernel` and
that settles it: a step holding the whole kernel mutably can do anything to it,
and several steps cannot hold it at once. Everything since has removed a reason
that has to be true — effects produced rather than performed (§4.4), applied at
the epoch boundary (§4.5), lanes reordered without changing a run (§4.6),
allocation given a lane-local form (§4.8).

What was never established is the part that sounds like bookkeeping: *what a
step actually does*. "A handler can do anything to the kernel" is a fact about a
type signature and not about the handlers, and the difference decides whether
the rest of the work is a week or a rewrite.

A step now takes a `LaneView`, and a `LaneView` offers **fifteen operations**.
Measured rather than chosen — they are what `cpu_scalar` and
`executives::ant_colony` already called, and the type was written around that
list. The count is not the point; the point is that the list is closed and the
compiler holds it closed, so an operation with no lane-local form is a compile
error inside a handler rather than something to find by audit. That is
`SOMA-CAPABILITIES.md`'s technique for closing the operation set and §4.1's for
sealing `Admission`, applied to the executive. `LaneView` has no `Deref` to
`Kernel` and no constructor outside the crate; both are `compile_fail`
doctests, with a passing one alongside them so a misspelled path cannot make
the failures vacuous.

Sorting the fifteen by what a concurrent lane would need turns the remaining
work into four items rather than a quantity:

- **Reads** (`continuations`, `epoch_number`, `future_value`, `object_bytes`,
  `read_u64_object`) need a shared borrow and nothing else. Two of them take
  `&mut` today only because reading is a governed effect whose authority
  decision is traced (I10c) — which is a real write, and the reason a concurrent
  lane needs the lane-local trace buffer §4.2's position scheme already
  anticipates.
- **Allocation** (`create_process`, `create_continuation`, `create_future`,
  `create_object`) has its lane-local form in §4.8's shards. Wiring is
  mechanical.
- **Own-frame writes** (`host_payload_mut`, `object_bytes_mut`) are disjoint
  across lanes by I8, which the checker already enforces — no two continuations
  share a frame object.
- **Cross-lane writes** (`enqueue_message`, `receive_message`, `resolve_future`,
  `await_future`) are the genuinely hard ones. They touch state another lane may
  touch, so they need journalling in the shape §4.4 used for bin entries, and
  I25 is what makes deferring them safe.

**Four operations, not a kernel.** That is the result of writing it down, and it
is the answer §4.3 could not give.

**At this point in the sequence the view still borrowed the kernel mutably.**
Retyping the handlers changed no run; §4.18 later uses one mutable view per
isolated snapshot and turns the closed operation surface into a replay journal.

### 4.11 A lane produces its events too

§4.10 sorted a step's fifteen operations into four groups and put the five
*reads* in the easiest one — "a shared borrow and nothing else". That was true
of four of them. It was not true of `object_bytes` and `read_u64_object`, which
take `&mut` because reading is a governed effect whose authority decision is
traced (I10c). A read is a write, and what it writes is the trace: `logical_time`
is one counter the whole run draws from, and `trace` is one vector the whole run
appends to. Two lanes reading two different objects contend on both.

So the trace gets §4.4's treatment, which is the same treatment the effects got
and for the same reason: a lane **produces** events into `lane_trace` and the
boundary appends them. `leave_lane` drains, in emission order.

**The clock is handed out at the drain, not at emission.** That is the part
worth stating, because it is what makes the counter host-side rather than
something a lane touches. I23 already says `logical_time` carries no information
beyond `(epoch, lane, lane_sequence)` — a position is assigned locally and
needs nothing shared. This makes that structural: a lane produces events that do
not have a `logical_time` yet, and one exists only because the sequential
interpreter is the reference and replay reads it.

**The drain does not sort.** Appending in emission order is what makes this
change no run, and sorting by position instead would be a different and worse
claim. §4.2 wrote I23's clause 2 as a property of the reference and exempted
implementations whose append order is not their position order — §4.6's
reordered lanes are exactly such a run. A reference that re-serialised its own
trace into position order would make clause 2 hold of anything that emits, which
is the same objection §4.6 raised against folding `in_position_order` into
`conforms`.

**One shared-clock read survives, and it is not the trace.** A sent message
carries a `logical_timestamp` (§11 of the P1 ABI), stamped from the counter
inside the lane that sends. Nothing reads it and nothing orders by it —
per-pair ordering is `sender_sequence`, which is what I6 checks — so it is
left reading the count, including what the lane has produced and not yet
drained, so the value a run stamps is the value it stamped before. Recorded
here rather than quietly changed: a concurrent executive either drops the field
or stamps it from a position, and that is an ABI decision rather than an
implementation one.

**Entering a lane over an undrained buffer is rejected**, not assumed away. It
would append one lane's events under the next lane's number, and a position is
the only thing that says which lane did what (I23 clause 3) — so the trace it
produces reads as legal. One branch per lane, not per event.

**What this does not do.** The buffer lives on the kernel, so `LaneView` still
borrows the kernel mutably and the reads still take `&mut`; a lane owning its
own buffer is what would change that, and it is the same move as a lane owning
its own table shards (§4.8). Five workloads produce byte-identical output, which
is the whole evidence offered. What changed is that the *read* category of §4.10
now writes one place, and that place is per-lane.

### 4.12 An epoch's lanes share a quota

§4.6 reordered an epoch's lanes and reported that **it found nothing**, and
recorded that as a result rather than as silence. It was a result about the
workloads it ran. Every one of them allocates in the root domain, which is
unbounded, and a bounded domain is the case it did not have.

A step creating a process consumes its domain's quota. Two lanes of one epoch
creating processes in one bounded domain therefore race for it — and the loser
is not slowed down, it is refused. The same workload under `LaneOrder::Plan` and
`LaneOrder::Reverse` refuses a *different pair* of processes and faults a
different pair of continuations. Both runs leave a legal state. Nothing but a
comparison of the two runs reports it, which is exactly the shape §4.6 built the
comparison for.

**I25 clause 1 does not see this**, and that is the interesting part rather than
an oversight to be embarrassed about. Clause 1 reads the semantic order and asks
for a ≺ edge joining two lanes. Here there is none: nothing is sent, resolved,
delivered or woken between the lanes. `cross_lane_edges()` is empty for a run
that demonstrably depends on its lane order. The dependence is carried by a
counter, not by an event, and an order built from the trace cannot contain it.

**I25 gains a second clause.** No two lanes of one epoch may be decided by one
bounded resource. The condition is that **one lane got the resource and a
different lane was refused it**, in the same epoch, and each half is doing work:

- Two lanes with room to spare decide nothing — every draw succeeds under every
  order.
- Everyone refused decides nothing either. A mailbox that was already full when
  the epoch began refuses all four of its senders whatever order they ran in.
- One lane refused after taking the last of something itself has raced nobody.

The clause fires on exactly the runs where the two lane orders disagree and on
none of the ones where they agree — checked in both directions, for both
resources.

That condition is now the one clause 2 asks of these two resources rather than
of all of them, and the first bullet above is where it turned out to be a
statement about *them*: a resource that hands out identified items rather than
interchangeable units decides something between two lanes that both succeed.
§4.13 is that case and carries the corrected clause.

**A refusal is now traced.** `ProcessCreationRefused` carries the domain in
`subject` and the quota in `auxiliary`; `MessageSendBlocked` carries the
receiver in `causal`, as `MessageSent` does, and the capacity in `auxiliary`.
Two reasons each, and the invariant is the second one. The first is that a
refusal is a thing that happened and the trace had no way to say it: a run
showed a process faulting, or a sender that had not sent, and could not say the
machine had told it no. The second is that the clause has to tell a bound that
bit from a bound with room, and only the refusal says which. `ProcessCreated`
gained the domain in `subject` for the same reason — a reclaimed process cannot
be asked which domain it drew on.

**The second resource was a prediction, and it held.** This section first said a
quota was the case the machine had, and named "a mailbox capacity a second lane
fills" as what to look at next. It behaves identically: several senders replying
to one receiver in one epoch with one slot free, and the sender that gets it is
the one whose lane ran first. Clause 1 is blind for the same reason — occupancy
is not an event — and the two runs are not I18-equivalent.
`tests/mailbox_capacity.rs` is the experiment; it is also where the clause's
condition got its final form, because "two lanes drew and one was refused" fires
on a mailbox that was already full, which is not a race.

**What that suggests about the rest.** Neither resource was found by a failing
test; both were found by asking what a lane reads a *decision* off. That
question is the tool, and the remaining candidates are the ones where an
operation can say no: a table that can refuse to allocate, and a supervision
queue with a bound. Neither is reachable from a step today.

That last sentence was wrong when it was written, and §4.13 is the case it
missed. The candidates were listed by looking for a *new* bounded thing, and the
one that was there already is this same mailbox read from the other end: a
receive that finds it empty says no exactly as a send that finds it full does,
and `receive_message` is one of §4.10's fifteen. The lesson is the narrower of
the two available: a list of "where else can an operation say no" has to be
taken over operations, not over resources, because one resource can refuse two
different callers for two different reasons.

**The section heading is now too narrow.** It says quota; the clause is about
bounded resources, and a quota is one of two. It is left as it is because §4.12
is what the commits and the handoff point at, and renaming a section to make it
tidier is how a reference stops matching what refers to it.

**What is *not* this clause.** `processes_created` is incremented by every
allocation, in the root domain as much as a bounded one, so two lanes writing it
is a data race in every workload. That one is mechanical: the increment
commutes, so it is a journalled effect in §4.4's shape, applied at commit. What
does not commute — and what no journal fixes — is the *decision* read off the
counter, because a lane needs the answer before it can continue and §4.3 (2)
already established that allocation cannot be deferred.

**The bug this found.** `LaneView::create_process` was infallible, so
`DomainQuotaExceeded` inside a handler reached an `expect` and aborted the host
process. The machine has a word for a step that cannot proceed — v0.2 §6.3's
`Fault`, with a supervision model behind it — and was not using it. The lane
surface's `create_process` returns a `Result`, and both spawning handlers store
their frame and fault. That is the second time §4.10's list has paid: the first
was learning what a step touches, and this is learning that one entry on it can
fail.

**What this costs.** A workload with a contended quota is one the concurrent
executive cannot take, and I25 names it at the point the workload does it. That
is the standing §4.3 gave clause 1 and it is unchanged: the kernel does not
refuse such a workload, the checker reports it.

### 4.13 The same mailbox, drained

§4.12 closed by naming the remaining candidates and saying none of them was
reachable from a step. One was, and it was not a new resource: it was the
mailbox §4.12 had just finished filling, read from the other end. A send that
finds a full mailbox says no; a receive that finds an empty one says no just as
definitely, and `receive_message` is one of the fifteen operations §4.10 sorted.

The experiment is `mailbox_capacity.rs` with the arrow reversed. One process, four
read-only continuations at a resume point that receives — so I13 admits all four
in one epoch and each is a lane — and one message waiting. The message goes to
lane 1 under `Plan` and to lane 4 under `Reverse`, the other three park, and the
two runs are not I18-equivalent. Clause 1 is blind for the third time and for the
third version of the same reason: `cross_lane_edges()` is empty, because how many
messages are in a mailbox is not an event.

`MessageReceiveBlocked` is `MessageSendBlocked`'s mirror, with the same two
justifications. A receive that found nothing left no record at all — a trace
showed a continuation that started and then waited, and could not say whether it
was waiting on a future or on an empty mailbox — and the checker cannot tell a
mailbox several lanes drained between them from one that had a message for each
of them without it. It carries the mailbox's owner in `process`, as
`MessageReceived` does, which is what lets the clause key both on one resource.

**Its `auxiliary` is a constant, on purpose.** A receive is refused by emptiness
and nothing else, so the occupancy that refused it is always zero. The more
informative number available — how many receivers are already parked — is the
wrong one to record, and the reason is worth keeping: that depth is the *parking
order*, and parking order differs between two lane orders in runs whose epoch
outcome does not. Putting it in the trace would make those runs disagree over
something no epoch decided. The same observation applies to `full_waiters` and is
why §4.12's already-full null holds: a queue of losers is order-dependent state
that this trace does not expose, and a later epoch that drains it is a race in
that epoch rather than in the one that parked them.

**The clause was asking the wrong question, and this is what showed it.** Four
receivers and four messages looks like the "room for everyone" null that a quota
and a capacity both have — every lane succeeds, no refusal anywhere — and it is
not one. The runs disagree. A quota and a capacity dispense **interchangeable**
units: one slot in a mailbox is like every other, and which one a sender got is
not in the trace. A mailbox's occupancy dispenses **identified** items — this
message, from that sender, with that sequence number — so two lanes that both
succeed still took different things, and which lane took which is precisely what
the order decided.

So clause 2 now asks a condition per resource kind rather than one condition:

- **Interchangeable** — a winner and a *different* loser, unchanged, which is
  what §4.12 established and what both of its nulls still turn on.
- **Identified** — a winner and any other lane that drew, won or lost. A refusal
  is one way to lose; getting the other message is another.

The nulls change shape with it. For a drained mailbox, "everybody succeeded" is
not a null and "everybody was refused" still is — an empty mailbox refuses every
receiver under every order — and the null that says the clause is about *one*
mailbox is two lanes each draining their own.

**A second defect, in the pairing.** The clause took the lowest-numbered winner
and looked for a loser that was not it. That misses a lane which both won and
lost: if it is also the lowest winner, asking only about it finds nothing, while
a higher-numbered winner that got in ahead of it plainly decided the outcome.
Every pair is now considered, and the minimum over pairs is taken — which also
makes the report the same text every run, where picking a loser out of a
`HashSet` did not. No workload reaches the shape, because a handler sends once,
so it is checked by appending the three events a workload that sent twice would
have produced.

**What is unchanged.** No existing workload emits a blocked receive: the suite's
receiving workloads ingest the message before the receiver runs, so every receive
in them finds one. Eight example reports are byte-identical, and the probe that
says so is stronger than the reports — making the emission `panic!` and running
the whole suite reaches it from exactly one test file, this one.

### 4.14 A future takes one value

§4.13's correction was to the method: enumerate the *operations* that can say
no, not the resources. `LaneView` offers fifteen and the compiler holds the list
closed (§4.10), so that enumeration is finite and can be walked. `resolve_future`
is on it, and single assignment (§12) is precisely an operation that says no.

Four processes hold `RESOLVE` on one future and each publishes a value computed
from its own input, in one epoch. The future takes the value of whichever lane
ran first — lane 1 under `Plan`, lane 4 under `Reverse` — and the other three
fault. Both runs leave a legal state, so only the comparison reports it, and
clause 1 is blind for the fourth time. This variant of the reason is the
starkest: what the losing lanes read is that a cell *had already been written*,
and a write that did not happen is not something an order can draw an edge from.

`FutureResolutionRefused` is the fourth traced refusal, and single assignment is
the one that was enforcing itself entirely silently — the loser faulted, and
`ProcessFailed` does not distinguish a program that went wrong from one told the
value was already published. It carries the future in `causal`, as
`FutureResolved` does, and the value the lane built and did not publish in
`subject`.

**This is the resource that does not discriminate.** §4.13 split clause 2's
condition in two, and a future has exactly one unit, so there is never a second
winner and the two conditions name the same set. It is registered as dispensing
interchangeable units — what a resolver wins is a permission to publish, and one
such permission is like another — but nothing in any run can tell that choice
from the other one. Recorded because a resource that cannot distinguish the two
rules is not evidence for either, and a later reader deciding where a fifth
resource belongs should not cite this one.

**What the enumeration says about the rest of the fifteen.** Walking the list
the way §4.13 said to leaves two operations that read a decision and are *not*
reachable in any workload, for a reason worth stating precisely — it is about
the handlers, not about the machine:

- `await_future` returns `AlreadySettled` rather than registering a waiter, which
  is a decision read off a future another lane may resolve in the same epoch. No
  handler can reach it: the only awaiting handler creates the future in the same
  step it awaits, so the resolver cannot have run yet. A handler that awaited a
  future it did not create would reach it immediately.
- `future_value` reads a resolution with no event at all, and the same argument
  applies for the same reason. (§4.15 reaches it and §4.16 finds that reaching
  it was not the same as racing it: the read there follows an await, so the
  resolution is already in the past. A poll is the case that races, and it is
  the one nothing could report.)

So the blind spot is a property of the handler set, and it moves the moment the
handler set does. That is a different kind of "not reachable" from §4.12's, which
was a claim about the machine and was wrong.

The remaining twelve either allocate (partitioned, §4.8), write a frame no other
lane holds (I8), or send and receive, which §4.12 and §4.13 covered.

### 4.15 An await that reads a future somebody else is resolving

§4.14 ended by naming two decisions no *handler* could reach and saying exactly
why: the only awaiting handler creates the future in the step it awaits, so a
resolver cannot have run yet. That was a claim about the handler set, and the
way to test a claim about the handler set is to write the handler. `JOIN_AWAIT`
awaits a future named by its frame — somebody else's — and `JOIN_RESUME` reads
what it took. Both blind spots close at once, because the second one,
`future_value`, is the read the second half does.

One resolver, one awaiter, one future, one epoch. Under `Plan` the resolver is
lane 1, so the awaiter finds the value published and continues without parking.
Under `Reverse` the awaiter goes first, parks, and the resolver wakes it. **The
two runs leave the same state** — the awaiter runs `JOIN_RESUME` in epoch 1 and
reads the same value either way — so nothing is wrong with either run's result.
What differs is only the route, and the route is what I18 is about.

**This is the first of the five that clause 1 can see, and it is blind in the
reference order.** Under `Reverse` the resolver's wake names the parked
continuation, so the awaiter's `ContinuationWaiting` and the resolver's
`ContinuationReady` are two events of one continuation's program order emitted
from two lanes, and `cross_lane_edges()` is finally not empty. The reason it is
not is worth stating, because it is what the other four lacked: a wake is the
only one of these events that names **another lane's continuation** rather than
only the resource. A resolve names a future, a send names a mailbox, a refusal
names what refused it — none of them says whose lane was affected.

Under `Plan` nothing is woken, there is no edge, and clause 1 reports nothing at
all. So a plan-order run of this workload passes clause 1 cleanly, and *that*
run is the reference. A single run satisfying clause 1 was never evidence of
independence; here is a workload where the run you would naturally do is the one
that cannot report it. §4.6's discipline — run it again in another order and
compare — is not an extra check on top of I25, it is what makes I25's first
clause reach this case.

**The clause the other order needs, and the correction it carries.** Clause 2
reports the `Plan` run, and needs a new event to do it: an await that did not
park emitted nothing to tell it from any other yield. `FutureAwaitSettled` is the
fifth traced decision and **the first that is not a refusal**. Every one so far
existed because an operation said no and `ProcessFailed` or a bare park could not
say so; this operation *succeeded*. `await_future` returns normally by both
routes. What another lane decided is which of two states of the future it read.

That widens the enumeration §4.13 established, and the widening — not the run —
is what this section is for:

> Enumerate the operations whose **result another lane can decide**, not the
> operations that can say no.

A refusal is the special case where one of the results is a failure. It was the
conspicuous case — it faults a step, so it announces itself — which is why four
rounds of walking the list found refusals and stopped. `await_future` was on the
list each time and was passed over each time, because nothing about it fails.

Registering it in the checker took one thing that was not obvious. `Bounded` now
holds an entry, `FutureSettlement`, that is **not a bounded resource**: nothing
is dispensed, the awaiter draws no unit, it reads which state the future is in.
It belongs in that clause anyway, and the reason is that the clause's subject was
never boundedness — it is a lane's outcome being decided by another lane of its
epoch, and a bounded resource was just the first mechanism found for that. The
same future is therefore keyed twice, once as an assignment two lanes may
contest (§4.14) and once as a state one lane writes and another reads, and a
`FutureResolved` counts as a win under both.

**The nulls.** Three, and each rules out a different reading:

- *Settled before the epoch.* The host publishes, the awaiter finds it settled
  under every order, no lane resolved anything. Silent — a state nobody wrote
  this epoch decided nothing — and it is also the run that shows the event alone
  is not the report.
- *Different futures.* The resolver publishes into one of its own and the
  awaited future is untouched, so the awaiter parks under both orders. Silent:
  the clause is about one future.
- *No resolver.* The awaiter parks and stays parked; the wake, if it ever comes,
  comes in another epoch. Silent: the clause is about one epoch.

**What is unchanged.** No existing workload reaches the settled arm — the probe
is the emission made to `panic!`, and with `--no-fail-fast` the whole suite
reaches it from two files: this section's, and `kernel_edge_cases`, which calls
`await_future` on the kernel directly rather than from a step. That second one is
the distinction §4.14 drew, showing up as a difference between test files: the
machine could always do this, and until now no handler asked it to. Ten example
reports are byte-identical.

**What this leaves.** Re-walking the fifteen under the widened question, the
candidate it now surfaces is not an operation at all but a read: `continuations()`
hands a step the whole table, and a continuation's status is written by
`apply_step_result` *inside* the lane that produced it. A handler that read
another continuation's status would see a write from a lane of its own epoch,
and no clause would report it — no event is emitted, so there is nothing to key
on. No handler does; every one of them reads its own `run_class` and its own
frame. That is the same kind of "not reachable" as §4.14's, with the same
expiry: it is a fact about the handlers. Unlike §4.14's, the fix is available
without waiting for a workload to prove it — the view could hand a step its own
descriptor rather than the table — and it is not done here because narrowing
`continuations()` is a change to the fifteen and belongs with the concurrent
executive that will need it.

### 4.16 A future looked at, and the epoch that did not report it

Re-walking the fifteen under §4.15's wider question — which operations have a
result another lane can decide — turns up one the narrow question had no reason
to stop at. `future_value` cannot fail and does not block. It was also the only
read on `LaneView` that took no capability and left no record.

§4.15 did reach it: `JOIN_RESUME` reads a future it did not create. It never
*raced* it, because that read comes after an await, and an await that parks or
returns settled has already put the resolution in the past. Take the await away
and the read is a poll, which is a perfectly ordinary thing for a program to do
— check whether a result is ready, carry on either way — and now the resolving
lane of the same epoch decides what it returns.

**Everything this project built to catch that was blind.** One resolver lane,
one polling lane, one future, one epoch:

- clause 1 has no edge to find: nothing parked, so nothing was woken, so no
  event names another lane's continuation. Unlike §4.15, this is not a
  limitation of what the trace records — there genuinely is no interaction
  between the two lanes beyond the read.
- clause 2 had no event to key on.
- and **the run comparison was blind too**, which is new. What the poll saw went
  into its frame, and a frame is not observable behaviour. `conforms_traces`
  reported nothing. Two runs, I18-equivalent, leaving different state.

That last one matters more than the clause. §4.6's discipline — run the workload
again in another order and compare — is what found all five earlier races, and
it is the fallback whenever a clause is blind. Here the fallback fails as well,
so the run that was decided by lane order has no detector of any kind.

**Making it undeniable.** `POLL_ACT` is a second resume point that sends a
message if the poll saw a value. Now the runs do disagree — and they disagree in
epoch 1, while the epoch that decided everything is epoch 0, which every clause
declares clean under both orders. An epoch passing I25 is supposed to mean its
lanes could have run in any order. Here it says so and is wrong, and the
evidence arrives an epoch later, attached to a lane that did nothing wrong.

**The fix is the one the machine already applies to its other reads.**
`object_bytes` and `read_u64_object` are governed: they authorize, they trace
the decision, and `LaneView` takes them by `&mut` for exactly that reason.
`future_value` now joins them. It authorizes `AWAIT` — the right that already
means "may observe this future", since a poll is the non-blocking form of the
same observation — and emits `FutureStateObserved`.

Two details of that event are load-bearing:

- **It records what the poll saw**, in `auxiliary` and `subject`. A poll that
  found the future still *pending* was decided by the resolving lane exactly as
  much as one that found the value — it would have seen the value under the
  other order — so recording only the resolved case would leave half the runs
  unreportable. That is the same mistake in miniature as the one this section
  exists to fix.
- **A denied read faults**, rather than reading as "not resolved yet". The
  ungoverned read collapsed those two answers into `None`, and they are
  different answers. A null future reference is a third thing again — the frame
  names nothing to look at — and skips the read entirely, which is what keeps
  handlers holding an empty frame behaving as they did.

Clause 2 keys the event under `FutureSettlement`, beside §4.15's settled await:
same resource, same question, two operations that read it. Both orders of the
poll workload now report, in epoch 0.

**The nulls** are §4.15's three, with the same readings — published before the
epoch (nobody won it), nobody resolving at all, and the two lanes on different
futures — plus one this section needs of its own: a poller holding no `AWAIT` is
denied and faults, which is what says the read is governed rather than merely
traced.

**What is not unchanged, for the first time in five rounds.** The four refusals
and the settled await were emitted by no existing workload, and each round could
say so with a probe. This one is emitted by fourteen of the suite's test files —
`future_value` is what `expand_resume_1` calls to collect its heuristic result,
so the Expand workload's trace grows by three events per read (the grant, the
effect, the observation). Nothing failed and the ten example reports are still
byte-identical, which is the useful part of the result: fourteen files' worth of
workloads look at futures, and not one of them looks at a future its own epoch
is resolving. The clause fires nowhere except where it was built to.

### 4.17 The read that was not a race, and the surface that would have been one

After §4.16 there was one read left on `LaneView` that neither authorized nor
traced: `continuations`, which handed a step the whole continuation table by
shared reference. The five before it each turned out to be a race once the
question was widened, so the honest thing is to report that this one is not.

**Measured first, and the measurement is negative.** Three call sites, all in
`cpu_scalar`, and every one of them named the continuation the lane was already
running. Between them they read two fields:

- `run_class`, which Phase E fixed when it built this lane's cohort, and which
  Phase F reads again on the host's side of the step. Nothing between those two
  reads can change it.
- `frame`, which is written once when the continuation is created and never
  again.

So §4.6's fallback — run it in another order, compare — has nothing to find, and
this time that is a fact about the handlers rather than a limit of the trace. No
step ever named a continuation other than its own.

**What the table offered was the ability to.** Descriptors really do change
between lanes of one epoch. Bin entries, statuses and run classes go through
effects and land at the boundary (§4.5), but `apply_step_result` does not: it
runs inside the lane loop, and a fault there carries containment into a
sibling's status and withdraws that sibling's journal entries mid-epoch. A step
that read a sibling descriptor would be reading precisely that, and the three
blindnesses would be §4.16's, unchanged: no ≺ edge, because reading is not
waking; no event to key clause 2 on, because the read emitted none; and the
answer disappearing into a frame, which is where §4.16 established that
`conforms_traces` stops being able to see.

**So this one is narrowed rather than governed**, which is the first time the
answer has not been "make it a governed read". Governing it would put an
authority pair and an event on every frame load and store — several per step,
the highest-frequency read in the machine — to record an answer the epoch loop
is already holding. Narrowing it makes the cross-continuation read a compile
error instead of a traced one, which is §4.10's own technique applied to the one
operation §4.10 left open.

The two fields are passed in. `run_class` becomes an argument to `dispatch`.
`frame` becomes `LaneView::frame`, a copy taken before the step begins.
`continuations` is gone from the view, with a `compile_fail` beside the three
that were already there.

**The nulls**, in the shape this section needs rather than §4.15's: the
constructor's `compile_fail` gained its new argument, so that it still fails on
`new` being crate-private and not on its arity — a block that fails for the
wrong reason passes vacuously, which is the same reason the positive block
exists at all.

Eight handlers now take `_cont`. Underscored rather than removed, because the
uniform signature is what makes `dispatch` a switch rather than eight unrelated
calls — and because the count is the measurement. Eight of the handlers wanted
their continuation for nothing but asking the table about it.

**Nothing in any trace moved.** No event was added, 402 tests pass, and the ten
example reports are byte-identical — not "still byte-identical after a growth of
three events per read", as §4.16 had to say, but unchanged event for event. That
is the claim of the section: it changes what a step can express, not what any
step does.

`LaneView` still offers fifteen operations, and its reads are now closed in a
stronger sense than being counted. Three are governed — `object_bytes`,
`read_u64_object`, `future_value` — and the other two, `epoch_number` and
`frame`, read nothing a lane is able to write.

---

### 4.18 Speculative concurrent epochs

The CPU executive now has the optimistic implementation §4.3 anticipated. It
clones one immutable pre-Phase-F state into isolated worker snapshots, runs the
lanes on scoped OS threads, and records every call through `LaneView` together
with object, future, process, mailbox, domain, and allocator-partition accesses.

The validator rejects every read/write or write/write overlap. It also replays
the concrete operation journals on a disposable kernel before touching the real
one. A conflict or replay mismatch discards all snapshots and runs the original
plan-order loop. A valid history replays through the ordinary kernel operations
and `apply_step_result` in lane-number order, so authority checks, trace events,
effects, allocation identities, and terminal state all cross the same semantic
boundary as the reference implementation.

The four writes §4.10 left open are no longer special cases outside the model:
message enqueue/receive use mailbox keys and future resolve/await use future
keys. Allocation uses the plan-derived partition and process creation also
writes its domain. Fault containment and pre-existing cancellation remain
reference-only because their footprint is intentionally broader than the closed
set declared so far.

This is an opt-in CPU executive, not a claim that optimism is always faster.
Snapshot cloning, a validation replay, and thread startup lose badly on small
steps. The measured crossover and controls are in
`docs/SPECULATIVE-EPOCHS.md`; the local M4 Pro sweep reaches 5.78× at eight
million arithmetic iterations across eight lanes and about 0.1× at one thousand
iterations per lane.

### 4.19 Discovery searches the implementation

The first real Discovery target is SOMA's own evaluator path. The experiment
searches placement, interpreted and native-compiled CPU thread count, epoch
command grouping, Metal scratch reuse, and threadgroup width across light,
medium, and compute-heavy bodies.
Evaluator construction and deterministic input preparation are content-addressed
nodes shared across configuration hypotheses. Wall-clock trials are
`Observation` nodes and therefore retain multiplicity even when their values
are byte-identical.

Acquisition is deliberately outside replay: each configuration is warmed, then
trials run in a deterministic rotating order. Every trial records elapsed time
and the digest of the published output. A disagreement between configuration
outputs invalidates the study before replay. The literal and optimized
Discovery executors then consume the same captured evidence, and D1–D7 require
identical terminal scientific state.

This does not make wall-clock time deterministic and does not place timing in
the SOMA semantic core. It makes the comparison scientifically controlled: one
noisy acquisition, two executions over exactly the same observations. The
protocol, negative controls, and M4 Pro measurements are in
`docs/SELF-TUNING.md`.

The resulting native CPU backend is not benchmark scaffolding. It lowers every
validated evaluator-body operation to Cranelift machine code, including gathers
and structured/divergent loops, and never substitutes the interpreter inside a
native timing. I20-style agreement keeps the scalar interpreter as the
definition.

## 5. C — distributed, trace-equivalent

**Started, and still the long pole.** Node-qualified identity, signed remote
delegation, logical-epoch validity, version pinning, and live revocation checks
are implemented and tested. `docs/DISTRIBUTED.md` fixes the failure model before
transport: a partition is not a process fault, and declared node loss discards
only uncommitted epoch journals before notifying remote supervisors.

The first transport is also implemented. `RemoteBatchBackend` sends signed,
content-addressed evaluator requests over framed TCP; the worker authorizes
before consulting an apply-once response ledger. Remote placement is distinct
from CPU and accelerator placement, and unavailable, lost, protocol, authority,
and evaluator failures cannot be confused with successful bytes.

The model is unusually well-positioned for the hard part: frames are durable
position-independent byte blobs (v0.2 §1.1), so a continuation can resume on a
node that did not suspend it without migrating register state.

What is not done:

- **Node identity in kernel tables.** `RemoteRef` gives the wire a disjoint node
  namespace without overloading allocator partitions; kernel entity ownership
  and migration still need to carry it.
- **Capability spaces across nodes.** Remote grants prove and attenuate
  authority, pin versions and logical epochs, and observe live revocation at
  use. Replicating revocation observations and connecting grants to every
  stateful remote operation remain.
- **Node-loss integration.** The partition/loss semantics are fixed in
  `docs/DISTRIBUTED.md`. Process ownership, system-only idempotent loss
  declaration, containment, rejected re-placement, `ProcessLost`, and distinct
  remote-supervisor notices are executable. Coordinated multi-node epoch
  journals still need transport integration.
- **Escrowed channel payloads** assume the kernel can hold a `READ` root for an
  in-flight message. Across nodes, "the kernel" is plural.

**Exit criteria.** A two-node run of the streaming graph and the supervision tree
is I18-equivalent to the single-node run; a node killed mid-epoch produces a
defined, tested outcome; and no test passes by routing all work to one node —
the control is a run where every process is remote from its supervisor.

The placement half of that exit criterion passes. The streaming graph is
I18-equivalent with both channel peers remote from its coordinator, and the
supervision tree is I18-equivalent under notify, escalate, and restart with all
four parent/child edges crossing nodes. Later implementation added authenticated
full-journal transport and authoritative owner-side future, bounded-channel,
object, terminal-supervision, and real Kernel mailbox-ingress services, including
two owner Kernel epoch loops. This still does not close C: generic remote
`LaneView`, remote process lifecycle/recovery, durability, and a multi-resource
application remain open; see `DREAM-COMPLETION.md`.

---

## 6. D — performance

Hardware measurement is now executable, but the completion result is negative.
The ant collective is slower end to end than its direct host control, and
independent grouped-versus-generic resident runs reverse ordering despite exact
outputs and competent controls. No reproducible G5 speedup has been established;
raw captures and caveats are in `PERFORMANCE.md` and `DREAM-COMPLETION.md`.

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
| I18 schedule conformance | checked | replaces trace equality (§2), up to a renaming (§2.6) |
| I19 placement neutrality | checked | cohort width, with a non-vacuity null |
| I20 backend agreement | checked | CPU interpreter is the definition |
| I21 bounded progress | checked | no withholding, plus a starvation bound |
| I22 admission determinism | checked | the decision is a function of the candidate set (§4.1) |
| I23 position-derived emission | checked | the trace's order needs no shared clock (§4.2) |
| I24 effect-mediated commit | checked | bins are written by an applier, in plan order (§4.4) |
| I25 lane independence | checked | two clauses: no ≺ edge joins two lanes of one epoch (§4.5), and no lane's outcome is decided by another through one resource — a domain quota, a mailbox's capacity, a mailbox's occupancy, a future's one assignment, or a future's settled state, whether awaited or merely looked at (§4.12–§4.16) |

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
- **Wider floating-point bodies.** Deterministic binary32 arithmetic,
  comparison, selection, and explicitly typed float locals are implemented.
  Wider and mixed precision remain outside v0.3.
- **Relaxed-determinism and wall-clock contracts.** Both contradict clauses the
  model relies on.
- **A general-purpose surface language.** v0.2 §7 item 1 says
  "language/compiler for arbitrary evaluator bodies". v0.3 delivers the
  compiler and a minimal body language. A programmer-facing language for whole
  SOMA programs is a separate project.
