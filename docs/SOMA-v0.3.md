# SOMA v0.3

**Status:** partially implemented. §1–§3 are specification and are
machine-checked, as are §4.1, §4.2, §4.4 and §4.5. §4's four obligations are
discharged; the concurrent executive they were for is not written. §5–§6 are
scope.

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
| **B** | Persistent device-resident scheduler | **started** (§4) — semantics done, executive not |
| **C** | Distributed / multi-node implementation | scope (§5) |
| **D** | Performance work on real hardware | scope (§6) |

Eight new clauses are now checked — I18 through I25 — and v0.2's only
`[modelled]` clause is gone. The test suite went from 151 to 284.

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
- **Integer-only.** Admitting `f32` would force I20 to either demand
  bit-identical results across backends — constraining what a GPU may do — or
  weaken to a tolerance, which guts it as an invariant. Deferred until there is
  a reason to pay for it.
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

What is *not* done is the executive that all of this was for. Lanes are
reorderable and §4.6 reorders them — deterministically, on one thread — so the
property is checked rather than claimed. Running them on several threads is
mechanical rather than semantic work, and it is not done.

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

**What is still missing for a threaded executive.** The shards are the
allocator. What has no lane-local form yet is the rest of what a step writes —
mailboxes, futures, capability spaces, object payloads — and the read-through
view that would let a handler take a lane rather than `&mut Kernel`. That is
mechanical work across the handler surface rather than a further semantic
question, which is the state §4 has been trying to reach: everything left is a
refactor, and nothing left is an unanswered question about the model.

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
| I18 schedule conformance | checked | replaces trace equality (§2), up to a renaming (§2.6) |
| I19 placement neutrality | checked | cohort width, with a non-vacuity null |
| I20 backend agreement | checked | CPU interpreter is the definition |
| I21 bounded progress | checked | no withholding, plus a starvation bound |
| I22 admission determinism | checked | the decision is a function of the candidate set (§4.1) |
| I23 position-derived emission | checked | the trace's order needs no shared clock (§4.2) |
| I24 effect-mediated commit | checked | bins are written by an applier, in plan order (§4.4) |
| I25 lane independence | checked | no ≺ edge joins two lanes of one epoch (§4.5) |

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
