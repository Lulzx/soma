# SOMA Semantic Specification v0.2

**Status:** draft. Machine-checked where marked; incomplete where marked.

SOMA is an abstract machine for irregular concurrent computation. It is defined
here without reference to any hardware: no SIMD width, no device, no host, no
placement. Those belong to *implementations* of SOMA, of which the CPU reference
interpreter in this repository is one and a GPU operating system might be
another. If a clause below mentions hardware, it is a defect in the clause.

The question this version exists to answer is whether persistent processes,
object capabilities, continuations, dataflow readiness, and collectives form a
coherent programming model. Coherence is testable, so most of this document is
written to be executable.

## 0. How to read this

Every clause is one of:

| Marker | Meaning |
| ------ | ------- |
| **[checked]** | Stated here, implemented, and machine-verified by `soma::semantics::invariants` against the reference interpreter after every transition. |
| **[modelled]** | Implemented in the reference interpreter, but not yet expressible as a state predicate, so it is verified only by targeted tests. |
| **[absent]** | Named by the model and *not* implemented. The machine does not have this property. |

The markers are load-bearing. An [absent] clause is not a promise about future
work being nearly done; it means a program relying on it will not be protected.
`tests/semantics.rs` asserts both directions for every [checked] clause: the
reference model satisfies it, *and* the checker catches a state that violates
it. A checker that cannot fail is not evidence.

---

## 1. Machine state

A SOMA machine state Σ is:

```text
Σ = ⟨ E, τ, T, Q, M, W, S, A ⟩

E  epoch counter
τ  logical clock and event trace
T  entity tables, one per kind: processes, objects, continuations,
   futures, capabilities
Q  runnable bins: a mapping from bin key to a queue of continuations
M  mailboxes: one bounded ordered queue per process
W  wait sets: for each future, the continuations registered on it
S  scheduler configuration: binning mode, cohort width, partial policy
A  accounting counters
```

Entities are never addressed by pointer. Every reference is a **generational
reference**: a slot, a generation, a kind, and flags. A reference resolves only
when the slot exists, the kind matches the table, and the generation equals the
slot's current generation. Deleting an entity increments its slot generation
before the slot is reused, so a reference retained across a delete fails to
resolve rather than silently addressing a new entity.

*Note.* The generation field is finite (16 bits in the reference ABI), so
staleness detection is bounded, not guaranteed: a slot recycled 2^16 times wraps.
This is an ABA window, documented rather than solved.

### 1.1 Entities

**Process** — a persistent, independently evolving unit of computation with an
identity, private state, an inbox, a mode, and a lifecycle status. A process is
*not* a unit of execution; it does not "run". It becomes ready, and its
continuations run.

**Object** — a byte payload with an ownership state, a version, and a kind. The
physical representation is private to the implementation; programs cannot
construct or inspect a mapping.

**Continuation** — a *bounded resumable segment* of a process's execution: the
schedulable unit. It names its process, its run class, its durable frame, what
it depends on, and a remaining step budget. Continuations exist because the
model does not assume preemption: a computation that must yield control does so
by ending a continuation and naming the next one.

**Frame** — an object holding a continuation's live state as a byte blob. Frames
are durable and position-independent, which is what allows a continuation to be
resumed by a different executor than the one that suspended it.

**Future** — a single-assignment cell. Resolving it publishes a value and makes
every registered waiter ready.

**Run class** — the identity of a resume point. Two continuations share a run
class exactly when they will execute the same code over the same frame schema.
A run class is simultaneously the interpreter's dispatch key and the scheduler's
bin key; that they are the same value is a design commitment, not a coincidence.

**Capability** — permission to act on an entity. **[absent]** — see §5.

**Channel**, **Collective**, **Domain** — named by the model, not implemented.
**[absent]**

### 1.2 Observable behaviour

The observable behaviour of a run is its **trace**: a totally ordered sequence
of events, each carrying a logical time, epoch, kind, process, continuation, run
class, and one auxiliary field.

Two runs are **equivalent** when their traces are equal. This is the only
equivalence the model defines. Internal state that never reaches the trace is
not observable, and an implementation may represent it however it likes.

---

## 2. Invariants

A state is **legal** when all of the following hold. `check(Σ)` returns every
violation, not the first.

**I1. Reference integrity [checked].** Every reference held by a live entity
resolves, or is null. A continuation's process, frame, and dependency; a
process's state object; a resolved future's value; a queued message's payload.

**I2. No continuation left running [checked].** Between transitions no
continuation is in the `Running` state. `Running` exists only *within* a step.

**I3. Process/continuation consistency [checked].** A continuation's process is
live, and a terminated process has no continuation that is still schedulable.

> This clause is the reason `Complete` is a statement about a continuation and
> not about its process. A process may have several continuations alive at once;
> retiring the process when the first finishes strands the rest. The reference
> interpreter got this wrong until this specification was written — see §6.1.

**I4. Future single assignment [checked].** A pending future carries no value. A
settled future was settled no later than the current epoch, and has an empty
wait set — because resolution drains the wait set exactly once, a continuation
registered afterwards would never wake.

**I5. Mailbox bound [checked].** No mailbox holds more messages than its
capacity. Send into a full mailbox fails and registers the sender for wakeup; it
does not block, spin, or drop.

**I6. Message ordering [checked].** Messages from one sender to one receiver are
delivered in send order. No ordering is defined between different senders.

**I7. Scheduler well-formedness [checked].** Every continuation in a bin is
live, is in the `Runnable` state, and sits in the bin its run class maps to
under the current binning mode.

**I8. Frame exclusivity [checked].** No two continuations share a frame object.
A frame is the private mutable state of exactly one continuation.

**I9. Ownership monotonicity [checked, partial].** A frozen object has been
published to at least one reader, and any owner it names is live. Freezing is
one-way: mutation of a frozen object requires allocating a new object.

> Only the structural half is checked. That a frozen object is never *written*
> is not verified, because the model has no write barrier — see §5.

**I10. Capability safety [absent].** *Intended:* no process may act on an entity
without holding a capability conferring that right. **Not implemented.** The
capability table exists and is never consulted. Any process can mutate any
object it can name. There is deliberately no checker for this clause, because a
check that cannot fail would misrepresent the machine as safe.

**I11. Trace monotonicity [checked].** Logical time strictly increases across
the trace; epochs never move backwards.

**I12. Accounting consistency [checked].** Issued lane-slots partition exactly
into useful and idle; full cohorts do not exceed total cohorts.

**I13. Serial process execution [modelled].** At most one continuation holding
mutable authority over a process's state executes per epoch. Verified by test
rather than by predicate, because it is a property of a transition, not of a
state.

**I14. Progress [modelled].** If any continuation is runnable, an epoch executes
at least one step. Deferral policies may delay work but may not withhold it
indefinitely; the reference interpreter forces a partial cohort when an epoch
would otherwise do nothing.

---

## 3. Transitions

Each rule below is written as preconditions, effect, and the trace it emits.
All transitions are deterministic: given Σ and the same external input, the
successor state and emitted trace are uniquely determined. No rule consults a
clock, a random source, or an iteration order over an unordered container.

### 3.1 Entity creation

```text
CREATE-PROCESS(mode) -> p
  effect  fresh process, fresh state object, empty mailbox, status = Created
  trace   ProcessCreated

CREATE-CONTINUATION(p, run_class, frame_bytes, budget) -> c
  pre     p resolves
  effect  fresh frame object holding frame_bytes;
          c.status = Runnable; c enqueued in bin(run_class)
  trace   ContinuationReady

CREATE-FUTURE() -> f
  effect  fresh future, state = Pending, value = null
```

### 3.2 Messaging

```text
SEND(sender, receiver, payload, from)
  pre     receiver has a mailbox
  effect  if mailbox is full: register `from` as a waiter, fail with MailboxFull
          otherwise append the message with the next per-pair sequence number,
          and wake one registered receiver if any
  trace   MessageSent (+ MessageReceived if a waiter was woken)

RECEIVE(c) -> message?
  effect  if the mailbox is non-empty: pop the oldest message and wake one
          sender blocked on capacity
          otherwise register c as a receive-waiter and yield nothing
  trace   MessageReceived when a message is delivered
```

Send is non-blocking and lossless: a full mailbox is back-pressure, reported to
the sender, never a dropped message. Every blocked sender is registered, not
just the first (I5).

### 3.3 Dataflow readiness

```text
AWAIT(c, f, next_run_class) -> Registered | AlreadyResolved
  pre     f resolves
  effect  c.run_class := next_run_class; c.dependency := f
          if f is pending: c.status := Waiting; c joins f's wait set
          otherwise: c stays Runnable and the caller must yield, not await
  trace   ContinuationWaiting, emitted once, by the commit rule

RESOLVE(f, v)
  pre     f resolves and is Pending, else AlreadyResolved
  effect  f.state := Resolved; f.value := v;
          every waiter becomes Runnable and is enqueued; wait set drained
  trace   ContinuationReady per waiter, then FutureResolved
```

The `AlreadyResolved` outcome is not an optimisation. Resolution drains the wait
set once and never revisits it, so registering on a settled future is a
permanent stall (I4). The rule returns the distinction so callers cannot make
that mistake silently.

### 3.4 Execution and commit

A continuation step returns a **step result**: `Complete`, `Yield(next)`,
`Await(target, next)`, `Send`, `Spawn(next)`, or `Fault`. Handlers perform their
own side effects; the commit rule finalises scheduling and status.

```text
COMMIT(c, p, result)
  Complete      c.status := Completed
                p retired iff no continuation of p remains schedulable  (I3)
  Yield(n)      c.run_class := n; c.status := Runnable; c enqueued
  Await(t, n)   c.run_class := n; c.status := Waiting; NOT enqueued
  Send | Spawn  if next ≠ 0: continue as Yield; else as Complete
  Fault         c.status := Faulted; p.status := Failed; failure count += 1
```

A continuation becomes runnable or terminal *only* through this rule. That is
what makes I7 checkable: no other code path may enqueue.

### 3.5 The epoch

```text
EPOCH(Σ) -> Σ'
  1. Boundary   promote each bin's next buffer to current
  2. Admit      drop non-runnable entries; claim at most one mutating
                continuation per serial process (I13); defer the rest
  3. Group      partition admitted work by run class
  4. Execute    for each continuation: if budget exhausted, Fault without
                dispatching; else dispatch, then COMMIT, then charge the budget
  5. Account    update counters and advance the epoch
```

The budget check precedes dispatch deliberately. Faulting *after* commit would
leave a faulted continuation enqueued by that commit, violating I7.

Work produced during an epoch lands in the next-epoch buffer, so an epoch
boundary is a consistent cut. This is a property of the reference interpreter's
scheduling, not a requirement of the model; an implementation may wake
continuations within an epoch provided it preserves I1–I12 and determinism.

---

## 4. What the model does not define

Stated explicitly, because silence reads as permission:

- **Placement.** Nothing in §1–§3 says where a continuation runs. An
  implementation may run everything on one executor or distribute across many.
- **Parallelism.** The model is defined by a deterministic sequential semantics.
  A parallel implementation must be observationally equivalent to it (§1.2).
- **Cohorting.** Grouping continuations of one run class for joint execution is
  an implementation strategy the model *enables* by making run class explicit.
  It is not part of the semantics, and a conforming implementation need not do
  it.
- **Time.** There is no wall clock. `deadline_ns` exists in the ABI and is
  inert.
- **Fairness.** Beyond I14, no fairness guarantee is made. A run class may
  starve another.

---

## 5. Conformance of the reference interpreter

| Clause | Status | Note |
| ------ | ------ | ---- |
| I1 reference integrity | checked | |
| I2 no continuation running | checked | |
| I3 process/continuation | checked | defect found and fixed, §6.1 |
| I4 future single assignment | checked | defect found and fixed previously |
| I5 mailbox bound | checked | |
| I6 message ordering | checked | per sender/receiver pair only |
| I7 scheduler well-formed | checked | |
| I8 frame exclusivity | checked | |
| I9 ownership monotonicity | checked, partial | structural half only |
| I10 capability safety | **absent** | table allocated, never consulted |
| I11 trace monotonicity | checked | |
| I12 accounting consistency | checked | |
| I13 serial execution | modelled | per-epoch, by test |
| I14 progress | modelled | by test |

Entities named but not implemented: **Channel**, **Collective**, **Domain**,
execution contracts, cancellation, and supervision. Messaging is per-process
mailboxes rather than first-class channels; there is no collective construct at
all, so the model's claim to cover cooperative execution shapes is currently
unsupported by any implementation.

`tests/semantics.rs::capability_authority_is_unenforced_and_the_spec_says_so`
demonstrates the I10 gap concretely: a process with no capability mutates an
object it does not own, ownership transfer changes nothing, and the machine
still reports itself legal.

---

## 6. Ambiguities the executable model exposed

The point of an executable specification is that writing it breaks things.

### 6.1 `Complete` conflated continuation and process lifetime

The first run of the I3 checker failed on the `Expand` workload. `Expand` spawns
its heuristic as a second continuation *of the same process*; when the heuristic
completed, the commit rule marked the whole process terminated, leaving
`Expand`'s main continuation waiting on a future belonging to a dead process.
It then woke and ran normally, because nothing checked.

The ambiguity was in the model, not only the code: §3.4 had never said whether
`Complete` is a claim about a continuation or about a process. It is about a
continuation. A process retires when its last continuation does.

### 6.2 Unresolved: what does a process *own*?

I8 gives each continuation an exclusive frame, and a process has a state object,
but nothing says how concurrent continuations of one process may touch that
shared state. I13 restricts *mutating* continuations to one per epoch without
defining which continuations mutate. The reference interpreter treats mode as
the answer, which is too coarse.

### 6.3 Unresolved: failure containment

`Fault` marks a process failed and increments a counter. Nothing says what
happens to its other continuations, its queued messages, its unresolved futures,
or anyone awaiting them. A future whose resolver faults is never resolved and
its waiters wait forever — I14 does not catch this, because those continuations
are not runnable.

### 6.4 Unresolved: cancellation

`CancelPending` and `Cancelled` exist as states with no transition reaching
them.

---

## 7. Next

In order of what most constrains the rest:

1. **Capabilities (I10).** The largest gap between what the model claims and
   what it does. Requires deciding whether authority is checked at reference
   resolution or at operation.
2. **Failure and cancellation (§6.3, §6.4).** Both need the answer to §6.2
   first.
3. **Channels and collectives.** Currently vocabulary, not machinery.
4. **Validation workloads** beyond dynamic search: streaming pipelines, actor
   systems, irregular dataflow — chosen to stress ordering, back-pressure, and
   failure rather than throughput.
5. **A minimal IR**, once §6.2 and §6.3 are settled — a surface syntax for a
   model with unresolved ownership and failure semantics would encode the
   ambiguity rather than remove it.

Performance work belongs after all of this, and the performance results already
in this repository (`docs/SOMA-P1.md`, and the cohorting studies) should be read
as measurements of *one implementation strategy*, not as properties of the
model.
