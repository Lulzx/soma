# SOMA Semantic Specification v0.2

**Status:** draft. Machine-checked where marked. Incomplete where marked.

SOMA is an abstract machine for irregular concurrent computation. It is defined
here without reference to any hardware: no SIMD width, no device, no host, no
placement. Those belong to *implementations* of SOMA, of which the CPU reference
interpreter in this repository is one and a GPU operating system might be
another. If a clause below mentions hardware, it is a defect in the clause.

This version defines persistent processes, object capabilities, continuations,
dataflow readiness, and collectives as one programming model. Most clauses map
to executable checks against the reference interpreter.

## 0. How to read this

Every clause is one of:

| Marker | Meaning |
| ------ | ------- |
| **[checked]** | Stated here, implemented, and machine-verified by `soma::semantics::invariants` against the reference interpreter after every transition. |
| **[modelled]** | Implemented in the reference interpreter, but not yet expressible as a state predicate, so it is verified only by targeted tests. |
| **[absent]** | Named by the model and *not* implemented. The machine does not have this property. |

An [absent] clause provides no guarantee to a program that relies on it.
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
   futures, channels, collectives, capabilities, domains, contracts, modules
Q  runnable bins: a mapping from bin key to a queue of continuations
M  mailboxes and channel queues: bounded ordered message stores
W  wait sets: future, mailbox, channel, multi-input, and supervision readiness
S  scheduler configuration: binning mode, cohort width, partial policy
A  accounting counters
```

Entities are addressed by **generational references**, not pointers. Each has
a slot, generation, kind, and flags. A
reference resolves only
when the slot exists, the kind matches the table, and the generation equals the
slot's current generation. Deleting an entity increments its slot generation
before the slot is reused, so a reference retained across a delete fails to
resolve rather than silently addressing a new entity.

*Note.* The generation field is finite (16 bits in the reference ABI). This used
to make staleness detection bounded rather than guaranteed — a slot recycled
2^16 times wrapped, an ABA window documented rather than solved. **Solved in
v0.3 §1.2:** a slot whose generation is exhausted is retired instead of
recycled, so detection is guaranteed at every generation width, at a cost of one
withdrawn slot per 65,535 recycles.

### 1.1 Entities

**Process**, a persistent, independently evolving unit of computation with an
identity, private state, an inbox, a mode, and a lifecycle status. A process is
*not* a unit of execution. It does not "run". It becomes ready, and its
continuations run.

**Object**, a byte payload with a version and a kind. Its unique-mutable or
frozen-shared state is derived from live capability holders rather than stored
in its descriptor. The physical representation is private to the
implementation. The program API does not construct or expose a mapping.

**Continuation**, a *bounded resumable segment* of a process's execution: the
schedulable unit. It names its process, its run class, its durable frame, what
it depends on, a `ReadOnly` or `Mutable` declaration for canonical process
state, and a remaining step budget. Continuations exist because the model does
not assume preemption: a computation that must yield control does so by ending
a continuation and naming the next one.

**Frame**, an object holding a continuation's live state as a byte blob. Frames
are durable and position-independent, which is what allows a continuation to be
resumed by a different executor than the one that suspended it.

**Future**, a single-assignment cell. Resolving it publishes a value and makes
every registered waiter ready.

**Run class**, the identity of a resume point. Two continuations share a run
class exactly when they will execute the same code over the same frame schema.
A run class is simultaneously the interpreter's dispatch key and the scheduler's
bin key. That they are the same value is a design commitment, not a coincidence.

**Capability**, actor-relative permission to act on an entity. Capability
spaces, genesis, attenuation, and enforcement for every reachable operation
right are implemented. Authority decisions and governed effects are observable,
making I10c trace-checkable.

**Channel**, a first-class bounded FIFO message queue. Send, receive, and close
are separate capability-governed operations. A committed message holds payload
read authority in kernel escrow until delivery. **[checked]**

**Collective**, a coordinated operation with an owner and completion future.
The implemented `BatchEvaluate` form consumes a frozen input array and
publishes a frozen output array. It carries the stable evaluator ID selected by
the minimal IR. **[checked]**

**Domain**, a logical authority and allocation boundary. Every process and
object belongs to one generationally referenced domain. A domain may nest under
another domain and may bound total process creation. Domain creation and
cross-domain process creation require actor-relative authority. **[checked]**

**Execution contract**, an actor-relative constraint attached to a
continuation. The current hardware-neutral contract bounds steps and frame
bytes and requires scalar, deterministic, placement-neutral execution with no
wall-clock deadline. Hardware placement, lane shape, and relaxed determinism
are rejected rather than promised without machinery. **[checked]**

**Module**, an immutable loaded manifest that gives a set of batch evaluators
stable identities and element strides. A module is created from the validated
minimal textual IR, is an actor-relative capability target, and is linked from
each collective instantiated through it. **[checked]**

**Supervision** is a direct parent/child relation. When a supervised child
terminates, fails, or is cancelled, the kernel appends exactly one typed exit
notice to the supervisor's reliable control queue and wakes one continuation
waiting on that queue. Notices are separate from bounded user mailboxes, so
ordinary traffic cannot cause a child exit to be dropped. Each relationship
selects notification-only containment, failure escalation, or bounded restart.
Restart preserves the failed identity for observation, creates a fresh process
identity with the registered entry-continuation template, records lineage in
the notice, and escalates when its retry budget is exhausted. **[checked]**

### 1.2 Observable behaviour

The observable behaviour of a run is its **trace**: a totally ordered sequence
of events, each carrying a logical time, epoch, kind, process, continuation, run
class, and one auxiliary field.

Two runs are **equivalent** when their traces are equal. The model defines no
other equivalence. Internal state absent from the trace is
not observable, and an implementation may represent it however it likes.

> **Superseded by v0.3 §2.** Trace equality is defined over a total order on
> logical time, and this interpreter's total order is an artifact of running one
> continuation at a time. Under this clause as written, every parallel
> implementation of SOMA is non-conforming by construction. `docs/SOMA-v0.3.md`
> replaces it with I18: a trace conforms when it contains the same events per
> epoch and is a linear extension of the semantic order ≺. Equality is the
> special case where ≺ happens to be total. Placement reporting
> (`CohortCreated`, `ContinuationPlaced`) is excluded from observable behaviour,
> without which §4's "cohorting is not part of the semantics" would be false.

---

## 2. Invariants

A state is **legal** when all of the following hold. `check(Σ)` returns every
violation, not the first.

**I1. Reference integrity [checked].** Every reference held by a live entity
resolves, or is null. A continuation's process, frame, dependency, and contract.
A process and object's domain. A domain's parent. A resolved future's value. A
queued message's payload. A channel's escrow target and waiters. A collective's
owner, arrays, completion future, and optional module. A supervision notice's
child and optional replacement.

**I2. No continuation left running [checked].** Between transitions no
continuation is in the `Running` state. `Running` exists only *within* a step.

**I3. Process/continuation consistency [checked].** A continuation's process is
live, and a terminated process has no continuation that is still schedulable.

> This clause is the reason `Complete` is a statement about a continuation and
> not about its process. A process may have several continuations alive at once.
> Retiring the process when the first finishes strands the rest. The reference
> interpreter got this wrong until this specification was written, see §6.1.

**I4. Future single assignment [checked].** A pending future carries no value. A
settled future was settled no later than the current epoch and has an empty
wait set. Resolution drains the wait set exactly once. A continuation
registered after resolution would never wake. A collective's state and output
must agree with the state and value of its completion future.

**I5. Mailbox bound [checked].** No process mailbox or first-class channel holds
more messages than its capacity. Send into a full queue fails and registers the
sender for wakeup. It does not block, spin, or drop.

**I6. Message ordering [checked].** Process-mailbox messages from one sender to
one receiver are delivered in send order. A channel is FIFO across all senders.

**I7. Scheduler well-formedness [checked].** Every continuation in a bin is
live, is in the `Runnable` state, and is stored in the bin its run class maps to
under the current binning mode.

**I8. Frame exclusivity [checked].** No two continuations share a frame object.
A frame is the private mutable state of exactly one continuation.

**I9. Ownership monotonicity [subsumed by I10b].** Ownership is not an
independent flag. Exactly one process may hold live full-object `WRITE`
authority; no writer plus at least one reader is frozen-shared. `WRITE` cannot
be copied across processes, transfer moves the sole writer, and freeze revokes
write-bearing capability trees. Returning to mutable state therefore requires
allocating a new object.

**I10a. Capability attenuation [checked].** A derived capability's rights are a
subset of its parent's rights and its byte range lies within its parent's.

**I10b. Capability integrity [checked].** Every capability target resolves, its
rights apply to the target kind, its parent is live in the same actor's
capability space, and no object has more than one mutable authority holder.

**I10c. No unauthorised effect [checked].** No process may apply a governed
effect without holding a capability conferring that right. Every reachable
operation records its granted or denied decision. A governed mutation emits an
adjacent `AuthorityEffect` with the same actor, right, and target; the checker
rejects an effect without that matching grant.

**I11. Trace monotonicity [checked].** Logical time strictly increases across
the trace. Epochs never move backwards.

**I12. Accounting consistency [checked].** Issued lane-slots partition exactly
into useful and idle. Full cohorts do not exceed total cohorts.

**I13. Process-state serialisation [checked, trace-level].** At most one
continuation declaring `Mutable` access to a process's canonical state starts in
an epoch. Read-only continuations may share the epoch. State mutation also
requires the declared continuation to be the process's active continuation.

**I14. Progress [superseded by I21].** If any continuation is runnable, an epoch
executes at least one step. Deferral policies may delay work but may not
withhold it indefinitely. The reference interpreter forces a partial cohort when
an epoch would otherwise do nothing.

> This was the specification's only `[modelled]` clause. v0.3's I21 checks it —
> the withholding half as a counter, since it describes a transition rather than
> a state — and adds a starvation bound, which §4 below explicitly declined to
> give.

**I15. Supervision integrity [checked].** A process cannot supervise itself.
Every non-null supervisor and every child named by a queued notice resolves.
Each notice belongs to the child's declared direct supervisor, agrees with the
child's terminal state and failure count, and every registered waiter belongs
to the supervisor whose queue it waits on. A failed child whose relationship
requires escalation has a failed direct supervisor.

**I16. Domain and contract integrity [checked].** The root domain resolves and
has no parent. Domain parent links are acyclic at creation, stored process counts
equal actual membership, and bounded domains do not exceed their creation
quota. Every object and process names a live domain. Every execution contract
is valid for the hardware-neutral machine; an attached continuation stays
within the contract's step and frame-byte bounds.

**I17. Module integrity [checked].** Every loaded module's identity and
evaluator count agree with its immutable manifest. Evaluator IDs are nonzero
and unique, and strides are nonzero. Every module-linked collective names an
evaluator present in that module with the same element stride.

---

## 3. Transitions

Each rule below is written as preconditions, effect, and the trace it emits.
All transitions are deterministic: given Σ and the same external input, the
successor state and emitted trace are uniquely determined. No rule consults a
clock, a random source, or an iteration order over an unordered container.

### 3.1 Entity creation

```text
CREATE-DOMAIN(parent, max_processes) -> d
  pre     caller has WRITE on parent
  effect  fresh logical domain, process count = 0
  trace   DomainCreated

CREATE-PROCESS(domain, mode) -> p
  pre     caller has WRITE on domain; domain creation quota is not exhausted
  effect  fresh process and state object in domain, empty mailbox,
          status = Created; increment domain process count
  trace   ProcessCreated

CREATE-CONTINUATION(p, state_access, run_class, frame_bytes, budget) -> c
  pre     p resolves; a Pure process rejects Mutable state_access
  effect  fresh frame object holding frame_bytes;
          c.state_access := state_access;
          c.status = Runnable; c enqueued in bin(run_class)
  trace   ContinuationReady

CREATE-FUTURE() -> f
  effect  fresh future, state = Pending, value = null

CREATE-SUPERVISED-PROCESS(supervisor, mode, policy) -> p
  pre     supervisor resolves and is non-terminal
  effect  CREATE-PROCESS(mode); p.supervisor := supervisor;
          p.supervision_policy := policy

CREATE-CONTRACT(step_limit, frame_byte_limit, deterministic) -> k
  pre     scalar shape; placement neutral; deterministic; no wall-clock deadline
  effect  fresh actor-relative execution contract
  trace   ContractCreated

CREATE-CONTRACTED-CONTINUATION(p, k, frame, budget) -> c
  pre     caller has READ on k; budget <= k.step_limit;
          bytes(frame) <= k.frame_byte_limit when that limit is nonzero
  effect  CREATE-CONTINUATION(...); c.execution_contract := k
  trace   ContractAttached

LOAD-MODULE(name, evaluators[]) -> m
  pre     caller is live; name is non-empty; evaluator IDs are nonzero and
          unique; evaluator strides are nonzero
  effect  fresh immutable module manifest; caller receives genesis authority
  trace   ModuleLoaded
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

First-class channels use the analogous rules:

```text
CREATE-CHANNEL(owner, capacity) -> ch
  effect  fresh open channel with an empty bounded FIFO

CHANNEL-SEND(sender, ch, payload, from)
  pre     sender has SEND on ch and READ|TRANSFER on payload; ch is open
  effect  if full: register from and fail with MailboxFull
          otherwise escrow a payload READ root and append in FIFO order
  trace   ChannelSent

CHANNEL-RECEIVE(receiver, ch, c) -> message?
  pre     receiver has RECEIVE on ch
  effect  pop the oldest entry, install its escrowed READ root in receiver's
          capability space, and wake one capacity waiter; if open and empty,
          register c; if closed and empty, fail with ChannelClosed
  trace   ChannelReceived on delivery

CLOSE-CHANNEL(actor, ch)
  pre     actor has DESTROY on ch
  effect  reject future sends, wake all waiters, preserve queued entries for drain
  trace   ChannelClosed
```

Supervision uses a reliable kernel control queue rather than the user mailbox:

```text
NOTIFY-SUPERVISOR(child, reason)
  pre     child has a non-null supervisor and just became terminal
  effect  append (child, reason, failure_count) exactly once;
          wake the oldest registered supervisor continuation, if any;
          if reason = Failed and policy = Escalate, fail the supervisor;
          if policy = Restart and retries remain, create a fresh replacement
          from the entry template and include it in the notice;
          if policy = Restart and retries are exhausted, fail the supervisor
  trace   SupervisionNotified

RECEIVE-SUPERVISION(supervisor, c) -> notice?
  pre     supervisor has RECEIVE on itself; c belongs to supervisor
  effect  pop the oldest notice, or register c if the queue is empty
```

For an irregular all-input join, receive is atomic across its input set:

```text
RECEIVE-ALL(actor, channels[], c) -> messages[]?
  pre     channels is non-empty and duplicate-free; actor has RECEIVE on each
  effect  if every channel is non-empty: pop exactly one from each, in input
          order; otherwise consume nothing and register c on missing inputs
          retry removes stale registrations before re-evaluating the whole set
```

### 3.3 Dataflow readiness

```text
AWAIT(c, f, next_run_class) -> Registered | AlreadySettled(state)
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

The `AlreadySettled` outcome is not an optimisation. Resolution, failure, and
cancellation drain the wait set once and never revisit it, so registering on a
settled future is a permanent stall (I4). The returned state lets the caller
distinguish a value from failure or cancellation.

`BatchEvaluate` is the first collective:

```text
CREATE-BATCH-EVALUATE(owner, evaluator, inputs, count, stride) -> (op, done)
  pre     inputs is a frozen array of at least count * stride bytes
  effect  fresh Pending collective and Pending completion future
  trace   CollectiveCreated

CREATE-BATCH-EVALUATE-IN-MODULE(owner, module, evaluator, inputs, count)
  pre     owner has READ on module; evaluator occurs in module's manifest
  effect  CREATE-BATCH-EVALUATE using the manifest stride;
          op.module := module

COMPLETE-BATCH-EVALUATE(owner, op, outputs)
  pre     owner has WRITE on op and RESOLVE on done;
          op and done are Pending; outputs is a sufficiently large frozen array
  effect  op.outputs := outputs; op.state := Completed;
          resolve done to outputs
  trace   CollectiveCompleted
```

The collective specifies lifecycle and publication, not evaluator code or
parallel execution strategy. The minimal IR gives `evaluator` a stable integer
identity, frozen-array element stride, and entry/completion resume points. Owner
failure or cancellation settles the pending collective and its completion
future together.

### 3.4 Execution and commit

A continuation step returns a **step result**: `Complete`, `Yield(next)`,
`Await(target, next)`, `Send`, `Spawn(next)`, or `Fault`. Handlers perform their
own side effects. The commit rule finalises scheduling and status.

```text
COMMIT(c, p, result)
  Complete      c.status := Completed
                p retired iff no continuation of p remains schedulable  (I3)
  Yield(n)      c.run_class := n; c.status := Runnable; c enqueued
  Await(t, n)   c.run_class := n; c.status := Waiting; NOT enqueued
  Send | Spawn  if next ≠ 0: continue as Yield; else as Complete
  Fault         c.status := Faulted; p.status := Failed; failure count += 1;
                cancel sibling continuations; fail p-owned pending futures;
                drain p's mailbox and wake external waiters/senders;
                reclaim p's local capability space

CANCEL(actor, p)
  pre     actor has WRITE on p; p is not terminal
  effect  if p has an active continuation: p.status := CancelPending and
          finalize at that continuation's commit boundary;
          otherwise finalize immediately;
          finalization cancels live continuations, cancels p-owned pending
          futures, drains the mailbox, wakes external waiters/senders,
          reclaims local capabilities, and sets p.status := Cancelled
  trace   ProcessCancelled
```

A continuation becomes runnable or terminal *only* through this rule. That is
what makes I7 checkable: no other code path may enqueue.

### 3.5 The epoch

```text
EPOCH(Σ) -> Σ'
  1. Boundary   promote each bin's next buffer to current
  2. Admit      drop non-runnable entries; claim at most one Mutable
                state-access continuation per process (I13); defer the rest
  3. Group      partition admitted work by run class
  4. Execute    for each continuation: if budget exhausted, Fault without
                dispatching; else dispatch, then COMMIT, then charge the budget
  5. Account    update counters and advance the epoch
```

The budget check precedes dispatch deliberately. Faulting *after* commit would
leave a faulted continuation enqueued by that commit, violating I7.

Work produced during an epoch goes into the next-epoch buffer, so an epoch
boundary is a consistent cut. This is a property of the reference interpreter's
scheduling, not a requirement of the model. An implementation may wake
continuations within an epoch provided it preserves I1–I13, I15–I17, and determinism.

---

## 4. Undefined by the model

- **Placement.** Nothing in §1–§3 says where a continuation runs. An
  implementation may run everything on one executor or distribute across many.
- **Parallelism.** The model is defined by a deterministic sequential semantics.
  A parallel implementation must be observationally equivalent to it (§1.2).
- **Cohorting.** Grouping continuations of one run class for joint execution is
  an implementation strategy the model *enables* by making run class explicit.
  It is not part of the semantics, and a conforming implementation need not do
  it.
- **Time.** There is no wall clock. Legacy `deadline_ns` fields remain inert,
  and a current execution contract with a nonzero deadline is invalid.
- **Fairness.** Beyond I14, no fairness guarantee is made. A run class may
  starve another. *(v0.3's I21 adds a bound: no runnable continuation waits
  longer than a declared number of consecutive epochs. Starvation is defensible
  when it is visible in a single sequential trace and much less so under
  territory placement, where it becomes a policy outcome nobody chose.)*

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
| I9 ownership monotonicity | subsumed by I10b | ownership derived from capabilities |
| I10a attenuation | checked | rights and ranges only shrink |
| I10b capability integrity | checked | actor-relative spaces and live links |
| I10c no unauthorised effect | checked | adjacent decision/effect trace proof |
| I11 trace monotonicity | checked | |
| I12 accounting consistency | checked | |
| I13 process-state serialisation | checked | trace-level, with fault injection |
| I14 progress | superseded | checked as I21 in v0.3, with a starvation bound |
| I15 supervision integrity | checked | notices, escalation, bounded restart, replacement lineage |
| I16 domain/contract integrity | checked | membership, quotas, contract bounds |
| I17 module integrity | checked | manifests and collective evaluator links |

No entity named by this v0.2 specification remains **[absent]**. Domains,
execution contracts, supervision, channels, modules, and the `BatchEvaluate`
collective are covered by structural invariants plus targeted lifecycle and
negative tests.

`tests/semantics.rs::i10c_records_grants_denials_and_authorized_effects` exercises
grants, denials, and an authorized write. The adjacent fault-injection test
proves that I10c catches an effect without a matching grant. Separate negative
tests prove that the I10a/I10b checkers catch amplification and a dead parent.

---

## 6. Ambiguities the executable model exposed

The executable checks exposed the following ambiguities.

### 6.1 `Complete` conflated continuation and process lifetime

The first run of the I3 checker failed on the `Expand` workload. `Expand` spawns
its heuristic as a second continuation *of the same process*. When the heuristic
completed, the commit rule marked the whole process terminated, leaving
`Expand`'s main continuation waiting on a future belonging to a dead process.
It then woke and ran normally, because nothing checked.

The ambiguity was in the model, not only the code: §3.4 did not specify whether
`Complete` is a claim about a continuation or about a process. It is about a
continuation. A process retires when its last continuation does.

### 6.2 Process-state ownership

A process owns one canonical state object through its capability space. Each
continuation explicitly declares `ReadOnly` or `Mutable` access to that state.
The scheduler serialises mutable declarations per process and I13 checks the
trace. During execution, `process_state_bytes_mut` requires the named mutable
continuation to be active; generic object mutation cannot open a process-state
object. Private continuation frames remain governed separately by I8. A `Pure`
process cannot create a mutable-state continuation, so purity is a construction
rule rather than a scheduler guess based on process mode.

### 6.3 Failure containment

`Fault` is a containment boundary. The triggering continuation becomes
`Faulted`; every other live continuation of the process becomes `Cancelled` and
is removed from scheduler and wait structures. Pending futures owned by the
process become `Failed`; external waiters become runnable so they can observe
the terminal future state. The mailbox is drained and external senders blocked
on its capacity are woken; subsequent sends fail with `ProcessUnavailable`.
Local capabilities are reclaimed, while previously exported roots survive.

### 6.4 Cooperative cancellation

`CANCEL(actor, p)` requires `WRITE` on the target process. With no active
continuation it finalizes immediately. During execution it records
`CancelPending`; the active continuation reaches its commit boundary, then the
same containment cleanup runs with futures becoming `Cancelled` and the process
ending in `Cancelled`. Cancellation does not preempt a continuation mid-step.

---

## 7. Extensions beyond v0.2 conformance

The semantic core is complete for every entity and invariant named above. The
bounded stream covers FIFO, back-pressure, and producer failure; the controlled
actor tree covers notification, escalation, restart, and sibling containment;
the irregular two-input join covers atomic readiness, skew, back-pressure, and
committed-prefix survival. The minimal textual evaluator IR loads immutable
manifests and is connected to module-linked `BatchEvaluate`. A backend-neutral
execution boundary realizes the reference evaluator on CPU or optional Apple
Metal, with CPU spill and collective-boundary placement accounting.

Future work extends rather than completes this specification:

1. A general-purpose language/compiler for arbitrary evaluator bodies.
2. Additional contract dimensions only when an implementation can enforce them.
3. A persistent device scheduler or distributed implementation proven
   trace-equivalent to this machine.

These are scoped in `docs/SOMA-v0.3.md`. Item 3 cannot be attempted under this
specification as written: §1.2 defines equivalence as trace equality over a
total order, which no concurrent implementation can satisfy. Weakening that
relation is v0.3's only specification deliverable.

Performance work belongs after all of this, and the performance results already
in this repository (`docs/SOMA-P1.md`, and the cohorting studies) should be read
as measurements of *one implementation strategy*, not as properties of the
model.
