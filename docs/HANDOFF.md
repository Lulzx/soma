# SOMA engineering handoff

## Discovery execution model

`src/discovery` is now the first application-level execution model above SOMA.
It implements deterministic trace replay, SHA-256 semantic node identities,
pending/ready deduplication, transitive hypothesis interest and cancellation,
pointwise-only batch fusion, D1-D7 checks, synthetic discovery search, regime
mapping, and CPU/Metal reports. It deliberately changes no SOMA ABI entity.
Start with [DISCOVERY.md](DISCOVERY.md), `tests/discovery_*`, and
`examples/discovery_report.rs`.

The default implementation trace compresses 2,184 logical requests to 910
physical evaluator realizations and 12 dispatches with identical terminal
scientific state. Local release smoke runs showed a modest CPU improvement and
a larger Metal improvement, but do not quote a single run or structural
compression as a performance result.

The narrowed §4.17 surface now feeds an opt-in speculative concurrent CPU
executive. It executes isolated lane snapshots on real threads, records all 15
`LaneView` operations and their resource accesses, validates conflicts, and
either replays canonically or discards everything and invokes the reference
loop. Start with [SPECULATIVE-EPOCHS.md](SPECULATIVE-EPOCHS.md),
`kernel/speculation.rs`, and `tests/speculative_epochs.rs`.

Read §1 for the project state and §6 for the test discipline before changing
the code.

Repository: https://github.com/Lulzx/soma. The default semantic core is
dependency-free. There are 410 default tests, seven additional tests behind the
`metal` feature, seven compile-fail doc tests, and no Clippy warnings. The optional `metal`
feature adds the `metal-rs` implementation dependency on macOS.

```sh
cargo test
cargo clippy --all-targets
cargo test --all-features            # real Metal dispatch on macOS
cargo clippy --all-targets --all-features
cargo run --example cohort_report      # cohorting vs a persistent FIFO
cargo run --example baseline_report    # vs a hand-written bulk frontier
cargo run --example irregular_report   # occupancy/latency frontiers
cargo run --example regime_map         # where cohorting helps and fails
cargo run --example territory_report   # distribution across territories
cargo run --example streaming_report   # channel back-pressure + failure
cargo run --example supervision_report # notification vs failure escalation
cargo run --example multi_input_report # atomic join + skew/failure controls
cargo run --release --example speculative_epoch_report # threaded epoch crossover
```

The measurement examples are separate, want `--release`, and are documented in
`docs/PERFORMANCE.md`:

```sh
cargo run --release --features metal --example backend_bench   # CPU vs Metal across batch sizes
cargo run --release --features metal --example metal_overhead  # where a Metal call's fixed cost goes
cargo run --release --example kernel_overhead                  # what a published cohort costs off-GPU
cargo run --release --example growth_sweep                     # cost against accumulated state, to 1M
cargo run --release --example memory_profile                   # bytes per unit of work
```

---

## 1. What the project currently is

SOMA is an **abstract machine** for irregular concurrent computation: persistent
processes, objects, continuations, dataflow readiness, capabilities,
collectives. It specifies no SIMD width, device, host, or placement. Those
belong to implementations.

The project changed shape recently and the git history reads misleadingly if you
don't know this. It began as a GPU operating system idea ("replace kernel
launches with persistent device-resident processes"), generalised into SOMA, and
is now focused on the abstract machine. A GPU OS may implement it later.
Performance work was paused and has since restarted, on the implementation
only. `docs/PERFORMANCE.md` covers it: a wall-clock harness (the repository had
none), the Metal backend, three accidentally-quadratic paths, and reclamation.
No invariant changed. Read its §1 before quoting any figure and §7 for what is
still open, including that the cohorting thesis in §4 below is *still* a
structural model rather than a hardware measurement.

Two documents, and they are not equals:

| Doc | Status |
| --- | --- |
| `docs/SOMA-v0.2.md` | **Current.** The semantic specification. Start here. |
| `docs/SOMA-v0.3.md` | **Current for anything added since v0.2.** §1–§3 and §4's semantic obligations are implemented and checked. §4.18 records the speculative concurrent CPU executive. Persistent device residency, distribution, and the remaining hardware work stay scoped by §4–§6. |
| `docs/SOMA-P1.md` | Historical. The original broad Phase-1 contract, still referenced by `§n` markers in code comments. Useful context, but it describes a wider system than the one being built, and its framing is what the refocus moved away from. |

The directory is still named `gpu-os` and the crate `soma`. Harmless, but expect
the mismatch.

---

## 2. Architecture in one pass

```text
src/
  abi/         Fixed-width ABI structs. Ref64 = slot+generation+kind+flags.
               Descriptors for objects, processes, continuations, cohorts,
               futures, messages, channels, collectives, capabilities,
               domains, contracts, traces.
  table.rs     Generational slot table, partitioned. Slot 0 is NULL in every
               partition. Delete bumps generation. Stale references fail.
  kernel/      The machine. mod.rs holds all state. epochs.rs runs epochs.
               commit.rs publishes effects. effects.rs is the effect log: a
               step produces its bin entries and the applier writes them.
               ownership.rs derives object state from live capabilities.
               accounting.rs records counters.
  executives/  cpu_scalar.rs is the continuation interpreter. batch.rs is the
               physical backend/publication boundary; metal.rs is optional.
  scheduler/   runnable_bins.rs contains double-buffered run-class bins.
               cohorts.rs builds cohorts and computes dispatch cost.
               admission.rs decides which continuations run this epoch, as a
               pure function of the candidate set.
  compiler/    frame.rs encodes frames. state_machine_lowering.rs contains the
               hand-lowered Expand example. body.rs is the evaluator body
               language and its MSL codegen; examples.rs holds the example
               module both backends are checked against.
  semantics/   invariants.rs checks the executable part of the specification.
               order.rs derives the semantic order, the identity
               correspondence, and checks I18/I19.
               schedule.rs checks I22, admission determinism.
  replay/      trace comparison for determinism checks.
  experiments/ measurement only. None of it is part of the machine.
```

A *run class* identifies a resume point. It is the interpreter's dispatch key
and the scheduler's bin key. A continuation that yields names its next bin, so
the scheduler does not inspect continuation metadata to decide what can run
together.

SOMA does not assume preemption. A computation that must yield ends a
continuation and names the next one. Frames are byte blobs in shared memory, not
register state, so another executor can resume the continuation.

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
- The semantic specification, with 11 original invariants plus capability
  attenuation, integrity, effect authorization, and supervision integrity
  machine-checked.
- Capability-derived object ownership: one mutable holder, linear `WRITE`
  transfer, and freeze by revoking write-bearing capability trees.
- Explicit `ReadOnly`/`Mutable` continuation declarations for canonical process
  state, with active-continuation enforcement and trace-checked I13 admission.
- Contained process failure and cancellation: sibling continuations terminate,
  owned futures settle, mailboxes drain, and external waiters/senders wake.
- First-class bounded channels with capability-gated send/receive/close,
  back-pressure, FIFO delivery, and kernel escrow of payload `READ` authority.
- A `BatchEvaluate` collective lifecycle over frozen arrays, publishing one
  frozen output array through a completion future.
- A generic bounded streaming-graph validation workload and a minimal
  hardware-neutral evaluator IR with frozen-array schema and resume points.
- Direct parent/child supervision with reliable typed terminal notices and
  deterministic waiter wakeup, opt-in failure escalation, and bounded restart
  through fresh replacement identities.
- Atomic all-input channel receive plus an irregular two-input join validation.
- Logical domains with authority and creation quotas; hardware-neutral
  execution contracts that bound continuation steps and frame bytes.
- A textual evaluator-module surface, immutable loaded manifests, and I17
  checking every module-linked collective.
- A physical batch backend boundary with a scalar reference evaluator, CPU
  spill, placement-change accounting, and an optional real Metal compute
  implementation.

Added in v0.3:

- A semantic order ≺ derived from the transition rules, and conformance
  (I18) and placement-neutrality (I19) checks against it. Trace equality no
  longer excludes every parallel implementation by construction.
- An evaluator body language with one source lowered to both the CPU
  interpreter and generated Metal Shading Language, and backend agreement
  (I20) checked against real GPU hardware. The hardcoded `2*x + 1` in both
  backends is gone.
- Bounded progress (I21), which absorbs v0.2's only `[modelled]` clause and
  adds a starvation bound.
- Per-process live-continuation counts in place of a table scan per commit,
  and generation-exhausted slots retired rather than wrapped.
- Admission as a pure function of the epoch's candidate set (I22), replacing
  the first-come `HashSet` claim.
- Lane-relative trace positions (I23), so the run's order is recoverable
  without a shared clock. Two of the device scheduler's four obligations.
- I18 up to a renaming of entity references, so an implementation whose
  allocator names entities differently is no longer non-conforming by
  construction. `Ref64` gains a `partition` byte in place of the unused
  `flags`, and `TraceEvent` a `subject` field so no entity is recorded as a
  bare slot number.
- Partitioned allocation: `GenTable` allocates from a partition chosen by the
  lane's position in the epoch's plan, so lanes need no shared allocator. I19
  varies it at 1/2/4/8 partitions.
- Gathering bodies. `Op::Index` gives an element its own position and
  `Op::Gather` reads a computed element of the frozen input array, so a body is
  no longer confined to its own fields. Two properties keep this from costing
  anything the machine relies on: a gather reads the frozen *input* and never
  the output, so lane order still cannot change the result (I19), and an
  out-of-range index clamps to the last element rather than faulting, so bodies
  stay total under a computed index. Both lowerings clamp identically and I20
  checks that on hardware. `examples::NEIGHBOUR_MAX` (a stencil) and
  `examples::PERMUTE` (a reversal, where every element overwrites the field its
  neighbours read) are the checked cases.
- Loops. `repeat` takes a trip count fixed at validation time, `breakif`
  leaves the innermost one early, and `get`/`set` over declared locals are how
  a value outlives an iteration -- values computed inside a loop do not escape
  it, which is the rule that replaces phi nodes. Totality survives because the
  trip count is static: `step_bound` multiplies out the nesting and `MAX_STEPS`
  is the ceiling, so a body's worst case is still known before it runs, which
  is what the continuation step budget needs. Branch-freedom stopped being a
  property of the language and became `is_uniform`, a property of a body: a
  counted loop is uniform, a `breakif` is not, and divergence costs occupancy
  rather than correctness. `examples::WINDOW_SUM` and `examples::RUN_LENGTH`
  are the checked cases, both agreeing with the CPU interpreter on real Metal.
- A second array binding. `gatheraux` reads a computed element of a second
  frozen, read-only array, declared with its own `aux` layout because an aux
  element has no reason to share the input element's shape. The array is bound
  to the collective (`create_batch_evaluate_bound`) rather than passed at the
  call, because that is what the capability escrow freezes; it is authorized
  and validated on exactly the same terms as the first, since a body gathering
  from an array it holds no READ on is a capability hole and one gathering from
  an unfrozen array breaks I19. The binding is checked in *both* directions at
  the backend boundary -- a body reading an array it was not given, and an
  array bound to a body with no name for it, are both `InvalidInput`. This is
  what put ant sensing on the GPU; see `experiments/ant_scoring.rs`.
- Reordered lanes (v0.3 §4.6). `scheduler::lane_order` decides the order an
  epoch's lanes are *run* in, separately from the order the plan numbers them
  in: plan order, reversed, or a per-epoch permutation. A lane's number never
  moves, so it keeps its position space and its allocation partition and a
  reordered run stays comparable to a plan-order one. This is what turns
  canonical commit and I25 from claims about a machine that only ever chose one
  order into properties a run can fail -- `tests/lane_order.rs` requires the
  effect log's application sequence identical across orders, and putting the
  applier back inside the lane loop fails it. A reordered run gives up I23's
  clause 2 and owes I18 after `order::in_position_order`, which is the exemption
  §4.2 wrote when the clause was written. Still one thread.
- The first threads (v0.3 §4.7). `CpuReferenceBackend::with_threads` evaluates a
  batch's elements across OS threads, split by `chunks_mut` so each owns a
  disjoint run of output elements over a shared immutable input. The safety
  argument is entirely the body language's: pure, reads the frozen input and
  never the output, writes only its own element -- rules stated for I19 that
  turn out to be exactly what threading needs. A knob rather than a default,
  because I20 makes this backend the definition and a definition should be the
  plainest reading of a body available. This is *not* the concurrent executive;
  lanes still run one after another.
- Lane-local allocation (v0.3 §4.8). `GenTable::shard` opens an allocator over
  one partition's unminted slot numbers, taken by `&self` so an epoch can open
  one per lane from a shared borrow and the lanes own them independently;
  `merge` folds one back. A shard holds only the slots its lane mints, so the
  table stays readable and a step reads pre-epoch state from the table and its
  own new entities from the shard -- which is the read-back §4.3 (2) requires,
  since a step stores references it allocated into opaque frame bytes. A shard
  does not recycle freed slots, because popping a shared free list is the
  coordination partitions exist to remove; that changes which slot numbers a run
  mints and nothing about what it does. This is the allocator a threaded
  executive needs. What has no lane-local form yet is mailboxes, futures,
  capability spaces and object payloads.
- Threaded epochs (v0.3 §4.9). `CpuReferenceBackend::evaluate_epoch` runs an
  epoch's collectives side by side, which is the axis element threading cannot
  serve: many small cohorts have too few elements each to fill a thread. The
  two are alternatives rather than layers, so a request inside a threaded epoch
  runs single-threaded. Results stay in request order -- the caller publishes
  result i into collective i -- and one failed request fails the epoch, which is
  the contract the sequential path already had.
- I25 clause 2 (v0.3 §4.12). An epoch's lanes must not be decided by one bounded
  resource. Two of them: a domain's process quota (`tests/domain_quota.rs`) and a
  receiver's mailbox capacity (`tests/mailbox_capacity.rs`). In both, two lanes
  draw and the one that ran first wins, so the same workload under `Plan` and
  `Reverse` is not I18-equivalent -- the counterexamples §4.6 did not have. Both
  runs leave a legal state, so only comparing them reports it, and clause 1
  cannot: the dependence carries no ≺ edge, just a counter or an occupancy. The
  condition is a winner *and* a different loser -- everyone refused is not a
  race. `ProcessCreationRefused` and `MessageSendBlocked` are what let the
  checker tell a bound that bit from a bound with room. The bug this turned up:
  `LaneView::create_process` was infallible, so a full domain aborted the host
  process instead of faulting the step.
- I25 clause 2's third resource, and the correction it forced (v0.3 §4.13). The
  same mailbox drained rather than filled (`tests/mailbox_drain.rs`): several
  continuations of one process receive in one epoch, the message goes to
  whichever lane ran first, and `MessageReceiveBlocked` is the record a parked
  receiver had none of. What it corrected is the condition. A quota and a
  capacity hand out interchangeable units, so a winner and a different loser is
  right for them; a mailbox hands out identified messages, so four receivers and
  four messages disagree across orders with nobody refused. Clause 2 now asks
  per resource kind. It also stopped taking the lowest winner and looking for an
  unequal loser, which missed a lane that both won and lost -- and made the
  report the same text every run.
- I25 clause 2's fourth resource (v0.3 §4.14). A future takes one value
  (`tests/future_assignment.rs`): four lanes publish into one future and it
  takes the one whose lane ran first, `FutureResolutionRefused` being the record
  single assignment was enforcing itself without. It is the resource that cannot
  distinguish clause 2's two conditions -- one unit means there is never a
  second winner -- so do not cite it as evidence for where a fifth resource
  belongs. Walking §4.10's fifteen operations the way §4.13 said to leaves
  `await_future`'s `AlreadySettled` and `future_value` as decisions no *handler*
  can reach, because the only awaiting handler creates the future in the step it
  awaits. That is a fact about the handler set and it stops being true the
  moment a handler awaits a future it did not create.
- The fifth decision, and the wider question (v0.3 §4.15). `JOIN_AWAIT` is that
  handler: it awaits a future named by its frame, which somebody else made and
  somebody else resolves (`tests/foreign_await.rs`). Both of §4.14's blind spots
  close with it, `future_value` being what its second half reads. Under `Plan`
  the resolver goes first and the awaiter does not park; under `Reverse` it
  parks and is woken. Same final state, different route. `FutureAwaitSettled` is
  the fifth traced decision and the first that is *not* a refusal -- the await
  succeeds either way -- which is what widened the method from "operations that
  can say no" to "operations whose result another lane can decide". It is also
  the first workload clause 1 can see, and it sees only the order in which the
  awaiter parked: the reference order reports nothing from it.
- The sixth decision, and the first the comparison missed (v0.3 §4.16).
  `POLL_FUTURE` reads a future without awaiting it (`tests/future_poll.rs`).
  Before this section nothing reported the race at all -- not clause 1, not
  clause 2, and not `conforms_traces`, because what the poll saw went into a
  frame. `POLL_ACT` makes it undeniable by sending a message in a later epoch if
  the poll saw a value: the runs then disagree in epoch 1 while epoch 0, which
  decided it, is clean by every clause. The fix makes the read governed like the
  view's other reads -- `AWAIT`, an authority pair, and `FutureStateObserved`
  recording what it saw. Unlike the five before it, this event fires in existing
  workloads: fourteen test files reach it, all pass, and the example reports are
  byte-identical, which says no existing workload polls a future its own epoch
  is resolving.
- The last ungoverned read, and the first that is not a race (v0.3 §4.17).
  `LaneView::continuations` handed a step the whole continuation table. All
  three call sites named the running continuation and read `run_class`, which
  Phase E fixed before any lane ran, or `frame`, which is written once -- so
  reordering finds nothing, and that is a fact about the handlers rather than a
  limit of the trace. The surface is what was wrong: `apply_step_result` runs
  inside the lane loop, so sibling descriptors do change mid-epoch, and a step
  reading one would have hit §4.16's three blindnesses exactly. Narrowed rather
  than governed, because the epoch loop already holds both answers and an
  authority pair per frame load would buy nothing: `dispatch` takes the run
  class as an argument, `LaneView::frame` is a copy taken before the step, and
  the table is a `compile_fail`. No event added, no trace changed. Eight
  handlers now take `_cont`, which is the count of handlers that wanted their
  continuation only to ask the table about it.
- A lane-local trace buffer (v0.3 §4.11). A lane produces trace events into
  `lane_trace` and `leave_lane` appends them, which is §4.4's move applied to
  the trace. `logical_time` is handed out at the drain rather than at emission,
  so the run's one counter is host-side and a lane touches nothing shared to
  emit. This is what §4.10's *read* category was waiting on: reading is a
  governed effect and the authority decision is traced, so a read writes one
  place and that place is now per-lane. The drain deliberately does not sort --
  see the trap below. Entering a lane over an undrained buffer panics.
- A sealed step surface (v0.3 §4.10). A handler takes `executives::lane::LaneView`
  rather than `&mut Kernel`, and the view offers fifteen operations -- measured
  from what `cpu_scalar` and `executives::ant_colony` already called, not
  chosen. No `Deref` to `Kernel` and no constructor outside the crate, both as
  `compile_fail` doctests with a passing null beside them. The value is that an
  operation with no lane-local form is now a compile error inside a step, so
  the concurrent executive journals all fifteen operations. Its access set
  explicitly covers the four cross-lane writes (`enqueue_message`,
  `receive_message`, `resolve_future`, `await_future`), while reads, allocation,
  and own-frame writes use the same closed journal and conflict validator.
- An effect log (I24). A step no longer writes a runnable bin; it produces the
  entries it wants and the kernel applies them, in the order the plan puts the
  producing lanes in. `Scheduler::enqueue` demands a token only the applier can
  build, so writing a bin inline is a compile error.
- Canonical commit (v0.3 §4.5). The applier moved out of the lane loop: an
  epoch runs every lane, then applies what they produced, sorted by the position
  each effect was produced at. Nothing an epoch commits depends on the order its
  lanes ran, so lanes are reorderable — which was the last of the device
  scheduler's four obligations. The price is I25: no ≺ edge may join two lanes
  of one epoch, which is §4.3 (3)'s measured precondition asked of every run
  instead of once. The existing suite passed unmodified, which is that
  measurement coming true rather than a weak result. Cancellation now withdraws
  the entries its epoch has not applied instead of applying the journal first.

### Semantic boundary

No entity or invariant named by `SOMA-v0.2.md` remains absent, and v0.3 §1–§3
are implemented and checked, as are §4.1, §4.2 and §4.4. A device-resident
scheduler, a distributed implementation, and hardware performance results remain
beyond conformance — scoped in v0.3 §4–§6, and not guarantees silently claimed
by the current machine. In particular, admission is order-independent, the
trace's order is reconstructible from position, and bin entries are applied
rather than performed — but *execution* is still not reorderable. The applier
runs at the end of each lane, and everything other than bin entry (mailboxes,
futures, capability spaces, the object tables) is still written as the step
runs, so what a lane observes depends on when it ran. Canonical commit is what
would change that, and it is not done.

---

## 4. Measurements

The largest ratio uses the weakest baseline. Keep the controls and caveats with
the result.

1. **Cohorting beats a persistent FIFO by 1.85–3.27× useful lane occupancy.**
   True, and against a weak baseline.
2. **Against a competent hand-written bulk frontier, it ties exactly, 1.00× in
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

These are *structural bounds* computed from how continuations group, not
hardware measurements. A lane group spanning `k` run classes is charged `k`
masked dispatches because a uniform-dispatch executive cannot do better. Real
hardware may do worse. The irregular-arrival result is a **trace-driven policy
comparison**: an arrival
trace is generated once and both policies are scored against it, so a node's
ready tick is an input rather than a consequence of when its parent ran, and the
SOMA side is a model of the scheduler's binning rather than the kernel running.
Nothing here touches throughput or scheduler overhead.

---

## 5. Semantic slices, in dependency order

### 5.1 Capabilities (I10)

Capability enforcement was the dependency that constrained every later slice.

**Settled in design: [docs/SOMA-CAPABILITIES.md](SOMA-CAPABILITIES.md).**
Authority is checked at operation, not at reference resolution, and the
operation set is closed by making kernel state private so a bypass is a compile
error. That note carries the full check surface, a staged implementation order
where every step passes tests, and the two consequences it forced: object
ownership is defined entirely by capabilities, and exported authority survives
the exporting holder's failure.

All eight steps are complete. Every right used by a reachable operation is
enforced, message sends delegate payload `READ` authority, parked `AWAIT`
authority is rechecked on resume, and authority decisions/effects make I10c
trace-checkable. Object ownership is derived from live capability holders; the
advisory owner/mode/count fields are gone and I9 is subsumed by I10b. A fault
reclaims the failed process's local capability space, while exported roots in
other spaces remain valid.

### 5.2 Process-state ownership (spec §6.2)

Settled. Every continuation declares `ReadOnly` or `Mutable` access to its
process's canonical state object. Admission permits any number of read-only
continuations but at most one mutable declaration per process per epoch.
`process_state_bytes_mut` also requires that exact continuation to be active;
the generic object mutation API rejects process-state objects. `Pure` processes
reject mutable continuation creation. I13 checks the resulting trace and has a
negative fault-injection test.

### 5.3 Failure containment and cancellation (spec §6.3, §6.4)

Settled. `Fault` marks the triggering continuation faulted and its siblings
cancelled. Explicit cancellation is immediate at quiescence or enters
`CancelPending` until the active continuation commits. Both paths settle every
owned pending future (`Failed` or `Cancelled`), drain the mailbox, wake external
future waiters and capacity-blocked senders, remove local waiter registrations,
and reclaim the local capability space. Terminal mailboxes reject new sends.
Exported authority roots remain valid.

### 5.4 Channels and collectives

Settled at the semantic-core level. Channels are first-class bounded entities.
`SEND`, `RECEIVE`, and `DESTROY` gate send, receive, and close; committed
messages carry an escrowed `READ` root so sender failure cannot revoke an
in-flight payload. Close rejects new sends, wakes waiters, and permits queued
messages to drain. I1, I5, I6, and I10b cover channel state.

`BatchEvaluate` accepts a frozen input array plus count and stride and creates a
completion future. Completion requires a frozen output array, resolves the
future exactly once, and failure or cancellation of the owner settles both
entities consistently. This is orchestration and publication semantics only.
The minimal module layer names evaluator bodies; the CPU and Metal reference
backends currently realize the example `2*x+1` body used to validate
placement-independent publication.

### 5.5 Validation workloads

Dynamic constraint search and a theoretical bounded streaming graph now exist.
The stream uses numbered records, deterministic source/sink stages, bounded
first-class channels, and optional producer failure. It verifies FIFO ordering,
lossless back-pressure, and survival of the committed prefix. The controlled
actor tree covers supervision propagation and restart. The irregular two-input
join verifies atomic readiness, waiter migration, FIFO pairing, skew-induced
back-pressure, and committed-prefix survival after producer failure.

### 5.6 Supervision

The semantic core is implemented. A supervisor may create a direct child; the
child's completion, failure, or cancellation appends exactly one typed notice
to a reliable kernel control queue. This queue is intentionally separate from
the bounded user mailbox, so failure reporting cannot be dropped by ordinary
back-pressure. One waiting supervisor continuation wakes deterministically.
Relationships select notification-only containment or escalation. On an
escalating child failure, the direct supervisor fails after the child notice is
committed; its own relationship may propagate failure farther up the tree. I15
checks relationship, notice, waiter, required-escalation, restart lineage, and
retry-exhaustion integrity. Restart creates a fresh process identity from a
registered entry-continuation template; exhaustion escalates.

`experiments/supervision_tree.rs` supplies the control that selected this slice.
With no fault, notification and escalation are identical. With one failed leaf,
notification leaves its branch alive and the root uninformed; escalation fails
that branch and delivers one root notice; restart replaces only the failed leaf.
All policies preserve the sibling branch and share the same fault-free control.

### 5.7 Minimal IR

Implemented in `compiler/ir.rs`. A module names batch evaluators, one
frozen-array element stride, and entry/completion resume points with run classes
and state-access declarations. Validation rejects empty/duplicate identities,
zero strides, and ambiguous resume points. Instantiation records the evaluator
ID in the `BatchEvaluate` descriptor. `Module::parse` provides a deliberately
small line-oriented surface; `Module::load` creates an actor-relative immutable
manifest, and loaded instantiation links the collective to it. I17 rejects
invalid manifests and evaluator/stride mismatches. There is intentionally no
device, lane width, placement, or launch concept in this IR.

### 5.8 Domains and execution contracts

Implemented. Every process and object belongs to a live logical domain. Domain
creation and explicit placement require actor-relative authority, and a domain's
monotonic process-creation quota is checked by I16. Execution contracts are
first-class capability targets attached to continuations; I16 checks step and
frame-byte bounds. Hardware placement, lane shape, relaxed determinism, and
wall-clock deadlines are rejected because the abstract machine cannot enforce
them.

### 5.9 Physical batch implementation

`executives/batch.rs` is the boundary between semantic collectives and physical
execution. A backend receives only frozen bytes and shape metadata. The common
path freezes its output and completes the collective, so backends cannot bypass
capability or publication rules. Underfilled work and `Unavailable` accelerator
responses spill to CPU; switching backend for an evaluator at a collective
boundary increments migration accounting.

With `--features metal` on macOS, `MetalBatchBackend` compiles a Metal Shading
Language compute kernel, dispatches it through shared buffers, waits for command
completion, and returns bytes through that common path. The integration test
runs on real Apple GPU hardware and compares its output and semantic trace with
the CPU path.
This is not yet a persistent device scheduler, arbitrary evaluator compiler, or
hardware performance result. `src/experiments/territories.rs` remains the
placement-policy model.

### 5.10 The effect log

Implemented in `kernel/effects.rs`, and the third of B's four obligations
(v0.3 §4.4). A step produces the runnable-bin entries it wants rather than
writing them, and the kernel applies the lane's journal in production order.
`Scheduler::enqueue` demands a token only the applier can construct, so an
inline bin write is a compile error; I24 checks that the applied indices are
complete, that sorting the log by production position recovers the application
order, and that the scheduler's own entry count is fully accounted for by the
log.

Read the scope claim exactly. Only bin entry is mediated. Mailboxes, futures,
capability spaces and the object tables are still written as the step runs, and
allocation is still eager because v0.3 §4.3 (2) shows it has to be. That does
not change with canonical commit and cannot: a step stores references it
allocated into opaque frame bytes, so eager allocation is not a stage of the
refactor that has yet to happen. What the applier's move to the epoch boundary
buys is that no lane can observe another lane's *bin entry or status write*, and
what it costs is I25 — the obligation that no workload lets a lane observe
another lane by any of the paths that remain.

---

## 6. Test discipline

Every invariant in
`semantics/invariants.rs` has two tests: the reference model satisfies it, *and*
it catches a state that violates it. The same rule applies to regression tests.
Three kernel fixes were verified by reintroducing the bug and confirming the
new test failed. If you add a check, add its failing case.

Every comparison needs a control that shows no effect. The cohorting study
reports 1.00× for a single run class. The irregular study reports 1.00× for zero
irregularity and 1.00× for a zero wait budget. Those nulls are what distinguish
measuring the mechanism from measuring the harness.

Use the strongest reasonable baseline. The bulk frontier's first version
sorted the frontier and then cut it into fixed lane groups that straddled class
boundaries. SOMA came out 14 percent ahead. Segmenting each class, which is
what a competent engineer would write, erased the advantage entirely. The near
miss is the reason `dispatch_cost` and `search_step` are *shared* by both sides:
neither can drift into different work or different scoring.

The arrival experiments verify that every policy dispatches every arrival
exactly once. Any occupancy figure from a policy that silently drops or double-counts is
meaningless.

Mark absent things absent. Do not write a vacuous check to make a table look
complete. Each newly implemented entity needs both behavioral tests and a
fault-injection case proving its invariant can reject illegal state.

Every measurement depends on comparable runs. Sort before any scheduling
decision that would otherwise depend on `HashMap` iteration order.
`trace_snapshot` is the equality used for run comparison.

---

## 7. Traps

- **An index narrows a search. It does not decide the answer.** `Ref64::key()`
  is partition and slot, not kind and not generation, so a process and an object
  in the same slot share a bucket in `kernel::capability_space`. Every lookup
  re-checks the whole reference. Skipping that check let `revoke_target_right`
  revoke a capability over the wrong entity, and `CapabilityIntegrity` reported
  it only once reclamation started deleting things, several commits later.
- **Deleting an entity means purging the capabilities that name it**, and
  reclaiming a process means giving one back to its domain's
  `processes_created`, which `DomainContractIntegrity` checks against the *live*
  count despite the name. Miss either and the machine is smaller and illegal.
- **Releasing authority is not exercising it.** `AuthorityEffect` is what
  `NoUnauthorizedEffect` demands an adjacent grant for. Letting go needs no
  permission beyond having held it. That is why `AuthorityReleased` is a
  separate event kind, and why reusing the first one makes every release
  illegal.
- **Benchmarks that start from `Kernel::new()` cannot see the defect that
  matters.** Three hot paths scanned a structure that only grows, so a run doing
  n operations did O(n²) work with every test passing — nothing about the
  *result* changes, only how long it takes. `examples/growth_sweep` is the shape
  that finds them: fix an operation, grow one structure underneath it, re-time.
  Point it at anything new. See `docs/PERFORMANCE.md` §4.
- **Continuation status must change through `set_continuation_status`.**
  `retire_process_if_idle` no longer scans the continuation table; it reads the
  per-process count that helper maintains. A status write that bypasses it
  leaves the count stale, which I3 reports. (Was a trap; now a rule.)
- **16-bit generations** no longer wrap: `GenTable::delete` retires an exhausted
  slot instead. Staleness detection is guaranteed rather than bounded, at a cost
  of one withdrawn slot per 65,535 recycles. What is *still* missing for a
  distributed implementation is a node identity in `Ref64`, so two nodes cannot
  allocate colliding references. See v0.3 §1.2.
- **Admission must not decide from position.** `scheduler::admission::admit`
  sees the whole candidate set before placing any of it, and every field it
  reads is state rather than a position in that set. Deciding as the list is
  walked — the natural way to write it — reintroduces the race I22 exists to
  rule out, and it changes no trace the reference interpreter produces, so no
  behavioural test would notice. `Admission` is sealed for that reason: an
  epoch that builds its own does not compile. If you find yourself wanting to
  widen that, the thing you are about to break is checkable only on hardware.
- **Trace emission must stay lane-attributed.** `enter_lane`/`leave_lane` in
  `epochs.rs` are what put an executing continuation's events in a sequence
  space of its own. Delete the call and I23's reconstruction test still passes —
  a single host counter satisfies "sorted by position equals emitted order"
  perfectly — while the trace goes back to needing a shared clock. Clause 3 of
  I23 is the only thing that reports it.
- **"The reordering found nothing" is a statement about the workloads run.**
  §4.6 said it, and it stood until someone built a workload it was wrong about:
  a bounded domain makes the same reordering disagree (v0.3 §4.12). Every
  reorderability result here is conditional on what the workload touches, so a
  new shared, bounded, or refusing resource needs its own reordered run rather
  than inheriting the old conclusion. The place to look is anything a lane reads
  a *decision* off rather than merely writes. That question found all three
  cases so far — a domain quota, a mailbox's capacity, and the same mailbox's
  occupancy — none by a failing test. Enumerate the *operations* that can say
  no, not the resources: §4.12 listed the resources, concluded nothing else was
  reachable from a step, and missed the mailbox it had just finished filling
  because a receive refuses a different caller for a different reason (§4.13).
- **Enumerate the operations whose result another lane can decide, not the ones
  that can say no.** §4.13's correction was itself too narrow, and §4.15 is what
  showed it. `await_future` is on `LaneView`'s fifteen, was walked past in four
  successive rounds, and never fails: it returns normally whether it registered
  a waiter or found the value already published. Which of the two it returns is
  decided by whether a resolving lane of the same epoch went first. A refusal is
  the special case where one of the results is a failure, and it is conspicuous
  because it faults a step — which is exactly why looking for refusals kept
  finding them and kept missing this. The next candidate under the wider
  question is a *read*: `continuations()` hands a step the whole table, and
  `apply_step_result` writes a continuation's status inside the lane that
  produced it. No handler reads another's, so it is unreachable in the §4.14
  sense — a fact about the handlers, with the same expiry.
- **A read can be a race, and the reordering discipline does not always catch
  it.** §4.16 is the one case where running the workload again in another order
  and comparing reported *nothing*: a poll of a future put what it saw in a
  frame, and a frame is not observable behaviour, so two I18-equivalent runs
  left different state. Clause 1 had no edge (nothing parked, so nothing was
  woken) and clause 2 had no event. When the divergence finally reached the
  trace it was an epoch later, in an epoch that had not raced anything, while
  the epoch that decided it passed every clause. If a step reads state, ask what
  records the read — "the comparison will catch it" is not always true.
- **`future_value` is a governed read, and `Kernel::future_value` is not.**
  `LaneView::future_value` takes `&mut` and an actor, authorizes `AWAIT`, and
  emits `FutureStateObserved`; the kernel's plain `future_value` stays the
  host's ungoverned read, for tests and the epoch machinery. Do not reach for
  the second from a step — the point of the split is that a lane's look at a
  future is recorded. The event carries *both* outcomes (`auxiliary` is 1 for
  resolved, 0 for pending) because a poll that saw nothing was decided by the
  lane that had not yet run, exactly as much as one that saw the value.
- **A denial, a pending future, and a null reference are three answers.** The
  ungoverned read collapsed all three into `None`. A handler now faults on a
  denial, treats pending as pending, and skips the read entirely when the frame
  names no future — that last one is what keeps `EXPAND_RESUME_1` working for
  the tests that run it over an empty frame, and it is why the arm exists.
- **`Bounded` has an entry that is not a bounded resource.** `FutureSettlement`
  dispenses nothing; it is a future's state, written by a resolver and read by
  an awaiter. It is in clause 2 because the clause's subject was never
  boundedness — it is one lane's outcome being decided by another lane of its
  epoch — and a bounded resource was the first mechanism found for that. One
  future is therefore keyed twice, as an assignment (§4.14) and as a state
  (§4.15), and a `FutureResolved` is a win under both keys. That is why the
  event-to-resource match in `bounded_resource_independence` produces a *list*
  rather than one entry; a resolve that registered under only one of them makes
  the other key winnerless and silently unreportable.
- **Clause 1 can see a race, and is blind in the reference order when it does.**
  §4.15 is the first workload where `cross_lane_edges()` is non-empty, because a
  wake names *another lane's continuation* and the other four races' events name
  only their resource. It is non-empty in the `Reverse` run and empty in the
  `Plan` one, where the awaiter never parked — so the plan-order run, the one
  you would naturally do, reports nothing from clause 1. A single run satisfying
  clause 1 is not evidence its lanes were independent. Running the workload in a
  second order and comparing is what reaches the case at all.
- **A null for one bounded resource is not a null for another.** Clause 2 asks a
  different question depending on what the resource hands out. A quota and a
  mailbox capacity dispense interchangeable units, so two lanes that both
  succeed decide nothing and contention needs a refusal. A mailbox's messages
  are *identified* — a sender, a sequence number — so two lanes that both
  succeed took different messages and the order decided which. "Four receivers,
  four messages" is not room for everyone; it is a race with no refusal in it.
  Adding a resource means asking which kind it is before reusing either
  condition (v0.3 §4.13).
- **The lane trace buffer drains in emission order, and must not sort.**
  `drain_lane_trace` appends what the lane produced in the order it produced it,
  which is why the buffer changed no run. Sorting by position there looks like an
  improvement — it would make I23's clause 2 hold even of §4.6's reordered runs
  — and it is the same mistake as folding `order::in_position_order` into
  `conforms`: a reference that re-serialises its own trace makes the clause hold
  of anything that emits. §4.2 wrote the exemption on purpose. No test fails if
  you do this; what breaks is what clause 2 means.
- **A message's `logical_timestamp` still reads the run's clock.** It is the one
  shared-clock read left inside a lane (v0.3 §4.11). Nothing reads the field and
  nothing orders by it — I6 orders per pair by `sender_sequence` — so `clock_now`
  keeps it reading the count including the lane's undrained events, and the value
  a run stamps is what it stamped before. A concurrent executive drops the field
  or stamps it from a position; that is an ABI decision, so it is recorded rather
  than quietly changed.
- **A bin entry is written by the applier, not by the step.** `kernel/effects.rs`
  is the only place that can build the `Committing` token `Scheduler::enqueue`
  demands, so a handler that enqueues inline does not compile. If you find
  yourself reaching for `raw::enqueue_unmediated` outside a fault injection, the
  thing you are about to undo is I24 clause 3 — and with it the reason canonical
  commit was one line away rather than a rewrite.
- **Cancellation withdraws; it does not apply.** `cancel_process_continuations`
  takes the entries its epoch has produced and not yet applied out of the
  journal, then `remove_all`s what earlier epochs applied. Calling
  `apply_epoch_effects()` there instead — which is what it used to do, back when
  the journal held one lane — commits the epoch's earlier lanes in the middle of
  a later one, which is the ordering canonical commit exists to remove.
  `cancelling_a_process_withdraws_the_effects_its_lane_produced` fails if you
  do. The note this replaced said no handler reached the case and no test
  covered it. One does reach it: the `CancelPending` check at the end of
  `apply_step_result` runs after the branch that emits the resume.
- **A bare slot is not an identity.** Two partitions each mint slot 7. Anything
  keyed or compared by `.slot` is wrong; use `Ref64::key()` for map keys and the
  whole reference for comparison. `.slot` belongs in error messages and nowhere
  else. I8 shipped with this bug for exactly as long as it took to turn
  partitioning on — the invariant checker caught it on the first run.
- **An entity never goes in `auxiliary`.** That field is numeric — sequence
  numbers, counts, right masks. A slot number put there carries no kind and no
  generation, so the identity correspondence cannot translate it and cannot
  tell it apart from a sequence. Use `subject` for the entity an event is
  about, `causal` for the entity two events are ordered through.
- **Trace `causal` is load-bearing for I18.** Adding an event that participates
  in a cross-entity happens-before edge without setting `causal` silently drops
  that edge, and a dropped edge makes the conformance checker weaker without
  making any test fail. `trace_caused` is the way to set it.
- **`§n` comments refer to `docs/SOMA-P1.md`**, the historical contract, not to
  the v0.2 spec. The v0.2 spec uses `I1..I17` and `§6.x`.
- **`experiments/` is not the machine.** Nothing there is part of SOMA's
  semantics. It is measurement scaffolding.
- **The step budget check must stay before dispatch** in `epochs.rs`. Faulting
  after commit leaves a faulted continuation enqueued by that commit, which
  violates I7. This was a real bug.
- **`Complete` is about a continuation, not a process.** A process retires when
  its last continuation does. Getting this wrong strands live continuations on a
  dead process. See spec §6.1.

---

## 8. Suggested first week

1. Read `docs/SOMA-v0.2.md` end to end, then `src/semantics/invariants.rs`
   alongside `tests/semantics.rs`. That pair is the fastest way into the model.
2. Run the examples. The numbers in §4 should reproduce exactly. If they don't,
   something is non-deterministic and that is a bug worth chasing before
   anything else.
3. Break something on purpose. Flip a condition in `commit.rs` and confirm the
   invariant checker catches it. That tells you the safety net is real before
   you rely on it.
4. Read `docs/SOMA-v0.3.md`. §2 and §3 explain the two pieces of machinery
   most likely to surprise you — why trace equality had to go, and why both
   backends used to agree about nothing. Then pick up §4 (the persistent device
   scheduler). All four of its obligations are discharged in §4.1, §4.2, §4.4
   and §4.5. Then read `docs/SPECULATIVE-EPOCHS.md` for the concurrent CPU
   implementation those obligations now support.
