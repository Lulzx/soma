# SOMA-P1: Minimal Executable Kernel Contract

## Version 0.1

## 1. Purpose

SOMA-P1 is the smallest implementation capable of testing SOMA's central hypothesis:

> Independent persistent processes can execute efficiently on heterogeneous SIMD hardware when their ready continuations are dynamically regrouped into coherent physical execution cohorts.

SOMA-P1 is not intended to implement the full architecture.

It must prove or disprove three mechanisms:

1. Persistent engine-independent processes.
2. Bounded resumable continuations.
3. Dynamic continuation cohorting.

Everything else is included only where required to test those mechanisms.

---

# 2. Phase-1 scope

SOMA-P1 implements:

| Component          | Phase-1 support                         |
| ------------------ | --------------------------------------- |
| Protection domains | Logical domains. Trusted modules        |
| Processes          | Persistent, serial-state processes      |
| Objects            | Unique mutable and frozen shared        |
| Capabilities       | Generation-checked table references     |
| Messages           | Typed, bounded asynchronous channels    |
| Futures            | Single-assignment futures               |
| Continuations      | Explicit bounded state machines         |
| Execution shapes   | Scalar and lane                         |
| Engines            | CPU scalar and GPU SIMD-lane            |
| Scheduling         | Epochal hierarchical scheduler          |
| Cohorting          | Dynamic grouping by continuation class  |
| Failure            | Software traps and process cancellation |
| Determinism        | Event tracing and deterministic replay  |
| Collectives        | One batch-evaluation collective         |
| Migration          | CPU/GPU placement at epoch boundaries   |

Deferred until later:

* arbitrary untrusted shader execution.
* hardware-enforced GPU address-space isolation.
* tensor, tile, media, and Neural Engine execution.
* general borrowing.
* shared mutable objects.
* transactional messages.
* transparent hardware preemption.
* distributed execution.
* process checkpoint rollback.
* and dynamic code loading.

---

# 3. ABI principles

All shared ABI structures use:

```text
fixed-width integer fields
little-endian representation
16-byte minimum alignment
no native language pointers
no implementation-defined enums
explicit ABI version numbers
generation-checked references
```

A SOMA program does not place raw CPU or GPU addresses in persistent state.

References point into operating-system-managed tables.

Every ABI object begins with:

```cpp
struct AbiHeader {
    uint16_t abi_version;
    uint16_t structure_kind;
    uint32_t byte_length;
};
```

Unknown trailing fields may be ignored when the ABI version permits forward compatibility.

---

# 4. Reference format

All kernel-managed entities use a compact generational reference:

```cpp
struct Ref64 {
    uint32_t slot;
    uint16_t generation;
    uint8_t kind;
    uint8_t flags;
};
```

Possible kinds include:

```text
domain
process
object
capability
continuation
channel
future
contract
collective
module
```

A reference is valid only when:

```text
reference.slot exists
reference.kind matches the table
reference.generation equals the current slot generation
the requesting domain has authority to use it
```

Deleting an entity increments its slot generation before reuse.

This prevents stale references from silently targeting newly allocated entities.

`Ref64` is not itself an authority. It is merely a table reference.

---

# 5. Capability ABI

Application code receives a `CapRef`, which resolves through the calling domain's capability table.

```cpp
using CapRef = Ref64;

struct CapabilityEntry {
    AbiHeader header;

    Ref64 object;

    uint64_t offset;
    uint64_t length;

    uint32_t rights;
    uint16_t ownership_mode;
    uint16_t transfer_policy;

    uint32_t object_version;
    uint32_t valid_until_epoch;

    Ref64 parent_capability;
};
```

Phase-1 rights are:

```text
READ
WRITE
FREEZE
TRANSFER
SEND
RECEIVE
RESOLVE
AWAIT
DESTROY
```

Phase-1 ownership modes are:

```text
UNIQUE_MUTABLE
FROZEN_SHARED
KERNEL_OWNED
```

Capability derivation can only reduce:

* accessible range.
* rights.
* lifetime.
* or transferability.

Derivation does not amplify authority.

---

# 6. Object ABI

```cpp
struct ObjectDescriptor {
    AbiHeader header;

    Ref64 id;
    Ref64 owner_domain;

    uint64_t byte_length;
    uint64_t physical_mapping_token;

    uint32_t version;
    uint16_t object_kind;
    uint16_t ownership_state;

    Ref64 unique_owner;
    uint32_t reader_count;
    uint32_t flags;
};
```

Phase-1 object kinds are:

```text
RAW_BYTES
PROCESS_STATE
CONTINUATION_FRAME
MESSAGE_PAYLOAD
FROZEN_ARRAY
FUTURE_VALUE
TRACE_BUFFER
```

The `physical_mapping_token` is private to the kernel and engine executives.
The user API does not expose or construct it.

## 6.1 Unique mutable object

A unique mutable object has exactly one write-capable owner.

The owner may:

* read it.
* write it.
* freeze it.
* or transfer ownership.

Sending a unique object transfers authority atomically:

```text
sender loses WRITE and TRANSFER rights
receiver receives the unique capability
message becomes visible only after transfer commits
```

## 6.2 Frozen shared object

Freezing performs:

1. Completion of all prior writes.
2. Version increment.
3. Transition to immutable state.
4. Publication to one or more readers.

A frozen object does not return to mutable state in Phase 1.

Mutation requires allocating a new object.

---

# 7. Process ABI

```cpp
struct ProcessDescriptor {
    AbiHeader header;

    Ref64 id;
    Ref64 domain;
    Ref64 supervisor;

    CapRef state;
    Ref64 inbox;
    Ref64 urgent_inbox;

    Ref64 active_continuation;
    Ref64 waiting_on;

    uint32_t status;
    uint16_t process_mode;
    uint16_t base_priority;

    uint64_t compute_quota;
    uint64_t memory_quota;
    uint64_t deadline_ns;

    uint32_t last_committed_epoch;
    uint32_t failure_count;
};
```

Phase-1 process modes:

```text
SERIAL
PURE
SYSTEM
```

A `SERIAL` process may have many suspended continuations, but no more than one state-mutating continuation may execute at once.

Process states are:

```text
CREATED
RUNNABLE
RUNNING
WAITING
CANCEL_PENDING
FAILED
TERMINATED
```

A process contains no permanently assigned execution thread.

---

# 8. Continuation ABI

A continuation is the actual schedulable unit.

```cpp
struct ContinuationDescriptor {
    AbiHeader header;

    Ref64 id;
    Ref64 process;

    uint32_t run_class;
    uint32_t resume_point;

    CapRef frame;
    Ref64 dependency;

    uint64_t deadline_ns;

    uint32_t remaining_steps;
    uint16_t priority;
    uint16_t status;

    uint32_t created_epoch;
    uint32_t last_run_epoch;
};
```

Continuation states are:

```text
NEW
RUNNABLE
RUNNING
WAITING
COMPLETED
CANCELLED
FAULTED
```

A continuation may return one of the following results:

```cpp
enum StepKind : uint32_t {
    STEP_COMPLETE,
    STEP_YIELD,
    STEP_AWAIT,
    STEP_SEND,
    STEP_SPAWN,
    STEP_FAULT
};

struct StepResult {
    uint32_t kind;
    uint32_t next_run_class;

    Ref64 target;
    CapRef value;

    uint32_t consumed_steps;
    uint32_t flags;
};
```

A continuation must return control before exhausting its declared maximum step budget.

In Phase 1, this is compiler- or programmer-enforced rather than hardware-enforced.

---

# 9. Run classes

A run class describes continuations that can execute through the same physical implementation.

```cpp
struct RunClassDescriptor {
    AbiHeader header;

    uint32_t id;
    uint32_t module_function;

    Ref64 execution_contract;

    uint16_t engine_mask;
    uint16_t cohort_width;
    uint16_t priority_class;
    uint16_t flags;

    uint32_t frame_type;
    uint32_t result_type;
};
```

A run class captures:

```text
code implementation
resume point
execution shape
frame layout
precision mode
resource class
supported engines
```

This means the scheduler does not repeatedly inspect arbitrary continuation metadata.

Every yielded continuation already knows the exact queue into which it belongs:

```text
next_run_class → runnable bin
```

This is the simplest implementation of continuation cohorting.

---

# 10. Execution contract ABI

```cpp
struct ExecutionContract {
    AbiHeader header;

    Ref64 id;

    uint8_t shape;
    uint8_t placement_policy;
    uint8_t precision_policy;
    uint8_t determinism_policy;

    uint16_t minimum_parallelism;
    uint16_t preferred_parallelism;

    uint32_t maximum_steps;
    uint32_t local_memory_bytes;

    uint64_t deadline_ns;
    uint64_t expected_read_bytes;
    uint64_t expected_write_bytes;

    uint32_t objective_flags;
    uint32_t contract_flags;
};
```

Phase-1 shapes are:

```text
SCALAR
LANES
```

Placement policies are:

```text
ANY
PREFER_CPU
PREFER_GPU
REQUIRE_CPU
REQUIRE_GPU
```

The runtime may override a preference but must obey a requirement.

---

# 11. Message ABI

```cpp
struct MessageDescriptor {
    AbiHeader header;

    uint32_t type_id;
    uint32_t flags;

    Ref64 sender;
    Ref64 receiver;

    uint64_t sender_sequence;
    uint64_t logical_timestamp;

    CapRef payload;
    CapRef transferred_capability;

    Ref64 completion_future;
};
```

Phase-1 guarantees:

```text
at-most-once delivery
ordered delivery per sender–receiver pair
release on committed send
acquire on receive
```

A message is not visible until:

1. Its payload is complete.
2. Any capability transfer is validated.
3. The sender's sequence number is assigned.
4. The message descriptor is committed.

Mailbox capacity is bounded.

Sending to a full mailbox returns a future that resolves when capacity becomes available. The sender does not spin.

---

# 12. Future ABI

```cpp
struct FutureDescriptor {
    AbiHeader header;

    Ref64 id;
    Ref64 owner_domain;

    uint32_t state;
    uint32_t waiter_count;

    CapRef value;
    Ref64 failure;

    Ref64 waiter_list;
    uint32_t resolved_epoch;
    uint32_t flags;
};
```

Future states are:

```text
PENDING
RESOLVED
FAILED
CANCELLED
```

Resolution is single-assignment.

Resolving a future publishes its result with release semantics.

Resuming a waiter acquires the result.

---

# 13. Runnable-bin structure

A general lock-free work queue is unnecessary for Phase 1.

SOMA-P1 uses **double-buffered append-only runnable bins**.

```cpp
struct RunnableBin {
    uint32_t write_count;
    uint32_t capacity;

    Ref64* entries;
};
```

Each run class has two buffers:

```text
current_epoch
next_epoch
```

During an epoch:

* workers consume only from `current_epoch`.
* new runnable continuations append only to `next_epoch`.
* an atomic increment reserves each append slot.
* the buffers swap at the next scheduling boundary.

Benefits:

* no concurrent pop operation.
* no per-entry reclamation.
* no ABA problem.
* deterministic epoch boundaries.
* cheap grouping by run class.
* natural bounded execution.

Messages and futures may awaken continuations into the next-epoch buffer.

Later implementations may permit same-epoch wakeups as an optimization.

---

# 14. Cohort construction

For a GPU lane run class with hardware width `W`:

```text
full cohorts     = floor(runnable_count / W)
remaining lanes  = runnable_count mod W
```

Each full cohort contains continuations from one run class.

```cpp
struct CohortDescriptor {
    uint32_t run_class;
    uint16_t width;
    uint16_t active_lanes;

    Ref64 continuations[MAX_COHORT_WIDTH];
};
```

All active lanes execute:

```text
same module function
same resume point
same frame schema
same numerical policy
different process and object state
```

The final partial cohort may be handled through one of four policies:

```text
RUN_PARTIAL
DEFER
SEND_TO_CPU
MERGE_WITH_GENERIC_CLASS
```

The policy is selected using measured cost rather than fixed ideology.

---

# 15. GPU executive

The Phase-1 GPU executive operates in bounded epochs.

Each epoch contains a fixed maximum number of scheduling rounds:

```text
1. Read eligible runnable-bin counts.
2. Reserve full cohorts.
3. Execute cohorts.
4. Store continuation results.
5. Append yielded or spawned work to next-epoch bins.
6. Publish messages and future resolutions.
7. Update accounting and trace state.
```

A cohort executes a uniform dispatch:

```cpp
switch (run_class) {
    case SEARCH_EXPAND_0:
        result = search_expand_0(frame, context);
        break;

    case SEARCH_EXPAND_1:
        result = search_expand_1(frame, context);
        break;

    case HEURISTIC_PREPARE:
        result = heuristic_prepare(frame, context);
        break;
}
```

Because the entire cohort shares one run class, this branch is uniform and does not introduce intra-cohort divergence.

Phase 1 may use a static dispatch table or generated switch. Dynamic module loading is not required.

---

# 16. CPU scalar executive

The CPU executive consumes:

* scalar-only run classes.
* partial GPU cohorts below the batching threshold.
* continuations nearing a deadline.
* and continuations whose divergence history predicts poor GPU efficiency.

The CPU implementation uses the same:

```text
continuation descriptor
frame layout
capability model
step result
message semantics
```

A continuation can move between CPU and GPU only at a continuation boundary.

No register state is migrated.

Its durable frame already resides in shared memory.

---

# 17. Placement policy

The initial placement score is:

```text
predicted execution time
+ queue delay
+ underfilled-cohort cost
+ expected memory interference
+ deadline risk
+ migration penalty
```

For each run class, the runtime records:

```cpp
struct RunClassStatistics {
    uint64_t cpu_executions;
    uint64_t gpu_executions;

    double cpu_mean_ns;
    double gpu_mean_ns;

    double gpu_active_lane_ratio;
    double cohort_wait_ns;

    double bytes_per_execution;
    double prediction_error;
};
```

The first policy is intentionally simple:

```text
if REQUIRE_CPU or REQUIRE_GPU:
    obey requirement

else if full GPU cohort exists:
    choose GPU

else if deadline is near:
    choose CPU

else if predicted batching benefit exceeds wait cost:
    defer for GPU cohort

else:
    choose CPU
```

This policy exists to generate evidence. It is not intended to be final.

---

# 18. Epoch lifecycle

Each epoch proceeds through eight phases.

## Phase A: Ingest

Import:

* newly created processes.
* CPU service responses.
* external messages.
* cancellation requests.
* and resolved I/O futures.

## Phase B: Validate

Check:

* capability generations.
* process status.
* object ownership.
* continuation budgets.
* and execution-contract validity.

## Phase C: Admit

Apply:

* domain quotas.
* process priorities.
* deadline policy.
* and runnable-bin capacity limits.

## Phase D: Place

Assign eligible continuations to:

* CPU scalar execution.
* GPU lane execution.
* or deferred batching.

## Phase E: Cohort

Partition GPU work into full and partial SIMD cohorts.

## Phase F: Execute

Run a bounded number of continuation steps and collect `StepResult` records.

## Phase G: Commit

Atomically publish:

* state transitions.
* messages.
* capability transfers.
* future resolutions.
* child processes.
* and continuation results.

## Phase H: Account

Record:

* execution time.
* active lanes.
* queue delay.
* bytes accessed.
* deadline outcomes.
* process quotas.
* and trace events.

The epoch then advances and runnable-bin buffers swap.

---

# 19. Process execution invariant

For every serial process:

```text
At most one continuation holding mutable authority over process state may be RUNNING.
```

A process may simultaneously have:

* one active mutating continuation.
* several waiting continuations.
* several pure child operations.
* and unresolved futures.

This invariant avoids general-purpose locking in the first implementation.

---

# 20. Failure model

Phase-1 failures include:

```text
INVALID_CAPABILITY
STALE_REFERENCE
BOUNDS_VIOLATION
OWNERSHIP_VIOLATION
STEP_BUDGET_EXCEEDED
INVALID_MESSAGE
MAILBOX_OVERFLOW
EXPLICIT_TRAP
CANCELLED
```

A software fault produces:

```cpp
struct FailureRecord {
    Ref64 process;
    Ref64 continuation;

    uint32_t failure_class;
    uint32_t run_class;

    uint32_t epoch;
    uint32_t source_location;

    CapRef diagnostic_payload;
};
```

The default supervisor policy is:

```text
cancel the failing continuation
mark the process failed
notify the supervisor
retain the trace and process state
continue unrelated processes
```

Phase 1 does not claim containment from arbitrary malicious native GPU code. All executable modules are trusted or compiler-validated.

---

# 21. Trace ABI

Every significant event emits a compact trace record:

```cpp
struct TraceEvent {
    uint64_t logical_time;

    uint32_t epoch;
    uint16_t event_kind;
    uint16_t engine;

    Ref64 process;
    Ref64 continuation;

    uint32_t run_class;
    uint32_t auxiliary;
};
```

Required events include:

```text
PROCESS_CREATED
MESSAGE_SENT
MESSAGE_RECEIVED
CONTINUATION_READY
CONTINUATION_PLACED
COHORT_CREATED
CONTINUATION_STARTED
CONTINUATION_YIELDED
CONTINUATION_WAITING
CONTINUATION_COMPLETED
FUTURE_RESOLVED
PROCESS_FAILED
PROCESS_CANCELLED
```

Deterministic replay consumes:

* initial object state.
* external inputs.
* message order.
* placement decisions.
* cohort membership.
* and random seeds.

A CPU interpreter executes the same continuation state machines for debugging.

---

# 22. Minimal source-level interface

The first source model can be implemented through generated state machines rather than a complete new language.

```rust
process SearchState(state: Unique<State>) {
    on Expand(request: ExpandRequest) {
        let node = state.current;

        let score = await heuristic(node);

        for movement in legal_moves(node) {
            spawn SearchState(apply(node, movement));
        }

        send request.reply <- score;
    }
}
```

The compiler or source transformer lowers this into:

```text
Expand.resume_0
    Receive request.
    Store request in frame.
    Spawn heuristic.
    Await heuristic future.

Expand.resume_1
    Load heuristic result.
    Generate a bounded group of moves.
    Yield if moves remain.

Expand.resume_2
    Finish child creation.
    Send reply.
    Complete.
```

Every resume point becomes a run class.

---

# 23. First collective

The only Phase-1 collective is:

```text
BATCH_EVALUATE
```

It accepts a frozen array of independent inputs and produces a frozen array of results.

```cpp
struct BatchEvaluateRequest {
    Ref64 implementation;
    CapRef inputs;
    CapRef outputs;

    uint32_t element_count;
    uint32_t element_stride;

    Ref64 completion_future;
};
```

It exists to compare:

* actor-by-actor lane execution.
* explicitly batched execution.
* and runtime-generated batching.

This reveals whether automatic physical shaping approaches the efficiency of manually expressed bulk work.

---

# 24. Prototype implementation layout

```text
soma/
    abi/
        refs
        capabilities
        objects
        processes
        continuations
        messages
        futures
        contracts
        traces

    kernel/
        domains
        ownership
        admission
        epochs
        commit
        failure

    executives/
        cpu_scalar
        gpu_lane

    scheduler/
        runnable_bins
        placement
        cohorting
        statistics

    compiler/
        state_machine_lowering
        frame_layout
        run_class_generation

    replay/
        trace_reader
        deterministic_interpreter

    experiments/
        dynamic_search
        synthetic_divergence
        latency_under_load
```

---

# 25. Dynamic-search experiment

The first workload should have controllable divergence rather than only a realistic application whose behavior is difficult to isolate.

Use two related workloads.

## 25.1 Synthetic branching search

Each state:

1. Executes one of several continuation paths.
2. Generates a configurable number of children.
3. Performs configurable amounts of arithmetic and memory access.
4. Optionally waits on an asynchronous heuristic.
5. May terminate, yield, or change priority.

Control variables:

```text
branching factor
continuation-class count
continuation duration
state size
memory intensity
future latency
priority distribution
arrival rate
process count
```

This maps the regime in which cohorting helps or fails.

## 25.2 Dynamic constraint search

This theoretical workload explores a finite state space defined only by an
initial state, a data-dependent transition relation, a goal predicate, and an
optional asynchronous scoring function. Each state becomes a process or a
process-owned work item. No application domain is assumed.

The workload includes:

```text
dynamic state generation
global state deduplication
variable legal-move counts
heuristic evaluation
priority queues
cancellation
solution propagation
multiple simultaneous searches
```

This determines whether the mechanism remains useful outside the regular
synthetic branching benchmark without tying the result to one puzzle or domain.

---

# 26. Required baselines

The experiment must compare:

| Baseline                  | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| CPU actor runtime         | Measures whether GPU participation is worthwhile |
| CPU-directed GPU dispatch | Represents conventional orchestration            |
| Bulk frontier kernel      | Strong manually batched implementation           |
| Persistent GPU FIFO       | Isolates launch elimination                      |
| SOMA without cohorting    | Isolates process/runtime overhead                |
| SOMA with cohorting       | Tests the central mechanism                      |
| SOMA with CPU spill       | Tests heterogeneous placement                    |

The bulk frontier implementation is essential.

Without it, SOMA might appear successful only because the baselines are weak.

---

# 27. Measurements

Primary measurements:

```text
completed state transitions per second
time to solution
p50 and p99 continuation wake-up latency
useful active SIMD-lane ratio
cohort fill ratio
time spent waiting for cohort formation
scheduler and commit overhead
host CPU utilization
memory bytes per state transition
deadline miss rate
fairness across concurrent searches
```

Secondary measurements:

```text
continuations created per second
message throughput
future-resolution latency
partial-cohort frequency
CPU/GPU placement accuracy
trace volume
replay overhead
```

---

# 28. Go/no-go criteria

SOMA-P1 should be considered promising only when all of the following hold.

## 28.1 Cohorting creates real value

Against the persistent FIFO baseline, continuation cohorting must produce either:

```text
at least 25% greater throughput
```

or:

```text
at least 1.5× greater useful lane occupancy
with no more than 10% throughput regression
```

in a meaningful divergent regime.

## 28.2 Scheduler cost is bounded

Cohorting, placement, commit, and tracing combined must consume:

```text
less than 20% of total execution time
```

in the regime where SOMA claims an advantage.

The longer-term target is below 10%.

## 28.3 Homogeneous work remains competitive

On homogeneous workloads already well suited to bulk execution, SOMA must remain within:

```text
15% of the manually batched implementation
```

or route the work into its collective path and close the difference.

## 28.4 Host orchestration disappears

The CPU must not submit or schedule individual search operations.

Host-side orchestration should remain approximately constant as the number of logical processes grows.

## 28.5 Latency remains controllable

A runnable high-priority continuation should begin execution within:

```text
two normal epochs
```

unless prevented by a hard resource requirement.

## 28.6 CPU spill has a measurable regime

CPU execution of partial cohorts must improve low-load or deadline-sensitive latency without causing more than a small throughput loss under load.

If no such regime exists, engine-independent migration should be deprioritized.

---

# 29. Failure interpretations

The experiment should be allowed to falsify more than the implementation.

Possible outcomes include:

## Outcome A: Cohorting wins substantially

Proceed toward:

* richer process semantics.
* more execution shapes.
* compiler automation.
* and stronger isolation.

## Outcome B: Cohorting helps only for narrow synthetic workloads

Reframe SOMA as a specialized runtime for irregular task systems rather than a general operating system.

## Outcome C: Cohorting overhead dominates

Investigate:

* coarser continuations.
* fewer run classes.
* compiler-generated batch fusion.
* and direct collective construction.

The abstract process model may survive while dynamic cohorting is rejected.

## Outcome D: Manual bulk batching always wins

SOMA's scheduler should become a graph- and batch-construction system rather than a lane-level actor executive.

## Outcome E: CPU/GPU migration rarely helps

Retain engine-independent semantics but make placement mostly static and explicit.

SOMA must permit these results. Its purpose is not to protect its original thesis.

---

# 30. Implementation order

The shortest evidence-producing sequence is:

```text
1. Define fixed ABI references and tables.
2. Build the deterministic CPU continuation interpreter.
3. Implement processes, messages, futures, and runnable bins.
4. Implement the synthetic branching-search workload.
5. Build the persistent FIFO GPU baseline.
6. Add run-class bins and continuation cohorting.
7. Add the bulk frontier baseline.
8. Measure cohorting before adding CPU/GPU migration.
9. Add CPU spill for partial cohorts.
10. Run the generic dynamic constraint-search workload.
```

The CPU interpreter comes first because it provides:

* semantic ground truth.
* deterministic tests.
* trace validation.
* and a debugging oracle for GPU behavior.

---

# 31. Phase-1 success condition

SOMA-P1 succeeds when it demonstrates:

SOMA-P1 succeeds if resident processes wake through messages and futures,
suspend through bounded continuations, and form SIMD cohorts that outperform a
generic persistent worker on irregular workloads without fine-grained CPU
scheduling.

Before that point, it remains a strong but unproven abstract machine.
