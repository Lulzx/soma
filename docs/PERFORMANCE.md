# Performance: what was measured, and what it cost

This document covers nine commits, `cae41c5..c3621a0`. Read §1 before using any
number in it, and §7 before deciding what to do next.

Everything here is about the *implementation*. No semantics changed except one
new operation (`release_authority`, §5.2) and one new trace event kind
(`AuthorityReleased`). The invariants are untouched, and every result in §§2–5
is a wall-clock measurement on one machine, an Apple M4 Pro, reproducible
from the examples listed with each section.

---

## 1. Why the existing measurements did not answer this

`docs/HANDOFF.md` §4 reports cohorting ratios of 1.55×–3.27×, correctly caveated
as *structural bounds computed from how continuations group*. They are not
timings. Before this work the repository contained no wall-clock measurement at
all: `grep Instant` over `src`, `tests`, and `examples` returned nothing, and
the Metal backend had never been timed once.

That mattered more than it sounds. Two classes of defect are invisible without a
clock, and the code had both:

- **Cost that belongs to nothing.** `MetalBatchBackend::evaluate` created a
  command queue per call, which is 68µs at best against a total call cost of
  ~330µs.
- **Cost that grows with the age of the run.** Three separate hot paths scanned
  a structure that only grows, so a run doing *n* operations did O(n²) work
  while every correctness test passed, because nothing about the *result*
  changes, only how long it takes.

The second class is the reason for the measurement rule in §6: a benchmark that
starts from a fresh `Kernel::new()` is measuring the one regime where these
cannot be seen.

---

## 2. The measurement harness

`src/experiments/backend_bench.rs` is the instrument the rest of this depends
on. It provides:

- `synthetic_program(id, fields, alu_ops)`: evaluator bodies with a chosen
  amount of arithmetic. The example bodies in `compiler::examples` are all two
  to eight instructions, which is right for checking a lowering and useless for
  telling a memory-bound regime from a compute-bound one.
- `time_evaluate` (the backend alone), `time_published_path`
  (`execute_with_spill`, including the kernel's copies and publication), and
  `time_epoch` (a whole epoch, batched or one collective at a time).
- `PlacementModel::fit` fits `time(n) = fixed + n·per_element` over a size
  sweep, so "profitable above n elements" becomes a measured number rather than
  the constant `minimum_accelerator_batch` that callers currently invent.
- `measured_crossover` beside the fitted `crossover`, because **the two
  disagree**. Neither backend is linear over four decades, and on a noisy CPU
  sweep the least-squares fit put the crossing at *one element* while the table
  plainly showed Metal first winning at 8,192. When they disagree the measured
  one is the observation and the fit is an extrapolation.

```sh
cargo run --release --features metal --example backend_bench   # CPU vs Metal, 32 → 4M elements
cargo run --release --features metal --example metal_overhead  # where a Metal call's fixed cost goes
cargo run --release --example kernel_overhead                  # what a published cohort costs off-GPU
cargo run --release --example growth_sweep                     # cost against accumulated state, to 1M
cargo run --release --example memory_profile                   # bytes per unit of work
```

---

## 3. The Metal backend

### 3.1 Persistent queue and reused buffers (`9c67ec8`)

`evaluate` created a command queue *and* both buffers per call. Decomposing the
fixed cost showed the queue at 68µs–420µs and buffer allocation at 1.4µs, which
made allocation look like a rounding error. It is not one: `new_buffer` is
cheap at any size, but a fresh shared buffer is fresh pages and the fill that
follows faults every one of them. Allocate-and-fill against memcpy into a warm
buffer is 59µs vs 13µs at 1MB, and 2016µs vs 503µs at 32MB. **Reusing the
allocation is a per-byte saving of roughly three quarters**, which moves the
fitted per-element cost rather than the intercept.

On the eight-instruction body:

| | before | after |
|---|---|---|
| fitted fixed cost | 475µs | 178µs |
| fitted per-element | 2.12ns | 0.98ns |
| 4M elements | 9.32ms | 4.25ms |
| first size where Metal wins by 15% | 65,536 | 8,192 |

Threadgroup sizing is **not** a finding here, though it is the usual advice: 32
through 512 threads land within run-to-run noise at both 1k and 1M elements, and
the ranking inverts between runs. The generated kernel's byte-wise copy loop
makes it memory-bound, so occupancy is not the constraint.

### 3.2 Publishing where the GPU wrote it (`4bd5c18`)

On Apple silicon the CPU and GPU address the same physical memory, so the round
trip `execute_with_spill` performed was two full passes over a batch to change
its *type*. Both are gone:

- The input `to_vec` existed only because the kernel borrow has to end before
  `publish` takes it mutably again. The borrow now ends at its last use.
- The output needed objects backed by an allocation the core did not make.
  `kernel::payload` splits a payload into `Host` and `Foreign`, and
  `BatchBackend::evaluate_payload` defaults to the old copy so only Metal
  implements it. I20 still compares backends through `evaluate`: agreement is
  about bytes, not about where they live.

Publication overhead over `evaluate()` alone, against Metal:

| elements | before | input copy gone | both gone |
|---|---|---|---|
| 65,536 | 45.5% | 16.1% | 10.0% |
| 1,048,576 | 32.3% | 25.6% | 15.5% |

A published buffer is *given away*, so the backend allocates a fresh output for
a batch it will publish rather than handing over the one it reuses. Overwriting
it on the next batch would mutate a frozen object behind the kernel's back and
no invariant would catch it. Since nothing semantic can observe the difference,
`object_provenance` exists so one test can assert the GPU path publishes
`metal-shared` and the CPU path `host`, with identical bytes.

### 3.3 One command buffer per epoch (`2e19cac`, `9be0e62`)

Pricing the three submission shapes at the Metal API, for 64 cohorts of 8,192
elements: per-cohort wait 9,897µs, deferred wait 1,611µs, one command buffer
757µs. Asynchronous submission buys 6.1× and one command buffer buys 13×, so
**the cheaper change is the larger one**: batching needs only a backend entry
point, while async reaches into the collective-completion path.

`BatchBackend::evaluate_epoch` defaults to a loop. Metal encodes the whole epoch
into one command buffer, staging inputs into one reused buffer at aligned
offsets. `execute_epoch_with_spill` gathers, submits, and publishes in the order
the epoch offered. Gathering needs every input alive at once, which
`object_bytes` cannot do, because authorization records an effect and so its
result borrows the kernel mutably. `object_bytes_many` authorizes each in turn
and only then reborrows shared.

End to end through the kernel, 8,192-element cohorts:

| cohorts | one by one | as an epoch | speedup |
|---|---|---|---|
| 1 | 0.191ms | 0.174ms | 1.0× |
| 4 | 0.740ms | 0.300ms | 2.5× |
| 16 | 3.602ms | 0.610ms | 5.9× |
| 64 | 15.818ms | 4.504ms | 3.5× |

The gap from 13× is the finding: at 64 cohorts the epoch spent ~4.2ms of its
4.5ms *outside* the GPU. That is what §4 went after. (After §4, the 64-cohort
epoch is 2.247ms and the ratio 6.6×.)

**The epoch's trace is not identical** to running the same collectives singly:
it authorizes every input before publishing anything, so read effects group
where they used to interleave. That reordering *is* the batching. The test pins
what a reader downstream of publication sees, the order collectives complete in,
and the bytes.

---

## 4. Three accidentally-quadratic paths (`51ab67f`, `d482235`)

Found by re-timing a fixed operation as one structure grew underneath it
(`examples/growth_sweep`, levels 0 → 1M). None of this is a novel result and
the fixes are the obvious ones. They are documented because all three were live
in the path SOMA's thesis depends on, and the whole test suite passed through
them.

Cost of one more operation, against how much the kernel already holds. Each
ratio is against the same operation on an empty kernel.

| operation | before | after |
|---|---|---|
| Publishing a batch | 485µs at 16k (81×) | ~1.9µs at 1M, flat |
| Running an epoch | 6.59ms at 1M (3,953×) | 1.83µs at 1M (1.4×) |
| Cancelling a process | 1.96ms at 1M (682×) | 1.00µs at 1M, flat |

Publishing was fixed first, which is why its "before" column stops at sixteen
thousand: at a million it was not worth waiting for.

**Authorization.** `find_authorized_capability` answered "may this actor do X to
Y?" by walking the actor's whole capability space, and `revoke_capability_tree`
found a capability's children the same way. Freezing an object revokes write
authority over it, so that ran for every published batch.
`kernel::capability_space` indexes by target and by parent. Roots are left out
of the parent index: they all share a null parent, so indexing them would put
most of the space in one bucket and make the `retain` on delete linear again.

**Epochs.** `run_epoch → apply_step_result → contain_process_failure →
cancel_process_continuations`, which a process reaches by *finishing*, not only
by failing. It found the process's continuations by walking every continuation
ever created, then swept the future waiters, every mailbox, every channel and
every supervision queue. There is one mailbox per process. Continuations are now
indexed by process, and the sweeps are skipped when nothing was cancelled: the
common case, where every `retain` would have kept every element.

**Cancellation.** `Scheduler::remove` walked both epoch buffers of every bin,
once per continuation. Batching the removals into one pass made it *worse*
(1.96ms → 9.22ms): hashing a `Ref64` per queue entry costs more than the four
integer comparisons it replaced. The pass itself was the problem. Both buffers
are only appended to and then drained whole, so an entry's index is stable while
it is queued. Entries are now `Option<Ref64>` and each bin records where every
continuation sits, so removal clears one slot. The position map stores the
*swap sequence* an entry arrived at rather than which buffer holds it, because
the buffers trade places every epoch.

Guards live in `tests/publication_scaling.rs`. They are timing tests, which is
worth the discomfort: what they separate is roughly-constant from
linear-in-a-kernel-eighty-times-larger, two orders of magnitude apart, so the
bound is loose enough that a loaded machine will not trip it. All three were
verified by putting the old code back.

---

## 5. Memory (`0b96d53`, `c3621a0`)

Making each operation's *time* independent of accumulated state only meant the
memory limit arrived no slower. `examples/memory_profile` priced it at ~1.3KB
per published batch against 32 bytes of payload, and 400,000 processes run to
completion with all 400,000 still resident.

### 5.1 Finished processes

`reclaim_finished_processes` releases exactly what `allocate_process` allocated
(state object, mailbox, supervision queue, capability space) plus the process's
continuations and their frames. It leaves alone any process whose
supervisor has not taken the notice, that supervises something live, or that a
restart blueprint could bring back, and reports *which*, since "nothing was
reclaimed" and "nothing was reclaimable" are different answers.

400,000 processes each run to completion: 567MB with 400k processes and 800k
objects resident, against 167MB with the tables back to empty.

### 5.2 Anything nothing can name

A published batch outlives its producer, and its collective and completion
future outlive both. Rather than add reference counting to the ABI, this uses the
definition the machine already has: a capability *is* the ability to name
something. `unreachable()` marks from every capability target and every live
process, continuation, message and waiter list, follows what those name, and
reports what nothing arrived at. `reclaim_unreachable()` releases it.

It must be a closure and not a scan: a collective names its input and output, so
an object is not garbage merely because no capability names it. But if the
collective is itself unreachable, it must not keep them alive.

`release_authority(actor, target)` is the counterpart and the one new semantic
operation: nothing becomes unreachable while its owner still holds authority. It
drops one actor's authority, so a frozen array several processes read stays
readable by the others and goes when the last lets go.

400,000 published batches: 494MB with 400k objects and 1.2M capabilities,
against 2MB with one object and five capabilities. That figure is with the logs
drained each round. Left in, they are the only thing still growing, which is
what `kernel::retention` is for.

**Both passes are explicit and called from nowhere.** When a run may forget a
process, or forget a published batch, is a policy question about supervision and
inspection, and this code decides only what is safe, not when.

---

## 6. Rules this produced

Additions to `docs/HANDOFF.md` §7. Each was a real defect in this work, not a
hypothetical.

- **Re-time as the kernel ages, not once on a fresh one.** Every benchmark that
  starts from `Kernel::new()` measures the one regime where a scan over
  accumulated state is invisible. Three of them were live. The `growth_sweep`
  shape (fix an operation, grow one structure underneath it, re-time) is what to
  point at anything new.
- **An index narrows a search. It does not decide the answer.** `Ref64::key()`
  is partition and slot, *not* kind or generation, so a process and an object in
  the same slot share a bucket. The indexed lookup must re-check the whole
  reference. It did not, and `revoke_target_right` could revoke a capability over
  the wrong entity. `CapabilityIntegrity` caught it, but only once reclamation
  started deleting things, several commits after it was introduced.
- **Deleting an entity means purging the capabilities that name it.** Otherwise
  the machine is smaller and illegal.
- **`processes_created` is a population, not a total.**
  `DomainContractIntegrity` checks it against the live count, so reclaiming a
  process has to give one back.
- **Releasing authority is not exercising it.** `AuthorityEffect` is checked by
  `NoUnauthorizedEffect` for an adjacent grant. Letting go needs no permission
  beyond having held it. That is why `AuthorityReleased` exists.
- **A backend that publishes a buffer has given it away.** It must not reuse it.
- **Profile before the third guess.** Two of the four scans here were found by
  `sample` in seconds after I had already guessed wrong twice from the code.

---

## 7. What is still open

- **The thesis is still unmeasured.** Whether continuation cohorting beats a
  bulk schedule *on hardware* remains an analytical occupancy model over
  simulated ticks (`regime_map`, `baseline_report`). This work made the machinery
  underneath fast and honest enough to ask the question. It did not ask it.
- **Asynchronous submission (§3.3) was not done.** It is worth 6.1× on its own
  and is subsumed by epoch batching wherever an epoch has several cohorts. It
  earns its cost only where epochs are narrow.
- **The input-side copy remains.** Metal still copies inputs into its staging
  buffer, worth ~10% at 1M elements. Removing it needs a chained collective to
  bind a prior GPU output directly, which means `evaluate` taking a `&Payload`
  and downcasting.
- **Nothing calls either reclamation pass.** See §5.
- **`kernel_overhead` and `growth_sweep` have only been pointed at the batch and
  epoch paths.** Channels, supervision, and the admission log have never been
  measured against accumulated state.
