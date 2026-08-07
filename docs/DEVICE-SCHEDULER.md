# Device-resident scheduler

SOMA's device scheduler is being built against the same epoch contract as the
reference kernel, not as a separate approximate policy. The first hardware
slice is implemented in `scheduler::device` and
`executives::metal_scheduler`.

## Implemented device phase

For a complete epoch candidate set, the Metal scheduler now performs:

1. deterministic I13 admission for mutable process state;
2. stable placement into run-class bins;
3. cohort and lane assignment at the configured width; and
4. explicit run, defer, or CPU-spill disposition for every partial-cohort
   policy.

Admission is a set decision. Each candidate thread compares itself with every
mutable candidate for the same process and yields to the smallest
`(waiting_since, continuation identity)` key. It therefore implements the same
longest-waiting, identity-tiebroken rule as `scheduler::admission::admit`, with
no first-arriving atomic winner.

That comparison kernel remains the low-fixed-cost path for fewer than 128
mutable claims. At 128 or more, indices are instead sorted by full process
identity, mutable-before-read-only access, waiting age, and continuation
identity; the first mutable claim in each process group wins. The threshold
changes only the algorithm, not the decision. In particular the device ABI now
carries the complete process `Ref64`, rather than its slot/partition key, so a
generation can never alias another claim.

Placement no longer performs a quadratic all-candidate scan. Admitted candidate
indices are initialized in resident storage, sorted by `(bin, input order)` by
a deterministic device-side bitonic network, and assigned bin rank through
binary-search bounds in the sorted index. Results are written back in original
candidate order. Every sorting stage and placement remain in one command
buffer, so no host read or round trip separates admission from placement.

The candidate and placement arrays use fixed-width, pointer-free structs whose
Rust sizes are compile-time asserted against the MSL layout. Their shared Metal
buffers grow geometrically and remain allocated across calls; a smaller later
epoch reuses the resident capacity. `tests/device_scheduler.rs` compares the
real GPU output field-for-field with an independent reference lowering for all
four partial policies, including a contested mutable claim.

## Concurrency and determinism

Candidates are evaluated concurrently by GPU threads. No thread claims a
process or appends to an order-sensitive queue. Instead, every output position
is a pure function of the complete candidate array. That avoids turning the
GPU's physical execution order into SOMA lane order and preserves I22 by
construction.

The same device now validates lane-local access journals. Each access is a
fixed-width `(lane, resource namespace, Ref64, read/write, ordinal)` record.
One GPU thread per canonical lane compares its journal with the complete epoch
and reports whether it conflicts and the smallest conflicting lane. A conflict
requires equal namespace and identity, different lanes, and at least one
write; read/read pairs, duplicate records within a lane, and identical Ref64
bits in different resource namespaces do not conflict. The decision is thus a
set function and cannot depend on physical completion order.

The access and result buffers grow geometrically and remain resident across
epochs. `tests/device_scheduler.rs` compares the Metal result to an independent
CPU oracle, reverses journal input order, exercises read/read and namespace
negative controls, and verifies buffer reuse.

This is wired into actual epochs through
`Kernel::run_epoch_with_lane_validator`. The speculative workers execute from
the same pre-epoch snapshot and emit the real `LaneView` journals; the journals
are sorted into the fixed device ABI; Metal decides whether the epoch can
commit; and any conflict or device error sends the entire epoch through the
reference path before an operation or payload escapes. Disjoint dynamic-search
lanes commit with a reference-equivalent trace, while two writers of one future
exercise the fallback. A debug build also compares the device decision with the
independent oracle at that boundary. This moves the conflict gate needed before
canonical commit onto the device. It does not yet move operation replay or
handler execution there.

## Device operation journal and canonical replay

The complete `LaneView` call surface now has one pointer-free output ABI.
`DeviceLaneOperation` is a fixed 72-byte record containing canonical lane and
ordinal, opcode, flags, actor/target/value references, result code, and an
offset into one byte arena. Continuation frames, created object bytes, and
received-message evidence use that arena; no record contains a Rust pointer or
implementation enum.

All eleven operation shapes are covered: governed future/object reads, process,
continuation, future, and object creation, frame/object writes, message enqueue
and receive, and future resolve and await. Runtime and ABI errors have stable
codes, while complex results retain enough evidence for replay to reject a
different answer.

This is not a dormant format. Snapshot handlers lower into it before conflict
validation, and accepted epochs canonically replay these records against the
real kernel. The previous replay over the internal Rust `LaneOperation` enum is
gone. `SpeculationStats::device_operation_kinds` proves the independent expand
workload lowered every opcode; allocation and mailbox conflict controls prove
rejected journals still leave no state behind. This establishes the exact ABI
a Metal handler must emit and removes host data-model design from the remaining
device lowering.

Real Metal also consumes the complete operation records from those epochs. One
thread per lane slice verifies contiguous ordinals, the closed eleven-opcode
set, one canonical lane identity, and every offset/length against that lane's
arena size. The operation, slice, and result buffers remain resident and grow
geometrically. Invalid opcode and out-of-bounds arena controls fail on hardware;
the actual speculative epoch path runs this gate before Metal's access-conflict
decision and canonical host replay.

Read-only admission takes the constant path. Small mutable sets retain the
bounded quadratic comparison path; large mutable sets and placement use
`O(n log² n)` sorting plus `O(n log n)` bound lookup. Irregular randomized
epochs from 1 through 127 candidates and a separate 257-candidate mixed mutable
epoch—including non-power-of-two sizes, contested claims, sparse bins, and all
four partial policies—agree field-for-field with the oracle.

## Remaining integration

`MetalResidentSearch` is the first end-to-end resident execution path. It keeps
double-buffered continuation descriptors, a run-class-ordered execution
frontier, counters, completion accounting, and result digests in Metal buffers
while a single command graph runs every level of a dynamic branching search.
Reset, stable run-class placement, cohort accounting, concurrent execution,
child publication, and frontier swap are device phases; the host reads state
only after the graph completes. A 7-root, three-way, depth-four test executes
all 847 nodes over five epochs and agrees exactly with the independent CPU
transition on node count, epoch count, wrapping checksum sum, XOR digest,
cohorts, lane slots, useful lanes, and overflow status.

The controls distinguish scheduling from mere execution. Four run classes
produce more physical cohorts than the one-class control at width 32, while
width one collapses cohort count and lane slots exactly to node count. Both
controls still agree field-for-field with the reference accounting.

This is not yet the complete persistent executive. The resident path currently
executes the bounded search transition. Lane access journals, operation
payloads, conflict validation, and canonical replay now share device-ready
ABIs; handler evaluation and lane-local trace emission still need Metal
lowering. Completion means the general lane program runs through that same
no-round-trip graph, followed by I19 and trace comparison against the reference
kernel.
