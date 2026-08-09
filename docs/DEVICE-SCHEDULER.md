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

This is not yet the complete persistent executive. The resident path currently executes the bounded search transition. That
transition now emits one pointer-free trace record per lane directly into its
canonical `(epoch, lane, lane_sequence)` slot; the trace stays resident until
the complete command graph finishes and is compared field-for-field with an
independent oracle. Child publication is likewise derived from canonical lane
position rather than an atomic completion-order append.

Lane access journals, operation payloads, conflict validation, and canonical
replay share device-ready ABIs. The resident frontier now also accepts a
validated `EvaluatorProgram` keyed by user run class (1024+), double-buffers
lane-private frames through an arbitrary number of evaluator dispatches in one
command buffer, and emits `DeviceLaneAccess` plus `DeviceLaneOperation` records
at graph completion. The first end-to-end operation subset is canonical object
write: the host receives the final private frame bodies together with ordered
read/write accesses and `OP_READ_OBJECT`/`OP_WRITE_OBJECT` records per lane, sufficient for snapshot
validation and canonical host commit without observing an intermediate frame.
Gather/aux bodies are deliberately rejected on this private-frame path.

The CPU oracle interprets the same validated body and constructs the same
commit ABI. `MetalResidentSearch` also implements `DeviceEpochBackend`: the
normal speculative `Kernel` epoch selects installed user continuations, packs
their private frame objects, executes the resident graph, checks the graph's
emitted object-write access/operation against the independently constructed
lane journal, and only then enters the existing disposable-validation plus
canonical replay gate. Final frame bytes are published through the ordinary
snapshot payload commit; a protocol mismatch falls back without changing the
kernel.

Tests compare CPU/Metal frames, operations, accesses, and every lane-local
trace event, then vary cohort width 1 versus 32 for an explicit I19
placement-neutrality check. A second end-to-end test runs real `Kernel` epochs
through this backend and checks canonical frame publication plus I18 trace
conformance and byte-identical I19 kernel/device traces. The older branching
search remains as the dynamic frontier/placement stress case; expanding
compiled handlers to allocation, mailbox, await, and child-publication
operations is the remaining vocabulary work, not installation of a hardcoded
search transition.


### Resident future resolution

The next exact effect subset is future resolution. A user frame handler may
bind two validated u64 fields as the future and value references. Both the CPU
continuation path and resident Metal path evaluate the private frame first,
then emit the exact `OP_RESOLVE_FUTURE` record plus future-write/value-read
accesses. Successful independent lanes pass disposable validation and resolve
through canonical replay; two lanes targeting one future are rejected by the
ordinary journal conflict rule and the whole epoch falls back before any
resident result commits. Tests cover real-Kernel CPU/Metal I18, cohort-width
I19, exact future values and frames, committed-device statistics, and the
conflict/fallback path.


### Standalone dynamic multi-handler graph

`run_dynamic_frame_graph` is now a standalone no-intermediate-read smoke for two
installed general evaluator handlers. A deterministic device pack pass compacts
only active lanes of each run class, binds the resulting device count directly
to the evaluator, and scatters by stable rank. Evaluator output drives
Yield/Complete and the next run class; unknown classes are rejected in the same
transition, including on the last bounded dispatch. Active/class state,
per-lane trace slots, invocation counters, and quiescence stay resident for one
command-buffer submission and one final host read. Width 1 and 32 select real
Metal threadgroup widths, while counters prove inactive and wrong-class lanes
do not logically invoke a handler. The initial stable pack is deliberately
O(N^2); replacing it with a prefix scan is performance work, not a semantic
requirement.

This graph is not yet the normal multi-step `Kernel` executive. The ordinary
`run_epoch_with_device_backend` path still invokes one frame step per epoch and
per host submission. Therefore these results establish dynamic device frontier,
rebin, quiescence, and trace correspondence only for the standalone graph; they
do not claim general resident Kernel completion until final frames/status and
journals are wired through one canonical Kernel commit.


`examples/resident_dynamic_bench.rs` is an exploratory release harness. Its
original O(N^2)-pack run was negative at every sampled scale. An audit found
that several CPU-only repeat samples had been mislabeled as genuine controls;
those labels are removed rather than used for any G5 conclusion.

#### Atomic compaction and exploratory scaling

Because installed resident handlers reject gather/aux and are pure element-wise,
within-class physical order is not semantic. Packing now resets a device count
and uses `atomic_fetch_add`; the rank is carried only to scatter that element
back to its original lane. Trace slots remain keyed by original lane. Repeated
width-1 and width-32 tests prove different atomic packing orders preserve exact
frames, traces, and I19. If cross-lane reads are ever admitted, a stable
partition must return.

The benchmark labels were corrected after audit. `cpu_repeat` and
`cpu_one_program` remain explicitly exploratory placeholders, not genuine
sorted-bulk or device one-class controls. `resident_level_sync` is now a real
16-submission Metal control, and `resident_low_arrival` is eight real resident
batch submissions; dynamic initial classes are loaded from each input frame so
multi-submit execution preserves device-produced rebin state.

| lanes | CPU | CPU repeat | resident w1 | resident w32 | CPU one-program | resident level-sync | resident low-arrival |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 382 | 368 | 2689 | 847 | 311 | 4033 | 5221 |
| 4096 | 1334 | 1329 | 1678 | 1494 | 1305 | 5496 | 6017 |
| 16384 | 5305 | 5418 | 4658 | 4412 | 5091 | 9669 | 13655 |

All rows remained exact with one resident submission. The 16K parity/crossover
is a useful regime lead, but genuine CPU class buckets, a generic-device persistent worker, resident
one-class nulls, and repeated raw/median samples are still required.


`run_dynamic_ungrouped_frame_graph` supplies the same-device persistent-worker
baseline: it atomically compacts all active lanes without class grouping and
runs one generic installed handler. The correspondence test uses a competent
generic body with two class-specific counted loops and `break_if` at each loop
head, while grouped handlers contain only their relevant loop. Thus divergence
cost is measured without forcing a branchless generic handler to do work it
could skip. Generic and grouped Metal runs match exact final frames, trace, steps, and
quiescence. The harness adds a genuine CPU class-bucket oracle, device one-class,
eight-submit level-sync, low-arrival, alternating samples, and submission counts.
However, the first timing input put every lane in A and switched all lanes in
lockstep, so it never exercised mixed-class SIMD divergence. Its reversing 16K
ratios (1.061, 0.890, 1.233, 0.768) are only compaction/one-class noise controls.

The corrected `resident_dynamic_stress` input alternates classes, interleaves
four depths, and asserts both classes at every epoch. Across two independent
processes and six recreated-backend AB/BA batches, all 66 exact primary samples
exceed 20 ms. Call-position-stratified grouped/generic ratios have median 0.9050
and a seeded bootstrap 95% interval [0.8568, 0.9482]. That is a narrow,
reproducible grouped win over the competent generic device worker. It is not a
G5 closure. A dedicated position-balanced 65,536-lane control confirms sorted
16-submit execution is faster in all six batches: median grouped/sorted is
1.1883 with bootstrap 95% interval [1.1031, 1.2384]. A coalesced-encoder barrier
experiment stayed slower and was reverted. This also remains a standalone graph
without canonical `Kernel` publication. Historical, corrected, and fair-sorted
raw captures are in `measurements/RESIDENT-DYNAMIC-M4-PRO-2026-08-07.txt`,
`measurements/RESIDENT-DYNAMIC-STRESS-M4-PRO-2026-08-07.txt`, and
`measurements/RESIDENT-GROUPED-SORTED-FAIR-M4-PRO-2026-08-07.txt`. An irregular
eight-host-queue frozen-chunk control overlaps calls but reverses across its two
batches (1.0287, 0.9889); it is not persistent resident ingress. Its raw record
is `measurements/RESIDENT-IRREGULAR-HOST-QUEUE-M4-PRO-2026-08-07.txt`.

### Resident synchronization ABI oracle

`executives::resident_sync` defines the standalone CPU oracle and bounded,
pointer-free handler bytecode for the next functional slice. Installed state
machines emit nonblocking future observe, future await/resolve, and mailbox
send/receive effects; an executive-owned resource table applies them in
canonical lane order, parks and
wakes continuations across logical epochs, and records exact
operation/access/trace journals. `executives::metal_resident_sync` lowers that
contract to one real command buffer with device-owned frames, effects,
futures, mailboxes, FIFO waiter tickets, retry/rebin state, and quiescence. Its
two device phases emit every sorted handler result before canonical table
mutation; a single elected thread preserves deterministic application while
real width-1/32 threadgroup shapes provide I19 controls. CPU and M4 Metal match
exactly for future wake, successful mailbox delivery, authority and target
refusals, frame bytes, resources, journals, trace, and completion.

`kernel::resident_sync` now adds a transactional canonical commit bridge for a
strict bounded subset. The device result includes applied disposition, causal
wake, invocation, and per-epoch runnable/completion journals. After one Metal
command buffer, the bridge validates those records and replays each operation
through ordinary governed Kernel future/mailbox methods, `apply_step_result`,
Phase-G effect application, admission, trace drain, and full Phase-H accounting
before atomically swapping the clone. An independent governed-Kernel fixture
matches I18 trace order, resources, scheduler/effect/admission logs, counters,
continuation fields, and accounting; CPU and width-1/32 Metal bridge results are
exact. The same bounded ABI now carries `FutureObserve` as a nonblocking
future-read operation and fixed eight-byte little-endian object range reads and
in-place writes. Object operations use dense plan-local targets, versioned and
range-qualified capability snapshots, exact `OP_READ_OBJECT`/`OP_WRITE_OBJECT`
access and payload/result journals, and canonical replay through ordinary
Kernel object methods. CPU and actual Metal widths 1/32 agree exactly, while an
independent ordinary-Kernel reference agrees on final bytes and ordered
authority traces. Plans and results refuse atomically on stale versions,
expired/insufficient/range authority, invalid targets, growth, conflicts, and
malformed operation records. Object count (4,096), capability count (16,384),
each payload, and both actual and stride arena bytes (16 MiB) are checked before
CPU cloning or Metal allocation. A domain-separated SHA-256 plan fingerprint
covers the relevant public and private commit state.

A quiescent graph may publish a final local pending future await, full mailbox
send, or empty mailbox receive rather than treating every park as deadlock.
Final pending metadata is tied to the last matching blocking effect, its exact
outcome (`Registered`, `Full`, or `Empty`), canonical global ticket order,
target/value, disposition, and next class. Replay reconstructs the ordinary
Kernel future waiter, mailbox full-waiter, or mailbox receive-waiter queue on a
clone and compares every nonempty queue exactly before publication. Ordinary
future resolution, mailbox receive, and mailbox enqueue wake the imported
waiters in FIFO order. CPU and actual Metal widths 1/32 produce the same
canonical parked state; tampered ticket/target/value/outcome or disposition
refuses atomically.

Shared installed run classes now form exact canonical cohorts rather than being
refused, while the per-process mutable-admission exclusion remains. This closes
canonical commit only for local unsupervised, completed-or-locally-parked
future/mailbox/fixed-range-object programs with no pre-existing waiter queue on
a resource the resident bytecode can touch, stable pre-existing capabilities, host-backed object payloads,
`RunClassBins`, and `RunPartial`. Existing mailbox entries are snapshotted as
exact FIFO `(sender,payload)` pairs on both CPU and Metal, so an imported empty
receive waiter can be woken by ordinary enqueue and then finish through a new
resident plan without a host-side mailbox shadow. Competing mutable
continuations are grouped by actor in a bounded Metal prepass and use the exact
longest-waiting/identity winner; ordinary admission records, requeues, and
serial-deferral accounting are reconstructed transactionally. CPU and actual
Metal widths 1/32 agree across repeated contention, and winner tampering refuses
atomically. Bounded handlers also have wrapping little-endian immediate and
frame-to-frame word addition, frame-equality completion, frame-selected dynamic
next classes, and zero/nonzero frame-conditional next classes. Private
arithmetic state can therefore drive bounded multi-handler, multi-epoch
completion identically on CPU and actual Metal; zero or uninstalled dynamic
classes refuse transactionally. Disjoint ordinary future/mailbox waiter queues
now survive a resident run exactly on CPU and actual Metal, while device
participation by those parked continuations or touching their queued resource
still refuses.

Channel send and receive are now canonical resident effects through the same
bounded ABI. The device journal carries `OP_CHANNEL_SEND`/`OP_CHANNEL_RECEIVE`,
the Metal shader applies channel send/receive and wakes exactly one parked send
or receive waiter, and the Kernel primitives park a sender on a full channel or
a receiver on an empty one with exact FIFO waiter queues. A wake is an
epoch-boundary effect, so it is traced at the start of the next epoch's host
phase, and a parked continuation re-executes only its single pending effect with
`Yield` disposition rather than replaying its whole handler. `Sent` stays
non-value-bearing, so a sender can never observe its own successful send and
re-parks correctly on a still-full channel. Receiver-empty-park (receiver parks,
sender fills and wakes, receiver retries and receives), prefill delivery, and
sender-full-park (receiver drains, wakes the sender, which delivers across the
epoch boundary) flows match the ordinary Kernel exactly on CPU and actual-Metal
widths 1/32; tampered tickets, outcomes, and dispositions refuse atomically, as
does a pending effect outside the channel vocabulary. Supervision, allocation or
resizing, foreign resources, device capability creation, sub-full-range ordinary
Kernel object authorization, and general loop/gather or broader effect shapes
still refuse. The general G2 executive therefore remains open.
