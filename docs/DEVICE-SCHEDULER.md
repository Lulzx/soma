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

Placement is a second GPU dispatch. Each admitted candidate counts earlier
admitted candidates in its bin, producing a deterministic bin rank, cohort,
and lane. The admission and placement dispatches are encoded into one command
buffer, so no host read or round trip separates the decision from placement.

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

The current algorithm is intentionally simple and quadratic in candidate
count. It establishes the device ABI and semantic equivalence before replacing
the comparison scans with parallel sort/scan primitives. Scheduler-overhead
benchmarks will measure where that crossover is justified.

## Remaining integration

This is the device-resident planning phase, not yet the complete persistent
executive. The next integration carries admitted descriptors through device
continuation execution, lane-local event/effect journals, and canonical commit
without returning the plan to host code between those phases. Completion means
one epoch command graph with no host round trip inside it, followed by I19 and
trace comparison against the reference run.
