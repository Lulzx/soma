# SOMA completion gates

This file is the refusal to call a research slice a finished GPU OS. A gate is
complete only when the executable evidence named here exists and passes on the
hardware it claims to cover.

## G1 — Machine semantics [complete]

The numbered invariants, deterministic reference interpreter, capability and
ownership rules, bounded continuations, messages, futures, channels,
collectives, supervision, cancellation, domains, contracts, replay, I18/I19,
and positive/negative invariant tests are implemented.

## G2 — General device-resident executive [in progress]

Completion requires one no-round-trip graph to admit, group, execute, journal,
validate, commit, trace, publish successors, and reach quiescence for arbitrary
installed bounded continuation handlers—not only the resident search
transition. Every `LaneView` operation shape and failure outcome must compare
against the reference kernel under I18/I19. Intermediate epoch state must not be
read or scheduled by the host.

Current evidence: resident admission/frontiers/search, canonical lane trace
positions, grouped conflict validation, and canonical replay exist. Installed
validated frame evaluators run through a no-round-trip Metal graph with private
double-buffered frames and exact object/future journals. A separate real
one-command-buffer synchronization backend now executes bounded pointer-free
handlers with device-owned futures and mailboxes, canonical two-phase effects,
park/wake/retry/rebin/quiescence, FIFO waiters, exact journals/traces, and
width-1/32 CPU equivalence. These standalone graphs are not yet the normal
`Kernel` Phase-G path. Allocation, channel, supervision, full dynamic successor
coverage, and the general canonical commit bridge remain to be integrated.

## G3 — Distributed ownership [in progress]

Completion requires at least two kernels to own canonical resources and route
authorized operations, waits, wakeups, messages, supervision, and recovery
without a single coordinator owning every queue/future. Exact retry, revocation,
partitions, declared loss, and committed-effect recovery must be executable.

Current evidence: signed delegation, remote evaluator and full-journal
transports, node ownership/loss, distributed placement equivalence, and
authoritative remotely owned futures, bounded channels, growable objects, and
terminal supervision notices exist. Deterministic node-qualified boundary
bridges park and wake local continuations without shadow resource state or
cross-node entity collisions. Narrow future/channel `RemoteNodeRuntime` smokes
run continuations on two real kernel owner threads, and signed apply-once remote
SEND now commits into the real owner Kernel inbox with an immutable
node-qualified envelope, wake, priority, back-pressure, revocation, loss, and
I18 controls. Generic remote `LaneView` effects, remote process
creation/restart/recovery, durable services, and a multi-resource application
remain integration gates.

## G4 — Bounded programming surface [complete]

Completion requires a useful typed source surface that preserves validation-time
step/frame bounds and identical backend semantics. Integer fields, gathers,
auxiliary arrays, locals, structured loops, divergent breaks, bounded
compile-time function calls, and the complete bounded typed canonical `f32`
surface exists across reference, native, and Metal: arithmetic, ordered
comparison, selection, and explicitly typed float locals.

## G5 — End-to-end thesis evidence [in progress]

The Phase-1 success condition is met only by measured irregular applications,
not structural occupancy alone. Required controls:

1. identical application state under reference, persistent-worker, cohorted,
   native, and Metal placements;
2. real wall-clock comparison against persistent-worker and sorted bulk
   baselines, including scheduler and migration overhead;
3. a workload/regime where cohorting wins materially and reproducibly;
4. width-one, one-class, level-synchronous, and low-arrival null controls;
5. bounded priority latency and approximately constant host orchestration for a
   resident run;
6. raw samples, hardware/toolchain identity, and repeated confidence summaries.

The collective ant sensing path now drives the real persistent colony and agrees
exactly with the independent host world, but its Metal full wall is slower than
the direct host control. The standalone dynamic graph has exact same-device
generic-worker, grouped, one-class, level-sync, low-arrival, and CPU class-bucket
controls; independent release runs reverse the grouped/generic ordering, so its
apparent 16K crossover is not reproducible. Discovery and self-tuning provide
additional equivalence controls. A qualifying end-to-end cohorting speedup is
not yet established.

## G6 — Release discipline [in progress]

Default/native tests and Clippy run on Linux CI; all targets compile on macOS
with Metal. A release requires the full suite and real-device equivalence tests,
no undocumented stubs, frozen raw measurements, and an updated audit of every
gate above. Capture the auditable device/toolchain record with
`scripts/release-audit.sh` on the qualifying Mac; see
[`RELEASE-AUDIT.md`](RELEASE-AUDIT.md) for clean-tree, checksum, quick, and
dry-run rules.
