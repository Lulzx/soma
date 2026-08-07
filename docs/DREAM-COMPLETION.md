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
width-1/32 CPU equivalence. A transactional all-complete Kernel bridge now maps exact live local
references, checks strong plan fingerprints, step budgets and capability
horizons, validates explicit invocation/disposition/wake/epoch journals, and
replays future/mailbox operations through ordinary governed Kernel methods,
Phase G, admission, causal trace, and full Phase-H accounting before atomic
publication. An independent governed reference matches I18 and the complete
scheduler/effect/admission/accounting state, including width-1/32 Metal. This is
a canonical commit slice, but only for local unsupervised, unique-run-class,
all-complete programs with pre-existing stable authority and initially empty
waiter/mailbox state. Allocation, channels, supervision, device capability
creation, admission deferral, parked final state, broader handler shapes, and
the general canonical bridge remain to be integrated.

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
I18 controls. A bounded authenticated multiplexed protocol carries six future/channel/object
operations with exact grant rebinding, issuer-qualified actors, canonical
apply-once positions, late/collision rejection, and preserved transport error
classes. A deliberately narrower validated Kernel dispatch can emit exactly one
blocking future-await, channel-send, or channel-receive operation, park in
private node-qualified state, and retry exact retained work across two real
runtimes/TCP; it is special dispatch, not general `LaneView`, and caller-carried
outcomes remain unsigned/trusted. Owner-side process templates provide
content-bound create/restart receipts, terminal lifecycle observation, bounded
snapshot/WAL apply-once durability, and exact terminal recovery. Signed bounded
lifecycle requests now cross TCP between two real runtimes, but responses are
not mutually authenticated and live-state recovery remains explicitly refused.
General remote `LaneView`, signed/bound multiplexed batch responses, direct
canonical remote park, object/mixed/result-frame dispatch, durable live process
recovery, and a real multi-resource multi-kernel application remain gates.

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

The collective ant sensing path drives the real persistent colony and agrees
exactly with the independent host world, but its Metal full wall is slower than
the direct host control. The first grouped/generic resident timing input was
later found to switch every lane between classes in lockstep, so it was a
one-class non-vacuity failure rather than an irregular divergence experiment;
its reversing ratios are retained only as negative noise/compaction controls.
A corrected 49,152-lane stress capture keeps both classes live in every epoch
and gives a call-position-stratified grouped/generic median ratio of 0.9050 with
bootstrap 95% interval [0.8568, 0.9482] across six batches. This closes only the
narrow standalone competent-generic comparison. A dedicated fair AB/BA control
finds sorted 16-submit execution faster in all six batches (median
grouped/sorted 1.1883, bootstrap 95% interval [1.1031, 1.2384]); an attempted
coalesced-encoder optimization stayed slower and was reverted. An eight-thread,
eight-queue irregular host-release control has overlapping calls and exact
combined results, but its two batch ratios reverse (1.0287 and 0.9889); it is
explicitly frozen chunks, not live ingress into one persistent resident command
buffer. The graph also has no canonical Kernel commit. Discovery and
self-tuning provide additional controls. A qualifying end-to-end cohorting
speedup is not yet established.

## G6 — Release discipline [complete]

The clean committed revision `8f48bb6cdea2eed19470c652107fd0dc15b9dc65`
has a qualifying physical-device capture in
`measurements/RELEASE-AUDIT-20260807T090150Z-full.log` with an adjacent verified
SHA-256 file. The record identifies the 16-core Apple M4 Pro, macOS 26.6,
Rust/Cargo 1.92.0, exact commit and initially clean tree; it ends `result: PASS`
after the complete all-feature suite, warnings-forbidden Clippy, explicit Metal
backend/scheduler/binary32/resident-sync suites, and bounded release benchmarks.
The benchmark outputs remain negative/unstable evidence rather than a speedup
claim. `scripts/release-audit.sh` and [`RELEASE-AUDIT.md`](RELEASE-AUDIT.md)
preserve clean-tree refusal, immutable log names, checksum, quick, and dry-run
rules for subsequent release commits.
