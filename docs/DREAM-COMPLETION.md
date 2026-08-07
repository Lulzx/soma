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
validated frame evaluators now run through a no-round-trip Metal graph with
private double-buffered frames and emit exact object read/write journals; the
normal `Kernel` speculative path validates and canonically commits them with
CPU/I18 and width-1/32 I19 controls. Allocation, mailbox, future, channel,
supervision, and dynamic successor operation vocabularies remain to be moved
into that resident pipeline.

## G3 — Distributed ownership [in progress]

Completion requires at least two kernels to own canonical resources and route
authorized operations, waits, wakeups, messages, supervision, and recovery
without a single coordinator owning every queue/future. Exact retry, revocation,
partitions, declared loss, and committed-effect recovery must be executable.

Current evidence: signed delegation, remote evaluator and full-journal
transports, node ownership/loss, distributed placement equivalence, and
authoritative remotely owned futures and bounded channels exist. Deterministic
epoch-boundary bridges park and wake local continuations without shadow resource
state. Remote mailboxes, supervision, process execution/recovery, and a
multi-kernel application remain integration gates.

## G4 — Bounded programming surface [in progress]

Completion requires a useful typed source surface that preserves validation-time
step/frame bounds and identical backend semantics. Integer fields, gathers,
auxiliary arrays, locals, structured loops, divergent breaks, bounded
compile-time function calls, and typed canonical `f32` add/multiply exist across
reference, native, and Metal. Float subtraction/division/comparison/select and
typed float locals remain open.

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
exactly with the independent host world. Discovery and self-tuning provide
additional semantic equivalence controls. A qualifying end-to-end cohorting
speedup is not yet established.

## G6 — Release discipline [in progress]

Default/native tests and Clippy run on Linux CI; all targets compile on macOS
with Metal. A release requires the full suite and real-device equivalence tests,
no undocumented stubs, frozen raw measurements, and an updated audit of every
gate above.
