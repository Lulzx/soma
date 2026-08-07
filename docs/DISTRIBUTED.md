# Distributed SOMA

Distributed execution preserves the single-node epoch contract. A node may
speculatively execute lanes, but its lane events and effects become visible
only through canonical epoch commit. Loss before commit discards that node's
entire uncommitted journal; loss after commit is already part of the run. A
coordinator never guesses that a process faulted merely because a transport
timed out.

## Identity and authority

`distributed::RemoteRef` pairs a 64-bit `NodeId` with the existing `Ref64`.
The `Ref64::partition` remains an allocator partition inside a node; it is not
overloaded as a cluster node identifier. Two nodes can therefore allocate the
same partition and slot without naming the same entity.

A remote reference is not authority. `RemoteGrant` is a fixed-width,
little-endian delegation containing issuer, audience, actor, remote target,
rights, object version, inclusive logical-epoch bounds, a unique nonce, and an
HMAC-SHA-256 signature. The signature proves provenance and prevents field
tampering. The issuer's live registry supplies the other half: an otherwise
valid signed grant is rejected after revocation. Authorization at use checks
protocol version, issuer, signature, registry membership, audience, exact
target, object version, rights, logical validity, and revocation.

`RemoteBatchBackend` and `RemoteBatchServer` carry evaluator requests over
framed TCP. Requests are content-addressed over shape, logical epoch, grant,
primary input, and auxiliary input. The worker authorizes first, then consults
its response ledger, so an exact retry applies once while a revoked grant cannot
retrieve a cached response. Wire responses distinguish success, authority
denial, malformed input, unsupported evaluator, and execution failure.

The client separately distinguishes connection refusal (`NodeUnavailable`),
loss after connection (`NodeLost`), and malformed responses (`ProtocolError`).
Integration tests compare remote output with the CPU definition, prove duplicate
application suppression, revoke before a cached retry, and show that an
unreachable node cannot be mistaken for evaluator bytes.

`BatchBackend::evaluate_epoch` uses one TCP session for every request offered
together and returns payloads in request order, preserving the backend's
all-or-error ordering contract without a connection handshake per cohort. A
separate negative control accepts a connection and drops it before replying;
the client reports `NodeLost`, distinct from connection refusal.

`RemoteJournalValidator` applies the same transport and authority rules to the
stateful pre-commit boundary. The coordinator serializes the actual read/write
sets emitted by `LaneView` into the fixed device/wire ABI and sends the complete
epoch to an authenticated worker. The same request carries every fixed-width
operation record and its byte arena: governed reads, allocation, object/frame
writes, messages, and future operations. The worker validates lane/ordinal
order, opcode range, and every arena bound before making the namespace-aware
conflict decision. The request identity covers all access, operation, result,
and payload bytes, and the service counts the operation records it durably
accepted into its in-memory response ledger.

Exact retry applies once; authorization is checked before the response ledger,
so revoking a grant denies a formerly cached request. The client preserves
distinct `Unavailable`, `NodeLost`, authority, protocol, and execution
outcomes. A malformed operation arena is an explicit negative control and is
rejected at the worker rather than reaching commit.

`RemoteFutureService` is the first resource whose canonical state is owned by
the node named in its `RemoteRef`, rather than replayed into coordinator state.
It serves authorized `AWAIT` observations and single-assignment `RESOLVE`
operations over TCP. Resolve requests are content-addressed and apply once;
revocation is checked before retry lookup, and a different second value is
rejected. Polls deliberately bypass the response ledger so a previously
observed `Pending` value cannot hide a later resolution. The client keeps
unavailable, lost, protocol, authority, and already-resolved outcomes distinct.
`RemoteFutureBridge` now supplies that scheduler edge. It observes the owner at
most once per logical epoch boundary, parks only a local continuation identity,
and uses the normal runnable-bin wake effect when the authoritative state is
resolved. No future descriptor or value is copied into the local kernel. The
kernel keeps the opaque remote dependency in a private map (never in an ABI
reference field), removes stale runnable entries before parking, and traces one
`ContinuationWaiting`/`ContinuationReady` pair with the remote entity as cause.
Duplicate registration, wake, and same-boundary polling are idempotent; tests
check I1/I7 immediately around the transition.

`RemoteChannelService` similarly owns a bounded FIFO and closed bit on the node
named by `RemoteRef`. SEND, RECEIVE, and DESTROY grants are operation-specific;
per-actor sequences, content-addressed mutation replay, authorization before
replay, and Full/Empty/Closed outcomes preserve FIFO, back-pressure, drain, and
close semantics. `RemoteChannelBridge` probes readiness once per epoch boundary
and parks/wakes local send or receive continuations through the same private-map
hooks without creating a shadow channel descriptor. TCP and kernel integration
controls cover apply-once retry, revocation, competing operations, node
unavailable/lost/protocol distinctions, and invariant legality before and after
wake.

This validator is wired directly into `Kernel::run_epoch_with_lane_validator`.
A clean remote decision permits canonical coordinator commit. A conflict or
transport error discards every speculative snapshot and takes the reference
path before an operation or payload escapes. Tests exercise both a disjoint
epoch that commits with an I18-equivalent trace and two future writers that
fall back. The worker now receives the complete speculative operation payload,
but operation replay and authoritative queue/future state remain on the
coordinator. Node loss before that replay therefore discards the journal; the
remote worker cannot make an operation visible by accepting its bytes.

## Failure semantics

Transport failure and process failure are different outcomes.

- A connection error before acknowledgement is `NodeUnavailable`; the request
  may be retried with the same request identity.
- A connection loss after acknowledgement is resolved through the node's
  request ledger. Pure evaluator requests are content-addressed and replayable.
  Stateful lane requests are not externally visible until epoch commit.
- A partition is not automatically a fault. The coordinator may wait or apply
  an explicit `declare_lost` policy at an epoch boundary.
- Declared node loss discards uncommitted journals and emits a node-loss notice
  to every remote process's supervisor. It does not claim those processes
  executed a `Fault` instruction.
- A committed remote effect is never rolled back. Recovery resumes from the
  last committed epoch and request identities suppress duplicate application.

The transport must make these states distinct on the wire; an empty payload or
generic execution error cannot stand in for any of them.

This boundary is executable in the kernel now. Every process descriptor records
its owning node separately from allocator partition. Only the system principal
may declare loss; declaration is monotonic and idempotent, rejects subsequent
placement on that node, contains every live process owned there, emits
`ProcessLost` rather than `ProcessFailed`, and delivers the distinct
`ExitReason::NodeLost` to supervisors. A partition control leaves remote
processes untouched until declaration. Existing supervision and semantic
invariant suites pass unchanged.

## Placement equivalence

The streaming graph and supervision tree now accept explicit node placement.
`tests/distributed_equivalence.rs` runs each locally and across two nodes, then
uses I18 rather than raw result equality alone. The streaming control places
both channel peers away from the coordinator and covers normal completion plus
source failure. The supervision control alternates nodes at every level, so all
four supervisor/child edges are remote, and covers notify, escalation, and
restart. Every run is legal, has the same outcome, and is I18-equivalent.

This is a non-vacuous semantic placement test. Complete stateful journals now
cross an authenticated remote boundary and gate real epoch commit, but one
coordinator kernel still owns and replays the channel, future, and supervision
operations. Moving authoritative resource ownership to several kernels would
be a stronger distributed machine; it is not required for this backend's
speculative-execute/central-canonical-commit model and is not claimed here.

## Completion evidence

The backend evidence is: the two-node streaming graph and supervision tree are
I18-equivalent to their single-node runs; every process is remote from its
supervisor in the non-vacuity control; evaluator and stateful journal requests
are authenticated and duplicate-safe; revocation is observed before cached
responses; full operation arenas are validated remotely; refusal and
accepted-then-lost are distinct; and explicit node loss produces the defined
discard-and-notify result rather than a program fault.
