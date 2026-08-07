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

This is a non-vacuous semantic placement test, not yet stateful message
transport: one coordinator kernel still owns the channel and supervision
queues. Those operations must next use authenticated remote journals before the
full distributed exit criterion is satisfied.

## Completion evidence

The backend is complete only when the two-node streaming graph and supervision
tree are I18-equivalent to their single-node runs, every process is remote from
its supervisor in the non-vacuity control, revocation is observed at remote
use, duplicate requests apply once, and killing a node mid-epoch produces the
defined discard-and-notify result.
