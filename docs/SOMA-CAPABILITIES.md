# Design note: capability enforcement (I10)

**Status:** proposal, not implemented. Settles the question `docs/HANDOFF.md`
§5.1 asks — authority at reference resolution or at operation — and enumerates
the check surface so implementation is mechanical.

**Decision: check at operation, with the operation set closed by construction.**

The rest of this note argues that, specifies the mechanism, and records what it
forces elsewhere in the model. Two of those consequences are larger than the
capability work itself; see §7.

---

## 1. What exists today

`src/abi/capabilities.rs` already defines the vocabulary: nine rights (`READ`,
`WRITE`, `FREEZE`, `TRANSFER`, `SEND`, `RECEIVE`, `RESOLVE`, `AWAIT`,
`DESTROY`), a `CapabilityEntry` with an attenuation parent, an epoch bound, and
a version pin, and a `transferred_capability` slot on every message.

Nothing consults any of it. `tests/semantics.rs` demonstrates a process mutating
an object it has no relationship to, and the machine reporting itself legal.

Three defects in the existing ABI need fixing before enforcement can be written:

1. **`CapabilityEntry.object` is misnamed and mistyped in intent.** Rights
   include `SEND`/`RECEIVE` (mailboxes) and `RESOLVE`/`AWAIT` (futures), which
   are not objects. Rename to `target: Ref64` and let the kind discriminate.
2. **There is no holder.** The doc comment says a capability "resolves through
   the calling domain's capability table", which is the right design — but the
   implementation is a single global `GenTable<CapabilityEntry>`. One or the
   other has to give. §3 chooses the per-space design the comment describes.
3. **Kernel state is public.** `kernel.objects`, `kernel.object_payloads`,
   `kernel.mailboxes` and the rest are `pub` fields. Any code can mutate
   anything without calling an operation at all, so *the check surface is
   currently unbounded*. This is the single biggest obstacle and §4 addresses
   it first.

---

## 2. Why not check at reference resolution

Resolution-checking — making every `table.get(r)` take a holder and consult
authority — is superficially attractive because it is one choke point and
therefore impossible to forget. It fails here for three reasons.

**The kernel is its own biggest dereferencer.** The scheduler reads every
continuation to bin it; commit reads processes to retire them; the invariant
checker walks every table. None of these act on behalf of a program, so each
needs an ambient-authority escape hatch — and that hatch becomes the path most
of the codebase uses. The guarantee degrades to "checked when a program does
it", which is exactly what operation-checking provides, reached by a longer
road and with a permanently open bypass.

**It contradicts a design commitment already made.** `abi/refs.rs` states that
`Ref64` "is not itself an authority; it is merely a table reference." Resolution
checking partially fuses the two back together. Keeping references
authority-free is what lets them appear in traces, be compared across runs, and
be handed to the scheduler without leaking rights.

**It charges per lookup, not per decision.** `enqueue_message` resolves sender,
receiver, payload, and the sending continuation — four resolutions for one
logical operation that warrants one `SEND` check.

Operation-checking's real weakness is exhaustiveness: someone adds an operation
and forgets the check. That is a discipline problem, and this project's norm is
to convert discipline problems into mechanical ones. §4 does that.

---

## 3. The mechanism

### 3.1 Capability spaces

Each process owns a capability space: a generational table private to it.

```text
capability_spaces : process slot -> GenTable<CapabilityEntry>
```

A `Ref64` of kind `Capability` is interpreted **relative to the acting
process**. The same slot number in two spaces names different authority, so a
capability reference cannot be forged by guessing an index — possession within
your own space is the authority. This is what the existing ABI comment intends,
and it removes the need for a holder field.

The current global `capabilities: GenTable<CapabilityEntry>` is replaced.

### 3.2 Genesis

Creating an entity mints a full-rights capability in the creator's space.

```text
create_object(actor, kind, bytes) -> (Ref64 object, Ref64 capability)
```

Creation itself is unauthorised — anyone may create. That is deliberate, and it
is also a hole: see §8.

### 3.3 Derivation

`derive(actor, cap, rights, offset, length) -> Ref64` mints a child capability
in the same space with `parent_capability` set. **Derivation may only reduce**:
the child's rights must be a subset of the parent's, and its byte range must lie
within the parent's. This is machine-checkable as a state invariant (§6, I10a)
and is the property that makes delegation safe to reason about.

### 3.4 Transfer

Capabilities move between processes on messages, via the existing
`MessageDescriptor.transferred_capability`. Send removes or copies (per
`transfer_policy`) the entry from the sender's space; delivery installs it in
the receiver's. Transfer requires `TRANSFER` on the capability being moved and
`SEND` on the receiver.

Transfer is the *only* way a process obtains authority it did not create. That
single sentence is what makes the model analysable: authority flows along
message edges, so the reachable authority of a process is bounded by its
communication graph.

### 3.5 Revocation

Two mechanisms, both already present in the ABI:

- **`valid_until_epoch`** — expiry, checked at use. Free, no bookkeeping.
- **`parent_capability`** — revoking a capability revokes its whole derivation
  subtree. Implemented by deleting the entry; descendants are detected at use by
  walking to a dead parent, or eagerly by sweeping the tree. Prefer the eager
  sweep: lazy detection makes the cost of a use depend on derivation depth,
  which is a divergence source under cohorted execution (§5.4).

`object_version` gives a third, narrower mechanism: pinning a capability to a
version invalidates it when the object is frozen. Keep it, but do not rely on it
for revocation — §7.1 changes what freezing means.

---

## 4. Closing the operation set

Exhaustiveness is enforced structurally, not by review:

1. **Make kernel state private.** `processes`, `objects`, `capabilities`,
   `continuations`, `futures`, `object_payloads`, `mailboxes`, `future_waiters`
   become private to the `kernel` module. Rust's module privacy then makes any
   bypass a *compile error* outside `kernel`, which is the same "impossible to
   forget" property resolution-checking promised, obtained at the boundary that
   matters rather than at every lookup.

2. **Every operation takes an explicit actor.** No default, no ambient
   authority, no `Option<Ref64>`. If the kernel itself is acting, it passes an
   explicit system principal, so the privileged path is visible in every call
   site rather than implied by absence.

3. **One gate.** All checks route through

   ```rust
   fn authorize(&mut self, actor: Ref64, right: u32, target: Ref64)
       -> Result<(), RuntimeError>
   ```

   which resolves the capability in the actor's space, checks the right bit,
   the epoch bound, the target match, and the derivation chain — and emits a
   trace event either way (§5.2).

4. **A test-only raw accessor.** `tests/semantics.rs` deliberately injects
   illegal states and needs unmediated access. Expose that as an explicitly
   named `raw` module gated on `#[cfg(any(test, feature = "raw"))]`, so the
   bypass is a deliberate, greppable act rather than the default.

Step 1 is the bulk of the work and touches every experiment. Do it as its own
commit, with no behaviour change, before any capability logic lands.

---

## 5. Rules specific to SOMA

### 5.1 Authority is checked at use, never at suspend

A continuation's frame is durable and may hold a capability reference across
many epochs. Authority **must be re-checked when the continuation resumes**, not
captured when it suspended. Otherwise a revoked capability keeps working for
every continuation that was already parked, and revocation becomes advisory.

This is the rule most likely to be violated by an optimisation ("cache the
resolved capability in the frame"). It should appear as a comment at the frame
encode/decode boundary.

### 5.2 Authority decisions are observable

A denial is not an internal detail; it changes what the program does, so it
belongs in the trace and must be deterministic. Two runs of the same program
must deny identically. Add `AuthorityGranted` / `AuthorityDenied` event kinds.

Emitting on grant as well as denial roughly doubles trace volume for mutating
operations, and buys the strongest form of I10 (§6, I10c). Make it a build
feature if the volume hurts; do not omit it by default, because the project's
position is that a guarantee you cannot test is a guarantee you do not have.

### 5.3 The acting principal is the process, not the continuation

Continuations are execution slices, not security principals. A continuation acts
with its process's authority. This keeps authority stable across a process's
resume points, which is necessary for §5.1 to be meaningful.

### 5.4 Do not make authority checks data-dependent per lane

Parked but constraining. Under cohorted execution a cohort's lanes execute
uniformly; a check whose *cost* varies per lane (walking a derivation chain of
unknown depth) reintroduces divergence into the mechanism the project spent its
measurement budget defending. Prefer constant-time checks: eager revocation
sweeps (§3.5), a resolved rights mask on the entry, no chain walk at use.

Do not design for SIMD now. Do avoid precluding it.

---

## 6. What I10 becomes

Split into three clauses, two checkable as state predicates immediately:

**I10a. Attenuation [checked].** Every capability's rights are a subset of its
parent's, and its byte range lies within its parent's. Derivation never
amplifies. A pure state predicate over the capability spaces — implement this
first, it is cheap and catches the classic bug.

**I10b. Capability integrity [checked].** Every capability's target resolves,
its kind is one the rights apply to, and its parent (if any) is live. Same shape
as I1.

**I10c. No unauthorised effect [checked, trace-level].** Every state-changing
event in the trace is immediately preceded by an `AuthorityGranted` event naming
the same actor, right, and target. This is the clause that actually says the
machine is safe, and it is checkable only because §5.2 makes authority
observable.

Then flip the marker in `docs/SOMA-v0.2.md` §5 from **absent** to **checked**,
and rewrite
`tests/semantics.rs::capability_authority_is_unenforced_and_the_spec_says_so`
into its opposite. That test is designed to fail when this lands; update it,
don't delete it.

---

## 7. What this forces elsewhere

These are the parts worth arguing about before writing code.

### 7.1 Ownership and capabilities are currently two mechanisms for one thing

`ObjectDescriptor.unique_owner` and a `WRITE` capability both claim to govern
mutation. Two mechanisms for one property is how models acquire contradictions:
nothing says what happens when they disagree.

**Recommendation: eliminate `unique_owner` and define ownership in terms of
capabilities.** An object is *uniquely owned* exactly when one live `WRITE`
capability for it exists; it is *frozen* when no `WRITE` capability for it
exists and at least one `READ` capability does. Freezing is then not a flag to
be set but the act of destroying the write capability, which makes the model's
one-way rule structural rather than enforced by a check.

This collapses I9 into I10, deletes `ownership.rs`'s advisory transfer, and
makes "mutation of a frozen object requires allocating a new object" true by
construction. It is a real simplification, and it is the reason to settle
capabilities before ownership (spec §6.2) rather than after.

### 7.2 Failure containment needs an answer for authority

Spec §6.3 is open: when a process faults, nothing says what happens to its
futures or their waiters. Capabilities widen the question — a faulted process
holds a capability space, and capabilities it derived and transferred are live
elsewhere. Two coherent positions:

- **Authority survives the holder.** Transferred capabilities remain valid; the
  space is reclaimed but its exports are not. Simple; means a faulted process's
  grants outlive it.
- **Authority is revoked transitively.** Faulting revokes the process's
  capabilities and everything derived from them, which can cascade through
  unrelated processes and needs a story for what a holder observes.

The first is the smaller change and the easier one to reason about. Pick it
unless there is a concrete requirement for the second, and record the choice in
the spec rather than leaving it to the implementation.

---

## 8. What this does not give

State plainly in the spec, so the guarantee is not overread:

- **No resource safety.** Creation is unauthorised (§3.2), so any process can
  exhaust memory by creating objects. That needs domain quotas, which the model
  names and does not implement.
- **No information-flow control.** A process with `READ` may relay what it read
  to anyone it can `SEND` to. Capabilities bound *access*, not propagation.
- **No availability guarantee.** Authority does not prevent a process from
  filling a mailbox it may legitimately send to.
- **No protection from the implementation.** The `raw` accessor (§4.4) and the
  system principal are unchecked by construction.

---

## 9. Implementation order

Each step should land green.

1. **Privatise kernel state.** No behaviour change, no capability logic.
   Mechanical, touches every experiment, and is the step that makes the rest
   enforceable. Largest diff, lowest risk.
2. **Rename `CapabilityEntry.object` to `target`.** Trivial, do it while the
   ABI is already being touched.
3. **Capability spaces, genesis, derivation**, plus **I10a** and **I10b** as
   state invariants. At this point capabilities exist and attenuate correctly
   but nothing is enforced yet — a good place to stop and review.
4. **Thread `actor` through the operation surface** (§10) with the gate
   returning `Ok` unconditionally. Another no-behaviour-change commit; isolates
   the API churn from the semantic change.
5. **Turn the gate on**, one right at a time, starting with `WRITE`. Expect the
   `Expand` and search workloads to need capabilities plumbed through their
   frames — that will exercise §5.1 immediately.
6. **Trace events and I10c.**
7. **Unify ownership (§7.1)**, delete `unique_owner`, collapse I9.
8. **Settle failure (§7.2)** and update the spec.

---

## 10. The check surface

Every operation that must gate, with its right and target. This is the complete
list for the machine as it stands; anything added later belongs in it.

| Operation | Right | Target | Notes |
| --- | --- | --- | --- |
| `create_object` | — | — | Genesis; mints full rights to actor. Unauthorised by design (§8). |
| `object_bytes` / `read_u64_object` | `READ` | object | Also bounds-check against the capability's offset/length. |
| `object_bytes_mut` | `WRITE` | object | After §7.1, implies unique ownership. |
| `freeze` | `FREEZE` | object | After §7.1, becomes "destroy the write capability". |
| `transfer_unique` | `TRANSFER` | object | Deleted entirely by §7.1. |
| table `delete` | `DESTROY` | any | Not currently reachable from any operation; wire it when process teardown lands. |
| `create_process` | — | — | Genesis. Needs a domain quota it does not have (§8). |
| `create_continuation` | `WRITE` | process | Creating execution in a process is a mutation of it. |
| `create_future` | — | — | Genesis. |
| `await_future` | `AWAIT` | future | Check on resume too, not only on registration (§5.1). |
| `resolve_future` | `RESOLVE` | future | The right that matters most: resolution is single-assignment, so holding it is the ability to decide a value nobody can overwrite. |
| `enqueue_message` | `SEND` | receiver process | Plus `TRANSFER` on any attached capability. |
| `receive_message` | `RECEIVE` | own mailbox | Trivially held today; becomes meaningful with channels. |
| `ingest_message` | system | receiver | External input; system principal only. |
| `run_epoch` and scheduling | system | — | The machine acting as itself. |

### 10.1 Introspection, deliberately left ungated

The remaining public operations — `process_state`, `continuation_state`,
`future_value`, `mailbox_len`, the scheduler counters, `trace_snapshot` — read
state without changing it, and this proposal does not gate them. Two different
justifications, and only one of them is sound:

- **`future_value` is fine.** It returns a `Ref64`, and a reference is not an
  authority (§2). Handing out a reference to a value the caller cannot then use
  without a capability leaks nothing. This is the design commitment paying off.
- **`mailbox_len`, `process_state` and the counters are a real leak.** They
  expose another process's progress and queue depth to anyone who can name it.
  That is an information-flow question, not an access-control one, and §8 says
  the model does not address information flow. Left open knowingly rather than
  overlooked — if observation should require a right, `READ` on the target
  process is the natural spelling, and it is a small change once the gate
  exists.

`trace_snapshot` is implementation observability, not a program-visible
operation, and should move behind the `raw` boundary in step 1.

Derived from the public surface of `kernel/mod.rs`, `kernel/ownership.rs`, and
`kernel/epochs.rs` as of `aed270b`. Regenerate with
`grep -n "    pub fn " src/kernel/*.rs` and diff against this table when adding
operations; if a new row is needed, the gate needs a new call.
