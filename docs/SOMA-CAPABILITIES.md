# Design note: capability enforcement (I10)

**Status:** partially implemented. Actor-relative spaces, genesis, derivation,
I10a, and I10b exist. `WRITE` authority is enforced; other rights are not.

**Decision: check at operation, with the operation set closed by construction.**

This note specifies the mechanism and its effect on ownership and failure
semantics. See §7.

---

## 1. What exists today

`src/abi/capabilities.rs` defines nine rights (`READ`,
`WRITE`, `FREEZE`, `TRANSFER`, `SEND`, `RECEIVE`, `RESOLVE`, `AWAIT`,
`DESTROY`), a `CapabilityEntry` with an attenuation parent, an epoch bound, and
a version pin, and a `transferred_capability` slot on every message.

Actor-relative capability spaces, genesis, derivation, and the I10a/I10b state
checks are implemented. Operations pass through one authorization gate. `WRITE`
consults the actor's space, epoch bound, version pin, and live parent at use;
other rights remain permissive.

Three prerequisites for enforcement are complete:

1. **`CapabilityEntry.target` names any governed entity.** Rights
   include `SEND`/`RECEIVE` (mailboxes) and `RESOLVE`/`AWAIT` (futures), which
   are not objects. The `target: Ref64` field lets the kind discriminate.
2. **Capabilities have holders.** The implementation uses the per-process
   spaces described in §3.
3. **Kernel state is private.** External code reaches unchecked state only
   through the unsafe, doc-hidden `raw` test surface described in §4.

---

## 2. Why not check at reference resolution

Resolution-checking would make every `table.get(r)` take a holder and consult
authority. That gives one check point but fails here for three reasons.

**The kernel is its own biggest dereferencer.** The scheduler reads every
continuation to bin it. Commit reads processes to retire them. The invariant
checker walks every table. None acts on behalf of a program, so resolution
checking needs a privileged bypass that most kernel code would use. Operations
already mark the boundary between program effects and kernel bookkeeping.

**It contradicts a design commitment already made.** `abi/refs.rs` states that
`Ref64` is a table reference, not authority. Resolution
checking partially fuses the two back together. Keeping references
authority-free is what lets them appear in traces, be compared across runs, and
be handed to the scheduler without leaking rights.

**It charges per lookup, not per decision.** `enqueue_message` resolves sender,
receiver, payload, and the sending continuation. Those four lookups implement
one logical operation that needs one `SEND` check.

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
guessing an index does not forge a capability. Possession within
the actor's space is the authority. This matches the ABI comment and removes the
need for a holder field.

The former global `capabilities: GenTable<CapabilityEntry>` has been replaced.

### 3.2 Genesis

Creating an entity mints a full-rights capability in the creator's space.

```text
create_object(actor, kind, bytes) -> (Ref64 object, Ref64 capability)
```

Creation requires no capability, so any process may create. Resource quotas are
outside this design. See §8.

### 3.3 Derivation

`derive(actor, cap, rights, offset, length) -> Ref64` mints a child capability
in the same space with `parent_capability` set. **Derivation may only reduce**:
the child's rights must be a subset of the parent's, and its byte range must lie
within the parent's. This is machine-checkable as a state invariant (§6, I10a)
and is the property that makes delegation safe to reason about.

### 3.4 Transfer

Capabilities move between processes on messages, via the existing
`MessageDescriptor.transferred_capability`. Send removes or copies (per
`transfer_policy`) the entry from the sender's space. Delivery installs it in
the receiver's. Transfer requires `TRANSFER` on the capability being moved and
`SEND` on the receiver.

Transfer is the *only* way a process obtains authority it did not create. That
single sentence is what makes the model analysable: authority flows along
message edges, so the reachable authority of a process is bounded by its
communication graph.

### 3.5 Revocation

Two mechanisms, both already present in the ABI:

- `valid_until_epoch` sets an expiry checked at use. It needs no bookkeeping.
- `parent_capability` makes revocation apply to the derivation subtree.
  Deleting an entry revokes it. Descendants are detected at use by
  walking to a dead parent, or eagerly by sweeping the tree. Prefer the eager
  sweep: lazy detection makes the cost of a use depend on derivation depth,
  which is a divergence source under cohorted execution (§5.4).

`object_version` gives a third, narrower mechanism: pinning a capability to a
version invalidates it when the object is frozen. Keep it, but do not rely on it
for revocation. Section 7.1 changes what freezing means.

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
   the epoch bound, target, and derivation chain. It emits a
   trace event either way (§5.2).

4. **An unsafe raw test accessor.** `tests/semantics.rs` deliberately injects
   illegal states and needs unmediated access. It lives in an explicitly named,
   doc-hidden `raw` module and requires `unsafe` at every call site, so bypasses
   are deliberate and greppable. Moving the negative integration tests into a
   crate-local harness would allow the module itself to be `#[cfg(test)]`. Until
   then, safe callers cannot use the bypass.

Step 1 is complete. It made kernel state private without changing runtime
behaviour.

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

A denial changes the program, so it belongs in the trace and must be
deterministic. Two runs of the same program
must deny identically. Add `AuthorityGranted` / `AuthorityDenied` event kinds.

Emitting on grant as well as denial roughly doubles trace volume for mutating
operations, and buys the strongest form of I10 (§6, I10c). Make it a build
feature if the volume is excessive. Keep it enabled by default so I10c remains
testable.

### 5.3 The acting principal is the process, not the continuation

Continuations are execution slices, not security principals. A continuation acts
with its process's authority. This keeps authority stable across a process's
resume points, which is necessary for §5.1 to be meaningful.

### 5.4 Do not make authority checks data-dependent per lane

In a cohorted executor, all lanes execute the authority check. Variable work per
lane, such as walking derivation chains of different depths, causes divergence.
Prefer constant-time checks: eager revocation
sweeps (§3.5), a resolved rights mask on the entry, no chain walk at use.

The scalar interpreter need not optimize this yet, but the capability model
should permit a constant-time implementation.

---

## 6. What I10 becomes

Split into three clauses, two checkable as state predicates immediately:

**I10a. Attenuation [checked].** Every capability's rights are a subset of its
parent's, and its byte range lies within its parent's. A state predicate over
the capability spaces checks this rule.

**I10b. Capability integrity [checked].** Every capability's target resolves,
its kind is one the rights apply to, and its parent (if any) is live. Same shape
as I1.

**I10c. No unauthorised effect [checked, trace-level].** Every state-changing
event in the trace is immediately preceded by an `AuthorityGranted` event naming
the same actor, right, and target. This clause says the
machine is safe, and it is checkable only because §5.2 makes authority
observable.

Then flip the marker in `docs/SOMA-v0.2.md` §5 from **absent** to **checked**,
and rewrite
`tests/semantics.rs::write_authority_is_enforced_while_other_rights_remain_permissive`
into a full-denial test. The current test deliberately records the boundary
between enforced `WRITE` and the remaining permissive rights.

---

## 7. What this forces elsewhere

These are the parts worth arguing about before writing code.

### 7.1 Ownership and capabilities are currently two mechanisms for one thing

`ObjectDescriptor.unique_owner` and a `WRITE` capability both claim to govern
mutation. Two mechanisms for one property is how models acquire contradictions:
nothing says what happens when they disagree.

**Recommendation: eliminate `unique_owner` and define ownership in terms of
capabilities.** An object is *uniquely owned* exactly when one live `WRITE`
capability for it exists. It is *frozen* when no `WRITE` capability for it
exists and at least one `READ` capability does. Freezing is then not a flag to
be set but the act of destroying the write capability, which makes the model's
one-way rule structural rather than enforced by a check.

This collapses I9 into I10, deletes `ownership.rs`'s advisory transfer, and
makes "mutation of a frozen object requires allocating a new object" true by
construction. It is a real simplification, and it is the reason to settle
capabilities before ownership (spec §6.2) rather than after.

### 7.2 Failure containment needs an answer for authority

Spec §6.3 is open: when a process faults, nothing says what happens to its
futures or their waiters. Capabilities widen the question, a faulted process
holds a capability space, and capabilities it derived and transferred are live
elsewhere. Two coherent positions:

- Transferred capabilities may remain valid after the holder faults. The
  space is reclaimed but its exports survive, so a faulted process's
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

1. **Privatise kernel state. Complete.** No behaviour change, no capability logic.
   Mechanical, touches every experiment, and is the step that makes the rest
   enforceable. Largest diff, lowest risk.
2. **Rename `CapabilityEntry.object` to `target`.** Complete.
3. **Capability spaces, genesis, derivation, I10a, and I10b. Complete.**
   Capability references are actor-relative and structural authority is now
   executable.
4. **Thread `actor` through the operation surface. Complete.** All governed
   operations name their principal and pass through one gate. This checkpoint
   originally landed permissive to isolate API churn from semantic change;
   step 5 now enables rights incrementally.
5. **Enforce rights, in progress.** `WRITE` is complete, including expiry and
   parent-revocation checks at use. Continue one right at a time across the
   remaining operation surface.
6. **Trace events and I10c.**
7. **Unify ownership (§7.1)**, delete `unique_owner`, collapse I9.
8. **Settle failure (§7.2)** and update the spec.

---

## 10. The check surface

The table lists every current operation that needs an authority check. Add a row
when adding an operation.

| Operation | Right | Target | Notes |
| --- | --- | --- | --- |
| `create_object` | none | none | Genesis. Mints full rights to actor. Unauthorised by design (§8). |
| `object_bytes` / `read_u64_object` | `READ` | object | Also bounds-check against the capability's offset/length. |
| `object_bytes_mut` | `WRITE` | object | After §7.1, implies unique ownership. |
| `freeze` | `FREEZE` | object | After §7.1, becomes "destroy the write capability". |
| `transfer_unique` | `TRANSFER` | object | Deleted entirely by §7.1. |
| table `delete` | `DESTROY` | any | Not currently reachable from an operation. Wire it when process teardown exists. |
| `create_process` | none | none | Genesis. Needs a domain quota it does not have (§8). |
| `create_continuation` | `WRITE` | process | Creating execution in a process is a mutation of it. |
| `create_future` | none | none | Genesis. |
| `await_future` | `AWAIT` | future | Check on resume too, not only on registration (§5.1). |
| `resolve_future` | `RESOLVE` | future | The right that matters most: resolution is single-assignment, so holding it is the ability to decide a value nobody can overwrite. |
| `enqueue_message` | `SEND` | receiver process | Plus `TRANSFER` on any attached capability. |
| `receive_message` | `RECEIVE` | own mailbox | Trivially held today. It becomes meaningful with channels. |
| `ingest_message` | system | receiver | External input from the system principal. |
| `run_epoch` and scheduling | system | none | The machine acting as itself. |

### 10.1 Introspection, deliberately left ungated

The remaining public operations, `process_state`, `continuation_state`,
`future_value`, `mailbox_len`, the scheduler counters, `trace_snapshot`, read
state without changing it, and this proposal does not gate them. Two different
justifications, and only one of them is sound:

- **`future_value` is fine.** It returns a `Ref64`, and a reference is not an
  authority (§2). Handing out a reference to a value the caller cannot then use
  without a capability leaks nothing. This is the design commitment paying off.
- `mailbox_len`, `process_state`, and the counters expose another process's
  progress and queue depth to anyone who can name it. Section 8 leaves
  information flow undefined. If observation should require authority, use
  `READ` on the target process after the gate exists.

`trace_snapshot` is implementation observability, not a program-visible
operation, and should move behind the `raw` boundary in step 1.

Derived from the public surface of `kernel/mod.rs`, `kernel/ownership.rs`, and
`kernel/epochs.rs` as of `aed270b`. Regenerate with
`grep -n "    pub fn " src/kernel/*.rs` and diff against this table when adding
operations. A new row requires a corresponding gate call.
