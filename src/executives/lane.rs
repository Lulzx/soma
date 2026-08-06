//! What a step is allowed to touch (`docs/SOMA-v0.3.md` §4.10).
//!
//! A handler took `&mut Kernel`, which is why §4.3 could name execute/commit
//! fusion as *the* obstacle to a concurrent executive without having to say
//! much more: a step holding the whole kernel mutably can do anything to it,
//! and several steps cannot hold it at once. Every piece of §4 since has been
//! removing a reason that has to be true — effects are produced rather than
//! performed (§4.4), applied at the epoch boundary (§4.5), lanes reorder
//! without changing a run (§4.6), and allocation has a lane-local form (§4.8).
//!
//! What was never established is the part that sounds like bookkeeping: *what a
//! step actually does*. "A handler can do anything to the kernel" is a fact
//! about a type signature, not about the handlers, and the difference decides
//! how much work the rest of the refactor is. This type answers it. A step
//! takes a `LaneView`, and a `LaneView` offers fifteen operations.
//!
//! Fifteen, measured rather than chosen — they are what `cpu_scalar` and
//! `executives::ant_colony` already called, and the type was written around
//! that list. The value is not the count; it is that the list is now closed and
//! the compiler holds it closed. An operation that has no lane-local form
//! cannot be reached from a step, so the remaining work is enumerable instead
//! of being "audit three thousand lines of kernel for what a handler might
//! call". That is the technique `docs/SOMA-CAPABILITIES.md` used to close the
//! operation set and §4.1 used to seal `Admission`, applied to the executive.
//!
//! **This view still borrows the kernel mutably**, so it does not make lanes
//! run concurrently and nothing here claims it does. It makes the set of things
//! that would have to change finite and visible. The categories are:
//!
//! * *Reads* — `continuations`, `epoch_number`, `future_value`, `object_bytes`,
//!   `read_u64_object`.
//!   These need a shared borrow and nothing else once the lane stops holding
//!   `&mut`.
//! * *Allocation* — `create_process`, `create_continuation`, `create_future`,
//!   `create_object`. §4.8's shards are the lane-local form; wiring them is
//!   mechanical.
//! * *Own-frame writes* — `host_payload_mut`, `object_bytes_mut`. I8 (frame
//!   exclusivity) already guarantees no two continuations share a frame, so
//!   these are disjoint across lanes by an invariant the checker enforces.
//! * *Cross-lane writes* — `enqueue_message`, `resolve_future`, `await_future`,
//!   `receive_message`. These are the genuinely hard four: they touch state
//!   another lane may also touch, so they need journalling in the shape §4.4
//!   used for bin entries. I25 is what makes that safe — no lane observes
//!   another within an epoch — and it is already checked.
//!
//! Four operations, not a kernel. That is the result of writing this down.

use crate::abi::{ObjectKind, ProcessMode, Ref64};
use crate::kernel::{ContinuationSpec, Kernel, RuntimeError};

/// The kernel as one lane of one epoch may use it.
///
/// Constructed by the epoch loop around the lane it is about to run, and
/// dropped when the lane ends. It deliberately does *not* implement `Deref` to
/// `Kernel`: the whole point is that the reachable set is the list below, and a
/// deref would reopen it. The seal is only worth having if there is no way
/// around it, so both halves are compile failures rather than conventions.
///
/// A lane is not a kernel:
///
/// ```compile_fail
/// fn f(lane: &mut soma::executives::lane::LaneView<'_>) {
///     let _: &soma::kernel::Kernel = lane;
/// }
/// ```
///
/// and a step cannot reach an operation the view does not offer:
///
/// ```compile_fail
/// fn f(lane: &mut soma::executives::lane::LaneView<'_>) {
///     let _ = lane.cancel_process(soma::abi::Ref64::NULL, soma::abi::Ref64::NULL);
/// }
/// ```
///
/// The null for those three: the same path, with an operation the view *does*
/// offer, compiles. Without it a misspelled path would make every block above
/// fail for the wrong reason and pass vacuously.
///
/// ```
/// fn f(lane: &soma::executives::lane::LaneView<'_>) -> Option<soma::abi::Ref64> {
///     lane.future_value(soma::abi::Ref64::NULL)
/// }
/// ```
///
/// nor build one over a kernel it was handed:
///
/// ```compile_fail
/// fn f(kernel: &mut soma::kernel::Kernel) {
///     let _ = soma::executives::lane::LaneView::new(kernel);
/// }
/// ```
pub struct LaneView<'a> {
    kernel: &'a mut Kernel,
}

impl<'a> LaneView<'a> {
    pub(crate) fn new(kernel: &'a mut Kernel) -> LaneView<'a> {
        LaneView { kernel }
    }

    // ---- reads -----------------------------------------------------------

    pub(crate) fn continuations(
        &self,
    ) -> &crate::table::GenTable<crate::abi::continuations::ContinuationDescriptor> {
        self.kernel.continuations()
    }

    pub(crate) fn epoch_number(&self) -> u32 {
        self.kernel.epoch_number()
    }

    pub fn future_value(&self, future: Ref64) -> Option<Ref64> {
        self.kernel.future_value(future)
    }

    /// Takes `&mut` because reading is a governed effect and the authority
    /// decision is traced (I10c). That is a genuine write to the trace, not an
    /// artefact — and it is why a concurrent lane needs a lane-local trace
    /// buffer, which §4.2's position scheme already anticipates.
    pub fn object_bytes(&mut self, actor: Ref64, obj: Ref64) -> Result<&[u8], RuntimeError> {
        self.kernel.object_bytes(actor, obj)
    }

    /// A convenience over `object_bytes`, and `&mut` for the same reason.
    pub fn read_u64_object(&mut self, actor: Ref64, obj: Ref64) -> Option<u64> {
        self.kernel.read_u64_object(actor, obj)
    }

    // ---- allocation (§4.8 shards are the lane-local form) -----------------

    /// Fallible, unlike `Kernel::create_process`.
    ///
    /// A step creates a process in its own process's domain, and that domain
    /// may be bounded — so this is the one allocation whose failure is a
    /// decision the machine already has a word for. It used to be the kernel's
    /// infallible form, which turned a `DomainQuotaExceeded` a handler could
    /// have faulted on into an abort of the host process. A step that cannot
    /// allocate should fault; nothing about the machine says it should stop.
    pub fn create_process(
        &mut self,
        actor: Ref64,
        mode: ProcessMode,
    ) -> Result<Ref64, RuntimeError> {
        self.kernel.try_create_process(actor, mode)
    }

    pub fn create_continuation(
        &mut self,
        actor: Ref64,
        process: Ref64,
        spec: ContinuationSpec,
    ) -> Result<Ref64, RuntimeError> {
        self.kernel.create_continuation(actor, process, spec)
    }

    pub fn create_future(&mut self, actor: Ref64) -> Ref64 {
        self.kernel.create_future(actor)
    }

    pub fn create_object(&mut self, actor: Ref64, kind: ObjectKind, bytes: Vec<u8>) -> Ref64 {
        self.kernel.create_object(actor, kind, bytes)
    }

    // ---- own-frame writes (disjoint across lanes by I8) -------------------

    pub fn host_payload_mut(
        &mut self,
        actor: Ref64,
        obj: Ref64,
    ) -> Result<&mut Vec<u8>, RuntimeError> {
        self.kernel.host_payload_mut(actor, obj)
    }

    pub fn object_bytes_mut(
        &mut self,
        actor: Ref64,
        obj: Ref64,
    ) -> Result<&mut [u8], RuntimeError> {
        self.kernel.object_bytes_mut(actor, obj)
    }

    // ---- cross-lane writes (the four that need journalling) ---------------

    pub fn enqueue_message(
        &mut self,
        actor: Ref64,
        receiver: Ref64,
        payload: Ref64,
        sender_cont: Ref64,
    ) -> Result<(), RuntimeError> {
        self.kernel.enqueue_message(actor, receiver, payload, sender_cont)
    }

    pub fn receive_message(
        &mut self,
        actor: Ref64,
        cont: Ref64,
    ) -> Result<Option<crate::abi::MessageDescriptor>, RuntimeError> {
        self.kernel.receive_message(actor, cont)
    }

    pub fn resolve_future(
        &mut self,
        actor: Ref64,
        future: Ref64,
        value: Ref64,
    ) -> Result<(), RuntimeError> {
        self.kernel.resolve_future(actor, future, value)
    }

    pub fn await_future(
        &mut self,
        actor: Ref64,
        cont: Ref64,
        future: Ref64,
        next_run_class: u32,
    ) -> Result<crate::kernel::AwaitOutcome, RuntimeError> {
        self.kernel.await_future(actor, cont, future, next_run_class)
    }
}
