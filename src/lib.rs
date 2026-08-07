//! SOMA: executable abstract-machine semantics and reference implementations.
//!
//! The crate contains the dependency-free semantic core, executable invariants,
//! validation workloads, and physical batch backends:
//!
//! 1. Fixed ABI references and generational tables (`abi`, `table`);
//! 2. A deterministic CPU continuation interpreter (`executives::cpu_scalar`)
//!    and an optional Metal batch backend (`executives::metal`);
//! 3. Processes, messages, futures, and double-buffered runnable bins over a
//!    deterministic reference and speculative concurrent epoch lifecycles
//!    (`kernel`, `scheduler`), together with
//!    the `Expand` state machine and a synthetic branching-search workload
//!    (`compiler`, `experiments`) that prove resumable continuations run
//!    deterministically.
//!
//! Metal includes both collective evaluation and a concurrent resident
//! scheduler. General continuation handlers still execute through the
//! speculative CPU snapshots before their device-ready journals are validated
//! and canonically replayed.

pub mod abi;
pub mod compiler;
pub mod discovery;
pub mod distributed;
pub mod executives;
pub mod experiments;
pub mod kernel;
pub mod replay;
pub mod scheduler;
pub mod semantics;
pub mod table;
