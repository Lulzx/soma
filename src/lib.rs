//! SOMA: executable abstract-machine semantics and reference implementations.
//!
//! The crate contains the dependency-free semantic core, executable invariants,
//! validation workloads, and physical batch backends:
//!
//! 1. Fixed ABI references and generational tables (`abi`, `table`);
//! 2. A deterministic CPU continuation interpreter (`executives::cpu_scalar`)
//!    and an optional Metal batch backend (`executives::metal`);
//! 3. Processes, messages, futures, and double-buffered runnable bins over a
//!    single-threaded epoch lifecycle (`kernel`, `scheduler`), together with
//!    the `Expand` state machine and a synthetic branching-search workload
//!    (`compiler`, `experiments`) that prove resumable continuations run
//!    deterministically.
//!
//! The Metal backend is a collective-level implementation, not a persistent
//! device-resident scheduler.

pub mod abi;
pub mod compiler;
pub mod executives;
pub mod experiments;
pub mod kernel;
pub mod replay;
pub mod scheduler;
pub mod semantics;
pub mod table;
