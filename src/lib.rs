//! SOMA-P1: minimal executable kernel contract — Phase 1 (steps 1–3).
//!
//! This crate implements the first slice of the evidence-producing path from
//! §30:
//!
//! 1. Fixed ABI references and generational tables (`abi`, `table`);
//! 2. A deterministic CPU continuation interpreter (`executives::cpu_scalar`);
//! 3. Processes, messages, futures, and double-buffered runnable bins over a
//!    single-threaded epoch lifecycle (`kernel`, `scheduler`), together with
//!    the `Expand` state machine and a synthetic branching-search workload
//!    (`compiler`, `experiments`) that prove resumable continuations run
//!    deterministically.
//!
//! GPU execution, real SIMD cohort construction, migration, and the baselines
//! are later slices.

pub mod abi;
pub mod compiler;
pub mod executives;
pub mod experiments;
pub mod kernel;
pub mod replay;
pub mod scheduler;
pub mod semantics;
pub mod table;
