//! The SOMA abstract machine's semantics, as executable checks.
//!
//! `docs/SOMA-v0.2.md` is the specification; this module is the part of it that
//! runs. Keeping the invariants in code rather than prose is what stops the
//! document from describing a machine the implementation does not build.

pub mod invariants;
pub mod order;
