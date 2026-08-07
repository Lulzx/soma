//! Application-level execution model for machine-generated discovery.
//!
//! Discovery deliberately lives above the SOMA ABI. It replays an identical
//! logical research trace through either a literal executor or an optimized
//! one which shares deterministic work, withdraws work with no remaining
//! scientific interest, and fuses compatible pointwise evaluations.

pub mod batcher;
pub mod executor;
pub mod graph;
pub mod interest;
pub mod invariants;
pub mod key;
pub mod metrics;
pub mod node;
pub mod registry;
pub mod trace;

pub use executor::{execute_naive, execute_optimized, DiscoveryError, DiscoveryResult};
pub use key::{ExperimentKey, ModuleDigest, NodeDigest, ObjectDigest};
pub use metrics::DiscoveryMetrics;
pub use node::{DiscoveryNode, EvaluationSpec, FusionClass, HypothesisId, RequestId};
pub use trace::{DiscoveryEvent, DiscoveryTrace};
