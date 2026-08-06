//! Engine executives (§16): scalar semantics plus physical batch backends.

pub mod ant_colony;
pub mod batch;
pub mod cpu_scalar;
pub mod lane;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
