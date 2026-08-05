//! Engine executives (§16): scalar semantics plus physical batch backends.

pub mod batch;
pub mod cpu_scalar;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
