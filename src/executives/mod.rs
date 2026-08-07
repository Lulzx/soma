//! Engine executives (§16): scalar semantics plus physical batch backends.

pub mod ant_colony;
pub mod batch;
pub mod cpu_scalar;
pub mod lane;
/// Standalone resident future/mailbox synchronization reference backend.
pub mod resident_sync;

#[cfg(feature = "native")]
pub mod native;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_scheduler;

/// Standalone one-command-buffer resident synchronization backend.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_resident_sync;
