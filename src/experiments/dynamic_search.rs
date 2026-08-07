//! Synthetic branching-search workload (§25.1).
//!
//! This workload has controllable divergence: each state executes one of a few
//! continuation paths, spawns a configurable number of children, does a
//! configurable amount of arithmetic/memory work, and either terminates or
//! yields. It exists to map the regime in which continuation cohorting helps or
//! fails — not to be a realistic application.

use crate::abi::{ProcessMode, StateAccess};
use crate::compiler::frame::Frame;
use crate::compiler::run_classes::DEFAULT_MAX_STEPS;
use crate::compiler::state_machine_lowering::SearchFrame;
use crate::kernel::{ContinuationSpec, Kernel};

/// Control variables exposed by §25.1.
#[derive(Clone, Copy, Debug)]
pub struct ControlKnobs {
    /// Number of children per internal node.
    pub branching_factor: u32,
    /// Tree depth (nodes at depth 0 are leaves).
    pub depth: u32,
    /// Number of independent root searches (processes) started.
    pub process_count: u32,
    /// Number of distinct continuation classes used (1 for Phase 1).
    pub class_count: u32,
    /// Bounded arithmetic iterations per node ("state duration").
    pub arithmetic_ops: u32,
    /// Bytes of "memory intensity" per node (reserved, unused in scalar phase).
    pub memory_intensity: u32,
    /// Whether internal nodes await a heuristic future before branching.
    pub use_heuristic: bool,
    /// Arrival rate (roots enqueued per epoch); unused in the one-shot phase.
    pub arrival_rate: u32,
}

impl Default for ControlKnobs {
    fn default() -> Self {
        ControlKnobs {
            branching_factor: 3,
            depth: 4,
            process_count: 2,
            class_count: 1,
            arithmetic_ops: 16,
            memory_intensity: 0,
            use_heuristic: false,
            arrival_rate: 1,
        }
    }
}

impl ControlKnobs {
    /// Total number of nodes that will be visited: roots * (b^0 + b^1 + ... + b^depth).
    pub fn node_count(&self) -> u64 {
        let b = self.branching_factor as u64;
        let mut total = 0u64;
        let mut level = 1u64; // nodes at current depth level
        for _ in 0..=self.depth {
            total += level;
            level = level.saturating_mul(b);
        }
        self.process_count as u64 * total
    }
}

/// Spawn the branching-search roots into a fresh kernel. Returns the kernel.
pub fn build(knobs: &ControlKnobs) -> Kernel {
    build_in(Kernel::new(), knobs)
}

/// Spawn the roots into an already-configured kernel — used to vary scheduling
/// mode, cohort width, or partial policy without duplicating workload setup.
pub fn build_in(mut kernel: Kernel, knobs: &ControlKnobs) -> Kernel {
    for root in 0..knobs.process_count {
        let p = kernel.create_process(crate::kernel::SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let frame = SearchFrame {
            value: root as u64 + 1,
            depth: knobs.depth,
            branching: knobs.branching_factor,
            work_iters: knobs.arithmetic_ops,
            class_count: knobs.class_count,
        };
        let run_class = frame.run_class();
        let mut bytes = Vec::new();
        frame.encode(&mut bytes);
        kernel
            .create_continuation(
                crate::kernel::SYSTEM_PRINCIPAL,
                p,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    run_class,
                    0,
                    bytes,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .expect("system may create root continuations");
    }
    kernel
}

/// Build and run to quiescence. Returns the number of epochs required.
pub fn run(knobs: &ControlKnobs) -> u32 {
    let mut kernel = build(knobs);
    kernel.run_to_quiescence(100_000)
}
