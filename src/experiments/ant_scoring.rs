//! Movement scoring as a batch evaluator, run on the CPU reference backend and
//! — with `--features metal` on macOS — on real Apple GPU hardware.
//!
//! # What this does and does not claim
//!
//! The obvious thing to say about an ant colony on a GPU is "pheromone sensing
//! runs on the accelerator". That is still not true here, though the reason has
//! moved.
//!
//! It used to be the body language: a body was straight-line SSA over *one
//! element's own fields*, with no indexed read of another element, and sensing
//! *is* a gather. `Op::Gather` and `Op::Index` removed that limit — a body can
//! now read any element of the frozen input array, and `examples::NEIGHBOUR_MAX`
//! is a stencil doing exactly that on real hardware.
//!
//! What remains is a collective-level limit rather than a language one. Sensing
//! gathers from the *trail grid*, and a `BatchEvaluate` binds exactly one input
//! array — the one the ants themselves are elements of (`ir.rs`,
//! `instantiate_batch`). A body can reach any ant; it has no name for the grid.
//! Expressing sensing needs a second read-only array binding, which is a change
//! to the collective, the capability escrow that freezes its inputs, and both
//! backends' signatures. Until that exists, the gather stays on the CPU.
//!
//! What is expressible is the decision that follows that gather. Once an ant's
//! eight neighbour readings are packed into its own element, choosing the best
//! direction is a fold of `CmpLt` and `Select` — branch-free by construction,
//! which is exactly the shape a uniform-dispatch executive wants. So:
//!
//! * the gather stays on the CPU, and this module packs it;
//! * the **scoring** is a real evaluator body, lowered to both backends from one
//!   source, and on macOS it is a real Metal compute dispatch.
//!
//! That is a narrower claim than "the simulation runs on the GPU", and it is the
//! one the machine actually supports. `agreement` checks the two backends
//! produce identical bytes, which is I20 applied to this body.
//!
//! # The body
//!
//! An element is nine `u32` fields: eight trail readings and a slot for the
//! answer. The body is an argmax written without branches —
//!
//! ```text
//! best, index := f0, 0
//! for i in 1..8:
//!     take := best < fi
//!     best  := select(take, fi, best)
//!     index := select(take, i,  index)
//! store field 8 := index
//! ```
//!
//! — which is thirty-seven operations and no control flow. Every lane of a
//! cohort running it executes the same instructions in the same order.

use crate::compiler::body::EvaluatorProgram;
use crate::compiler::ir::Module;
use crate::experiments::ant_colony::DIRECTIONS;

/// Evaluator id. The example module occupies 7–10.
pub const ANT_MOVEMENT_SCORE: u32 = 11;

/// Readings per element, one per neighbour.
pub const READINGS: usize = 8;
/// Nine `u32` fields: eight readings and the chosen direction.
pub const FIELDS: usize = READINGS + 1;
/// Bytes per element.
pub const STRIDE: u32 = (FIELDS * 4) as u32;
/// The field the body writes.
pub const RESULT_FIELD: usize = READINGS;

/// Generate the module source.
///
/// Written as a generator rather than a literal because the body is a regular
/// unrolled fold: hand-transcribing thirty-seven operations with interlocking
/// indices is a way to introduce a typo that both backends would then agree
/// about, which is precisely the failure I20 exists to catch.
pub fn source() -> String {
    let mut s = String::from("module soma.ants\n");
    s.push_str(&format!(
        "evaluator {ANT_MOVEMENT_SCORE} ant_movement_score {STRIDE} 110 111 ro 112 113 ro\n"
    ));
    for _ in 0..FIELDS {
        s.push_str("  field u32\n");
    }
    // best := f0, index := 0
    s.push_str("  op 0 load 0\n");
    s.push_str("  op 1 const 0\n");
    let (mut best, mut index) = (0u32, 1u32);
    for reading in 1..READINGS as u32 {
        let base = 2 + (reading - 1) * 5;
        s.push_str(&format!("  op {} load {reading}\n", base));
        s.push_str(&format!("  op {} cmplt {best} {}\n", base + 1, base));
        s.push_str(&format!("  op {} select {} {} {best}\n", base + 2, base + 1, base));
        s.push_str(&format!("  op {} const {reading}\n", base + 3));
        s.push_str(&format!(
            "  op {} select {} {} {index}\n",
            base + 4,
            base + 1,
            base + 3
        ));
        best = base + 2;
        index = base + 4;
    }
    s.push_str(&format!("  store {RESULT_FIELD} {index}\n"));
    s
}

/// Parse the module.
pub fn module() -> Module {
    Module::parse(&source()).expect("the generated ant-scoring module must parse")
}

/// The scoring body.
pub fn program() -> EvaluatorProgram {
    let module = module();
    module
        .evaluators()
        .iter()
        .find(|e| e.id == ANT_MOVEMENT_SCORE)
        .and_then(|e| e.body.clone())
        .expect("the module declares the scoring body")
}

/// One ant's gathered neighbourhood, ready to be scored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Neighbourhood {
    /// Trail strength at each of the eight neighbours, in `DIRECTIONS` order.
    /// An impassable neighbour reads zero, so it can never win the argmax and
    /// the body needs no notion of passability.
    pub readings: [u32; READINGS],
}

impl Neighbourhood {
    /// What the body should decide, computed here so a test has something
    /// independent to compare a backend against.
    pub fn expected(&self) -> u32 {
        let mut best = self.readings[0];
        let mut index = 0u32;
        for (candidate, value) in self.readings.iter().enumerate().skip(1) {
            if best < *value {
                best = *value;
                index = candidate as u32;
            }
        }
        index
    }

    /// The offset the chosen direction corresponds to.
    pub fn direction(&self) -> (i32, i32) {
        DIRECTIONS[self.expected() as usize % 8]
    }
}

/// Pack gathered neighbourhoods into the frozen input array the collective
/// takes. This is the gather boundary: everything before it is CPU work over a
/// shared field, everything after it is one uniform dispatch.
pub fn pack(batch: &[Neighbourhood]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(batch.len() * STRIDE as usize);
    for item in batch {
        for reading in item.readings {
            bytes.extend_from_slice(&reading.to_le_bytes());
        }
        // The result field goes out zeroed; the body overwrites it.
        bytes.extend_from_slice(&0u32.to_le_bytes());
    }
    bytes
}

/// Read the chosen directions back out of a published output array.
pub fn unpack(bytes: &[u8], count: usize) -> Vec<u32> {
    (0..count)
        .map(|index| {
            let at = index * STRIDE as usize + RESULT_FIELD * 4;
            match bytes.get(at..at + 4) {
                Some(word) => u32::from_le_bytes([word[0], word[1], word[2], word[3]]),
                None => 0,
            }
        })
        .collect()
}

/// A deterministic batch of neighbourhoods, including the edges worth hitting:
/// an all-zero neighbourhood, a tie at the maximum, a single winner in each
/// position, and saturated readings.
pub fn sample_batch() -> Vec<Neighbourhood> {
    let mut batch = vec![
        Neighbourhood { readings: [0; 8] },
        // A tie. `CmpLt` is strict, so the first maximum wins — the same rule
        // the reference above applies.
        Neighbourhood {
            readings: [9, 9, 9, 9, 9, 9, 9, 9],
        },
        Neighbourhood {
            readings: [u32::MAX, 1, 2, 3, 4, 5, 6, 7],
        },
        Neighbourhood {
            readings: [1, 2, 3, 4, 5, 6, 7, u32::MAX],
        },
    ];
    // A single winner in each position, so every `select` arm is exercised.
    for winner in 0..READINGS {
        let mut readings = [3u32; READINGS];
        readings[winner] = 1000;
        batch.push(Neighbourhood { readings });
    }
    // A deterministic spread.
    let mut state = 0x5EED_A417_C0FF_EE11u64;
    for _ in 0..24 {
        let mut readings = [0u32; READINGS];
        for slot in readings.iter_mut() {
            state = crate::experiments::ant_colony::split_mix(&mut state.clone())
                ^ state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            *slot = (state >> 33) as u32;
        }
        batch.push(Neighbourhood { readings });
    }
    batch
}
