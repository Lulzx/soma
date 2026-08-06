//! Movement scoring as a batch evaluator, run on the CPU reference backend and
//! — with `--features metal` on macOS — on real Apple GPU hardware.
//!
//! # What this does and does not claim
//!
//! Pheromone sensing runs on the accelerator. Getting to that sentence took two
//! separate limits, and both are gone:
//!
//! * The body language. A body was straight-line SSA over *one element's own
//!   fields*, with no indexed read of another element, and sensing *is* a
//!   gather. `Op::Gather` and `Op::Index` removed that.
//! * The collective. Sensing gathers from the *trail grid*, and a
//!   `BatchEvaluate` bound exactly one input array — the one the ants are
//!   elements of. A body could reach any ant and had no name for the grid.
//!   `Op::GatherAux` and the second binding on `create_batch_evaluate_bound`
//!   removed that: the grid is a second frozen, read-only array, authorized and
//!   escrowed on the same terms as the first.
//!
//! So there are two bodies here, and keeping both is deliberate.
//!
//! * `ant_movement_score` takes an element holding eight readings some caller
//!   gathered and folds them into a direction. This is the shape the machine
//!   supported before the second binding, and it is still what a caller wants
//!   when the readings come from somewhere that is not a frozen array.
//! * `ant_sense_and_score` takes an ant's position and the grid, does the eight
//!   reads itself with `gatheraux`, and folds the same fold. This is the whole
//!   of sense-then-choose as one dispatch.
//!
//! The second is checked three ways, because two would not be enough. It agrees
//! with an independent reference computed from `executives/ant_colony.rs`'s own
//! sensing code, so a body with a transposed direction or an off-by-one bound
//! fails rather than being confirmed; it agrees between the CPU interpreter and
//! real Metal, which is I20; and the batch covers every cell of the grid on both
//! trail channels, so every edge and corner exercises the bounds test.
//!
//! What is still not claimed is that the *simulation* runs on the GPU.
//! `executives/ant_colony.rs` still senses on the host during a colony run —
//! wiring the sensing collective into the ant step is a change to the
//! executive, not to the language or the collective, and it is not made here.
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
        s.push_str(&format!(
            "  op {} select {} {} {best}\n",
            base + 2,
            base + 1,
            base
        ));
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

// ---- sensing ---------------------------------------------------------------

/// Evaluator id for the body that senses *and* scores.
pub const ANT_SENSE_AND_SCORE: u32 = 16;

/// Element fields for the sensing body: `x`, `y`, `width`, `height`, the trail
/// channel to read, and the chosen direction.
///
/// `width`, `height` and the channel are the same for every ant in a batch and
/// are still carried per element, because the language has no uniform. That is
/// 12 bytes an ant of redundancy against a trail grid of tens of kilobytes,
/// which is the trade a constant buffer would remove and nothing else would.
pub const SENSE_X: usize = 0;
pub const SENSE_Y: usize = 1;
pub const SENSE_WIDTH: usize = 2;
pub const SENSE_HEIGHT: usize = 3;
pub const SENSE_CHANNEL: usize = 4;
pub const SENSE_RESULT: usize = 5;
pub const SENSE_FIELDS: usize = 6;
pub const SENSE_STRIDE: u32 = (SENSE_FIELDS * 4) as u32;

/// Instruction accumulator, so the generator names instructions by what they
/// are rather than by arithmetic on a running index.
///
/// The scoring body's generator computes its operand indices from a base, which
/// works because its shape is one repeated block. The sensing body's is not
/// regular — a preamble, eight direction blocks, then a fold — and computing
/// indices by hand across that is the way to introduce a typo both backends
/// would then agree about.
struct Emit {
    lines: Vec<String>,
}

impl Emit {
    fn new() -> Self {
        Self { lines: Vec::new() }
    }

    fn op(&mut self, text: String) -> u32 {
        let index = self.lines.len() as u32;
        self.lines.push(format!("  op {index} {text}"));
        index
    }

    fn constant(&mut self, value: u64) -> u32 {
        self.op(format!("const {value}"))
    }
}

/// Generate the module source for the sensing body.
///
/// This is the gather that used to stay on the CPU. `executives/ant_colony.rs`
/// reads eight neighbours out of the trail grid per ant and packs them into the
/// element the scoring body reads; the same eight reads are expressed here as
/// `gatheraux` against the grid, so the whole of sense-then-choose is one
/// dispatch rather than a host loop feeding one.
///
/// It reproduces `sense`'s semantics exactly, because a body that computed
/// something *nearly* the same would be a body nothing could check:
///
/// * a neighbour outside the grid reads zero, and so can never win;
/// * the grid is `(channel * cells + cell)` in `u16`, which is
///   `ant_colony::field_offset` with the multiply by two supplied by the aux
///   element stride; and
/// * the winner is the first strict maximum, which is `Neighbourhood::expected`.
///
/// The bounds test is unsigned rather than signed. `x - 1` at `x == 0` wraps to
/// a value far past `width`, so one `cmplt` against `width` rejects both edges
/// and there is no need for a comparison the language does not have.
pub fn sensing_source() -> String {
    let mut e = Emit::new();

    let x = e.op(format!("load {SENSE_X}"));
    let y = e.op(format!("load {SENSE_Y}"));
    let width = e.op(format!("load {SENSE_WIDTH}"));
    let height = e.op(format!("load {SENSE_HEIGHT}"));
    let channel = e.op(format!("load {SENSE_CHANNEL}"));
    let cells = e.op(format!("mul {width} {height}"));
    let base = e.op(format!("mul {channel} {cells}"));
    let zero = e.constant(0);

    let mut readings = Vec::with_capacity(READINGS);
    for (dx, dy) in DIRECTIONS {
        // A negative offset is added as its wrapping `u64`, which is what makes
        // the single unsigned bounds test below correct.
        let ox = e.constant(dx as i64 as u64);
        let oy = e.constant(dy as i64 as u64);
        let nx = e.op(format!("add {x} {ox}"));
        let ny = e.op(format!("add {y} {oy}"));
        let in_x = e.op(format!("cmplt {nx} {width}"));
        let in_y = e.op(format!("cmplt {ny} {height}"));
        let inside = e.op(format!("and {in_x} {in_y}"));
        let row = e.op(format!("mul {ny} {width}"));
        let cell = e.op(format!("add {row} {nx}"));
        let at = e.op(format!("add {base} {cell}"));
        let raw = e.op(format!("gatheraux {at} 0"));
        readings.push(e.op(format!("select {inside} {raw} {zero}")));
    }

    // The same first-strict-maximum fold the scoring body performs, over values
    // this body gathered rather than values the host packed.
    let mut best = readings[0];
    let mut index = zero;
    for (direction, reading) in readings.iter().enumerate().skip(1) {
        let take = e.op(format!("cmplt {best} {reading}"));
        best = e.op(format!("select {take} {reading} {best}"));
        let candidate = e.constant(direction as u64);
        index = e.op(format!("select {take} {candidate} {index}"));
    }

    let mut s = String::from("module soma.ants.sensing\n");
    s.push_str(&format!(
        "evaluator {ANT_SENSE_AND_SCORE} ant_sense_and_score {SENSE_STRIDE} 160 161 ro 162 163 ro\n"
    ));
    for _ in 0..SENSE_FIELDS {
        s.push_str("  field u32\n");
    }
    // One `u16` per aux element: the trail grid is a flat array of `u16`
    // channels laid out `(channel * cells + cell)`, which is exactly an element
    // index once the stride supplies the multiply by two.
    s.push_str("  aux u16\n");
    for line in &e.lines {
        s.push_str(line);
        s.push('\n');
    }
    s.push_str(&format!("  store {SENSE_RESULT} {index}\n"));
    s
}

pub fn sensing_module() -> Module {
    Module::parse(&sensing_source()).expect("the generated ant-sensing module must parse")
}

/// The sensing body.
pub fn sensing_program() -> EvaluatorProgram {
    sensing_module()
        .evaluators()
        .iter()
        .find(|e| e.id == ANT_SENSE_AND_SCORE)
        .and_then(|e| e.body.clone())
        .expect("the module declares the sensing body")
}

/// One ant's position and the grid it is on, as the sensing body reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sensor {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub channel: u32,
}

/// Pack sensors into the frozen input array the sensing collective takes.
pub fn pack_sensors(batch: &[Sensor]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(batch.len() * SENSE_STRIDE as usize);
    for sensor in batch {
        for value in [
            sensor.x,
            sensor.y,
            sensor.width,
            sensor.height,
            sensor.channel,
            0,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// Read the chosen directions out of a published sensing output array.
pub fn unpack_sensors(bytes: &[u8], count: usize) -> Vec<u32> {
    (0..count)
        .map(|index| {
            let at = index * SENSE_STRIDE as usize + SENSE_RESULT * 4;
            match bytes.get(at..at + 4) {
                Some(word) => u32::from_le_bytes([word[0], word[1], word[2], word[3]]),
                None => 0,
            }
        })
        .collect()
}

/// What the sensing body should decide, computed the way
/// `executives/ant_colony.rs` gathers and `Neighbourhood::expected` chooses.
///
/// This is the independent answer a test compares a backend against, and it is
/// written from the host-side sensing code rather than from the body, so a body
/// with a transposed direction or an off-by-one bound has something to fail.
pub fn expected_direction(sensor: &Sensor, grid: &[u8]) -> u32 {
    let cells = sensor.width as usize * sensor.height as usize;
    let mut readings = [0u32; READINGS];
    for (index, (dx, dy)) in DIRECTIONS.iter().enumerate() {
        let nx = sensor.x as i64 + *dx as i64;
        let ny = sensor.y as i64 + *dy as i64;
        if nx < 0 || ny < 0 || nx >= sensor.width as i64 || ny >= sensor.height as i64 {
            continue;
        }
        let cell = ny as usize * sensor.width as usize + nx as usize;
        readings[index] = u32::from(crate::experiments::ant_colony::read_trail(
            grid,
            cells,
            sensor.channel as usize,
            cell,
        ));
    }
    Neighbourhood { readings }.expected()
}
