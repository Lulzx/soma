//! An ant colony as a population of persistent processes.
//!
//! Every existing workload in this directory is shaped like a computation: a
//! search tree, a streaming graph, a join. This one is shaped like a *world*.
//! Ten thousand ants each hold their own state, decide their own next behaviour
//! from it, and never synchronise with each other. Nothing in the workload sorts
//! them, batches them, or tells them to take a turn.
//!
//! What it is for is the claim in §9 and §14: that dispatch shape can be
//! *discovered* from continuations rather than imposed on the program. An ant
//! that has just found food yields to `ANT_CARRY_FOOD`; an ant that walked into
//! a rock yields to `ANT_AVOID_OBSTACLE`. Neither ant knows the other exists.
//! The scheduler groups them anyway, because the run class an ant names is the
//! bin it lands in, and a bin is uniform by construction. Run the identical
//! population under `SchedulingMode::PersistentFifo` and the same continuations
//! arrive in the same order but land in one bin, so a lane group spans whatever
//! behaviours happened to be adjacent. The difference between those two runs is
//! the mechanism, with the workload held fixed.
//!
//! # Why the population is a two-level tree
//!
//! An ant senses a shared pheromone field and deposits into it. That is a
//! many-writers-to-one-place pattern, and this machine does not have one:
//!
//! * `CapabilityIntegrity` fails an object with more than one `WRITE` holder, so
//!   ants cannot share a deposit buffer;
//! * authority lookup scans the holder's capability space linearly, so no
//!   process may hold thousands of capabilities;
//! * a sub-range grant does not authorise object access at all —
//!   `find_authorized_capability` requires `offset == 0 && length >= byte_length`
//!   for `READ`/`WRITE`/`FREEZE` — so a partitioned buffer is not a way around
//!   the first constraint;
//! * and message payloads deposit a capability in the *receiver's* space per
//!   message, which turns a per-epoch report from every ant into unbounded growth
//!   in the one space every ant talks to.
//!
//! Those four together admit exactly one bounded shape: a tree whose every
//! capability space is bounded by a constant. An ant solely owns its own deposit
//! object. A colony holds `READ` on its own ants' deposits — a fixed number —
//! and folds them into a summary it solely owns. The world holds `READ` on the
//! colony summaries — also a fixed number — and folds those into the field. No
//! space grows with the population, and nothing is allocated per epoch.
//!
//! The tree is not a workaround dressed up as a design. It is what the machine's
//! own constraints leave, and it is why a colony is a real supervision subtree
//! with real work in it rather than a label on a group of ants.
//!
//! # The field is double-buffered
//!
//! Ants read the field while the world writes it. Two buffers make that legal
//! without a lock: in epoch `n` ants read buffer `n % 2` and the world writes
//! buffer `(n + 1) % 2`, having first copied the readable one forward. So the
//! buffer being read is never the buffer being written, and the world remains
//! the sole `WRITE` holder of both. This is the same trick `DoubleBin` plays on
//! the runnable bins, for the same reason.
//!
//! # Determinism
//!
//! Every ant carries its own SplitMix64 generator seeded from its identity, so
//! the run is a pure function of `ColonyKnobs`. Nothing consults a clock and
//! nothing depends on iteration order of a map.
//!
//! # What this does not measure
//!
//! Occupancy here is the same structural quantity the rest of this directory
//! reports — derived from how continuations group, not from silicon. The CPU
//! scalar executive runs a cohort's lanes one after another. This workload makes
//! the grouping *visible*; it does not make it fast, and it says nothing about
//! throughput.

use crate::abi::PartialCohortPolicy;
use crate::abi::{ObjectKind, ProcessMode, Ref64, Rights, StateAccess, SupervisionPolicy};
use crate::compiler::frame::{put_ref64, put_u16, put_u32, put_u64, put_u8, put_vec_u64};
use crate::compiler::frame::{ByteCursor, Frame, FrameError};
use crate::compiler::run_classes::{ANT_EXPLORE, COLONY_AGGREGATE, WORLD_STEP};
use crate::kernel::accounting::Accounting;
use crate::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use crate::scheduler::runnable_bins::SchedulingMode;

/// Trail channels held in the field. Two, because an outbound ant and a laden
/// ant are following different gradients.
pub const TRAIL_FOOD: usize = 0;
pub const TRAIL_HOME: usize = 1;
pub const TRAIL_COUNT: usize = 2;

/// Bytes of one deposit record.
pub const DEPOSIT_RECORD: usize = 16;

/// Bytes of a per-ant deposit object: two records, one per epoch parity.
///
/// Every stage of the ant → colony → world pipeline is double-buffered on epoch
/// parity, and that is what makes the workload a *fair* comparison rather than
/// just a pretty one. Bins drain in bin order, so under `RunClassBins` ants
/// (classes 20–25) happen to run before colonies (26) and the world (27) within
/// an epoch — but under `PersistentFifo` everything shares one bin and the three
/// stages interleave in arrival order. If a stage read what the previous stage
/// wrote *this* epoch, the two scheduling modes would simulate different worlds
/// and any occupancy comparison between them would be meaningless.
///
/// Writing slot `epoch % 2` and reading slot `(epoch + 1) % 2` removes the
/// question: a reader never touches the slot its writer is filling, so the
/// result does not depend on which ran first. The cost is one epoch of latency
/// per stage, which an ant colony can well afford.
pub const DEPOSIT_BYTES: usize = DEPOSIT_RECORD * 2;

/// The deposit slot written during `epoch`.
pub fn write_slot(epoch: u32) -> usize {
    (epoch % 2) as usize
}

/// The deposit slot filled during the previous epoch, and therefore the one a
/// downstream stage reads.
pub fn read_slot(epoch: u32) -> usize {
    ((epoch + 1) % 2) as usize
}

/// The eight neighbour offsets, in a fixed order so a direction index is stable
/// across a run.
pub const DIRECTIONS: [(i32, i32); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

/// Control variables for the colony.
#[derive(Clone, Copy, Debug)]
pub struct ColonyKnobs {
    pub width: u16,
    pub height: u16,
    /// Colonies, each a supervision subtree with its own nest.
    pub colonies: u16,
    pub ants_per_colony: u16,
    pub food_sources: u16,
    /// Percent of cells that are impassable.
    pub obstacle_percent: u8,
    /// Pheromone subtracted from every cell each epoch.
    pub decay: u16,
    /// Pheromone laid by one ant in one step.
    pub deposit: u16,
    /// Trail strength below which an ant treats a cell as unmarked.
    pub sense_threshold: u16,
    /// Epochs a `Wait` ant idles for.
    pub wait_ticks: u8,
    /// Epochs the run is expected to last. It sizes each ant's step budget:
    /// an ant resumes once per epoch and a resume costs one step, so a budget
    /// below the run length faults every ant partway through (§10).
    pub epochs: u32,
    pub seed: u64,
}

impl Default for ColonyKnobs {
    fn default() -> Self {
        ColonyKnobs {
            width: 96,
            height: 96,
            colonies: 4,
            ants_per_colony: 64,
            food_sources: 5,
            obstacle_percent: 6,
            decay: 3,
            deposit: 220,
            sense_threshold: 12,
            wait_ticks: 3,
            epochs: 400,
            seed: 0x5EED_A417_C0FF_EE01,
        }
    }
}

impl ColonyKnobs {
    pub fn cells(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn ant_count(&self) -> u32 {
        self.colonies as u32 * self.ants_per_colony as u32
    }

    /// Bytes of one field buffer: two `u16` trail channels per cell.
    pub fn field_bytes(&self) -> usize {
        self.cells() * TRAIL_COUNT * 2
    }

    /// Bytes of one parity slot of a colony summary: a count plus one record
    /// per ant.
    pub fn summary_slot_bytes(&self) -> usize {
        4 + self.ants_per_colony as usize * 8
    }

    /// Bytes of one colony's summary object: two parity slots, for the reason
    /// given on [`DEPOSIT_BYTES`].
    pub fn summary_bytes(&self) -> usize {
        self.summary_slot_bytes() * 2
    }
}

// ---- deterministic generator ---------------------------------------------

/// SplitMix64. Each ant owns one, so no two ants share a stream and no ant
/// depends on when it ran.
pub fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---- terrain --------------------------------------------------------------

/// Immutable world geometry: bounds, nests, food sources, obstacles.
///
/// Written once during setup and never again, so the world stays its sole
/// `WRITE` holder and every ant can hold `READ` on it. It is not `freeze`d: a
/// freeze bumps the object version, and a capability is pinned to the version it
/// was minted against.
pub struct Terrain;

impl Terrain {
    pub const HEADER: usize = 8;

    /// `width | height | food_count | nest_count`, then food sources, then nests,
    /// then the obstacle bitmap.
    pub fn encode(knobs: &ColonyKnobs, nests: &[Nest], food: &[FoodSource]) -> Vec<u8> {
        let mut out = Vec::new();
        put_u16(&mut out, knobs.width);
        put_u16(&mut out, knobs.height);
        put_u16(&mut out, food.len() as u16);
        put_u16(&mut out, nests.len() as u16);
        for (x, y, r) in food {
            put_u16(&mut out, *x);
            put_u16(&mut out, *y);
            put_u16(&mut out, *r);
        }
        for (x, y) in nests {
            put_u16(&mut out, *x);
            put_u16(&mut out, *y);
        }
        out.extend_from_slice(&Self::obstacle_bitmap(knobs));
        out
    }

    fn obstacle_bitmap(knobs: &ColonyKnobs) -> Vec<u8> {
        let cells = knobs.cells();
        let mut bits = vec![0u8; cells.div_ceil(8)];
        if knobs.obstacle_percent == 0 {
            return bits;
        }
        let mut rng = knobs.seed ^ 0xB10C_C0DE;
        for index in 0..cells {
            let roll = (split_mix(&mut rng) % 100) as u8;
            if roll < knobs.obstacle_percent {
                bits[index / 8] |= 1 << (index % 8);
            }
        }
        bits
    }

    fn bitmap_offset(bytes: &[u8]) -> usize {
        let food = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        let nests = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        Self::HEADER + food * 6 + nests * 4
    }

    /// Whether `(x, y)` is impassable. Out-of-bounds counts as impassable, so a
    /// bounds check and an obstacle check are the same test to the caller.
    pub fn is_obstacle(bytes: &[u8], width: u16, height: u16, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return true;
        }
        let index = y as usize * width as usize + x as usize;
        let base = Self::bitmap_offset(bytes);
        match bytes.get(base + index / 8) {
            Some(byte) => byte & (1 << (index % 8)) != 0,
            None => true,
        }
    }

    /// Whether `(x, y)` lies inside a food source.
    pub fn is_food(bytes: &[u8], x: i32, y: i32) -> bool {
        let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        (0..count).any(|i| {
            let at = Self::HEADER + i * 6;
            let fx = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as i32;
            let fy = u16::from_le_bytes([bytes[at + 2], bytes[at + 3]]) as i32;
            let r = u16::from_le_bytes([bytes[at + 4], bytes[at + 5]]) as i32;
            let (dx, dy) = (x - fx, y - fy);
            dx * dx + dy * dy <= r * r
        })
    }

    /// The food sources, for the trace export.
    pub fn food_sources(bytes: &[u8]) -> Vec<FoodSource> {
        let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        (0..count)
            .map(|i| {
                let at = Self::HEADER + i * 6;
                (
                    u16::from_le_bytes([bytes[at], bytes[at + 1]]),
                    u16::from_le_bytes([bytes[at + 2], bytes[at + 3]]),
                    u16::from_le_bytes([bytes[at + 4], bytes[at + 5]]),
                )
            })
            .collect()
    }
}

// ---- field ----------------------------------------------------------------

/// Byte offset of one cell's trail value in a field buffer.
pub fn field_offset(cells: usize, trail: usize, index: usize) -> usize {
    (trail * cells + index) * 2
}

pub fn read_trail(bytes: &[u8], cells: usize, trail: usize, index: usize) -> u16 {
    let at = field_offset(cells, trail, index);
    match (bytes.get(at), bytes.get(at + 1)) {
        (Some(a), Some(b)) => u16::from_le_bytes([*a, *b]),
        _ => 0,
    }
}

pub fn write_trail(bytes: &mut [u8], cells: usize, trail: usize, index: usize, value: u16) {
    let at = field_offset(cells, trail, index);
    if at + 1 < bytes.len() {
        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }
}

// ---- frames ---------------------------------------------------------------

/// One ant's durable state.
///
/// This is the whole ant: it holds no register state and nothing outside the
/// frame, so any executive can resume it at a continuation boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AntFrame {
    pub id: u32,
    pub colony: u16,
    pub x: u16,
    pub y: u16,
    pub home_x: u16,
    pub home_y: u16,
    pub width: u16,
    pub height: u16,
    pub rng: u64,
    pub carrying: u8,
    /// The direction index the ant is currently travelling in.
    ///
    /// An ant with no momentum jitters on the spot: a fresh uniform direction
    /// every epoch is a random walk with a displacement that grows like the
    /// square root of the steps, so a population of them never finds anything.
    /// Carrying the heading and mostly keeping it turns the same generator into
    /// a search that actually covers ground, and it is one byte of frame.
    pub heading: u8,
    pub wait_ticks: u8,
    /// Epochs an ant idles for when it decides to wait.
    pub wait_reset: u8,
    /// Pheromone laid in one step, and the strength below which a cell reads as
    /// unmarked. Carried in the frame because a handler has no access to the
    /// knobs — the frame is the whole of an ant's world.
    pub deposit_amount: u16,
    pub sense_threshold: u16,
    /// Loads delivered home, for the report.
    pub delivered: u16,
    pub deposit: Ref64,
    pub terrain: Ref64,
    pub field_a: Ref64,
    pub field_b: Ref64,
}

impl AntFrame {
    /// The field buffer readable during `epoch`. The world writes the other one.
    pub fn readable_field(&self, epoch: u32) -> Ref64 {
        if epoch.is_multiple_of(2) {
            self.field_a
        } else {
            self.field_b
        }
    }

    pub fn cell(&self) -> usize {
        self.y as usize * self.width as usize + self.x as usize
    }
}

impl Frame for AntFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.id);
        put_u16(out, self.colony);
        put_u16(out, self.x);
        put_u16(out, self.y);
        put_u16(out, self.home_x);
        put_u16(out, self.home_y);
        put_u16(out, self.width);
        put_u16(out, self.height);
        put_u64(out, self.rng);
        put_u8(out, self.carrying);
        put_u8(out, self.heading);
        put_u8(out, self.wait_ticks);
        put_u8(out, self.wait_reset);
        put_u16(out, self.deposit_amount);
        put_u16(out, self.sense_threshold);
        put_u16(out, self.delivered);
        put_ref64(out, self.deposit);
        put_ref64(out, self.terrain);
        put_ref64(out, self.field_a);
        put_ref64(out, self.field_b);
    }

    fn decode(cursor: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(AntFrame {
            id: cursor.u32()?,
            colony: cursor.u16()?,
            x: cursor.u16()?,
            y: cursor.u16()?,
            home_x: cursor.u16()?,
            home_y: cursor.u16()?,
            width: cursor.u16()?,
            height: cursor.u16()?,
            rng: cursor.u64()?,
            carrying: cursor.u8()?,
            heading: cursor.u8()?,
            wait_ticks: cursor.u8()?,
            wait_reset: cursor.u8()?,
            deposit_amount: cursor.u16()?,
            sense_threshold: cursor.u16()?,
            delivered: cursor.u16()?,
            deposit: cursor.ref64()?,
            terrain: cursor.ref64()?,
            field_a: cursor.ref64()?,
            field_b: cursor.ref64()?,
        })
    }
}

/// One colony's aggregation state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColonyFrame {
    pub id: u16,
    pub summary: Ref64,
    /// The deposit objects this colony holds `READ` on. Bounded by
    /// `ants_per_colony`, which is what keeps the colony's capability space
    /// bounded.
    pub deposits: Vec<u64>,
}

impl Frame for ColonyFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_u16(out, self.id);
        put_ref64(out, self.summary);
        put_vec_u64(out, &self.deposits);
    }

    fn decode(cursor: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(ColonyFrame {
            id: cursor.u16()?,
            summary: cursor.ref64()?,
            deposits: cursor.vec_u64()?,
        })
    }
}

/// The world's aggregation state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldFrame {
    pub width: u16,
    pub height: u16,
    pub decay: u16,
    pub field_a: Ref64,
    pub field_b: Ref64,
    pub summaries: Vec<u64>,
}

impl WorldFrame {
    pub fn cells(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// In epoch `n` ants read `n % 2`; the world writes the other buffer, so
    /// the buffer being read is never the buffer being written.
    pub fn buffers(&self, epoch: u32) -> (Ref64, Ref64) {
        if epoch.is_multiple_of(2) {
            (self.field_a, self.field_b)
        } else {
            (self.field_b, self.field_a)
        }
    }
}

impl Frame for WorldFrame {
    fn encode(&self, out: &mut Vec<u8>) {
        put_u16(out, self.width);
        put_u16(out, self.height);
        put_u16(out, self.decay);
        put_ref64(out, self.field_a);
        put_ref64(out, self.field_b);
        put_vec_u64(out, &self.summaries);
    }

    fn decode(cursor: &mut ByteCursor) -> Result<Self, FrameError> {
        Ok(WorldFrame {
            width: cursor.u16()?,
            height: cursor.u16()?,
            decay: cursor.u16()?,
            field_a: cursor.ref64()?,
            field_b: cursor.ref64()?,
            summaries: cursor.vec_u64()?,
        })
    }
}

// ---- deposits -------------------------------------------------------------

/// One ant's report for one epoch.
///
/// The epoch stamp is what makes a dead ant harmless: the colony ignores any
/// record not stamped with the epoch it is aggregating, and an ant that has
/// failed simply stops stamping. Nothing has to notice the death to stop
/// counting it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Deposit {
    pub epoch: u32,
    pub x: u16,
    pub y: u16,
    pub food: u16,
    pub home: u16,
}

impl Deposit {
    pub fn write(&self, bytes: &mut [u8]) {
        if bytes.len() < DEPOSIT_RECORD {
            return;
        }
        bytes[0..4].copy_from_slice(&self.epoch.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.x.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.y.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.food.to_le_bytes());
        bytes[10..12].copy_from_slice(&self.home.to_le_bytes());
    }

    pub fn read(bytes: &[u8]) -> Deposit {
        if bytes.len() < DEPOSIT_RECORD {
            return Deposit::default();
        }
        Deposit {
            epoch: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            x: u16::from_le_bytes([bytes[4], bytes[5]]),
            y: u16::from_le_bytes([bytes[6], bytes[7]]),
            food: u16::from_le_bytes([bytes[8], bytes[9]]),
            home: u16::from_le_bytes([bytes[10], bytes[11]]),
        }
    }
}

// ---- the built colony -----------------------------------------------------

/// One ant, as the harness needs to find it again.
///
/// The continuation is kept because an ant's whole state lives in its frame, so
/// reading a frame is the only way to observe where an ant is. Nothing inside
/// the machine uses these; they are the harness's index into the run.
#[derive(Clone, Copy, Debug)]
pub struct AntHandle {
    pub process: Ref64,
    pub continuation: Ref64,
    pub deposit: Ref64,
}

/// One colony: a supervisor process, the summary it owns, and its ants.
#[derive(Clone, Debug)]
pub struct ColonyHandle {
    pub process: Ref64,
    pub continuation: Ref64,
    pub summary: Ref64,
    pub ants: Vec<AntHandle>,
    pub nest: (u16, u16),
}

/// Everything a consumer needs to read the run back out.
#[derive(Clone, Debug)]
pub struct AntColony {
    pub knobs: ColonyKnobs,
    pub world: Ref64,
    pub world_continuation: Ref64,
    pub terrain: Ref64,
    pub field_a: Ref64,
    pub field_b: Ref64,
    pub colonies: Vec<ColonyHandle>,
}

impl AntColony {
    pub fn ant_count(&self) -> usize {
        self.colonies.iter().map(|c| c.ants.len()).sum()
    }

    /// The field buffer holding what ants read during `epoch`.
    pub fn readable_field(&self, epoch: u32) -> Ref64 {
        if epoch.is_multiple_of(2) {
            self.field_a
        } else {
            self.field_b
        }
    }
}

/// A nest position.
type Nest = (u16, u16);
/// A food source: position and radius.
type FoodSource = (u16, u16, u16);

/// Lay out nests and food sources deterministically, spread across the grid.
fn layout(knobs: &ColonyKnobs) -> (Vec<Nest>, Vec<FoodSource>) {
    let mut rng = knobs.seed ^ 0x1A1D_0002;
    let margin = 6i32;
    let span_x = (knobs.width as i32 - 2 * margin).max(1);
    let span_y = (knobs.height as i32 - 2 * margin).max(1);

    // Nests sit on a lattice rather than a line. A row of colonies works for
    // four and puts a hundred of them two cells apart, which is one nest as far
    // as any ant is concerned.
    let count = knobs.colonies.max(1) as i32;
    let columns = (count as f64).sqrt().ceil() as i32;
    let rows = count.div_euclid(columns) + i32::from(count % columns != 0);
    let step_x = span_x / columns.max(1);
    let step_y = span_y / rows.max(1);

    let nests = (0..knobs.colonies)
        .map(|i| {
            let index = i as i32;
            let (column, row) = (index % columns, index / columns);
            // A deterministic jitter inside the cell, so a large lattice does
            // not read as wallpaper.
            let mut cell_rng = knobs.seed ^ ((index as u64) << 17) ^ 0x4E57_0000u64;
            let jx = (split_mix(&mut cell_rng) % (step_x.max(2) as u64 / 2)) as i32;
            let jy = (split_mix(&mut cell_rng) % (step_y.max(2) as u64 / 2)) as i32;
            let x = margin + column * step_x + step_x / 4 + jx;
            let y = margin + row * step_y + step_y / 4 + jy;
            (
                x.clamp(0, knobs.width as i32 - 1) as u16,
                y.clamp(0, knobs.height as i32 - 1) as u16,
            )
        })
        .collect();

    let food = (0..knobs.food_sources)
        .map(|_| {
            let x = margin + (split_mix(&mut rng) % span_x as u64) as i32;
            let y = margin + (split_mix(&mut rng) % span_y as u64) as i32;
            let r = 3 + (split_mix(&mut rng) % 3) as u16;
            (
                x.clamp(0, knobs.width as i32 - 1) as u16,
                y.clamp(0, knobs.height as i32 - 1) as u16,
                r,
            )
        })
        .collect();

    (nests, food)
}

/// Build the colony into a fresh kernel.
pub fn build(knobs: &ColonyKnobs) -> (Kernel, AntColony) {
    build_in(Kernel::new(), knobs)
}

/// Build the colony into an already-configured kernel, so scheduling mode,
/// cohort width, and retention can be varied without duplicating setup.
pub fn build_in(mut kernel: Kernel, knobs: &ColonyKnobs) -> (Kernel, AntColony) {
    let (nests, food) = layout(knobs);
    let cells = knobs.cells();

    // The world owns the terrain and both field buffers, and is their sole
    // `WRITE` holder for the whole run.
    let world = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let terrain_bytes = Terrain::encode(knobs, &nests, &food);
    let terrain = kernel.create_object(world, ObjectKind::RawBytes, terrain_bytes);
    let field_a = kernel.create_object(world, ObjectKind::RawBytes, vec![0u8; knobs.field_bytes()]);
    let field_b = kernel.create_object(world, ObjectKind::RawBytes, vec![0u8; knobs.field_bytes()]);

    let mut colonies: Vec<ColonyHandle> = Vec::new();
    let mut summaries: Vec<u64> = Vec::new();

    for colony_index in 0..knobs.colonies {
        let nest = nests[colony_index as usize % nests.len()];
        // A colony is supervised by the world, and its ants by the colony. The
        // tree is the same one the capability constraints forced, so failure
        // containment and data flow have the same shape.
        let colony = kernel
            .create_supervised_process_with_policy(
                SYSTEM_PRINCIPAL,
                world,
                ProcessMode::Serial,
                SupervisionPolicy::Notify,
            )
            .expect("the world may supervise a colony");
        let summary = kernel.create_object(
            colony,
            ObjectKind::RawBytes,
            vec![0u8; knobs.summary_bytes()],
        );
        // The world reads the summary; the colony remains its only writer.
        grant_read(&mut kernel, colony, world, summary);
        summaries.push(summary.to_u64());

        let mut ants = Vec::new();
        let mut deposits = Vec::new();
        for ant_index in 0..knobs.ants_per_colony {
            let ant = kernel
                .create_supervised_process_with_policy(
                    SYSTEM_PRINCIPAL,
                    colony,
                    ProcessMode::Serial,
                    SupervisionPolicy::Notify,
                )
                .expect("a colony may supervise an ant");

            let deposit = kernel.create_object(ant, ObjectKind::RawBytes, vec![0u8; DEPOSIT_BYTES]);
            // The colony reads it; the ant remains its only writer.
            grant_read(&mut kernel, ant, colony, deposit);
            deposits.push(deposit.to_u64());

            // The ant reads the world's terrain and both field buffers. Three
            // capabilities, granted once — the ant's space never grows.
            grant_read(&mut kernel, world, ant, terrain);
            grant_read(&mut kernel, world, ant, field_a);
            grant_read(&mut kernel, world, ant, field_b);

            let seed = knobs
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((colony_index as u64) << 32)
                .wrapping_add(ant_index as u64 + 1);
            let mut rng = seed;
            // Ants start scattered around their nest rather than stacked on it,
            // so the first epoch is not a single cell's worth of identical
            // decisions.
            let dx = (split_mix(&mut rng) % 7) as i32 - 3;
            let dy = (split_mix(&mut rng) % 7) as i32 - 3;
            let frame = AntFrame {
                id: colony_index as u32 * knobs.ants_per_colony as u32 + ant_index as u32,
                colony: colony_index,
                x: (nest.0 as i32 + dx).clamp(0, knobs.width as i32 - 1) as u16,
                y: (nest.1 as i32 + dy).clamp(0, knobs.height as i32 - 1) as u16,
                home_x: nest.0,
                home_y: nest.1,
                width: knobs.width,
                height: knobs.height,
                rng,
                carrying: 0,
                heading: (split_mix(&mut rng) % 8) as u8,
                wait_ticks: 0,
                wait_reset: knobs.wait_ticks,
                deposit_amount: knobs.deposit,
                sense_threshold: knobs.sense_threshold,
                delivered: 0,
                deposit,
                terrain,
                field_a,
                field_b,
            };
            let continuation = spawn(&mut kernel, ant, ANT_EXPLORE, &frame, ant_budget(knobs));
            ants.push(AntHandle {
                process: ant,
                continuation,
                deposit,
            });
        }

        let colony_frame = ColonyFrame {
            id: colony_index,
            summary,
            deposits,
        };
        let colony_continuation = spawn(
            &mut kernel,
            colony,
            COLONY_AGGREGATE,
            &colony_frame,
            ant_budget(knobs),
        );

        colonies.push(ColonyHandle {
            process: colony,
            continuation: colony_continuation,
            summary,
            ants,
            nest,
        });
    }

    let world_frame = WorldFrame {
        width: knobs.width,
        height: knobs.height,
        decay: knobs.decay,
        field_a,
        field_b,
        summaries,
    };
    let world_continuation = spawn(
        &mut kernel,
        world,
        WORLD_STEP,
        &world_frame,
        ant_budget(knobs),
    );

    debug_assert_eq!(cells, knobs.cells());

    (
        kernel,
        AntColony {
            knobs: *knobs,
            world,
            world_continuation,
            terrain,
            field_a,
            field_b,
            colonies,
        },
    )
}

/// A resume costs one step and every participant resumes once per epoch, so the
/// budget has to clear the run length or the population faults partway through.
/// The slack covers the epochs the world and colonies spend before and after the
/// ants.
fn ant_budget(knobs: &ColonyKnobs) -> u32 {
    knobs.epochs.saturating_add(64)
}

fn grant_read(kernel: &mut Kernel, owner: Ref64, receiver: Ref64, target: Ref64) {
    let length = kernel
        .object_byte_length(target)
        .expect("granting read on a live object");
    kernel
        .grant_capability(owner, receiver, target, Rights::READ, 0, length)
        .expect("the sole owner may delegate read authority");
}

fn spawn<F: Frame>(
    kernel: &mut Kernel,
    process: Ref64,
    run_class: u32,
    frame: &F,
    max_steps: u32,
) -> Ref64 {
    let mut bytes = Vec::new();
    frame.encode(&mut bytes);
    kernel
        .create_continuation(
            SYSTEM_PRINCIPAL,
            process,
            ContinuationSpec::new(StateAccess::ReadOnly, run_class, 0, bytes, max_steps),
        )
        .expect("the system principal may create a root continuation")
}

/// Run the colony for `epochs`, or until it goes quiet.
pub fn run(knobs: &ColonyKnobs) -> (Kernel, AntColony, u32) {
    let (mut kernel, colony) = build(knobs);
    let mut epochs = 0;
    while epochs < knobs.epochs && kernel.total_pending() > 0 {
        kernel.run_epoch();
        epochs += 1;
    }
    (kernel, colony, epochs)
}

// ---- observation ----------------------------------------------------------

/// One ant as an observer sees it. Decoded from the ant's frame, which is the
/// only place an ant's state exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AntView {
    pub id: u32,
    pub colony: u16,
    pub x: u16,
    pub y: u16,
    pub carrying: bool,
    pub delivered: u16,
    /// The behaviour the ant is about to run — the run class it last named.
    pub run_class: u32,
    pub alive: bool,
}

/// Decode every ant's frame. Used by the trace exporter and by the checks that
/// compare two runs, so both see exactly the state the machine holds.
pub fn observe_ants(kernel: &mut Kernel, colony: &AntColony) -> Vec<AntView> {
    let mut views = Vec::with_capacity(colony.ant_count());
    for handle in colony.colonies.iter().flat_map(|c| c.ants.iter()) {
        let Some((run_class, alive)) = continuation_status(kernel, handle.continuation) else {
            continue;
        };
        let Some(frame) = read_frame::<AntFrame>(kernel, handle.process, handle.continuation)
        else {
            continue;
        };
        views.push(AntView {
            id: frame.id,
            colony: frame.colony,
            x: frame.x,
            y: frame.y,
            carrying: frame.carrying == 1,
            delivered: frame.delivered,
            run_class,
            alive,
        });
    }
    views
}

/// The run class and liveness of a continuation, or `None` if it is gone.
fn continuation_status(kernel: &Kernel, continuation: Ref64) -> Option<(u32, bool)> {
    let descriptor = kernel.continuations().get(continuation).ok()?;
    let alive = matches!(
        descriptor.status,
        crate::abi::continuations::ContinuationState::Runnable
            | crate::abi::continuations::ContinuationState::Waiting
    );
    Some((descriptor.run_class, alive))
}

/// Decode a frame belonging to `process`.
pub fn read_frame<F: Frame>(kernel: &mut Kernel, process: Ref64, continuation: Ref64) -> Option<F> {
    let frame_object = kernel.continuations().get(continuation).ok()?.frame;
    let bytes = kernel.object_bytes(process, frame_object).ok()?.to_vec();
    F::decode(&mut ByteCursor::new(&bytes)).ok()
}

/// Total pheromone in each trail channel of the buffer readable at `epoch`.
///
/// Used as a cheap whole-world checksum: two runs that simulated the same world
/// agree on it, and two that did not almost certainly do not.
pub fn field_totals(kernel: &mut Kernel, colony: &AntColony, epoch: u32) -> (u64, u64) {
    let cells = colony.knobs.cells();
    let field = colony.readable_field(epoch);
    match kernel.object_bytes(colony.world, field) {
        Ok(bytes) => {
            let food = (0..cells)
                .map(|cell| read_trail(bytes, cells, TRAIL_FOOD, cell) as u64)
                .sum();
            let home = (0..cells)
                .map(|cell| read_trail(bytes, cells, TRAIL_HOME, cell) as u64)
                .sum();
            (food, home)
        }
        Err(_) => (0, 0),
    }
}

/// Loads delivered to the nests over the whole run.
pub fn total_delivered(kernel: &mut Kernel, colony: &AntColony) -> u64 {
    observe_ants(kernel, colony)
        .iter()
        .map(|ant| ant.delivered as u64)
        .sum()
}

// ---- predation ------------------------------------------------------------

/// A predator taking a bite out of one colony.
#[derive(Clone, Copy, Debug)]
pub struct PredatorStrike {
    /// Which colony is hit. Every other colony is the control.
    pub colony: u16,
    /// How many of its ants are taken.
    pub victims: u16,
}

/// What the machine did about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredatorOutcome {
    pub struck: Vec<Ref64>,
    /// Ants of the struck colony still running.
    pub survivors: usize,
    /// Ants of every other colony still running. The containment claim is about
    /// this number: it must not move.
    pub bystanders: usize,
    /// Terminal notices queued at the struck colony.
    pub notices: usize,
    /// `ProcessFailed` events the run emitted.
    pub failures: usize,
    /// Whether the struck colony itself survived — `Notify` contains the failure
    /// at the child, so it must.
    pub colony_alive: bool,
    /// Whether the world above it survived.
    pub world_alive: bool,
}

/// Take `strike.victims` ants out of one colony.
///
/// The ants are not deleted behind the machine's back. Their frames are
/// invalidated, so each victim *faults* at its next resume, through the ordinary
/// step path — which is the point. What the demo is showing is not the killing;
/// it is that a fault inside a subtree stops there, that the supervisor is told,
/// and that no ant of any other colony notices.
pub fn inject_predator(
    kernel: &mut Kernel,
    colony: &AntColony,
    strike: PredatorStrike,
) -> Vec<Ref64> {
    let Some(handle) = colony.colonies.get(strike.colony as usize) else {
        return Vec::new();
    };
    let mut struck = Vec::new();
    for ant in handle.ants.iter().take(strike.victims as usize) {
        let Ok(frame_object) = kernel
            .continuations()
            .get(ant.continuation)
            .map(|c| c.frame)
        else {
            continue;
        };
        // A zeroed frame decodes to a zero-width world, which no ant can be in.
        // `ant_step` faults rather than stepping it.
        if let Ok(bytes) = kernel.object_bytes_mut(SYSTEM_PRINCIPAL, frame_object) {
            bytes.iter_mut().for_each(|byte| *byte = 0);
            struck.push(ant.process);
        }
    }
    struck
}

/// Observe the aftermath.
pub fn predator_outcome(
    kernel: &mut Kernel,
    colony: &AntColony,
    strike: PredatorStrike,
    struck: Vec<Ref64>,
) -> PredatorOutcome {
    let ants = observe_ants(kernel, colony);
    let survivors = ants
        .iter()
        .filter(|ant| ant.colony == strike.colony && ant.alive)
        .count();
    let bystanders = ants
        .iter()
        .filter(|ant| ant.colony != strike.colony && ant.alive)
        .count();

    let handle = colony.colonies.get(strike.colony as usize);
    let notices = handle
        .map(|h| kernel.pending_supervision_notices(h.process))
        .unwrap_or(0);
    let colony_alive = handle
        .and_then(|h| kernel.process_state(h.process).ok())
        .is_some_and(|state| {
            !matches!(
                state,
                crate::abi::ProcessState::Failed
                    | crate::abi::ProcessState::Terminated
                    | crate::abi::ProcessState::Cancelled
            )
        });
    let world_alive = kernel.process_state(colony.world).is_ok_and(|state| {
        !matches!(
            state,
            crate::abi::ProcessState::Failed
                | crate::abi::ProcessState::Terminated
                | crate::abi::ProcessState::Cancelled
        )
    });
    let failures = kernel
        .trace_events()
        .iter()
        .filter(|event| event.event_kind == crate::abi::EventKind::ProcessFailed)
        .count();

    PredatorOutcome {
        struck,
        survivors,
        bystanders,
        notices,
        failures,
        colony_alive,
        world_alive,
    }
}

// ---- scheduling comparison (§28.1 on a population instead of a tree) ------

/// One configured run of the colony.
#[derive(Clone, Debug)]
pub struct ColonyRun {
    pub mode: SchedulingMode,
    pub cohort_width: u16,
    pub epochs: u32,
    pub accounting: Accounting,
    /// Everything below is the *world*, not the schedule. Two runs that differ
    /// in any of it did not simulate the same colony, and comparing their
    /// occupancy would be comparing two different workloads.
    pub delivered: u64,
    pub food_trail: u64,
    pub home_trail: u64,
    /// Order-independent digest of every ant's final position and behaviour.
    pub population: u64,
}

impl ColonyRun {
    pub fn lane_occupancy(&self) -> f64 {
        self.accounting.lane_occupancy()
    }

    pub fn dispatches(&self) -> u64 {
        self.accounting.cohorts
    }

    pub fn cohort_fill_ratio(&self) -> f64 {
        self.accounting.cohort_fill_ratio()
    }
}

/// Digest the population so two runs can be compared without depending on the
/// order the ants happen to be visited in.
fn population_digest(ants: &[AntView]) -> u64 {
    let mut digest = 0u64;
    for ant in ants {
        let mut word = (ant.id as u64) << 40
            | (ant.x as u64) << 24
            | (ant.y as u64) << 8
            | (ant.run_class as u64 & 0xFF);
        word ^= (ant.delivered as u64) << 16;
        word ^= (ant.carrying as u64) << 63;
        // Commutative, so the visiting order cannot change the answer.
        digest = digest.wrapping_add(word.rotate_left(ant.id % 61));
    }
    digest
}

/// Run the colony under one scheduling mode.
pub fn run_mode(knobs: &ColonyKnobs, mode: SchedulingMode, cohort_width: u16) -> ColonyRun {
    let mut kernel = Kernel::with_mode(mode);
    kernel.configure_cohorts(cohort_width, PartialCohortPolicy::RunPartial);
    let (mut kernel, colony) = build_in(kernel, knobs);

    let mut epochs = 0;
    while epochs < knobs.epochs && kernel.total_pending() > 0 {
        kernel.run_epoch();
        epochs += 1;
    }

    let (food_trail, home_trail) = field_totals(&mut kernel, &colony, epochs);
    let ants = observe_ants(&mut kernel, &colony);
    ColonyRun {
        mode,
        cohort_width,
        epochs,
        accounting: *kernel.accounting(),
        delivered: ants.iter().map(|ant| ant.delivered as u64).sum(),
        food_trail,
        home_trail,
        population: population_digest(&ants),
    }
}

/// The same colony scheduled two ways.
#[derive(Clone, Debug)]
pub struct ColonyComparison {
    pub fifo: ColonyRun,
    pub cohorted: ColonyRun,
}

impl ColonyComparison {
    /// Useful-lane-occupancy ratio.
    pub fn occupancy_ratio(&self) -> f64 {
        let baseline = self.fifo.lane_occupancy();
        if baseline <= 0.0 {
            return 0.0;
        }
        self.cohorted.lane_occupancy() / baseline
    }

    /// How much of the baseline's dispatch count run-class binning removes.
    pub fn dispatch_reduction(&self) -> f64 {
        let baseline = self.fifo.dispatches();
        if baseline == 0 {
            return 0.0;
        }
        1.0 - (self.cohorted.dispatches() as f64 / baseline as f64)
    }

    /// The control that makes the ratio mean anything: both runs must have
    /// simulated the same colony, step for step and ant for ant. Only the
    /// binning is allowed to differ.
    ///
    /// This is what the parity-slot double buffering in the workload buys. If a
    /// stage read what the previous stage wrote in the same epoch, the two modes
    /// would order the stages differently, the worlds would diverge, and this
    /// would return false.
    pub fn simulated_identical_world(&self) -> bool {
        self.fifo.accounting.steps == self.cohorted.accounting.steps
            && self.fifo.accounting.useful_lane_slots == self.cohorted.accounting.useful_lane_slots
            && self.fifo.epochs == self.cohorted.epochs
            && self.fifo.delivered == self.cohorted.delivered
            && self.fifo.food_trail == self.cohorted.food_trail
            && self.fifo.home_trail == self.cohorted.home_trail
            && self.fifo.population == self.cohorted.population
    }
}

/// Compare the two scheduling modes at one lane width.
pub fn compare(knobs: &ColonyKnobs, cohort_width: u16) -> ColonyComparison {
    ColonyComparison {
        fifo: run_mode(knobs, SchedulingMode::PersistentFifo, cohort_width),
        cohorted: run_mode(knobs, SchedulingMode::RunClassBins, cohort_width),
    }
}

/// A human-readable comparison, including the width-1 null control.
pub fn report(knobs: &ColonyKnobs, cohort_width: u16) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "colonies={} ants={} grid={}x{} epochs={} width={}\n",
        knobs.colonies,
        knobs.ant_count(),
        knobs.width,
        knobs.height,
        knobs.epochs,
        cohort_width
    ));

    let c = compare(knobs, cohort_width);
    for (label, run) in [("persistent-fifo", &c.fifo), ("run-class", &c.cohorted)] {
        s.push_str(&format!(
            "  {label:<16} dispatches={:<8} occupancy={:.3}  fill={:.3}\n",
            run.dispatches(),
            run.lane_occupancy(),
            run.cohort_fill_ratio(),
        ));
    }
    s.push_str(&format!(
        "  occupancy ratio {:.2}x, dispatch reduction {:.1}%\n",
        c.occupancy_ratio(),
        c.dispatch_reduction() * 100.0
    ));
    s.push_str(&format!(
        "  same world under both schedules: {}  (delivered={} food_trail={})\n",
        if c.simulated_identical_world() {
            "yes"
        } else {
            "NO — the comparison is invalid"
        },
        c.cohorted.delivered,
        c.cohorted.food_trail,
    ));

    // The null control. At width 1 every dispatch is a single lane, so how
    // continuations were binned cannot change occupancy. A ratio other than
    // 1.00 here would mean the harness is measuring itself.
    let null = compare(knobs, 1);
    s.push_str(&format!(
        "  null control at width 1: {:.2}x  [{}]\n",
        null.occupancy_ratio(),
        if (null.occupancy_ratio() - 1.0).abs() < 1e-9 {
            "no effect, as required"
        } else {
            "UNEXPECTED — binning must not matter at width 1"
        }
    ));
    s
}
