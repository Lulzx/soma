//! Ant-colony handlers: the resume points behind the run classes in
//! `compiler::run_classes`, executed by the scalar interpreter.
//!
//! There are three kinds of participant and they form the pipeline the workload
//! module explains: ants write their own deposit, colonies fold their ants'
//! deposits into a summary, the world folds the summaries into the pheromone
//! field. Each stage reads the parity slot the stage before it filled *last*
//! epoch, so no stage depends on which ran first within an epoch — which is what
//! lets the same population be scheduled two different ways and still simulate
//! the same world.
//!
//! Every handler is one bounded segment ending in a `yield`, and the run class
//! it yields to is the ant's own decision about what it is doing next. That is
//! the only thing the scheduler ever learns about an ant.

use crate::abi::{Ref64, StepResult};
use crate::compiler::run_classes::{
    ANT_AVOID_OBSTACLE, ANT_CARRY_FOOD, ANT_EXPLORE, ANT_FOLLOW_TRAIL, ANT_RETURN_HOME, ANT_WAIT,
    COLONY_AGGREGATE, WORLD_STEP,
};
use crate::executives::cpu_scalar::{load_frame, store_frame};
use crate::experiments::ant_colony::{
    read_trail, split_mix, write_slot, write_trail, AntFrame, ColonyFrame, Deposit, Terrain,
    WorldFrame, DEPOSIT_RECORD, DIRECTIONS, TRAIL_FOOD, TRAIL_HOME,
};
use crate::executives::lane::LaneView;

/// What an ant can see from where it stands. Gathered once per step so the
/// decision below is a pure function of it.
struct Senses {
    /// Whether each of the eight neighbours can be entered.
    passable: [bool; 8],
    /// Food-trail strength at each neighbour.
    food_trail: [u16; 8],
    /// Home-trail strength at each neighbour.
    home_trail: [u16; 8],
    /// Whether the ant is standing on a food source.
    on_food: bool,
}

impl Senses {
    fn any_passable(&self) -> bool {
        self.passable.iter().any(|p| *p)
    }
}

/// The strongest neighbour on one trail, if it clears `threshold`.
///
/// The scan starts at a generator-chosen offset so that two ants standing on
/// equal readings do not both pick direction zero. Without it the population
/// falls into lockstep and the run-class histogram stops being interesting for
/// reasons that have nothing to do with scheduling.
fn best_direction(
    scores: &[u16; 8],
    passable: &[bool; 8],
    threshold: u16,
    rng: &mut u64,
) -> Option<usize> {
    let start = (split_mix(rng) % 8) as usize;
    let mut best: Option<(usize, u16)> = None;
    for offset in 0..8 {
        let index = (start + offset) % 8;
        if !passable[index] || scores[index] <= threshold {
            continue;
        }
        if best.is_none_or(|(_, value)| scores[index] > value) {
            best = Some((index, scores[index]));
        }
    }
    best.map(|(index, _)| index)
}

/// A passable direction chosen at random, if there is one.
fn random_direction(passable: &[bool; 8], rng: &mut u64) -> Option<usize> {
    let start = (split_mix(rng) % 8) as usize;
    (0..8)
        .map(|offset| (start + offset) % 8)
        .find(|index| passable[*index])
}

/// The neighbour that most nearly points at `(dx, dy)`.
fn direction_towards(dx: i32, dy: i32) -> usize {
    let want = (dx.signum(), dy.signum());
    DIRECTIONS
        .iter()
        .position(|d| *d == want)
        .unwrap_or_else(|| {
            // Straight along the dominant axis when the sign pair is (0, 0).
            if dx.abs() >= dy.abs() {
                if dx >= 0 {
                    2
                } else {
                    6
                }
            } else if dy >= 0 {
                4
            } else {
                0
            }
        })
}

/// Prefer the trail, fall back to heading for the nest.
fn homeward(senses: &Senses, frame: &AntFrame, rng: &mut u64) -> Option<usize> {
    let dx = frame.home_x as i32 - frame.x as i32;
    let dy = frame.home_y as i32 - frame.y as i32;
    let direct = direction_towards(dx, dy);
    if senses.passable[direct] {
        return Some(direct);
    }
    best_direction(
        &senses.home_trail,
        &senses.passable,
        frame.sense_threshold,
        rng,
    )
    .or_else(|| random_direction(&senses.passable, rng))
}

fn at_home(frame: &AntFrame) -> bool {
    let dx = (frame.home_x as i32 - frame.x as i32).abs();
    let dy = (frame.home_y as i32 - frame.y as i32).abs();
    dx <= 1 && dy <= 1
}

/// Read the terrain and the readable field buffer into a `Senses`.
///
/// Two borrows, taken one after the other: `object_bytes` borrows the lane,
/// so the values are copied out of each before the next call. Only the eight
/// neighbours are read — the field is tens of kilobytes and an ant has no reason
/// to touch more of it than it can see.
fn sense(lane: &mut LaneView<'_>, process: Ref64, frame: &AntFrame, epoch: u32) -> Option<Senses> {
    let width = frame.width;
    let height = frame.height;
    let cells = width as usize * height as usize;
    let (x, y) = (frame.x as i32, frame.y as i32);

    let (passable, on_food) = {
        let bytes = lane.object_bytes(process, frame.terrain).ok()?;
        let mut passable = [false; 8];
        for (index, (dx, dy)) in DIRECTIONS.iter().enumerate() {
            passable[index] = !Terrain::is_obstacle(bytes, width, height, x + dx, y + dy);
        }
        (passable, Terrain::is_food(bytes, x, y))
    };

    let (food_trail, home_trail) = {
        let field = frame.readable_field(epoch);
        let bytes = lane.object_bytes(process, field).ok()?;
        let mut food = [0u16; 8];
        let mut home = [0u16; 8];
        for (index, (dx, dy)) in DIRECTIONS.iter().enumerate() {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let cell = ny as usize * width as usize + nx as usize;
            food[index] = read_trail(bytes, cells, TRAIL_FOOD, cell);
            home[index] = read_trail(bytes, cells, TRAIL_HOME, cell);
        }
        (food, home)
    };

    Some(Senses {
        passable,
        food_trail,
        home_trail,
        on_food,
    })
}

/// One ant's step. `behaviour` is the index within the ant block, so this is
/// the same switch the scheduler binned on — the run class is both.
pub fn ant_step(
    lane: &mut LaneView<'_>,
    _cont: Ref64,
    process: Ref64,
    behaviour: u32,
) -> StepResult {
    let epoch = lane.epoch_number();
    let run_class = ANT_EXPLORE + behaviour;
    let mut frame: AntFrame = match load_frame_ant(lane, process) {
        Some(frame) => frame,
        None => return StepResult::fault(process, run_class),
    };

    let Some(senses) = sense(lane, process, &frame, epoch) else {
        return StepResult::fault(process, run_class);
    };

    let mut rng = frame.rng;
    let next;
    let mut step: Option<usize> = None;

    match run_class {
        // Idle for a bounded number of epochs, then rejoin the search. `Wait`
        // exists so the population is not uniformly busy: a class that empties
        // and refills is what makes the bin histogram move.
        ANT_WAIT => {
            if frame.wait_ticks > 0 {
                frame.wait_ticks -= 1;
                next = ANT_WAIT;
            } else {
                next = ANT_EXPLORE;
            }
        }

        // Blocked last step. Pick a fresh heading, then resume whatever the ant
        // was actually doing.
        ANT_AVOID_OBSTACLE => {
            step = random_direction(&senses.passable, &mut rng);
            next = if frame.carrying == 1 {
                ANT_CARRY_FOOD
            } else if best_direction(
                &senses.food_trail,
                &senses.passable,
                frame.sense_threshold,
                &mut rng,
            )
            .is_some()
            {
                ANT_FOLLOW_TRAIL
            } else {
                ANT_EXPLORE
            };
        }

        ANT_EXPLORE => {
            if senses.on_food && frame.carrying == 0 {
                frame.carrying = 1;
                next = ANT_CARRY_FOOD;
            } else if best_direction(
                &senses.food_trail,
                &senses.passable,
                frame.sense_threshold,
                &mut rng,
            )
            .is_some()
            {
                next = ANT_FOLLOW_TRAIL;
            } else if !senses.any_passable() {
                next = ANT_AVOID_OBSTACLE;
            } else {
                // Occasionally stop. A bounded, self-chosen pause, which is the
                // cheapest way to keep readiness irregular without inventing an
                // external event to wait on.
                if split_mix(&mut rng).is_multiple_of(64) {
                    frame.wait_ticks = frame.wait_reset;
                    next = ANT_WAIT;
                } else {
                    // Keep going, mostly. Three steps in four continue the
                    // current heading, which is what turns the walk from a
                    // jitter into a search.
                    let heading = if !split_mix(&mut rng).is_multiple_of(4)
                        && senses.passable[frame.heading as usize % 8]
                    {
                        frame.heading as usize % 8
                    } else {
                        (split_mix(&mut rng) % 8) as usize
                    };
                    if senses.passable[heading] {
                        step = Some(heading);
                        next = ANT_EXPLORE;
                    } else {
                        next = ANT_AVOID_OBSTACLE;
                    }
                }
            }
        }

        ANT_FOLLOW_TRAIL => {
            if senses.on_food && frame.carrying == 0 {
                frame.carrying = 1;
                next = ANT_CARRY_FOOD;
            } else {
                // The cell an ant just came from carries the trail it just
                // walked, so an unconstrained gradient climb walks back and
                // forth forever. Excluding the reverse of the current heading
                // is what makes trail-following go somewhere.
                let mut allowed = senses.passable;
                allowed[(frame.heading as usize + 4) % 8] = false;
                match best_direction(
                    &senses.food_trail,
                    &allowed,
                    frame.sense_threshold,
                    &mut rng,
                ) {
                    Some(heading) => {
                        step = Some(heading);
                        next = ANT_FOLLOW_TRAIL;
                    }
                    // The trail evaporated. Back to searching.
                    None => next = ANT_EXPLORE,
                }
            }
        }

        ANT_CARRY_FOOD => {
            if at_home(&frame) {
                frame.carrying = 0;
                frame.delivered = frame.delivered.saturating_add(1);
                next = ANT_EXPLORE;
            } else {
                match homeward(&senses, &frame, &mut rng) {
                    Some(heading) => {
                        step = Some(heading);
                        next = ANT_CARRY_FOOD;
                    }
                    None => next = ANT_AVOID_OBSTACLE,
                }
            }
        }

        ANT_RETURN_HOME => {
            if at_home(&frame) {
                next = ANT_EXPLORE;
            } else {
                match homeward(&senses, &frame, &mut rng) {
                    Some(heading) => {
                        step = Some(heading);
                        next = ANT_RETURN_HOME;
                    }
                    None => next = ANT_AVOID_OBSTACLE,
                }
            }
        }

        _ => return StepResult::fault(process, run_class),
    }

    if let Some(heading) = step {
        frame.heading = heading as u8;
        let (dx, dy) = DIRECTIONS[heading];
        frame.x = (frame.x as i32 + dx).clamp(0, frame.width as i32 - 1) as u16;
        frame.y = (frame.y as i32 + dy).clamp(0, frame.height as i32 - 1) as u16;
    }
    frame.rng = rng;

    // An outbound ant lays the trail home; a laden ant lays the trail to food.
    // The two gradients are what let the other behaviours be simple.
    let deposit = Deposit {
        epoch,
        x: frame.x,
        y: frame.y,
        food: if frame.carrying == 1 {
            frame.deposit_amount
        } else {
            0
        },
        home: if frame.carrying == 0 {
            frame.deposit_amount
        } else {
            0
        },
    };
    let slot = write_slot(epoch) * DEPOSIT_RECORD;
    if let Ok(bytes) = lane.object_bytes_mut(process, frame.deposit) {
        if bytes.len() >= slot + DEPOSIT_RECORD {
            deposit.write(&mut bytes[slot..slot + DEPOSIT_RECORD]);
        }
    }

    store_frame(lane, process, &frame);
    StepResult::yield_next(next)
}

fn load_frame_ant(lane: &mut LaneView<'_>, process: Ref64) -> Option<AntFrame> {
    let fallback = AntFrame {
        id: u32::MAX,
        colony: 0,
        x: 0,
        y: 0,
        home_x: 0,
        home_y: 0,
        width: 0,
        height: 0,
        rng: 0,
        carrying: 0,
        heading: 0,
        wait_ticks: 0,
        wait_reset: 0,
        deposit_amount: 0,
        sense_threshold: 0,
        delivered: 0,
        deposit: Ref64::NULL,
        terrain: Ref64::NULL,
        field_a: Ref64::NULL,
        field_b: Ref64::NULL,
    };
    let frame: AntFrame = load_frame(lane, process, fallback);
    // A frame that failed to decode comes back as the sentinel, and a zero-sized
    // world is not a world. Faulting is better than stepping an ant that does
    // not exist.
    (frame.id != u32::MAX && frame.width > 0).then_some(frame)
}

/// Fold this colony's ants' deposits into the colony summary.
///
/// Reads the parity slot the ants filled last epoch and writes this epoch's
/// slot, so it neither races the ants below it nor the world above it. An ant
/// that has failed simply stops stamping its record, and a stale stamp is
/// skipped — nothing has to be told about the death.
pub fn colony_aggregate(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64) -> StepResult {
    let epoch = lane.epoch_number();
    let frame: ColonyFrame = load_frame(
        lane,
        process,
        ColonyFrame {
            id: 0,
            summary: Ref64::NULL,
            deposits: Vec::new(),
        },
    );
    if frame.summary.is_null() {
        return StepResult::fault(process, COLONY_AGGREGATE);
    }

    let want = epoch.wrapping_sub(1);
    let read_at = write_slot(want) * DEPOSIT_RECORD;
    let mut records: Vec<Deposit> = Vec::with_capacity(frame.deposits.len());
    for raw in &frame.deposits {
        let deposit = Ref64::from_u64(*raw);
        let Ok(bytes) = lane.object_bytes(process, deposit) else {
            continue;
        };
        if bytes.len() < read_at + DEPOSIT_RECORD {
            continue;
        }
        let record = Deposit::read(&bytes[read_at..read_at + DEPOSIT_RECORD]);
        // Epoch zero writes nothing, and a dead ant's slot keeps whatever it
        // last wrote. The stamp is what tells them apart.
        if record.epoch == want && (record.food > 0 || record.home > 0) {
            records.push(record);
        }
    }

    let slot_bytes = 4 + frame.deposits.len() * 8;
    let write_at = write_slot(epoch) * slot_bytes;
    if let Ok(bytes) = lane.object_bytes_mut(process, frame.summary) {
        if bytes.len() >= write_at + slot_bytes {
            let slot = &mut bytes[write_at..write_at + slot_bytes];
            let count = records.len().min(frame.deposits.len());
            slot[0..4].copy_from_slice(&(count as u32).to_le_bytes());
            for (index, record) in records.iter().take(count).enumerate() {
                let at = 4 + index * 8;
                slot[at..at + 2].copy_from_slice(&record.x.to_le_bytes());
                slot[at + 2..at + 4].copy_from_slice(&record.y.to_le_bytes());
                slot[at + 4..at + 6].copy_from_slice(&record.food.to_le_bytes());
                slot[at + 6..at + 8].copy_from_slice(&record.home.to_le_bytes());
            }
        }
    }

    store_frame(lane, process, &frame);
    StepResult::yield_next(COLONY_AGGREGATE)
}

/// Advance the pheromone field by one epoch.
///
/// Copy the readable buffer forward, decay it, add the colonies' deposits, and
/// leave the result in the buffer the ants will read next epoch. The world is
/// the sole `WRITE` holder of both buffers throughout, so nothing here needs a
/// lock and nothing needs to be frozen.
pub fn world_step(lane: &mut LaneView<'_>, _cont: Ref64, process: Ref64) -> StepResult {
    let epoch = lane.epoch_number();
    let frame: WorldFrame = load_frame(
        lane,
        process,
        WorldFrame {
            width: 0,
            height: 0,
            decay: 0,
            field_a: Ref64::NULL,
            field_b: Ref64::NULL,
            summaries: Vec::new(),
        },
    );
    if frame.width == 0 {
        return StepResult::fault(process, WORLD_STEP);
    }
    let cells = frame.cells();
    let (readable, writable) = frame.buffers(epoch);

    // Collect the colonies' summaries before touching the field: the field
    // borrow and the summary borrows cannot be held at once.
    let want = epoch.wrapping_sub(1);
    let mut deposits: Vec<(u16, u16, u16, u16)> = Vec::new();
    for raw in &frame.summaries {
        let summary = Ref64::from_u64(*raw);
        let Ok(bytes) = lane.object_bytes(process, summary) else {
            continue;
        };
        let slot_bytes = bytes.len() / 2;
        let read_at = write_slot(want) * slot_bytes;
        if slot_bytes < 4 || bytes.len() < read_at + slot_bytes {
            continue;
        }
        let slot = &bytes[read_at..read_at + slot_bytes];
        let count = u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]) as usize;
        let capacity = (slot_bytes - 4) / 8;
        for index in 0..count.min(capacity) {
            let at = 4 + index * 8;
            deposits.push((
                u16::from_le_bytes([slot[at], slot[at + 1]]),
                u16::from_le_bytes([slot[at + 2], slot[at + 3]]),
                u16::from_le_bytes([slot[at + 4], slot[at + 5]]),
                u16::from_le_bytes([slot[at + 6], slot[at + 7]]),
            ));
        }
    }

    let carried: Vec<u8> = match lane.object_bytes(process, readable) {
        Ok(bytes) => bytes.to_vec(),
        Err(_) => return StepResult::fault(process, WORLD_STEP),
    };

    let decay = frame.decay;
    let width = frame.width as usize;
    if let Ok(bytes) = lane.object_bytes_mut(process, writable) {
        bytes.copy_from_slice(&carried);
        for trail in [TRAIL_FOOD, TRAIL_HOME] {
            for cell in 0..cells {
                let value = read_trail(bytes, cells, trail, cell);
                write_trail(bytes, cells, trail, cell, value.saturating_sub(decay));
            }
        }
        for (x, y, food, home) in deposits {
            let cell = y as usize * width + x as usize;
            if cell >= cells {
                continue;
            }
            if food > 0 {
                let value = read_trail(bytes, cells, TRAIL_FOOD, cell);
                write_trail(bytes, cells, TRAIL_FOOD, cell, value.saturating_add(food));
            }
            if home > 0 {
                let value = read_trail(bytes, cells, TRAIL_HOME, cell);
                write_trail(bytes, cells, TRAIL_HOME, cell, value.saturating_add(home));
            }
        }
    }

    store_frame(lane, process, &frame);
    StepResult::yield_next(WORLD_STEP)
}
