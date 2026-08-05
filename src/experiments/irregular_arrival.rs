//! Irregular arrival: the experiment that decides the thesis (§25.1, §27).
//!
//! The bulk frontier baseline ties SOMA on level-synchronous work, because when
//! readiness arrives in neat levels a host can sort each level by run class and
//! get the same groups SOMA forms dynamically. The thesis therefore rests on the
//! case where readiness does *not* arrive in levels: roots trickle in over time
//! (§25.1's arrival rate) and siblings become ready at different moments because
//! their heuristics resolve at different times (future latency).
//!
//! # Methodology
//!
//! This is a **trace-driven policy comparison**, not a causal simulation. An
//! arrival trace is generated once from the search tree — each node tagged with
//! the tick it becomes ready and the run class it belongs to — and both dispatch
//! policies are then scored against that identical trace. This is deliberate:
//! if each system generated its own readiness stream, each would be measured on
//! a different workload and the comparison would be worthless. Fixing the
//! arrivals and varying only the dispatch policy is the standard way to compare
//! schedulers, and it is the only way to make this particular claim honestly.
//!
//! The consequence is that a node's ready tick does not depend on when its
//! parent actually ran. The trace is an input, not a causal history.
//!
//! # The hypothesis under test
//!
//! Both policies face the same tension: dispatch now with idle lanes, or wait to
//! accumulate a fuller cohort. The claim is that they resolve it at different
//! granularities.
//!
//! * The bulk frontier launches from the host, so its accumulation window is
//!   **global** — every run class waits for the same barrier, and the window is
//!   set by whichever class fills slowest.
//! * SOMA holds partial cohorts **per run class**, so a class that has filled
//!   dispatches immediately while a sparse class keeps accumulating.
//!
//! If that distinction is real, SOMA reaches a given lane occupancy at lower
//! waiting time, and the two occupancy/latency frontiers separate. If the
//! frontiers coincide, the distinction does not pay, and §29's Outcome D applies
//! to irregular work too.

use std::collections::{BTreeMap, VecDeque};

use crate::compiler::run_classes::{search_class, SEARCH_BRANCH};
use crate::compiler::state_machine_lowering::search_step;
use crate::scheduler::cohorts::{dispatch_cost, DispatchCost};

/// One continuation becoming ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arrival {
    /// The tick at which this work becomes runnable.
    pub tick: u32,
    pub run_class: u32,
    /// Stable identity, used to keep ordering deterministic.
    pub id: u64,
}

/// A fixed readiness stream that every policy is scored against.
#[derive(Clone, Debug, Default)]
pub struct ArrivalTrace {
    /// Sorted by `(tick, id)`.
    pub events: Vec<Arrival>,
}

impl ArrivalTrace {
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The last tick on which anything becomes ready.
    pub fn horizon(&self) -> u32 {
        self.events.last().map(|e| e.tick).unwrap_or(0)
    }

    /// How many distinct run classes appear.
    pub fn distinct_classes(&self) -> usize {
        let mut v: Vec<u32> = self.events.iter().map(|e| e.run_class).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Mean arrivals per tick over the trace's span — the load the policies see.
    pub fn arrival_rate(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        self.events.len() as f64 / (self.horizon() as f64 + 1.0)
    }
}

/// Workload shape for the irregular regime.
#[derive(Clone, Copy, Debug)]
pub struct IrregularKnobs {
    pub roots: u32,
    /// Ticks over which the roots arrive. 0 means all at once, which recovers
    /// the level-synchronous regime.
    pub arrival_span: u32,
    pub depth: u32,
    pub branching_factor: u32,
    pub class_count: u32,
    /// Spread of per-node readiness jitter — §25.1's future latency. 0 means
    /// siblings become ready together, which is again level-synchronous.
    pub jitter: u32,
    pub arithmetic_ops: u32,
}

impl Default for IrregularKnobs {
    fn default() -> Self {
        IrregularKnobs {
            roots: 8,
            arrival_span: 12,
            depth: 4,
            branching_factor: 3,
            class_count: 4,
            jitter: 3,
            arithmetic_ops: 16,
        }
    }
}

/// Generate the arrival trace for a search tree.
///
/// Roots are spread across `arrival_span`; a child becomes ready one tick after
/// its parent plus a deterministic jitter derived from its own value, modelling
/// a heuristic future that resolves after a variable delay.
pub fn trace(knobs: &IrregularKnobs) -> ArrivalTrace {
    let mut events = Vec::new();
    let mut queue: VecDeque<(u64, u32, u32)> = VecDeque::new();
    let mut next_id = 0u64;

    for root in 0..knobs.roots {
        let tick = if knobs.roots <= 1 {
            0
        } else {
            root * knobs.arrival_span / knobs.roots
        };
        queue.push_back((root as u64 + 1, knobs.depth, tick));
    }

    while let Some((value, depth, ready)) = queue.pop_front() {
        let run_class = search_class(value, knobs.class_count);
        events.push(Arrival {
            tick: ready,
            run_class,
            id: next_id,
        });
        next_id += 1;

        if depth > 0 {
            let stepped = search_step(value, knobs.arithmetic_ops, run_class - SEARCH_BRANCH);
            for i in 0..knobs.branching_factor {
                let child = stepped.wrapping_add(i as u64);
                let jitter = if knobs.jitter == 0 {
                    0
                } else {
                    (child % (knobs.jitter as u64 + 1)) as u32
                };
                queue.push_back((child, depth - 1, ready + 1 + jitter));
            }
        }
    }

    events.sort_by_key(|e| (e.tick, e.id));
    ArrivalTrace { events }
}

/// What a dispatch policy achieved on a trace.
#[derive(Clone, Debug, Default)]
pub struct PolicyOutcome {
    pub cost: DispatchCost,
    /// Host round-trips required (§28.4).
    pub host_launches: u64,
    /// Per-item wait, in ticks, between becoming ready and being dispatched.
    pub waits: Vec<u32>,
    /// The tick on which the last item was dispatched.
    pub makespan: u32,
}

impl PolicyOutcome {
    pub fn occupancy(&self) -> f64 {
        self.cost.occupancy()
    }

    pub fn dispatches(&self) -> u64 {
        self.cost.dispatches
    }

    pub fn items(&self) -> usize {
        self.waits.len()
    }

    fn percentile(&self, p: f64) -> u32 {
        if self.waits.is_empty() {
            return 0;
        }
        let mut sorted = self.waits.clone();
        sorted.sort_unstable();
        let rank = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[rank.min(sorted.len() - 1)]
    }

    /// Median wait between readiness and dispatch (§27).
    pub fn p50_wait(&self) -> u32 {
        self.percentile(0.50)
    }

    /// Tail wait (§27).
    pub fn p99_wait(&self) -> u32 {
        self.percentile(0.99)
    }

    pub fn mean_wait(&self) -> f64 {
        if self.waits.is_empty() {
            return 0.0;
        }
        self.waits.iter().map(|w| *w as f64).sum::<f64>() / self.waits.len() as f64
    }
}

/// Group a trace's arrivals by tick, so a policy can consume them in order.
fn by_tick(trace: &ArrivalTrace) -> BTreeMap<u32, Vec<Arrival>> {
    let mut m: BTreeMap<u32, Vec<Arrival>> = BTreeMap::new();
    for e in &trace.events {
        m.entry(e.tick).or_default().push(*e);
    }
    m
}

fn accumulate(cost: &mut DispatchCost, add: DispatchCost) {
    cost.dispatches += add.dispatches;
    cost.lane_slots += add.lane_slots;
    cost.useful_lane_slots += add.useful_lane_slots;
    cost.full_dispatches += add.full_dispatches;
}

/// SOMA's policy: per-run-class bins that dispatch a full cohort the moment one
/// exists, and hold a partial remainder for up to `max_defer` ticks before
/// running it anyway.
///
/// The waiting is per class. A class that has filled dispatches immediately even
/// while another class is still accumulating — there is no global barrier.
pub fn soma_policy(trace: &ArrivalTrace, width: u16, max_defer: u32) -> PolicyOutcome {
    let mut out = PolicyOutcome::default();
    // run class -> queued (ready tick, id)
    let mut bins: BTreeMap<u32, VecDeque<Arrival>> = BTreeMap::new();
    let ticks = by_tick(trace);
    let last_tick = trace.horizon();
    let w = width.max(1) as usize;

    for tick in 0..=last_tick {
        if let Some(arrivals) = ticks.get(&tick) {
            for a in arrivals {
                bins.entry(a.run_class).or_default().push_back(*a);
            }
        }

        for (run_class, queue) in bins.iter_mut() {
            // Any full cohort goes out now, regardless of what other classes
            // are doing.
            while queue.len() >= w {
                let batch: Vec<Arrival> = queue.drain(..w).collect();
                accumulate(&mut out.cost, dispatch_cost(&vec![*run_class; w], width));
                for a in &batch {
                    out.waits.push(tick - a.tick);
                }
                out.makespan = out.makespan.max(tick);
            }

            // A remainder that has waited long enough runs partial rather than
            // starving. `max_defer` is this policy's analogue of the bulk
            // window, but it is applied per class.
            if let Some(oldest) = queue.front() {
                if tick - oldest.tick >= max_defer || tick == last_tick {
                    let n = queue.len();
                    let batch: Vec<Arrival> = queue.drain(..).collect();
                    accumulate(&mut out.cost, dispatch_cost(&vec![*run_class; n], width));
                    for a in &batch {
                        out.waits.push(tick - a.tick);
                    }
                    out.makespan = out.makespan.max(tick);
                }
            }
        }
    }

    // Anything still queued after the horizon drains on the final tick.
    for (run_class, queue) in bins.iter_mut() {
        if queue.is_empty() {
            continue;
        }
        let n = queue.len();
        let batch: Vec<Arrival> = queue.drain(..).collect();
        accumulate(&mut out.cost, dispatch_cost(&vec![*run_class; n], width));
        for a in &batch {
            out.waits.push(last_tick - a.tick);
        }
        out.makespan = out.makespan.max(last_tick);
    }

    out
}

/// The bulk frontier's policy: the host launches every `window` ticks over
/// everything that has accumulated, partitioning that batch by run class.
///
/// The window is global. Every class waits for the same launch, so the window
/// that suits a sparse class also delays a busy one.
pub fn bulk_policy(trace: &ArrivalTrace, width: u16, window: u32) -> PolicyOutcome {
    let mut out = PolicyOutcome::default();
    let mut pending: Vec<Arrival> = Vec::new();
    let ticks = by_tick(trace);
    let last_tick = trace.horizon();
    let window = window.max(1);

    for tick in 0..=last_tick {
        if let Some(arrivals) = ticks.get(&tick) {
            pending.extend_from_slice(arrivals);
        }

        let due = (tick + 1) % window == 0 || tick == last_tick;
        if !due || pending.is_empty() {
            continue;
        }

        // One host launch, segmented by run class — the strong manual variant.
        out.host_launches += 1;
        let mut classes: Vec<u32> = pending.iter().map(|a| a.run_class).collect();
        classes.sort_unstable();
        classes.dedup();
        for run_class in classes {
            let n = pending.iter().filter(|a| a.run_class == run_class).count();
            accumulate(&mut out.cost, dispatch_cost(&vec![run_class; n], width));
        }
        for a in &pending {
            out.waits.push(tick - a.tick);
        }
        out.makespan = out.makespan.max(tick);
        pending.clear();
    }

    out
}

/// One point on an occupancy/latency frontier.
#[derive(Clone, Copy, Debug)]
pub struct FrontierPoint {
    /// The policy's waiting knob: `max_defer` for SOMA, `window` for bulk.
    pub knob: u32,
    pub occupancy: f64,
    pub mean_wait: f64,
    pub p50_wait: u32,
    pub p99_wait: u32,
    pub dispatches: u64,
    pub host_launches: u64,
}

fn point(knob: u32, o: &PolicyOutcome) -> FrontierPoint {
    FrontierPoint {
        knob,
        occupancy: o.occupancy(),
        mean_wait: o.mean_wait(),
        p50_wait: o.p50_wait(),
        p99_wait: o.p99_wait(),
        dispatches: o.dispatches(),
        host_launches: o.host_launches,
    }
}

/// Sweep both policies' waiting knobs to trace their occupancy/latency frontiers.
#[derive(Clone, Debug)]
pub struct Frontiers {
    pub soma: Vec<FrontierPoint>,
    pub bulk: Vec<FrontierPoint>,
}

impl Frontiers {
    /// The best occupancy either policy reaches without exceeding `budget`
    /// mean ticks of waiting.
    fn best_within(points: &[FrontierPoint], budget: f64) -> Option<FrontierPoint> {
        points
            .iter()
            .filter(|p| p.mean_wait <= budget + 1e-9)
            .max_by(|a, b| a.occupancy.total_cmp(&b.occupancy))
            .copied()
    }

    pub fn soma_at(&self, budget: f64) -> Option<FrontierPoint> {
        Self::best_within(&self.soma, budget)
    }

    pub fn bulk_at(&self, budget: f64) -> Option<FrontierPoint> {
        Self::best_within(&self.bulk, budget)
    }

    /// Occupancy advantage at a matched latency budget. Above 1.0 means SOMA
    /// reaches higher occupancy than the manual batch for the same waiting.
    pub fn advantage_at(&self, budget: f64) -> Option<f64> {
        match (self.soma_at(budget), self.bulk_at(budget)) {
            (Some(s), Some(b)) if b.occupancy > 0.0 => Some(s.occupancy / b.occupancy),
            _ => None,
        }
    }
}

/// Trace both frontiers over a range of waiting knobs.
pub fn frontiers(trace: &ArrivalTrace, width: u16, knobs: &[u32]) -> Frontiers {
    Frontiers {
        soma: knobs
            .iter()
            .map(|k| point(*k, &soma_policy(trace, width, *k)))
            .collect(),
        bulk: knobs
            .iter()
            .map(|k| point(*k, &bulk_policy(trace, width, (*k).max(1))))
            .collect(),
    }
}

/// One cell of the regime map (§25: "this maps the regime in which cohorting
/// helps or fails").
#[derive(Clone, Copy, Debug)]
pub struct RegimePoint {
    pub arrival_span: u32,
    pub jitter: u32,
    pub class_count: u32,
    /// Occupancy advantage at a matched mean-wait budget.
    pub advantage: f64,
    pub soma_occupancy: f64,
    pub bulk_occupancy: f64,
    /// Mean wait each policy needs to reach SOMA's best occupancy.
    pub soma_wait_at_peak: f64,
    pub bulk_wait_at_peak: f64,
}

impl RegimePoint {
    /// Whether the workload has any irregularity at all. With neither staggered
    /// arrival nor jitter, readiness comes in levels and a host can batch it.
    pub fn is_level_synchronous(&self) -> bool {
        self.arrival_span == 0 && self.jitter == 0
    }

    /// How much less waiting SOMA needs to reach the same occupancy.
    pub fn wait_reduction(&self) -> f64 {
        if self.soma_wait_at_peak <= 0.0 {
            return f64::INFINITY;
        }
        self.bulk_wait_at_peak / self.soma_wait_at_peak
    }
}

/// Evaluate one workload shape at a matched latency budget.
pub fn regime_point(knobs: &IrregularKnobs, width: u16, budget: f64) -> RegimePoint {
    let t = trace(knobs);
    let sweep: Vec<u32> = vec![0, 1, 2, 4, 8, 16, 32, 64];
    let f = frontiers(&t, width, &sweep);

    let peak = f
        .soma
        .iter()
        .map(|p| p.occupancy)
        .fold(0.0f64, f64::max)
        - 1e-9;
    let wait_to_peak = |points: &[FrontierPoint]| {
        points
            .iter()
            .filter(|p| p.occupancy >= peak)
            .map(|p| p.mean_wait)
            .fold(f64::INFINITY, f64::min)
    };

    RegimePoint {
        arrival_span: knobs.arrival_span,
        jitter: knobs.jitter,
        class_count: knobs.class_count,
        advantage: f.advantage_at(budget).unwrap_or(f64::NAN),
        soma_occupancy: f.soma_at(budget).map(|p| p.occupancy).unwrap_or(0.0),
        bulk_occupancy: f.bulk_at(budget).map(|p| p.occupancy).unwrap_or(0.0),
        soma_wait_at_peak: wait_to_peak(&f.soma),
        bulk_wait_at_peak: wait_to_peak(&f.bulk),
    }
}

/// Sweep arrival irregularity and class count to map where cohorting pays.
pub fn regime_map(
    base: &IrregularKnobs,
    spans: &[u32],
    jitters: &[u32],
    class_counts: &[u32],
    width: u16,
    budget: f64,
) -> Vec<RegimePoint> {
    let mut out = Vec::new();
    for span in spans {
        for jitter in jitters {
            for classes in class_counts {
                out.push(regime_point(
                    &IrregularKnobs {
                        arrival_span: *span,
                        jitter: *jitter,
                        class_count: *classes,
                        ..*base
                    },
                    width,
                    budget,
                ));
            }
        }
    }
    out
}

/// A human-readable frontier table.
pub fn report(knobs: &IrregularKnobs, width: u16) -> String {
    let t = trace(knobs);
    let sweep: Vec<u32> = vec![0, 1, 2, 4, 8, 16, 32];
    let f = frontiers(&t, width, &sweep);

    let mut s = String::new();
    s.push_str(&format!(
        "roots={} span={} depth={} branch={} classes={} jitter={} width={}\n",
        knobs.roots,
        knobs.arrival_span,
        knobs.depth,
        knobs.branching_factor,
        knobs.class_count,
        knobs.jitter,
        width
    ));
    s.push_str(&format!(
        "  trace: {} arrivals over {} ticks ({:.1}/tick), {} classes\n",
        t.len(),
        t.horizon() + 1,
        t.arrival_rate(),
        t.distinct_classes()
    ));
    s.push_str("  knob | soma occ  mean p50 p99 | bulk occ  mean p50 p99 launches\n");
    for (a, b) in f.soma.iter().zip(f.bulk.iter()) {
        s.push_str(&format!(
            "  {:>4} | {:.3}  {:>5.2} {:>3} {:>3} | {:.3}  {:>5.2} {:>3} {:>3} {:>6}\n",
            a.knob,
            a.occupancy,
            a.mean_wait,
            a.p50_wait,
            a.p99_wait,
            b.occupancy,
            b.mean_wait,
            b.p50_wait,
            b.p99_wait,
            b.host_launches,
        ));
    }
    for budget in [0.5f64, 1.0, 2.0, 4.0] {
        if let Some(adv) = f.advantage_at(budget) {
            let s_pt = f.soma_at(budget).unwrap();
            let b_pt = f.bulk_at(budget).unwrap();
            s.push_str(&format!(
                "  at mean wait <= {:.1} ticks: soma {:.3} vs bulk {:.3} = {:.2}x\n",
                budget, s_pt.occupancy, b_pt.occupancy, adv
            ));
        }
    }
    s
}
