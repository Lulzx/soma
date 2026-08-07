//! Execution territories: does cohorting survive distribution?
//!
//! Everything measured so far assumed one global pool of ready continuations to
//! cut cohorts from. A device-resident scheduler does not have that. A GPU is
//! many independently schedulable compute territories — SMs, compute units,
//! shader cores, whatever the vendor calls them — each with its own queues, and
//! a cohort must be assembled from continuations that are *in the same
//! territory*. Distribution is therefore not a detail of the implementation; it
//! is the thing that decides whether the mechanism is real.
//!
//! # Why this is a threat and not a formality
//!
//! With `T` territories and `K` run classes, a territory holds roughly
//! `N/(T·K)` ready continuations of any given class, and a full cohort needs
//! `W` of one class in one territory. At 80 territories, 4 classes and width
//! 32 that wants ten thousand concurrently ready continuations before cohorts
//! fill reliably. Below it, fragmentation shreds occupancy — and the global
//! numbers reported elsewhere in this crate are an upper bound that quietly
//! assumed the problem away.
//!
//! The hand-written bulk frontier is largely immune, which sharpens the
//! problem. It sorts the whole frontier on the host before dispatching, so its
//! lane groups are uniform no matter which territory ends up running them. A
//! global sort beats local grouping at cohort formation, by construction.
//! Distributed cohorting can only match it by *placing* work so that a class
//! concentrates instead of scattering.
//!
//! # What is being compared
//!
//! Placement is the whole experiment. Each routing policy is a different answer
//! to "which territory should this continuation run in", and each trades cohort
//! fill against load balance:
//!
//! * `Local` — stay where the parent ran. No decision at all, which is what a
//!   GPU does today; work scatters with the tree.
//! * `RoundRobin` — spread evenly. Perfect balance, worst possible fill.
//! * `ClassAffinity` — send a run class to a fixed territory. Best possible
//!   fill, and badly imbalanced the moment classes are not equally busy.
//! * `ClassAffinityBalanced` — affinity until a territory's queue for that
//!   class exceeds a cap, then spill to the least-loaded territory.
//!
//! Only a device-resident scheduler can make this choice at all. A host-launched
//! model has no place to put the decision.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::experiments::irregular_arrival::{Arrival, ArrivalTrace};
use crate::scheduler::cohorts::{dispatch_cost, DispatchCost};

/// How a ready continuation is assigned to a territory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    /// Run where the parent ran; roots are spread round-robin. Models a GPU
    /// with no placement policy, where spawned work lands in the creating
    /// territory's queue.
    Local,
    /// Spread arrivals evenly, ignoring run class.
    RoundRobin,
    /// A run class always goes to the same territory.
    ClassAffinity,
    /// Class affinity until the target's queue for that class reaches `cap`,
    /// then spill to the least-loaded territory.
    ClassAffinityBalanced { cap: usize },
    /// Give each run class a *block* of territories sized by its share of the
    /// load, and spread that class's work across its own block.
    ///
    /// This is the policy the other three are missing. Plain class affinity
    /// concentrates perfectly but cannot occupy more territories than there are
    /// run classes, so it idles the rest of the machine; the locality-blind
    /// policies occupy everything but shred every cohort. Sizing a block per
    /// class decouples the two: concentration is what fills cohorts, and block
    /// *width* is what keeps territories busy, so both can be satisfied at once
    /// as long as a class's load can keep its block above the cohort width.
    ProportionalAffinity,
}

/// One distributed-scheduler configuration.
#[derive(Clone, Copy, Debug)]
pub struct TerritoryConfig {
    pub territories: u32,
    pub width: u16,
    pub routing: Routing,
    /// Ticks a partial cohort may wait before being dispatched anyway.
    pub max_defer: u32,
}

/// What a distributed configuration achieved.
#[derive(Clone, Debug, Default)]
pub struct TerritoryOutcome {
    pub cost: DispatchCost,
    /// Dispatches issued by each territory, in territory order.
    pub per_territory: Vec<u64>,
    pub waits: Vec<u32>,
    /// Arrivals placed somewhere other than their affinity territory.
    pub spills: u64,
    pub makespan: u32,
}

impl TerritoryOutcome {
    pub fn occupancy(&self) -> f64 {
        self.cost.occupancy()
    }

    pub fn dispatches(&self) -> u64 {
        self.cost.dispatches
    }

    pub fn items(&self) -> usize {
        self.waits.len()
    }

    /// Load imbalance: the busiest territory's dispatch count over the mean.
    /// 1.0 is perfect balance; 2.0 means one territory did twice its share, so
    /// the other territories are idle waiting for it.
    pub fn imbalance(&self) -> f64 {
        if self.per_territory.is_empty() {
            return 1.0;
        }
        let total: u64 = self.per_territory.iter().sum();
        if total == 0 {
            return 1.0;
        }
        let mean = total as f64 / self.per_territory.len() as f64;
        let max = *self.per_territory.iter().max().unwrap() as f64;
        max / mean
    }

    /// Territories that issued nothing at all.
    pub fn idle_territories(&self) -> usize {
        self.per_territory.iter().filter(|d| **d == 0).count()
    }

    pub fn mean_wait(&self) -> f64 {
        if self.waits.is_empty() {
            return 0.0;
        }
        self.waits.iter().map(|w| *w as f64).sum::<f64>() / self.waits.len() as f64
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

    pub fn p50_wait(&self) -> u32 {
        self.percentile(0.50)
    }

    pub fn p99_wait(&self) -> u32 {
        self.percentile(0.99)
    }

    /// A single figure of merit: occupancy discounted by imbalance. A scheduler
    /// that fills every cohort by funnelling all work into one territory has
    /// not actually won, and this is what says so.
    pub fn effective_occupancy(&self) -> f64 {
        self.occupancy() / self.imbalance().max(1.0)
    }
}

/// Per-territory, per-class queues.
type Queues = Vec<BTreeMap<u32, VecDeque<Arrival>>>;

fn queued_for(queues: &Queues, territory: usize, run_class: u32) -> usize {
    queues[territory]
        .get(&run_class)
        .map(|q| q.len())
        .unwrap_or(0)
}

fn total_queued(queues: &Queues, territory: usize) -> usize {
    queues[territory].values().map(|q| q.len()).sum()
}

/// Territory blocks for `ProportionalAffinity`: each class gets a contiguous
/// run of territories sized by its share of the trace, with at least one each.
fn proportional_blocks(trace: &ArrivalTrace, territories: usize) -> HashMap<u32, (usize, usize)> {
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for e in &trace.events {
        *counts.entry(e.run_class).or_insert(0) += 1;
    }
    let total: usize = counts.values().sum();
    let mut blocks = HashMap::new();
    if total == 0 || counts.is_empty() {
        return blocks;
    }

    let classes: Vec<(u32, usize)> = counts.into_iter().collect();
    let k = classes.len();

    // Fewer territories than classes: there is nothing to apportion, and the
    // policy degenerates to plain class affinity with classes sharing homes.
    if territories <= k {
        for (i, (class, _)) in classes.iter().enumerate() {
            blocks.insert(*class, (i % territories, 1));
        }
        return blocks;
    }

    // Largest-remainder apportionment with a floor of one territory per class.
    let mut sizes: Vec<usize> = classes
        .iter()
        .map(|(_, n)| (n * territories / total).max(1))
        .collect();
    let mut assigned: usize = sizes.iter().sum();

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|i| {
        let remainder = (classes[*i].1 * territories) % total;
        (std::cmp::Reverse(remainder), classes[*i].0)
    });

    let mut cursor = 0;
    while assigned < territories {
        sizes[order[cursor % k]] += 1;
        assigned += 1;
        cursor += 1;
    }
    // The floor can over-allocate; take back from the largest blocks, never
    // below one, and stop if nothing can be reduced.
    while assigned > territories {
        match (0..k)
            .filter(|i| sizes[*i] > 1)
            .max_by_key(|i| (sizes[*i], std::cmp::Reverse(*i)))
        {
            Some(i) => {
                sizes[i] -= 1;
                assigned -= 1;
            }
            None => break,
        }
    }

    let mut start = 0;
    for (i, (class, _)) in classes.iter().enumerate() {
        let len = sizes[i]
            .max(1)
            .min(territories - start.min(territories - 1));
        blocks.insert(*class, (start.min(territories - 1), len));
        start = (start + len).min(territories);
    }
    blocks
}

/// Choose a territory for `arrival`.
#[allow(clippy::too_many_arguments)]
fn route(
    arrival: &Arrival,
    routing: Routing,
    territories: usize,
    queues: &Queues,
    placed: &HashMap<u64, usize>,
    round_robin: &mut usize,
    blocks: &HashMap<u32, (usize, usize)>,
    class_rr: &mut HashMap<u32, usize>,
) -> (usize, bool) {
    match routing {
        Routing::ProportionalAffinity => {
            let (start, len) = blocks
                .get(&arrival.run_class)
                .copied()
                .unwrap_or((0, territories));
            let counter = class_rr.entry(arrival.run_class).or_insert(0);
            let t = (start + (*counter % len.max(1))) % territories;
            *counter += 1;
            (t, false)
        }
        Routing::RoundRobin => {
            let t = *round_robin % territories;
            *round_robin += 1;
            (t, false)
        }
        Routing::Local => match arrival.parent.and_then(|p| placed.get(&p)) {
            Some(t) => (*t, false),
            None => {
                let t = *round_robin % territories;
                *round_robin += 1;
                (t, false)
            }
        },
        Routing::ClassAffinity => ((arrival.run_class as usize) % territories, false),
        Routing::ClassAffinityBalanced { cap } => {
            let home = (arrival.run_class as usize) % territories;
            if queued_for(queues, home, arrival.run_class) < cap {
                (home, false)
            } else {
                // Least loaded overall, ties broken by lowest index so the
                // choice stays deterministic.
                let target = (0..territories)
                    .min_by_key(|t| (total_queued(queues, *t), *t))
                    .unwrap_or(home);
                (target, target != home)
            }
        }
    }
}

/// Run the distributed scheduler over a trace.
///
/// Each territory forms cohorts only from its own queues — that is the whole
/// point — and dispatches a full cohort as soon as one exists, independently of
/// every other territory.
pub fn territory_policy(trace: &ArrivalTrace, cfg: &TerritoryConfig) -> TerritoryOutcome {
    let territories = cfg.territories.max(1) as usize;
    let w = cfg.width.max(1) as usize;

    let mut out = TerritoryOutcome {
        per_territory: vec![0; territories],
        ..Default::default()
    };
    let mut queues: Queues = vec![BTreeMap::new(); territories];
    let mut placed: HashMap<u64, usize> = HashMap::new();
    let mut round_robin = 0usize;
    let blocks = proportional_blocks(trace, territories);
    let mut class_rr: HashMap<u32, usize> = HashMap::new();

    let mut by_tick: BTreeMap<u32, Vec<Arrival>> = BTreeMap::new();
    for e in &trace.events {
        by_tick.entry(e.tick).or_default().push(*e);
    }
    let last_tick = trace.horizon();

    let dispatch = |out: &mut TerritoryOutcome,
                    queue: &mut VecDeque<Arrival>,
                    territory: usize,
                    run_class: u32,
                    count: usize,
                    tick: u32| {
        let batch: Vec<Arrival> = queue.drain(..count).collect();
        let cost = dispatch_cost(&vec![run_class; count], cfg.width);
        out.cost.dispatches += cost.dispatches;
        out.cost.lane_slots += cost.lane_slots;
        out.cost.useful_lane_slots += cost.useful_lane_slots;
        out.cost.full_dispatches += cost.full_dispatches;
        out.per_territory[territory] += cost.dispatches;
        for a in &batch {
            out.waits.push(tick - a.tick);
        }
        out.makespan = out.makespan.max(tick);
    };

    for tick in 0..=last_tick {
        if let Some(arrivals) = by_tick.get(&tick) {
            for a in arrivals {
                let (territory, spilled) = route(
                    a,
                    cfg.routing,
                    territories,
                    &queues,
                    &placed,
                    &mut round_robin,
                    &blocks,
                    &mut class_rr,
                );
                if spilled {
                    out.spills += 1;
                }
                placed.insert(a.id, territory);
                queues[territory]
                    .entry(a.run_class)
                    .or_default()
                    .push_back(*a);
            }
        }

        // Every territory schedules itself, with no global coordination.
        for territory in 0..territories {
            let classes: Vec<u32> = queues[territory].keys().copied().collect();
            for run_class in classes {
                loop {
                    let len = queued_for(&queues, territory, run_class);
                    if len < w {
                        break;
                    }
                    let queue = queues[territory].get_mut(&run_class).unwrap();
                    dispatch(&mut out, queue, territory, run_class, w, tick);
                }

                let (len, oldest) = {
                    let queue = queues[territory].get(&run_class);
                    (
                        queue.map(|q| q.len()).unwrap_or(0),
                        queue.and_then(|q| q.front()).map(|a| a.tick),
                    )
                };
                if let (true, Some(oldest)) = (len > 0, oldest) {
                    if tick - oldest >= cfg.max_defer || tick == last_tick {
                        let queue = queues[territory].get_mut(&run_class).unwrap();
                        dispatch(&mut out, queue, territory, run_class, len, tick);
                    }
                }
            }
        }
    }

    // Drain anything still queued past the horizon.
    for territory in 0..territories {
        let classes: Vec<u32> = queues[territory].keys().copied().collect();
        for run_class in classes {
            let len = queued_for(&queues, territory, run_class);
            if len == 0 {
                continue;
            }
            let queue = queues[territory].get_mut(&run_class).unwrap();
            dispatch(&mut out, queue, territory, run_class, len, last_tick);
        }
    }

    out
}

/// The globally-sorted bulk frontier under the same distribution.
///
/// It sorts on the host before dispatching, so its lane groups are uniform
/// regardless of which territory runs them — distribution costs it nothing in
/// cohort fill. This is the number distributed cohorting has to match, and it
/// is deliberately the strongest form of the baseline.
pub fn global_sort_reference(trace: &ArrivalTrace, width: u16, window: u32) -> DispatchCost {
    crate::experiments::irregular_arrival::bulk_policy(trace, width, window).cost
}

/// One row of the placement comparison.
#[derive(Clone, Copy, Debug)]
pub struct PlacementResult {
    pub territories: u32,
    pub routing: Routing,
    pub occupancy: f64,
    pub imbalance: f64,
    pub effective_occupancy: f64,
    pub idle_territories: usize,
    pub mean_wait: f64,
    pub p99_wait: u32,
    pub dispatches: u64,
    pub spills: u64,
}

/// Evaluate one configuration.
pub fn evaluate(trace: &ArrivalTrace, cfg: &TerritoryConfig) -> PlacementResult {
    let o = territory_policy(trace, cfg);
    PlacementResult {
        territories: cfg.territories,
        routing: cfg.routing,
        occupancy: o.occupancy(),
        imbalance: o.imbalance(),
        effective_occupancy: o.effective_occupancy(),
        idle_territories: o.idle_territories(),
        mean_wait: o.mean_wait(),
        p99_wait: o.p99_wait(),
        dispatches: o.dispatches(),
        spills: o.spills,
    }
}

/// Sweep territory counts and routing policies.
pub fn sweep(
    trace: &ArrivalTrace,
    territory_counts: &[u32],
    routings: &[Routing],
    width: u16,
    max_defer: u32,
) -> Vec<PlacementResult> {
    let mut out = Vec::new();
    for territories in territory_counts {
        for routing in routings {
            out.push(evaluate(
                trace,
                &TerritoryConfig {
                    territories: *territories,
                    width,
                    routing: *routing,
                    max_defer,
                },
            ));
        }
    }
    out
}

fn routing_name(r: Routing) -> &'static str {
    match r {
        Routing::Local => "local",
        Routing::RoundRobin => "round-robin",
        Routing::ClassAffinity => "class-affinity",
        Routing::ClassAffinityBalanced { .. } => "affinity+spill",
        Routing::ProportionalAffinity => "proportional",
    }
}

/// A human-readable placement table.
pub fn report(trace: &ArrivalTrace, width: u16, max_defer: u32, cap: usize) -> String {
    let routings = [
        Routing::Local,
        Routing::RoundRobin,
        Routing::ClassAffinity,
        Routing::ClassAffinityBalanced { cap },
        Routing::ProportionalAffinity,
    ];
    let counts = [1u32, 4, 16, 64];
    let rows = sweep(trace, &counts, &routings, width, max_defer);

    let reference = global_sort_reference(trace, width, max_defer.max(1));

    let mut s = String::new();
    s.push_str(&format!(
        "{} arrivals over {} ticks, {} classes, width={} defer={}\n",
        trace.len(),
        trace.horizon() + 1,
        trace.distinct_classes(),
        width,
        max_defer
    ));
    s.push_str(&format!(
        "global-sort reference (host-launched): occupancy={:.3}\n",
        reference.occupancy()
    ));
    s.push_str("  terr routing         | occ    imbal  eff    idle | mean p99 spills\n");
    for r in &rows {
        s.push_str(&format!(
            "  {:>4} {:<15} | {:.3}  {:>5.2}  {:.3}  {:>4} | {:>4.2} {:>3} {:>6}\n",
            r.territories,
            routing_name(r.routing),
            r.occupancy,
            r.imbalance,
            r.effective_occupancy,
            r.idle_territories,
            r.mean_wait,
            r.p99_wait,
            r.spills,
        ));
    }
    s
}
