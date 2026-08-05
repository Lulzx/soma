//! JSONL export of an ant-colony run, for a viewer outside the crate.
//!
//! One JSON object per line, written as the run happens rather than collected
//! and serialised at the end. That is what `LogRetention::PerEpoch` is for: the
//! kernel's trace is drained every epoch, the cohort records and failure events
//! are read out of it, and the rest is discarded. A run therefore costs bounded
//! memory however long it goes on, and the census on the way out says whether
//! anything was missed.
//!
//! The encoder is written by hand. The semantic core has no dependencies and
//! there is no reason for a viewer format to be the thing that adds one.
//!
//! # Stream shape
//!
//! ```text
//! {"kind":"header", ...}      once, first
//! {"kind":"epoch", ...}       once per epoch
//! {"kind":"field", ...}       every `field_stride` epochs
//! {"kind":"summary", ...}     once, last
//! ```
//!
//! Nothing downstream has to parse the whole file before drawing: a viewer can
//! render each line as it arrives.

use std::io::{self, Write};

use crate::abi::EventKind;
use crate::compiler::run_classes::ant_class_name;
use crate::experiments::ant_colony::{
    field_totals, observe_ants, read_trail, AntColony, ColonyKnobs, PredatorStrike, Terrain,
    TRAIL_FOOD, TRAIL_HOME,
};
use crate::kernel::retention::LogRetention;
use crate::kernel::Kernel;
use crate::scheduler::runnable_bins::SchedulingMode;

/// How much of the run to write out.
#[derive(Clone, Copy, Debug)]
pub struct ExportOptions {
    /// Lane width the run is cohorted at.
    pub cohort_width: u16,
    pub mode: SchedulingMode,
    /// Emit the pheromone field every `field_stride` epochs. The field is by far
    /// the largest thing in the stream, and it changes slowly.
    pub field_stride: u32,
    /// Emit at most this many ants per epoch, sampled by stride. Zero means all
    /// of them. A viewer showing fifty thousand ants does not need every one in
    /// the file to look right, but the file should say that it sampled.
    pub ant_sample: u32,
    /// Optional predator strike, and the epoch it lands on.
    pub predator: Option<(u32, PredatorStrike)>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            cohort_width: 32,
            mode: SchedulingMode::RunClassBins,
            field_stride: 4,
            ant_sample: 0,
            predator: None,
        }
    }
}

fn mode_name(mode: SchedulingMode) -> &'static str {
    match mode {
        SchedulingMode::RunClassBins => "run-class",
        SchedulingMode::PersistentFifo => "persistent-fifo",
    }
}

// ---- base64 ---------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding. The field is tens of kilobytes per frame and a
/// JSON array of numbers would be four times the size for no benefit.
fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

// ---- the exporter ---------------------------------------------------------

/// Runs a colony and writes the stream.
pub struct Exporter<W: Write> {
    sink: W,
    options: ExportOptions,
    /// Accounting at the end of the previous epoch, so each line can carry the
    /// epoch's own numbers rather than a running total.
    previous: EpochCounters,
}

#[derive(Clone, Copy, Debug, Default)]
struct EpochCounters {
    cohorts: u64,
    lane_slots: u64,
    useful_lane_slots: u64,
    full_cohorts: u64,
    steps: u64,
    deferred_lanes: u64,
}

impl EpochCounters {
    fn of(kernel: &Kernel) -> EpochCounters {
        let accounting = kernel.accounting();
        EpochCounters {
            cohorts: accounting.cohorts,
            lane_slots: accounting.lane_slots,
            useful_lane_slots: accounting.useful_lane_slots,
            full_cohorts: accounting.full_cohorts,
            steps: accounting.steps,
            deferred_lanes: accounting.deferred_lanes,
        }
    }

    fn since(&self, earlier: &EpochCounters) -> EpochCounters {
        EpochCounters {
            cohorts: self.cohorts - earlier.cohorts,
            lane_slots: self.lane_slots - earlier.lane_slots,
            useful_lane_slots: self.useful_lane_slots - earlier.useful_lane_slots,
            full_cohorts: self.full_cohorts - earlier.full_cohorts,
            steps: self.steps - earlier.steps,
            deferred_lanes: self.deferred_lanes - earlier.deferred_lanes,
        }
    }
}

impl<W: Write> Exporter<W> {
    pub fn new(sink: W, options: ExportOptions) -> Exporter<W> {
        Exporter {
            sink,
            options,
            previous: EpochCounters::default(),
        }
    }

    /// Build the colony, run it, and write the whole stream.
    pub fn export(mut self, knobs: &ColonyKnobs) -> io::Result<ExportSummary> {
        let mut kernel = Kernel::with_mode(self.options.mode);
        kernel.configure_cohorts(
            self.options.cohort_width,
            crate::abi::PartialCohortPolicy::RunPartial,
        );
        let (mut kernel, colony) =
            crate::experiments::ant_colony::build_in(kernel, knobs);

        self.write_header(&mut kernel, &colony)?;

        // From here the kernel keeps only the running epoch's records and this
        // loop drains them. Everything the stream reports about cohorts and
        // failures comes out of that drain.
        kernel.set_log_retention(LogRetention::PerEpoch);
        let _ = kernel.take_trace_events();
        let _ = kernel.take_effect_log();
        let _ = kernel.take_admission_log();

        let mut epochs = 0u32;
        let mut struck = 0usize;
        let mut observation = 0u64;
        while epochs < knobs.epochs && kernel.total_pending() > 0 {
            if let Some((at, strike)) = self.options.predator {
                if at == epochs {
                    struck = crate::experiments::ant_colony::inject_predator(
                        &mut kernel,
                        &colony,
                        strike,
                    )
                    .len();
                    // The strike is the harness reaching into the run, and its
                    // writes are traced like any other. Drained here for the
                    // same reason the observation reads are: the next
                    // `run_epoch` would otherwise discard them and the census
                    // would report the stream as lossy when it is not.
                    observation += kernel.take_trace_events().len() as u64;
                }
            }

            kernel.run_epoch();
            let events = kernel.take_trace_events();
            let _ = kernel.take_effect_log();
            let _ = kernel.take_admission_log();

            self.write_epoch(&mut kernel, &colony, epochs, &events)?;
            if self.options.field_stride > 0 && epochs.is_multiple_of(self.options.field_stride) {
                self.write_field(&mut kernel, &colony, epochs)?;
            }

            // Reading the run perturbs its trace: every `object_bytes` the
            // exporter performs to observe an ant is an authorised operation and
            // emits an authority event, attributed to the host rather than to
            // any lane. Those events are the observer's, not the run's, so they
            // are drained and counted separately — otherwise the census would
            // report them as dropped and the stream would look incomplete when
            // nothing about the run had been missed.
            observation += kernel.take_trace_events().len() as u64;
            epochs += 1;
        }

        let summary = self.write_summary(&mut kernel, &colony, epochs, struck, observation)?;
        self.sink.flush()?;
        Ok(summary)
    }

    fn write_header(&mut self, kernel: &mut Kernel, colony: &AntColony) -> io::Result<()> {
        let knobs = &colony.knobs;
        let terrain = kernel
            .object_bytes(colony.world, colony.terrain)
            .map(|b| b.to_vec())
            .unwrap_or_default();
        let food = Terrain::food_sources(&terrain);
        // The obstacle bitmap, verbatim: one bit per cell, row-major.
        let bitmap_start = Terrain::HEADER + food.len() * 6 + knobs.colonies as usize * 4;
        let obstacles = terrain.get(bitmap_start..).unwrap_or(&[]);

        write!(
            self.sink,
            "{{\"kind\":\"header\",\"width\":{},\"height\":{},\"colonies\":{},\
             \"ants\":{},\"epochs\":{},\"seed\":{},\"mode\":\"{}\",\"lane_width\":{},\
             \"field_stride\":{},\"ant_sample\":{}",
            knobs.width,
            knobs.height,
            knobs.colonies,
            colony.ant_count(),
            knobs.epochs,
            knobs.seed,
            mode_name(self.options.mode),
            self.options.cohort_width,
            self.options.field_stride,
            self.options.ant_sample,
        )?;

        write!(self.sink, ",\"nests\":[")?;
        for (index, handle) in colony.colonies.iter().enumerate() {
            if index > 0 {
                write!(self.sink, ",")?;
            }
            write!(
                self.sink,
                "[{},{},{}]",
                handle.nest.0,
                handle.nest.1,
                handle.ants.len()
            )?;
        }
        write!(self.sink, "],\"food\":[")?;
        for (index, (x, y, r)) in food.iter().enumerate() {
            if index > 0 {
                write!(self.sink, ",")?;
            }
            write!(self.sink, "[{x},{y},{r}]")?;
        }
        write!(self.sink, "],\"obstacles\":\"{}\"", base64(obstacles))?;

        // The run-class table, so the viewer never hardcodes a behaviour name.
        write!(self.sink, ",\"run_classes\":{{")?;
        let mut first = true;
        for run_class in crate::compiler::run_classes::ANT_BASE
            ..=crate::compiler::run_classes::WORLD_STEP
        {
            if let Some(name) = ant_class_name(run_class) {
                if !first {
                    write!(self.sink, ",")?;
                }
                write!(self.sink, "\"{run_class}\":\"{name}\"")?;
                first = false;
            }
        }
        writeln!(self.sink, "}}}}")
    }

    fn write_epoch(
        &mut self,
        kernel: &mut Kernel,
        colony: &AntColony,
        epoch: u32,
        events: &[crate::abi::TraceEvent],
    ) -> io::Result<()> {
        let counters = EpochCounters::of(kernel);
        let delta = counters.since(&self.previous);
        self.previous = counters;

        write!(self.sink, "{{\"kind\":\"epoch\",\"epoch\":{epoch}")?;

        // Ants, sampled by stride if asked. `[id, x, y, carrying, run_class]`.
        let ants = observe_ants(kernel, colony);
        let stride = if self.options.ant_sample == 0 {
            1
        } else {
            (ants.len() as u32).div_ceil(self.options.ant_sample).max(1) as usize
        };
        write!(self.sink, ",\"ants\":[")?;
        let mut written = 0usize;
        for ant in ants.iter().step_by(stride) {
            if !ant.alive {
                continue;
            }
            if written > 0 {
                write!(self.sink, ",")?;
            }
            write!(
                self.sink,
                "[{},{},{},{},{}]",
                ant.id, ant.x, ant.y, ant.carrying as u8, ant.run_class
            )?;
            written += 1;
        }
        write!(self.sink, "],\"ants_shown\":{written},\"ants_alive\":{}",
            ants.iter().filter(|a| a.alive).count())?;

        // Ready continuations per run class — the bin histogram.
        write!(self.sink, ",\"bins\":[")?;
        for (index, (run_class, count)) in kernel.pending_counts().iter().enumerate() {
            if index > 0 {
                write!(self.sink, ",")?;
            }
            write!(self.sink, "[{run_class},{count}]")?;
        }
        write!(self.sink, "]")?;

        // The epoch's actual dispatch shape, read out of the drained trace.
        // `auxiliary` on a `CohortCreated` event is the cohort's active lanes.
        write!(self.sink, ",\"cohorts\":[")?;
        for (index, event) in events
            .iter()
            .filter(|e| e.event_kind == EventKind::CohortCreated)
            .enumerate()
        {
            if index > 0 {
                write!(self.sink, ",")?;
            }
            write!(self.sink, "[{},{}]", event.run_class, event.auxiliary)?;
        }
        write!(self.sink, "]")?;

        // Supervision and failure, also from the drained trace.
        let failures = events
            .iter()
            .filter(|e| e.event_kind == EventKind::ProcessFailed)
            .count();
        let notices = events
            .iter()
            .filter(|e| e.event_kind == EventKind::SupervisionNotified)
            .count();
        let restarts = events
            .iter()
            .filter(|e| e.event_kind == EventKind::ProcessRestarted)
            .count();

        writeln!(
            self.sink,
            ",\"metrics\":{{\"dispatches\":{},\"lane_slots\":{},\"useful_lane_slots\":{},\
             \"full_cohorts\":{},\"steps\":{},\"deferred\":{},\"failures\":{},\
             \"notices\":{},\"restarts\":{},\"host_launches\":0}}}}",
            delta.cohorts,
            delta.lane_slots,
            delta.useful_lane_slots,
            delta.full_cohorts,
            delta.steps,
            delta.deferred_lanes,
            failures,
            notices,
            restarts,
        )
    }

    /// The pheromone field, quantised to a byte per cell per trail.
    ///
    /// A `u16` of trail is more precision than a colour ramp can show, and the
    /// field is the largest thing in the stream, so it is scaled to its own
    /// maximum and sent as bytes. The scale factor goes out with it so the
    /// viewer is not guessing.
    fn write_field(
        &mut self,
        kernel: &mut Kernel,
        colony: &AntColony,
        epoch: u32,
    ) -> io::Result<()> {
        let cells = colony.knobs.cells();
        let field = colony.readable_field(epoch + 1);
        let Ok(bytes) = kernel.object_bytes(colony.world, field) else {
            return Ok(());
        };

        let mut food = vec![0u8; cells];
        let mut home = vec![0u8; cells];
        let mut food_max = 1u16;
        let mut home_max = 1u16;
        for cell in 0..cells {
            food_max = food_max.max(read_trail(bytes, cells, TRAIL_FOOD, cell));
            home_max = home_max.max(read_trail(bytes, cells, TRAIL_HOME, cell));
        }
        for cell in 0..cells {
            food[cell] =
                ((read_trail(bytes, cells, TRAIL_FOOD, cell) as u32 * 255) / food_max as u32) as u8;
            home[cell] =
                ((read_trail(bytes, cells, TRAIL_HOME, cell) as u32 * 255) / home_max as u32) as u8;
        }

        writeln!(
            self.sink,
            "{{\"kind\":\"field\",\"epoch\":{epoch},\"food_max\":{food_max},\
             \"home_max\":{home_max},\"food\":\"{}\",\"home\":\"{}\"}}",
            base64(&food),
            base64(&home)
        )
    }

    fn write_summary(
        &mut self,
        kernel: &mut Kernel,
        colony: &AntColony,
        epochs: u32,
        struck: usize,
        observation: u64,
    ) -> io::Result<ExportSummary> {
        let (food_trail, home_trail) = field_totals(kernel, colony, epochs);
        let ants = observe_ants(kernel, colony);
        let accounting = kernel.accounting();
        let census = kernel.log_accounting();

        let summary = ExportSummary {
            epochs,
            ants_alive: ants.iter().filter(|a| a.alive).count(),
            delivered: ants.iter().map(|a| a.delivered as u64).sum(),
            dispatches: accounting.cohorts,
            lane_occupancy: accounting.lane_occupancy(),
            food_trail,
            home_trail,
            struck,
            trace_events_streamed: census.trace.taken.saturating_sub(observation),
            trace_events_observed: observation,
            trace_events_dropped: census.trace.dropped,
        };

        writeln!(
            self.sink,
            "{{\"kind\":\"summary\",\"epochs\":{},\"ants_alive\":{},\"delivered\":{},\
             \"dispatches\":{},\"lane_occupancy\":{:.6},\"food_trail\":{},\"home_trail\":{},\
             \"struck\":{},\"trace_events_streamed\":{},\"trace_events_observed\":{},\
             \"trace_events_dropped\":{}}}",
            summary.epochs,
            summary.ants_alive,
            summary.delivered,
            summary.dispatches,
            summary.lane_occupancy,
            summary.food_trail,
            summary.home_trail,
            summary.struck,
            summary.trace_events_streamed,
            summary.trace_events_observed,
            summary.trace_events_dropped,
        )?;
        Ok(summary)
    }
}

/// What the exported run amounted to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExportSummary {
    pub epochs: u32,
    pub ants_alive: usize,
    pub delivered: u64,
    pub dispatches: u64,
    pub lane_occupancy: f64,
    pub food_trail: u64,
    pub home_trail: u64,
    pub struck: usize,
    /// Trace events the run produced and the exporter consumed.
    pub trace_events_streamed: u64,
    /// Trace events produced by the exporter's own reads of the run. Reading an
    /// ant requires authority, and exercising authority is a traced event, so
    /// observing the machine adds to its trace. Counted apart from the run's own
    /// events so neither number misrepresents the other.
    pub trace_events_observed: u64,
    /// Trace events discarded without being consumed. Non-zero means the stream
    /// is missing something, and the stream says so rather than looking whole.
    pub trace_events_dropped: u64,
}

impl ExportSummary {
    /// Whether the export saw every trace event the run produced.
    pub fn is_complete(&self) -> bool {
        self.trace_events_dropped == 0
    }
}

/// Export a run to any sink.
pub fn export<W: Write>(
    sink: W,
    knobs: &ColonyKnobs,
    options: ExportOptions,
) -> io::Result<ExportSummary> {
    Exporter::new(sink, options).export(knobs)
}
