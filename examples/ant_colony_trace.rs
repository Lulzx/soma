//! Write an ant-colony run to JSONL for the viewer in `viz/`.
//!
//! Two runs of the identical colony, one binned by run class and one through a
//! single persistent FIFO, plus a third with a predator strike. The viewer
//! cross-fades between the first two; the third is the failure demo.
//!
//! Usage: `cargo run --release --example ant_colony_trace [out_dir]`

use std::fs::{create_dir_all, File};
use std::io::BufWriter;

use soma::experiments::ant_colony::{ColonyKnobs, PredatorStrike};
use soma::experiments::ant_trace::{export, ExportOptions};
use soma::scheduler::runnable_bins::SchedulingMode;

fn main() -> std::io::Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "viz/data".into());
    create_dir_all(&out)?;

    // Ten thousand ants across a hundred colonies. The colony size is what keeps
    // every capability space bounded, so the population scales by adding
    // colonies rather than by making them bigger.
    let knobs = ColonyKnobs {
        colonies: 100,
        ants_per_colony: 100,
        width: 320,
        height: 320,
        food_sources: 40,
        epochs: 260,
        ..ColonyKnobs::default()
    };

    // The whole population is simulated; the stream carries a sample of it. Ten
    // thousand ants a frame is tens of megabytes that a viewer drawing dots does
    // not need, and the header records the sampling so the page can say so.
    let sampling = ExportOptions {
        ant_sample: 2500,
        field_stride: 6,
        field_scale: 2,
        ..ExportOptions::default()
    };

    let runs = [
        (
            "run-class.jsonl",
            ExportOptions {
                mode: SchedulingMode::RunClassBins,
                ..sampling
            },
        ),
        (
            "persistent-fifo.jsonl",
            ExportOptions {
                mode: SchedulingMode::PersistentFifo,
                ..sampling
            },
        ),
        (
            "predator.jsonl",
            ExportOptions {
                mode: SchedulingMode::RunClassBins,
                // A whole colony, so the containment is visible against
                // ninety-nine that carry on.
                predator: Some((150, PredatorStrike { colony: 42, victims: 100 })),
                ..sampling
            },
        ),
    ];

    for (name, options) in runs {
        let path = format!("{out}/{name}");
        let summary = export(BufWriter::new(File::create(&path)?), &knobs, options)?;
        let bytes = std::fs::metadata(&path)?.len();
        println!(
            "{path:<32} {:>4} epochs  {:>5} ants  {:>6} delivered  \
             occupancy {:.3}  {:>7} dispatches  {:.1} MB  complete={}",
            summary.epochs,
            summary.ants_alive,
            summary.delivered,
            summary.lane_occupancy,
            summary.dispatches,
            bytes as f64 / 1e6,
            summary.is_complete(),
        );
    }
    Ok(())
}
