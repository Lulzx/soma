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

    let knobs = ColonyKnobs {
        colonies: 4,
        ants_per_colony: 96,
        epochs: 400,
        ..ColonyKnobs::default()
    };

    let runs = [
        (
            "run-class.jsonl",
            ExportOptions {
                mode: SchedulingMode::RunClassBins,
                ..ExportOptions::default()
            },
        ),
        (
            "persistent-fifo.jsonl",
            ExportOptions {
                mode: SchedulingMode::PersistentFifo,
                ..ExportOptions::default()
            },
        ),
        (
            "predator.jsonl",
            ExportOptions {
                mode: SchedulingMode::RunClassBins,
                predator: Some((200, PredatorStrike { colony: 1, victims: 48 })),
                ..ExportOptions::default()
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
