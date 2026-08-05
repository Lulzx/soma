use soma::experiments::streaming_graph::{run, StreamingGraphConfig};

fn main() {
    for capacity in [1, 2, 4, 8] {
        let report = run(StreamingGraphConfig {
            records: 32,
            channel_capacity: capacity,
            fail_source_after: Some(21),
        })
        .expect("streaming graph must execute");
        println!(
            "capacity={capacity:<2} committed={:<2} delivered={:<2} backpressure={:<2} ordered={} source_failed={}",
            report.committed_records,
            report.delivered.len(),
            report.backpressure_events,
            report.ordered(),
            report.source_failed,
        );
    }
}
