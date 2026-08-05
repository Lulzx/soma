use soma::experiments::streaming_graph::{run, StreamingGraphConfig};

#[test]
fn bounded_stream_preserves_every_record_in_fifo_order() {
    let report = run(StreamingGraphConfig {
        records: 23,
        channel_capacity: 3,
        fail_source_after: None,
    })
    .unwrap();
    assert_eq!(report.committed_records, 23);
    assert_eq!(report.delivered.len(), 23);
    assert!(report.ordered());
}

#[test]
fn a_small_channel_exercises_backpressure_without_dropping_work() {
    let report = run(StreamingGraphConfig {
        records: 12,
        channel_capacity: 2,
        fail_source_after: None,
    })
    .unwrap();
    assert!(report.backpressure_events > 0);
    assert_eq!(report.delivered.len(), 12);
    assert!(report.ordered());
}

#[test]
fn committed_prefix_survives_source_failure() {
    let report = run(StreamingGraphConfig {
        records: 20,
        channel_capacity: 4,
        fail_source_after: Some(9),
    })
    .unwrap();
    assert!(report.source_failed);
    assert_eq!(report.committed_records, 9);
    assert_eq!(report.delivered, (0..9).collect::<Vec<_>>());
    assert!(report.ordered());
}
