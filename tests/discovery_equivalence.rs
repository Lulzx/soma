use soma::experiments::discovery_search::{run_cpu, DiscoveryKnobs};

#[test]
fn identical_trace_has_identical_scientific_state() {
    let report = run_cpu(&DiscoveryKnobs::default()).unwrap();
    assert_eq!(report.naive.scientific, report.optimized.scientific);
    assert!(report.invariants.all_hold(), "{:#?}", report.invariants);
}

#[test]
fn semantic_change_is_detected() {
    let mut a = run_cpu(&DiscoveryKnobs::default()).unwrap();
    let b = a.clone();
    let record = a.optimized.scientific.evidence.values_mut().next().unwrap();
    record.output.0[0] ^= 1;
    assert_ne!(a.optimized.scientific, b.optimized.scientific);
}
