use soma::abi::SupervisionPolicy;
use soma::experiments::streaming_graph::{
    run_placed as run_stream, StreamingGraphConfig, StreamingPlacement,
};
use soma::experiments::supervision_tree::{run, run_placed as run_tree, TreeKnobs, TreePlacement};
use soma::semantics::order::conforms_traces;

#[test]
fn two_node_streaming_graph_is_i18_equivalent() {
    for fail_source_after in [None, Some(19)] {
        let config = StreamingGraphConfig {
            records: 41,
            channel_capacity: 3,
            fail_source_after,
        };
        let local = run_stream(config, StreamingPlacement::default()).unwrap();
        let distributed = run_stream(
            config,
            StreamingPlacement {
                coordinator: 1,
                source: 2,
                sink: 2,
            },
        )
        .unwrap();
        assert!(local.legal && distributed.legal);
        assert_eq!(local.report, distributed.report);
        assert_eq!(distributed.remote_channel_edges, 2);
        assert!(
            conforms_traces(&local.trace, &distributed.trace).is_empty(),
            "node placement changed streaming semantics"
        );
    }
}

#[test]
fn every_supervision_edge_is_remote_and_i18_equivalent() {
    let placement = TreePlacement {
        root: 1,
        left_branch: 2,
        right_branch: 2,
        left_worker: 1,
        right_worker: 1,
    };
    for worker_policy in [
        SupervisionPolicy::Notify,
        SupervisionPolicy::Escalate,
        SupervisionPolicy::Restart,
    ] {
        let knobs = TreeKnobs {
            worker_policy,
            inject_left_failure: true,
        };
        let local = run(&knobs);
        let distributed = run_tree(&knobs, placement);
        assert!(local.legal && distributed.legal);
        assert_eq!(local.outcome, distributed.outcome);
        assert_eq!(distributed.remote_edges, 4);
        assert!(
            conforms_traces(&local.trace, &distributed.trace).is_empty(),
            "node placement changed supervision semantics for {worker_policy:?}"
        );
    }
}
