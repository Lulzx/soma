use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use soma::abi::{Kind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_HEURISTIC};
use soma::compiler::state_machine_lowering::HeuristicFrame;
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore, RemoteGrant};
use soma::distributed::remote_journal::{
    RemoteJournalServer, RemoteJournalService, RemoteJournalValidator,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::experiments::dynamic_search::{build, ControlKnobs};
use soma::kernel::speculation::EpochExecutive;
use soma::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::device::{DeviceLaneAccess, LaneConflictValidator, LaneValidationError};
use soma::scheduler::device_ops::{DeviceLaneOperation, DeviceOperationJournal};
use soma::semantics::invariants::assert_legal;
use soma::semantics::order::conforms_traces;

struct Fixture {
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    grant: RemoteGrant,
    service: Arc<Mutex<RemoteJournalService>>,
    listener: TcpListener,
}

fn fixture() -> Fixture {
    let issuer = NodeId(1);
    let worker = NodeId(2);
    let target = RemoteRef {
        node: worker,
        entity: Ref64::new(1, 1, Kind::Module),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0xA7; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: worker,
        actor: Ref64::new(1, 1, Kind::Process),
        target,
        rights: Rights::READ,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: u32::MAX,
    });
    let service = Arc::new(Mutex::new(RemoteJournalService::new(
        worker,
        target,
        1,
        authority.clone(),
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    Fixture {
        authority,
        grant,
        service,
        listener,
    }
}

fn leaf_knobs() -> ControlKnobs {
    ControlKnobs {
        branching_factor: 0,
        depth: 0,
        process_count: 8,
        class_count: 3,
        arithmetic_ops: 2_000,
        ..ControlKnobs::default()
    }
}

#[test]
fn an_actual_epoch_can_use_an_authenticated_remote_commit_gate() {
    let fixture = fixture();
    let endpoint = fixture.listener.local_addr().unwrap();
    let service = fixture.service.clone();
    let server = std::thread::spawn(move || {
        RemoteJournalServer::serve_n(fixture.listener, service, 1).unwrap()
    });
    let mut validator = RemoteJournalValidator::new(endpoint, fixture.grant, 0);
    let mut reference = build(&leaf_knobs());
    let mut remote = build(&leaf_knobs());
    remote.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 16 });

    reference.run_epoch();
    remote.run_epoch_with_lane_validator(&mut validator);

    server.join().unwrap();
    assert_eq!(fixture.service.lock().unwrap().applied_requests(), 1);
    assert!(fixture.service.lock().unwrap().operation_records() > 0);
    assert_eq!(remote.speculation_stats().committed_epochs, 1);
    assert!(conforms_traces(&reference.trace_snapshot(), &remote.trace_snapshot()).is_empty());
    assert_legal(&remote);
}

fn contested_future() -> Kernel {
    let mut kernel = Kernel::new();
    kernel.set_allocation_partitions(8);
    let future = kernel.create_future(SYSTEM_PRINCIPAL);
    for input in [7, 13] {
        let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        kernel
            .grant_capability(SYSTEM_PRINCIPAL, process, future, Rights::RESOLVE, 0, 0)
            .unwrap();
        let mut frame = Vec::new();
        HeuristicFrame { future, input }.encode(&mut frame);
        kernel
            .create_continuation(
                SYSTEM_PRINCIPAL,
                process,
                ContinuationSpec::new(
                    StateAccess::ReadOnly,
                    SEARCH_HEURISTIC,
                    0,
                    frame,
                    DEFAULT_MAX_STEPS,
                ),
            )
            .unwrap();
    }
    kernel
}

#[test]
fn a_remote_conflict_discards_journals_and_replays_the_epoch() {
    let fixture = fixture();
    let endpoint = fixture.listener.local_addr().unwrap();
    let service = fixture.service.clone();
    let server = std::thread::spawn(move || {
        RemoteJournalServer::serve_n(fixture.listener, service, 1).unwrap()
    });
    let mut validator = RemoteJournalValidator::new(endpoint, fixture.grant, 0);
    let mut reference = contested_future();
    let mut remote = contested_future();
    remote.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });

    reference.run_epoch();
    remote.run_epoch_with_lane_validator(&mut validator);

    server.join().unwrap();
    assert_eq!(remote.speculation_stats().conflict_fallbacks, 1);
    assert!(conforms_traces(&reference.trace_snapshot(), &remote.trace_snapshot()).is_empty());
}

#[test]
fn exact_retry_applies_once_but_revocation_precedes_the_cache() {
    let fixture = fixture();
    let endpoint = fixture.listener.local_addr().unwrap();
    let service = fixture.service.clone();
    let server_service = service.clone();
    let server = std::thread::spawn(move || {
        RemoteJournalServer::serve_n(fixture.listener, server_service, 3).unwrap()
    });
    let mut validator = RemoteJournalValidator::new(endpoint, fixture.grant, 4);
    let object = Ref64::new(7, 1, Kind::Object);
    let accesses = [
        DeviceLaneAccess::read(0, 1, object, 0),
        DeviceLaneAccess::write(1, 1, object, 0),
    ];

    let first = validator.validate_lane_journals(&accesses, 2).unwrap();
    assert_eq!(
        validator.validate_lane_journals(&accesses, 2).unwrap(),
        first
    );
    assert_eq!(service.lock().unwrap().applied_requests(), 1);
    fixture
        .authority
        .lock()
        .unwrap()
        .revoke(fixture.grant.nonce);
    assert_eq!(
        validator.validate_lane_journals(&accesses, 2),
        Err(LaneValidationError::AuthorityDenied)
    );
    server.join().unwrap();
    assert_eq!(service.lock().unwrap().applied_requests(), 1);
}

#[test]
fn unavailable_and_accepted_then_lost_are_distinct() {
    let fixture = fixture();
    let endpoint = fixture.listener.local_addr().unwrap();
    drop(fixture.listener);
    let mut unavailable = RemoteJournalValidator::new(endpoint, fixture.grant, 0);
    assert_eq!(
        unavailable.validate_lane_journals(&[], 1),
        Err(LaneValidationError::Unavailable)
    );

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let dropper = std::thread::spawn(move || drop(listener.accept().unwrap()));
    let mut lost = RemoteJournalValidator::new(endpoint, fixture.grant, 0);
    assert_eq!(
        lost.validate_lane_journals(&[], 1),
        Err(LaneValidationError::NodeLost)
    );
    dropper.join().unwrap();
}

#[test]
fn malformed_operation_arena_is_rejected_by_the_worker() {
    let fixture = fixture();
    let endpoint = fixture.listener.local_addr().unwrap();
    let service = fixture.service.clone();
    let server = std::thread::spawn(move || {
        RemoteJournalServer::serve_n(fixture.listener, service, 1).unwrap()
    });
    let mut validator = RemoteJournalValidator::new(endpoint, fixture.grant, 0);
    let journal = DeviceOperationJournal {
        operations: vec![DeviceLaneOperation {
            lane: 0,
            ordinal: 0,
            opcode: 1,
            payload_offset: 4,
            payload_len: 8,
            ..DeviceLaneOperation::default()
        }],
        payload: vec![0; 4],
    };
    assert_eq!(
        validator.validate_epoch(&[], 1, &[&journal]),
        Err(LaneValidationError::InvalidInput)
    );
    server.join().unwrap();
}
