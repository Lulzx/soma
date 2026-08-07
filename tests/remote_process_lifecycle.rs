use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use soma::abi::{Kind, ProcessMode, ProcessState, Ref64, Rights, StateAccess};
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_node_runtime::RemoteNodeRuntime;
use soma::distributed::remote_process::{
    create_request_id, restart_request_id, RemoteProcessClient, RemoteProcessError,
    RemoteProcessResponse, RemoteProcessServer, RemoteProcessService, RemoteProcessStatus,
    RemoteProcessTcpClient, RemoteProcessTemplate, MAX_REMOTE_PROCESS_DURABLE_BYTES,
};
use soma::distributed::{NodeId, RemoteRef};
use soma::kernel::{ContinuationSpec, Kernel};
use soma::semantics::invariants::{check, Invariant};
use soma::semantics::order;

fn grant(
    store: &Arc<Mutex<RemoteAuthorityStore>>,
    audience: NodeId,
    target: RemoteRef,
    rights: u32,
    version: u32,
) -> soma::distributed::authority::RemoteGrant {
    store.lock().unwrap().issue(GrantSpec {
        audience,
        actor: Ref64::NULL,
        target,
        rights,
        object_version: version,
        valid_from_epoch: 0,
        valid_until_epoch: 100,
    })
}
fn assert_i1_i15(k: &Kernel) {
    let bad: Vec<_> = check(k)
        .into_iter()
        .filter(|v| {
            matches!(
                v.invariant,
                Invariant::ReferenceIntegrity | Invariant::SupervisionIntegrity
            )
        })
        .collect();
    assert!(bad.is_empty(), "{bad:?}");
}
fn temp_dir() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "soma-rps-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn two_kernel_create_fail_restart_crash_recover_exact_retry() {
    let owner = NodeId(41);
    // The caller is a genuinely independent kernel. Its only local process is
    // unrelated client work; remote lifecycle calls must not install a shadow.
    let mut client_kernel = Kernel::new();
    let cp = client_kernel.create_process(Ref64::NULL, ProcessMode::Serial);
    client_kernel
        .create_continuation(
            cp,
            cp,
            ContinuationSpec::new(StateAccess::ReadOnly, 996, 0, vec![], 1),
        )
        .unwrap();
    client_kernel.run_epoch();
    let mut i18_reference = Kernel::new();
    let rp = i18_reference.create_process(Ref64::NULL, ProcessMode::Serial);
    i18_reference
        .create_continuation(
            rp,
            rp,
            ContinuationSpec::new(StateAccess::ReadOnly, 996, 0, vec![], 1),
        )
        .unwrap();
    i18_reference.run_epoch();
    let client_processes = client_kernel.process_count();
    assert!(order::conforms(&i18_reference, &client_kernel).is_empty()); // I18, independent runs
    let client_runtime = RemoteNodeRuntime::new(NodeId(7), client_kernel);
    let service_ref = RemoteRef {
        node: owner,
        entity: Ref64::new(900, 3, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(owner, [9; 32])));
    let path = temp_dir();
    let service = Arc::new(Mutex::new(
        RemoteProcessService::open(owner, service_ref, 4, authority.clone(), &path).unwrap(),
    ));
    service
        .lock()
        .unwrap()
        .register_template(RemoteProcessTemplate {
            id: 5,
            mode: ProcessMode::Serial,
            // An unknown bounded run class deterministically faults in the real executive.
            entry: ContinuationSpec::new(StateAccess::ReadOnly, 999, 0, vec![], 2),
            restart_limit: 1,
        })
        .unwrap();
    service
        .lock()
        .unwrap()
        .register_template(RemoteProcessTemplate {
            id: 6,
            mode: ProcessMode::Serial,
            entry: ContinuationSpec::new(StateAccess::ReadOnly, 998, 0, vec![], 2),
            restart_limit: 0,
        })
        .unwrap();
    let mut owner_runtime = RemoteNodeRuntime::new(owner, Kernel::new());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = owner_runtime
        .register_owned_process_server(service_ref, service.clone(), listener)
        .unwrap();
    let client = RemoteProcessTcpClient::new(endpoint, owner);
    let create_grant = grant(&authority, owner, service_ref, Rights::WRITE, 4);
    let create_id = create_request_id(5, 0, &create_grant);
    assert_eq!(client.create(create_id, 5, 0, &create_grant).unwrap(), None);
    // A request id is content-bound, not a replay nonce for different bytes.
    assert_eq!(
        client.create(create_id, 6, 0, &create_grant),
        Err(RemoteProcessError::ProtocolError)
    );
    owner_runtime.run_epoch().unwrap();
    let first = match client
        .create(create_id, 5, 0, &create_grant)
        .unwrap()
        .unwrap()
    {
        RemoteProcessResponse::Created(x) => x,
        x => panic!("{x:?}"),
    };
    let read_first = grant(
        &authority,
        owner,
        first.process,
        Rights::READ,
        first.version,
    );
    assert!(matches!(
        client.query(first, 1, &read_first).unwrap().status,
        RemoteProcessStatus::Terminal(_)
    ));
    assert_i1_i15(owner_runtime.kernel());

    let restart_grant = grant(
        &authority,
        owner,
        first.process,
        Rights::WRITE,
        first.version,
    );
    let restart_id = restart_request_id(first, 1, &restart_grant);
    let staged_restart = (0..4)
        .find_map(
            |_| match client.restart(restart_id, first, 1, &restart_grant) {
                Ok(x) => Some(Ok(x)),
                Err(RemoteProcessError::NodeLost) => None, // ambiguous: exact content-addressed retry
                Err(e) => Some(Err(e)),
            },
        )
        .expect("bounded loopback retry")
        .unwrap();
    assert_eq!(staged_restart, None);
    owner_runtime.run_epoch().unwrap();
    let restarted = (0..4)
        .find_map(
            |_| match client.restart(restart_id, first, 1, &restart_grant) {
                Ok(Some(x)) => Some(Ok(x)),
                Ok(None) | Err(RemoteProcessError::NodeLost) => None,
                Err(e) => Some(Err(e)),
            },
        )
        .expect("bounded loopback result retry")
        .unwrap();
    let second = match restarted {
        RemoteProcessResponse::Restarted(x) => x,
        x => panic!("{x:?}"),
    };
    assert_eq!(second.restart_of, first.process);
    assert_eq!(second.restart_attempt, 1);
    let read_second = grant(
        &authority,
        owner,
        second.process,
        Rights::READ,
        second.version,
    );
    assert!(matches!(
        client.query(second, 2, &read_second).unwrap().status,
        RemoteProcessStatus::Terminal(_)
    ));
    assert_i1_i15(owner_runtime.kernel());
    assert_eq!(client_runtime.kernel().process_count(), client_processes); // no local descriptor shadow
    assert!(order::conforms(&i18_reference, client_runtime.kernel()).is_empty()); // I18 remains unchanged

    // Simulated owner crash: only disk plus the live authority registry survives.
    owner_runtime.join_servers().unwrap();
    drop(owner_runtime);
    drop(service);
    // Crash after WAL fsync but before snapshot replacement: the valid bounded
    // WAL image wins over a torn snapshot.
    std::fs::write(path.join("process.snapshot"), b"torn").unwrap();
    let recovered_service = Arc::new(Mutex::new(
        RemoteProcessService::open(owner, service_ref, 4, authority.clone(), &path).unwrap(),
    ));
    let mut recovered_kernel = Kernel::new();
    recovered_service
        .lock()
        .unwrap()
        .recover_kernel(&mut recovered_kernel)
        .unwrap();
    let mut recovered_runtime = RemoteNodeRuntime::new(owner, recovered_kernel);
    let recovered_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let recovered_endpoint = recovered_runtime
        .register_owned_process_server(service_ref, recovered_service.clone(), recovered_listener)
        .unwrap();
    let recovered_client = RemoteProcessTcpClient::new(recovered_endpoint, owner);
    // Exact retry survives restart and returns the byte-identical receipt.
    assert_eq!(
        recovered_client
            .restart(restart_id, first, 1, &restart_grant)
            .unwrap(),
        Some(RemoteProcessResponse::Restarted(second))
    );
    assert_eq!(recovered_service.lock().unwrap().ledger_len(), 2);
    assert_i1_i15(recovered_runtime.kernel());
    assert_eq!(
        recovered_runtime
            .kernel()
            .process_state(first.process.entity)
            .unwrap(),
        ProcessState::Failed
    );
    assert_eq!(
        recovered_runtime
            .kernel()
            .process_state(second.process.entity)
            .unwrap(),
        ProcessState::Failed
    );
    let recovered_read = grant(
        &authority,
        owner,
        second.process,
        Rights::READ,
        second.version,
    );
    assert_eq!(
        recovered_client
            .query(second, 2, &recovered_read)
            .unwrap()
            .status,
        RemoteProcessStatus::Terminal(soma::abi::ExitReason::Failed)
    );
    assert_eq!(client_runtime.kernel().process_count(), client_processes);
    recovered_runtime.join_servers().unwrap();
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn stale_receipt_auth_before_replay_and_node_qualified_collision() {
    let make = |node: NodeId| {
        let target = RemoteRef {
            node,
            entity: Ref64::new(77, 0, Kind::Object),
        };
        let auth = Arc::new(Mutex::new(RemoteAuthorityStore::new(
            node,
            [node.0 as u8; 32],
        )));
        let service = Arc::new(Mutex::new(
            RemoteProcessService::new(node, target, 1, auth.clone()).unwrap(),
        ));
        service
            .lock()
            .unwrap()
            .register_template(RemoteProcessTemplate {
                id: 1,
                mode: ProcessMode::Serial,
                entry: ContinuationSpec::new(StateAccess::ReadOnly, 997, 0, vec![], 1),
                restart_limit: 0,
            })
            .unwrap();
        (target, auth, service)
    };
    let (ta, aa, sa) = make(NodeId(1));
    let (tb, ab, sb) = make(NodeId(2));
    let ga = grant(&aa, NodeId(1), ta, Rights::WRITE, 1);
    let gb = grant(&ab, NodeId(2), tb, Rights::WRITE, 1);
    let ia = create_request_id(1, 0, &ga);
    let ib = create_request_id(1, 0, &gb);
    let ca = RemoteProcessClient::new(&sa);
    let cb = RemoteProcessClient::new(&sb);
    ca.create(ia, 1, 0, &ga).unwrap();
    cb.create(ib, 1, 0, &gb).unwrap();
    let mut ra = RemoteNodeRuntime::new(NodeId(1), Kernel::new());
    ra.register_owned_process_service(ta, sa.clone()).unwrap();
    ra.run_epoch().unwrap();
    let mut rb = RemoteNodeRuntime::new(NodeId(2), Kernel::new());
    rb.register_owned_process_service(tb, sb.clone()).unwrap();
    rb.run_epoch().unwrap();
    let pa = match ca.create(ia, 1, 0, &ga).unwrap().unwrap() {
        RemoteProcessResponse::Created(x) => x,
        _ => unreachable!(),
    };
    let pb = match cb.create(ib, 1, 0, &gb).unwrap().unwrap() {
        RemoteProcessResponse::Created(x) => x,
        _ => unreachable!(),
    };
    assert_eq!(pa.process.entity, pb.process.entity);
    assert_ne!(pa.process, pb.process);
    let mut stale = pa;
    stale.version += 1;
    let stale_grant = grant(&aa, NodeId(1), pa.process, Rights::READ, stale.version);
    assert_eq!(
        ca.query(stale, 1, &stale_grant),
        Err(RemoteProcessError::StaleReceipt)
    );
    let mut stale_ref = pa;
    stale_ref.process.entity.generation = stale_ref.process.entity.generation.wrapping_add(1);
    let stale_ref_grant = grant(
        &aa,
        NodeId(1),
        stale_ref.process,
        Rights::READ,
        stale_ref.version,
    );
    assert_eq!(
        ca.query(stale_ref, 1, &stale_ref_grant),
        Err(RemoteProcessError::StaleReceipt)
    );
    // Revocation is checked before a completed request is replayed.
    aa.lock().unwrap().revoke(ga.nonce);
    assert_eq!(
        ca.create(ia, 1, 0, &ga),
        Err(RemoteProcessError::AuthorityDenied)
    );
}

#[test]
fn oversized_durable_images_are_rejected_before_read() {
    let path = temp_dir();
    let file = std::fs::File::create(path.join("process.snapshot")).unwrap();
    file.set_len(MAX_REMOTE_PROCESS_DURABLE_BYTES + 41).unwrap();
    let node = NodeId(99);
    let target = RemoteRef {
        node,
        entity: Ref64::new(5, 0, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(node, [1; 32])));
    assert!(matches!(
        RemoteProcessService::open(node, target, 1, authority, &path),
        Err(RemoteProcessError::CapacityExceeded)
    ));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn tcp_transport_distinguishes_unavailable_and_ambiguous_loss() {
    let node = NodeId(55);
    let target = RemoteRef {
        node,
        entity: Ref64::new(5, 0, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(node, [3; 32])));
    let grant = grant(&authority, node, target, Rights::WRITE, 1);
    let id = create_request_id(1, 0, &grant);

    let unused = TcpListener::bind("127.0.0.1:0").unwrap();
    let unavailable_addr = unused.local_addr().unwrap();
    drop(unused);
    let unavailable = RemoteProcessTcpClient::new(unavailable_addr, node);
    assert_eq!(
        unavailable.create(id, 1, 0, &grant),
        Err(RemoteProcessError::NodeUnavailable)
    );

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let lost_addr = listener.local_addr().unwrap();
    let closer = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        drop(stream);
    });
    let lost = RemoteProcessTcpClient::new(lost_addr, node);
    assert_eq!(
        lost.create(id, 1, 0, &grant),
        Err(RemoteProcessError::NodeLost)
    );
    closer.join().unwrap();
}

#[test]
fn partial_header_peer_cannot_stall_process_server_shutdown() {
    let node = NodeId(81);
    let target = RemoteRef {
        node,
        entity: Ref64::new(8, 0, Kind::Object),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(node, [8; 32])));
    let service = Arc::new(Mutex::new(
        RemoteProcessService::new(node, target, 1, authority).unwrap(),
    ));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = shutdown.clone();
    let server = std::thread::spawn(move || {
        RemoteProcessServer::serve_until_with_timeout(
            listener,
            service,
            server_shutdown,
            Duration::from_millis(40),
        )
    });
    let mut peer = TcpStream::connect(endpoint).unwrap();
    peer.write_all(&[1, 2]).unwrap(); // less than the four-byte frame prefix
    let started = Instant::now();
    shutdown.store(true, Ordering::Release);
    server.join().unwrap().unwrap();
    assert!(started.elapsed() < Duration::from_secs(1));
}
