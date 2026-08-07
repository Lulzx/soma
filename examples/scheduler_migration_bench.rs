//! Real scheduler-overhead and end-to-end remote-migration measurements.
//!
//! Run in release mode; raw nanosecond trials are printed beside every
//! percentile summary so the report can be independently reanalysed.

use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use soma::abi::cohorts::PartialCohortPolicy;
use soma::abi::{Kind, ObjectKind, ProcessMode, Ref64, Rights, StateAccess};
use soma::compiler::examples;
use soma::distributed::authority::{GrantSpec, RemoteAuthorityStore};
use soma::distributed::remote_batch::{RemoteBatchBackend, RemoteBatchServer, RemoteBatchService};
use soma::distributed::{NodeId, RemoteRef};
use soma::executives::batch::{
    execute_with_spill, BatchBackend, CpuReferenceBackend, PlacementStats,
};
use soma::experiments::backend_bench::{synthetic_inputs, synthetic_program};
use soma::kernel::ownership::freeze;
use soma::kernel::{Kernel, SYSTEM_PRINCIPAL};
use soma::scheduler::admission::Candidate;

fn main() {
    println!(
        "SOMA scheduler/migration benchmark\narch={} os={} release={}\n",
        std::env::consts::ARCH,
        std::env::consts::OS,
        !cfg!(debug_assertions)
    );
    scheduler_benchmark();
    remote_benchmark();
}

fn summarize(name: &str, samples: &[Duration]) {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let at = |fraction: f64| {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index].as_secs_f64() * 1e6
    };
    println!(
        "{name}\tmedian_us={:.3}\tp10_us={:.3}\tp90_us={:.3}\traw_ns={}",
        at(0.5),
        at(0.1),
        at(0.9),
        samples
            .iter()
            .map(|sample| sample.as_nanos().to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
}

fn candidates(count: u32, classes: u32) -> Vec<Candidate> {
    (0..count)
        .map(|index| Candidate {
            bin: index % classes.max(1),
            continuation: Ref64::new(index + 1, 1, Kind::Continuation),
            process: Ref64::new(index + 1, 1, Kind::Process),
            run_class: index % classes.max(1),
            state_access: StateAccess::ReadOnly,
            waiting_since: index % 7,
        })
        .collect()
}

fn scheduler_benchmark() {
    println!("[scheduler planning: complete admission + bin/cohort placement]");
    for count in [32, 128, 512, 2_048] {
        for classes in [1, 4, 16] {
            let candidates = candidates(count, classes);
            let repetitions = if count <= 128 { 31 } else { 11 };
            let mut cpu = Vec::with_capacity(repetitions);
            for _ in 0..repetitions {
                let started = Instant::now();
                std::hint::black_box(soma::scheduler::device::reference_device_schedule(
                    &candidates,
                    32,
                    PartialCohortPolicy::RunPartial,
                ));
                cpu.push(started.elapsed());
            }
            summarize(&format!("cpu candidates={count} classes={classes}"), &cpu);

            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                use soma::executives::metal_scheduler::MetalDeviceScheduler;
                let mut metal = MetalDeviceScheduler::new().expect("Metal scheduler available");
                for _ in 0..3 {
                    metal
                        .schedule(&candidates, 32, PartialCohortPolicy::RunPartial)
                        .unwrap();
                }
                let mut samples = Vec::with_capacity(repetitions);
                for _ in 0..repetitions {
                    let started = Instant::now();
                    std::hint::black_box(
                        metal
                            .schedule(&candidates, 32, PartialCohortPolicy::RunPartial)
                            .unwrap(),
                    );
                    samples.push(started.elapsed());
                }
                summarize(
                    &format!("metal candidates={count} classes={classes}"),
                    &samples,
                );
            }
        }
    }
}

struct RemoteFixture {
    client: RemoteBatchBackend,
    server: std::thread::JoinHandle<()>,
}

fn remote_fixture(
    program: &soma::compiler::body::EvaluatorProgram,
    requests: usize,
) -> RemoteFixture {
    let issuer = NodeId(1);
    let worker = NodeId(2);
    let target = RemoteRef {
        node: worker,
        entity: Ref64::new(1, 1, Kind::Module),
    };
    let authority = Arc::new(Mutex::new(RemoteAuthorityStore::new(issuer, [0xC3; 32])));
    let grant = authority.lock().unwrap().issue(GrantSpec {
        audience: worker,
        actor: Ref64::new(1, 1, Kind::Process),
        target,
        rights: Rights::READ,
        object_version: 1,
        valid_from_epoch: 0,
        valid_until_epoch: u32::MAX,
    });
    let service = Arc::new(Mutex::new(RemoteBatchService::with(
        worker,
        target,
        1,
        authority,
        &[program],
    )));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        RemoteBatchServer::serve_n(listener, service, requests).unwrap()
    });
    let client = RemoteBatchBackend::with(endpoint, grant, 0, &[program]).unwrap();
    RemoteFixture { client, server }
}

fn remote_benchmark() {
    println!("\n[remote backend: loopback TCP, 32 ALU ops, 8-byte elements]");
    let program = synthetic_program(45_000, 2, 32);
    for count in [64, 4_096, 65_536] {
        let repetitions = if count < 65_536 { 21 } else { 9 };
        let total_remote = repetitions * 2 + 3;
        let mut fixture = remote_fixture(&program, total_remote);
        let inputs = synthetic_inputs(count, program.stride());
        let mut cpu = CpuReferenceBackend::with(&[&program]);
        for _ in 0..3 {
            fixture.client.set_epoch(1);
            fixture
                .client
                .evaluate(program.id(), &inputs, count, program.stride())
                .unwrap();
        }
        let mut local = Vec::with_capacity(repetitions);
        let mut remote = Vec::with_capacity(repetitions);
        for trial in 0..repetitions {
            let started = Instant::now();
            std::hint::black_box(
                cpu.evaluate(program.id(), &inputs, count, program.stride())
                    .unwrap(),
            );
            local.push(started.elapsed());

            fixture.client.set_epoch((trial + 10) as u32);
            let started = Instant::now();
            std::hint::black_box(
                fixture
                    .client
                    .evaluate(program.id(), &inputs, count, program.stride())
                    .unwrap(),
            );
            remote.push(started.elapsed());
        }
        summarize(&format!("local-backend elements={count}"), &local);
        summarize(&format!("remote-backend elements={count}"), &remote);

        let mut migration = Vec::with_capacity(repetitions);
        for trial in 0..repetitions {
            fixture.client.set_epoch((trial + 100) as u32);
            let mut kernel = Kernel::new();
            let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
            let input = kernel.create_object(owner, ObjectKind::FrozenArray, inputs.clone());
            freeze(&mut kernel, owner, input).unwrap();
            let mut stats = PlacementStats::default();
            let started = Instant::now();
            let (remote_collective, _) = kernel
                .create_batch_evaluate_for(owner, program.id(), input, count, program.stride())
                .unwrap();
            execute_with_spill(
                &mut kernel,
                owner,
                remote_collective,
                1,
                &mut fixture.client,
                &mut cpu,
                &mut stats,
            )
            .unwrap();
            let (cpu_collective, _) = kernel
                .create_batch_evaluate_for(owner, program.id(), input, count, program.stride())
                .unwrap();
            execute_with_spill(
                &mut kernel,
                owner,
                cpu_collective,
                u32::MAX,
                &mut fixture.client,
                &mut cpu,
                &mut stats,
            )
            .unwrap();
            migration.push(started.elapsed());
            assert_eq!(stats.remote_executions, 1);
            assert_eq!(stats.cpu_executions, 1);
            assert_eq!(stats.migrations, 1);
        }
        summarize(
            &format!("remote-to-cpu-full-publication elements={count}"),
            &migration,
        );
        fixture.server.join().unwrap();
    }

    // Keep the built-in example module linked in this benchmark too: it guards
    // against accidentally benchmarking only a synthetic private evaluator.
    std::hint::black_box(examples::DOUBLE_PLUS_ONE);
}
