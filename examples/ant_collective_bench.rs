//! End-to-end wall benchmark for ant sensing implementations.
//!
//! Each sample builds a fresh persistent colony, supplies both sensing choices,
//! executes every colony/world continuation for every epoch, and observes the
//! complete final world. Runs rotate order to avoid always assigning warm-up to
//! one implementation. This is a benchmark, not a speedup claim: it prints raw
//! wall, backend, setup, and remaining host time.

use std::time::{Duration, Instant};

use soma::compiler::frame::Frame;
use soma::executives::batch::{
    AuxArray, BackendError, BackendKind, BatchBackend, BatchRequest, CpuReferenceBackend,
};
use soma::experiments::ant_colony::{
    build, observe_ants, read_frame, AntFrame, AntView, ColonyFrame, ColonyKnobs, WorldFrame,
};
use soma::experiments::ant_scoring::{
    prepare_colony_epoch, prepare_colony_epoch_timed, sensing_program, ColonySensing,
};
use soma::kernel::payload::Payload;

#[derive(Debug)]
struct TimedBackend<B> {
    inner: B,
    elapsed: Duration,
}

impl<B> TimedBackend<B> {
    fn new(inner: B) -> Self {
        Self {
            inner,
            elapsed: Duration::ZERO,
        }
    }
}

impl<B: BatchBackend> BatchBackend for TimedBackend<B> {
    fn kind(&self) -> BackendKind {
        self.inner.kind()
    }
    fn install(
        &mut self,
        program: &soma::compiler::body::EvaluatorProgram,
    ) -> Result<(), BackendError> {
        self.inner.install(program)
    }
    fn evaluate(
        &mut self,
        id: u32,
        input: &[u8],
        count: u32,
        stride: u32,
    ) -> Result<Vec<u8>, BackendError> {
        let start = Instant::now();
        let result = self.inner.evaluate(id, input, count, stride);
        self.elapsed += start.elapsed();
        result
    }
    fn evaluate_with_aux(
        &mut self,
        id: u32,
        input: &[u8],
        count: u32,
        stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Vec<u8>, BackendError> {
        let start = Instant::now();
        let result = self.inner.evaluate_with_aux(id, input, count, stride, aux);
        self.elapsed += start.elapsed();
        result
    }
    fn evaluate_payload(
        &mut self,
        id: u32,
        input: &[u8],
        count: u32,
        stride: u32,
        aux: AuxArray<'_>,
    ) -> Result<Payload, BackendError> {
        let start = Instant::now();
        let result = self.inner.evaluate_payload(id, input, count, stride, aux);
        self.elapsed += start.elapsed();
        result
    }
    fn evaluate_epoch(
        &mut self,
        requests: &[BatchRequest<'_>],
    ) -> Result<Vec<Payload>, BackendError> {
        let start = Instant::now();
        let result = self.inner.evaluate_epoch(requests);
        self.elapsed += start.elapsed();
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct World {
    ants: Vec<AntView>,
    /// Exact bytes of terrain, every persistent continuation frame, every ant
    /// deposit, every colony summary, and both field buffers.
    durable_objects: Vec<Vec<u8>>,
    accounting: soma::kernel::accounting::Accounting,
    pending: usize,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    wall: Duration,
    setup: Duration,
    backend: Duration,
}

impl Sample {
    fn host(self) -> Duration {
        self.wall.saturating_sub(self.setup + self.backend)
    }
}

fn finish(
    mut kernel: soma::kernel::Kernel,
    colony: &soma::experiments::ant_colony::AntColony,
) -> World {
    soma::semantics::invariants::assert_legal(&kernel);
    let mut durable_objects = vec![
        kernel
            .object_bytes(colony.world, colony.terrain)
            .unwrap()
            .to_vec(),
        kernel
            .object_bytes(colony.world, colony.field_a)
            .unwrap()
            .to_vec(),
        kernel
            .object_bytes(colony.world, colony.field_b)
            .unwrap()
            .to_vec(),
    ];
    let world_frame =
        read_frame::<WorldFrame>(&mut kernel, colony.world, colony.world_continuation)
            .expect("world frame exists");
    let mut encoded = Vec::new();
    world_frame.encode(&mut encoded);
    durable_objects.push(encoded);
    for group in &colony.colonies {
        let frame = read_frame::<ColonyFrame>(&mut kernel, group.process, group.continuation)
            .expect("colony frame exists");
        let mut encoded = Vec::new();
        frame.encode(&mut encoded);
        durable_objects.push(encoded);
        durable_objects.push(
            kernel
                .object_bytes(group.process, group.summary)
                .unwrap()
                .to_vec(),
        );
        for ant in &group.ants {
            let frame = read_frame::<AntFrame>(&mut kernel, ant.process, ant.continuation)
                .expect("ant frame exists");
            let mut encoded = Vec::new();
            frame.encode(&mut encoded);
            durable_objects.push(encoded);
            durable_objects.push(
                kernel
                    .object_bytes(ant.process, ant.deposit)
                    .unwrap()
                    .to_vec(),
            );
        }
    }
    World {
        ants: observe_ants(&mut kernel, colony),
        durable_objects,
        accounting: *kernel.accounting(),
        pending: kernel.total_pending(),
    }
}

fn run_host(knobs: &ColonyKnobs) -> Result<(Sample, World), BackendError> {
    let wall_start = Instant::now();
    let setup_start = Instant::now();
    let (mut kernel, colony) = build(knobs);
    let setup = setup_start.elapsed();
    let mut sensing = Duration::ZERO;
    for epoch in 0..knobs.epochs {
        let (scored, elapsed) =
            prepare_colony_epoch_timed(&mut kernel, &colony, epoch, ColonySensing::HostReference)?;
        sensing += elapsed;
        assert_eq!(scored, colony.ant_count());
        kernel.run_epoch();
    }
    let world = finish(kernel, &colony);
    Ok((
        Sample {
            wall: wall_start.elapsed(),
            setup,
            backend: sensing,
        },
        world,
    ))
}

fn run_collective<B: BatchBackend>(
    knobs: &ColonyKnobs,
    make: impl FnOnce() -> Result<B, BackendError>,
) -> Result<(Sample, World), BackendError> {
    let wall_start = Instant::now();
    let setup_start = Instant::now();
    let backend = make()?;
    let (mut kernel, colony) = build(knobs);
    let setup = setup_start.elapsed();
    let mut backend = TimedBackend::new(backend);
    for epoch in 0..knobs.epochs {
        let scored = prepare_colony_epoch(
            &mut kernel,
            &colony,
            epoch,
            ColonySensing::Collective(&mut backend),
        )?;
        assert_eq!(scored, colony.ant_count());
        kernel.run_epoch();
    }
    let backend_elapsed = backend.elapsed;
    let world = finish(kernel, &colony);
    Ok((
        Sample {
            wall: wall_start.elapsed(),
            setup,
            backend: backend_elapsed,
        },
        world,
    ))
}

fn raw(samples: &[Sample], field: impl Fn(Sample) -> Duration) -> String {
    samples
        .iter()
        .map(|s| field(*s).as_nanos().to_string())
        .collect::<Vec<_>>()
        .join(",")
}
fn median(samples: &[Sample]) -> Duration {
    let mut values: Vec<_> = samples.iter().map(|s| s.wall).collect();
    values.sort();
    values[values.len() / 2]
}
fn print_samples(label: &str, samples: &[Sample]) {
    println!(
        "{label:<16} median_ms={:.3} wall_ns={} setup_ns={} backend_ns={} host_ns={}",
        median(samples).as_secs_f64() * 1e3,
        raw(samples, |s| s.wall),
        raw(samples, |s| s.setup),
        raw(samples, |s| s.backend),
        raw(samples, |s| s.host())
    );
}

fn main() -> Result<(), BackendError> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let full = args.iter().any(|a| a == "--full");
    let trials = args
        .iter()
        .find_map(|a| a.parse::<usize>().ok())
        .unwrap_or(if full { 3 } else { 5 })
        .max(2);
    let knobs = if full {
        ColonyKnobs {
            colonies: 100,
            ants_per_colony: 100,
            width: 320,
            height: 320,
            epochs: 260,
            ..ColonyKnobs::default()
        }
    } else {
        ColonyKnobs {
            colonies: 4,
            ants_per_colony: 64,
            width: 96,
            height: 96,
            epochs: 40,
            ..ColonyKnobs::default()
        }
    };
    let program = sensing_program();
    let mut host = Vec::with_capacity(trials);
    let mut cpu = Vec::with_capacity(trials);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let mut metal = Vec::with_capacity(trials);
    let mut expected: Option<World> = None;

    // Rotate all implementations, rather than merely reversing two, so each
    // occupies every position in a three-way trial equally often.
    for trial in 0..trials {
        for position in 0..3 {
            let which = (trial + position) % 3;
            let result = match which {
                0 => run_host(&knobs).map(|x| ("host", x)),
                1 => run_collective(&knobs, || Ok(CpuReferenceBackend::with(&[&program])))
                    .map(|x| ("cpu", x)),
                2 => {
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    {
                        run_collective(&knobs, || {
                            soma::executives::metal::MetalBatchBackend::with(&[&program])
                        })
                        .map(|x| ("metal", x))
                    }
                    #[cfg(not(all(feature = "metal", target_os = "macos")))]
                    {
                        continue;
                    }
                }
                _ => unreachable!(),
            }?;
            let (label, (sample, world)) = result;
            if let Some(reference) = &expected {
                assert_eq!(reference, &world, "{label} changed the exact final world");
            } else {
                expected = Some(world);
            }
            match label {
                "host" => host.push(sample),
                "cpu" => cpu.push(sample),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                "metal" => metal.push(sample),
                _ => unreachable!(),
            }
        }
    }
    println!(
        "SOMA ant collective wall benchmark: ants={} epochs={} trials={trials}",
        knobs.ant_count(),
        knobs.epochs
    );
    println!("timing: wall includes backend construction, colony setup, all persistent steps, and final observation");
    println!("backend: host=sensing loop; collectives=physical backend call only; host=wall-setup-backend");
    print_samples("host-reference", &host);
    print_samples("cpu-collective", &cpu);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    print_samples("metal-collective", &metal);
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("metal-collective skipped (requires macOS and --features metal)");
    println!(
        "identical_world=true (all persistent object bytes, ant views, accounting, and pending count)"
    );
    Ok(())
}
