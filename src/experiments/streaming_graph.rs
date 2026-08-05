//! Generic bounded streaming-graph validation workload.
//!
//! A source emits monotonically numbered records into one first-class channel;
//! a sink drains records whenever capacity applies back-pressure. An optional
//! source failure occurs only after a committed send, proving that the queued
//! prefix remains observable independently of the producer's lifetime.

use crate::abi::{ObjectKind, ProcessMode, ProcessState, Ref64, Rights, StateAccess};
use crate::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use crate::semantics::invariants::assert_legal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingGraphConfig {
    pub records: u32,
    pub channel_capacity: u32,
    pub fail_source_after: Option<u32>,
}

impl Default for StreamingGraphConfig {
    fn default() -> Self {
        Self {
            records: 32,
            channel_capacity: 4,
            fail_source_after: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamingGraphReport {
    pub attempted_records: u32,
    pub committed_records: u32,
    pub delivered: Vec<u64>,
    pub backpressure_events: u32,
    pub source_failed: bool,
}

impl StreamingGraphReport {
    pub fn ordered(&self) -> bool {
        self.delivered
            .iter()
            .copied()
            .eq(0..u64::from(self.committed_records))
    }
}

/// Execute the graph to quiescence using only public machine transitions.
pub fn run(config: StreamingGraphConfig) -> Result<StreamingGraphReport, RuntimeError> {
    let mut kernel = Kernel::new();
    let coordinator = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let source = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let sink = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let channel = kernel.create_channel(coordinator, config.channel_capacity);
    kernel.grant_capability(coordinator, source, channel, Rights::SEND, 0, 0)?;
    kernel.grant_capability(coordinator, sink, channel, Rights::RECEIVE, 0, 0)?;

    let failure_limit = config.fail_source_after.unwrap_or(config.records);
    let target = config.records.min(failure_limit);
    let mut delivered = Vec::new();
    let mut backpressure_events = 0;
    let mut committed = 0;

    for sequence in 0..target {
        let payload = kernel.create_object(
            source,
            ObjectKind::MessagePayload,
            u64::from(sequence).to_le_bytes().to_vec(),
        );
        loop {
            match kernel.send_channel(source, channel, payload, Ref64::NULL) {
                Ok(()) => {
                    committed += 1;
                    break;
                }
                Err(RuntimeError::MailboxFull) => {
                    backpressure_events += 1;
                    drain_one(&mut kernel, sink, channel, &mut delivered)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    let source_failed = if config.fail_source_after.is_some() {
        kernel.create_continuation(
            source,
            source,
            ContinuationSpec::new(StateAccess::ReadOnly, u32::MAX, 0, Vec::new(), 1),
        )?;
        kernel.run_epoch();
        kernel.process_state(source)? == ProcessState::Failed
    } else {
        false
    };
    kernel.close_channel(coordinator, channel)?;
    while kernel.channel_len(channel)? > 0 {
        drain_one(&mut kernel, sink, channel, &mut delivered)?;
    }
    assert_legal(&kernel);

    Ok(StreamingGraphReport {
        attempted_records: config.records,
        committed_records: committed,
        delivered,
        backpressure_events,
        source_failed,
    })
}

fn drain_one(
    kernel: &mut Kernel,
    sink: Ref64,
    channel: Ref64,
    delivered: &mut Vec<u64>,
) -> Result<(), RuntimeError> {
    let message = kernel
        .receive_channel(sink, channel, Ref64::NULL)?
        .ok_or(RuntimeError::MissingPayload)?;
    let bytes = kernel.object_bytes(sink, message.payload)?;
    let value = u64::from_le_bytes(
        bytes
            .get(..8)
            .ok_or(RuntimeError::MissingPayload)?
            .try_into()
            .map_err(|_| RuntimeError::MissingPayload)?,
    );
    delivered.push(value);
    Ok(())
}
