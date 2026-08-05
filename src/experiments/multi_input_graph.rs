//! Irregular two-input join over first-class bounded channels.
//!
//! The left source starts immediately and the right source is delayed. Atomic
//! all-input receive must preserve both FIFO streams without consuming a lone
//! input, while the bounded queues apply measurable back-pressure. An optional
//! left-source failure verifies that the already joined prefix survives.

use crate::abi::{ObjectKind, ProcessMode, ProcessState, Ref64, Rights, StateAccess};
use crate::compiler::frame::Frame;
use crate::compiler::run_classes::SEARCH_BRANCH;
use crate::compiler::state_machine_lowering::SearchFrame;
use crate::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiInputConfig {
    pub records: usize,
    pub capacity: u32,
    pub right_delay: u32,
    pub fail_left_after: Option<usize>,
}

impl Default for MultiInputConfig {
    fn default() -> Self {
        Self {
            records: 16,
            capacity: 2,
            right_delay: 4,
            fail_left_after: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiInputReport {
    pub joined: Vec<(u64, u64)>,
    pub ordered: bool,
    pub left_backpressure: usize,
    pub right_backpressure: usize,
    pub left_state: ProcessState,
    pub right_state: ProcessState,
    pub committed_prefix_preserved: bool,
    pub legal: bool,
}

fn value(kernel: &mut Kernel, owner: Ref64, number: usize) -> Ref64 {
    kernel.create_object(
        owner,
        ObjectKind::MessagePayload,
        (number as u64).to_le_bytes().to_vec(),
    )
}

fn decode(kernel: &mut Kernel, actor: Ref64, object: Ref64) -> u64 {
    let bytes = kernel
        .object_bytes(actor, object)
        .expect("joined payload is readable");
    u64::from_le_bytes(bytes[..8].try_into().expect("payload schema is u64"))
}

fn fault_source(kernel: &mut Kernel, source: Ref64) {
    let mut frame = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut frame);
    kernel
        .create_continuation(
            source,
            source,
            ContinuationSpec::new(StateAccess::ReadOnly, SEARCH_BRANCH, 0, frame, 0),
        )
        .expect("source may create its faulting continuation");
    kernel.run_epoch();
}

pub fn run(config: MultiInputConfig) -> Result<MultiInputReport, RuntimeError> {
    if config.records == 0
        || config.capacity == 0
        || config
            .fail_left_after
            .is_some_and(|prefix| prefix > config.records)
    {
        return Err(RuntimeError::InvalidMultiInput);
    }

    let mut kernel = Kernel::new();
    let join = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let left = kernel.create_supervised_process(join, join, ProcessMode::Serial)?;
    let right = kernel.create_supervised_process(join, join, ProcessMode::Serial)?;
    let left_channel = kernel.create_channel(join, config.capacity);
    let right_channel = kernel.create_channel(join, config.capacity);
    kernel.grant_capability(join, left, left_channel, Rights::SEND, 0, 0)?;
    kernel.grant_capability(join, right, right_channel, Rights::SEND, 0, 0)?;

    let left_values = (0..config.records)
        .map(|number| value(&mut kernel, left, number))
        .collect::<Vec<_>>();
    let right_values = (0..config.records)
        .map(|number| value(&mut kernel, right, number))
        .collect::<Vec<_>>();
    let left_limit = config.fail_left_after.unwrap_or(config.records);
    let mut next_left = 0usize;
    let mut next_right = 0usize;
    let mut left_backpressure = 0usize;
    let mut right_backpressure = 0usize;
    let mut joined = Vec::new();
    let max_ticks = config.records.saturating_mul(8) + config.right_delay as usize + 64;

    for tick in 0..max_ticks {
        if next_left < left_limit && tick >= next_left {
            match kernel.send_channel(left, left_channel, left_values[next_left], Ref64::NULL) {
                Ok(()) => next_left += 1,
                Err(RuntimeError::MailboxFull) => left_backpressure += 1,
                Err(error) => return Err(error),
            }
        }
        if next_right < config.records
            && tick >= next_right.saturating_add(config.right_delay as usize)
        {
            match kernel.send_channel(right, right_channel, right_values[next_right], Ref64::NULL) {
                Ok(()) => next_right += 1,
                Err(RuntimeError::MailboxFull) => right_backpressure += 1,
                Err(error) => return Err(error),
            }
        }

        while let Some(messages) =
            kernel.receive_channels_all(join, &[left_channel, right_channel], Ref64::NULL)?
        {
            joined.push((
                decode(&mut kernel, join, messages[0].payload),
                decode(&mut kernel, join, messages[1].payload),
            ));
        }

        if config.fail_left_after == Some(joined.len()) && joined.len() == left_limit {
            fault_source(&mut kernel, left);
            kernel.close_channel(join, left_channel)?;
            kernel.close_channel(join, right_channel)?;
            break;
        }
        if config.fail_left_after.is_none() && joined.len() == config.records {
            break;
        }
    }

    let ordered = joined
        .iter()
        .enumerate()
        .all(|(index, pair)| *pair == (index as u64, index as u64));
    let expected = config.fail_left_after.unwrap_or(config.records);
    let report = MultiInputReport {
        committed_prefix_preserved: ordered && joined.len() == expected,
        ordered,
        joined,
        left_backpressure,
        right_backpressure,
        left_state: kernel.process_state(left)?,
        right_state: kernel.process_state(right)?,
        legal: crate::semantics::invariants::check(&kernel).is_empty(),
    };
    if report.joined.len() != expected {
        return Err(RuntimeError::InvalidMultiInput);
    }
    Ok(report)
}

pub fn report() -> String {
    let complete = run(MultiInputConfig::default()).expect("complete join");
    let failed = run(MultiInputConfig {
        fail_left_after: Some(7),
        ..MultiInputConfig::default()
    })
    .expect("failed-source join");
    format!(
        "complete joined={} ordered={} left_backpressure={} right_backpressure={}\nfailed   joined={} prefix_preserved={} left={:?} right={:?}\n",
        complete.joined.len(),
        complete.ordered,
        complete.left_backpressure,
        complete.right_backpressure,
        failed.joined.len(),
        failed.committed_prefix_preserved,
        failed.left_state,
        failed.right_state,
    )
}
