//! Message ABI (§11).

use super::refs::Ref64;
use super::AbiHeader;

/// Message descriptor (§11). Phase-1 guarantees: at-most-once delivery, ordered
/// delivery per sender–receiver pair, release on committed send, acquire on
/// receive.
#[derive(Clone, Debug)]
pub struct MessageDescriptor {
    pub header: AbiHeader,

    pub type_id: u32,
    pub flags: u32,

    pub sender: Ref64,
    pub receiver: Ref64,

    pub sender_sequence: u64,
    pub logical_timestamp: u64,

    pub payload: Ref64,               // CapRef
    pub transferred_capability: Ref64, // CapRef

    pub completion_future: Ref64,
}

impl MessageDescriptor {
    pub fn new(sender: Ref64, receiver: Ref64, payload: Ref64) -> MessageDescriptor {
        MessageDescriptor {
            header: AbiHeader::new(11, std::mem::size_of::<MessageDescriptor>() as u32),
            type_id: 0,
            flags: 0,
            sender,
            receiver,
            sender_sequence: 0,
            logical_timestamp: 0,
            payload,
            transferred_capability: Ref64::NULL,
            completion_future: Ref64::NULL,
        }
    }
}
