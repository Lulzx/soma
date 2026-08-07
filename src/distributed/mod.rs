//! Distributed execution primitives.
//!
//! Remote references name both the node that owns an entity and the ordinary
//! SOMA reference on that node. A reference is still not authority: every
//! request carries a signed, attenuated grant which the receiving node checks
//! against its live revocation registry at use.

pub mod authority;
pub mod remote_batch;
pub mod remote_channel;
pub mod remote_future;
pub mod remote_journal;
pub mod remote_lane_effect;
pub mod remote_lane_transport;
pub mod remote_mailbox_ingress;
pub mod remote_node_runtime;
pub mod remote_object;
pub mod remote_process;
pub mod remote_supervision;

use crate::abi::Ref64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RemoteRef {
    pub node: NodeId,
    pub entity: Ref64,
}
