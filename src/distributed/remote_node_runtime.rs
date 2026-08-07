//! Small node-owned epoch runtime for cross-node future progress.
//!
//! This is deliberately a narrow first node runtime: it multiplexes a node's
//! owned future/channel services in a registry and couples their scheduling
//! bridges to one real `Kernel` epoch loop. Signed mailbox ingress is staged
//! by TCP threads and committed only through the real owner `Kernel` inbox at
//! its epoch boundary. Service state remains at its owning node; there is no coordinator containing
//! application state and no foreign descriptor is installed in a
//! client kernel. The canonical future is a node-runtime service adjacent to,
//! not a descriptor inside, the owner `Kernel`. Resolution below is an explicit
//! configured post-continuation hook (future resolve and channel send/receive),
//! not yet journaled `LaneView` remote operations.

use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, ThreadId};

use super::remote_channel::{
    RemoteChannelBridge, RemoteChannelBridgeError, RemoteChannelClient, RemoteChannelError,
    RemoteChannelServer, RemoteChannelService, RemoteChannelWaitKind, RemoteReceiveOutcome,
    RemoteSendOutcome,
};
use super::remote_future::{
    RemoteAwaitOutcome, RemoteFutureBridge, RemoteFutureBridgeError, RemoteFutureClient,
    RemoteFutureError, RemoteFutureServer, RemoteFutureService, RemoteFutureState,
};
use super::remote_lane_effect::{
    KernelRemoteLaneEmission, RemoteLaneApply, RemoteLaneClientRouter, RemoteLaneEffectService,
    RemoteLaneError, RemoteLaneOperation, RemoteLaneOutcome, RemoteLaneProgram,
    RemoteLaneRequestId, RemoteLaneValue,
};
use super::remote_lane_transport::{RemoteLaneClientSession, VerifiedRemoteLaneOutcomes};
use super::remote_mailbox_ingress::{
    RemoteMailboxApplyOutcome, RemoteMailboxError, RemoteMailboxIngress, RemoteMailboxServer,
};
use super::remote_process::{RemoteProcessError, RemoteProcessServer, RemoteProcessService};
use super::{NodeId, RemoteRef};
use crate::abi::Ref64;
use crate::kernel::Kernel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteNodeRuntimeError {
    WrongOwner,
    WrongNode,
    DuplicateResource,
    UnknownResource,
    Kernel(crate::kernel::RuntimeError),
    Remote(RemoteFutureError),
    Channel(RemoteChannelError),
    Mailbox(RemoteMailboxError),
    Process(RemoteProcessError),
    Lane(RemoteLaneError),
}
impl From<RemoteFutureBridgeError> for RemoteNodeRuntimeError {
    fn from(value: RemoteFutureBridgeError) -> Self {
        match value {
            RemoteFutureBridgeError::Kernel(e) => Self::Kernel(e),
            RemoteFutureBridgeError::Remote(e) => Self::Remote(e),
        }
    }
}
impl From<RemoteChannelBridgeError> for RemoteNodeRuntimeError {
    fn from(value: RemoteChannelBridgeError) -> Self {
        match value {
            RemoteChannelBridgeError::Kernel(e) => Self::Kernel(e),
            RemoteChannelBridgeError::Remote(e) => Self::Channel(e),
        }
    }
}

struct OwnedFuture {
    target: RemoteRef,
    endpoint: SocketAddr,
    service: Arc<Mutex<RemoteFutureService>>,
    server: Option<JoinHandle<std::io::Result<()>>>,
    shutdown: Arc<AtomicBool>,
}
struct ForeignFuture {
    target: RemoteRef,
    bridge: RemoteFutureBridge,
}
struct ResolutionEffect {
    continuation: Ref64,
    client: RemoteFutureClient,
    value: Ref64,
    resolution_epoch: Option<u32>,
}
struct OwnedChannel {
    target: RemoteRef,
    endpoint: SocketAddr,
    service: Arc<Mutex<RemoteChannelService>>,
    server: Option<JoinHandle<std::io::Result<()>>>,
    shutdown: Arc<AtomicBool>,
}
struct ChannelBridgeEntry {
    target: RemoteRef,
    bridge: RemoteChannelBridge,
}
enum ChannelEffectKind {
    Send {
        client: RemoteChannelClient,
        sequence: u64,
        value: Ref64,
    },
    Receive {
        client: RemoteChannelClient,
        sequence: u64,
    },
}
struct ChannelEffect {
    target: RemoteRef,
    continuation: Ref64,
    epoch: Option<u32>,
    kind: ChannelEffectKind,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteChannelEffectOutcome {
    Send {
        target: RemoteRef,
        outcome: RemoteSendOutcome,
    },
    Receive {
        target: RemoteRef,
        outcome: RemoteReceiveOutcome,
    },
}

struct OwnedProcessService {
    target: RemoteRef,
    endpoint: Option<SocketAddr>,
    service: Arc<Mutex<RemoteProcessService>>,
    server: Option<JoinHandle<std::io::Result<()>>>,
    shutdown: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RemoteLaneOperationKind {
    FutureAwait,
    FutureResolve,
    ChannelSend,
    ChannelReceive,
    ObjectWrite,
}
impl RemoteLaneOperationKind {
    fn from_operation(operation: &RemoteLaneOperation) -> Option<Self> {
        match operation {
            RemoteLaneOperation::FutureAwait => Some(Self::FutureAwait),
            RemoteLaneOperation::FutureResolve { .. } => Some(Self::FutureResolve),
            RemoteLaneOperation::ChannelSend { .. } => Some(Self::ChannelSend),
            RemoteLaneOperation::ChannelReceive { .. } => Some(Self::ChannelReceive),
            RemoteLaneOperation::ObjectWrite { .. } => Some(Self::ObjectWrite),
            _ => None,
        }
    }
    fn is_blocking(self) -> bool {
        matches!(
            self,
            Self::FutureAwait | Self::ChannelSend | Self::ChannelReceive
        )
    }
}
#[derive(Clone)]
struct RemoteLaneExpectedOutcome {
    request_id: RemoteLaneRequestId,
    target: RemoteRef,
    kind: RemoteLaneOperationKind,
}
#[derive(Clone)]
struct RemoteLaneWaitReceipt {
    continuation: Ref64,
    blocking_target: RemoteRef,
    blocking_kind: RemoteLaneOperationKind,
    /// The blocking request id is the stable identity of this entire emission.
    group_id: RemoteLaneRequestId,
    expected: Vec<RemoteLaneExpectedOutcome>,
    session_binding: Option<([u8; 16], NodeId, NodeId)>,
}

fn valid_remote_lane_shape(
    kind: RemoteLaneOperationKind,
    result: &Result<RemoteLaneApply, RemoteLaneError>,
) -> bool {
    match result {
        Err(_) | Ok(RemoteLaneApply::WouldBlock) => true,
        Ok(RemoteLaneApply::Applied(value)) => matches!(
            (kind, value),
            (
                RemoteLaneOperationKind::FutureAwait,
                RemoteLaneValue::Ref(_)
            ) | (
                RemoteLaneOperationKind::FutureResolve,
                RemoteLaneValue::Unit
            ) | (
                RemoteLaneOperationKind::ChannelReceive,
                RemoteLaneValue::Ref(_)
            ) | (RemoteLaneOperationKind::ChannelSend, RemoteLaneValue::Unit)
                | (
                    RemoteLaneOperationKind::ObjectWrite,
                    RemoteLaneValue::Version { .. }
                )
        ),
        Ok(RemoteLaneApply::Closed) => matches!(
            kind,
            RemoteLaneOperationKind::ChannelSend | RemoteLaneOperationKind::ChannelReceive
        ),
        Ok(RemoteLaneApply::Lost) => true,
    }
}

struct OwnedMailboxIngress {
    target: RemoteRef,
    endpoint: SocketAddr,
    ingress: Arc<Mutex<RemoteMailboxIngress>>,
    server: Option<JoinHandle<std::io::Result<()>>>,
    shutdown: Arc<AtomicBool>,
}

/// One kernel, one owner thread, and the resources/endpoints belonging to that
/// node. Configuration may occur before moving the runtime to its owner thread;
/// the first epoch/park operation binds ownership.
pub struct RemoteNodeRuntime {
    node: NodeId,
    kernel: Kernel,
    owner: Option<ThreadId>,
    owned_futures: Vec<OwnedFuture>,
    foreign_futures: Vec<ForeignFuture>,
    resolution_effects: Vec<ResolutionEffect>,
    owned_channels: Vec<OwnedChannel>,
    channel_bridges: Vec<ChannelBridgeEntry>,
    channel_effects: Vec<ChannelEffect>,
    channel_outcomes: Vec<RemoteChannelEffectOutcome>,
    owned_mailbox_ingresses: Vec<OwnedMailboxIngress>,
    owned_processes: Vec<OwnedProcessService>,
    mailbox_outcomes: Vec<RemoteMailboxApplyOutcome>,
    remote_lane_service: Option<RemoteLaneEffectService>,
    remote_lane_router: Option<RemoteLaneClientRouter>,
    remote_lane_outcomes: Vec<RemoteLaneOutcome>,
    outbound_remote_lane: Vec<KernelRemoteLaneEmission>,
    remote_lane_waiters: std::collections::HashMap<RemoteLaneRequestId, RemoteLaneWaitReceipt>,
}
impl RemoteNodeRuntime {
    pub fn new(node: NodeId, kernel: Kernel) -> Self {
        Self {
            node,
            kernel,
            owner: None,
            owned_futures: vec![],
            foreign_futures: vec![],
            resolution_effects: vec![],
            owned_channels: vec![],
            channel_bridges: vec![],
            channel_effects: vec![],
            channel_outcomes: vec![],
            owned_mailbox_ingresses: vec![],
            owned_processes: vec![],
            mailbox_outcomes: vec![],
            remote_lane_service: None,
            remote_lane_router: None,
            remote_lane_outcomes: vec![],
            outbound_remote_lane: vec![],
            remote_lane_waiters: std::collections::HashMap::new(),
        }
    }
    /// Install the one owner-side multiplexed remote effect boundary.  The
    /// router contains transport proxies only, never local ABI descriptors.
    pub fn install_remote_lane_owner(
        &mut self,
        service: RemoteLaneEffectService,
        router: RemoteLaneClientRouter,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if self.remote_lane_service.is_some() {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        self.remote_lane_service = Some(service);
        self.remote_lane_router = Some(router);
        Ok(())
    }

    pub fn install_remote_lane_program(
        &mut self,
        run_class: u32,
        program: RemoteLaneProgram,
    ) -> Result<(), RemoteNodeRuntimeError> {
        self.kernel
            .install_remote_lane_program(run_class, program)
            .map_err(RemoteNodeRuntimeError::Kernel)
    }
    /// Drain journals emitted only by real special Kernel continuation dispatch. The
    /// caller supplies transport to the target owner runtime.
    pub fn pending_outbound_remote_lane(&self) -> Vec<KernelRemoteLaneEmission> {
        self.outbound_remote_lane.clone()
    }
    /// Bind an emitted waiter to the exact authenticated transport session that
    /// will carry it. This must be called before exchanging the containing batch.
    /// Repeating the same binding is harmless; changing it is refused.
    pub fn bind_remote_lane_waiter_session(
        &mut self,
        request_id: RemoteLaneRequestId,
        session: &RemoteLaneClientSession,
    ) -> Result<(), RemoteNodeRuntimeError> {
        let receipt = self
            .remote_lane_waiters
            .get(&request_id)
            .cloned()
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?;
        if session.issuer != self.node {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if session.owner != receipt.blocking_target.node {
            return Err(RemoteNodeRuntimeError::WrongOwner);
        }
        let binding = (session.session_id, session.issuer, session.owner);
        // Refuse a conflicting route for any member before changing any member.
        if receipt.expected.iter().any(|expected| {
            self.remote_lane_waiters
                .get(&expected.request_id)
                .is_none_or(|member| {
                    member.group_id != receipt.group_id
                        || member
                            .session_binding
                            .is_some_and(|existing| existing != binding)
                })
        }) {
            return Err(RemoteNodeRuntimeError::Lane(
                RemoteLaneError::InvalidEnvelope,
            ));
        }
        for expected in &receipt.expected {
            self.remote_lane_waiters
                .get_mut(&expected.request_id)
                .expect("members were checked above")
                .session_binding = Some(binding);
        }
        Ok(())
    }

    /// Accept only outcomes authenticated and content-bound by the remote-lane
    /// session transport. Verification happens before this method can observe a
    /// receipt, and therefore before any continuation is woken or faulted.
    pub fn accept_authenticated_remote_lane_outcomes(
        &mut self,
        outcomes: VerifiedRemoteLaneOutcomes,
    ) -> Result<(), RemoteNodeRuntimeError> {
        let binding = outcomes.routing_binding();
        let (_, issuer, owner) = binding;
        if issuer != self.node || outcomes.outcomes().iter().any(|o| o.target.node != owner) {
            return Err(RemoteNodeRuntimeError::Lane(
                RemoteLaneError::InvalidEnvelope,
            ));
        }
        self.apply_remote_lane_outcomes(&outcomes.into_outcomes(), binding)
    }

    fn apply_remote_lane_outcomes(
        &mut self,
        outcomes: &[RemoteLaneOutcome],
        session_binding: ([u8; 16], NodeId, NodeId),
    ) -> Result<(), RemoteNodeRuntimeError> {
        let mut seen = std::collections::HashSet::new();
        let mut group_ids = Vec::new();
        for outcome in outcomes {
            if !seen.insert(outcome.request_id) {
                return Err(RemoteNodeRuntimeError::Lane(
                    RemoteLaneError::InvalidEnvelope,
                ));
            }
            let receipt = self.remote_lane_waiters.get(&outcome.request_id).ok_or(
                RemoteNodeRuntimeError::Lane(RemoteLaneError::InvalidEnvelope),
            )?;
            let expected = receipt
                .expected
                .iter()
                .find(|expected| expected.request_id == outcome.request_id)
                .ok_or(RemoteNodeRuntimeError::Lane(
                    RemoteLaneError::InvalidEnvelope,
                ))?;
            if receipt.session_binding != Some(session_binding)
                || outcome.target != expected.target
                || !valid_remote_lane_shape(expected.kind, &outcome.result)
            {
                return Err(RemoteNodeRuntimeError::Lane(
                    RemoteLaneError::InvalidEnvelope,
                ));
            }
            if !group_ids.contains(&receipt.group_id) {
                group_ids.push(receipt.group_id);
            }
        }
        // A verified transport frame must contain the complete result vector for
        // every emission it mentions. Missing members are an atomic refusal.
        for group_id in &group_ids {
            let receipt =
                self.remote_lane_waiters
                    .get(group_id)
                    .ok_or(RemoteNodeRuntimeError::Lane(
                        RemoteLaneError::InvalidEnvelope,
                    ))?;
            if receipt
                .expected
                .iter()
                .any(|expected| !seen.contains(&expected.request_id))
            {
                return Err(RemoteNodeRuntimeError::Lane(
                    RemoteLaneError::InvalidEnvelope,
                ));
            }
        }

        // Attempt all terminal group transitions on a clone.  A retryable member
        // keeps the complete group parked, including already-applied members.
        let mut kernel = self.kernel.clone();
        let mut terminal_groups = Vec::new();
        for group_id in group_ids {
            let receipt = self
                .remote_lane_waiters
                .get(&group_id)
                .expect("group validated above");
            let group_outcomes: Vec<_> = receipt
                .expected
                .iter()
                .map(|expected| {
                    outcomes
                        .iter()
                        .find(|outcome| outcome.request_id == expected.request_id)
                        .expect("complete group validated above")
                })
                .collect();
            if group_outcomes.iter().any(|outcome| {
                matches!(
                    outcome.result,
                    Ok(RemoteLaneApply::WouldBlock) | Err(RemoteLaneError::NodeUnavailable)
                )
            }) {
                continue;
            }
            let success = group_outcomes
                .iter()
                .all(|outcome| matches!(outcome.result, Ok(RemoteLaneApply::Applied(_))));
            if success {
                kernel
                    .complete_remote_lane_program(receipt.continuation)
                    .map_err(RemoteNodeRuntimeError::Kernel)?;
            } else {
                kernel
                    .fail_remote_lane_program(receipt.continuation)
                    .map_err(RemoteNodeRuntimeError::Kernel)?;
            }
            match receipt.blocking_kind {
                RemoteLaneOperationKind::FutureAwait => kernel.wake_remote_future_waiter(
                    receipt.continuation,
                    receipt.blocking_target.node.0,
                    receipt.blocking_target.entity,
                ),
                RemoteLaneOperationKind::ChannelSend | RemoteLaneOperationKind::ChannelReceive => {
                    kernel.wake_remote_channel_waiter(
                        receipt.continuation,
                        receipt.blocking_target.node.0,
                        receipt.blocking_target.entity,
                    )
                }
                _ => unreachable!("validated programs have exactly one blocking operation"),
            }
            terminal_groups.push(group_id);
        }
        if terminal_groups.is_empty() {
            return Ok(());
        }
        self.kernel = kernel;
        let mut removed = std::collections::HashSet::new();
        for group_id in terminal_groups {
            let receipt = self
                .remote_lane_waiters
                .get(&group_id)
                .expect("terminal group still present")
                .clone();
            for expected in receipt.expected {
                removed.insert(expected.request_id);
                self.remote_lane_waiters.remove(&expected.request_id);
            }
        }
        self.outbound_remote_lane.retain(|emission| {
            !emission
                .batch
                .effects()
                .iter()
                .any(|effect| removed.contains(&effect.request_id))
        });
        Ok(())
    }

    /// Convert an unrecoverable batch transport/stage refusal into an exact
    /// fault receipt; temporary failures should instead retain/retry the outbox.
    pub fn fail_outbound_remote_lane(
        &mut self,
        request_id: RemoteLaneRequestId,
    ) -> Result<(), RemoteNodeRuntimeError> {
        let receipt = self
            .remote_lane_waiters
            .get(&request_id)
            .cloned()
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?;
        self.kernel
            .fail_remote_lane_program(receipt.continuation)
            .map_err(RemoteNodeRuntimeError::Kernel)?;
        let ids: std::collections::HashSet<_> = receipt
            .expected
            .iter()
            .map(|expected| expected.request_id)
            .collect();
        for id in &ids {
            self.remote_lane_waiters.remove(id);
        }
        self.outbound_remote_lane.retain(|emission| {
            !emission
                .batch
                .effects()
                .iter()
                .any(|effect| ids.contains(&effect.request_id))
        });
        match receipt.blocking_kind {
            RemoteLaneOperationKind::FutureAwait => self.kernel.wake_remote_future_waiter(
                receipt.continuation,
                receipt.blocking_target.node.0,
                receipt.blocking_target.entity,
            ),
            RemoteLaneOperationKind::ChannelSend | RemoteLaneOperationKind::ChannelReceive => {
                self.kernel.wake_remote_channel_waiter(
                    receipt.continuation,
                    receipt.blocking_target.node.0,
                    receipt.blocking_target.entity,
                )
            }
            _ => unreachable!("validated programs have exactly one blocking operation"),
        }
        Ok(())
    }

    /// The validated special Kernel dispatch submits this protocol;
    /// this runtime deliberately does not expose an arbitrary host closure as a lane handler.
    /// Authenticate and freeze a journal now; mutation occurs only in
    /// `run_epoch` at the owner boundary.
    pub fn stage_remote_lane_effects(
        &mut self,
        frame: &[u8],
    ) -> Result<(), RemoteNodeRuntimeError> {
        let router = self
            .remote_lane_router
            .as_ref()
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?;
        self.remote_lane_service
            .as_mut()
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?
            .stage(frame, router)
            .map_err(RemoteNodeRuntimeError::Lane)
    }
    pub fn drain_remote_lane_outcomes(&mut self) -> Vec<RemoteLaneOutcome> {
        std::mem::take(&mut self.remote_lane_outcomes)
    }

    pub fn node(&self) -> NodeId {
        self.node
    }
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
    pub fn kernel_mut(&mut self) -> &mut Kernel {
        &mut self.kernel
    }
    pub fn owns(&self, target: RemoteRef) -> bool {
        self.owned_futures.iter().any(|r| r.target == target)
            || self.owned_channels.iter().any(|r| r.target == target)
            || self
                .owned_mailbox_ingresses
                .iter()
                .any(|r| r.target == target)
            || self.owned_processes.iter().any(|r| r.target == target)
    }
    pub fn owned_future_service(
        &self,
        target: RemoteRef,
    ) -> Option<&Arc<Mutex<RemoteFutureService>>> {
        self.owned_futures
            .iter()
            .find(|r| r.target == target)
            .map(|r| &r.service)
    }

    /// Install the canonical owner-side process lifecycle service. It has no
    /// client-side descriptor mirror; clients retain only remote receipts.
    pub fn register_owned_process_service(
        &mut self,
        target: RemoteRef,
        service: Arc<Mutex<RemoteProcessService>>,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if target.node != self.node {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.owns(target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        if service
            .lock()
            .map_err(|_| RemoteNodeRuntimeError::Process(RemoteProcessError::NodeLost))?
            .service_ref()
            != target
        {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        self.owned_processes.push(OwnedProcessService {
            target,
            endpoint: None,
            service,
            server: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        });
        Ok(())
    }
    pub fn register_owned_process_server(
        &mut self,
        target: RemoteRef,
        service: Arc<Mutex<RemoteProcessService>>,
        listener: TcpListener,
    ) -> Result<SocketAddr, RemoteNodeRuntimeError> {
        if target.node != self.node
            || service
                .lock()
                .map_err(|_| RemoteNodeRuntimeError::Process(RemoteProcessError::NodeLost))?
                .service_ref()
                != target
        {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.owns(target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        let endpoint = listener
            .local_addr()
            .map_err(|_| RemoteNodeRuntimeError::WrongNode)?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = shutdown.clone();
        let server_service = service.clone();
        let server = std::thread::spawn(move || {
            RemoteProcessServer::serve_until(listener, server_service, server_shutdown)
        });
        self.owned_processes.push(OwnedProcessService {
            target,
            endpoint: Some(endpoint),
            service,
            server: Some(server),
            shutdown,
        });
        Ok(endpoint)
    }
    pub fn owned_process_service(
        &self,
        target: RemoteRef,
    ) -> Option<&Arc<Mutex<RemoteProcessService>>> {
        self.owned_processes
            .iter()
            .find(|r| r.target == target)
            .map(|r| &r.service)
    }

    /// Register and start a bounded test/server lifetime for an owned future.
    /// The endpoint is resource-specific in this narrow slice; later registry
    /// protocol multiplexing can replace it without changing kernel coupling.
    pub fn register_owned_future(
        &mut self,
        target: RemoteRef,
        service: Arc<Mutex<RemoteFutureService>>,
        listener: TcpListener,
    ) -> Result<SocketAddr, RemoteNodeRuntimeError> {
        if target.node != self.node {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.owns(target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        let endpoint = listener
            .local_addr()
            .map_err(|_| RemoteNodeRuntimeError::WrongNode)?;
        let server_service = service.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = shutdown.clone();
        let server = std::thread::spawn(move || {
            RemoteFutureServer::serve_until(listener, server_service, server_shutdown)
        });
        self.owned_futures.push(OwnedFuture {
            target,
            endpoint,
            service,
            server: Some(server),
            shutdown,
        });
        Ok(endpoint)
    }
    pub fn endpoint(&self, target: RemoteRef) -> Option<SocketAddr> {
        self.owned_futures
            .iter()
            .find(|r| r.target == target)
            .map(|r| r.endpoint)
            .or_else(|| {
                self.owned_channels
                    .iter()
                    .find(|r| r.target == target)
                    .map(|r| r.endpoint)
            })
            .or_else(|| {
                self.owned_mailbox_ingresses
                    .iter()
                    .find(|r| r.target == target)
                    .map(|r| r.endpoint)
            })
            .or_else(|| {
                self.owned_processes
                    .iter()
                    .find(|r| r.target == target)
                    .and_then(|r| r.endpoint)
            })
    }
    /// Expose a local process inbox over signed TCP ingress. The server only
    /// validates and stages; `run_epoch` performs the canonical enqueue.
    pub fn register_mailbox_ingress(
        &mut self,
        target: RemoteRef,
        ingress: Arc<Mutex<RemoteMailboxIngress>>,
        listener: TcpListener,
    ) -> Result<SocketAddr, RemoteNodeRuntimeError> {
        if target.node != self.node {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.owns(target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        let endpoint = listener
            .local_addr()
            .map_err(|_| RemoteNodeRuntimeError::WrongNode)?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = shutdown.clone();
        let server_ingress = ingress.clone();
        let server = std::thread::spawn(move || {
            RemoteMailboxServer::serve_until(listener, server_ingress, server_shutdown)
        });
        self.owned_mailbox_ingresses.push(OwnedMailboxIngress {
            target,
            endpoint,
            ingress,
            server: Some(server),
            shutdown,
        });
        Ok(endpoint)
    }
    pub fn register_owned_mailbox(
        &mut self,
        target: RemoteRef,
        ingress: Arc<Mutex<RemoteMailboxIngress>>,
        listener: TcpListener,
    ) -> Result<SocketAddr, RemoteNodeRuntimeError> {
        self.register_mailbox_ingress(target, ingress, listener)
    }

    pub fn owned_mailbox_ingress(
        &self,
        target: RemoteRef,
    ) -> Option<&Arc<Mutex<RemoteMailboxIngress>>> {
        self.owned_mailbox_ingresses
            .iter()
            .find(|resource| resource.target == target)
            .map(|resource| &resource.ingress)
    }
    pub fn drain_mailbox_outcomes(&mut self) -> Vec<RemoteMailboxApplyOutcome> {
        std::mem::take(&mut self.mailbox_outcomes)
    }

    pub fn register_owned_channel(
        &mut self,
        target: RemoteRef,
        service: Arc<Mutex<RemoteChannelService>>,
        listener: TcpListener,
    ) -> Result<SocketAddr, RemoteNodeRuntimeError> {
        if target.node != self.node {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.owns(target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        let endpoint = listener
            .local_addr()
            .map_err(|_| RemoteNodeRuntimeError::WrongNode)?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = shutdown.clone();
        let server_service = service.clone();
        let server = std::thread::spawn(move || {
            RemoteChannelServer::serve_until(listener, server_service, server_shutdown)
        });
        self.owned_channels.push(OwnedChannel {
            target,
            endpoint,
            service,
            server: Some(server),
            shutdown,
        });
        Ok(endpoint)
    }
    pub fn owned_channel_service(
        &self,
        target: RemoteRef,
    ) -> Option<&Arc<Mutex<RemoteChannelService>>> {
        self.owned_channels
            .iter()
            .find(|r| r.target == target)
            .map(|r| &r.service)
    }
    pub fn register_channel_bridge(
        &mut self,
        target: RemoteRef,
        send_client: RemoteChannelClient,
        receive_client: RemoteChannelClient,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if send_client.target() != target || receive_client.target() != target {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.channel_bridges.iter().any(|r| r.target == target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        self.channel_bridges.push(ChannelBridgeEntry {
            target,
            bridge: RemoteChannelBridge::new(target, send_client, receive_client),
        });
        Ok(())
    }
    pub fn park_on_remote_channel(
        &mut self,
        target: RemoteRef,
        kind: RemoteChannelWaitKind,
        continuation: Ref64,
        next_run_class: u32,
    ) -> Result<(), RemoteNodeRuntimeError> {
        self.bind_owner()?;
        let entry = self
            .channel_bridges
            .iter_mut()
            .find(|r| r.target == target)
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?;
        entry
            .bridge
            .register(&mut self.kernel, kind, continuation, next_run_class)
            .map_err(Into::into)
    }
    pub fn send_after_continuation_runs(
        &mut self,
        target: RemoteRef,
        continuation: Ref64,
        client: RemoteChannelClient,
        sequence: u64,
        value: Ref64,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if client.target() != target {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        self.channel_effects.push(ChannelEffect {
            target,
            continuation,
            epoch: None,
            kind: ChannelEffectKind::Send {
                client,
                sequence,
                value,
            },
        });
        Ok(())
    }
    pub fn receive_after_continuation_runs(
        &mut self,
        target: RemoteRef,
        continuation: Ref64,
        client: RemoteChannelClient,
        sequence: u64,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if client.target() != target {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        self.channel_effects.push(ChannelEffect {
            target,
            continuation,
            epoch: None,
            kind: ChannelEffectKind::Receive { client, sequence },
        });
        Ok(())
    }
    pub fn drain_channel_outcomes(&mut self) -> Vec<RemoteChannelEffectOutcome> {
        std::mem::take(&mut self.channel_outcomes)
    }
    pub fn register_foreign_future(
        &mut self,
        target: RemoteRef,
        client: RemoteFutureClient,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if target.node == self.node {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        if self.foreign_futures.iter().any(|r| r.target == target) {
            return Err(RemoteNodeRuntimeError::DuplicateResource);
        }
        self.foreign_futures.push(ForeignFuture {
            target,
            bridge: RemoteFutureBridge::new(target, client),
        });
        Ok(())
    }
    pub fn park_on_remote_future(
        &mut self,
        target: RemoteRef,
        continuation: Ref64,
        next_run_class: u32,
    ) -> Result<RemoteAwaitOutcome, RemoteNodeRuntimeError> {
        self.bind_owner()?;
        let entry = self
            .foreign_futures
            .iter_mut()
            .find(|r| r.target == target)
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?;
        entry
            .bridge
            .await_at_epoch_boundary(&mut self.kernel, continuation, next_run_class)
            .map_err(Into::into)
    }
    /// Arrange a real local continuation execution to emit one idempotent
    /// resolution. Any terminal state counts as execution completion; semantic
    /// success/fault policy remains an application concern.
    pub fn resolve_after_continuation_runs(
        &mut self,
        target: RemoteRef,
        continuation: Ref64,
        client: RemoteFutureClient,
        value: Ref64,
    ) -> Result<(), RemoteNodeRuntimeError> {
        if target.node != self.node
            || !self
                .owned_futures
                .iter()
                .any(|resource| resource.target == target)
            || client.target() != target
        {
            return Err(RemoteNodeRuntimeError::WrongNode);
        }
        self.resolution_effects.push(ResolutionEffect {
            continuation,
            client,
            value,
            resolution_epoch: None,
        });
        Ok(())
    }
    /// Run one ordinary kernel epoch on the owner thread, apply completed local
    /// effects, then poll/wake foreign waiters at the new epoch boundary.
    pub fn run_epoch(
        &mut self,
    ) -> Result<Vec<(RemoteRef, RemoteFutureState)>, RemoteNodeRuntimeError> {
        self.bind_owner()?;
        // Phase A: freeze the validated network stage and commit it through the
        // real inbox before admission. Socket threads never touch the kernel.
        let boundary_epoch = self.kernel.current_epoch();
        if let (Some(service), Some(router)) =
            (&mut self.remote_lane_service, &mut self.remote_lane_router)
        {
            self.remote_lane_outcomes
                .extend(service.apply_epoch(boundary_epoch, router));
        }
        for resource in &self.owned_mailbox_ingresses {
            let mut ingress = resource
                .ingress
                .lock()
                .map_err(|_| RemoteNodeRuntimeError::Mailbox(RemoteMailboxError::NodeLost))?;
            self.mailbox_outcomes
                .extend(ingress.apply_boundary(&mut self.kernel, boundary_epoch));
        }
        for resource in &self.owned_processes {
            resource
                .service
                .lock()
                .map_err(|_| RemoteNodeRuntimeError::Process(RemoteProcessError::NodeLost))?
                .apply_boundary(&mut self.kernel, boundary_epoch)
                .map_err(RemoteNodeRuntimeError::Process)?;
        }
        let pending_bytes = self
            .outbound_remote_lane
            .iter()
            .map(|emission| emission.batch.encode().len())
            .sum();
        self.kernel
            .set_remote_lane_outbox_usage(self.outbound_remote_lane.len(), pending_bytes);
        self.kernel.run_epoch();
        let mut kernel = self.kernel.clone();
        let emissions = kernel.drain_remote_lane_emissions();
        let mut pending = Vec::with_capacity(emissions.len());
        let mut all_ids = std::collections::HashSet::new();
        // Validate every emission before mutating any waiter, receipt, or
        // outbox. Installed programs promise exactly one blocking member.
        for emission in emissions {
            let mut expected = Vec::with_capacity(emission.batch.effects().len());
            let mut blocking = Vec::new();
            let mut owner = None;
            for effect in emission.batch.effects() {
                let kind = RemoteLaneOperationKind::from_operation(&effect.operation).ok_or(
                    RemoteNodeRuntimeError::Lane(RemoteLaneError::InvalidProgram),
                )?;
                if !all_ids.insert(effect.request_id)
                    || self.remote_lane_waiters.contains_key(&effect.request_id)
                    || owner.is_some_and(|node| node != effect.target.node)
                {
                    return Err(RemoteNodeRuntimeError::Lane(
                        RemoteLaneError::InvalidProgram,
                    ));
                }
                owner = Some(effect.target.node);
                if kind.is_blocking() {
                    blocking.push((effect.request_id, effect.target, kind));
                }
                expected.push(RemoteLaneExpectedOutcome {
                    request_id: effect.request_id,
                    target: effect.target,
                    kind,
                });
            }
            if blocking.len() != 1 || expected.is_empty() {
                return Err(RemoteNodeRuntimeError::Lane(
                    RemoteLaneError::InvalidProgram,
                ));
            }
            let (group_id, blocking_target, blocking_kind) = blocking[0];
            let continuation = emission.continuation;
            pending.push((
                emission,
                RemoteLaneWaitReceipt {
                    continuation,
                    blocking_target,
                    blocking_kind,
                    group_id,
                    expected,
                    session_binding: None,
                },
            ));
        }

        // Waiter registration is fallible, so perform the complete publication
        // against clones and swap it into the live runtime only after all groups
        // have registered successfully.
        let mut waiters = self.remote_lane_waiters.clone();
        let mut outbox = self.outbound_remote_lane.clone();
        for (emission, receipt) in pending {
            match receipt.blocking_kind {
                RemoteLaneOperationKind::FutureAwait => kernel
                    .register_remote_future_waiter(
                        emission.continuation,
                        receipt.blocking_target.node.0,
                        receipt.blocking_target.entity,
                        emission.run_class,
                    )
                    .map_err(RemoteNodeRuntimeError::Kernel)?,
                RemoteLaneOperationKind::ChannelSend | RemoteLaneOperationKind::ChannelReceive => {
                    kernel
                        .register_remote_channel_waiter(
                            emission.continuation,
                            receipt.blocking_target.node.0,
                            receipt.blocking_target.entity,
                            emission.run_class,
                        )
                        .map_err(RemoteNodeRuntimeError::Kernel)?
                }
                _ => unreachable!(),
            }
            for member in &receipt.expected {
                waiters.insert(member.request_id, receipt.clone());
            }
            outbox.push(emission);
        }
        self.kernel = kernel;
        self.remote_lane_waiters = waiters;
        self.outbound_remote_lane = outbox;
        let epoch = self.kernel.current_epoch();
        for resource in &self.owned_processes {
            resource
                .service
                .lock()
                .map_err(|_| RemoteNodeRuntimeError::Process(RemoteProcessError::NodeLost))?
                .observe_after_epoch(&self.kernel, epoch)
                .map_err(RemoteNodeRuntimeError::Process)?;
        }
        let mut index = 0;
        while index < self.resolution_effects.len() {
            let ran = self
                .kernel
                .continuation_state(self.resolution_effects[index].continuation)
                .is_ok_and(|s| {
                    !matches!(
                        s,
                        crate::abi::continuations::ContinuationState::New
                            | crate::abi::continuations::ContinuationState::Runnable
                            | crate::abi::continuations::ContinuationState::Running
                            | crate::abi::continuations::ContinuationState::Waiting
                    )
                });
            if ran {
                let effect = &mut self.resolution_effects[index];
                let resolution_epoch = *effect.resolution_epoch.get_or_insert(epoch);
                effect.client.set_epoch(resolution_epoch);
                // Retain the exact content-addressed effect on ambiguous loss;
                // retrying uses the same epoch/value and the owner applies once.
                effect
                    .client
                    .resolve(effect.value)
                    .map_err(RemoteNodeRuntimeError::Remote)?;
                self.resolution_effects.swap_remove(index);
            } else {
                index += 1;
            }
        }
        let mut channel_index = 0;
        while channel_index < self.channel_effects.len() {
            let ran = self
                .kernel
                .continuation_state(self.channel_effects[channel_index].continuation)
                .is_ok_and(|s| {
                    !matches!(
                        s,
                        crate::abi::continuations::ContinuationState::New
                            | crate::abi::continuations::ContinuationState::Runnable
                            | crate::abi::continuations::ContinuationState::Running
                            | crate::abi::continuations::ContinuationState::Waiting
                    )
                });
            if !ran {
                channel_index += 1;
                continue;
            }
            let effect = &mut self.channel_effects[channel_index];
            let effect_epoch = *effect.epoch.get_or_insert(epoch);
            let outcome = match &mut effect.kind {
                ChannelEffectKind::Send {
                    client,
                    sequence,
                    value,
                } => {
                    client.set_epoch(effect_epoch);
                    RemoteChannelEffectOutcome::Send {
                        target: effect.target,
                        outcome: client
                            .send(*sequence, *value)
                            .map_err(RemoteNodeRuntimeError::Channel)?,
                    }
                }
                ChannelEffectKind::Receive { client, sequence } => {
                    client.set_epoch(effect_epoch);
                    RemoteChannelEffectOutcome::Receive {
                        target: effect.target,
                        outcome: client
                            .receive(*sequence)
                            .map_err(RemoteNodeRuntimeError::Channel)?,
                    }
                }
            };
            self.channel_outcomes.push(outcome);
            self.channel_effects.swap_remove(channel_index);
        }
        for entry in &mut self.channel_bridges {
            entry.bridge.sync_epoch_boundary(&mut self.kernel)?;
        }
        let mut observations = Vec::with_capacity(self.foreign_futures.len());
        for entry in &mut self.foreign_futures {
            observations.push((
                entry.target,
                entry.bridge.sync_epoch_boundary(&mut self.kernel)?,
            ));
        }
        Ok(observations)
    }
    pub fn join_servers(&mut self) -> std::io::Result<()> {
        for resource in &self.owned_futures {
            resource.shutdown.store(true, Ordering::Release);
        }
        for resource in &self.owned_channels {
            resource.shutdown.store(true, Ordering::Release);
        }
        for resource in &self.owned_mailbox_ingresses {
            resource.shutdown.store(true, Ordering::Release);
        }
        for resource in &self.owned_processes {
            resource.shutdown.store(true, Ordering::Release);
        }
        for resource in &mut self.owned_futures {
            if let Some(handle) = resource.server.take() {
                handle
                    .join()
                    .map_err(|_| std::io::Error::other("remote node server panicked"))??;
            }
        }
        for resource in &mut self.owned_channels {
            if let Some(handle) = resource.server.take() {
                handle
                    .join()
                    .map_err(|_| std::io::Error::other("remote node channel server panicked"))??;
            }
        }
        for resource in &mut self.owned_mailbox_ingresses {
            if let Some(handle) = resource.server.take() {
                handle
                    .join()
                    .map_err(|_| std::io::Error::other("remote mailbox server panicked"))??;
            }
        }
        for resource in &mut self.owned_processes {
            if let Some(handle) = resource.server.take() {
                handle
                    .join()
                    .map_err(|_| std::io::Error::other("remote process server panicked"))??;
            }
        }
        Ok(())
    }
    fn bind_owner(&mut self) -> Result<(), RemoteNodeRuntimeError> {
        let current = std::thread::current().id();
        match self.owner {
            None => {
                self.owner = Some(current);
                Ok(())
            }
            Some(owner) if owner == current => Ok(()),
            Some(_) => Err(RemoteNodeRuntimeError::WrongOwner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{valid_remote_lane_shape, RemoteLaneOperationKind};
    use crate::distributed::remote_lane_effect::{RemoteLaneApply, RemoteLaneValue};

    #[test]
    fn verified_terminal_shapes_accept_lost_but_not_future_closed() {
        assert!(valid_remote_lane_shape(
            RemoteLaneOperationKind::FutureAwait,
            &Ok(RemoteLaneApply::Lost),
        ));
        assert!(!valid_remote_lane_shape(
            RemoteLaneOperationKind::FutureAwait,
            &Ok(RemoteLaneApply::Closed),
        ));
        assert!(valid_remote_lane_shape(
            RemoteLaneOperationKind::ChannelReceive,
            &Ok(RemoteLaneApply::Closed),
        ));
        assert!(!valid_remote_lane_shape(
            RemoteLaneOperationKind::FutureAwait,
            &Ok(RemoteLaneApply::Applied(RemoteLaneValue::Unit)),
        ));
    }

    #[test]
    fn mixed_partial_terminal_group_remains_parked_and_malformed_is_atomic() {
        use crate::abi::{Kind, ProcessMode, Ref64, StateAccess};
        use crate::distributed::{NodeId, RemoteRef};
        use crate::kernel::{ContinuationSpec, Kernel, SYSTEM_PRINCIPAL};

        let worker = NodeId(900);
        let owner = NodeId(901);
        let mut kernel = Kernel::new();
        let actor = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
        let continuation = kernel
            .create_continuation(
                actor,
                actor,
                ContinuationSpec::new(StateAccess::ReadOnly, 9001, 9001, vec![], 2),
            )
            .unwrap();
        let future = RemoteRef {
            node: owner,
            entity: Ref64::new(1, 1, Kind::Future),
        };
        let object = RemoteRef {
            node: owner,
            entity: Ref64::new(2, 1, Kind::Object),
        };
        kernel
            .register_remote_future_waiter(continuation, owner.0, future.entity, 9001)
            .unwrap();
        let mut runtime = super::RemoteNodeRuntime::new(worker, kernel);
        let wait_id = crate::distributed::remote_lane_effect::RemoteLaneRequestId([1; 32]);
        let write_id = crate::distributed::remote_lane_effect::RemoteLaneRequestId([2; 32]);
        let binding = ([3; 16], worker, owner);
        let expected = vec![
            super::RemoteLaneExpectedOutcome {
                request_id: wait_id,
                target: future,
                kind: RemoteLaneOperationKind::FutureAwait,
            },
            super::RemoteLaneExpectedOutcome {
                request_id: write_id,
                target: object,
                kind: RemoteLaneOperationKind::ObjectWrite,
            },
        ];
        let receipt = super::RemoteLaneWaitReceipt {
            continuation,
            blocking_target: future,
            blocking_kind: RemoteLaneOperationKind::FutureAwait,
            group_id: wait_id,
            expected,
            session_binding: Some(binding),
        };
        runtime.remote_lane_waiters.insert(wait_id, receipt.clone());
        runtime.remote_lane_waiters.insert(write_id, receipt);

        let partial = vec![
            crate::distributed::remote_lane_effect::RemoteLaneOutcome {
                request_id: wait_id,
                target: future,
                result: Ok(RemoteLaneApply::Applied(RemoteLaneValue::Ref(object))),
            },
            crate::distributed::remote_lane_effect::RemoteLaneOutcome {
                request_id: write_id,
                target: object,
                result: Err(
                    crate::distributed::remote_lane_effect::RemoteLaneError::NodeUnavailable,
                ),
            },
        ];
        assert!(runtime
            .apply_remote_lane_outcomes(&partial[..1], binding)
            .is_err());
        assert_eq!(runtime.remote_lane_waiters.len(), 2);
        runtime
            .apply_remote_lane_outcomes(&partial, binding)
            .unwrap();
        assert_eq!(runtime.remote_lane_waiters.len(), 2);
        assert_eq!(
            runtime.kernel.continuation_state(continuation).unwrap(),
            crate::abi::ContinuationState::Waiting
        );

        let malformed = vec![
            partial[0].clone(),
            crate::distributed::remote_lane_effect::RemoteLaneOutcome {
                request_id: write_id,
                target: object,
                result: Ok(RemoteLaneApply::Applied(RemoteLaneValue::Unit)),
            },
        ];
        assert!(runtime
            .apply_remote_lane_outcomes(&malformed, binding)
            .is_err());
        assert_eq!(runtime.remote_lane_waiters.len(), 2);
        assert_eq!(
            runtime.kernel.continuation_state(continuation).unwrap(),
            crate::abi::ContinuationState::Waiting
        );
    }
}
