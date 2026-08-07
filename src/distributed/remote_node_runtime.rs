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
    RemoteLaneRequestId,
};
use super::remote_lane_transport::VerifiedRemoteLaneOutcomes;
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

#[derive(Clone, Copy)]
enum RemoteLaneWaitKind {
    Future,
    Channel,
}
struct RemoteLaneWaitReceipt {
    continuation: Ref64,
    target: RemoteRef,
    kind: RemoteLaneWaitKind,
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
    /// Accept only outcomes authenticated and content-bound by the remote-lane
    /// session transport. Verification happens before this method can observe a
    /// receipt, and therefore before any continuation is woken or faulted.
    pub fn accept_authenticated_remote_lane_outcomes(
        &mut self,
        outcomes: VerifiedRemoteLaneOutcomes,
    ) -> Result<(), RemoteNodeRuntimeError> {
        self.apply_remote_lane_outcomes(&outcomes.into_outcomes())
    }

    fn apply_remote_lane_outcomes(
        &mut self,
        outcomes: &[RemoteLaneOutcome],
    ) -> Result<(), RemoteNodeRuntimeError> {
        for outcome in outcomes {
            if !self.remote_lane_waiters.contains_key(&outcome.request_id) {
                continue;
            }
            if matches!(
                outcome.result,
                Ok(RemoteLaneApply::WouldBlock) | Err(RemoteLaneError::NodeUnavailable)
            ) {
                continue;
            }
            let receipt = self
                .remote_lane_waiters
                .remove(&outcome.request_id)
                .expect("checked");
            if matches!(outcome.result, Ok(RemoteLaneApply::Applied(_))) {
                self.kernel
                    .complete_remote_lane_program(receipt.continuation)
                    .map_err(RemoteNodeRuntimeError::Kernel)?;
            } else {
                self.kernel
                    .fail_remote_lane_program(receipt.continuation)
                    .map_err(RemoteNodeRuntimeError::Kernel)?;
            }
            self.outbound_remote_lane.retain(|emission| {
                !emission
                    .batch
                    .effects()
                    .iter()
                    .any(|effect| effect.request_id == outcome.request_id)
            });
            match receipt.kind {
                RemoteLaneWaitKind::Future => self.kernel.wake_remote_future_waiter(
                    receipt.continuation,
                    receipt.target.node.0,
                    receipt.target.entity,
                ),
                RemoteLaneWaitKind::Channel => self.kernel.wake_remote_channel_waiter(
                    receipt.continuation,
                    receipt.target.node.0,
                    receipt.target.entity,
                ),
            }
        }
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
            .remove(&request_id)
            .ok_or(RemoteNodeRuntimeError::UnknownResource)?;
        self.kernel
            .fail_remote_lane_program(receipt.continuation)
            .map_err(RemoteNodeRuntimeError::Kernel)?;
        self.outbound_remote_lane.retain(|emission| {
            !emission
                .batch
                .effects()
                .iter()
                .any(|effect| effect.request_id == request_id)
        });
        match receipt.kind {
            RemoteLaneWaitKind::Future => self.kernel.wake_remote_future_waiter(
                receipt.continuation,
                receipt.target.node.0,
                receipt.target.entity,
            ),
            RemoteLaneWaitKind::Channel => self.kernel.wake_remote_channel_waiter(
                receipt.continuation,
                receipt.target.node.0,
                receipt.target.entity,
            ),
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
        for emission in self.kernel.drain_remote_lane_emissions() {
            // One blocking dependency per v1 program. Refuse ambiguous mixed
            // waits transactionally before publishing the outbound frame.
            let waits: Vec<_> = emission
                .batch
                .effects()
                .iter()
                .filter_map(|effect| match effect.operation {
                    RemoteLaneOperation::FutureAwait => {
                        Some((effect.request_id, effect.target, RemoteLaneWaitKind::Future))
                    }
                    RemoteLaneOperation::ChannelSend { .. }
                    | RemoteLaneOperation::ChannelReceive { .. } => Some((
                        effect.request_id,
                        effect.target,
                        RemoteLaneWaitKind::Channel,
                    )),
                    _ => None,
                })
                .collect();
            if waits.len() > 1 {
                return Err(RemoteNodeRuntimeError::Lane(
                    RemoteLaneError::InvalidProgram,
                ));
            }
            if let Some((id, target, kind)) = waits.first().copied() {
                match kind {
                    RemoteLaneWaitKind::Future => self
                        .kernel
                        .register_remote_future_waiter(
                            emission.continuation,
                            target.node.0,
                            target.entity,
                            emission.run_class,
                        )
                        .map_err(RemoteNodeRuntimeError::Kernel)?,
                    RemoteLaneWaitKind::Channel => self
                        .kernel
                        .register_remote_channel_waiter(
                            emission.continuation,
                            target.node.0,
                            target.entity,
                            emission.run_class,
                        )
                        .map_err(RemoteNodeRuntimeError::Kernel)?,
                }
                self.remote_lane_waiters.insert(
                    id,
                    RemoteLaneWaitReceipt {
                        continuation: emission.continuation,
                        target,
                        kind,
                    },
                );
            }
            self.outbound_remote_lane.push(emission);
        }
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
