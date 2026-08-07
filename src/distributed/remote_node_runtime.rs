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
use super::remote_mailbox_ingress::{
    RemoteMailboxApplyOutcome, RemoteMailboxError, RemoteMailboxIngress, RemoteMailboxServer,
};
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
    mailbox_outcomes: Vec<RemoteMailboxApplyOutcome>,
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
            mailbox_outcomes: vec![],
        }
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
        for resource in &self.owned_mailbox_ingresses {
            let mut ingress = resource
                .ingress
                .lock()
                .map_err(|_| RemoteNodeRuntimeError::Mailbox(RemoteMailboxError::NodeLost))?;
            self.mailbox_outcomes
                .extend(ingress.apply_boundary(&mut self.kernel, boundary_epoch));
        }
        self.kernel.run_epoch();
        let epoch = self.kernel.current_epoch();
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
