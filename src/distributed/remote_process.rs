//! Owner-side, authenticated remote process lifecycle.
//!
//! Only pre-registered bounded continuation templates may be instantiated.  A
//! client holds a [`RemoteRef`] and versioned receipts; the `ProcessDescriptor`
//! remains exclusively in the owner's real [`Kernel`].  Requests are staged
//! and applied at owner epoch boundaries.  The small snapshot/WAL format is
//! deterministic and bounded, and includes the apply-once request ledger.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::authority::{RemoteAuthorityStore, RemoteGrant};
use super::{NodeId, RemoteRef};
use crate::abi::{ExitReason, ProcessMode, ProcessState, Ref64, Rights};
use crate::kernel::{ContinuationSpec, Kernel, RuntimeError, SYSTEM_PRINCIPAL};

pub const MAX_REMOTE_PROCESS_TEMPLATES: usize = 64;
pub const MAX_REMOTE_PROCESSES: usize = 4096;
pub const MAX_REMOTE_PROCESS_LEDGER: usize = 8192;
pub const MAX_REMOTE_PROCESS_FRAME: usize = 64 * 1024;
pub const MAX_REMOTE_PROCESS_DURABLE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct RemoteProcessTemplate {
    pub id: u32,
    pub mode: ProcessMode,
    pub entry: ContinuationSpec,
    pub restart_limit: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteProcessReceipt {
    pub process: RemoteRef,
    pub version: u32,
    pub template_id: u32,
    pub restart_of: RemoteRef,
    pub restart_attempt: u32,
    pub restart_limit: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteProcessStatus {
    Created,
    Runnable,
    Running,
    Waiting,
    Terminal(ExitReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteProcessObservation {
    pub receipt: RemoteProcessReceipt,
    pub status: RemoteProcessStatus,
    pub owner_epoch: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteProcessResponse {
    Created(RemoteProcessReceipt),
    Restarted(RemoteProcessReceipt),
    RestartBudgetExhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteProcessError {
    NodeUnavailable,
    NodeLost,
    ProtocolError,
    AuthorityDenied,
    WrongNode,
    UnknownTemplate,
    UnknownProcess,
    StaleReceipt,
    DuplicateTemplate,
    CapacityExceeded,
    RestartBudgetExhausted,
    Kernel(RuntimeError),
    Persistence,
    LiveRecoveryUnsupported,
}

#[derive(Clone)]
enum PendingKind {
    Create { template_id: u32 },
    Restart { receipt: RemoteProcessReceipt },
}
#[derive(Clone)]
struct Pending {
    id: [u8; 32],
    kind: PendingKind,
}
#[derive(Clone)]
struct Record {
    receipt: RemoteProcessReceipt,
    status: RemoteProcessStatus,
    owner_epoch: u32,
}

/// Canonical owner state. Socket/RPC adapters may stage calls concurrently;
/// only `apply_boundary` is allowed to touch the kernel.
#[derive(Clone)]
pub struct RemoteProcessService {
    node: NodeId,
    service_ref: RemoteRef,
    service_version: u32,
    authority: Arc<Mutex<RemoteAuthorityStore>>,
    templates: BTreeMap<u32, RemoteProcessTemplate>,
    records: BTreeMap<u64, Record>,
    pending: VecDeque<Pending>,
    ledger: HashMap<[u8; 32], RemoteProcessResponse>,
    ledger_order: VecDeque<[u8; 32]>,
    store: Option<PathBuf>,
}

/// Content address for a create request. Exact retry means repeating all of
/// these bytes, not merely recycling a caller-chosen nonce.
pub fn create_request_id(template_id: u32, epoch: u32, grant: &RemoteGrant) -> [u8; 32] {
    let mut b = Vec::new();
    b.extend_from_slice(b"create");
    b.extend_from_slice(&template_id.to_le_bytes());
    b.extend_from_slice(&epoch.to_le_bytes());
    b.extend_from_slice(&grant.encode());
    Sha256::digest(b).into()
}
pub fn restart_request_id(
    receipt: RemoteProcessReceipt,
    epoch: u32,
    grant: &RemoteGrant,
) -> [u8; 32] {
    let mut b = Vec::new();
    b.extend_from_slice(b"restart");
    encode_receipt(&mut b, receipt);
    b.extend_from_slice(&epoch.to_le_bytes());
    b.extend_from_slice(&grant.encode());
    Sha256::digest(b).into()
}

impl RemoteProcessService {
    pub fn new(
        node: NodeId,
        service_ref: RemoteRef,
        service_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
    ) -> Result<Self, RemoteProcessError> {
        if service_ref.node != node {
            return Err(RemoteProcessError::WrongNode);
        }
        Ok(Self {
            node,
            service_ref,
            service_version,
            authority,
            templates: BTreeMap::new(),
            records: BTreeMap::new(),
            pending: VecDeque::new(),
            ledger: HashMap::new(),
            ledger_order: VecDeque::new(),
            store: None,
        })
    }

    /// Open durable state. Templates and records are restored from the newest
    /// valid snapshot/WAL image. The durable scope is templates, terminal
    /// receipts/lineage, and the request ledger; live continuation checkpoint
    /// recovery is explicitly unsupported.
    pub fn open(
        node: NodeId,
        service_ref: RemoteRef,
        service_version: u32,
        authority: Arc<Mutex<RemoteAuthorityStore>>,
        directory: impl AsRef<Path>,
    ) -> Result<Self, RemoteProcessError> {
        let mut s = Self::new(node, service_ref, service_version, authority)?;
        fs::create_dir_all(directory.as_ref()).map_err(|_| RemoteProcessError::Persistence)?;
        s.store = Some(directory.as_ref().to_path_buf());
        if let Some(bytes) = load_latest(directory.as_ref())? {
            s.decode_state(&bytes)?;
        }
        Ok(s)
    }

    pub fn service_ref(&self) -> RemoteRef {
        self.service_ref
    }
    pub fn register_template(
        &mut self,
        template: RemoteProcessTemplate,
    ) -> Result<(), RemoteProcessError> {
        if template.entry.frame_bytes.len() > MAX_REMOTE_PROCESS_FRAME
            || template.entry.max_steps == 0
        {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        if self.templates.contains_key(&template.id) {
            return Err(RemoteProcessError::DuplicateTemplate);
        }
        if self.templates.len() >= MAX_REMOTE_PROCESS_TEMPLATES {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        self.templates.insert(template.id, template);
        self.persist()?;
        Ok(())
    }

    /// Stage a creation. Authorization precedes the ledger lookup deliberately:
    /// revocation remains effective even for an exact replay.
    pub fn stage_create(
        &mut self,
        request_id: [u8; 32],
        template_id: u32,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        self.authorize(
            grant,
            self.service_ref,
            Rights::WRITE,
            self.service_version,
            epoch,
        )?;
        if request_id != create_request_id(template_id, epoch, grant) {
            return Err(RemoteProcessError::ProtocolError);
        }
        if let Some(r) = self.ledger.get(&request_id) {
            return Ok(Some(*r));
        }
        if !self.templates.contains_key(&template_id) {
            return Err(RemoteProcessError::UnknownTemplate);
        }
        if !self.pending.iter().any(|p| p.id == request_id) {
            self.pending.push_back(Pending {
                id: request_id,
                kind: PendingKind::Create { template_id },
            });
        }
        Ok(None)
    }

    pub fn stage_restart(
        &mut self,
        request_id: [u8; 32],
        receipt: RemoteProcessReceipt,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        self.authorize(
            grant,
            receipt.process,
            Rights::WRITE,
            receipt.version,
            epoch,
        )?;
        if request_id != restart_request_id(receipt, epoch, grant) {
            return Err(RemoteProcessError::ProtocolError);
        }
        self.validate_receipt(receipt)?;
        if let Some(r) = self.ledger.get(&request_id) {
            return Ok(Some(*r));
        }
        if !self.pending.iter().any(|p| p.id == request_id) {
            self.pending.push_back(Pending {
                id: request_id,
                kind: PendingKind::Restart { receipt },
            });
        }
        Ok(None)
    }

    pub fn query(
        &self,
        receipt: RemoteProcessReceipt,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<RemoteProcessObservation, RemoteProcessError> {
        self.authorize(grant, receipt.process, Rights::READ, receipt.version, epoch)?;
        let r = self.validate_receipt(receipt)?;
        Ok(RemoteProcessObservation {
            receipt: r.receipt,
            status: r.status,
            owner_epoch: r.owner_epoch,
        })
    }

    pub fn process_count(&self) -> usize {
        self.records.len()
    }
    pub fn ledger_len(&self) -> usize {
        self.ledger.len()
    }

    pub fn apply_boundary(
        &mut self,
        kernel: &mut Kernel,
        epoch: u32,
    ) -> Result<(), RemoteProcessError> {
        let novel = self
            .pending
            .iter()
            .filter(|p| !self.ledger.contains_key(&p.id))
            .count();
        if self.ledger.len().saturating_add(novel) > MAX_REMOTE_PROCESS_LEDGER {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        let mut next = self.clone();
        let mut next_kernel = kernel.clone();
        next.apply_boundary_inner(&mut next_kernel, epoch)?;
        next.persist()?;
        *kernel = next_kernel;
        *self = next;
        Ok(())
    }

    fn apply_boundary_inner(
        &mut self,
        kernel: &mut Kernel,
        epoch: u32,
    ) -> Result<(), RemoteProcessError> {
        while let Some(p) = self.pending.pop_front() {
            if self.ledger.contains_key(&p.id) {
                continue;
            }
            let response = match p.kind {
                PendingKind::Create { template_id } => {
                    let receipt = self.instantiate(kernel, template_id, None, 0)?;
                    RemoteProcessResponse::Created(receipt)
                }
                PendingKind::Restart { receipt } => {
                    let old = self.validate_receipt(receipt)?.clone();
                    if !matches!(old.status, RemoteProcessStatus::Terminal(_)) {
                        return Err(RemoteProcessError::ProtocolError);
                    }
                    if old.receipt.restart_attempt >= old.receipt.restart_limit {
                        RemoteProcessResponse::RestartBudgetExhausted
                    } else {
                        let next = self.instantiate(
                            kernel,
                            old.receipt.template_id,
                            Some(old.receipt.process),
                            old.receipt.restart_attempt + 1,
                        )?;
                        RemoteProcessResponse::Restarted(next)
                    }
                }
            };
            self.insert_ledger(p.id, response)?;
        }
        for record in self.records.values_mut() {
            if matches!(record.status, RemoteProcessStatus::Terminal(_)) {
                continue;
            }
            let state = kernel
                .process_state(record.receipt.process.entity)
                .map_err(RemoteProcessError::Kernel)?;
            record.status = map_state(state);
            record.owner_epoch = epoch;
        }
        Ok(())
    }

    /// Refresh terminal state after the real kernel has completed the epoch.
    pub fn observe_after_epoch(
        &mut self,
        kernel: &Kernel,
        epoch: u32,
    ) -> Result<(), RemoteProcessError> {
        let mut next = self.clone();
        for record in next.records.values_mut() {
            if matches!(record.status, RemoteProcessStatus::Terminal(_)) {
                continue;
            }
            let state = kernel
                .process_state(record.receipt.process.entity)
                .map_err(RemoteProcessError::Kernel)?;
            record.status = map_state(state);
            record.owner_epoch = epoch;
        }
        next.persist()?;
        *self = next;
        Ok(())
    }

    fn instantiate(
        &mut self,
        kernel: &mut Kernel,
        template_id: u32,
        restart_of: Option<RemoteRef>,
        attempt: u32,
    ) -> Result<RemoteProcessReceipt, RemoteProcessError> {
        if self.records.len() >= MAX_REMOTE_PROCESSES {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        let template = self
            .templates
            .get(&template_id)
            .ok_or(RemoteProcessError::UnknownTemplate)?
            .clone();
        let process = kernel
            .create_process_on_node(SYSTEM_PRINCIPAL, template.mode, self.node.0)
            .map_err(RemoteProcessError::Kernel)?;
        kernel
            .create_continuation(process, process, template.entry)
            .map_err(RemoteProcessError::Kernel)?;
        let remote = RemoteRef {
            node: self.node,
            entity: process,
        };
        let receipt = RemoteProcessReceipt {
            process: remote,
            version: 1,
            template_id,
            restart_of: restart_of.unwrap_or(RemoteRef {
                node: self.node,
                entity: Ref64::NULL,
            }),
            restart_attempt: attempt,
            restart_limit: template.restart_limit,
        };
        self.records.insert(
            process.to_u64(),
            Record {
                receipt,
                status: RemoteProcessStatus::Runnable,
                owner_epoch: kernel.current_epoch(),
            },
        );
        Ok(receipt)
    }

    fn validate_receipt(
        &self,
        receipt: RemoteProcessReceipt,
    ) -> Result<&Record, RemoteProcessError> {
        if receipt.process.node != self.node {
            return Err(RemoteProcessError::WrongNode);
        }
        let Some(r) = self.records.get(&receipt.process.entity.to_u64()) else {
            let e = receipt.process.entity;
            if self.records.values().any(|record| {
                let known = record.receipt.process.entity;
                known.slot == e.slot && known.kind == e.kind && known.partition == e.partition
            }) {
                return Err(RemoteProcessError::StaleReceipt);
            }
            return Err(RemoteProcessError::UnknownProcess);
        };
        if r.receipt != receipt {
            return Err(RemoteProcessError::StaleReceipt);
        }
        Ok(r)
    }
    fn authorize(
        &self,
        grant: &RemoteGrant,
        target: RemoteRef,
        rights: u32,
        version: u32,
        epoch: u32,
    ) -> Result<(), RemoteProcessError> {
        let ok = self.authority.lock().ok().is_some_and(|a| {
            a.authorize(grant, self.node, target, rights, version, epoch)
                .is_ok()
        });
        if ok {
            Ok(())
        } else {
            Err(RemoteProcessError::AuthorityDenied)
        }
    }
    fn insert_ledger(
        &mut self,
        id: [u8; 32],
        response: RemoteProcessResponse,
    ) -> Result<(), RemoteProcessError> {
        if self.ledger.len() >= MAX_REMOTE_PROCESS_LEDGER {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        self.ledger.insert(id, response);
        self.ledger_order.push_back(id);
        Ok(())
    }

    /// Rebuild allocation identities for terminal records after a simulated
    /// owner crash. Live work is rejected rather than restarted from an
    /// uncommitted template frame. The service remains the canonical terminal
    /// record; no foreign descriptor is manufactured.
    pub fn recover_kernel(&self, kernel: &mut Kernel) -> Result<(), RemoteProcessError> {
        if self
            .records
            .values()
            .any(|r| !matches!(r.status, RemoteProcessStatus::Terminal(_)))
        {
            return Err(RemoteProcessError::LiveRecoveryUnsupported);
        }
        let mut recovered = kernel.clone();
        for record in self.records.values() {
            let t = self
                .templates
                .get(&record.receipt.template_id)
                .ok_or(RemoteProcessError::ProtocolError)?;
            let p = recovered
                .create_process_on_node(SYSTEM_PRINCIPAL, t.mode, self.node.0)
                .map_err(RemoteProcessError::Kernel)?;
            if p != record.receipt.process.entity {
                return Err(RemoteProcessError::ProtocolError);
            }
            let RemoteProcessStatus::Terminal(reason) = record.status else {
                unreachable!()
            };
            recovered
                .restore_remote_process_terminal(p, reason)
                .map_err(RemoteProcessError::Kernel)?;
        }
        *kernel = recovered;
        Ok(())
    }

    fn validate_durable_bound(&self) -> Result<(), RemoteProcessError> {
        // Fixed record/ledger overhead is deliberately overestimated. This is
        // checked before allocating the encoded image.
        let frames = self
            .templates
            .values()
            .try_fold(0u64, |n, template| {
                n.checked_add(template.entry.frame_bytes.len() as u64)
            })
            .ok_or(RemoteProcessError::CapacityExceeded)?;
        let fixed = 64u64
            .checked_add((self.templates.len() as u64).saturating_mul(64))
            .and_then(|n| n.checked_add((self.records.len() as u64).saturating_mul(96)))
            .and_then(|n| n.checked_add((self.ledger.len() as u64).saturating_mul(128)))
            .ok_or(RemoteProcessError::CapacityExceeded)?;
        if frames
            .checked_add(fixed)
            .is_none_or(|n| n > MAX_REMOTE_PROCESS_DURABLE_BYTES)
        {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        Ok(())
    }

    fn persist(&self) -> Result<(), RemoteProcessError> {
        let Some(dir) = &self.store else {
            return Ok(());
        };
        self.validate_durable_bound()?;
        let payload = self.encode_state();
        if payload.len() as u64 > MAX_REMOTE_PROCESS_DURABLE_BYTES {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        let mut wal = Vec::with_capacity(40 + payload.len());
        wal.extend_from_slice(b"RPSW");
        wal.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        wal.extend_from_slice(&Sha256::digest(&payload));
        wal.extend_from_slice(&payload);
        let wal_path = dir.join("process.wal");
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&wal_path)
            .map_err(|_| RemoteProcessError::Persistence)?;
        f.write_all(&wal)
            .and_then(|_| f.sync_all())
            .map_err(|_| RemoteProcessError::Persistence)?;
        let tmp = dir.join("process.snapshot.tmp");
        fs::write(&tmp, &payload).map_err(|_| RemoteProcessError::Persistence)?;
        fs::rename(tmp, dir.join("process.snapshot"))
            .map_err(|_| RemoteProcessError::Persistence)?;
        Ok(())
    }

    fn encode_state(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"RPS1");
        put_u64(&mut b, self.node.0);
        put_u64(&mut b, self.service_ref.entity.to_u64());
        put_u32(&mut b, self.service_version);
        put_u32(&mut b, self.templates.len() as u32);
        for t in self.templates.values() {
            put_u32(&mut b, t.id);
            b.push(t.mode as u8);
            b.push(match t.entry.state_access {
                crate::abi::StateAccess::ReadOnly => 0,
                crate::abi::StateAccess::Mutable => 1,
            });
            put_u32(&mut b, t.entry.run_class);
            put_u32(&mut b, t.entry.resume_point);
            put_u32(&mut b, t.entry.max_steps);
            put_u32(&mut b, t.restart_limit);
            put_bytes(&mut b, &t.entry.frame_bytes);
        }
        put_u32(&mut b, self.records.len() as u32);
        for r in self.records.values() {
            encode_receipt(&mut b, r.receipt);
            encode_status(&mut b, r.status);
            put_u32(&mut b, r.owner_epoch);
        }
        put_u32(&mut b, self.ledger_order.len() as u32);
        for id in &self.ledger_order {
            b.extend_from_slice(id);
            encode_response(&mut b, self.ledger[id]);
        }
        b
    }
    fn decode_state(&mut self, bytes: &[u8]) -> Result<(), RemoteProcessError> {
        let mut c = Cursor { b: bytes, p: 0 };
        if c.take(4)? != b"RPS1" {
            return Err(RemoteProcessError::ProtocolError);
        }
        if c.u64()? != self.node.0
            || c.u64()? != self.service_ref.entity.to_u64()
            || c.u32()? != self.service_version
        {
            return Err(RemoteProcessError::ProtocolError);
        }
        let nt = c.u32()? as usize;
        if nt > MAX_REMOTE_PROCESS_TEMPLATES {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        for _ in 0..nt {
            let id = c.u32()?;
            let mode = match c.u8()? {
                1 => ProcessMode::Serial,
                2 => ProcessMode::Pure,
                3 => ProcessMode::System,
                _ => return Err(RemoteProcessError::ProtocolError),
            };
            let access = match c.u8()? {
                0 => crate::abi::StateAccess::ReadOnly,
                1 => crate::abi::StateAccess::Mutable,
                _ => return Err(RemoteProcessError::ProtocolError),
            };
            let run = c.u32()?;
            let resume = c.u32()?;
            let steps = c.u32()?;
            let limit = c.u32()?;
            let frame = c.bytes()?;
            if frame.len() > MAX_REMOTE_PROCESS_FRAME {
                return Err(RemoteProcessError::CapacityExceeded);
            }
            self.templates.insert(
                id,
                RemoteProcessTemplate {
                    id,
                    mode,
                    entry: ContinuationSpec::new(access, run, resume, frame, steps),
                    restart_limit: limit,
                },
            );
        }
        let nr = c.u32()? as usize;
        if nr > MAX_REMOTE_PROCESSES {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        for _ in 0..nr {
            let receipt = decode_receipt(&mut c, self.node)?;
            let status = decode_status(&mut c)?;
            let owner_epoch = c.u32()?;
            self.records.insert(
                receipt.process.entity.to_u64(),
                Record {
                    receipt,
                    status,
                    owner_epoch,
                },
            );
        }
        let nl = c.u32()? as usize;
        if nl > MAX_REMOTE_PROCESS_LEDGER {
            return Err(RemoteProcessError::CapacityExceeded);
        }
        for _ in 0..nl {
            let id = c.array32()?;
            let response = decode_response(&mut c, self.node)?;
            self.ledger.insert(id, response);
            self.ledger_order.push_back(id);
        }
        if c.p != bytes.len() {
            return Err(RemoteProcessError::ProtocolError);
        }
        Ok(())
    }
}

/// In-memory transport used by two-kernel tests and embedders.  It retains a
/// weak endpoint, grants and receipts only--never a descriptor or process table.
pub struct RemoteProcessClient {
    endpoint: Weak<Mutex<RemoteProcessService>>,
}
impl RemoteProcessClient {
    pub fn new(endpoint: &Arc<Mutex<RemoteProcessService>>) -> Self {
        Self {
            endpoint: Arc::downgrade(endpoint),
        }
    }
    fn service(&self) -> Result<Arc<Mutex<RemoteProcessService>>, RemoteProcessError> {
        self.endpoint
            .upgrade()
            .ok_or(RemoteProcessError::NodeUnavailable)
    }
    pub fn create(
        &self,
        id: [u8; 32],
        template: u32,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        self.service()?
            .lock()
            .map_err(|_| RemoteProcessError::NodeLost)?
            .stage_create(id, template, epoch, grant)
    }
    pub fn restart(
        &self,
        id: [u8; 32],
        receipt: RemoteProcessReceipt,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        self.service()?
            .lock()
            .map_err(|_| RemoteProcessError::NodeLost)?
            .stage_restart(id, receipt, epoch, grant)
    }
    pub fn query(
        &self,
        r: RemoteProcessReceipt,
        e: u32,
        g: &RemoteGrant,
    ) -> Result<RemoteProcessObservation, RemoteProcessError> {
        self.service()?
            .lock()
            .map_err(|_| RemoteProcessError::NodeLost)?
            .query(r, e, g)
    }
}

fn map_state(s: ProcessState) -> RemoteProcessStatus {
    match s {
        ProcessState::Created => RemoteProcessStatus::Created,
        ProcessState::Runnable => RemoteProcessStatus::Runnable,
        ProcessState::Running => RemoteProcessStatus::Running,
        ProcessState::Waiting => RemoteProcessStatus::Waiting,
        ProcessState::Failed => RemoteProcessStatus::Terminal(ExitReason::Failed),
        ProcessState::Terminated => RemoteProcessStatus::Terminal(ExitReason::Completed),
        ProcessState::Cancelled | ProcessState::CancelPending => {
            RemoteProcessStatus::Terminal(ExitReason::Cancelled)
        }
    }
}
fn bounded_read(path: &Path, max: u64) -> Result<Option<Vec<u8>>, RemoteProcessError> {
    match fs::metadata(path) {
        Ok(meta) if meta.len() > max => return Err(RemoteProcessError::CapacityExceeded),
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(RemoteProcessError::Persistence),
    }
    fs::read(path)
        .map(Some)
        .map_err(|_| RemoteProcessError::Persistence)
}
fn load_latest(dir: &Path) -> Result<Option<Vec<u8>>, RemoteProcessError> {
    let wal = dir.join("process.wal");
    if let Some(w) = bounded_read(&wal, MAX_REMOTE_PROCESS_DURABLE_BYTES + 40)? {
        if w.len() >= 40 && &w[..4] == b"RPSW" {
            let n = u32::from_le_bytes(w[4..8].try_into().unwrap()) as usize;
            if w.len() == 40 + n && Sha256::digest(&w[40..]).as_slice() == &w[8..40] {
                return Ok(Some(w[40..].to_vec()));
            }
        }
    }
    bounded_read(
        &dir.join("process.snapshot"),
        MAX_REMOTE_PROCESS_DURABLE_BYTES,
    )
}
fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes())
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes())
}
fn put_bytes(b: &mut Vec<u8>, x: &[u8]) {
    put_u32(b, x.len() as u32);
    b.extend_from_slice(x)
}
struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], RemoteProcessError> {
        let x = self
            .b
            .get(self.p..self.p + n)
            .ok_or(RemoteProcessError::ProtocolError)?;
        self.p += n;
        Ok(x)
    }
    fn u8(&mut self) -> Result<u8, RemoteProcessError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, RemoteProcessError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, RemoteProcessError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn array32(&mut self) -> Result<[u8; 32], RemoteProcessError> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    fn bytes(&mut self) -> Result<Vec<u8>, RemoteProcessError> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }
}
fn encode_ref(b: &mut Vec<u8>, r: RemoteRef) {
    put_u64(b, r.node.0);
    put_u64(b, r.entity.to_u64())
}
fn decode_ref(c: &mut Cursor<'_>) -> Result<RemoteRef, RemoteProcessError> {
    Ok(RemoteRef {
        node: NodeId(c.u64()?),
        entity: Ref64::from_u64(c.u64()?),
    })
}
fn encode_receipt(b: &mut Vec<u8>, r: RemoteProcessReceipt) {
    encode_ref(b, r.process);
    put_u32(b, r.version);
    put_u32(b, r.template_id);
    encode_ref(b, r.restart_of);
    put_u32(b, r.restart_attempt);
    put_u32(b, r.restart_limit)
}
fn decode_receipt(
    c: &mut Cursor<'_>,
    node: NodeId,
) -> Result<RemoteProcessReceipt, RemoteProcessError> {
    let r = RemoteProcessReceipt {
        process: decode_ref(c)?,
        version: c.u32()?,
        template_id: c.u32()?,
        restart_of: decode_ref(c)?,
        restart_attempt: c.u32()?,
        restart_limit: c.u32()?,
    };
    if r.process.node != node {
        return Err(RemoteProcessError::WrongNode);
    }
    Ok(r)
}
fn encode_status(b: &mut Vec<u8>, s: RemoteProcessStatus) {
    b.push(match s {
        RemoteProcessStatus::Created => 1,
        RemoteProcessStatus::Runnable => 2,
        RemoteProcessStatus::Running => 3,
        RemoteProcessStatus::Waiting => 4,
        RemoteProcessStatus::Terminal(ExitReason::Completed) => 7,
        RemoteProcessStatus::Terminal(ExitReason::Failed) => 6,
        RemoteProcessStatus::Terminal(ExitReason::Cancelled) => 8,
        RemoteProcessStatus::Terminal(ExitReason::NodeLost) => 9,
    })
}
fn decode_status(c: &mut Cursor<'_>) -> Result<RemoteProcessStatus, RemoteProcessError> {
    Ok(match c.u8()? {
        1 => RemoteProcessStatus::Created,
        2 => RemoteProcessStatus::Runnable,
        3 => RemoteProcessStatus::Running,
        4 => RemoteProcessStatus::Waiting,
        6 => RemoteProcessStatus::Terminal(ExitReason::Failed),
        7 => RemoteProcessStatus::Terminal(ExitReason::Completed),
        8 => RemoteProcessStatus::Terminal(ExitReason::Cancelled),
        9 => RemoteProcessStatus::Terminal(ExitReason::NodeLost),
        _ => return Err(RemoteProcessError::ProtocolError),
    })
}
fn encode_response(b: &mut Vec<u8>, r: RemoteProcessResponse) {
    match r {
        RemoteProcessResponse::Created(x) => {
            b.push(1);
            encode_receipt(b, x)
        }
        RemoteProcessResponse::Restarted(x) => {
            b.push(2);
            encode_receipt(b, x)
        }
        RemoteProcessResponse::RestartBudgetExhausted => b.push(3),
    }
}
fn decode_response(
    c: &mut Cursor<'_>,
    node: NodeId,
) -> Result<RemoteProcessResponse, RemoteProcessError> {
    Ok(match c.u8()? {
        1 => RemoteProcessResponse::Created(decode_receipt(c, node)?),
        2 => RemoteProcessResponse::Restarted(decode_receipt(c, node)?),
        3 => RemoteProcessResponse::RestartBudgetExhausted,
        _ => return Err(RemoteProcessError::ProtocolError),
    })
}

const PROCESS_WIRE_MAGIC: u32 = 0x5250_5331;
const PROCESS_WIRE_VERSION: u16 = 1;
const PROCESS_CREATE: u16 = 1;
const PROCESS_RESTART: u16 = 2;
const PROCESS_QUERY: u16 = 3;
const PROCESS_REQUEST_LEN: usize = 4 + 2 + 2 + 32 + 4 + RemoteGrant::ENCODED_LEN + 4 + 48;

#[derive(Clone, Copy)]
struct ProcessWireRequest {
    opcode: u16,
    id: [u8; 32],
    epoch: u32,
    grant: RemoteGrant,
    template: u32,
    receipt: RemoteProcessReceipt,
}
impl ProcessWireRequest {
    fn encode(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(PROCESS_REQUEST_LEN);
        put_u32(&mut b, PROCESS_WIRE_MAGIC);
        b.extend_from_slice(&PROCESS_WIRE_VERSION.to_le_bytes());
        b.extend_from_slice(&self.opcode.to_le_bytes());
        b.extend_from_slice(&self.id);
        put_u32(&mut b, self.epoch);
        b.extend_from_slice(&self.grant.encode());
        put_u32(&mut b, self.template);
        encode_receipt(&mut b, self.receipt);
        b
    }
    fn decode(b: &[u8]) -> Result<Self, RemoteProcessError> {
        if b.len() != PROCESS_REQUEST_LEN {
            return Err(RemoteProcessError::ProtocolError);
        }
        let mut c = Cursor { b, p: 0 };
        if c.u32()? != PROCESS_WIRE_MAGIC {
            return Err(RemoteProcessError::ProtocolError);
        }
        let version = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        if version != PROCESS_WIRE_VERSION {
            return Err(RemoteProcessError::ProtocolError);
        }
        let opcode = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        let id = c.array32()?;
        let epoch = c.u32()?;
        let grant = RemoteGrant::decode(c.take(RemoteGrant::ENCODED_LEN)?)
            .ok_or(RemoteProcessError::ProtocolError)?;
        let template = c.u32()?;
        let receipt = decode_receipt(&mut c, grant.target.node)?;
        Ok(Self {
            opcode,
            id,
            epoch,
            grant,
            template,
            receipt,
        })
    }
}
fn null_receipt(node: NodeId) -> RemoteProcessReceipt {
    RemoteProcessReceipt {
        process: RemoteRef {
            node,
            entity: Ref64::NULL,
        },
        version: 0,
        template_id: 0,
        restart_of: RemoteRef {
            node,
            entity: Ref64::NULL,
        },
        restart_attempt: 0,
        restart_limit: 0,
    }
}

#[derive(Clone, Copy)]
struct ProcessWireResponse {
    status: u16,
    response: Option<RemoteProcessResponse>,
    observation: Option<RemoteProcessObservation>,
}
impl ProcessWireResponse {
    fn ok(response: Option<RemoteProcessResponse>) -> Self {
        Self {
            status: 0,
            response,
            observation: None,
        }
    }
    fn observed(x: RemoteProcessObservation) -> Self {
        Self {
            status: 0,
            response: None,
            observation: Some(x),
        }
    }
    fn error(e: RemoteProcessError) -> Self {
        Self {
            status: wire_error(e),
            response: None,
            observation: None,
        }
    }
    fn encode(self) -> Vec<u8> {
        let mut b = Vec::new();
        put_u32(&mut b, PROCESS_WIRE_MAGIC);
        b.extend_from_slice(&PROCESS_WIRE_VERSION.to_le_bytes());
        b.extend_from_slice(&self.status.to_le_bytes());
        match (self.response, self.observation) {
            (None, None) => b.push(0),
            (Some(RemoteProcessResponse::Created(r)), _) => {
                b.push(1);
                encode_receipt(&mut b, r)
            }
            (Some(RemoteProcessResponse::Restarted(r)), _) => {
                b.push(2);
                encode_receipt(&mut b, r)
            }
            (Some(RemoteProcessResponse::RestartBudgetExhausted), _) => b.push(3),
            (_, Some(o)) => {
                b.push(4);
                encode_receipt(&mut b, o.receipt);
                encode_status(&mut b, o.status);
                put_u32(&mut b, o.owner_epoch)
            }
        }
        b
    }
    fn decode(b: &[u8], node: NodeId) -> Result<Self, RemoteProcessError> {
        let mut c = Cursor { b, p: 0 };
        if c.u32()? != PROCESS_WIRE_MAGIC {
            return Err(RemoteProcessError::ProtocolError);
        }
        let version = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        if version != PROCESS_WIRE_VERSION {
            return Err(RemoteProcessError::ProtocolError);
        }
        let status = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        let tag = c.u8()?;
        let (response, observation) = match tag {
            0 => (None, None),
            1 => (
                Some(RemoteProcessResponse::Created(decode_receipt(
                    &mut c, node,
                )?)),
                None,
            ),
            2 => (
                Some(RemoteProcessResponse::Restarted(decode_receipt(
                    &mut c, node,
                )?)),
                None,
            ),
            3 => (Some(RemoteProcessResponse::RestartBudgetExhausted), None),
            4 => {
                let receipt = decode_receipt(&mut c, node)?;
                let status = decode_status(&mut c)?;
                let owner_epoch = c.u32()?;
                (
                    None,
                    Some(RemoteProcessObservation {
                        receipt,
                        status,
                        owner_epoch,
                    }),
                )
            }
            _ => return Err(RemoteProcessError::ProtocolError),
        };
        if c.p != b.len() {
            return Err(RemoteProcessError::ProtocolError);
        }
        Ok(Self {
            status,
            response,
            observation,
        })
    }
}
fn wire_error(e: RemoteProcessError) -> u16 {
    match e {
        RemoteProcessError::AuthorityDenied => 1,
        RemoteProcessError::WrongNode => 2,
        RemoteProcessError::UnknownTemplate => 3,
        RemoteProcessError::UnknownProcess => 4,
        RemoteProcessError::StaleReceipt => 5,
        RemoteProcessError::CapacityExceeded => 6,
        RemoteProcessError::RestartBudgetExhausted => 7,
        RemoteProcessError::ProtocolError => 8,
        RemoteProcessError::Persistence => 9,
        RemoteProcessError::LiveRecoveryUnsupported => 10,
        RemoteProcessError::Kernel(_) => 11,
        RemoteProcessError::DuplicateTemplate => 12,
        RemoteProcessError::NodeUnavailable => 13,
        RemoteProcessError::NodeLost => 14,
    }
}
fn decode_wire_error(v: u16) -> RemoteProcessError {
    match v {
        1 => RemoteProcessError::AuthorityDenied,
        2 => RemoteProcessError::WrongNode,
        3 => RemoteProcessError::UnknownTemplate,
        4 => RemoteProcessError::UnknownProcess,
        5 => RemoteProcessError::StaleReceipt,
        6 => RemoteProcessError::CapacityExceeded,
        7 => RemoteProcessError::RestartBudgetExhausted,
        9 => RemoteProcessError::Persistence,
        10 => RemoteProcessError::LiveRecoveryUnsupported,
        12 => RemoteProcessError::DuplicateTemplate,
        13 => RemoteProcessError::NodeUnavailable,
        14 => RemoteProcessError::NodeLost,
        _ => RemoteProcessError::ProtocolError,
    }
}
fn write_process_frame(s: &mut TcpStream, b: &[u8]) -> std::io::Result<()> {
    if b.len() > 4096 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "oversized process frame",
        ));
    }
    s.write_all(&(b.len() as u32).to_le_bytes())?;
    s.write_all(b)
}
fn read_process_frame(s: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut n = [0; 4];
    s.read_exact(&mut n)?;
    let n = u32::from_le_bytes(n) as usize;
    if n > 4096 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "oversized process frame",
        ));
    }
    let mut b = vec![0; n];
    s.read_exact(&mut b)?;
    Ok(b)
}

pub struct RemoteProcessServer;
impl RemoteProcessServer {
    pub fn serve_until(
        listener: TcpListener,
        service: Arc<Mutex<RemoteProcessService>>,
        shutdown: Arc<AtomicBool>,
    ) -> std::io::Result<()> {
        Self::serve_until_with_timeout(listener, service, shutdown, Duration::from_millis(250))
    }
    pub fn serve_until_with_timeout(
        listener: TcpListener,
        service: Arc<Mutex<RemoteProcessService>>,
        shutdown: Arc<AtomicBool>,
        peer_timeout: Duration,
    ) -> std::io::Result<()> {
        listener.set_nonblocking(true)?;
        while !shutdown.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    // Accepted sockets may inherit O_NONBLOCK from the polling
                    // listener on some platforms; restore deadline-bounded
                    // blocking I/O before consuming the fixed frame.
                    stream.set_nonblocking(false)?;
                    // The protocol is bounded in bytes and now in wall time: a
                    // peer holding a partial prefix can delay this narrow
                    // single-threaded adapter only until this deadline.
                    stream.set_read_timeout(Some(peer_timeout))?;
                    stream.set_write_timeout(Some(peer_timeout))?;
                    let frame = match read_process_frame(&mut stream) {
                        Ok(x) => x,
                        Err(_) => continue,
                    };
                    let output = match ProcessWireRequest::decode(&frame) {
                        Err(e) => ProcessWireResponse::error(e),
                        Ok(r) => {
                            let mut s = service.lock().map_err(|_| {
                                std::io::Error::other("remote process service poisoned")
                            })?;
                            match r.opcode {
                                PROCESS_CREATE => {
                                    match s.stage_create(r.id, r.template, r.epoch, &r.grant) {
                                        Ok(x) => ProcessWireResponse::ok(x),
                                        Err(e) => ProcessWireResponse::error(e),
                                    }
                                }
                                PROCESS_RESTART => {
                                    match s.stage_restart(r.id, r.receipt, r.epoch, &r.grant) {
                                        Ok(x) => ProcessWireResponse::ok(x),
                                        Err(e) => ProcessWireResponse::error(e),
                                    }
                                }
                                PROCESS_QUERY => match s.query(r.receipt, r.epoch, &r.grant) {
                                    Ok(x) => ProcessWireResponse::observed(x),
                                    Err(e) => ProcessWireResponse::error(e),
                                },
                                _ => ProcessWireResponse::error(RemoteProcessError::ProtocolError),
                            }
                        }
                    };
                    let _ = write_process_frame(&mut stream, &output.encode());
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Signed-request, authority-checked operation protocol over bounded TCP.
/// This is not mutually authenticated transport: responses have request-shape
/// validation but no session MAC. Endpoint, grants, and receipts are retained;
/// owner memory is never shared.
pub struct RemoteProcessTcpClient {
    endpoint: SocketAddr,
    timeout: Duration,
    node: NodeId,
}
impl RemoteProcessTcpClient {
    pub fn new(endpoint: SocketAddr, node: NodeId) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_secs(5),
            node,
        }
    }
    pub fn set_timeout(&mut self, d: Duration) {
        self.timeout = d
    }
    pub fn create(
        &self,
        id: [u8; 32],
        template: u32,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        self.mutate(ProcessWireRequest {
            opcode: PROCESS_CREATE,
            id,
            epoch,
            grant: *grant,
            template,
            receipt: null_receipt(self.node),
        })
    }
    pub fn restart(
        &self,
        id: [u8; 32],
        receipt: RemoteProcessReceipt,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        self.mutate(ProcessWireRequest {
            opcode: PROCESS_RESTART,
            id,
            epoch,
            grant: *grant,
            template: 0,
            receipt,
        })
    }
    pub fn query(
        &self,
        receipt: RemoteProcessReceipt,
        epoch: u32,
        grant: &RemoteGrant,
    ) -> Result<RemoteProcessObservation, RemoteProcessError> {
        let x = self.round_trip(ProcessWireRequest {
            opcode: PROCESS_QUERY,
            id: [0; 32],
            epoch,
            grant: *grant,
            template: 0,
            receipt,
        })?;
        x.observation.ok_or(RemoteProcessError::ProtocolError)
    }
    fn mutate(
        &self,
        r: ProcessWireRequest,
    ) -> Result<Option<RemoteProcessResponse>, RemoteProcessError> {
        Ok(self.round_trip(r)?.response)
    }
    fn round_trip(&self, r: ProcessWireRequest) -> Result<ProcessWireResponse, RemoteProcessError> {
        let mut s = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(|_| RemoteProcessError::NodeUnavailable)?;
        s.set_read_timeout(Some(self.timeout))
            .map_err(|_| RemoteProcessError::NodeUnavailable)?;
        s.set_write_timeout(Some(self.timeout))
            .map_err(|_| RemoteProcessError::NodeUnavailable)?;
        write_process_frame(&mut s, &r.encode()).map_err(|_| RemoteProcessError::NodeLost)?;
        let b = read_process_frame(&mut s).map_err(|_| RemoteProcessError::NodeLost)?;
        let x = ProcessWireResponse::decode(&b, self.node)?;
        if x.status != 0 {
            return Err(decode_wire_error(x.status));
        }
        Ok(x)
    }
}
