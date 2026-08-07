//! Metal lowering of the standalone resident synchronization ABI.
//!
//! The complete epoch loop, handler interpreter, canonical effect applier,
//! park/wake/retry machinery, and quiescence test execute in one Metal command
//! buffer. The host performs one readback after completion. This is a
//! standalone backend and deliberately does not claim `Kernel` integration.

use std::collections::BTreeMap;
use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::batch::BackendError;
use super::resident_sync::*;
use crate::scheduler::device::{DeviceLaneAccess, DEVICE_ACCESS_WRITE};
use crate::scheduler::device_ops::{DeviceLaneOperation, DeviceOperationJournal};

const MAX_PROGRAM_INSTRUCTIONS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuConfig {
    continuation_count: u32,
    handler_count: u32,
    instruction_count: u32,
    future_count: u32,
    mailbox_count: u32,
    capability_count: u32,
    max_epochs: u32,
    max_effects: u32,
    frame_stride: u32,
    effect_capacity: u32,
    trace_capacity: u32,
    mailbox_stride: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuContinuation {
    id: u64,
    actor: u64,
    run_class: u32,
    frame_len: u32,
    state: u32,
    last_epoch: u32,
    previous_kind: u32,
    pending_opcode: u32,
    pending_target: u32,
    reserved: u32,
    previous_value: u64,
    previous_sender: u64,
    pending_value: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuHandler {
    run_class: u32,
    instruction_offset: u32,
    instruction_count: u32,
    reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuFuture {
    resolved: u32,
    reserved: u32,
    value: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuMailbox {
    capacity: u32,
    head: u32,
    count: u32,
    reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuMailEntry {
    sender: u64,
    value: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuCapability {
    actor: u64,
    kind: u32,
    target: u32,
    rights: u32,
    reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuEffectRecord {
    epoch: u32,
    lane: u32,
    ordinal: u32,
    opcode: u32,
    continuation: u64,
    target: u32,
    outcome: u32,
    value: u64,
    result_value: u64,
    result_sender: u64,
    run_class: u32,
    reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuStage {
    continuation_index: u32,
    effect_offset: u32,
    effect_count: u32,
    disposition: u32,
    next_run_class: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuStatus {
    effect_count: u32,
    trace_count: u32,
    completed_count: u32,
    epochs: u32,
    quiescent: u32,
    error: u32,
    reserved0: u32,
    reserved1: u32,
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Config { uint continuation_count, handler_count, instruction_count, future_count; uint mailbox_count, capability_count, max_epochs, max_effects; uint frame_stride, effect_capacity, trace_capacity, mailbox_stride; };
struct Cont { ulong id, actor; uint run_class, frame_len, state, last_epoch; uint previous_kind, pending_opcode, pending_target, reserved; ulong previous_value, previous_sender, pending_value; };
struct Handler { uint run_class, instruction_offset, instruction_count, reserved; };
struct Instruction { uint opcode, argument; ulong value; };
struct Future { uint resolved, reserved; ulong value; };
struct Mailbox { uint capacity, head, count, reserved; };
struct MailEntry { ulong sender, value; };
struct Capability { ulong actor; uint kind, target, rights, reserved; };
struct EffectRecord { uint epoch, lane, ordinal, opcode; ulong continuation; uint target, outcome; ulong value, result_value, result_sender; uint run_class, reserved; };
struct Trace { uint epoch, lane; ulong continuation; uint run_class, event, word; };
struct Status { uint effect_count, trace_count, completed_count, epochs; uint quiescent, error, reserved0, reserved1; };
struct Stage { uint continuation_index, effect_offset, effect_count, disposition; uint next_run_class, reserved0, reserved1, reserved2; };
constant uint RUNNABLE=1, PARKED=2, COMPLETE=3;
constant uint AWAIT=1, RESOLVE=2, SEND=3, RECEIVE=4;
constant uint OUT_RESOLVED=1, OUT_REGISTERED=2, OUT_SENT=3, OUT_RECEIVED=4, OUT_DENIED=5, OUT_INVALID=6, OUT_FULL=7, OUT_EMPTY=8, OUT_DOUBLE=9;
inline uint outcome_code(uint o) { if(o==OUT_RESOLVED||o==OUT_RECEIVED)return 2; if(o==OUT_REGISTERED)return 3; if(o==OUT_EMPTY)return 1; if(o==OUT_INVALID)return 0x101; if(o==OUT_DENIED)return 0x104; if(o==OUT_FULL)return 0x10c; if(o==OUT_DOUBLE)return 0x111; return 0; }
inline uint journal_opcode(uint op) { return op==AWAIT?11:(op==RESOLVE?10:(op==SEND?8:9)); }
inline uint hash_frame(device uchar* f,uint n){uint h=2166136261u;for(uint i=0;i<n;i++)h=(h^uint(f[i]))*16777619u;return h;}
inline bool allowed(device const Capability* caps,uint n,ulong actor,uint kind,uint target,uint right){for(uint i=0;i<n;i++){Capability c=caps[i];if(c.actor==actor&&c.kind==kind&&c.target==target&&(c.rights&right)!=0)return true;}return false;}
inline void add_trace(device Trace* traces,device Status* s,constant Config& cfg,uint epoch,uint lane,Cont c,uint event,uint word){if(s->trace_count>=cfg.trace_capacity){s->error=2;return;}Trace t={epoch,lane,c.id,c.run_class,event,word};traces[s->trace_count++]=t;}
inline void wake_future(device Cont* cs,uint n,uint target,ulong value,uint epoch){for(uint i=0;i<n;i++)if(cs[i].state==PARKED&&cs[i].pending_opcode==AWAIT&&cs[i].pending_target==target){cs[i].state=RUNNABLE;cs[i].last_epoch=epoch;cs[i].pending_opcode=0;cs[i].previous_kind=OUT_RESOLVED;cs[i].previous_value=value;}}
inline void wake_one(device Cont* cs,uint n,uint opcode,uint target,uint epoch){uint best=0xffffffffu;for(uint i=0;i<n;i++)if(cs[i].state==PARKED&&cs[i].pending_opcode==opcode&&cs[i].pending_target==target&&(best==0xffffffffu||cs[i].reserved<cs[best].reserved||(cs[i].reserved==cs[best].reserved&&cs[i].id<cs[best].id)))best=i;if(best!=0xffffffffu){cs[best].state=RUNNABLE;cs[best].last_epoch=epoch;}}
kernel void resident_sync(
 constant Config& cfg [[buffer(0)]], device Cont* cs [[buffer(1)]], device uchar* frames [[buffer(2)]],
 device const Handler* handlers [[buffer(3)]], device const Instruction* ins [[buffer(4)]], device Future* futures [[buffer(5)]],
 device Mailbox* mails [[buffer(6)]], device MailEntry* entries [[buffer(7)]], device const Capability* caps [[buffer(8)]],
 device EffectRecord* records [[buffer(9)]], device Trace* traces [[buffer(10)]], device ulong* completed [[buffer(11)]], device Status* status [[buffer(12)]], device Stage* stages [[buffer(13)]], uint gid [[thread_position_in_grid]]) {
 if(gid!=0)return; Status s={0,0,0,0,0,0,0,0}; *status=s; if(cfg.continuation_count==0){status->quiescent=1;return;}
 for(uint epoch=0;epoch<cfg.max_epochs;epoch++){
  uint lane=0;
  // Handler phase: snapshot every runnable lane and emit bounded effects only.
  // No future/mailbox/capability table is mutated in this phase.
  while(true){
   uint at=0xffffffffu;
   for(uint i=0;i<cfg.continuation_count;i++)if(cs[i].state==RUNNABLE&&cs[i].last_epoch!=epoch&&(at==0xffffffffu||cs[i].run_class<cs[at].run_class||(cs[i].run_class==cs[at].run_class&&cs[i].id<cs[at].id)))at=i;
   if(at==0xffffffffu)break;
   Cont c=cs[at]; c.last_epoch=epoch; c.state=0; uint disposition=0,next_class=c.run_class,emitted=0; uint effect_start=status->effect_count;
   uint retry_opcode=c.pending_opcode,retry_target=c.pending_target;ulong retry_value=c.pending_value;
   uint input_previous_kind=c.previous_kind;ulong input_previous_value=c.previous_value;
   uint pc=0,end=0;
   if(retry_opcode==0){c.previous_kind=0;c.previous_value=0;c.previous_sender=0;uint hi=0xffffffffu;for(uint h=0;h<cfg.handler_count;h++)if(handlers[h].run_class==c.run_class)hi=h;if(hi==0xffffffffu){status->error=3;return;}pc=handlers[hi].instruction_offset;end=pc+handlers[hi].instruction_count;}
   else{c.pending_opcode=0;c.pending_target=0;c.pending_value=0;}
   while(retry_opcode!=0||pc<end){
    uint op,arg;ulong val;
    if(retry_opcode!=0){op=retry_opcode;arg=retry_target;val=retry_value;retry_opcode=0;disposition=1;next_class=c.run_class;}
    else{Instruction x=ins[pc++];op=x.opcode;arg=x.argument;val=x.value;
     if(op==5||op==6){if(arg+8>c.frame_len||(op==6&&input_previous_kind!=OUT_RESOLVED&&input_previous_kind!=OUT_RECEIVED)){status->error=4;return;}ulong v=op==5?val:input_previous_value;for(uint bb=0;bb<8;bb++)frames[at*cfg.frame_stride+arg+bb]=uchar(v>>(bb*8));continue;}
     if(op==7){if((input_previous_kind!=OUT_RESOLVED&&input_previous_kind!=OUT_RECEIVED)||input_previous_value!=val)pc+=arg;continue;}
     if(op==8){disposition=1;next_class=arg;break;}if(op==9){disposition=2;break;}
    }
    if(op<1||op>4||emitted>=cfg.max_effects||status->effect_count>=cfg.effect_capacity){status->error=5;return;}
    EffectRecord r={epoch,lane,emitted,journal_opcode(op),c.id,arg,0,val,0,0,c.run_class,0};records[status->effect_count++]=r;emitted++;
   }
   if(disposition==0){status->error=7;return;}
   Stage st={at,effect_start,emitted,disposition,next_class,0,0,0};stages[lane]=st;cs[at]=c;
   add_trace(traces,status,cfg,epoch,lane,c,0,hash_frame(frames+at*cfg.frame_stride,c.frame_len));lane++;
  }
  // Canonical applier phase. All invocation traces already occupy the prefix.
  for(uint li=0;li<lane;li++){
   Stage st=stages[li];Cont c=cs[st.continuation_index];bool parked=false;
   for(uint ri=st.effect_offset;ri<st.effect_offset+st.effect_count;ri++){
    EffectRecord r=records[ri];uint op=r.opcode==11?AWAIT:(r.opcode==10?RESOLVE:(r.opcode==8?SEND:RECEIVE));uint arg=r.target;ulong val=r.value;
    uint kind=op<=2?2:3;uint right=(op==AWAIT||op==RECEIVE)?1:2;uint outcome=0;ulong result_value=0,result_sender=0;
    if(!allowed(caps,cfg.capability_count,c.actor,kind,arg,right))outcome=OUT_DENIED;
    else if((kind==2&&arg>=cfg.future_count)||(kind==3&&arg>=cfg.mailbox_count))outcome=OUT_INVALID;
    else if(op==AWAIT){if(futures[arg].resolved!=0){outcome=OUT_RESOLVED;result_value=futures[arg].value;}else{outcome=OUT_REGISTERED;parked=true;c.pending_opcode=AWAIT;c.pending_target=arg;}}
    else if(op==RESOLVE){if(futures[arg].resolved!=0)outcome=OUT_DOUBLE;else{futures[arg].resolved=1;futures[arg].value=val;outcome=OUT_RESOLVED;result_value=val;wake_future(cs,cfg.continuation_count,arg,val,epoch);}}
    else if(op==SEND){Mailbox m=mails[arg];if(m.count>=m.capacity){outcome=OUT_FULL;parked=true;c.pending_opcode=SEND;c.pending_target=arg;c.pending_value=val;}else{uint slot=arg*cfg.mailbox_stride+(m.head+m.count)%cfg.mailbox_stride;MailEntry ne={c.actor,val};entries[slot]=ne;m.count++;mails[arg]=m;outcome=OUT_SENT;wake_one(cs,cfg.continuation_count,RECEIVE,arg,epoch);}}
    else{Mailbox m=mails[arg];if(m.count==0){outcome=OUT_EMPTY;parked=true;c.pending_opcode=RECEIVE;c.pending_target=arg;}else{uint slot=arg*cfg.mailbox_stride+m.head;MailEntry e=entries[slot];m.head=(m.head+1)%cfg.mailbox_stride;m.count--;mails[arg]=m;outcome=OUT_RECEIVED;result_value=e.value;result_sender=e.sender;c.pending_opcode=0;wake_one(cs,cfg.continuation_count,SEND,arg,epoch);}}
    if(outcome==OUT_REGISTERED||outcome==OUT_FULL||outcome==OUT_EMPTY)c.reserved=++status->reserved0;
    r.outcome=outcome;r.result_value=result_value;r.result_sender=result_sender;records[ri]=r;add_trace(traces,status,cfg,epoch,li,c,r.opcode,outcome_code(outcome));
    if(outcome==OUT_RESOLVED||outcome==OUT_RECEIVED){c.previous_kind=outcome;c.previous_value=result_value;c.previous_sender=result_sender;}
   }
   if(parked)c.state=PARKED;else if(st.disposition==1){c.state=RUNNABLE;c.run_class=st.next_run_class;}else{c.state=COMPLETE;completed[status->completed_count++]=c.id;}cs[st.continuation_index]=c;
  }
  status->epochs=epoch+1;bool any=false;for(uint i=0;i<cfg.continuation_count;i++)if(cs[i].state==RUNNABLE)any=true;if(!any){status->quiescent=1;break;}
 }
 bool any=false;for(uint i=0;i<cfg.continuation_count;i++)if(cs[i].state==RUNNABLE)any=true;status->quiescent=any?0:1;
}
"#;

pub struct MetalResidentSync {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalResidentSync {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Unavailable)?;
        let library = device
            .new_library_with_source(SOURCE, &CompileOptions::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let function = library
            .get_function("resident_sync", None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self {
            queue: device.new_command_queue(),
            device,
            pipeline,
        })
    }

    /// Executes the complete resident run in one command buffer and performs a
    /// single final readback after that command buffer completes.
    pub fn run(
        &self,
        config: &ResidentSyncConfig,
        mut continuations: Vec<ResidentContinuation>,
        programs: &BTreeMap<u32, ResidentHandlerProgram>,
    ) -> Result<ResidentSyncResult, BackendError> {
        validate(config, &continuations, programs)?;
        if u64::from(config.cohort_width) > self.pipeline.max_total_threads_per_threadgroup() {
            return Err(BackendError::InvalidInput);
        }
        continuations.sort_by_key(|c| c.id);
        let effect_capacity =
            checked_capacity(config, continuations.len(), config.max_effects_per_step)?;
        let trace_capacity = effect_capacity
            .checked_add(
                (config.max_epochs as usize)
                    .checked_mul(continuations.len())
                    .ok_or(BackendError::InvalidInput)?,
            )
            .ok_or(BackendError::InvalidInput)?;
        let mut gpu_conts: Vec<GpuContinuation> = continuations
            .iter()
            .map(|c| GpuContinuation {
                id: c.id,
                actor: c.actor,
                run_class: c.run_class,
                frame_len: c.frame.len() as u32,
                state: 1,
                last_epoch: u32::MAX,
                ..Default::default()
            })
            .collect();
        let stride = config.max_frame_bytes as usize;
        let mut frames = vec![
            0u8;
            stride
                .checked_mul(continuations.len())
                .ok_or(BackendError::InvalidInput)?
        ];
        for (index, c) in continuations.iter().enumerate() {
            frames[index * stride..index * stride + c.frame.len()].copy_from_slice(&c.frame);
        }
        let mut instructions = Vec::new();
        let mut handlers = Vec::new();
        for (&run_class, program) in programs {
            handlers.push(GpuHandler {
                run_class,
                instruction_offset: instructions.len() as u32,
                instruction_count: program.instructions.len() as u32,
                reserved: 0,
            });
            instructions.extend_from_slice(&program.instructions);
        }
        let futures: Vec<GpuFuture> = config
            .futures
            .iter()
            .map(|f| match f {
                InitialFuture::Pending => GpuFuture::default(),
                InitialFuture::Resolved(value) => GpuFuture {
                    resolved: 1,
                    value: *value,
                    reserved: 0,
                },
            })
            .collect();
        let mailboxes: Vec<GpuMailbox> = config
            .mailbox_capacities
            .iter()
            .map(|capacity| GpuMailbox {
                capacity: *capacity,
                ..Default::default()
            })
            .collect();
        let capabilities: Vec<GpuCapability> = config
            .capabilities
            .iter()
            .map(|c| GpuCapability {
                actor: c.actor,
                kind: c.resource_kind,
                target: c.target,
                rights: c.rights,
                reserved: 0,
            })
            .collect();
        let cfg = GpuConfig {
            continuation_count: gpu_conts.len() as u32,
            handler_count: handlers.len() as u32,
            instruction_count: instructions.len() as u32,
            future_count: futures.len() as u32,
            mailbox_count: mailboxes.len() as u32,
            capability_count: capabilities.len() as u32,
            max_epochs: config.max_epochs,
            max_effects: config.max_effects_per_step,
            frame_stride: config.max_frame_bytes,
            effect_capacity: effect_capacity as u32,
            trace_capacity: trace_capacity as u32,
            mailbox_stride: config.max_continuations,
        };
        let cfg_b = self.buffer_from(std::slice::from_ref(&cfg));
        let cs_b = self.buffer_from(&gpu_conts);
        let frames_b = self.buffer_from(&frames);
        let handlers_b = self.buffer_from(&handlers);
        let ins_b = self.buffer_from(&instructions);
        let futures_b = self.buffer_from(&futures);
        let mail_b = self.buffer_from(&mailboxes);
        let entry_count = config
            .mailbox_capacities
            .len()
            .checked_mul(config.max_continuations as usize)
            .ok_or(BackendError::InvalidInput)?;
        let entries_b = self.zero_buffer::<GpuMailEntry>(entry_count);
        let caps_b = self.buffer_from(&capabilities);
        let effects_b = self.zero_buffer::<GpuEffectRecord>(effect_capacity);
        let traces_b = self.zero_buffer::<ResidentSyncTrace>(trace_capacity);
        let completed_b = self.zero_buffer::<u64>(continuations.len());
        let status_b = self.zero_buffer::<GpuStatus>(1);
        let stages_b = self.zero_buffer::<GpuStage>(continuations.len());
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, buffer) in [
            &cfg_b,
            &cs_b,
            &frames_b,
            &handlers_b,
            &ins_b,
            &futures_b,
            &mail_b,
            &entries_b,
            &caps_b,
            &effects_b,
            &traces_b,
            &completed_b,
            &status_b,
            &stages_b,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        // All physical cohort lanes are dispatched. Lane zero is the elected
        // serial canonical applier; the other lanes perform no writes.
        let width = u64::from(config.cohort_width);
        encoder.dispatch_threads(MTLSize::new(width, 1, 1), MTLSize::new(width, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(BackendError::ExecutionFailed);
        }
        // The only device-to-host read phase begins here.
        gpu_conts = read_vec(&cs_b, continuations.len());
        let final_frames: Vec<u8> = read_vec(&frames_b, frames.len());
        let final_futures: Vec<GpuFuture> = read_vec(&futures_b, futures.len());
        let final_mails: Vec<GpuMailbox> = read_vec(&mail_b, mailboxes.len());
        let final_entries: Vec<GpuMailEntry> = read_vec(&entries_b, entry_count);
        let status: GpuStatus = read_vec(&status_b, 1)[0];
        if status.error != 0
            || status.effect_count as usize > effect_capacity
            || status.trace_count as usize > trace_capacity
            || status.completed_count as usize > continuations.len()
        {
            return Err(BackendError::ExecutionFailed);
        }
        let raw_effects: Vec<GpuEffectRecord> = read_vec(&effects_b, status.effect_count as usize);
        let trace = read_vec(&traces_b, status.trace_count as usize);
        let completed = read_vec(&completed_b, status.completed_count as usize);
        decode(
            config,
            &continuations,
            &gpu_conts,
            &final_frames,
            &final_futures,
            &final_mails,
            &final_entries,
            raw_effects,
            trace,
            completed,
            status,
        )
    }
    fn buffer_from<T: Copy>(&self, values: &[T]) -> Buffer {
        let b = self.device.new_buffer(
            (values.len().max(1) * size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        if !values.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr().cast::<u8>(),
                    b.contents().cast::<u8>(),
                    size_of_val(values),
                );
            }
        }
        b
    }
    fn zero_buffer<T>(&self, count: usize) -> Buffer {
        self.device.new_buffer(
            (count.max(1) * size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

fn checked_capacity(
    config: &ResidentSyncConfig,
    count: usize,
    effects: u32,
) -> Result<usize, BackendError> {
    (config.max_epochs as usize)
        .checked_mul(count)
        .and_then(|v| v.checked_mul(effects as usize))
        .filter(|v| *v <= u32::MAX as usize)
        .ok_or(BackendError::InvalidInput)
}
fn validate(
    config: &ResidentSyncConfig,
    continuations: &[ResidentContinuation],
    programs: &BTreeMap<u32, ResidentHandlerProgram>,
) -> Result<(), BackendError> {
    let unique_ids = continuations
        .iter()
        .map(|continuation| continuation.id)
        .collect::<std::collections::BTreeSet<_>>();
    if config.max_epochs == 0
        || config.max_effects_per_step == 0
        || config.max_frame_bytes == 0
        || config.max_continuations == 0
        || config.cohort_width == 0
        || continuations.len() > config.max_continuations as usize
        || unique_ids.len() != continuations.len()
        || u32::try_from(continuations.len()).is_err()
        || u32::try_from(programs.len()).is_err()
        || u32::try_from(config.futures.len()).is_err()
        || u32::try_from(config.mailbox_capacities.len()).is_err()
        || u32::try_from(config.capabilities.len()).is_err()
        || config
            .mailbox_capacities
            .iter()
            .any(|capacity| *capacity > config.max_continuations)
    {
        return Err(BackendError::InvalidInput);
    }
    if continuations.iter().any(|continuation| {
        continuation.frame.len() > config.max_frame_bytes as usize
            || !programs.contains_key(&continuation.run_class)
    }) {
        return Err(BackendError::InvalidInput);
    }
    let mut instruction_count = 0usize;
    for (&run_class, program) in programs {
        instruction_count = instruction_count
            .checked_add(program.instructions.len())
            .ok_or(BackendError::InvalidInput)?;
        if program.run_class != run_class
            || program.instructions.len() > MAX_PROGRAM_INSTRUCTIONS
            || !validate_handler_program(
                program,
                config.max_effects_per_step,
                config.max_frame_bytes,
            )
        {
            return Err(BackendError::InvalidInput);
        }
    }
    u32::try_from(instruction_count).map_err(|_| BackendError::InvalidInput)?;
    let effect_capacity =
        checked_capacity(config, continuations.len(), config.max_effects_per_step)?;
    let invocation_capacity = (config.max_epochs as usize)
        .checked_mul(continuations.len())
        .ok_or(BackendError::InvalidInput)?;
    let trace_capacity = effect_capacity
        .checked_add(invocation_capacity)
        .ok_or(BackendError::InvalidInput)?;
    u32::try_from(trace_capacity).map_err(|_| BackendError::InvalidInput)?;
    config
        .mailbox_capacities
        .len()
        .checked_mul(config.max_continuations as usize)
        .ok_or(BackendError::InvalidInput)?;
    Ok(())
}

fn read_vec<T: Copy>(buffer: &Buffer, count: usize) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    // Call sites validate `count` against the exact allocation capacity before
    // reading device-written variable-length outputs.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), count).to_vec() }
}
fn raw_outcome(r: &GpuEffectRecord) -> Result<ResidentOutcome, BackendError> {
    Ok(match r.outcome {
        1 => ResidentOutcome::Resolved(r.result_value),
        2 => ResidentOutcome::Registered,
        3 => ResidentOutcome::Sent,
        4 => ResidentOutcome::Received {
            value: r.result_value,
            sender: r.result_sender,
        },
        5 => ResidentOutcome::CapabilityDenied,
        6 => ResidentOutcome::InvalidTarget,
        7 => ResidentOutcome::Full,
        8 => ResidentOutcome::Empty,
        9 => ResidentOutcome::DoubleResolve,
        _ => return Err(BackendError::ExecutionFailed),
    })
}
fn raw_effect(r: &GpuEffectRecord) -> Result<ResidentEffect, BackendError> {
    Ok(match r.opcode {
        crate::scheduler::device_ops::OP_AWAIT_FUTURE => {
            ResidentEffect::FutureAwait { target: r.target }
        }
        crate::scheduler::device_ops::OP_RESOLVE_FUTURE => ResidentEffect::FutureResolve {
            target: r.target,
            value: r.value,
        },
        crate::scheduler::device_ops::OP_ENQUEUE_MESSAGE => ResidentEffect::MailboxSend {
            target: r.target,
            value: r.value,
        },
        crate::scheduler::device_ops::OP_RECEIVE_MESSAGE => {
            ResidentEffect::MailboxReceive { target: r.target }
        }
        _ => return Err(BackendError::ExecutionFailed),
    })
}
#[allow(clippy::too_many_arguments)]
fn decode(
    config: &ResidentSyncConfig,
    specs: &[ResidentContinuation],
    cs: &[GpuContinuation],
    frames: &[u8],
    futures: &[GpuFuture],
    mails: &[GpuMailbox],
    entries: &[GpuMailEntry],
    raw: Vec<GpuEffectRecord>,
    trace: Vec<ResidentSyncTrace>,
    completed: Vec<u64>,
    status: GpuStatus,
) -> Result<ResidentSyncResult, BackendError> {
    let stride = config.max_frame_bytes as usize;
    let mut result = ResidentSyncResult::default();
    for (index, continuation) in cs.iter().enumerate() {
        result.frames.insert(
            continuation.id,
            frames[index * stride..index * stride + continuation.frame_len as usize].to_vec(),
        );
    }
    let mut journals: BTreeMap<(u32, u32), DeviceOperationJournal> = trace
        .iter()
        .filter(|entry| entry.event == 0)
        .map(|entry| ((entry.epoch, entry.lane), DeviceOperationJournal::default()))
        .collect();
    for record in raw {
        let effect = raw_effect(&record)?;
        let outcome = raw_outcome(&record)?;
        result.accesses.push(DeviceLaneAccess::new(
            record.lane,
            match effect {
                ResidentEffect::FutureAwait { .. } | ResidentEffect::FutureResolve { .. } => {
                    RESOURCE_FUTURE
                }
                _ => RESOURCE_MAILBOX,
            },
            u64::from(record.target),
            DEVICE_ACCESS_WRITE,
            record.ordinal,
        ));
        let actor = specs
            .iter()
            .find(|continuation| continuation.id == record.continuation)
            .ok_or(BackendError::ExecutionFailed)?
            .actor;
        journals
            .get_mut(&(record.epoch, record.lane))
            .ok_or(BackendError::ExecutionFailed)?
            .operations
            .push(DeviceLaneOperation {
                lane: record.lane,
                ordinal: record.ordinal,
                opcode: record.opcode,
                actor,
                target: u64::from(record.target),
                value: record.value,
                result_code: match outcome {
                    ResidentOutcome::Resolved(_) | ResidentOutcome::Received { .. } => 2,
                    ResidentOutcome::Registered => 3,
                    ResidentOutcome::Empty => 1,
                    ResidentOutcome::InvalidTarget => 0x101,
                    ResidentOutcome::CapabilityDenied => 0x104,
                    ResidentOutcome::Full => 0x10c,
                    ResidentOutcome::DoubleResolve => 0x111,
                    ResidentOutcome::Sent => 0,
                },
                result_ref: record.result_value,
                ..Default::default()
            });
        result.effects.push(ResidentEffectRecord {
            epoch: record.epoch,
            lane: record.lane,
            ordinal: record.ordinal,
            continuation: record.continuation,
            effect,
            outcome,
        });
    }
    result.operations = journals.into_values().collect();
    result.trace = trace;
    result.completed = completed;
    result.epochs = status.epochs;
    result.quiescent = status.quiescent != 0;
    result.future_values = futures
        .iter()
        .map(|future| (future.resolved != 0).then_some(future.value))
        .collect();
    result.mailboxes = mails
        .iter()
        .enumerate()
        .map(|(index, mailbox)| {
            (0..mailbox.count)
                .map(|offset| {
                    let entry = entries[index * config.max_continuations as usize
                        + ((mailbox.head + offset) % config.max_continuations) as usize];
                    (entry.sender, entry.value)
                })
                .collect()
        })
        .collect();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(actor: u64, kind: u32, target: u32, rights: u32) -> ResidentCapability {
        ResidentCapability {
            actor,
            resource_kind: kind,
            target,
            rights,
        }
    }

    fn future_input(
        width: u32,
    ) -> (
        ResidentSyncConfig,
        Vec<ResidentContinuation>,
        BTreeMap<u32, ResidentHandlerProgram>,
    ) {
        let config = ResidentSyncConfig {
            max_epochs: 8,
            max_effects_per_step: 2,
            max_frame_bytes: 8,
            max_continuations: 2,
            cohort_width: width,
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![],
            capabilities: vec![
                cap(10, RESOURCE_FUTURE, 0, RIGHT_READ),
                cap(20, RESOURCE_FUTURE, 0, RIGHT_WRITE),
            ],
        };
        let continuations = vec![
            ResidentContinuation {
                id: 1,
                actor: 10,
                run_class: 100,
                frame: vec![0; 8],
            },
            ResidentContinuation {
                id: 2,
                actor: 20,
                run_class: 200,
                frame: vec![],
            },
        ];
        let mut programs = BTreeMap::new();
        programs.insert(
            100,
            ResidentHandlerProgram {
                run_class: 100,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        value: 77,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_AWAIT,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 100,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            200,
            ResidentHandlerProgram {
                run_class: 200,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 0,
                        value: 77,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        );
        (config, continuations, programs)
    }

    #[test]
    fn width_1_and_32_match_cpu_oracle_exactly() {
        let metal = MetalResidentSync::new().unwrap();
        let mut results = Vec::new();
        for width in [1, 32] {
            let (config, continuations, programs) = future_input(width);
            let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
            let gpu = metal.run(&config, continuations, &programs).unwrap();
            assert_eq!(gpu, cpu);
            results.push(gpu);
        }
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn mailbox_wake_delivery_and_authority_errors_match_cpu() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 8,
            max_effects_per_step: 2,
            max_frame_bytes: 8,
            max_continuations: 3,
            cohort_width: 32,
            futures: vec![],
            mailbox_capacities: vec![1],
            capabilities: vec![
                cap(10, RESOURCE_MAILBOX, 0, RIGHT_READ),
                cap(20, RESOURCE_MAILBOX, 0, RIGHT_WRITE),
            ],
        };
        let continuations = vec![
            ResidentContinuation {
                id: 1,
                actor: 10,
                run_class: 100,
                frame: vec![0; 8],
            },
            ResidentContinuation {
                id: 2,
                actor: 20,
                run_class: 200,
                frame: vec![],
            },
            ResidentContinuation {
                id: 3,
                actor: 30,
                run_class: 300,
                frame: vec![],
            },
        ];
        let mut programs = BTreeMap::new();
        programs.insert(
            100,
            ResidentHandlerProgram {
                run_class: 100,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        value: 55,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_RECEIVE,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 100,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            200,
            ResidentHandlerProgram {
                run_class: 200,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        value: 55,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            300,
            ResidentHandlerProgram {
                run_class: 300,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        value: 99,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        );
        let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
        let gpu = metal.run(&config, continuations, &programs).unwrap();
        assert_eq!(gpu, cpu);
        assert!(gpu.effects.iter().any(|e| e.outcome
            == ResidentOutcome::Received {
                value: 55,
                sender: 20
            }));
        assert!(gpu
            .effects
            .iter()
            .any(|e| e.outcome == ResidentOutcome::CapabilityDenied));
    }

    #[test]
    fn mailbox_waiters_wake_in_registration_not_identity_order() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 2,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 3,
            cohort_width: 32,
            futures: vec![],
            mailbox_capacities: vec![2],
            capabilities: vec![
                cap(10, RESOURCE_MAILBOX, 0, RIGHT_READ),
                cap(20, RESOURCE_MAILBOX, 0, RIGHT_READ),
                cap(30, RESOURCE_MAILBOX, 0, RIGHT_WRITE),
            ],
        };
        let continuations = vec![
            ResidentContinuation {
                id: 2,
                actor: 20,
                run_class: 100,
                frame: vec![],
            },
            ResidentContinuation {
                id: 1,
                actor: 10,
                run_class: 200,
                frame: vec![],
            },
            ResidentContinuation {
                id: 3,
                actor: 30,
                run_class: 300,
                frame: vec![],
            },
        ];
        let mut programs = BTreeMap::new();
        for run_class in [100, 200] {
            programs.insert(
                run_class,
                ResidentHandlerProgram {
                    run_class,
                    instructions: vec![
                        ResidentInstruction {
                            opcode: HANDLER_EFFECT_MAILBOX_RECEIVE,
                            argument: 0,
                            value: 0,
                        },
                        ResidentInstruction {
                            opcode: HANDLER_YIELD,
                            argument: run_class,
                            value: 0,
                        },
                    ],
                },
            );
        }
        programs.insert(
            300,
            ResidentHandlerProgram {
                run_class: 300,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        value: 31,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        );
        let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
        let gpu = metal.run(&config, continuations, &programs).unwrap();
        assert_eq!(gpu, cpu);
        let deliveries = gpu
            .effects
            .iter()
            .filter_map(|effect| match effect.outcome {
                ResidentOutcome::Received { value, .. } => Some((effect.continuation, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(deliveries, vec![(2, 31)]);
        assert!(!gpu.completed.contains(&1));
    }

    #[test]
    fn value_bearing_previous_survives_later_denial() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 4,
            max_effects_per_step: 2,
            max_frame_bytes: 8,
            max_continuations: 1,
            cohort_width: 1,
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![1],
            capabilities: vec![cap(7, RESOURCE_FUTURE, 0, RIGHT_WRITE)],
        };
        let continuations = vec![ResidentContinuation {
            id: 1,
            actor: 7,
            run_class: 1,
            frame: vec![0; 8],
        }];
        let mut programs = BTreeMap::new();
        programs.insert(
            1,
            ResidentHandlerProgram {
                run_class: 1,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 0,
                        value: 5,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        value: 9,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 2,
                        value: 0,
                    },
                ],
            },
        );
        programs.insert(
            2,
            ResidentHandlerProgram {
                run_class: 2,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_IF_PREVIOUS_VALUE_NE_SKIP,
                        argument: 2,
                        value: 5,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        );
        let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
        let gpu = metal.run(&config, continuations, &programs).unwrap();
        assert_eq!(gpu, cpu);
        assert_eq!(gpu.frames[&1], 5u64.to_le_bytes());
        assert_eq!(gpu.effects[1].outcome, ResidentOutcome::CapabilityDenied);
    }

    #[test]
    fn class_zero_and_invalid_target_match_oracle() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 1,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 1,
            cohort_width: 1,
            futures: vec![],
            mailbox_capacities: vec![],
            capabilities: vec![cap(7, RESOURCE_FUTURE, 9, RIGHT_WRITE)],
        };
        let continuations = vec![ResidentContinuation {
            id: 1,
            actor: 7,
            run_class: 0,
            frame: vec![],
        }];
        let programs = BTreeMap::from([(
            0,
            ResidentHandlerProgram {
                run_class: 0,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_RESOLVE,
                        argument: 9,
                        value: 1,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        )]);
        let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
        let gpu = metal.run(&config, continuations, &programs).unwrap();
        assert_eq!(gpu, cpu);
        assert_eq!(gpu.effects[0].outcome, ResidentOutcome::InvalidTarget);
    }

    #[test]
    fn absent_previous_store_fails_boundedly() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 1,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 1,
            cohort_width: 1,
            futures: vec![],
            mailbox_capacities: vec![],
            capabilities: vec![],
        };
        let continuations = vec![ResidentContinuation {
            id: 1,
            actor: 1,
            run_class: 1,
            frame: vec![0; 8],
        }];
        let programs = BTreeMap::from([(
            1,
            ResidentHandlerProgram {
                run_class: 1,
                instructions: vec![
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        value: 0,
                    },
                ],
            },
        )]);
        assert!(run_resident_sync(&config, continuations.clone(), &programs).is_none());
        assert!(matches!(
            metal.run(&config, continuations, &programs),
            Err(BackendError::ExecutionFailed)
        ));
    }

    #[test]
    fn rejects_unbounded_frames_effects_and_control_flow() {
        let metal = MetalResidentSync::new().unwrap();
        let (mut config, continuations, mut programs) = future_input(1);
        config.max_frame_bytes = 4;
        assert!(matches!(
            metal.run(&config, continuations.clone(), &programs),
            Err(BackendError::InvalidInput)
        ));
        config.max_frame_bytes = 8;
        programs.get_mut(&100).unwrap().instructions[0].argument = u32::MAX;
        assert!(matches!(
            metal.run(&config, continuations, &programs),
            Err(BackendError::InvalidInput)
        ));
    }
}
