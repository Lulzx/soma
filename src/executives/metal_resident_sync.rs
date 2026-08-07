//! Metal lowering of the standalone resident synchronization ABI.
//!
//! The complete epoch loop, handler interpreter, canonical effect applier,
//! park/wake/retry machinery, governed fixed-range object mutations, and
//! quiescence test execute in one Metal command buffer. The host performs one readback after completion. This is a
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
    invocation_capacity: u32,
    wake_capacity: u32,
    epoch_capacity: u32,
    object_count: u32,
    object_capability_count: u32,
    object_stride: u32,
    initial_epoch: u32,
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
struct GpuObject {
    len: u32,
    version: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuObjectCapability {
    actor: u64,
    offset: u64,
    length: u64,
    target: u32,
    rights: u32,
    object_version: u32,
    valid_until_epoch: u32,
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
    invocation_count: u32,
    wake_count: u32,
    epoch_count: u32,
    ticket: u32,
    reserved0: u32,
    reserved1: u32,
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Config { uint continuation_count, handler_count, instruction_count, future_count; uint mailbox_count, capability_count, max_epochs, max_effects; uint frame_stride, effect_capacity, trace_capacity, mailbox_stride; uint invocation_capacity,wake_capacity,epoch_capacity,object_count; uint object_capability_count,object_stride,initial_epoch; };
struct Cont { ulong id, actor; uint run_class, frame_len, state, last_epoch; uint previous_kind, pending_opcode, pending_target, reserved; ulong previous_value, previous_sender, pending_value; };
struct Handler { uint run_class, instruction_offset, instruction_count, reserved; };
struct Instruction { uint opcode, argument, target, reserved; ulong value; };
struct Future { uint resolved, reserved; ulong value; };
struct Mailbox { uint capacity, head, count, reserved; };
struct MailEntry { ulong sender, value; };
struct Capability { ulong actor; uint kind, target, rights, reserved; };
struct Object { uint len,version; };
struct ObjectCapability { ulong actor,offset,length; uint target,rights,object_version,valid_until_epoch; };
struct EffectRecord { uint epoch, lane, ordinal, opcode; ulong continuation; uint target, outcome; ulong value, result_value, result_sender; uint run_class, reserved; };
struct Trace { uint epoch, lane; ulong continuation; uint run_class, event, word; };
struct Status { uint effect_count, trace_count, completed_count, epochs; uint quiescent,error,invocation_count,wake_count; uint epoch_count,ticket,reserved0,reserved1; };
struct Stage { uint continuation_index, effect_offset, effect_count, disposition; uint next_run_class, reserved0, reserved1, reserved2; };
struct Invocation { uint epoch,lane; ulong continuation; uint run_class,disposition,next_run_class; };
struct Wake { uint epoch,lane,cause_opcode,target; ulong cause_continuation,continuation; uint run_class,ticket,ordinal,reserved; };
struct EpochRecord { uint epoch,invocations,runnable_after,completed_after; };
constant uint RUNNABLE=1, PARKED=2, COMPLETE=3;
constant uint AWAIT=1, RESOLVE=2, SEND=3, RECEIVE=4, OBSERVE=10, OBJECT_READ=11, OBJECT_WRITE=12;
constant uint OUT_RESOLVED=1, OUT_REGISTERED=2, OUT_SENT=3, OUT_RECEIVED=4, OUT_DENIED=5, OUT_INVALID=6, OUT_FULL=7, OUT_EMPTY=8, OUT_DOUBLE=9, OUT_PENDING=10, OUT_OBJECT_READ=11, OUT_OBJECT_WRITTEN=12;
inline uint outcome_code(uint o) { if(o==OUT_RESOLVED||o==OUT_RECEIVED||o==OUT_OBJECT_READ)return 2; if(o==OUT_REGISTERED)return 3; if(o==OUT_EMPTY||o==OUT_PENDING)return 1; if(o==OUT_INVALID)return 0x101; if(o==OUT_DENIED)return 0x104; if(o==OUT_FULL)return 0x10c; if(o==OUT_DOUBLE)return 0x111; return 0; }
inline uint journal_opcode(uint op) { return op==OBJECT_READ?2:(op==OBJECT_WRITE?7:(op==OBSERVE?1:(op==AWAIT?11:(op==RESOLVE?10:(op==SEND?8:9))))); }
inline uint hash_frame(device uchar* f,uint n){uint h=2166136261u;for(uint i=0;i<n;i++)h=(h^uint(f[i]))*16777619u;return h;}
inline bool allowed(device const Capability* caps,uint n,ulong actor,uint kind,uint target,uint right){for(uint i=0;i<n;i++){Capability c=caps[i];if(c.actor==actor&&c.kind==kind&&c.target==target&&(c.rights&right)!=0)return true;}return false;}
inline bool object_allowed(device const ObjectCapability* caps,uint n,device const Object* objects,uint object_count,ulong actor,uint target,uint offset,uint right,uint epoch){if(target>=object_count)return false;ulong end=ulong(offset)+8;for(uint i=0;i<n;i++){ObjectCapability c=caps[i];if(c.actor==actor&&c.target==target&&(c.rights&right)!=0&&c.object_version==objects[target].version&&c.valid_until_epoch>=epoch&&ulong(offset)>=c.offset&&end>=ulong(offset)&&end<=c.offset+c.length)return true;}return false;}
inline void add_trace(device Trace* traces,device Status* s,constant Config& cfg,uint epoch,uint lane,Cont c,uint event,uint word){if(s->trace_count>=cfg.trace_capacity){s->error=2;return;}Trace t={epoch,lane,c.id,c.run_class,event,word};traces[s->trace_count++]=t;}
inline void add_wake(device Wake* wakes,device Status* s,constant Config& cfg,uint epoch,uint lane,uint cause,uint target,ulong cause_continuation,uint ordinal,Cont c){if(s->wake_count>=cfg.wake_capacity){s->error=8;return;}Wake w={epoch,lane,cause,target,cause_continuation,c.id,c.run_class,c.reserved,ordinal,0};wakes[s->wake_count++]=w;}
inline void wake_future(device Cont* cs,uint n,uint target,ulong value,uint epoch,uint lane,ulong cause_continuation,uint ordinal,device Wake* wakes,device Status* s,constant Config& cfg){while(true){uint best=0xffffffffu;for(uint i=0;i<n;i++)if(cs[i].state==PARKED&&cs[i].pending_opcode==AWAIT&&cs[i].pending_target==target&&(best==0xffffffffu||cs[i].reserved<cs[best].reserved||(cs[i].reserved==cs[best].reserved&&cs[i].id<cs[best].id)))best=i;if(best==0xffffffffu)break;add_wake(wakes,s,cfg,epoch,lane,10,target,cause_continuation,ordinal,cs[best]);cs[best].state=RUNNABLE;cs[best].last_epoch=epoch;cs[best].pending_opcode=0;cs[best].previous_kind=OUT_RESOLVED;cs[best].previous_value=value;}}
inline void wake_one(device Cont* cs,uint n,uint opcode,uint target,uint epoch,uint lane,uint cause,ulong cause_continuation,uint ordinal,device Wake* wakes,device Status* s,constant Config& cfg){uint best=0xffffffffu;for(uint i=0;i<n;i++)if(cs[i].state==PARKED&&cs[i].pending_opcode==opcode&&cs[i].pending_target==target&&(best==0xffffffffu||cs[i].reserved<cs[best].reserved||(cs[i].reserved==cs[best].reserved&&cs[i].id<cs[best].id)))best=i;if(best!=0xffffffffu){add_wake(wakes,s,cfg,epoch,lane,cause,target,cause_continuation,ordinal,cs[best]);cs[best].state=RUNNABLE;cs[best].last_epoch=epoch;}}
kernel void resident_sync(
 constant Config& cfg [[buffer(0)]], device Cont* cs [[buffer(1)]], device uchar* frames [[buffer(2)]],
 device const Handler* handlers [[buffer(3)]], device const Instruction* ins [[buffer(4)]], device Future* futures [[buffer(5)]],
 device Mailbox* mails [[buffer(6)]], device MailEntry* entries [[buffer(7)]], device const Capability* caps [[buffer(8)]],
 device EffectRecord* records [[buffer(9)]], device Trace* traces [[buffer(10)]], device ulong* completed [[buffer(11)]], device Status* status [[buffer(12)]], device Stage* stages [[buffer(13)]], device Invocation* invocations [[buffer(14)]], device Wake* wakes [[buffer(15)]], device EpochRecord* epoch_records [[buffer(16)]], device Object* objects [[buffer(17)]], device uchar* object_bytes [[buffer(18)]], device const ObjectCapability* object_caps [[buffer(19)]], device EffectRecord* scratch [[buffer(20)]], device uint* evaluation_counts [[buffer(21)]], uint tid [[thread_index_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
 threadgroup uint selected[32];
 threadgroup uint cohort_count;
 threadgroup uint epoch_lanes;
 threadgroup uint selection_class;
 threadgroup uint selection_cursor;
 threadgroup uint selection_class_active;
 threadgroup uint stop;
 if(tid==0){Status s={0,0,0,0,0,0,0,0,0,0,0,0};*status=s;stop=0;}
 evaluation_counts[tid]=0;
 threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
 if(cfg.continuation_count==0){if(tid==0)status->quiescent=1;return;}
 for(uint epoch=0;epoch<cfg.max_epochs;epoch++){
  if(tid==0){epoch_lanes=0;selection_class_active=0;selection_cursor=0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Select bounded same-run-class cohorts in canonical (run_class,id) order.
  // The host stores continuations in id order. Lane zero finds each next class
  // once, then continues one forward scan across as many physical cohorts as
  // necessary. Thus selection is bounded by O(H*N), rather than rescanning N
  // candidates for every selected lane. last_epoch is published before worker
  // evaluation, preserving the full-epoch snapshot and deferring wakes to the
  // next epoch. No class sentinel is used, so run class zero is ordinary.
  while(true){
   if(tid==0){
    cohort_count=0;
    while(cohort_count==0){
     if(selection_class_active==0){
      bool found=false;uint next_class=0;
      for(uint i=0;i<cfg.continuation_count;i++)if(cs[i].state==RUNNABLE&&cs[i].last_epoch!=epoch&&(!found||cs[i].run_class<next_class)){next_class=cs[i].run_class;found=true;}
      if(!found)break;
      selection_class=next_class;selection_cursor=0;selection_class_active=1;
     }
     while(selection_cursor<cfg.continuation_count&&cohort_count<tpg.x){
      uint at=selection_cursor++;
      if(cs[at].state==RUNNABLE&&cs[at].last_epoch!=epoch&&cs[at].run_class==selection_class){selected[cohort_count++]=at;cs[at].last_epoch=epoch;}
     }
     if(selection_cursor==cfg.continuation_count)selection_class_active=0;
    }
   }
   threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
   uint count=cohort_count;
   if(count==0)break;
   if(tid<count){
    uint at=selected[tid],lane=epoch_lanes+tid;
    Cont c=cs[at];c.state=0;uint disposition=0,next_class=c.run_class,emitted=0,error=0;
    uint retry_opcode=c.pending_opcode,retry_target=c.pending_target;ulong retry_value=c.pending_value;
    uint input_previous_kind=c.previous_kind;ulong input_previous_value=c.previous_value;
    uint pc=0,end=0;
    if(retry_opcode==0){
     c.previous_kind=0;c.previous_value=0;c.previous_sender=0;uint hi=0xffffffffu;
     for(uint h=0;h<cfg.handler_count;h++)if(handlers[h].run_class==c.run_class)hi=h;
     if(hi==0xffffffffu)error=3;else{pc=handlers[hi].instruction_offset;end=pc+handlers[hi].instruction_count;}
    }else{c.pending_opcode=0;c.pending_target=0;c.pending_value=0;}
    while(error==0&&(retry_opcode!=0||pc<end)){
     uint op,arg,target;ulong val;
     if(retry_opcode!=0){op=retry_opcode;arg=retry_target;target=retry_target;val=retry_value;retry_opcode=0;disposition=1;next_class=c.run_class;}
     else{
      Instruction x=ins[pc++];op=x.opcode;arg=x.argument;target=(op==OBJECT_READ||op==OBJECT_WRITE)?x.target:x.argument;val=x.value;
      if(op==5||op==6){if(arg+8>c.frame_len||(op==6&&input_previous_kind!=OUT_RESOLVED&&input_previous_kind!=OUT_RECEIVED&&input_previous_kind!=OUT_OBJECT_READ)){error=4;break;}ulong v=op==5?val:input_previous_value;for(uint bb=0;bb<8;bb++)frames[at*cfg.frame_stride+arg+bb]=uchar(v>>(bb*8));continue;}
      if(op==7){if((input_previous_kind!=OUT_RESOLVED&&input_previous_kind!=OUT_RECEIVED&&input_previous_kind!=OUT_OBJECT_READ)||input_previous_value!=val)pc+=arg;continue;}
      if(op==8){disposition=1;next_class=arg;break;}if(op==9){disposition=2;next_class=0;break;}
     }
     if(!((op>=1&&op<=4)||op==OBSERVE||op==OBJECT_READ||op==OBJECT_WRITE)||emitted>=cfg.max_effects){error=5;break;}
     uint scratch_at=lane*cfg.max_effects+emitted;
     EffectRecord r={epoch,lane,emitted,journal_opcode(op),c.id,target,0,val,0,0,c.run_class,(op==OBJECT_READ||op==OBJECT_WRITE)?arg:0};scratch[scratch_at]=r;emitted++;
    }
    if(error==0&&disposition==0)error=7;
    Stage st={at,lane*cfg.max_effects,emitted,disposition,next_class,0,hash_frame(frames+at*cfg.frame_stride,c.frame_len),error};stages[lane]=st;cs[at]=c;evaluation_counts[tid]++;
   }
   threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
   if(tid==0)epoch_lanes+=count;
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Phase E is epoch-wide: only after every runnable handler has finished does
  // lane zero compact scratch and publish the canonical invocation/trace prefix.
  if(tid==0){
   for(uint lane=0;lane<epoch_lanes;lane++){
    Stage st=stages[lane];if(st.reserved2!=0){status->error=st.reserved2;stop=1;break;}
    if(status->invocation_count>=cfg.invocation_capacity){status->error=9;stop=1;break;}
    Cont c=cs[st.continuation_index];uint invocation_index=status->invocation_count;
    Invocation iv={epoch,lane,c.id,c.run_class,st.disposition,st.next_run_class};invocations[status->invocation_count++]=iv;
    stages[lane].reserved0=invocation_index;
    add_trace(traces,status,cfg,epoch,lane,c,0,st.reserved1);if(status->error!=0){stop=1;break;}
    stages[lane].effect_offset=status->effect_count;
    for(uint ordinal=0;ordinal<st.effect_count;ordinal++){
     if(status->effect_count>=cfg.effect_capacity){status->error=5;stop=1;break;}
     records[status->effect_count++]=scratch[st.effect_offset+ordinal];
    }
    if(stop!=0)break;
   }
  }
  threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
  if(stop!=0)return;
  // Canonical applier. No worker writes futures, mailboxes, objects,
  // capabilities, journals, wake state, or shared counters.
  if(tid==0)for(uint li=0;li<epoch_lanes;li++){
   Stage st=stages[li];Cont c=cs[st.continuation_index];bool parked=false;
   for(uint ri=st.effect_offset;ri<st.effect_offset+st.effect_count;ri++){
    EffectRecord r=records[ri];uint op=r.opcode==2?OBJECT_READ:(r.opcode==7?OBJECT_WRITE:(r.opcode==1?OBSERVE:(r.opcode==11?AWAIT:(r.opcode==10?RESOLVE:(r.opcode==8?SEND:RECEIVE)))));uint arg=r.target;uint offset=r.reserved;ulong val=r.value;
    uint kind=(op==OBJECT_READ||op==OBJECT_WRITE)?1:((op<=2||op==OBSERVE)?2:3);uint right=(op==OBJECT_READ||op==OBSERVE||op==AWAIT||op==RECEIVE)?1:2;uint outcome=0;ulong result_value=0,result_sender=0;
    bool authorized=kind==1?object_allowed(object_caps,cfg.object_capability_count,objects,cfg.object_count,c.actor,arg,offset,right,cfg.initial_epoch+epoch):allowed(caps,cfg.capability_count,c.actor,kind,arg,right);
    if(!authorized)outcome=OUT_DENIED;
    else if((kind==1&&arg>=cfg.object_count)||(kind==2&&arg>=cfg.future_count)||(kind==3&&arg>=cfg.mailbox_count))outcome=OUT_INVALID;
    else if(kind==1&&(ulong(offset)+8>objects[arg].len||ulong(offset)+8<ulong(offset)))outcome=OUT_INVALID;
    else if(op==OBJECT_READ){uint base=arg*cfg.object_stride+offset;for(uint bb=0;bb<8;bb++)result_value|=ulong(object_bytes[base+bb])<<(bb*8);outcome=OUT_OBJECT_READ;}
    else if(op==OBJECT_WRITE){uint base=arg*cfg.object_stride+offset;for(uint bb=0;bb<8;bb++)object_bytes[base+bb]=uchar(val>>(bb*8));result_value=val;outcome=OUT_OBJECT_WRITTEN;}
    else if(op==OBSERVE){if(futures[arg].resolved!=0){outcome=OUT_RESOLVED;result_value=futures[arg].value;}else outcome=OUT_PENDING;}
    else if(op==AWAIT){if(futures[arg].resolved!=0){outcome=OUT_RESOLVED;result_value=futures[arg].value;}else{outcome=OUT_REGISTERED;parked=true;c.pending_opcode=AWAIT;c.pending_target=arg;}}
    else if(op==RESOLVE){if(futures[arg].resolved!=0)outcome=OUT_DOUBLE;else{futures[arg].resolved=1;futures[arg].value=val;outcome=OUT_RESOLVED;result_value=val;wake_future(cs,cfg.continuation_count,arg,val,epoch,li,c.id,r.ordinal,wakes,status,cfg);}}
    else if(op==SEND){Mailbox m=mails[arg];if(m.count>=m.capacity){outcome=OUT_FULL;parked=true;c.pending_opcode=SEND;c.pending_target=arg;c.pending_value=val;}else{uint slot=arg*cfg.mailbox_stride+(m.head+m.count)%cfg.mailbox_stride;MailEntry ne={c.actor,val};entries[slot]=ne;m.count++;mails[arg]=m;outcome=OUT_SENT;wake_one(cs,cfg.continuation_count,RECEIVE,arg,epoch,li,8,c.id,r.ordinal,wakes,status,cfg);}}
    else{Mailbox m=mails[arg];if(m.count==0){outcome=OUT_EMPTY;parked=true;c.pending_opcode=RECEIVE;c.pending_target=arg;}else{uint slot=arg*cfg.mailbox_stride+m.head;MailEntry e=entries[slot];m.head=(m.head+1)%cfg.mailbox_stride;m.count--;mails[arg]=m;outcome=OUT_RECEIVED;result_value=e.value;result_sender=e.sender;c.pending_opcode=0;wake_one(cs,cfg.continuation_count,SEND,arg,epoch,li,9,c.id,r.ordinal,wakes,status,cfg);}}
    if(outcome==OUT_REGISTERED||outcome==OUT_FULL||outcome==OUT_EMPTY){c.reserved=++status->ticket;c.state=PARKED;cs[st.continuation_index]=c;}
    r.outcome=outcome;r.result_value=result_value;r.result_sender=result_sender;records[ri]=r;add_trace(traces,status,cfg,epoch,li,c,r.opcode,outcome_code(outcome));
    // A later effect in this same lane may wake an effect that parked above.
    // Reconcile the lane-local copy with that authoritative wake before the
    // final disposition write, otherwise it would overwrite RUNNABLE/pending.
    Cont authoritative=cs[st.continuation_index];if(c.state==PARKED&&authoritative.state==RUNNABLE){c.state=RUNNABLE;c.pending_opcode=authoritative.pending_opcode;c.pending_target=authoritative.pending_target;c.pending_value=authoritative.pending_value;c.previous_kind=authoritative.previous_kind;c.previous_value=authoritative.previous_value;c.previous_sender=authoritative.previous_sender;}
    if(outcome==OUT_RESOLVED||outcome==OUT_RECEIVED||outcome==OUT_OBJECT_READ){c.previous_kind=outcome;c.previous_value=result_value;c.previous_sender=result_sender;}
   }
   if(parked){if(c.state!=RUNNABLE)c.state=PARKED;invocations[st.reserved0].disposition=3;invocations[st.reserved0].next_run_class=c.run_class;}else if(st.disposition==1){c.state=RUNNABLE;c.run_class=st.next_run_class;}else{c.state=COMPLETE;completed[status->completed_count++]=c.id;}cs[st.continuation_index]=c;
  }
  threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
  if(tid==0){
   if(status->error!=0){stop=1;}else{
    status->epochs=epoch+1;uint runnable_after=0,completed_after=0;for(uint i=0;i<cfg.continuation_count;i++){if(cs[i].state==RUNNABLE)runnable_after++;if(cs[i].state==COMPLETE)completed_after++;}
    if(status->epoch_count>=cfg.epoch_capacity){status->error=10;stop=1;}else{EpochRecord er={epoch,epoch_lanes,runnable_after,completed_after};epoch_records[status->epoch_count++]=er;if(runnable_after==0){status->quiescent=1;stop=1;}}
   }
  }
  // This barrier is also the epoch publication point: all lanes observe the
  // canonical state transitions and wakes before the next selection pass.
  threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
  if(stop!=0)break;
 }
 if(tid==0&&status->error==0){bool any=false;for(uint i=0;i<cfg.continuation_count;i++)if(cs[i].state==RUNNABLE)any=true;status->quiescent=any?0:1;}
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
        continuations: Vec<ResidentContinuation>,
        programs: &BTreeMap<u32, ResidentHandlerProgram>,
    ) -> Result<ResidentSyncResult, BackendError> {
        self.run_internal(config, continuations, programs)
            .map(|(result, _)| result)
    }

    #[cfg(test)]
    fn run_with_evaluation_counts(
        &self,
        config: &ResidentSyncConfig,
        continuations: Vec<ResidentContinuation>,
        programs: &BTreeMap<u32, ResidentHandlerProgram>,
    ) -> Result<(ResidentSyncResult, Vec<u32>), BackendError> {
        self.run_internal(config, continuations, programs)
    }

    fn run_internal(
        &self,
        config: &ResidentSyncConfig,
        mut continuations: Vec<ResidentContinuation>,
        programs: &BTreeMap<u32, ResidentHandlerProgram>,
    ) -> Result<(ResidentSyncResult, Vec<u32>), BackendError> {
        validate(config, &continuations, programs)?;
        if config.cohort_width > 32
            || u64::from(config.cohort_width) > self.pipeline.max_total_threads_per_threadgroup()
        {
            return Err(BackendError::InvalidInput);
        }
        continuations.sort_by_key(|c| c.id);
        let effect_capacity =
            checked_capacity(config, continuations.len(), config.max_effects_per_step)?;
        let invocation_capacity = (config.max_epochs as usize)
            .checked_mul(continuations.len())
            .ok_or(BackendError::InvalidInput)?;
        let trace_capacity = effect_capacity
            .checked_add(
                (config.max_epochs as usize)
                    .checked_mul(continuations.len())
                    .ok_or(BackendError::InvalidInput)?,
            )
            .ok_or(BackendError::InvalidInput)?;
        let frame_count = continuations
            .len()
            .checked_mul(config.max_frame_bytes as usize)
            .ok_or(BackendError::InvalidInput)?;
        let instruction_count = programs.values().try_fold(0usize, |count, program| {
            count
                .checked_add(program.instructions.len())
                .ok_or(BackendError::InvalidInput)
        })?;
        let entry_count = config
            .mailbox_capacities
            .len()
            .checked_mul(config.max_continuations as usize)
            .ok_or(BackendError::InvalidInput)?;
        let object_byte_count = config
            .objects
            .len()
            .checked_mul(config.max_frame_bytes as usize)
            .ok_or(BackendError::InvalidInput)?;
        let scratch_count = continuations
            .len()
            .checked_mul(config.max_effects_per_step as usize)
            .ok_or(BackendError::InvalidInput)?;
        // Every device offset in the shader is u32. Refuse padded arenas whose
        // element counts would wrap even when the host usize multiplication fits.
        for count in [frame_count, entry_count, object_byte_count, scratch_count] {
            u32::try_from(count).map_err(|_| BackendError::InvalidInput)?;
        }
        self.check_buffer::<GpuConfig>(1)?;
        self.check_buffer::<GpuContinuation>(continuations.len())?;
        self.check_buffer::<u8>(frame_count)?;
        self.check_buffer::<GpuHandler>(programs.len())?;
        self.check_buffer::<ResidentInstruction>(instruction_count)?;
        self.check_buffer::<GpuFuture>(config.futures.len())?;
        self.check_buffer::<GpuMailbox>(config.mailbox_capacities.len())?;
        self.check_buffer::<GpuMailEntry>(entry_count)?;
        self.check_buffer::<GpuCapability>(config.capabilities.len())?;
        self.check_buffer::<GpuEffectRecord>(effect_capacity)?;
        self.check_buffer::<ResidentSyncTrace>(trace_capacity)?;
        self.check_buffer::<u64>(continuations.len())?;
        self.check_buffer::<GpuStatus>(1)?;
        self.check_buffer::<GpuStage>(continuations.len())?;
        self.check_buffer::<ResidentInvocationRecord>(invocation_capacity)?;
        self.check_buffer::<ResidentWakeRecord>(invocation_capacity)?;
        self.check_buffer::<ResidentEpochRecord>(config.max_epochs as usize)?;
        self.check_buffer::<GpuObject>(config.objects.len())?;
        self.check_buffer::<u8>(object_byte_count)?;
        self.check_buffer::<GpuObjectCapability>(config.object_capabilities.len())?;
        self.check_buffer::<GpuEffectRecord>(scratch_count)?;
        self.check_buffer::<u32>(config.cohort_width as usize)?;
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
        let mut frames = vec![0u8; frame_count];
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
            .zip(&config.mailbox_messages)
            .map(|(capacity, messages)| GpuMailbox {
                capacity: *capacity,
                count: messages.len() as u32,
                ..Default::default()
            })
            .collect();
        let mut mailbox_entries = vec![GpuMailEntry::default(); entry_count];
        for (mailbox, messages) in config.mailbox_messages.iter().enumerate() {
            for (offset, (sender, value)) in messages.iter().copied().enumerate() {
                mailbox_entries[mailbox * config.max_continuations as usize + offset] =
                    GpuMailEntry { sender, value };
            }
        }
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
        let object_stride = config.max_frame_bytes as usize;
        if config
            .objects
            .iter()
            .any(|object| object.bytes.len() > object_stride)
        {
            return Err(BackendError::InvalidInput);
        }
        let mut object_bytes = vec![0u8; object_byte_count];
        let objects: Vec<GpuObject> = config
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                object_bytes[index * object_stride..index * object_stride + object.bytes.len()]
                    .copy_from_slice(&object.bytes);
                GpuObject {
                    len: object.bytes.len() as u32,
                    version: object.version,
                }
            })
            .collect();
        let object_capabilities: Vec<GpuObjectCapability> = config
            .object_capabilities
            .iter()
            .map(|cap| GpuObjectCapability {
                actor: cap.actor,
                offset: cap.offset,
                length: cap.length,
                target: cap.target,
                rights: cap.rights,
                object_version: cap.object_version,
                valid_until_epoch: cap.valid_until_epoch,
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
            invocation_capacity: invocation_capacity as u32,
            wake_capacity: invocation_capacity as u32,
            epoch_capacity: config.max_epochs,
            object_count: objects.len() as u32,
            object_capability_count: object_capabilities.len() as u32,
            object_stride: object_stride as u32,
            initial_epoch: config.initial_epoch,
        };
        let cfg_b = self.buffer_from(std::slice::from_ref(&cfg));
        let cs_b = self.buffer_from(&gpu_conts);
        let frames_b = self.buffer_from(&frames);
        let handlers_b = self.buffer_from(&handlers);
        let ins_b = self.buffer_from(&instructions);
        let futures_b = self.buffer_from(&futures);
        let mail_b = self.buffer_from(&mailboxes);
        let entries_b = self.buffer_from(&mailbox_entries);
        let caps_b = self.buffer_from(&capabilities);
        let effects_b = self.zero_buffer::<GpuEffectRecord>(effect_capacity);
        let traces_b = self.zero_buffer::<ResidentSyncTrace>(trace_capacity);
        let completed_b = self.zero_buffer::<u64>(continuations.len());
        let status_b = self.zero_buffer::<GpuStatus>(1);
        let stages_b = self.zero_buffer::<GpuStage>(continuations.len());
        let invocations_b = self.zero_buffer::<ResidentInvocationRecord>(invocation_capacity);
        let wakes_b = self.zero_buffer::<ResidentWakeRecord>(invocation_capacity);
        let epochs_b = self.zero_buffer::<ResidentEpochRecord>(config.max_epochs as usize);
        let objects_b = self.buffer_from(&objects);
        let object_bytes_b = self.buffer_from(&object_bytes);
        let object_caps_b = self.buffer_from(&object_capabilities);
        // Reused every epoch. Each epoch lane owns exactly max_effects records,
        // so handler evaluation never contends on the canonical journal.
        let scratch_b = self.zero_buffer::<GpuEffectRecord>(scratch_count);
        // Per-physical-lane instrumentation is deliberately not part of the
        // semantic result. It makes parallel evaluation observable to focused
        // backend tests without introducing atomics or shared counter races.
        let evaluation_counts_b = self.zero_buffer::<u32>(config.cohort_width as usize);
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
            &invocations_b,
            &wakes_b,
            &epochs_b,
            &objects_b,
            &object_bytes_b,
            &object_caps_b,
            &scratch_b,
            &evaluation_counts_b,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        // One bounded threadgroup evaluates cohorts in parallel; lane zero is
        // the sole canonical compactor/applier between uniform barriers.
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
        let final_object_bytes: Vec<u8> = read_vec(&object_bytes_b, object_bytes.len());
        let final_futures: Vec<GpuFuture> = read_vec(&futures_b, futures.len());
        let final_mails: Vec<GpuMailbox> = read_vec(&mail_b, mailboxes.len());
        let final_entries: Vec<GpuMailEntry> = read_vec(&entries_b, entry_count);
        let status: GpuStatus = read_vec(&status_b, 1)[0];
        if status.error != 0
            || status.effect_count as usize > effect_capacity
            || status.trace_count as usize > trace_capacity
            || status.completed_count as usize > continuations.len()
            || status.invocation_count as usize > invocation_capacity
            || status.wake_count as usize > invocation_capacity
            || status.epoch_count > config.max_epochs
        {
            return Err(BackendError::ExecutionFailed);
        }
        let raw_effects: Vec<GpuEffectRecord> = read_vec(&effects_b, status.effect_count as usize);
        let trace = read_vec(&traces_b, status.trace_count as usize);
        let completed = read_vec(&completed_b, status.completed_count as usize);
        let invocations = read_vec(&invocations_b, status.invocation_count as usize);
        let wakes = read_vec(&wakes_b, status.wake_count as usize);
        let epoch_records = read_vec(&epochs_b, status.epoch_count as usize);
        let evaluation_counts = read_vec(&evaluation_counts_b, config.cohort_width as usize);
        let result = decode(
            config,
            &continuations,
            &gpu_conts,
            &final_frames,
            &final_object_bytes,
            &final_futures,
            &final_mails,
            &final_entries,
            raw_effects,
            trace,
            completed,
            invocations,
            wakes,
            epoch_records,
            status,
        )?;
        Ok((result, evaluation_counts))
    }
    fn check_buffer<T>(&self, count: usize) -> Result<(), BackendError> {
        let bytes = count
            .max(1)
            .checked_mul(size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(BackendError::InvalidInput)?;
        if bytes > self.device.max_buffer_length() {
            return Err(BackendError::InvalidInput);
        }
        Ok(())
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
        || config
            .initial_epoch
            .checked_add(config.max_epochs)
            .is_none()
        || !object_storage_is_bounded(config)
        || config
            .objects
            .iter()
            .any(|object| object.bytes.len() > config.max_frame_bytes as usize)
        || config.object_capabilities.iter().any(|cap| {
            let Some(object) = config.objects.get(cap.target as usize) else {
                return true;
            };
            cap.object_version != object.version
                || cap
                    .offset
                    .checked_add(cap.length)
                    .is_none_or(|end| end > object.bytes.len() as u64)
        })
        || u32::try_from(config.futures.len()).is_err()
        || u32::try_from(config.mailbox_capacities.len()).is_err()
        || config.mailbox_messages.len() != config.mailbox_capacities.len()
        || u32::try_from(config.capabilities.len()).is_err()
        || config
            .mailbox_capacities
            .iter()
            .zip(&config.mailbox_messages)
            .any(|(capacity, messages)| {
                *capacity > config.max_continuations || messages.len() > *capacity as usize
            })
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
        10 => ResidentOutcome::Pending,
        11 => ResidentOutcome::ObjectRead(r.result_value),
        12 => ResidentOutcome::ObjectWritten,
        _ => return Err(BackendError::ExecutionFailed),
    })
}
fn raw_effect(r: &GpuEffectRecord) -> Result<ResidentEffect, BackendError> {
    Ok(match r.opcode {
        crate::scheduler::device_ops::OP_READ_OBJECT => ResidentEffect::ObjectRead {
            target: r.target,
            offset: r.reserved,
        },
        crate::scheduler::device_ops::OP_WRITE_OBJECT => ResidentEffect::ObjectWrite {
            target: r.target,
            offset: r.reserved,
            value: r.value,
        },
        crate::scheduler::device_ops::OP_OBSERVE_FUTURE => {
            ResidentEffect::FutureObserve { target: r.target }
        }
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
    object_bytes: &[u8],
    futures: &[GpuFuture],
    mails: &[GpuMailbox],
    entries: &[GpuMailEntry],
    raw: Vec<GpuEffectRecord>,
    trace: Vec<ResidentSyncTrace>,
    completed: Vec<u64>,
    invocations: Vec<ResidentInvocationRecord>,
    wakes: Vec<ResidentWakeRecord>,
    epoch_records: Vec<ResidentEpochRecord>,
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
    result.object_values = config
        .objects
        .iter()
        .enumerate()
        .map(|(index, object)| {
            object_bytes[index * stride..index * stride + object.bytes.len()].to_vec()
        })
        .collect();
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
                ResidentEffect::ObjectRead { .. } | ResidentEffect::ObjectWrite { .. } => {
                    RESOURCE_OBJECT
                }
                ResidentEffect::FutureObserve { .. }
                | ResidentEffect::FutureAwait { .. }
                | ResidentEffect::FutureResolve { .. } => RESOURCE_FUTURE,
                _ => RESOURCE_MAILBOX,
            },
            u64::from(record.target),
            if matches!(
                effect,
                ResidentEffect::FutureObserve { .. } | ResidentEffect::ObjectRead { .. }
            ) {
                crate::scheduler::device::DEVICE_ACCESS_READ
            } else {
                DEVICE_ACCESS_WRITE
            },
            record.ordinal,
        ));
        let actor = specs
            .iter()
            .find(|continuation| continuation.id == record.continuation)
            .ok_or(BackendError::ExecutionFailed)?
            .actor;
        let journal = journals
            .get_mut(&(record.epoch, record.lane))
            .ok_or(BackendError::ExecutionFailed)?;
        let payload = match outcome {
            ResidentOutcome::ObjectRead(value) => value.to_le_bytes().to_vec(),
            ResidentOutcome::ObjectWritten => record.value.to_le_bytes().to_vec(),
            _ => Vec::new(),
        };
        let (payload_offset, payload_len) = journal
            .append_payload(&payload)
            .ok_or(BackendError::ExecutionFailed)?;
        journal.operations.push(DeviceLaneOperation {
            lane: record.lane,
            ordinal: record.ordinal,
            opcode: record.opcode,
            actor,
            target: u64::from(record.target),
            value: record.value,
            auxiliary: u64::from(record.reserved),
            payload_offset,
            payload_len,
            result_code: match outcome {
                ResidentOutcome::ObjectRead(_)
                | ResidentOutcome::Resolved(_)
                | ResidentOutcome::Received { .. } => 2,
                ResidentOutcome::ObjectWritten => 0,
                ResidentOutcome::Registered => 3,
                ResidentOutcome::Empty | ResidentOutcome::Pending => 1,
                ResidentOutcome::InvalidTarget => 0x101,
                ResidentOutcome::CapabilityDenied => 0x104,
                ResidentOutcome::Full => 0x10c,
                ResidentOutcome::DoubleResolve => 0x111,
                ResidentOutcome::Sent => 0,
            },
            result_ref: if matches!(outcome, ResidentOutcome::ObjectWritten) {
                0
            } else {
                record.result_value
            },
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
    result.invocations = invocations;
    result.wakes = wakes;
    result.epoch_records = epoch_records;
    result.final_continuations = cs
        .iter()
        .map(|continuation| ResidentFinalContinuation {
            id: continuation.id,
            run_class: continuation.run_class,
            completed: continuation.state == 3,
            pending: match continuation.pending_opcode {
                0 => None,
                1 => Some(ResidentEffect::FutureAwait {
                    target: continuation.pending_target,
                }),
                3 => Some(ResidentEffect::MailboxSend {
                    target: continuation.pending_target,
                    value: continuation.pending_value,
                }),
                4 => Some(ResidentEffect::MailboxReceive {
                    target: continuation.pending_target,
                }),
                _ => None,
            },
            waiter_order: if continuation.pending_opcode == 0 {
                0
            } else {
                continuation.reserved
            },
        })
        .collect();
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

    fn instruction(opcode: u32, argument: u32, target: u32, value: u64) -> ResidentInstruction {
        ResidentInstruction {
            opcode,
            argument,
            target,
            reserved: 0,
            value,
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
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
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
                        target: 0,
                        reserved: 0,
                        value: 77,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_FUTURE_AWAIT,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 100,
                        target: 0,
                        reserved: 0,
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
                        target: 0,
                        reserved: 0,
                        value: 77,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                ],
            },
        );
        (config, continuations, programs)
    }

    fn observe_input(
        width: u32,
    ) -> (
        ResidentSyncConfig,
        Vec<ResidentContinuation>,
        BTreeMap<u32, ResidentHandlerProgram>,
    ) {
        let config = ResidentSyncConfig {
            max_epochs: 1,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 4,
            cohort_width: width,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending, InitialFuture::Resolved(44)],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![
                cap(7, RESOURCE_FUTURE, 0, RIGHT_READ),
                cap(7, RESOURCE_FUTURE, 1, RIGHT_READ),
                cap(7, RESOURCE_FUTURE, 9, RIGHT_READ),
            ],
        };
        let continuations = [(1, 7, 10, 0), (2, 7, 11, 1), (3, 8, 12, 0), (4, 7, 13, 9)]
            .into_iter()
            .map(|(id, actor, run_class, _)| ResidentContinuation {
                id,
                actor,
                run_class,
                frame: vec![],
            })
            .collect();
        let programs = [(10, 0), (11, 1), (12, 0), (13, 9)]
            .into_iter()
            .map(|(run_class, target)| {
                (
                    run_class,
                    ResidentHandlerProgram {
                        run_class,
                        instructions: vec![
                            ResidentInstruction {
                                opcode: HANDLER_EFFECT_FUTURE_OBSERVE,
                                argument: target,
                                target: 0,
                                reserved: 0,
                                value: 0,
                            },
                            ResidentInstruction {
                                opcode: HANDLER_COMPLETE,
                                argument: 0,
                                target: 0,
                                reserved: 0,
                                value: 0,
                            },
                        ],
                    },
                )
            })
            .collect();
        (config, continuations, programs)
    }

    #[test]
    fn observe_pending_resolved_denied_invalid_width_1_32_match_cpu_exactly() {
        let metal = MetalResidentSync::new().unwrap();
        let mut results = Vec::new();
        for width in [1, 32] {
            let (config, continuations, programs) = observe_input(width);
            let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
            let gpu = metal.run(&config, continuations, &programs).unwrap();
            assert_eq!(gpu, cpu);
            assert_eq!(
                gpu.effects
                    .iter()
                    .map(|effect| effect.outcome)
                    .collect::<Vec<_>>(),
                vec![
                    ResidentOutcome::Pending,
                    ResidentOutcome::Resolved(44),
                    ResidentOutcome::CapabilityDenied,
                    ResidentOutcome::InvalidTarget,
                ]
            );
            assert!(gpu
                .accesses
                .iter()
                .all(|access| { access.mode == crate::scheduler::device::DEVICE_ACCESS_READ }));
            assert!(gpu
                .operations
                .iter()
                .flat_map(|journal| &journal.operations)
                .all(
                    |operation| operation.opcode == crate::scheduler::device_ops::OP_OBSERVE_FUTURE
                ));
            results.push(gpu);
        }
        assert_eq!(results[0], results[1]);
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
            assert_eq!(
                gpu.wakes
                    .iter()
                    .map(|wake| {
                        (
                            wake.epoch,
                            wake.lane,
                            wake.cause_continuation,
                            wake.continuation,
                            wake.ticket,
                        )
                    })
                    .collect::<Vec<_>>(),
                vec![(0, 1, 2, 1, 1)]
            );
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
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![],
            mailbox_capacities: vec![1],
            mailbox_messages: vec![vec![]],
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
                        target: 0,
                        reserved: 0,
                        value: 55,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_RECEIVE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 100,
                        target: 0,
                        reserved: 0,
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
                        target: 0,
                        reserved: 0,
                        value: 55,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
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
                        target: 0,
                        reserved: 0,
                        value: 99,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
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
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![],
            mailbox_capacities: vec![2],
            mailbox_messages: vec![vec![]],
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
                            target: 0,
                            reserved: 0,
                            value: 0,
                        },
                        ResidentInstruction {
                            opcode: HANDLER_YIELD,
                            argument: run_class,
                            target: 0,
                            reserved: 0,
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
                        target: 0,
                        reserved: 0,
                        value: 31,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
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
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![1],
            mailbox_messages: vec![vec![]],
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
                        target: 0,
                        reserved: 0,
                        value: 5,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_EFFECT_MAILBOX_SEND,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 9,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_YIELD,
                        argument: 2,
                        target: 0,
                        reserved: 0,
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
                        target: 0,
                        reserved: 0,
                        value: 5,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_STORE_PREVIOUS_VALUE_U64,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
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
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
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
                        target: 0,
                        reserved: 0,
                        value: 1,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
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
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
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
                        target: 0,
                        reserved: 0,
                        value: 0,
                    },
                    ResidentInstruction {
                        opcode: HANDLER_COMPLETE,
                        argument: 0,
                        target: 0,
                        reserved: 0,
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
    #[test]
    fn bounded_object_read_write_width_1_32_match_cpu_and_exact_journals() {
        fn case(width: u32) -> (ResidentSyncResult, ResidentSyncResult) {
            let metal = MetalResidentSync::new().unwrap();
            let config = ResidentSyncConfig {
                max_epochs: 1,
                max_effects_per_step: 2,
                max_frame_bytes: 16,
                max_continuations: 1,
                cohort_width: width,
                initial_epoch: 7,
                objects: vec![ResidentObject {
                    version: 3,
                    bytes: (0u8..16).collect(),
                }],
                object_capabilities: vec![ResidentObjectCapability {
                    actor: 41,
                    target: 0,
                    offset: 0,
                    length: 16,
                    rights: RIGHT_READ | RIGHT_WRITE,
                    object_version: 3,
                    valid_until_epoch: 8,
                }],
                futures: vec![],
                mailbox_capacities: vec![],
                mailbox_messages: vec![],
                capabilities: vec![],
            };
            let continuations = vec![ResidentContinuation {
                id: 9,
                actor: 41,
                run_class: 1200,
                frame: vec![0; 8],
            }];
            let programs = BTreeMap::from([(
                1200,
                ResidentHandlerProgram {
                    run_class: 1200,
                    instructions: vec![
                        ResidentInstruction {
                            opcode: HANDLER_EFFECT_OBJECT_READ,
                            argument: 4,
                            target: 0,
                            reserved: 0,
                            value: 0,
                        },
                        ResidentInstruction {
                            opcode: HANDLER_EFFECT_OBJECT_WRITE,
                            argument: 8,
                            target: 0,
                            reserved: 0,
                            value: 0x8877665544332211,
                        },
                        ResidentInstruction {
                            opcode: HANDLER_COMPLETE,
                            argument: 0,
                            target: 0,
                            reserved: 0,
                            value: 0,
                        },
                    ],
                },
            )]);
            let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
            let gpu = metal.run(&config, continuations, &programs).unwrap();
            (cpu, gpu)
        }
        let (cpu1, gpu1) = case(1);
        let (cpu32, gpu32) = case(32);
        assert_eq!(gpu1, cpu1);
        assert_eq!(gpu32, cpu32);
        assert_eq!(gpu1, gpu32);
        assert_eq!(
            gpu1.effects[0].outcome,
            ResidentOutcome::ObjectRead(0x0b0a090807060504)
        );
        assert_eq!(
            &gpu1.object_values[0][8..16],
            &0x8877665544332211u64.to_le_bytes()
        );
        let journal = &gpu1.operations[0];
        assert_eq!(
            journal
                .operations
                .iter()
                .map(|op| (op.opcode, op.auxiliary, op.payload_len, op.result_ref))
                .collect::<Vec<_>>(),
            vec![
                (
                    crate::scheduler::device_ops::OP_READ_OBJECT,
                    4,
                    8,
                    0x0b0a090807060504
                ),
                (crate::scheduler::device_ops::OP_WRITE_OBJECT, 8, 8, 0)
            ]
        );
        assert_eq!(
            journal.payload,
            [
                0x0b0a090807060504u64.to_le_bytes(),
                0x8877665544332211u64.to_le_bytes()
            ]
            .concat()
        );
    }

    #[test]
    fn refuses_explicit_object_count_capability_and_arena_bounds_before_allocation() {
        let metal = MetalResidentSync::new().unwrap();
        let base = |objects, object_capabilities, max_frame_bytes| ResidentSyncConfig {
            max_epochs: 1,
            max_effects_per_step: 1,
            max_frame_bytes,
            max_continuations: 1,
            cohort_width: 1,
            initial_epoch: 0,
            objects,
            object_capabilities,
            futures: vec![],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![],
        };
        let assert_refused = |config: ResidentSyncConfig| {
            let programs = BTreeMap::new();
            assert!(run_resident_sync(&config, vec![], &programs).is_none());
            assert!(matches!(
                metal.run(&config, vec![], &programs),
                Err(BackendError::InvalidInput)
            ));
        };
        assert_refused(base(
            vec![
                ResidentObject {
                    version: 1,
                    bytes: vec![]
                };
                MAX_RESIDENT_OBJECTS + 1
            ],
            vec![],
            1,
        ));
        assert_refused(base(
            vec![ResidentObject {
                version: 1,
                bytes: vec![0],
            }],
            vec![
                ResidentObjectCapability {
                    actor: 1,
                    target: 0,
                    offset: 0,
                    length: 1,
                    rights: RIGHT_READ,
                    object_version: 1,
                    valid_until_epoch: 1,
                };
                MAX_RESIDENT_OBJECT_CAPABILITIES + 1
            ],
            1,
        ));
        let half = MAX_RESIDENT_OBJECT_ARENA_BYTES / 2 + 1;
        assert_refused(base(
            vec![
                ResidentObject {
                    version: 1,
                    bytes: vec![0; half],
                },
                ResidentObject {
                    version: 1,
                    bytes: vec![0; half],
                },
            ],
            vec![],
            half as u32,
        ));
    }
    #[test]
    fn same_step_await_then_resolve_publishes_park_before_self_wake() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 2,
            max_effects_per_step: 2,
            max_frame_bytes: 8,
            max_continuations: 1,
            cohort_width: 32,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![cap(7, RESOURCE_FUTURE, 0, RIGHT_READ | RIGHT_WRITE)],
        };
        let continuations = vec![ResidentContinuation {
            id: 11,
            actor: 7,
            run_class: 1,
            frame: vec![],
        }];
        let programs = BTreeMap::from([(
            1,
            ResidentHandlerProgram {
                run_class: 1,
                instructions: vec![
                    instruction(HANDLER_EFFECT_FUTURE_AWAIT, 0, 0, 0),
                    instruction(HANDLER_EFFECT_FUTURE_RESOLVE, 0, 0, 73),
                    instruction(HANDLER_COMPLETE, 0, 0, 0),
                ],
            },
        )]);
        let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
        let gpu = metal.run(&config, continuations, &programs).unwrap();
        assert_eq!(gpu, cpu);
        assert_eq!(
            gpu.effects
                .iter()
                .map(|record| record.outcome)
                .collect::<Vec<_>>(),
            vec![
                ResidentOutcome::Registered,
                ResidentOutcome::Resolved(73),
                ResidentOutcome::Resolved(73),
                ResidentOutcome::DoubleResolve,
            ]
        );
        assert_eq!(
            gpu.wakes
                .iter()
                .map(|wake| (wake.continuation, wake.ticket, wake.ordinal))
                .collect::<Vec<_>>(),
            vec![(11, 1, 1)]
        );
        assert_eq!(gpu.completed, vec![11]);
    }

    #[test]
    fn later_lane_resolve_wakes_waiters_in_ticket_then_id_order() {
        let metal = MetalResidentSync::new().unwrap();
        let config = ResidentSyncConfig {
            max_epochs: 2,
            max_effects_per_step: 1,
            max_frame_bytes: 8,
            max_continuations: 3,
            cohort_width: 32,
            initial_epoch: 0,
            objects: vec![],
            object_capabilities: vec![],
            futures: vec![InitialFuture::Pending],
            mailbox_capacities: vec![],
            mailbox_messages: vec![],
            capabilities: vec![
                cap(1, RESOURCE_FUTURE, 0, RIGHT_READ),
                cap(2, RESOURCE_FUTURE, 0, RIGHT_READ),
                cap(3, RESOURCE_FUTURE, 0, RIGHT_WRITE),
            ],
        };
        // Identity order is 1, 9, but class order registers 9 before 1.
        let continuations = vec![
            ResidentContinuation {
                id: 1,
                actor: 1,
                run_class: 200,
                frame: vec![],
            },
            ResidentContinuation {
                id: 9,
                actor: 2,
                run_class: 100,
                frame: vec![],
            },
            ResidentContinuation {
                id: 5,
                actor: 3,
                run_class: 50,
                frame: vec![],
            },
        ];
        let mut programs = BTreeMap::new();
        programs.insert(
            50,
            ResidentHandlerProgram {
                run_class: 50,
                instructions: vec![instruction(HANDLER_YIELD, 300, 0, 0)],
            },
        );
        for run_class in [100, 200] {
            programs.insert(
                run_class,
                ResidentHandlerProgram {
                    run_class,
                    instructions: vec![
                        instruction(HANDLER_EFFECT_FUTURE_AWAIT, 0, 0, 0),
                        instruction(HANDLER_YIELD, run_class, 0, 0),
                    ],
                },
            );
        }
        programs.insert(
            300,
            ResidentHandlerProgram {
                run_class: 300,
                instructions: vec![
                    instruction(HANDLER_EFFECT_FUTURE_RESOLVE, 0, 0, 88),
                    instruction(HANDLER_COMPLETE, 0, 0, 0),
                ],
            },
        );
        let cpu = run_resident_sync(&config, continuations.clone(), &programs).unwrap();
        let gpu = metal.run(&config, continuations, &programs).unwrap();
        assert_eq!(gpu, cpu);
        assert_eq!(
            gpu.wakes
                .iter()
                .map(|wake| (wake.continuation, wake.ticket))
                .collect::<Vec<_>>(),
            vec![(9, 1), (1, 2)]
        );
        assert_eq!(gpu.wakes[0].cause_continuation, 5);
        assert_eq!(gpu.wakes[0].epoch, 1);
    }

    #[test]
    fn full_epoch_parallel_evaluation_is_physical_and_compacts_irregular_effects_exactly() {
        let metal = MetalResidentSync::new().unwrap();
        let make_input = |width| {
            let config = ResidentSyncConfig {
                max_epochs: 4,
                max_effects_per_step: 2,
                max_frame_bytes: 16,
                max_continuations: 256,
                cohort_width: width,
                initial_epoch: 4,
                objects: vec![ResidentObject {
                    version: 2,
                    bytes: 0x8877665544332211u64.to_le_bytes().to_vec(),
                }],
                object_capabilities: vec![ResidentObjectCapability {
                    actor: 1,
                    target: 0,
                    offset: 0,
                    length: 8,
                    rights: RIGHT_READ,
                    object_version: 2,
                    valid_until_epoch: 8,
                }],
                futures: vec![InitialFuture::Resolved(19), InitialFuture::Pending],
                mailbox_capacities: vec![5],
                mailbox_messages: vec![vec![]],
                capabilities: vec![
                    cap(1, RESOURCE_MAILBOX, 0, RIGHT_WRITE),
                    cap(1, RESOURCE_FUTURE, 1, RIGHT_READ),
                    cap(2, RESOURCE_FUTURE, 1, RIGHT_READ),
                ],
            };
            let continuations = (1..=256u64)
                .rev()
                .map(|id| ResidentContinuation {
                    id,
                    actor: if id % 3 == 0 { 1 } else { 2 },
                    run_class: if id % 2 == 0 { 10 } else { 20 },
                    frame: vec![id as u8; (id as usize % 13) + 1],
                })
                .collect::<Vec<_>>();
            let programs = BTreeMap::from([
                (
                    10,
                    ResidentHandlerProgram {
                        run_class: 10,
                        instructions: vec![
                            instruction(HANDLER_EFFECT_MAILBOX_SEND, 0, 0, 100),
                            instruction(HANDLER_YIELD, 20, 0, 0),
                        ],
                    },
                ),
                (
                    20,
                    ResidentHandlerProgram {
                        run_class: 20,
                        instructions: vec![
                            instruction(HANDLER_EFFECT_OBJECT_READ, 0, 0, 0),
                            instruction(HANDLER_EFFECT_FUTURE_AWAIT, 1, 0, 0),
                            instruction(HANDLER_COMPLETE, 0, 0, 0),
                        ],
                    },
                ),
            ]);
            (config, continuations, programs)
        };

        let (config1, continuations1, programs1) = make_input(1);
        let cpu1 = run_resident_sync(&config1, continuations1.clone(), &programs1).unwrap();
        let (gpu1, counts1) = metal
            .run_with_evaluation_counts(&config1, continuations1, &programs1)
            .unwrap();
        assert_eq!(gpu1, cpu1);
        assert_eq!(counts1.len(), 1);
        assert!(counts1[0] > 256);
        assert_eq!(counts1.iter().sum::<u32>() as usize, gpu1.invocations.len());

        let (config32, continuations32, programs32) = make_input(32);
        let cpu32 = run_resident_sync(&config32, continuations32.clone(), &programs32).unwrap();
        let (gpu32, counts32) = metal
            .run_with_evaluation_counts(&config32, continuations32, &programs32)
            .unwrap();
        assert_eq!(gpu32, cpu32);
        assert_eq!(gpu32, gpu1);
        assert_eq!(counts32.iter().filter(|&&count| count != 0).count(), 32);
        assert!(counts32.iter().all(|&count| count > 0));
        assert_eq!(
            counts32.iter().sum::<u32>() as usize,
            gpu32.invocations.len()
        );
        assert!(gpu32
            .effects
            .iter()
            .any(|record| record.outcome == ResidentOutcome::Full));
        assert!(gpu32
            .effects
            .iter()
            .any(|record| record.outcome == ResidentOutcome::CapabilityDenied));
        assert!(gpu32
            .effects
            .iter()
            .any(|record| { record.outcome == ResidentOutcome::ObjectRead(0x8877665544332211) }));
        assert!(gpu32
            .final_continuations
            .iter()
            .any(|continuation| continuation.pending.is_some()));
        assert!(gpu32
            .invocations
            .windows(2)
            .all(|pair| pair[0].epoch <= pair[1].epoch));
    }
}
