//! Ownership transitions (§6): unique-mutable and frozen-shared objects.
//!
//! Phase 1 implements a light version sufficient to demonstrate the object
//! model: freezing an object bumps its version and moves it to immutable state;
//! a frozen object cannot return to mutable state (mutation requires allocating
//! a new object, §6.2). Transfer of a unique object's authority is tracked here.

use crate::abi::objects::OwnershipState;
use crate::abi::{Kind, Ref64};
use crate::kernel::Kernel;
use crate::kernel::RuntimeError;

/// Freeze a unique-mutable object into the frozen-shared state (§6.2):
/// 1. complete all prior writes (no-op here: single-threaded),
/// 2. increment the version,
/// 3. transition to immutable state,
/// 4. publish to readers (increment reader count).
pub fn freeze(kernel: &mut Kernel, actor: Ref64, object: Ref64) -> Result<u32, RuntimeError> {
    let _ = kernel.objects.get(object)?;
    let (version, byte_length, ownership) = {
        let o = kernel.objects.get(object)?;
        (o.version, o.byte_length, o.ownership_state)
    };
    if ownership == OwnershipState::FrozenShared {
        kernel.authorize(actor, crate::abi::Rights::READ, object)?;
        return Ok(version);
    }
    kernel.authorize(actor, crate::abi::Rights::FREEZE, object)?;
    kernel.authority_effect(actor, crate::abi::Rights::FREEZE, object);
    let new_version = version.wrapping_add(1);
    {
        let o = kernel.objects.get_mut(object)?;
        o.version = new_version;
        o.ownership_state = OwnershipState::FrozenShared;
        o.reader_count = 1;
    }
    let _ = kernel.mint_object_read(actor, object, byte_length, new_version);
    Ok(new_version)
}

/// Transfer the unique authority over `object` from one owner to another.
/// Returns `Err` if the object is frozen (not transferable) or a stale ref.
pub fn transfer_unique(
    kernel: &mut Kernel,
    actor: Ref64,
    object: Ref64,
    new_owner: Ref64,
) -> Result<(), RuntimeError> {
    kernel.authorize(actor, crate::abi::Rights::TRANSFER, object)?;
    let _ = kernel.objects.get(object)?;
    if kernel.objects.get(object)?.ownership_state == OwnershipState::FrozenShared {
        return Err(RuntimeError::Abi(crate::abi::AbiError::NoAuthority));
    }
    let _ = kernel.processes.get(new_owner)?;
    kernel.authority_effect(actor, crate::abi::Rights::TRANSFER, object);
    kernel.move_target_authority(actor, new_owner, object)?;
    kernel.objects.get_mut(object)?.unique_owner = new_owner;
    Ok(())
}

/// Assert that `r` is a live reference of the given kind (used by tests and
/// validation phases). Mirrors the §4 validity predicate.
pub fn assert_live(kernel: &Kernel, r: Ref64, expected: Kind) -> Result<(), RuntimeError> {
    if r.kind != expected {
        return Err(RuntimeError::Abi(crate::abi::AbiError::KindMismatch));
    }
    match expected {
        Kind::Process => kernel.processes.get(r).map(|_| ()),
        Kind::Object => kernel.objects.get(r).map(|_| ()),
        Kind::Continuation => kernel.continuations.get(r).map(|_| ()),
        Kind::Future => kernel.futures.get(r).map(|_| ()),
        _ => Ok(()),
    }
    .map_err(Into::into)
}

/// Read the current ownership state of an object.
pub fn ownership_state(kernel: &Kernel, object: Ref64) -> Result<OwnershipState, RuntimeError> {
    Ok(kernel.objects.get(object)?.ownership_state)
}
