//! Ownership transitions (§6), derived from capability authority.
//!
//! There is no independent owner field or ownership flag. One process holding
//! live `WRITE` authority means unique-mutable; no writer plus at least one
//! reader means frozen-shared. Freezing destroys mutable authority.

use crate::abi::objects::OwnershipState;
use crate::abi::{Kind, Ref64};
use crate::kernel::Kernel;
use crate::kernel::RuntimeError;

/// Freeze a unique-mutable object into the frozen-shared state (§6.2):
/// 1. complete all prior writes (no-op here: single-threaded),
/// 2. increment the version,
/// 3. revoke every write-bearing capability tree,
/// 4. publish version-pinned read authority.
pub fn freeze(kernel: &mut Kernel, actor: Ref64, object: Ref64) -> Result<u32, RuntimeError> {
    let (version, byte_length) = {
        let o = kernel.objects.get(object)?;
        (o.version, o.byte_length)
    };
    if ownership_state(kernel, object)? == OwnershipState::FrozenShared {
        kernel.authorize(actor, crate::abi::Rights::READ, object)?;
        return Ok(version);
    }
    kernel.authorize(actor, crate::abi::Rights::FREEZE, object)?;
    kernel.authority_effect(actor, crate::abi::Rights::FREEZE, object);
    let new_version = version.wrapping_add(1);
    {
        let o = kernel.objects.get_mut(object)?;
        o.version = new_version;
    }
    kernel.revoke_target_right(object, crate::abi::Rights::WRITE);
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
    if ownership_state(kernel, object)? != OwnershipState::UniqueMutable {
        return Err(RuntimeError::Abi(crate::abi::AbiError::NoAuthority));
    }
    let _ = kernel.processes.get(new_owner)?;
    kernel.move_target_authority(actor, new_owner, object)?;
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
    let _ = kernel.objects.get(object)?;
    let writers = kernel.authority_holder_count(object, crate::abi::Rights::WRITE);
    let readers = kernel.authority_holder_count(object, crate::abi::Rights::READ);
    match (writers, readers) {
        (1, _) => Ok(OwnershipState::UniqueMutable),
        (0, 1..) => Ok(OwnershipState::FrozenShared),
        _ => Err(RuntimeError::Abi(crate::abi::AbiError::NoAuthority)),
    }
}
