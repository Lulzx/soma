use soma::abi::{ObjectKind, Ref64, Rights};
use soma::kernel::{Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::kernel::raw;
use soma::semantics::invariants::assert_legal;

#[test]
fn genesis_mints_full_target_appropriate_rights() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::RawBytes, vec![1, 2, 3, 4]);
    let capability = kernel
        .find_capability(Ref64::NULL, object, Rights::WRITE)
        .expect("creation mints authority in the creator's space");
    let entry = kernel.capability_entry(Ref64::NULL, capability).unwrap();

    assert_eq!(entry.target, object);
    assert_eq!(entry.rights, Rights::for_target(object.kind));
    assert_eq!(entry.offset, 0);
    assert_eq!(entry.length, 4);
    assert_legal(&kernel);
}

#[test]
fn derivation_can_only_reduce_rights_and_range() {
    let mut kernel = Kernel::new();
    let object = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::RawBytes, vec![0; 16]);
    let parent = kernel
        .find_capability(Ref64::NULL, object, Rights::READ | Rights::WRITE)
        .unwrap();
    let child = kernel
        .derive_capability(Ref64::NULL, parent, Rights::READ, 4, 8)
        .unwrap();
    let entry = kernel.capability_entry(Ref64::NULL, child).unwrap();
    assert_eq!(entry.rights, Rights::READ);
    assert_eq!((entry.offset, entry.length), (4, 8));
    assert_eq!(entry.parent_capability, parent);

    assert_eq!(
        kernel.derive_capability(
            Ref64::NULL,
            child,
            Rights::READ | Rights::WRITE,
            4,
            8,
        ),
        Err(RuntimeError::InvalidCapabilityDerivation)
    );
    assert_eq!(
        kernel.derive_capability(Ref64::NULL, child, Rights::READ, 0, 16),
        Err(RuntimeError::InvalidCapabilityDerivation)
    );
    assert_legal(&kernel);
}

#[test]
fn capability_references_are_relative_to_the_process_space() {
    let mut kernel = Kernel::new();
    let first = kernel.create_process(SYSTEM_PRINCIPAL, soma::abi::ProcessMode::Serial);
    let second = kernel.create_process(SYSTEM_PRINCIPAL, soma::abi::ProcessMode::Serial);
    let first_self = kernel
        .find_capability(first, first, Rights::WRITE)
        .unwrap();
    let second_self = kernel
        .find_capability(second, second, Rights::WRITE)
        .unwrap();

    assert_eq!(first_self, second_self, "slots may coincide across spaces");
    assert_ne!(
        kernel.capability_entry(first, first_self).unwrap().target,
        kernel.capability_entry(second, second_self).unwrap().target
    );
    assert_legal(&kernel);
}

#[test]
fn write_requires_live_unexpired_authority_at_use() {
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, soma::abi::ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, soma::abi::ProcessMode::Serial);
    let object = kernel.create_object(actor, ObjectKind::RawBytes, vec![1]);

    kernel.object_bytes_mut(actor, object).unwrap().push(2);
    assert!(matches!(
        kernel.object_bytes_mut(stranger, object),
        Err(RuntimeError::AuthorityDenied)
    ));

    let capability = kernel.find_capability(actor, object, Rights::WRITE).unwrap();
    unsafe { raw::state(&mut kernel) }
        .capability_spaces
        .get_mut(&actor.slot)
        .unwrap()
        .get_mut(capability)
        .unwrap()
        .valid_until_epoch = 0;
    kernel.run_epoch();

    assert!(matches!(
        kernel.object_bytes_mut(actor, object),
        Err(RuntimeError::AuthorityDenied)
    ));
}

#[test]
fn revoking_a_parent_is_observed_when_a_parked_capability_is_reused() {
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, soma::abi::ProcessMode::Serial);
    let object = kernel.create_object(actor, ObjectKind::RawBytes, vec![0; 8]);
    let parent = kernel.find_capability(actor, object, Rights::WRITE).unwrap();
    let child = kernel
        .derive_capability(actor, parent, Rights::WRITE, 0, 8)
        .unwrap();
    kernel
        .derive_capability(actor, child, Rights::WRITE, 0, 8)
        .unwrap();

    unsafe { raw::state(&mut kernel) }
        .capability_spaces
        .get_mut(&actor.slot)
        .unwrap()
        .delete(parent)
        .unwrap();

    assert!(matches!(
        kernel.object_bytes_mut(actor, object),
        Err(RuntimeError::AuthorityDenied)
    ));
}

#[test]
fn a_subrange_capability_cannot_open_whole_object_mutation() {
    let mut kernel = Kernel::new();
    let actor = kernel.create_process(SYSTEM_PRINCIPAL, soma::abi::ProcessMode::Serial);
    let object = kernel.create_object(actor, ObjectKind::RawBytes, vec![0; 8]);
    let parent = kernel.find_capability(actor, object, Rights::WRITE).unwrap();
    let child = kernel
        .derive_capability(actor, parent, Rights::WRITE, 2, 2)
        .unwrap();

    let state = unsafe { raw::state(&mut kernel) };
    let space = state.capability_spaces.get_mut(&actor.slot).unwrap();
    space.delete(parent).unwrap();
    space.get_mut(child).unwrap().parent_capability = Ref64::NULL;

    assert!(matches!(
        kernel.object_bytes_mut(actor, object),
        Err(RuntimeError::AuthorityDenied)
    ));
}
