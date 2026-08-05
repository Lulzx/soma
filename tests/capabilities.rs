use soma::abi::{ObjectKind, ProcessMode, ProcessState, Ref64, Rights};
use soma::compiler::frame::Frame;
use soma::compiler::run_classes::{DEFAULT_MAX_STEPS, SEARCH_BRANCH};
use soma::compiler::state_machine_lowering::SearchFrame;
use soma::kernel::{Kernel, RuntimeError, SYSTEM_PRINCIPAL};
use soma::kernel::raw;
use soma::kernel::ownership::freeze;
use soma::semantics::invariants::assert_legal;

fn leaf(kernel: &mut Kernel, process: Ref64) -> Ref64 {
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);
    kernel
        .create_continuation(
            process,
            process,
            SEARCH_BRANCH,
            0,
            bytes,
            DEFAULT_MAX_STEPS,
        )
        .unwrap()
}

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

#[test]
fn messaging_and_future_operations_require_their_specific_rights() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let owner_cont = leaf(&mut kernel, owner);
    let receiver_cont = leaf(&mut kernel, receiver);
    let future = kernel.create_future(owner);
    let payload = kernel.create_object(owner, ObjectKind::MessagePayload, vec![7]);

    assert!(matches!(
        kernel.await_future(stranger, owner_cont, future, SEARCH_BRANCH),
        Err(RuntimeError::AuthorityDenied)
    ));
    assert!(matches!(
        kernel.resolve_future(stranger, future, payload),
        Err(RuntimeError::AuthorityDenied)
    ));
    assert!(matches!(
        kernel.receive_message(stranger, receiver_cont),
        Err(RuntimeError::AuthorityDenied)
    ));
    assert!(matches!(
        kernel.enqueue_message(owner, receiver, payload, owner_cont),
        Err(RuntimeError::AuthorityDenied)
    ));

    kernel
        .grant_capability(
            SYSTEM_PRINCIPAL,
            owner,
            receiver,
            Rights::SEND,
            0,
            0,
        )
        .unwrap();
    kernel
        .enqueue_message(owner, receiver, payload, owner_cont)
        .unwrap();
    let message = kernel
        .receive_message(receiver, receiver_cont)
        .unwrap()
        .unwrap();
    assert!(!message.transferred_capability.is_null());
    assert_eq!(kernel.object_bytes(receiver, message.payload).unwrap(), &[7]);
}

#[test]
fn creating_execution_requires_write_on_the_target_process() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let mut bytes = Vec::new();
    SearchFrame::leaf(1, 0).encode(&mut bytes);

    assert!(matches!(
        kernel.create_continuation(
            stranger,
            owner,
            SEARCH_BRANCH,
            0,
            bytes,
            DEFAULT_MAX_STEPS,
        ),
        Err(RuntimeError::AuthorityDenied)
    ));
}

#[test]
fn await_authority_is_rechecked_when_a_continuation_resumes() {
    let mut kernel = Kernel::new();
    let process = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let continuation = leaf(&mut kernel, process);
    let future = kernel.create_future(process);
    kernel
        .await_future(process, continuation, future, SEARCH_BRANCH)
        .unwrap();

    let capability = kernel
        .find_capability(process, future, Rights::AWAIT)
        .unwrap();
    unsafe { raw::state(&mut kernel) }
        .capability_spaces
        .get_mut(&process.slot)
        .unwrap()
        .get_mut(capability)
        .unwrap()
        .valid_until_epoch = 0;

    // Drain the parked continuation's original bin and advance beyond expiry.
    kernel.run_epoch();
    let value = kernel.create_object(SYSTEM_PRINCIPAL, ObjectKind::FutureValue, vec![0; 8]);
    kernel
        .resolve_future(SYSTEM_PRINCIPAL, future, value)
        .unwrap();
    kernel.run_epoch();

    assert_eq!(kernel.process_state(process).unwrap(), ProcessState::Failed);
}

#[test]
fn grants_require_transfer_authority_and_must_attenuate() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let stranger = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let object = kernel.create_object(owner, ObjectKind::RawBytes, vec![0; 4]);

    assert_eq!(
        kernel.grant_capability(stranger, receiver, object, Rights::READ, 0, 4),
        Err(RuntimeError::AuthorityDenied)
    );
    assert_eq!(
        kernel.grant_capability(owner, receiver, object, Rights::READ, 0, 5),
        Err(RuntimeError::InvalidCapabilityDerivation)
    );

    let granted = kernel
        .grant_capability(owner, receiver, object, Rights::READ, 1, 2)
        .unwrap();
    let entry = kernel.capability_entry(receiver, granted).unwrap();
    assert_eq!(entry.rights, Rights::READ);
    assert_eq!((entry.offset, entry.length), (1, 2));
}

#[test]
fn external_ingest_is_reserved_for_the_system_principal() {
    let mut kernel = Kernel::new();
    let sender = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let receiver = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let payload = kernel.create_object(sender, ObjectKind::MessagePayload, vec![1]);

    assert_eq!(
        kernel.ingest_message(sender, sender, receiver, payload, Ref64::NULL),
        Err(RuntimeError::AuthorityDenied)
    );
    assert!(kernel.mailbox_entries(receiver).unwrap().is_empty());
}

#[test]
fn freezing_replaces_mutable_authority_with_read_authority_for_the_new_version() {
    let mut kernel = Kernel::new();
    let owner = kernel.create_process(SYSTEM_PRINCIPAL, ProcessMode::Serial);
    let object = kernel.create_object(owner, ObjectKind::RawBytes, vec![1, 2]);
    let old_write = kernel.find_capability(owner, object, Rights::WRITE).unwrap();
    let old_version = kernel.capability_entry(owner, old_write).unwrap().object_version;

    let new_version = freeze(&mut kernel, owner, object).unwrap();

    assert_eq!(new_version, old_version + 1);
    assert_eq!(kernel.object_bytes(owner, object).unwrap(), &[1, 2]);
    assert_eq!(
        kernel.object_bytes_mut(owner, object),
        Err(RuntimeError::AuthorityDenied)
    );
    let has_current_read = unsafe { raw::state(&mut kernel) }
        .capability_spaces
        .get(&owner.slot)
        .unwrap()
        .iter()
        .any(|(_, entry)| {
            entry.target == object
                && entry.rights & Rights::READ == Rights::READ
                && entry.object_version == new_version
        });
    assert!(has_current_read);
}
