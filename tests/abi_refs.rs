//! Step 1 verification: ABI references and the generational table (§4).

use soma::abi::{AbiError, Kind, Ref64};
use soma::table::GenTable;

#[test]
fn alloc_returns_valid_ref_and_get_reads_value() {
    let mut t = GenTable::new(Kind::Process);
    let r = t.alloc(42u32);
    assert_eq!(r.kind, Kind::Process);
    assert!(!r.is_null());
    assert_eq!(*t.get(r).unwrap(), 42);
    assert_eq!(t.len(), 1);
}

#[test]
fn get_rejects_wrong_kind() {
    let mut t = GenTable::new(Kind::Process);
    let r = t.alloc(1u32);
    let wrong = Ref64::new(r.slot, r.generation, Kind::Object);
    assert_eq!(t.get(wrong), Err(AbiError::KindMismatch));
}

#[test]
fn get_rejects_bad_slot() {
    let t: GenTable<u32> = GenTable::new(Kind::Process);
    let r = Ref64::new(999, 0, Kind::Process);
    assert_eq!(t.get(r), Err(AbiError::BadSlot));
}

#[test]
fn stale_reference_rejected_after_delete() {
    let mut t = GenTable::new(Kind::Process);
    let r = t.alloc(7u32);
    assert_eq!(t.delete(r), Ok(7));
    // Deleted: value gone, generation bumped.
    assert_eq!(t.get(r), Err(AbiError::StaleReference));
    assert_eq!(t.len(), 0);
}

#[test]
fn slot_reuse_bumps_generation() {
    let mut t = GenTable::new(Kind::Process);
    let a = t.alloc(1u32);
    let slot = a.slot;
    let gen_a = a.generation;
    assert_eq!(t.delete(a), Ok(1));

    // Reused slot gets a higher generation; the old ref is now stale.
    let b = t.alloc(2u32);
    assert_eq!(b.slot, slot);
    assert!(b.generation > gen_a || b.generation != 0);
    assert_eq!(t.get(b), Ok(&2));
    assert_eq!(t.get(a), Err(AbiError::StaleReference));
}

#[test]
fn delete_with_stale_ref_is_noop() {
    let mut t = GenTable::new(Kind::Process);
    let _a = t.alloc(5u32);
    let b = t.alloc(6u32);
    // Trying to delete with a stale generation fails, leaving b live.
    let bad = Ref64::new(b.slot, 0xFFFF, Kind::Process);
    assert!(t.delete(bad).is_err());
    assert_eq!(*t.get(b).unwrap(), 6);
}

#[test]
fn null_ref_is_invalid() {
    let t: GenTable<u32> = GenTable::new(Kind::Process);
    // A reference to slot 0 (the reserved null slot) is always invalid.
    let null_process = Ref64::new(0, 0, Kind::Process);
    assert_eq!(t.get(null_process), Err(AbiError::BadSlot));
    assert!(null_process.is_null());
}

#[test]
fn ref64_to_u64_roundtrips() {
    let r = Ref64::new(0xDEAD_BEEF, 0x1234, Kind::Future);
    assert_eq!(Ref64::from_u64(r.to_u64()), r);
    assert_eq!(Ref64::from_u64(Ref64::NULL.to_u64()), Ref64::NULL);
    assert!(Ref64::from_u64(0).is_null());
}

#[test]
fn kind_u8_roundtrips() {
    for k in [
        Kind::Domain,
        Kind::Process,
        Kind::Object,
        Kind::Capability,
        Kind::Continuation,
        Kind::Channel,
        Kind::Future,
        Kind::Contract,
        Kind::Collective,
        Kind::Module,
    ] {
        assert_eq!(Kind::from_u8(k.as_u8()), Some(k));
    }
    assert_eq!(Kind::from_u8(0), None);
    assert_eq!(Kind::from_u8(255), None);
}

#[test]
fn table_iterates_live_entries_in_order() {
    let mut t = GenTable::new(Kind::Object);
    let r1 = t.alloc("a");
    let r2 = t.alloc("b");
    let entries: Vec<(&str, u32)> = t.iter().map(|(r, v)| (*v, r.slot)).collect();
    assert_eq!(entries, vec![("a", r1.slot), ("b", r2.slot)]);
}

// ---- generation exhaustion -----------------------------------------------

#[test]
fn a_slot_is_retired_rather_than_wrapping_its_generation() {
    // The ABA window `abi/refs.rs` used to document as unsolved: recycle one
    // slot 65,536 times and its generation wraps, so a reference held across
    // that churn matches again and silently addresses a different entity.
    //
    // Retiring the slot instead makes staleness detection guaranteed at every
    // generation width. A distributed implementation persists references
    // across a network, where "held for a long time" is the normal case.
    let mut table: GenTable<u32> = GenTable::new(Kind::Object);
    let first = table.alloc(1);
    assert_eq!(first.generation, 0);
    assert_eq!(table.retired_slots(), 0);

    // Churn the same slot until its generation is exhausted. Each iteration
    // reuses the freed slot, so the table never grows.
    let mut last = first;
    for _ in 0..u16::MAX {
        table.delete(last).unwrap();
        last = table.alloc(2);
    }
    assert_eq!(last.slot, first.slot, "the churn should reuse one slot");
    assert_eq!(last.generation, u16::MAX);

    // The final delete cannot bump the generation without wrapping, so the
    // slot is withdrawn instead of returned to the free list.
    table.delete(last).unwrap();
    assert_eq!(table.retired_slots(), 1);

    // The retired slot is never handed out again, and the stale reference that
    // would have collided with it does not resolve.
    let fresh = table.alloc(3);
    assert_ne!(fresh.slot, first.slot, "a retired slot was recycled");
    assert!(table.get(first).is_err());
    assert!(table.get(last).is_err());
}

#[test]
fn ordinary_churn_still_recycles_slots() {
    // The control. If retirement fired on every delete, the table would leak a
    // slot per entity and the test above would pass for the wrong reason.
    let mut table: GenTable<u32> = GenTable::new(Kind::Object);
    let first = table.alloc(1);
    for _ in 0..1000 {
        let current = table.alloc(2);
        table.delete(current).unwrap();
    }
    assert_eq!(table.retired_slots(), 0);
    assert_eq!(table.len(), 1);
    assert!(table.get(first).is_ok());
}
