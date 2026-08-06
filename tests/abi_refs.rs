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

// ---- lane-local allocation (v0.3 §4.8) ------------------------------------

/// A shard's references resolve after the merge to the entities the lane
/// allocated.
///
/// This is the property §4.3 (2) needs and the reason allocation can stay
/// eager under a concurrent executive: a step creates an entity, stores its
/// `Ref64` in opaque frame bytes, and the reference has to still mean that
/// entity once the epoch commits. A symbolic name resolved later could not
/// survive the byte blob; a partitioned slot number can.
#[test]
fn a_shards_references_survive_the_merge() {
    let mut table: GenTable<u32> = GenTable::new(Kind::Process);
    table.set_active_partition(0);
    let existing = table.alloc(100);

    let mut shard = table.shard(1);
    let a = shard.alloc(7);
    let b = shard.alloc(8);

    // Readable from the shard before the merge, which is what lets a step use
    // what it just allocated.
    assert_eq!(shard.get(a), Ok(&7));
    assert_eq!(shard.get(b), Ok(&8));
    assert!(shard.holds(a) && shard.holds(b));
    assert!(!shard.holds(existing));

    table.merge(shard);

    assert_eq!(table.get(a), Ok(&7));
    assert_eq!(table.get(b), Ok(&8));
    assert_eq!(table.get(existing), Ok(&100), "the merge disturbed the table");
    assert_eq!(table.len(), 3);
}

#[test]
fn two_lanes_allocate_into_their_own_partitions_at_the_same_time() {
    // The whole point. Two shards are two independent allocators over disjoint
    // slot spaces, so they can be filled from two threads with nothing shared
    // and no coordination — which is what `docs/SOMA-v0.3.md` §4.3 says
    // partitioned allocation is for.
    let table: GenTable<u64> = GenTable::new(Kind::Process);
    let mut first = table.shard(1);
    let mut second = table.shard(2);

    let (a, b) = std::thread::scope(|scope| {
        let one = scope.spawn(|| (0..64u64).map(|v| first.alloc(v)).collect::<Vec<_>>());
        let two = scope.spawn(|| (0..64u64).map(|v| second.alloc(v + 1000)).collect::<Vec<_>>());
        (one.join().unwrap(), two.join().unwrap())
    });

    // Both minted slot numbers from the same range and meant different
    // entities, which is exactly what a partition is.
    assert_eq!(a[0].slot, b[0].slot);
    assert_ne!(a[0].partition, b[0].partition);

    let mut table = table;
    table.merge(first);
    table.merge(second);

    for (index, r) in a.iter().enumerate() {
        assert_eq!(table.get(*r), Ok(&(index as u64)));
    }
    for (index, r) in b.iter().enumerate() {
        assert_eq!(table.get(*r), Ok(&(index as u64 + 1000)));
    }
    assert_eq!(table.len(), 128);
}

#[test]
fn merging_a_shard_matches_allocating_inline() {
    // The shard is an optimisation of where allocation happens, not a change to
    // what it produces. Against a partition with no freed slots the two are
    // identical, references included.
    let mut inline: GenTable<u32> = GenTable::new(Kind::Object);
    inline.set_active_partition(3);
    let direct: Vec<_> = (0..5u32).map(|v| inline.alloc(v)).collect();

    let mut sharded: GenTable<u32> = GenTable::new(Kind::Object);
    sharded.set_active_partition(3);
    let mut shard = sharded.shard(3);
    let staged: Vec<_> = (0..5u32).map(|v| shard.alloc(v)).collect();
    sharded.merge(shard);

    assert_eq!(direct, staged, "a shard minted different references");
    assert_eq!(inline.len(), sharded.len());
    for r in &direct {
        assert_eq!(inline.get(*r), sharded.get(*r));
    }
}

#[test]
fn a_shard_appends_rather_than_recycling_a_freed_slot() {
    // The one behavioural difference, tested rather than left in a comment.
    // Reusing a slot means popping the partition's free list, and two lanes
    // popping one list is the coordination partitions exist to remove. So the
    // shard appends, the freed slot stays free, and it becomes available again
    // after the merge.
    let mut table: GenTable<u32> = GenTable::new(Kind::Object);
    table.set_active_partition(0);
    let first = table.alloc(1);
    let second = table.alloc(2);
    table.delete(first).unwrap();

    // Inline, the next allocation would recycle `first`'s slot.
    let mut recycling: GenTable<u32> = GenTable::new(Kind::Object);
    let r1 = recycling.alloc(1);
    let _ = recycling.alloc(2);
    recycling.delete(r1).unwrap();
    assert_eq!(recycling.alloc(3).slot, r1.slot, "inline allocation recycles");

    let mut shard = table.shard(0);
    let third = shard.alloc(3);
    assert_ne!(third.slot, first.slot, "the shard recycled a freed slot");
    table.merge(shard);

    assert_eq!(table.get(third), Ok(&3));
    assert_eq!(table.get(second), Ok(&2));
    assert_eq!(
        table.get(first),
        Err(soma::abi::AbiError::StaleReference),
        "the deleted reference came back to life"
    );
    // And the freed slot is reusable again now the shard is folded in.
    assert_eq!(table.alloc(4).slot, first.slot);
}

#[test]
fn a_shard_refuses_a_reference_it_did_not_mint() {
    // A lane looks in its shard first and falls through to the table. That only
    // works if the shard is honest about what it holds, rather than resolving a
    // neighbouring partition's slot number to its own entity.
    let mut table: GenTable<u32> = GenTable::new(Kind::Process);
    table.set_active_partition(0);
    let older = table.alloc(1);

    let mut shard = table.shard(1);
    let mine = shard.alloc(2);

    assert_eq!(shard.get(older), Err(soma::abi::AbiError::BadSlot));
    assert!(!shard.holds(older));
    assert_eq!(shard.get(mine), Ok(&2));

    // A same-partition slot below the base belongs to the table, not the shard.
    let mut zero_shard = table.shard(0);
    let _ = zero_shard.alloc(9);
    assert_eq!(zero_shard.get(older), Err(soma::abi::AbiError::BadSlot));
    assert!(!zero_shard.holds(older));
}

#[test]
fn a_shard_of_the_wrong_kind_does_not_resolve() {
    // Kind mismatch is structural everywhere else in the ABI and stays so here.
    let table: GenTable<u32> = GenTable::new(Kind::Process);
    let mut shard = table.shard(0);
    let r = shard.alloc(1);
    let wrong = Ref64::in_partition(r.slot, r.generation, Kind::Object, r.partition);
    assert_eq!(shard.get(wrong), Err(soma::abi::AbiError::KindMismatch));
}
