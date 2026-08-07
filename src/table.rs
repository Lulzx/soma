//! Generational slot table (§4).
//!
//! Every kernel-managed entity lives in a typed generational table. Deleting an
//! entity increments its slot generation before reuse, so stale references can
//! never silently target newly allocated entities.

use crate::abi::{AbiError, Kind, Ref64};

/// One occupied-or-free slot.
#[derive(Clone, Debug)]
struct Slot<T> {
    generation: u16,
    value: Option<T>,
}

/// One allocator's slots. Partitions never share a slot space, which is what
/// lets two of them allocate at the same time without agreeing on anything.
#[derive(Clone, Debug)]
struct Partition<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> Partition<T> {
    /// Slot 0 is reserved in every partition, so `Ref64::NULL` is invalid
    /// everywhere rather than only in partition 0.
    fn new() -> Partition<T> {
        Partition {
            slots: vec![Slot {
                generation: 0,
                value: None,
            }],
            free: Vec::new(),
        }
    }
}

/// A generational table of `T` objects, addressed by `Ref64`.
///
/// Each table is typed to a single `Kind`; `Ref64` dereferences against a table
/// only when the reference's `kind` matches the table's kind. This makes
/// kind-mismatch structural rather than a runtime string comparison.
///
/// Slots are partitioned. A reference names its partition, and a partition's
/// slot numbering is its own, so two allocators may both mint slot 7 and mean
/// different entities. That is what a device's lanes and a cluster's nodes
/// need: allocation with no shared counter to contend on.
///
/// Which partition an allocation lands in is a *placement* decision, and
/// nothing about the entity. Two runs of one program that partition
/// differently name their entities differently and behave identically, which
/// is why I18 compares up to a correspondence between names (v0.3 §2.6) rather
/// than comparing references.
#[derive(Clone, Debug)]
pub struct GenTable<T> {
    kind: Kind,
    partitions: Vec<Partition<T>>,
    /// The partition `alloc` mints from.
    active: u8,
    /// Occupied slot count, maintained incrementally so `len` is O(1) — it is
    /// called from per-epoch accounting paths.
    live: usize,
    /// Slots withdrawn from reuse because their generation was exhausted.
    retired: usize,
}

impl<T> GenTable<T> {
    /// Create an empty table for the given kind. Slot 0 is reserved and never
    /// returned, so `Ref64::NULL` (slot 0) is always an invalid reference.
    pub fn new(kind: Kind) -> GenTable<T> {
        GenTable {
            kind,
            partitions: vec![Partition::new()],
            active: 0,
            live: 0,
            retired: 0,
        }
    }

    /// Mint subsequent allocations from `partition`, creating it if needed.
    ///
    /// Deterministic by construction: the caller derives the partition from the
    /// epoch's plan, never from which worker got there first.
    pub fn set_active_partition(&mut self, partition: u8) {
        while self.partitions.len() <= partition as usize {
            self.partitions.push(Partition::new());
        }
        self.active = partition;
    }

    pub fn active_partition(&self) -> u8 {
        self.active
    }

    /// How many partitions have been opened. One means the table allocates
    /// exactly as it did before partitions existed.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    fn partition_of(&self, r: Ref64) -> Option<&Partition<T>> {
        self.partitions.get(r.partition as usize)
    }

    fn partition_of_mut(&mut self, r: Ref64) -> Option<&mut Partition<T>> {
        self.partitions.get_mut(r.partition as usize)
    }

    /// Insert a value, returning a reference whose generation is the slot's
    /// current generation and whose kind matches the table.
    pub fn alloc(&mut self, value: T) -> Ref64 {
        let active = self.active;
        let partition = &mut self.partitions[active as usize];
        let slot;
        let generation;
        if let Some(idx) = partition.free.pop() {
            slot = idx;
            generation = partition.slots[idx as usize].generation;
            partition.slots[idx as usize].value = Some(value);
        } else {
            slot = partition.slots.len() as u32;
            generation = 0;
            partition.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
        }
        self.live += 1;
        Ref64::in_partition(slot, generation, self.kind, active)
    }

    /// Look up a value, enforcing §4 validity: slot exists, kind matches the
    /// table, generation equals the current slot generation.
    pub fn get(&self, r: Ref64) -> Result<&T, AbiError> {
        if r.kind != self.kind {
            return Err(AbiError::KindMismatch);
        }
        let slot = self
            .partition_of(r)
            .and_then(|p| p.slots.get(r.slot as usize))
            .ok_or(AbiError::BadSlot)?;
        if slot.generation != r.generation {
            return Err(AbiError::StaleReference);
        }
        slot.value.as_ref().ok_or(AbiError::BadSlot)
    }

    /// Mutable lookup with the same validity checks.
    pub fn get_mut(&mut self, r: Ref64) -> Result<&mut T, AbiError> {
        if r.kind != self.kind {
            return Err(AbiError::KindMismatch);
        }
        let slot = self
            .partition_of_mut(r)
            .and_then(|p| p.slots.get_mut(r.slot as usize))
            .ok_or(AbiError::BadSlot)?;
        if slot.generation != r.generation {
            return Err(AbiError::StaleReference);
        }
        slot.value.as_mut().ok_or(AbiError::BadSlot)
    }

    /// Delete a value. Increments the slot generation before reuse and returns
    /// the deleted value. If the reference is stale, returns `Err` and leaves
    /// the table unchanged.
    pub fn delete(&mut self, r: Ref64) -> Result<T, AbiError> {
        if r.kind != self.kind {
            return Err(AbiError::KindMismatch);
        }
        let partition = self
            .partitions
            .get_mut(r.partition as usize)
            .ok_or(AbiError::BadSlot)?;
        let slot = partition
            .slots
            .get_mut(r.slot as usize)
            .ok_or(AbiError::BadSlot)?;
        if slot.generation != r.generation {
            return Err(AbiError::StaleReference);
        }
        let value = slot.value.take().ok_or(AbiError::BadSlot)?;
        // Increment the generation before reuse, so a stale reference to the
        // previous occupant no longer resolves.
        //
        // When the generation would wrap, retire the slot instead of recycling
        // it. Wrapping is the ABA window the ABI note in `abi/refs.rs` used to
        // document as unsolved: a reference held across 65,536 recycles of one
        // slot would match again and silently address a different entity.
        // Retiring costs one slot per 65,535 recycles and makes staleness
        // detection guaranteed rather than bounded, which is what a
        // distributed implementation needs — it persists references across a
        // network, where "held for a long time" is the normal case rather than
        // the pathological one.
        match slot.generation.checked_add(1) {
            Some(next) => {
                slot.generation = next;
                partition.free.push(r.slot);
            }
            None => self.retired += 1,
        }
        self.live -= 1;
        Ok(value)
    }

    /// Slots permanently withdrawn because their generation was exhausted.
    ///
    /// Nonzero here is not an error, but it is the signal that a workload
    /// churns one slot hard enough to matter.
    pub fn retired_slots(&self) -> usize {
        self.retired
    }

    /// Number of live entries (occupied slots).
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over live (ref, &value) pairs in slot order.
    /// Iterate over live (ref, &value) pairs, partition-major then slot order.
    /// Deterministic, which matters because invariant checks and ownership
    /// derivation walk this.
    pub fn iter(&self) -> impl Iterator<Item = (Ref64, &T)> {
        let kind = self.kind;
        self.partitions
            .iter()
            .enumerate()
            .flat_map(move |(partition, p)| {
                p.slots.iter().enumerate().filter_map(move |(i, s)| {
                    s.value.as_ref().map(|v| {
                        (
                            Ref64::in_partition(i as u32, s.generation, kind, partition as u8),
                            v,
                        )
                    })
                })
            })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    // ---- lane-local allocation (v0.3 §4.8) -------------------------------

    /// Open an empty shard of `partition` that a lane can allocate into alone.
    ///
    /// This is the mechanism a threaded executive needs and the reason
    /// partitions exist at all. A lane cannot hold `&mut GenTable` while other
    /// lanes are running, but it can hold a shard: an allocator over slot
    /// numbers no one else will mint, because the partition is decided from the
    /// lane's position in the epoch's plan (§4.3) and no two lanes share one.
    ///
    /// Taken by `&self`, so an epoch opens one shard per lane from a single
    /// shared borrow of the table and the lanes then own them independently.
    /// The table itself stays readable throughout, which is what a step needs:
    /// it allocates into its shard and reads everything that existed before the
    /// epoch out of the table.
    ///
    /// A shard does **not** recycle freed slots. Reuse means popping the
    /// partition's free list, and two lanes popping one list is precisely the
    /// coordination partitions exist to remove — so a shard appends, and freed
    /// slots become available again after the merge. The cost is that an
    /// epoch's allocations do not reuse slots freed earlier in that epoch. That
    /// changes which slot numbers a run mints and nothing about what it does,
    /// which is the situation partitioned allocation was already in: I18
    /// compares up to a correspondence between names (§2.6), not by reference.
    pub fn shard(&self, partition: u8) -> PartitionShard<T> {
        let base = self
            .partitions
            .get(partition as usize)
            .map(|p| p.slots.len() as u32)
            .unwrap_or(1);
        PartitionShard {
            kind: self.kind,
            partition,
            base,
            slots: Vec::new(),
        }
    }

    /// Fold a lane's shard back into the table.
    ///
    /// Appending in shard order reproduces exactly the slots the shard minted:
    /// it based its numbering on the partition's length at `shard` time, it is
    /// the only allocator for that partition, and nothing else appends there
    /// while it is out. So a reference the lane handed out — and stored in
    /// opaque frame bytes, which §4.3 (2) says it must be able to do — resolves
    /// after the merge to the entity the lane allocated.
    ///
    /// Merging in plan order is the caller's job, and matters for the same
    /// reason applying effects in plan order does: two shards of one partition
    /// would otherwise interleave by arrival. An epoch gives each lane its own
    /// partition, so in practice each merge touches a different one.
    pub fn merge(&mut self, shard: PartitionShard<T>) {
        debug_assert_eq!(shard.kind, self.kind, "a shard belongs to one table");
        while self.partitions.len() <= shard.partition as usize {
            self.partitions.push(Partition::new());
        }
        let partition = &mut self.partitions[shard.partition as usize];
        debug_assert_eq!(
            partition.slots.len() as u32,
            shard.base,
            "the partition grew while its shard was out"
        );
        for slot in shard.slots {
            if slot.value.is_some() {
                self.live += 1;
            }
            partition.slots.push(slot);
        }
    }
}

/// An allocator over one partition's unused slot numbers, owned by one lane.
///
/// Holds only the slots the lane mints, not the partition's existing ones, so
/// opening a shard leaves the table fully readable. A lane therefore reads
/// pre-epoch state from the table and its own new entities from here, which is
/// what §4.3 (2) requires: a step that creates a future and stores it in its
/// frame has to be able to read it back before commit.
#[derive(Debug)]
pub struct PartitionShard<T> {
    kind: Kind,
    partition: u8,
    /// The partition's slot count when this shard was opened. Slot numbering
    /// continues from here, which is what makes the merge an append.
    base: u32,
    slots: Vec<Slot<T>>,
}

impl<T> PartitionShard<T> {
    pub fn alloc(&mut self, value: T) -> Ref64 {
        let slot = self.base + self.slots.len() as u32;
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        Ref64::in_partition(slot, 0, self.kind, self.partition)
    }

    /// Look up an entity this shard minted, with the same validity checks the
    /// table makes. A reference to anything else is `BadSlot` here — the caller
    /// falls through to the table, which is where everything else lives.
    pub fn get(&self, r: Ref64) -> Result<&T, AbiError> {
        self.slot(r)?.value.as_ref().ok_or(AbiError::BadSlot)
    }

    pub fn get_mut(&mut self, r: Ref64) -> Result<&mut T, AbiError> {
        if r.kind != self.kind || r.partition != self.partition || r.slot < self.base {
            return Err(AbiError::BadSlot);
        }
        let index = (r.slot - self.base) as usize;
        let slot = self.slots.get_mut(index).ok_or(AbiError::BadSlot)?;
        if slot.generation != r.generation {
            return Err(AbiError::StaleReference);
        }
        slot.value.as_mut().ok_or(AbiError::BadSlot)
    }

    /// Whether `r` names something this shard minted, so a caller knows which
    /// of the two places to look without probing both.
    pub fn holds(&self, r: Ref64) -> bool {
        r.kind == self.kind
            && r.partition == self.partition
            && r.slot >= self.base
            && ((r.slot - self.base) as usize) < self.slots.len()
    }

    fn slot(&self, r: Ref64) -> Result<&Slot<T>, AbiError> {
        if r.kind != self.kind {
            return Err(AbiError::KindMismatch);
        }
        if r.partition != self.partition || r.slot < self.base {
            return Err(AbiError::BadSlot);
        }
        let index = (r.slot - self.base) as usize;
        let slot = self.slots.get(index).ok_or(AbiError::BadSlot)?;
        if slot.generation != r.generation {
            return Err(AbiError::StaleReference);
        }
        Ok(slot)
    }

    pub fn partition(&self) -> u8 {
        self.partition
    }

    /// How many entities this lane allocated.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.value.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}
