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
#[derive(Debug)]
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
#[derive(Debug)]
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
}
