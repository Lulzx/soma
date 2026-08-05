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

/// A generational table of `T` objects, addressed by `Ref64`.
///
/// Each table is typed to a single `Kind`; `Ref64` dereferences against a table
/// only when the reference's `kind` matches the table's kind. This makes
/// kind-mismatch structural rather than a runtime string comparison.
#[derive(Debug)]
pub struct GenTable<T> {
    kind: Kind,
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    /// Occupied slot count, maintained incrementally so `len` is O(1) — it is
    /// called from per-epoch accounting paths.
    live: usize,
}

impl<T> GenTable<T> {
    /// Create an empty table for the given kind. Slot 0 is reserved and never
    /// returned, so `Ref64::NULL` (slot 0) is always an invalid reference.
    pub fn new(kind: Kind) -> GenTable<T> {
        GenTable {
            kind,
            // Index 0 = reserved null slot, generation 0, unoccupied.
            slots: vec![Slot {
                generation: 0,
                value: None,
            }],
            free: Vec::new(),
            live: 0,
        }
    }

    /// Insert a value, returning a reference whose generation is the slot's
    /// current generation and whose kind matches the table.
    pub fn alloc(&mut self, value: T) -> Ref64 {
        let slot;
        let generation;
        if let Some(idx) = self.free.pop() {
            slot = idx;
            generation = self.slots[idx as usize].generation;
            self.slots[idx as usize].value = Some(value);
        } else {
            slot = self.slots.len() as u32;
            generation = 0;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
        }
        self.live += 1;
        Ref64::new(slot, generation, self.kind)
    }

    /// Look up a value, enforcing §4 validity: slot exists, kind matches the
    /// table, generation equals the current slot generation.
    pub fn get(&self, r: Ref64) -> Result<&T, AbiError> {
        if r.kind != self.kind {
            return Err(AbiError::KindMismatch);
        }
        let slot = self.slots.get(r.slot as usize).ok_or(AbiError::BadSlot)?;
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
            .slots
            .get_mut(r.slot as usize)
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
        let slot = self
            .slots
            .get_mut(r.slot as usize)
            .ok_or(AbiError::BadSlot)?;
        if slot.generation != r.generation {
            return Err(AbiError::StaleReference);
        }
        let value = slot.value.take().ok_or(AbiError::BadSlot)?;
        // Increment generation before reuse; skip 0 so a reused slot is never
        // addressable with a zero generation from a previous occupant.
        slot.generation = slot.generation.wrapping_add(1);
        if slot.generation == 0 {
            slot.generation = 1;
        }
        self.free.push(r.slot);
        self.live -= 1;
        Ok(value)
    }

    /// Number of live entries (occupied slots).
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over live (ref, &value) pairs in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Ref64, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.value.as_ref().map(|v| (Ref64::new(i as u32, s.generation, self.kind), v))
            })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }
}
