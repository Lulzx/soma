//! One actor's capabilities, indexed by what they authorize.
//!
//! A capability space used to be a bare `GenTable<CapabilityEntry>`, and
//! `find_authorized_capability` answered "may this actor do X to object Y?" by
//! scanning all of it. That is fine for a space holding a handful of
//! capabilities and it is the wrong shape for a long run: every published
//! batch mints capabilities that stay in the space, so the scan grows with the
//! age of the run and every authorization pays for every object the actor has
//! ever touched.
//!
//! `examples/kernel_overhead` measures what that costs. Publishing one more
//! cohort takes 6µs into an empty kernel, 36µs after a thousand, 125µs after
//! four thousand, and 485µs after sixteen thousand — linear in what is already
//! there, so a run publishing n cohorts does O(n²) work and the constant is
//! authorization, not anything the run asked for.
//!
//! So the space keeps a map from target to the capabilities naming it. Lookup
//! by target is the only query on the hot path; everything else still walks
//! the table, because nothing else is called per authorization.
//!
//! There are two indexes: by target, for authorization, and by parent, for
//! revocation, which walks the tree of capabilities derived from one root.
//! Both are maintained by `alloc` and `delete`, the only ways in and out.
//!
//! `get_mut` hands out an entry a caller may edit, including its target and
//! parent, and there is no way to tell from here whether it did. So `get_mut`
//! marks the indexes possibly stale and both queries fall back to scanning
//! from then on — slower, and never wrong. Nothing in the kernel takes an
//! entry mutably; the raw fault-injection tests do, and they build small
//! kernels where a scan costs nothing.

use std::collections::HashMap;

use crate::abi::capabilities::CapabilityEntry;
use crate::abi::{AbiError, Kind, Ref64};
use crate::table::GenTable;

/// The capabilities under one index key.
///
/// Almost every bucket holds one entry — an object is typically named by a
/// single capability, and a capability has few children — and a `Vec` for one
/// `Ref64` is a heap allocation per key. `examples/memory_profile` prices a
/// published batch at about 1.3KB against 32 bytes of payload, and three
/// capabilities with two index entries each is a large part of that.
#[derive(Clone, Debug)]
enum Bucket {
    One(Ref64),
    Many(Vec<Ref64>),
}

impl Bucket {
    fn push(&mut self, reference: Ref64) {
        match self {
            Bucket::One(held) => *self = Bucket::Many(vec![*held, reference]),
            Bucket::Many(held) => held.push(reference),
        }
    }

    /// Remove `reference`; reports whether the bucket is now empty.
    fn remove(&mut self, reference: Ref64) -> bool {
        match self {
            Bucket::One(held) => *held == reference,
            Bucket::Many(held) => {
                held.retain(|entry| *entry != reference);
                held.is_empty()
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = Ref64> + '_ {
        match self {
            Bucket::One(held) => std::slice::from_ref(held).iter().copied(),
            Bucket::Many(held) => held.as_slice().iter().copied(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CapabilitySpace {
    table: GenTable<CapabilityEntry>,
    by_target: HashMap<u64, Bucket>,
    by_parent: HashMap<u64, Bucket>,
    /// Set when an entry has been handed out mutably, since a caller may have
    /// changed the target or parent the indexes are built on. While set, both
    /// queries fall back to scanning, which is slow and correct. Nothing in
    /// the kernel takes an entry mutably; the raw fault-injection tests do.
    indexes_may_be_stale: bool,
}

impl CapabilitySpace {
    pub fn new(kind: Kind) -> Self {
        Self {
            table: GenTable::new(kind),
            by_target: HashMap::new(),
            by_parent: HashMap::new(),
            indexes_may_be_stale: false,
        }
    }

    pub fn alloc(&mut self, entry: CapabilityEntry) -> Ref64 {
        let target = entry.target.key();
        // A root capability has no parent, and every root would otherwise
        // share one bucket keyed on null — which is most of the space, making
        // the bucket maintenance on `delete` linear in the whole space and
        // undoing the point of the index. Roots are never looked up by parent.
        let parent = (!entry.parent_capability.is_null()).then(|| entry.parent_capability.key());
        let reference = self.table.alloc(entry);
        insert_into(&mut self.by_target, target, reference);
        if let Some(parent) = parent {
            insert_into(&mut self.by_parent, parent, reference);
        }
        reference
    }

    pub fn delete(&mut self, reference: Ref64) -> Result<CapabilityEntry, AbiError> {
        let entry = self.table.delete(reference)?;
        remove_from(&mut self.by_target, entry.target.key(), reference);
        if !entry.parent_capability.is_null() {
            remove_from(
                &mut self.by_parent,
                entry.parent_capability.key(),
                reference,
            );
        }
        Ok(entry)
    }

    pub fn get(&self, reference: Ref64) -> Result<&CapabilityEntry, AbiError> {
        self.table.get(reference)
    }

    /// A capability, mutably.
    ///
    /// Both indexes are marked possibly stale, because the caller may change
    /// the target or the parent and there is no way to tell from here. The
    /// space keeps answering correctly afterwards, by scanning.
    pub fn get_mut(&mut self, reference: Ref64) -> Result<&mut CapabilityEntry, AbiError> {
        self.indexes_may_be_stale = true;
        self.table.get_mut(reference)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Ref64, &CapabilityEntry)> {
        self.table.iter()
    }

    /// The capabilities naming `target`, which is the only question
    /// authorization asks.
    ///
    /// The bucket key is `Ref64::key()`, which is partition and slot and *not*
    /// kind or generation, so a process and an object occupying the same slot
    /// share a bucket. Every lookup therefore re-checks the whole reference:
    /// the index narrows the search and does not decide the answer. Skipping
    /// that check let a revocation aimed at an object delete a capability over
    /// the domain in the same slot, which the `CapabilityIntegrity` invariant
    /// caught only because reclamation started deleting things.
    ///
    /// A stale reference in a bucket — one whose slot was reused under a new
    /// generation — is filtered by `get`, so the index may lag deletions
    /// without admitting a capability that no longer exists.
    pub fn for_target(&self, target: Ref64) -> Vec<(Ref64, &CapabilityEntry)> {
        self.lookup(&self.by_target, target.key(), |entry| {
            entry.target == target
        })
    }

    /// The capabilities whose parent is `parent`, which is how a revocation
    /// walks the tree it is revoking.
    ///
    /// Revoking is not rare: freezing an object revokes write authority over
    /// it, so this runs for every published batch. Finding children by
    /// scanning the space made that walk linear in every capability the actor
    /// held, which `examples/kernel_overhead` measured as the dominant cost of
    /// publishing into a kernel that had been running for a while.
    pub fn children_of(&self, parent: Ref64) -> Vec<Ref64> {
        if parent.is_null() {
            // Roots are not in the parent index, so answer this one by
            // scanning rather than by reporting that nothing has a null
            // parent. Nothing calls it: a revocation starts from a capability.
            return self
                .table
                .iter()
                .filter(|(_, entry)| entry.parent_capability.is_null())
                .map(|(reference, _)| reference)
                .collect();
        }
        self.lookup(&self.by_parent, parent.key(), |entry| {
            entry.parent_capability == parent
        })
        .into_iter()
        .map(|(reference, _)| reference)
        .collect()
    }

    fn lookup(
        &self,
        index: &HashMap<u64, Bucket>,
        key: u64,
        matches: impl Fn(&CapabilityEntry) -> bool,
    ) -> Vec<(Ref64, &CapabilityEntry)> {
        if self.indexes_may_be_stale {
            return self
                .table
                .iter()
                .filter(|(_, entry)| matches(entry))
                .collect();
        }
        index
            .get(&key)
            .into_iter()
            .flat_map(Bucket::iter)
            .filter_map(|reference| Some((reference, self.table.get(reference).ok()?)))
            .filter(|(_, entry)| matches(entry))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    pub fn set_active_partition(&mut self, partition: u8) {
        self.table.set_active_partition(partition);
    }

    /// Rebuild the target index from the table.
    ///
    /// Only needed if a capability's `target` was changed in place, which
    /// nothing in the kernel does.
    pub fn reindex(&mut self) {
        let mut by_target: HashMap<u64, Bucket> = HashMap::new();
        let mut by_parent: HashMap<u64, Bucket> = HashMap::new();
        for (reference, entry) in self.table.iter() {
            insert_into(&mut by_target, entry.target.key(), reference);
            if !entry.parent_capability.is_null() {
                insert_into(&mut by_parent, entry.parent_capability.key(), reference);
            }
        }
        self.by_target = by_target;
        self.by_parent = by_parent;
        self.indexes_may_be_stale = false;
    }
}

fn insert_into(index: &mut HashMap<u64, Bucket>, key: u64, reference: Ref64) {
    match index.get_mut(&key) {
        Some(bucket) => bucket.push(reference),
        None => {
            index.insert(key, Bucket::One(reference));
        }
    }
}

fn remove_from(index: &mut HashMap<u64, Bucket>, key: u64, reference: Ref64) {
    if let Some(bucket) = index.get_mut(&key) {
        if bucket.remove(reference) {
            index.remove(&key);
        }
    }
}
