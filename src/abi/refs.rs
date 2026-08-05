//! Fixed-width ABI references and the structure-kind namespace.
//!
//! Per §3 of the SOMA-P1 contract: all shared ABI structures use fixed-width
//! integer fields, little-endian representation, 16-byte minimum alignment, no
//! native language pointers, no implementation-defined enums, explicit ABI
//! version numbers, and generation-checked references.
//!
//! On x86-64 and aarch64 the in-memory layout of the integer fields below is
//! already little-endian. Byte-for-byte serialization to an on-disk / on-wire
//! ABI (via `to_le_bytes`) is deferred until the GPU/wire step; the in-memory
//! structs are the source of truth here.

/// The structure-kind namespace. Mirrors §4's list.
///
/// This is a plain `u8` discriminator, not a native enum, so it can live in a
/// fixed-width ABI field without an implementation-defined encoding. We wrap it
/// in a Rust enum only for the ergonomic `Kind::Process` spelling; it still
/// serializes as its `u8` discriminant.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Kind {
    #[default]
    Domain = 1,
    Process = 2,
    Object = 3,
    Capability = 4,
    Continuation = 5,
    Channel = 6,
    Future = 7,
    Contract = 8,
    Collective = 9,
    Module = 10,
}

impl Kind {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(v: u8) -> Option<Kind> {
        match v {
            1 => Some(Kind::Domain),
            2 => Some(Kind::Process),
            3 => Some(Kind::Object),
            4 => Some(Kind::Capability),
            5 => Some(Kind::Continuation),
            6 => Some(Kind::Channel),
            7 => Some(Kind::Future),
            8 => Some(Kind::Contract),
            9 => Some(Kind::Collective),
            10 => Some(Kind::Module),
            _ => None,
        }
    }
}

/// Compact generational reference into a kernel-managed table (§4).
///
/// A reference is valid only when:
/// - `slot` exists,
/// - `kind` matches the table it is dereferenced against,
/// - `generation` equals the current slot generation, and
/// - the requesting domain has authority to use it (deferred in Phase 1).
///
/// `Ref64` is not itself an authority; it is merely a table reference.
///
/// The generation field is 16 bits. That used to bound staleness detection
/// rather than guarantee it — a slot recycled 65,536 times wrapped back to a
/// generation a long-held stale reference could match, a classic ABA window.
///
/// `GenTable::delete` now retires a slot rather than wrapping its generation,
/// so staleness detection is guaranteed at every generation width. The cost
/// moved from correctness to space: one permanently withdrawn slot per 65,535
/// recycles, reported by `GenTable::retired_slots`. Keeping the field at 16
/// bits is therefore a space/churn tradeoff rather than a soundness one, and a
/// reference still fits in 64 bits.
///
/// `partition` is the allocator a reference was minted by. Two allocators may
/// hand out slot 7 at the same time and produce different references, which is
/// what lets a device's lanes and a cluster's nodes allocate without
/// coordinating. The sequential interpreter uses partition 0 throughout, so its
/// references are exactly what they were.
///
/// It occupies the byte that used to be `flags`, which was written as zero
/// everywhere and read nowhere, so a reference still fits in 64 bits.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Ref64 {
    pub slot: u32,
    pub generation: u16,
    pub kind: Kind,
    pub partition: u8,
}

impl Ref64 {
    pub const NULL: Ref64 = Ref64 {
        slot: 0,
        generation: 0,
        kind: Kind::Domain,
        partition: 0,
    };

    /// A reference minted by the default allocator (partition 0).
    pub fn new(slot: u32, generation: u16, kind: Kind) -> Ref64 {
        Ref64::in_partition(slot, generation, kind, 0)
    }

    /// A reference minted by `partition`'s allocator.
    pub fn in_partition(slot: u32, generation: u16, kind: Kind, partition: u8) -> Ref64 {
        Ref64 {
            slot,
            generation,
            kind,
            partition,
        }
    }

    /// Rebuild a process reference from a `key`, with a generation of zero.
    ///
    /// Used only where the kernel walks a map keyed by process identity and
    /// needs a reference back. The generation is not recoverable from a key and
    /// is not wanted here: these lookups ask "which process", not "is this
    /// reference still valid".
    pub fn process_at(key: u64) -> Ref64 {
        Ref64 {
            slot: (key & 0xFFFF_FFFF) as u32,
            generation: 0,
            kind: Kind::Process,
            partition: ((key >> 32) & 0xFF) as u8,
        }
    }

    /// The identity of the entity a reference names, ignoring generation: the
    /// key under which kernel-side maps that outlive a single reference — a
    /// process's mailbox, an object's payload — are stored.
    ///
    /// A bare slot stopped being that key when partitions arrived, because two
    /// allocators may both mint slot 7.
    pub fn key(&self) -> u64 {
        ((self.partition as u64) << 32) | (self.slot as u64)
    }

    pub fn is_null(&self) -> bool {
        self.slot == 0
    }

    /// Raw 64-bit little-endian encoding of the reference.
    pub fn to_u64(&self) -> u64 {
        let slot = (self.slot as u64) & 0xFFFF_FFFF;
        let generation = (self.generation as u64) & 0xFFFF;
        let kind = (self.kind.as_u8() as u64) & 0xFF;
        let partition = (self.partition as u64) & 0xFF;
        slot | (generation << 32) | (kind << 48) | (partition << 56)
    }

    /// Rebuild a reference from its `to_u64` encoding.
    pub fn from_u64(v: u64) -> Ref64 {
        let slot = (v & 0xFFFF_FFFF) as u32;
        let generation = ((v >> 32) & 0xFFFF) as u16;
        let kind = Kind::from_u8(((v >> 48) & 0xFF) as u8).unwrap_or(Kind::Domain);
        let partition = ((v >> 56) & 0xFF) as u8;
        Ref64 {
            slot,
            generation,
            kind,
            partition,
        }
    }
}
