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
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Ref64 {
    pub slot: u32,
    pub generation: u16,
    pub kind: Kind,
    pub flags: u8,
}

impl Ref64 {
    pub const NULL: Ref64 = Ref64 {
        slot: 0,
        generation: 0,
        kind: Kind::Domain,
        flags: 0,
    };

    pub fn new(slot: u32, generation: u16, kind: Kind) -> Ref64 {
        Ref64 {
            slot,
            generation,
            kind,
            flags: 0,
        }
    }

    pub fn is_null(&self) -> bool {
        self.slot == 0
    }

    /// Raw 64-bit little-endian encoding of the reference.
    pub fn to_u64(&self) -> u64 {
        let slot = (self.slot as u64) & 0xFFFF_FFFF;
        let generation = (self.generation as u64) & 0xFFFF;
        let kind = (self.kind.as_u8() as u64) & 0xFF;
        let flags = (self.flags as u64) & 0xFF;
        slot | (generation << 32) | (kind << 48) | (flags << 56)
    }

    /// Rebuild a reference from its `to_u64` encoding.
    pub fn from_u64(v: u64) -> Ref64 {
        let slot = (v & 0xFFFF_FFFF) as u32;
        let generation = ((v >> 32) & 0xFFFF) as u16;
        let kind = Kind::from_u8(((v >> 48) & 0xFF) as u8).unwrap_or(Kind::Domain);
        let flags = ((v >> 56) & 0xFF) as u8;
        Ref64 {
            slot,
            generation,
            kind,
            flags,
        }
    }
}
