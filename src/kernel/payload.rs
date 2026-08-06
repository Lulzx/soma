//! Where an object's bytes actually live.
//!
//! Every payload used to be a `Vec<u8>`, which is the right default and the
//! wrong requirement. On Apple silicon the CPU and GPU address the same
//! physical memory, so a batch that a GPU has just written is already in a
//! location the CPU can read; copying it into a `Vec` so that the kernel can
//! own it is a full pass over the batch performed for the sake of a type.
//!
//! `Payload` lets an object be backed by an allocation the core did not make,
//! without the core learning what Metal is. `ForeignPayload` is deliberately
//! the smallest possible interface — bytes, a length, and a destructor by way
//! of `Box` — because everything the semantic layer does with a payload it
//! does through a slice. An object whose bytes live in an `MTLBuffer` is
//! frozen, authorized, traced, and published exactly like any other.
//!
//! The safety obligation lives with whoever hands over a `ForeignPayload`: it
//! transfers ownership. A backend that keeps writing to a buffer it has
//! published would be mutating a frozen object behind the kernel's back, which
//! no invariant here can catch. That is why `MetalBatchBackend` allocates a
//! fresh output buffer for a batch it intends to publish rather than handing
//! over the one it reuses.

/// Bytes owned elsewhere, exposed to the kernel as a slice.
///
/// Implementors must keep the returned slice valid and stable for as long as
/// the payload lives, and must not alias it with anything still writing.
pub trait ForeignPayload: Send {
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
    /// What kind of memory this is, for traces and reports. Not semantic.
    fn provenance(&self) -> &'static str {
        "foreign"
    }
}

pub enum Payload {
    Host(Vec<u8>),
    Foreign(Box<dyn ForeignPayload>),
}

impl Payload {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Payload::Host(bytes) => bytes.as_slice(),
            Payload::Foreign(bytes) => bytes.as_slice(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Payload::Host(bytes) => bytes.as_mut_slice(),
            Payload::Foreign(bytes) => bytes.as_mut_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The payload as a growable `Vec`, when it is one.
    ///
    /// Only host payloads can grow: a foreign allocation has a fixed size
    /// decided by whoever made it. Process state is always host-backed and is
    /// the one payload that legitimately resizes in place, so it uses this;
    /// everything else takes a slice and cannot change the length behind
    /// `ObjectDescriptor::byte_length`'s back.
    pub fn as_mut_vec(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            Payload::Host(bytes) => Some(bytes),
            Payload::Foreign(_) => None,
        }
    }

    pub fn provenance(&self) -> &'static str {
        match self {
            Payload::Host(_) => "host",
            Payload::Foreign(bytes) => bytes.provenance(),
        }
    }
}

impl From<Vec<u8>> for Payload {
    fn from(bytes: Vec<u8>) -> Self {
        Payload::Host(bytes)
    }
}

impl std::fmt::Debug for Payload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Length and provenance rather than contents: a payload can be tens of
        // megabytes, and `Kernel` derives `Debug`.
        f.debug_struct("Payload")
            .field("provenance", &self.provenance())
            .field("len", &self.len())
            .finish()
    }
}
