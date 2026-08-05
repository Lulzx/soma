//! Frame byte-blob encoding.
//!
//! Per the chosen design, continuation frames are plain little-endian byte
//! blobs stored in an Object of kind `CONTINUATION_FRAME`. Each run class has a
//! `frame_type`; the corresponding frame struct implements `Frame`, providing
//! a deterministic `encode`/`decode` pair. This is forward-compatible with GPU
//! lane execution and deterministic replay, where the same bytes move between
//! executives without register migration (§16).

use crate::abi::Ref64;

/// A frame type that can be serialized to and from a little-endian byte blob.
pub trait Frame: Sized {
    fn encode(&self, out: &mut Vec<u8>);
    fn decode(cursor: &mut ByteCursor) -> Result<Self, FrameError>;
}

/// Read/write cursor over a byte blob.
pub struct ByteCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    pub fn new(bytes: &'a [u8]) -> ByteCursor<'a> {
        ByteCursor { bytes, pos: 0 }
    }

    pub fn u8(&mut self) -> Result<u8, FrameError> {
        let b = *self.bytes.get(self.pos).ok_or(FrameError::Truncated)?;
        self.pos += 1;
        Ok(b)
    }

    pub fn u16(&mut self) -> Result<u16, FrameError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    pub fn u32(&mut self) -> Result<u32, FrameError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub fn u64(&mut self) -> Result<u64, FrameError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }

    pub fn ref64(&mut self) -> Result<Ref64, FrameError> {
        Ok(Ref64::from_u64(self.u64()?))
    }

    /// Read a length-prefixed `Vec<u64>`. Length is a u32 prefix.
    pub fn vec_u64(&mut self) -> Result<Vec<u64>, FrameError> {
        let n = self.u32()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.u64()?);
        }
        Ok(v)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], FrameError> {
        if self.pos + n > self.bytes.len() {
            return Err(FrameError::Truncated);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

/// Little-endian writers.
pub fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
pub fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub fn put_ref64(out: &mut Vec<u8>, r: Ref64) {
    put_u64(out, r.to_u64());
}
pub fn put_vec_u64(out: &mut Vec<u8>, v: &[u64]) {
    put_u32(out, v.len() as u32);
    for x in v {
        put_u64(out, *x);
    }
}

/// Frame encoding / decoding error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    Truncated,
    Trailing,
    BadMagic(u32),
}
