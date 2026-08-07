//! Stable content identities for discovery computations.

use sha2::{Digest, Sha256};

macro_rules! digest_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub fn of(bytes: &[u8]) -> Self {
                let mut hasher = Sha256::new();
                hasher.update(bytes);
                Self(hasher.finalize().into())
            }
        }
    };
}

digest_type!(ObjectDigest);
digest_type!(ModuleDigest);
digest_type!(NodeDigest);
digest_type!(ExperimentKey);

/// Length-prefix every field so `(a, bc)` and `(ab, c)` cannot alias.
pub(crate) fn hash_fields(tag: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((tag.len() as u64).to_le_bytes());
    hasher.update(tag);
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}
