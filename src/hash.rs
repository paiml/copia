//! Strong hash implementation using BLAKE3.
//!
//! BLAKE3 provides cryptographic verification of block matches after
//! the rolling checksum identifies potential candidates.

use std::io::Read;

use serde::{Deserialize, Serialize};

/// Strong cryptographic hash for block verification.
///
/// Uses BLAKE3 for 256-bit security with high performance.
/// BLAKE3 is significantly faster than SHA-256 while providing
/// equivalent security guarantees.
///
/// # Example
///
/// ```rust
/// use copia::StrongHash;
///
/// let hash1 = StrongHash::compute(b"hello world");
/// let hash2 = StrongHash::compute(b"hello world");
/// assert_eq!(hash1, hash2);
///
/// let hash3 = StrongHash::compute(b"different data");
/// assert_ne!(hash1, hash3);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrongHash([u8; 32]);

impl StrongHash {
    /// Compute BLAKE3 hash of data.
    ///
    /// # Arguments
    ///
    /// * `data` - Byte slice to hash
    ///
    /// # Example
    ///
    /// ```rust
    /// use copia::StrongHash;
    ///
    /// let hash = StrongHash::compute(b"test data");
    /// ```
    #[must_use]
    pub fn compute(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Compute hash from a reader with streaming interface.
    ///
    /// This is useful for large files that shouldn't be loaded entirely
    /// into memory.
    ///
    /// # Arguments
    ///
    /// * `reader` - Any type implementing `Read`
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use copia::StrongHash;
    /// use std::io::Cursor;
    ///
    /// let data = b"streaming data";
    /// let mut cursor = Cursor::new(data);
    /// let hash = StrongHash::compute_streaming(&mut cursor).unwrap();
    /// ```
    pub fn compute_streaming<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 8192];

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        Ok(Self(*hasher.finalize().as_bytes()))
    }

    /// Create a `StrongHash` from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - 32-byte array
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes of the hash.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Get a truncated view of the hash.
    ///
    /// Useful for memory-efficient signature tables where full
    /// 32-byte hashes aren't necessary.
    ///
    /// # Arguments
    ///
    /// * `len` - Number of bytes to return (clamped to 32)
    #[must_use]
    pub fn truncated(&self, len: usize) -> &[u8] {
        &self.0[..len.min(32)]
    }

    /// Check equality of truncated hashes.
    ///
    /// # Arguments
    ///
    /// * `other` - Other hash to compare
    /// * `len` - Number of bytes to compare
    #[must_use]
    pub fn eq_truncated(&self, other: &Self, len: usize) -> bool {
        self.truncated(len) == other.truncated(len)
    }

    /// Constant-time equality comparison.
    ///
    /// Prevents timing attacks when comparing hashes in security-sensitive contexts.
    #[must_use]
    pub fn ct_eq(&self, other: &Self) -> bool {
        // XOR all bytes and OR results together
        // If any byte differs, result will be non-zero
        let mut result = 0u8;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            result |= a ^ b;
        }
        result == 0
    }

    /// Create a zero hash (for testing/initialization).
    #[must_use]
    pub const fn zero() -> Self {
        Self([0u8; 32])
    }
}

impl std::fmt::Debug for StrongHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StrongHash({:016x}...)", u64::from_be_bytes(self.0[..8].try_into().unwrap_or([0u8; 8])))
    }
}

impl std::fmt::Display for StrongHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Default for StrongHash {
    fn default() -> Self {
        Self::zero()
    }
}

impl AsRef<[u8]> for StrongHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ==========================================================================
    // UNIT TESTS - Basic functionality
    // ==========================================================================

    #[test]
    fn compute_empty() {
        let hash = StrongHash::compute(b"");
        // BLAKE3 has a well-defined hash for empty input
        assert_ne!(hash, StrongHash::zero());
    }

    #[test]
    fn compute_single_byte() {
        let hash = StrongHash::compute(b"a");
        assert_ne!(hash, StrongHash::zero());
        assert_ne!(hash, StrongHash::compute(b""));
    }

    #[test]
    fn compute_deterministic() {
        let data = b"test data for hashing";
        let hash1 = StrongHash::compute(data);
        let hash2 = StrongHash::compute(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn compute_different_data() {
        let hash1 = StrongHash::compute(b"hello");
        let hash2 = StrongHash::compute(b"world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn compute_case_sensitive() {
        let hash1 = StrongHash::compute(b"Hello");
        let hash2 = StrongHash::compute(b"hello");
        assert_ne!(hash1, hash2);
    }

    // ==========================================================================
    // STREAMING TESTS
    // ==========================================================================

    #[test]
    fn streaming_matches_direct() {
        let data = b"test data for streaming hash computation";
        let direct = StrongHash::compute(data);

        let mut cursor = Cursor::new(data);
        let streaming = StrongHash::compute_streaming(&mut cursor).unwrap();

        assert_eq!(direct, streaming);
    }

    #[test]
    fn streaming_empty() {
        let data: &[u8] = b"";
        let direct = StrongHash::compute(data);

        let mut cursor = Cursor::new(data);
        let streaming = StrongHash::compute_streaming(&mut cursor).unwrap();

        assert_eq!(direct, streaming);
    }

    #[test]
    fn streaming_large_data() {
        let data = vec![42u8; 100_000];
        let direct = StrongHash::compute(&data);

        let mut cursor = Cursor::new(&data);
        let streaming = StrongHash::compute_streaming(&mut cursor).unwrap();

        assert_eq!(direct, streaming);
    }

    // ==========================================================================
    // TRUNCATION TESTS
    // ==========================================================================

    #[test]
    fn truncated_returns_prefix() {
        let hash = StrongHash::compute(b"test");
        let full = hash.as_bytes();
        let truncated = hash.truncated(8);

        assert_eq!(truncated.len(), 8);
        assert_eq!(truncated, &full[..8]);
    }

    #[test]
    fn truncated_clamps_to_32() {
        let hash = StrongHash::compute(b"test");
        let truncated = hash.truncated(100);
        assert_eq!(truncated.len(), 32);
    }

    #[test]
    fn truncated_zero() {
        let hash = StrongHash::compute(b"test");
        let truncated = hash.truncated(0);
        assert!(truncated.is_empty());
    }

    #[test]
    fn eq_truncated_equal() {
        let hash1 = StrongHash::compute(b"test");
        let hash2 = StrongHash::compute(b"test");
        assert!(hash1.eq_truncated(&hash2, 8));
        assert!(hash1.eq_truncated(&hash2, 32));
    }

    #[test]
    fn eq_truncated_different() {
        let hash1 = StrongHash::compute(b"test1");
        let hash2 = StrongHash::compute(b"test2");
        assert!(!hash1.eq_truncated(&hash2, 8));
    }

    // ==========================================================================
    // CONSTANT TIME COMPARISON
    // ==========================================================================

    #[test]
    fn ct_eq_equal() {
        let hash1 = StrongHash::compute(b"test");
        let hash2 = StrongHash::compute(b"test");
        assert!(hash1.ct_eq(&hash2));
    }

    #[test]
    fn ct_eq_different() {
        let hash1 = StrongHash::compute(b"test1");
        let hash2 = StrongHash::compute(b"test2");
        assert!(!hash1.ct_eq(&hash2));
    }

    #[test]
    fn ct_eq_matches_regular_eq() {
        let hash1 = StrongHash::compute(b"same data");
        let hash2 = StrongHash::compute(b"same data");
        let hash3 = StrongHash::compute(b"different");

        assert_eq!(hash1 == hash2, hash1.ct_eq(&hash2));
        assert_eq!(hash1 == hash3, hash1.ct_eq(&hash3));
    }

    // ==========================================================================
    // CONSTRUCTORS AND CONVERSIONS
    // ==========================================================================

    #[test]
    fn from_bytes() {
        let bytes = [42u8; 32];
        let hash = StrongHash::from_bytes(bytes);
        assert_eq!(*hash.as_bytes(), bytes);
    }

    #[test]
    fn zero() {
        let hash = StrongHash::zero();
        assert_eq!(*hash.as_bytes(), [0u8; 32]);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(StrongHash::default(), StrongHash::zero());
    }

    #[test]
    fn as_ref() {
        let hash = StrongHash::compute(b"test");
        let bytes: &[u8] = hash.as_ref();
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes, hash.as_bytes());
    }

    // ==========================================================================
    // DISPLAY AND DEBUG
    // ==========================================================================

    #[test]
    fn display_format() {
        let hash = StrongHash::compute(b"test");
        let display = format!("{hash}");
        assert_eq!(display.len(), 64); // 32 bytes * 2 hex chars
        assert!(display.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn debug_format() {
        let hash = StrongHash::compute(b"test");
        let debug = format!("{hash:?}");
        assert!(debug.starts_with("StrongHash("));
        assert!(debug.contains("..."));
    }

    // ==========================================================================
    // HASHING (as HashMap key)
    // ==========================================================================

    #[test]
    fn hashable() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let hash1 = StrongHash::compute(b"test1");
        let hash2 = StrongHash::compute(b"test2");
        let hash3 = StrongHash::compute(b"test1"); // Same as hash1

        set.insert(hash1);
        set.insert(hash2);
        set.insert(hash3);

        assert_eq!(set.len(), 2); // hash1 and hash3 are equal
    }

    // ==========================================================================
    // SERDE SERIALIZATION
    // ==========================================================================

    #[test]
    fn serde_roundtrip() {
        let original = StrongHash::compute(b"test data");
        let serialized = bincode::serialize(&original).unwrap();
        let deserialized: StrongHash = bincode::deserialize(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn serde_size() {
        let hash = StrongHash::compute(b"test");
        let serialized = bincode::serialize(&hash).unwrap();
        // Should be exactly 32 bytes (no overhead with bincode)
        assert_eq!(serialized.len(), 32);
    }

    // ==========================================================================
    // EDGE CASES
    // ==========================================================================

    #[test]
    fn binary_data() {
        let data: Vec<u8> = (0..=255).collect();
        let hash = StrongHash::compute(&data);
        assert_ne!(hash, StrongHash::zero());
    }

    #[test]
    fn large_data() {
        let data = vec![0xABu8; 1_000_000];
        let hash = StrongHash::compute(&data);
        assert_ne!(hash, StrongHash::zero());
    }

    #[test]
    fn null_bytes() {
        let hash1 = StrongHash::compute(&[0u8; 10]);
        let hash2 = StrongHash::compute(&[0u8; 11]);
        assert_ne!(hash1, hash2); // Different lengths should produce different hashes
    }

    // ==========================================================================
    // CLONE AND COPY
    // ==========================================================================

    #[test]
    fn clone_equals_original() {
        let original = StrongHash::compute(b"test");
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn copy_semantics() {
        let original = StrongHash::compute(b"test");
        let copied = original; // Copy, not move
        assert_eq!(original, copied);
        // Both should still be usable
        let _ = original.as_bytes();
        let _ = copied.as_bytes();
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Hash computation is deterministic
        #[test]
        fn deterministic(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let hash1 = StrongHash::compute(&data);
            let hash2 = StrongHash::compute(&data);
            prop_assert_eq!(hash1, hash2);
        }

        /// Different data (usually) produces different hashes
        #[test]
        fn collision_resistant(
            data1 in prop::collection::vec(any::<u8>(), 1..100),
            data2 in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            if data1 != data2 {
                let hash1 = StrongHash::compute(&data1);
                let hash2 = StrongHash::compute(&data2);
                // With 256-bit hashes, collisions are astronomically unlikely
                prop_assert_ne!(hash1, hash2);
            }
        }

        /// Streaming and direct produce same result
        #[test]
        fn streaming_equivalence(data in prop::collection::vec(any::<u8>(), 0..10000)) {
            let direct = StrongHash::compute(&data);
            let mut cursor = std::io::Cursor::new(&data);
            let streaming = StrongHash::compute_streaming(&mut cursor).unwrap();
            prop_assert_eq!(direct, streaming);
        }

        /// Truncation returns correct prefix
        #[test]
        fn truncation_correct(
            data in prop::collection::vec(any::<u8>(), 1..100),
            len in 0usize..40
        ) {
            let hash = StrongHash::compute(&data);
            let truncated = hash.truncated(len);
            let expected_len = len.min(32);
            prop_assert_eq!(truncated.len(), expected_len);
            prop_assert_eq!(truncated, &hash.as_bytes()[..expected_len]);
        }

        /// ct_eq matches regular equality
        #[test]
        fn ct_eq_matches_eq(
            data1 in prop::collection::vec(any::<u8>(), 0..100),
            data2 in prop::collection::vec(any::<u8>(), 0..100)
        ) {
            let hash1 = StrongHash::compute(&data1);
            let hash2 = StrongHash::compute(&data2);
            prop_assert_eq!(hash1 == hash2, hash1.ct_eq(&hash2));
        }

        /// Serde roundtrip preserves data
        #[test]
        fn serde_roundtrip_preserves(data in prop::collection::vec(any::<u8>(), 0..100)) {
            let original = StrongHash::compute(&data);
            let serialized = bincode::serialize(&original).unwrap();
            let deserialized: StrongHash = bincode::deserialize(&serialized).unwrap();
            prop_assert_eq!(original, deserialized);
        }
    }
}
