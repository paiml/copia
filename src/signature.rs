//! Signature generation and lookup for rsync delta computation.
//!
//! The signature contains rolling checksums and strong hashes for each block
//! of the basis file, enabling efficient block matching during delta computation.

use std::io::Read;

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::checksum::RollingChecksum;
use crate::error::{CopiaError, Result};
use crate::hash::StrongHash;

/// Signature for a single block in the basis file.
///
/// Contains both the weak (rolling) checksum for fast filtering
/// and the strong (BLAKE3) hash for verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSignature {
    /// Block index (0-based position in file).
    pub index: u32,
    /// Rolling checksum for fast matching.
    pub weak_hash: u32,
    /// Strong cryptographic hash for verification.
    pub strong_hash: StrongHash,
}

impl BlockSignature {
    /// Create a new block signature.
    ///
    /// # Arguments
    ///
    /// * `index` - Block index in the file
    /// * `weak_hash` - Rolling checksum value
    /// * `strong_hash` - BLAKE3 hash
    #[must_use]
    pub const fn new(index: u32, weak_hash: u32, strong_hash: StrongHash) -> Self {
        Self {
            index,
            weak_hash,
            strong_hash,
        }
    }

    /// Compute signature for a data block.
    ///
    /// # Arguments
    ///
    /// * `index` - Block index
    /// * `data` - Block data
    #[must_use]
    pub fn compute(index: u32, data: &[u8]) -> Self {
        Self {
            index,
            weak_hash: RollingChecksum::new(data).digest(),
            strong_hash: StrongHash::compute(data),
        }
    }
}

/// Complete signature of a basis file.
///
/// Contains all block signatures and metadata needed for delta computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    /// Block size used during signature generation.
    pub block_size: usize,
    /// Total size of the basis file in bytes.
    pub file_size: u64,
    /// Signatures for each block.
    pub blocks: Vec<BlockSignature>,
}

impl Signature {
    /// Create a new signature.
    #[must_use]
    pub const fn new(block_size: usize, file_size: u64) -> Self {
        Self {
            block_size,
            file_size,
            blocks: Vec::new(),
        }
    }

    /// Generate signature from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Data source to generate signature from
    /// * `block_size` - Size of each block
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading fails.
    pub fn generate<R: Read>(reader: &mut R, block_size: usize) -> Result<Self> {
        // Read all data first for parallel processing
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let file_size = data.len() as u64;

        if data.is_empty() {
            return Ok(Self {
                block_size,
                file_size: 0,
                blocks: Vec::new(),
            });
        }

        // Parallel signature computation for large files
        let blocks: Vec<BlockSignature> = if data.len() > 64 * 1024 {
            // For files >64KB, use parallel processing
            data.par_chunks(block_size)
                .enumerate()
                .map(|(i, chunk)| {
                    #[allow(clippy::cast_possible_truncation)]
                    BlockSignature::compute(i as u32, chunk)
                })
                .collect()
        } else {
            // For small files, sequential is faster
            data.chunks(block_size)
                .enumerate()
                .map(|(i, chunk)| {
                    #[allow(clippy::cast_possible_truncation)]
                    BlockSignature::compute(i as u32, chunk)
                })
                .collect()
        };

        Ok(Self {
            block_size,
            file_size,
            blocks,
        })
    }

    /// Get the number of blocks in the signature.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Check if the signature is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Build a lookup table from this signature.
    #[must_use]
    pub fn into_table(self) -> SignatureTable {
        SignatureTable::from_signature(self)
    }
}

/// Efficient lookup table for block matching.
///
/// Uses a two-level lookup: first by weak hash (fast), then by strong hash (verification).
/// This enables O(1) average-case block matching with `FxHash` for fast u32 keys.
#[derive(Debug)]
pub struct SignatureTable {
    /// First level: rolling checksum -> candidate block indices.
    /// Uses `FxHashMap` for ~2x faster lookups with integer keys.
    weak_index: FxHashMap<u32, Vec<usize>>,
    /// Full signature data.
    signature: Signature,
}

impl SignatureTable {
    /// Build a signature table from a signature.
    #[must_use]
    pub fn from_signature(signature: Signature) -> Self {
        let mut weak_index: FxHashMap<u32, Vec<usize>> =
            FxHashMap::with_capacity_and_hasher(signature.blocks.len(), rustc_hash::FxBuildHasher);

        for (i, block) in signature.blocks.iter().enumerate() {
            weak_index.entry(block.weak_hash).or_default().push(i);
        }

        Self {
            weak_index,
            signature,
        }
    }

    /// Build a signature table directly from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Data source
    /// * `block_size` - Block size for signature generation
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading fails.
    pub fn build<R: Read>(reader: &mut R, block_size: usize) -> Result<Self> {
        let signature = Signature::generate(reader, block_size)?;
        Ok(Self::from_signature(signature))
    }

    /// Find a matching block for the given weak hash and data.
    ///
    /// First filters by weak hash, then verifies with strong hash.
    ///
    /// # Arguments
    ///
    /// * `weak` - Rolling checksum of the candidate data
    /// * `data` - Actual data to verify against strong hash
    ///
    /// # Returns
    ///
    /// The matching block signature if found, or `None` if no match.
    #[must_use]
    pub fn find_match(&self, weak: u32, data: &[u8]) -> Option<&BlockSignature> {
        let candidates = self.weak_index.get(&weak)?;
        let strong = StrongHash::compute(data);

        candidates
            .iter()
            .map(|&i| &self.signature.blocks[i])
            .find(|sig| sig.strong_hash == strong)
    }

    /// Find a matching block with optimized strong hash computation.
    ///
    /// For sequential matching (checking block at position `expected_index`),
    /// checks that index first and skips strong hash if it's the only candidate.
    /// This provides a significant speedup for mostly-identical files.
    #[must_use]
    pub fn find_match_optimized(
        &self,
        weak: u32,
        data: &[u8],
        expected_index: u32,
    ) -> Option<&BlockSignature> {
        let candidates = self.weak_index.get(&weak)?;

        // Fast path: if there's only one candidate and it's the expected one,
        // verify with strong hash (can't skip entirely for correctness)
        if candidates.len() == 1 {
            let sig = &self.signature.blocks[candidates[0]];
            if sig.index == expected_index {
                // Still verify, but this is the common case
                let strong = StrongHash::compute(data);
                if sig.strong_hash == strong {
                    return Some(sig);
                }
            }
        }

        // Check expected index first if present
        for &i in candidates {
            let sig = &self.signature.blocks[i];
            if sig.index == expected_index {
                let strong = StrongHash::compute(data);
                if sig.strong_hash == strong {
                    return Some(sig);
                }
                // Expected didn't match, check others
                break;
            }
        }

        // Fall back to checking all candidates
        let strong = StrongHash::compute(data);
        candidates
            .iter()
            .map(|&i| &self.signature.blocks[i])
            .find(|sig| sig.strong_hash == strong)
    }

    /// Find a matching block using only weak hash for sequential matching.
    ///
    /// Returns the block index if weak hash matches and the expected index is found.
    /// This is safe for sequential matching where final checksum verification is used.
    #[must_use]
    pub fn find_weak_match(&self, weak: u32, expected_index: u32) -> Option<u32> {
        let candidates = self.weak_index.get(&weak)?;

        for &i in candidates {
            let sig = &self.signature.blocks[i];
            if sig.index == expected_index {
                return Some(sig.index);
            }
        }
        None
    }

    /// Find a matching block using only strong hash (slower but definitive).
    ///
    /// # Arguments
    ///
    /// * `data` - Data to find match for
    #[must_use]
    pub fn find_match_strong(&self, data: &[u8]) -> Option<&BlockSignature> {
        let strong = StrongHash::compute(data);
        self.signature
            .blocks
            .iter()
            .find(|sig| sig.strong_hash == strong)
    }

    /// Check if a weak hash has any candidates.
    #[must_use]
    pub fn has_weak_match(&self, weak: u32) -> bool {
        self.weak_index.contains_key(&weak)
    }

    /// Get the number of weak hash buckets.
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.weak_index.len()
    }

    /// Get the underlying signature.
    #[must_use]
    pub const fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Get the block size.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.signature.block_size
    }

    /// Get the file size.
    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.signature.file_size
    }

    /// Get block count.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.signature.blocks.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signature.blocks.is_empty()
    }

    /// Validate that the block size is within acceptable bounds.
    ///
    /// # Errors
    ///
    /// Returns `InvalidBlockSize` if block size is invalid.
    pub fn validate_block_size(block_size: usize) -> Result<()> {
        if !(512..=65536).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(CopiaError::InvalidBlockSize(block_size));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ==========================================================================
    // BLOCK SIGNATURE TESTS
    // ==========================================================================

    #[test]
    fn block_signature_new() {
        let sig = BlockSignature::new(0, 12345, StrongHash::zero());
        assert_eq!(sig.index, 0);
        assert_eq!(sig.weak_hash, 12345);
        assert_eq!(sig.strong_hash, StrongHash::zero());
    }

    #[test]
    fn block_signature_compute() {
        let data = b"test block data";
        let sig = BlockSignature::compute(5, data);

        assert_eq!(sig.index, 5);
        assert_eq!(sig.weak_hash, RollingChecksum::new(data).digest());
        assert_eq!(sig.strong_hash, StrongHash::compute(data));
    }

    #[test]
    fn block_signature_compute_empty() {
        let sig = BlockSignature::compute(0, b"");
        assert_eq!(sig.index, 0);
        assert_eq!(sig.weak_hash, 0); // Empty data has zero checksum
    }

    #[test]
    fn block_signature_deterministic() {
        let data = b"consistent data";
        let sig1 = BlockSignature::compute(0, data);
        let sig2 = BlockSignature::compute(0, data);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn block_signature_different_data() {
        let sig1 = BlockSignature::compute(0, b"data1");
        let sig2 = BlockSignature::compute(0, b"data2");
        assert_ne!(sig1.weak_hash, sig2.weak_hash);
        assert_ne!(sig1.strong_hash, sig2.strong_hash);
    }

    #[test]
    fn block_signature_serde_roundtrip() {
        let sig = BlockSignature::compute(42, b"serialization test");
        let serialized = bincode::serialize(&sig).unwrap();
        let deserialized: BlockSignature = bincode::deserialize(&serialized).unwrap();
        assert_eq!(sig, deserialized);
    }

    // ==========================================================================
    // SIGNATURE TESTS
    // ==========================================================================

    #[test]
    fn signature_new() {
        let sig = Signature::new(1024, 0);
        assert_eq!(sig.block_size, 1024);
        assert_eq!(sig.file_size, 0);
        assert!(sig.is_empty());
    }

    #[test]
    fn signature_generate_empty() {
        let data: &[u8] = b"";
        let mut cursor = Cursor::new(data);
        let sig = Signature::generate(&mut cursor, 1024).unwrap();

        assert_eq!(sig.block_size, 1024);
        assert_eq!(sig.file_size, 0);
        assert_eq!(sig.block_count(), 0);
        assert!(sig.is_empty());
    }

    #[test]
    fn signature_generate_single_block() {
        let data = b"small data";
        let mut cursor = Cursor::new(data.as_slice());
        let sig = Signature::generate(&mut cursor, 1024).unwrap();

        assert_eq!(sig.block_size, 1024);
        assert_eq!(sig.file_size, data.len() as u64);
        assert_eq!(sig.block_count(), 1);
        assert_eq!(sig.blocks[0].index, 0);
    }

    #[test]
    fn signature_generate_multiple_blocks() {
        let data = vec![42u8; 3000]; // Will create 3 blocks with block_size=1024
        let mut cursor = Cursor::new(data.as_slice());
        let sig = Signature::generate(&mut cursor, 1024).unwrap();

        assert_eq!(sig.block_size, 1024);
        assert_eq!(sig.file_size, 3000);
        assert_eq!(sig.block_count(), 3);
        assert_eq!(sig.blocks[0].index, 0);
        assert_eq!(sig.blocks[1].index, 1);
        assert_eq!(sig.blocks[2].index, 2);
    }

    #[test]
    fn signature_generate_exact_block_boundary() {
        let data = vec![0u8; 2048]; // Exactly 2 blocks
        let mut cursor = Cursor::new(data.as_slice());
        let sig = Signature::generate(&mut cursor, 1024).unwrap();

        assert_eq!(sig.block_count(), 2);
        assert_eq!(sig.file_size, 2048);
    }

    #[test]
    fn signature_deterministic() {
        let data = b"reproducible signature generation";
        let sig1 = Signature::generate(&mut Cursor::new(data.as_slice()), 512).unwrap();
        let sig2 = Signature::generate(&mut Cursor::new(data.as_slice()), 512).unwrap();

        assert_eq!(sig1.block_size, sig2.block_size);
        assert_eq!(sig1.file_size, sig2.file_size);
        assert_eq!(sig1.blocks.len(), sig2.blocks.len());
        for (b1, b2) in sig1.blocks.iter().zip(sig2.blocks.iter()) {
            assert_eq!(b1, b2);
        }
    }

    #[test]
    fn signature_into_table() {
        let data = vec![42u8; 2048];
        let mut cursor = Cursor::new(data.as_slice());
        let sig = Signature::generate(&mut cursor, 1024).unwrap();
        let block_count = sig.block_count();

        let table = sig.into_table();
        assert_eq!(table.block_count(), block_count);
    }

    #[test]
    fn signature_serde_roundtrip() {
        let data = vec![1, 2, 3, 4, 5];
        let mut cursor = Cursor::new(data.as_slice());
        let original = Signature::generate(&mut cursor, 512).unwrap();

        let serialized = bincode::serialize(&original).unwrap();
        let deserialized: Signature = bincode::deserialize(&serialized).unwrap();

        assert_eq!(original.block_size, deserialized.block_size);
        assert_eq!(original.file_size, deserialized.file_size);
        assert_eq!(original.blocks.len(), deserialized.blocks.len());
    }

    // ==========================================================================
    // SIGNATURE TABLE TESTS
    // ==========================================================================

    #[test]
    fn signature_table_build_empty() {
        let data: &[u8] = b"";
        let mut cursor = Cursor::new(data);
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        assert!(table.is_empty());
        assert_eq!(table.block_count(), 0);
        assert_eq!(table.bucket_count(), 0);
    }

    #[test]
    fn signature_table_build_single_block() {
        let data = b"single block";
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        assert_eq!(table.block_count(), 1);
        assert_eq!(table.block_size(), 1024);
        assert_eq!(table.file_size(), data.len() as u64);
    }

    #[test]
    fn signature_table_find_match_exists() {
        let data = b"block data for matching";
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        let weak = RollingChecksum::new(data).digest();
        let result = table.find_match(weak, data);

        assert!(result.is_some());
        let sig = result.unwrap();
        assert_eq!(sig.index, 0);
    }

    #[test]
    fn signature_table_find_match_not_exists() {
        let data = b"original block";
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        let other_data = b"different data!";
        let weak = RollingChecksum::new(other_data).digest();
        let result = table.find_match(weak, other_data);

        assert!(result.is_none());
    }

    #[test]
    fn signature_table_find_match_weak_collision() {
        // Create data that might have weak hash collision
        let block1 = vec![1u8; 100];
        let block2 = vec![2u8; 100];

        let mut data = block1.clone();
        data.extend(&block2);

        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 100).unwrap();

        // Try to find block1
        let weak = RollingChecksum::new(&block1).digest();
        let result = table.find_match(weak, &block1);

        assert!(result.is_some());
        assert_eq!(result.unwrap().index, 0);
    }

    #[test]
    fn signature_table_has_weak_match() {
        let data = b"test data";
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        let weak = RollingChecksum::new(data).digest();
        assert!(table.has_weak_match(weak));
        assert!(!table.has_weak_match(weak.wrapping_add(1)));
    }

    #[test]
    fn signature_table_find_match_strong() {
        let data = b"block for strong match";
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        let result = table.find_match_strong(data);
        assert!(result.is_some());

        let no_match = table.find_match_strong(b"different");
        assert!(no_match.is_none());
    }

    #[test]
    fn signature_table_multiple_blocks_same_hash() {
        // All zero blocks will have the same weak hash
        let data = vec![0u8; 2048];
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        // Both blocks have weak_hash = 0
        let weak = RollingChecksum::new(&[0u8; 1024]).digest();
        assert_eq!(weak, 0);

        // Should find one of the blocks
        let result = table.find_match(weak, &[0u8; 1024]);
        assert!(result.is_some());
    }

    #[test]
    fn signature_table_validate_block_size_valid() {
        assert!(SignatureTable::validate_block_size(512).is_ok());
        assert!(SignatureTable::validate_block_size(1024).is_ok());
        assert!(SignatureTable::validate_block_size(2048).is_ok());
        assert!(SignatureTable::validate_block_size(4096).is_ok());
        assert!(SignatureTable::validate_block_size(65536).is_ok());
    }

    #[test]
    fn signature_table_validate_block_size_invalid() {
        // Too small
        assert!(SignatureTable::validate_block_size(256).is_err());
        assert!(SignatureTable::validate_block_size(0).is_err());

        // Too large
        assert!(SignatureTable::validate_block_size(131072).is_err());

        // Not power of 2
        assert!(SignatureTable::validate_block_size(1000).is_err());
        assert!(SignatureTable::validate_block_size(1023).is_err());
        assert!(SignatureTable::validate_block_size(1025).is_err());
    }

    // ==========================================================================
    // EDGE CASES
    // ==========================================================================

    #[test]
    fn signature_table_large_file() {
        let data = vec![42u8; 100_000];
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 1024).unwrap();

        // 100_000 / 1024 = 97.65... so 98 blocks
        assert_eq!(table.block_count(), 98);
        assert_eq!(table.file_size(), 100_000);
    }

    #[test]
    fn signature_table_binary_data() {
        let data: Vec<u8> = (0..=255).cycle().take(5000).collect();
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 512).unwrap();

        assert_eq!(table.block_count(), 10);
    }

    #[test]
    fn signature_table_getters() {
        let data = vec![0u8; 4096];
        let mut cursor = Cursor::new(data.as_slice());
        let table = SignatureTable::build(&mut cursor, 2048).unwrap();

        assert_eq!(table.block_size(), 2048);
        assert_eq!(table.file_size(), 4096);
        assert_eq!(table.block_count(), 2);
        assert!(!table.is_empty());

        let sig = table.signature();
        assert_eq!(sig.block_size, 2048);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    proptest! {
        /// Generated signatures are deterministic
        #[test]
        fn signature_deterministic(
            data in prop::collection::vec(any::<u8>(), 0..5000),
            block_size in prop::sample::select(vec![512usize, 1024, 2048, 4096])
        ) {
            let sig1 = Signature::generate(&mut Cursor::new(&data), block_size).unwrap();
            let sig2 = Signature::generate(&mut Cursor::new(&data), block_size).unwrap();

            prop_assert_eq!(sig1.block_count(), sig2.block_count());
            prop_assert_eq!(sig1.file_size, sig2.file_size);

            for (b1, b2) in sig1.blocks.iter().zip(sig2.blocks.iter()) {
                prop_assert_eq!(b1.weak_hash, b2.weak_hash);
                prop_assert_eq!(b1.strong_hash, b2.strong_hash);
            }
        }

        /// File size equals sum of data
        #[test]
        fn signature_file_size_correct(
            data in prop::collection::vec(any::<u8>(), 0..10000)
        ) {
            let sig = Signature::generate(&mut Cursor::new(&data), 1024).unwrap();
            prop_assert_eq!(sig.file_size, data.len() as u64);
        }

        /// Block count is ceiling division
        #[test]
        fn signature_block_count_correct(
            data in prop::collection::vec(any::<u8>(), 1..10000),
            block_size in prop::sample::select(vec![512usize, 1024, 2048])
        ) {
            let sig = Signature::generate(&mut Cursor::new(&data), block_size).unwrap();
            let expected = data.len().div_ceil(block_size);
            prop_assert_eq!(sig.block_count(), expected);
        }

        /// Find match returns correct block
        #[test]
        fn find_match_correct(
            data in prop::collection::vec(any::<u8>(), 512..5000)
        ) {
            let block_size = 512;
            let table = SignatureTable::build(&mut Cursor::new(&data), block_size).unwrap();

            // Try to find the first block
            let first_block = &data[..block_size.min(data.len())];
            let weak = RollingChecksum::new(first_block).digest();

            if let Some(found) = table.find_match(weak, first_block) {
                prop_assert_eq!(found.index, 0);
            }
        }

        /// Serde roundtrip preserves data
        #[test]
        fn signature_serde_preserves(
            data in prop::collection::vec(any::<u8>(), 0..2000)
        ) {
            let original = Signature::generate(&mut Cursor::new(&data), 512).unwrap();
            let serialized = bincode::serialize(&original).unwrap();
            let restored: Signature = bincode::deserialize(&serialized).unwrap();

            prop_assert_eq!(original.block_size, restored.block_size);
            prop_assert_eq!(original.file_size, restored.file_size);
            prop_assert_eq!(original.blocks.len(), restored.blocks.len());
        }
    }
}
