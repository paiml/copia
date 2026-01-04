//! Core synchronization engine for rsync-style delta transfers.
//!
//! This module provides the main `Sync` trait and `CopiaSync` implementation
//! for generating signatures, computing deltas, and applying patches.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::checksum::FastRollingChecksum;
use crate::delta::{Delta, DeltaOp};
use crate::error::{CopiaError, Result};
use crate::hash::StrongHash;
use crate::signature::{Signature, SignatureTable};

/// Core synchronization operations trait.
///
/// This trait defines the three fundamental rsync operations:
/// 1. **Signature**: Generate block signatures from a basis file
/// 2. **Delta**: Compute differences between source and basis
/// 3. **Patch**: Apply delta to basis to reconstruct source
pub trait Sync {
    /// Generate signature from basis file.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    fn signature<R: Read>(&self, basis: R) -> Result<Signature>;

    /// Compute delta between source and signature.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or signature is invalid.
    fn delta<R: Read>(&self, source: R, signature: &Signature) -> Result<Delta>;

    /// Apply delta to basis file, producing output.
    ///
    /// # Errors
    ///
    /// Returns an error if reading/writing fails or delta is invalid.
    fn patch<R: Read + Seek, W: Write>(&self, basis: R, delta: &Delta, output: W) -> Result<()>;
}

/// Configuration for sync operations.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Block size for signature generation (must be power of 2, 512-65536).
    pub block_size: usize,
    /// Strong hash length for signature comparison (4-32 bytes).
    pub strong_hash_len: usize,
    /// I/O buffer size.
    pub buffer_size: usize,
    /// Verify checksum after patch.
    pub verify_checksum: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            block_size: 2048,
            strong_hash_len: 8,
            buffer_size: 64 * 1024,
            verify_checksum: true,
        }
    }
}

/// Builder for creating sync engines with custom configuration.
///
/// # Example
///
/// ```rust
/// use copia::SyncBuilder;
///
/// let sync = SyncBuilder::new()
///     .block_size(4096)
///     .buffer_size(128 * 1024)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct SyncBuilder {
    config: SyncConfig,
}

impl SyncBuilder {
    /// Create a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SyncConfig::default(),
        }
    }

    /// Set the block size for signature generation.
    ///
    /// Must be a power of 2 between 512 and 65536.
    ///
    /// # Panics
    ///
    /// Panics if block size is invalid.
    #[must_use]
    pub fn block_size(mut self, size: usize) -> Self {
        assert!(
            size.is_power_of_two() && (512..=65536).contains(&size),
            "Block size must be power of 2, 512-65536"
        );
        self.config.block_size = size;
        self
    }

    /// Set the strong hash length for comparisons.
    ///
    /// Must be between 4 and 32 bytes.
    ///
    /// # Panics
    ///
    /// Panics if length is invalid.
    #[must_use]
    pub fn strong_hash_len(mut self, len: usize) -> Self {
        assert!((4..=32).contains(&len), "Hash length must be 4-32");
        self.config.strong_hash_len = len;
        self
    }

    /// Set the I/O buffer size.
    #[must_use]
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Enable or disable checksum verification after patch.
    #[must_use]
    pub fn verify_checksum(mut self, verify: bool) -> Self {
        self.config.verify_checksum = verify;
        self
    }

    /// Build the sync engine.
    #[must_use]
    pub fn build(self) -> CopiaSync {
        CopiaSync {
            config: self.config,
        }
    }
}

impl Default for SyncBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main synchronization engine implementing the rsync algorithm.
///
/// `CopiaSync` provides the core functionality for computing and applying
/// file deltas using the rsync algorithm.
#[derive(Debug, Clone)]
pub struct CopiaSync {
    config: SyncConfig,
}

impl CopiaSync {
    /// Create a new sync engine with default configuration.
    #[must_use]
    pub fn new() -> Self {
        SyncBuilder::new().build()
    }

    /// Create a sync engine with custom block size.
    ///
    /// # Panics
    ///
    /// Panics if block size is invalid.
    #[must_use]
    pub fn with_block_size(block_size: usize) -> Self {
        SyncBuilder::new().block_size(block_size).build()
    }

    /// Get the configured block size.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &SyncConfig {
        &self.config
    }
}

impl Default for CopiaSync {
    fn default() -> Self {
        Self::new()
    }
}

impl Sync for CopiaSync {
    fn signature<R: Read>(&self, mut basis: R) -> Result<Signature> {
        Signature::generate(&mut basis, self.config.block_size)
    }

    fn delta<R: Read>(&self, mut source: R, signature: &Signature) -> Result<Delta> {
        let table = SignatureTable::from_signature(signature.clone());
        let block_size = signature.block_size;

        // Read entire source into memory
        let mut source_data = Vec::new();
        source.read_to_end(&mut source_data)?;

        let source_size = source_data.len() as u64;
        let source_hash = StrongHash::compute(&source_data);

        #[allow(clippy::cast_possible_truncation)]
        let mut delta = Delta::with_checksum(
            block_size as u32,
            source_size,
            signature.file_size,
            source_hash,
        );

        if source_data.is_empty() {
            return Ok(delta);
        }

        // Empty signature means all data is literal
        if table.is_empty() {
            delta.push_literal(&source_data);
            return Ok(delta);
        }

        let mut pos = 0usize;

        // Initialize rolling checksum with first block
        let init_len = block_size.min(source_data.len());
        let mut rolling = FastRollingChecksum::new(&source_data[..init_len]);

        while pos + block_size <= source_data.len() {
            let weak = rolling.digest();

            // Fast path: check weak hash first before computing strong hash
            if table.has_weak_match(weak) {
                let block_data = &source_data[pos..pos + block_size];
                if let Some(sig) = table.find_match(weak, block_data) {
                    // Found a match - emit copy operation
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        let offset = u64::from(sig.index) * block_size as u64;
                        delta.push_copy(offset, block_size as u32);
                    }
                    pos += block_size;

                    // After match, we need new checksum at new position
                    // Roll forward by block_size bytes OR reinit (same cost)
                    if pos + block_size <= source_data.len() {
                        rolling = FastRollingChecksum::new(&source_data[pos..pos + block_size]);
                    }
                    continue;
                }
            }

            // No match - emit literal byte and roll window
            delta.push_literal_byte(source_data[pos]);

            if pos + block_size < source_data.len() {
                rolling.roll(source_data[pos], source_data[pos + block_size]);
            }
            pos += 1;
        }

        // Handle remaining bytes as literals
        if pos < source_data.len() {
            delta.push_literal(&source_data[pos..]);
        }

        Ok(delta)
    }

    fn patch<R: Read + Seek, W: Write>(
        &self,
        mut basis: R,
        delta: &Delta,
        mut output: W,
    ) -> Result<()> {
        // Validate delta first
        delta.validate()?;

        let mut hasher = blake3::Hasher::new();

        for op in &delta.ops {
            match op {
                DeltaOp::Copy { offset, len } => {
                    basis.seek(SeekFrom::Start(*offset))?;
                    let mut buffer = vec![0u8; *len as usize];
                    basis.read_exact(&mut buffer)?;
                    output.write_all(&buffer)?;
                    hasher.update(&buffer);
                }
                DeltaOp::Literal(data) => {
                    output.write_all(data)?;
                    hasher.update(data);
                }
            }
        }

        // Verify checksum if enabled
        if self.config.verify_checksum {
            let computed = StrongHash::from_bytes(*hasher.finalize().as_bytes());
            if computed != delta.checksum {
                return Err(CopiaError::ChecksumMismatch {
                    expected: *delta.checksum.as_bytes(),
                    actual: *computed.as_bytes(),
                });
            }
        }

        Ok(())
    }
}

/// Statistics from a sync operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyncStats {
    /// Bytes copied from basis file.
    pub bytes_matched: u64,
    /// Literal bytes transmitted.
    pub bytes_literal: u64,
    /// Compression ratio (0.0-1.0).
    pub compression_ratio: f64,
}

impl SyncStats {
    /// Create stats from a delta.
    #[must_use]
    pub fn from_delta(delta: &Delta) -> Self {
        Self {
            bytes_matched: delta.bytes_matched(),
            bytes_literal: delta.bytes_literal(),
            compression_ratio: delta.compression_ratio(delta.source_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ==========================================================================
    // SYNC BUILDER TESTS
    // ==========================================================================

    #[test]
    fn builder_default() {
        let sync = SyncBuilder::new().build();
        assert_eq!(sync.block_size(), 2048);
    }

    #[test]
    fn builder_block_size() {
        let sync = SyncBuilder::new().block_size(4096).build();
        assert_eq!(sync.block_size(), 4096);
    }

    #[test]
    fn builder_all_options() {
        let sync = SyncBuilder::new()
            .block_size(1024)
            .strong_hash_len(16)
            .buffer_size(32 * 1024)
            .verify_checksum(false)
            .build();

        assert_eq!(sync.config().block_size, 1024);
        assert_eq!(sync.config().strong_hash_len, 16);
        assert_eq!(sync.config().buffer_size, 32 * 1024);
        assert!(!sync.config().verify_checksum);
    }

    #[test]
    #[should_panic(expected = "Block size must be power of 2")]
    fn builder_invalid_block_size_not_power_of_2() {
        let _ = SyncBuilder::new().block_size(1000);
    }

    #[test]
    #[should_panic(expected = "Block size must be power of 2")]
    fn builder_invalid_block_size_too_small() {
        let _ = SyncBuilder::new().block_size(256);
    }

    #[test]
    #[should_panic(expected = "Block size must be power of 2")]
    fn builder_invalid_block_size_too_large() {
        let _ = SyncBuilder::new().block_size(131072);
    }

    #[test]
    #[should_panic(expected = "Hash length must be 4-32")]
    fn builder_invalid_hash_length_small() {
        let _ = SyncBuilder::new().strong_hash_len(2);
    }

    #[test]
    #[should_panic(expected = "Hash length must be 4-32")]
    fn builder_invalid_hash_length_large() {
        let _ = SyncBuilder::new().strong_hash_len(64);
    }

    // ==========================================================================
    // COPIA SYNC TESTS
    // ==========================================================================

    #[test]
    fn sync_new() {
        let sync = CopiaSync::new();
        assert_eq!(sync.block_size(), 2048);
    }

    #[test]
    fn sync_with_block_size() {
        let sync = CopiaSync::with_block_size(4096);
        assert_eq!(sync.block_size(), 4096);
    }

    #[test]
    fn sync_default() {
        let sync = CopiaSync::default();
        assert_eq!(sync.block_size(), 2048);
    }

    // ==========================================================================
    // SIGNATURE TESTS
    // ==========================================================================

    #[test]
    fn signature_empty() {
        let sync = CopiaSync::new();
        let data: &[u8] = b"";
        let sig = sync.signature(Cursor::new(data)).unwrap();

        assert!(sig.is_empty());
        assert_eq!(sig.file_size, 0);
    }

    #[test]
    fn signature_small_file() {
        let sync = CopiaSync::with_block_size(512);
        let data = b"small file content";
        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();

        assert_eq!(sig.block_count(), 1);
        assert_eq!(sig.file_size, data.len() as u64);
    }

    #[test]
    fn signature_multiple_blocks() {
        let sync = CopiaSync::with_block_size(512);
        let data = vec![42u8; 2000];
        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();

        assert_eq!(sig.block_count(), 4); // 2000 / 512 = 3.9... -> 4 blocks
    }

    // ==========================================================================
    // DELTA TESTS
    // ==========================================================================

    #[test]
    fn delta_identical_files() {
        let sync = CopiaSync::with_block_size(512);
        let data = vec![42u8; 1024];

        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(data.as_slice()), &sig).unwrap();

        assert_eq!(delta.bytes_matched(), 1024);
        assert_eq!(delta.bytes_literal(), 0);
        assert!((delta.compression_ratio(1024) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_completely_different() {
        let sync = CopiaSync::with_block_size(512);
        let basis = vec![0u8; 1024];
        let source = vec![1u8; 1024];

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        assert_eq!(delta.bytes_matched(), 0);
        assert_eq!(delta.bytes_literal(), 1024);
    }

    #[test]
    fn delta_empty_basis() {
        let sync = CopiaSync::with_block_size(512);
        let basis: &[u8] = b"";
        let source = b"new content";

        let sig = sync.signature(Cursor::new(basis)).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        assert_eq!(delta.bytes_literal(), source.len() as u64);
        assert_eq!(delta.bytes_matched(), 0);
    }

    #[test]
    fn delta_empty_source() {
        let sync = CopiaSync::with_block_size(512);
        let basis = b"existing content";
        let source: &[u8] = b"";

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source), &sig).unwrap();

        assert!(delta.is_empty());
        assert_eq!(delta.source_size, 0);
    }

    #[test]
    fn delta_partial_match() {
        let sync = CopiaSync::with_block_size(512);

        // Basis: [block1][block2]
        let mut basis = vec![1u8; 512];
        basis.extend(vec![2u8; 512]);

        // Source: [block1][block3] - first block matches
        let mut source = vec![1u8; 512];
        source.extend(vec![3u8; 512]);

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        // First block should match
        assert!(delta.bytes_matched() >= 512);
    }

    // ==========================================================================
    // PATCH TESTS
    // ==========================================================================

    #[test]
    fn patch_roundtrip() {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(true)
            .build();

        let basis = vec![42u8; 1024];
        let source = vec![42u8; 1024];

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        assert_eq!(output, source);
    }

    #[test]
    fn patch_modified_file() {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(true)
            .build();

        let basis = b"Hello, World! This is a test file for rsync.".to_vec();
        let source = b"Hello, Universe! This is a test file for rsync.".to_vec();

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        assert_eq!(output, source);
    }

    #[test]
    fn patch_large_file() {
        let sync = SyncBuilder::new()
            .block_size(1024)
            .verify_checksum(true)
            .build();

        // Create basis with pattern
        let basis: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

        // Modify some parts
        let mut source = basis.clone();
        for i in (0..source.len()).step_by(1000) {
            source[i] = 0xFF;
        }

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        assert_eq!(output, source);
    }

    #[test]
    fn patch_empty_to_content() {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(true)
            .build();

        let basis: &[u8] = b"";
        let source = b"New content created from empty basis";

        let sig = sync.signature(Cursor::new(basis)).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis), &delta, &mut output).unwrap();

        assert_eq!(output.as_slice(), source);
    }

    #[test]
    fn patch_content_to_empty() {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(true)
            .build();

        let basis = b"Existing content to be replaced";
        let source: &[u8] = b"";

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        assert!(output.is_empty());
    }

    #[test]
    fn patch_checksum_verification() {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(true)
            .build();

        let basis = b"basis content";
        let source = b"source content";

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let mut delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        // Corrupt the checksum
        delta.checksum = StrongHash::zero();

        let mut output = Vec::new();
        let result = sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            CopiaError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn patch_no_verification() {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(false)
            .build();

        let basis = b"basis content";
        let source = b"source content";

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let mut delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        // Corrupt the checksum - should still succeed without verification
        delta.checksum = StrongHash::zero();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        // Output should still be correct
        assert_eq!(output.as_slice(), source);
    }

    // ==========================================================================
    // SYNC STATS TESTS
    // ==========================================================================

    #[test]
    fn sync_stats_from_delta() {
        let sync = CopiaSync::with_block_size(512);
        let data = vec![42u8; 1024];

        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(data.as_slice()), &sig).unwrap();

        let stats = SyncStats::from_delta(&delta);
        assert_eq!(stats.bytes_matched, 1024);
        assert_eq!(stats.bytes_literal, 0);
        assert!((stats.compression_ratio - 1.0).abs() < f64::EPSILON);
    }

    // ==========================================================================
    // EDGE CASES
    // ==========================================================================

    #[test]
    fn sync_binary_data() {
        let sync = CopiaSync::with_block_size(512);

        let basis: Vec<u8> = (0..=255).cycle().take(2000).collect();
        let mut source = basis.clone();
        source[500] = 0xFF;
        source[1500] = 0xFF;

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();

        assert_eq!(output, source);
    }

    #[test]
    fn sync_repeated_blocks() {
        let sync = CopiaSync::with_block_size(512);

        // Basis with repeated pattern
        let block = vec![42u8; 512];
        let mut basis = block.clone();
        basis.extend(&block);
        basis.extend(&block);

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(basis.as_slice()), &sig).unwrap();

        // Should match all blocks
        assert_eq!(delta.bytes_matched(), 1536);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    proptest! {
        /// Roundtrip: patch(basis, delta(source, sig(basis))) == source
        #[test]
        fn roundtrip(
            basis in prop::collection::vec(any::<u8>(), 0..5000),
            source in prop::collection::vec(any::<u8>(), 0..5000)
        ) {
            let sync = SyncBuilder::new()
                .block_size(512)
                .verify_checksum(true)
                .build();

            let sig = sync.signature(Cursor::new(&basis)).unwrap();
            let delta = sync.delta(Cursor::new(&source), &sig).unwrap();

            let mut output = Vec::new();
            sync.patch(Cursor::new(&basis), &delta, &mut output).unwrap();

            prop_assert_eq!(output, source);
        }

        /// Identical block-aligned files produce high compression ratio
        #[test]
        fn identical_high_ratio(
            block_count in 1usize..10,
            byte_val in any::<u8>()
        ) {
            let block_size = 512;
            let data = vec![byte_val; block_count * block_size];
            let sync = CopiaSync::with_block_size(block_size);

            let sig = sync.signature(Cursor::new(&data)).unwrap();
            let delta = sync.delta(Cursor::new(&data), &sig).unwrap();

            let ratio = delta.compression_ratio(data.len() as u64);
            // Block-aligned identical data should match perfectly
            prop_assert!(ratio >= 0.99, "Expected high ratio for block-aligned identical files, got {}", ratio);
        }

        /// Delta output size is source size
        #[test]
        fn delta_produces_source_size(
            basis in prop::collection::vec(any::<u8>(), 0..2000),
            source in prop::collection::vec(any::<u8>(), 0..2000)
        ) {
            let sync = CopiaSync::with_block_size(512);

            let sig = sync.signature(Cursor::new(&basis)).unwrap();
            let delta = sync.delta(Cursor::new(&source), &sig).unwrap();

            let expected_size = delta.expected_output_size();
            prop_assert_eq!(expected_size, source.len() as u64);
        }

        /// Signature block count is correct
        #[test]
        fn signature_block_count(
            data in prop::collection::vec(any::<u8>(), 1..10000),
            block_size in prop::sample::select(vec![512usize, 1024, 2048, 4096])
        ) {
            let sync = CopiaSync::with_block_size(block_size);
            let sig = sync.signature(Cursor::new(&data)).unwrap();

            let expected = (data.len() + block_size - 1) / block_size;
            prop_assert_eq!(sig.block_count(), expected);
        }
    }
}
