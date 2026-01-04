//! Delta encoding and decoding for rsync-style synchronization.
//!
//! A delta represents the difference between a source file and a basis file,
//! expressed as a sequence of copy and literal operations.

use serde::{Deserialize, Serialize};

use crate::error::{CopiaError, Result};
use crate::hash::StrongHash;

/// Delta instruction types.
///
/// The rsync algorithm produces deltas consisting of two types of operations:
/// - **Copy**: Copy bytes from the basis file at a given offset
/// - **Literal**: Insert new bytes directly
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaOp {
    /// Copy `len` bytes from basis file starting at `offset`.
    Copy {
        /// Byte offset in the basis file.
        offset: u64,
        /// Number of bytes to copy.
        len: u32,
    },
    /// Insert literal bytes directly.
    Literal(Vec<u8>),
}

impl DeltaOp {
    /// Create a new copy operation.
    #[must_use]
    pub const fn copy(offset: u64, len: u32) -> Self {
        Self::Copy { offset, len }
    }

    /// Create a new literal operation.
    #[must_use]
    pub fn literal(data: Vec<u8>) -> Self {
        Self::Literal(data)
    }

    /// Create a literal from a slice.
    #[must_use]
    pub fn literal_from_slice(data: &[u8]) -> Self {
        Self::Literal(data.to_vec())
    }

    /// Check if this is a copy operation.
    #[must_use]
    pub const fn is_copy(&self) -> bool {
        matches!(self, Self::Copy { .. })
    }

    /// Check if this is a literal operation.
    #[must_use]
    pub const fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }

    /// Get the number of bytes this operation produces.
    #[must_use]
    pub fn output_len(&self) -> u64 {
        match self {
            Self::Copy { len, .. } => u64::from(*len),
            Self::Literal(data) => data.len() as u64,
        }
    }

    /// Get the number of bytes transmitted (for bandwidth calculation).
    #[must_use]
    pub fn transmission_size(&self) -> usize {
        match self {
            // Copy: 8 bytes offset + 4 bytes len + 1 byte tag
            Self::Copy { .. } => 13,
            // Literal: 4 bytes len + data + 1 byte tag
            Self::Literal(data) => 5 + data.len(),
        }
    }
}

/// Encoded delta representing the difference between source and basis files.
///
/// The delta can be applied to a basis file to reconstruct the source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    /// Block size used during delta computation.
    pub block_size: u32,
    /// Total size of the source file.
    pub source_size: u64,
    /// Total size of the basis file.
    pub basis_size: u64,
    /// Sequence of delta operations.
    pub ops: Vec<DeltaOp>,
    /// Checksum of the expected output (source file).
    pub checksum: StrongHash,
}

impl Delta {
    /// Create a new empty delta.
    #[must_use]
    pub fn new(block_size: u32, source_size: u64, basis_size: u64) -> Self {
        Self {
            block_size,
            source_size,
            basis_size,
            ops: Vec::new(),
            checksum: StrongHash::zero(),
        }
    }

    /// Create a delta with a checksum.
    #[must_use]
    pub fn with_checksum(
        block_size: u32,
        source_size: u64,
        basis_size: u64,
        checksum: StrongHash,
    ) -> Self {
        Self {
            block_size,
            source_size,
            basis_size,
            ops: Vec::new(),
            checksum,
        }
    }

    /// Add a copy operation.
    pub fn push_copy(&mut self, offset: u64, len: u32) {
        // Try to merge with previous copy if contiguous
        if let Some(DeltaOp::Copy {
            offset: prev_offset,
            len: prev_len,
        }) = self.ops.last_mut()
        {
            if *prev_offset + u64::from(*prev_len) == offset {
                // Contiguous: merge
                if let Some(new_len) = prev_len.checked_add(len) {
                    *prev_len = new_len;
                    return;
                }
            }
        }
        self.ops.push(DeltaOp::copy(offset, len));
    }

    /// Add a literal operation.
    pub fn push_literal(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // Try to merge with previous literal
        if let Some(DeltaOp::Literal(prev_data)) = self.ops.last_mut() {
            prev_data.extend_from_slice(data);
            return;
        }
        self.ops.push(DeltaOp::literal_from_slice(data));
    }

    /// Add a single literal byte.
    pub fn push_literal_byte(&mut self, byte: u8) {
        if let Some(DeltaOp::Literal(prev_data)) = self.ops.last_mut() {
            prev_data.push(byte);
            return;
        }
        self.ops.push(DeltaOp::literal(vec![byte]));
    }

    /// Get the number of operations.
    #[must_use]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Check if the delta is empty (no operations).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Calculate total bytes copied from basis.
    #[must_use]
    pub fn bytes_matched(&self) -> u64 {
        self.ops
            .iter()
            .filter_map(|op| match op {
                DeltaOp::Copy { len, .. } => Some(u64::from(*len)),
                DeltaOp::Literal(_) => None,
            })
            .sum()
    }

    /// Calculate total literal bytes.
    #[must_use]
    pub fn bytes_literal(&self) -> u64 {
        self.ops
            .iter()
            .filter_map(|op| match op {
                DeltaOp::Literal(data) => Some(data.len() as u64),
                DeltaOp::Copy { .. } => None,
            })
            .sum()
    }

    /// Calculate compression ratio.
    ///
    /// Returns a value between 0.0 and 1.0 where higher is better.
    /// 1.0 means all data was copied (100% match).
    /// 0.0 means all data is literal (0% match).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // acceptable for ratio calculation
    pub fn compression_ratio(&self, original_size: u64) -> f64 {
        if original_size == 0 {
            return 1.0;
        }
        let matched = self.bytes_matched();
        matched as f64 / original_size as f64
    }

    /// Calculate the transmission size (delta wire size).
    #[must_use]
    pub fn transmission_size(&self) -> usize {
        // Header: block_size(4) + source_size(8) + basis_size(8) + checksum(32) + op_count(4)
        let header = 56;
        let ops: usize = self.ops.iter().map(DeltaOp::transmission_size).sum();
        header + ops
    }

    /// Validate that all copy operations are within bounds.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCopyBounds` if any copy operation exceeds basis size.
    pub fn validate(&self) -> Result<()> {
        for op in &self.ops {
            if let DeltaOp::Copy { offset, len } = op {
                let end = offset.saturating_add(u64::from(*len));
                if end > self.basis_size {
                    return Err(CopiaError::InvalidCopyBounds {
                        offset: *offset,
                        len: *len,
                        basis_size: self.basis_size,
                    });
                }
            }
        }
        Ok(())
    }

    /// Calculate expected output size from operations.
    #[must_use]
    pub fn expected_output_size(&self) -> u64 {
        self.ops.iter().map(DeltaOp::output_len).sum()
    }

    /// Get the source size.
    #[must_use]
    pub const fn source_len(&self) -> u64 {
        self.source_size
    }
}

impl Default for Delta {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

/// Statistics from delta computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaStats {
    /// Number of copy operations.
    pub copy_ops: usize,
    /// Number of literal operations.
    pub literal_ops: usize,
    /// Total bytes copied from basis.
    pub bytes_copied: u64,
    /// Total literal bytes.
    pub bytes_literal: u64,
    /// Compression ratio (0.0-1.0).
    pub ratio: f64,
}

impl DeltaStats {
    /// Compute statistics from a delta.
    #[must_use]
    pub fn from_delta(delta: &Delta) -> Self {
        let copy_ops = delta.ops.iter().filter(|op| op.is_copy()).count();
        let literal_ops = delta.ops.iter().filter(|op| op.is_literal()).count();
        let bytes_copied = delta.bytes_matched();
        let bytes_literal = delta.bytes_literal();
        let ratio = delta.compression_ratio(delta.source_size);

        Self {
            copy_ops,
            literal_ops,
            bytes_copied,
            bytes_literal,
            ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // DELTA OP TESTS
    // ==========================================================================

    #[test]
    fn delta_op_copy() {
        let op = DeltaOp::copy(100, 50);
        assert!(op.is_copy());
        assert!(!op.is_literal());
        assert_eq!(op.output_len(), 50);
    }

    #[test]
    fn delta_op_literal() {
        let op = DeltaOp::literal(vec![1, 2, 3, 4, 5]);
        assert!(op.is_literal());
        assert!(!op.is_copy());
        assert_eq!(op.output_len(), 5);
    }

    #[test]
    fn delta_op_literal_from_slice() {
        let op = DeltaOp::literal_from_slice(b"hello");
        assert!(op.is_literal());
        assert_eq!(op.output_len(), 5);
    }

    #[test]
    fn delta_op_transmission_size_copy() {
        let op = DeltaOp::copy(0, 1000);
        assert_eq!(op.transmission_size(), 13); // Fixed size for copy
    }

    #[test]
    fn delta_op_transmission_size_literal() {
        let op = DeltaOp::literal(vec![0; 100]);
        assert_eq!(op.transmission_size(), 105); // 5 + 100
    }

    #[test]
    fn delta_op_serde_copy() {
        let op = DeltaOp::copy(12345, 67890);
        let serialized = bincode::serialize(&op).unwrap();
        let restored: DeltaOp = bincode::deserialize(&serialized).unwrap();
        assert_eq!(op, restored);
    }

    #[test]
    fn delta_op_serde_literal() {
        let op = DeltaOp::literal(vec![1, 2, 3, 4, 5]);
        let serialized = bincode::serialize(&op).unwrap();
        let restored: DeltaOp = bincode::deserialize(&serialized).unwrap();
        assert_eq!(op, restored);
    }

    // ==========================================================================
    // DELTA TESTS
    // ==========================================================================

    #[test]
    fn delta_new() {
        let delta = Delta::new(1024, 5000, 4000);
        assert_eq!(delta.block_size, 1024);
        assert_eq!(delta.source_size, 5000);
        assert_eq!(delta.basis_size, 4000);
        assert!(delta.is_empty());
        assert_eq!(delta.op_count(), 0);
    }

    #[test]
    fn delta_default() {
        let delta = Delta::default();
        assert_eq!(delta.block_size, 0);
        assert!(delta.is_empty());
    }

    #[test]
    fn delta_push_copy() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 500);
        delta.push_copy(600, 200); // Non-contiguous

        assert_eq!(delta.op_count(), 2);
        assert_eq!(delta.bytes_matched(), 700);
    }

    #[test]
    fn delta_push_copy_merge_contiguous() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 500);
        delta.push_copy(500, 200); // Contiguous - should merge

        assert_eq!(delta.op_count(), 1);
        assert_eq!(delta.bytes_matched(), 700);
    }

    #[test]
    fn delta_push_literal() {
        let mut delta = Delta::new(1024, 100, 0);
        delta.push_literal(b"hello");
        delta.push_literal(b" world"); // Should merge

        assert_eq!(delta.op_count(), 1);
        assert_eq!(delta.bytes_literal(), 11);
    }

    #[test]
    fn delta_push_literal_empty() {
        let mut delta = Delta::new(1024, 100, 0);
        delta.push_literal(b"");
        assert!(delta.is_empty());
    }

    #[test]
    fn delta_push_literal_byte() {
        let mut delta = Delta::new(1024, 100, 0);
        delta.push_literal_byte(b'a');
        delta.push_literal_byte(b'b');
        delta.push_literal_byte(b'c');

        assert_eq!(delta.op_count(), 1);
        assert_eq!(delta.bytes_literal(), 3);
    }

    #[test]
    fn delta_mixed_ops() {
        let mut delta = Delta::new(1024, 1000, 800);
        delta.push_literal(b"prefix");
        delta.push_copy(0, 500);
        delta.push_literal(b"middle");
        delta.push_copy(600, 200);
        delta.push_literal(b"suffix");

        assert_eq!(delta.op_count(), 5);
        assert_eq!(delta.bytes_matched(), 700);
        assert_eq!(delta.bytes_literal(), 18);
    }

    #[test]
    fn delta_compression_ratio_full_match() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 1000);

        assert!((delta.compression_ratio(1000) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_compression_ratio_no_match() {
        let mut delta = Delta::new(1024, 1000, 500);
        delta.push_literal(&vec![0u8; 1000]);

        assert!(delta.compression_ratio(1000).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_compression_ratio_partial() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 500);
        delta.push_literal(&vec![0u8; 500]);

        assert!((delta.compression_ratio(1000) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_compression_ratio_empty() {
        let delta = Delta::new(1024, 0, 0);
        assert!((delta.compression_ratio(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_validate_valid() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 500);
        delta.push_copy(500, 500);

        assert!(delta.validate().is_ok());
    }

    #[test]
    fn delta_validate_invalid() {
        let mut delta = Delta::new(1024, 1000, 500);
        delta.push_copy(0, 600); // Exceeds basis_size of 500

        let result = delta.validate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CopiaError::InvalidCopyBounds { .. }));
    }

    #[test]
    fn delta_validate_overflow() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(900, 200); // 900 + 200 = 1100 > 1000

        assert!(delta.validate().is_err());
    }

    #[test]
    fn delta_expected_output_size() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 500);
        delta.push_literal(b"hello");
        delta.push_copy(600, 200);

        assert_eq!(delta.expected_output_size(), 705);
    }

    #[test]
    fn delta_source_len() {
        let delta = Delta::new(1024, 12345, 6789);
        assert_eq!(delta.source_len(), 12345);
    }

    #[test]
    fn delta_serde_roundtrip() {
        let mut delta = Delta::new(1024, 1000, 800);
        delta.push_copy(0, 400);
        delta.push_literal(b"inserted data");
        delta.push_copy(500, 300);

        let serialized = bincode::serialize(&delta).unwrap();
        let restored: Delta = bincode::deserialize(&serialized).unwrap();

        assert_eq!(delta.block_size, restored.block_size);
        assert_eq!(delta.source_size, restored.source_size);
        assert_eq!(delta.basis_size, restored.basis_size);
        assert_eq!(delta.ops.len(), restored.ops.len());
    }

    // ==========================================================================
    // DELTA STATS TESTS
    // ==========================================================================

    #[test]
    fn delta_stats_from_delta() {
        let mut delta = Delta::new(1024, 1000, 800);
        delta.push_copy(0, 400);
        delta.push_literal(b"data");
        delta.push_copy(500, 200);
        delta.push_literal(b"more");

        let stats = DeltaStats::from_delta(&delta);

        assert_eq!(stats.copy_ops, 2);
        assert_eq!(stats.literal_ops, 2);
        assert_eq!(stats.bytes_copied, 600);
        assert_eq!(stats.bytes_literal, 8);
        assert!((stats.ratio - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn delta_stats_empty() {
        let delta = Delta::new(1024, 0, 0);
        let stats = DeltaStats::from_delta(&delta);

        assert_eq!(stats.copy_ops, 0);
        assert_eq!(stats.literal_ops, 0);
        assert_eq!(stats.bytes_copied, 0);
        assert_eq!(stats.bytes_literal, 0);
    }

    // ==========================================================================
    // EDGE CASES
    // ==========================================================================

    #[test]
    fn delta_large_copy() {
        let mut delta = Delta::new(1024, u64::MAX, u64::MAX);
        delta.push_copy(0, u32::MAX);

        assert_eq!(delta.bytes_matched(), u64::from(u32::MAX));
    }

    #[test]
    fn delta_many_ops() {
        let mut delta = Delta::new(1024, 10000, 10000);
        for i in 0..100 {
            delta.push_literal(&[i as u8]); // Forces new op each time after copy
            delta.push_copy(i * 10, 10);
        }

        // Due to merging, we have alternating literal/copy
        assert!(delta.op_count() > 0);
    }

    #[test]
    fn delta_transmission_size() {
        let mut delta = Delta::new(1024, 1000, 1000);
        delta.push_copy(0, 500);
        delta.push_literal(&[0u8; 100]);

        let size = delta.transmission_size();
        // Header (56) + copy (13) + literal (5 + 100)
        assert_eq!(size, 56 + 13 + 105);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Bytes matched + bytes literal equals expected output
        #[test]
        fn bytes_sum_equals_output(
            copies in prop::collection::vec((0u64..10000, 1u32..1000), 0..10),
            literals in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..100), 0..10)
        ) {
            let mut delta = Delta::new(1024, 100000, 100000);

            for (offset, len) in &copies {
                delta.ops.push(DeltaOp::copy(*offset, *len));
            }
            for lit in &literals {
                delta.ops.push(DeltaOp::literal(lit.clone()));
            }

            let expected = delta.expected_output_size();
            let sum = delta.bytes_matched() + delta.bytes_literal();
            prop_assert_eq!(expected, sum);
        }

        /// Compression ratio is between 0 and 1
        #[test]
        fn compression_ratio_bounded(
            source_size in 1u64..100000,
            matched in 0u64..100000
        ) {
            let matched = matched.min(source_size);
            let mut delta = Delta::new(1024, source_size, source_size);
            if matched > 0 {
                delta.push_copy(0, matched.min(u64::from(u32::MAX)) as u32);
            }

            let ratio = delta.compression_ratio(source_size);
            prop_assert!(ratio >= 0.0);
            prop_assert!(ratio <= 1.0);
        }

        /// Contiguous copies are merged
        #[test]
        fn contiguous_copies_merged(
            base_offset in 0u64..1000,
            lens in prop::collection::vec(1u32..100, 2..10)
        ) {
            let mut delta = Delta::new(1024, 100000, 100000);
            let mut offset = base_offset;

            for &len in &lens {
                delta.push_copy(offset, len);
                offset += u64::from(len);
            }

            // Should be merged into single op
            prop_assert_eq!(delta.op_count(), 1);

            let total_len: u32 = lens.iter().sum();
            prop_assert_eq!(delta.bytes_matched(), u64::from(total_len));
        }

        /// Consecutive literals are merged
        #[test]
        fn consecutive_literals_merged(
            chunks in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..50), 2..10)
        ) {
            let mut delta = Delta::new(1024, 10000, 0);

            for chunk in &chunks {
                delta.push_literal(chunk);
            }

            prop_assert_eq!(delta.op_count(), 1);

            let total_len: usize = chunks.iter().map(Vec::len).sum();
            prop_assert_eq!(delta.bytes_literal(), total_len as u64);
        }

        /// Serde roundtrip preserves delta
        #[test]
        fn serde_roundtrip(
            block_size in 512u32..65536,
            source_size in 0u64..100000,
            basis_size in 0u64..100000
        ) {
            let delta = Delta::new(block_size, source_size, basis_size);
            let serialized = bincode::serialize(&delta).unwrap();
            let restored: Delta = bincode::deserialize(&serialized).unwrap();

            prop_assert_eq!(delta.block_size, restored.block_size);
            prop_assert_eq!(delta.source_size, restored.source_size);
            prop_assert_eq!(delta.basis_size, restored.basis_size);
        }
    }
}
