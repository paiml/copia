//! Async synchronization operations using tokio.
//!
//! This module provides async versions of the core sync operations,
//! enabling non-blocking file synchronization for I/O-bound workloads.

#[cfg(feature = "async")]
use std::path::Path;

#[cfg(feature = "async")]
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, AsyncWrite, AsyncWriteExt};

use crate::checksum::FastRollingChecksum;
use crate::delta::{Delta, DeltaOp};
use crate::error::{CopiaError, Result};
use crate::hash::StrongHash;
use crate::signature::{BlockSignature, Signature, SignatureTable};
use crate::sync::SyncConfig;

/// Async synchronization engine.
///
/// Provides async versions of signature, delta, and patch operations
/// for use with tokio runtime.
#[derive(Debug, Clone)]
pub struct AsyncCopiaSync {
    config: SyncConfig,
}

impl AsyncCopiaSync {
    /// Create a new async sync engine with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SyncConfig::default(),
        }
    }

    /// Create with custom block size.
    ///
    /// # Panics
    ///
    /// Panics if block size is invalid.
    #[must_use]
    pub fn with_block_size(block_size: usize) -> Self {
        assert!(
            block_size.is_power_of_two() && (512..=65536).contains(&block_size),
            "Block size must be power of 2, 512-65536"
        );
        Self {
            config: SyncConfig {
                block_size,
                ..SyncConfig::default()
            },
        }
    }

    /// Get the configured block size.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Generate signature from an async reader.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    #[cfg(feature = "async")]
    pub async fn signature<R>(&self, mut reader: R) -> Result<Signature>
    where
        R: AsyncRead + Unpin,
    {
        let block_size = self.config.block_size;
        let mut blocks = Vec::new();
        let mut buffer = vec![0u8; block_size];
        let mut index = 0u32;
        let mut file_size = 0u64;

        loop {
            let mut bytes_read = 0;
            while bytes_read < block_size {
                match reader.read(&mut buffer[bytes_read..]).await? {
                    0 => break,
                    n => bytes_read += n,
                }
            }

            if bytes_read == 0 {
                break;
            }

            let data = &buffer[..bytes_read];
            blocks.push(BlockSignature::compute(index, data));
            file_size += bytes_read as u64;
            index = index.saturating_add(1);
        }

        Ok(Signature {
            block_size,
            file_size,
            blocks,
        })
    }

    /// Compute delta between source and signature asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    #[cfg(feature = "async")]
    #[allow(clippy::cast_possible_truncation)] // block_size validated to be <= 65536
    pub async fn delta<R>(&self, mut source: R, signature: &Signature) -> Result<Delta>
    where
        R: AsyncRead + Unpin,
    {
        let table = SignatureTable::from_signature(signature.clone());
        let block_size = signature.block_size;

        // Read entire source into memory
        let mut source_data = Vec::new();
        source.read_to_end(&mut source_data).await?;

        let source_size = source_data.len() as u64;
        let source_hash = StrongHash::compute(&source_data);

        let mut delta = Delta::with_checksum(
            block_size as u32,
            source_size,
            signature.file_size,
            source_hash,
        );

        if source_data.is_empty() {
            return Ok(delta);
        }

        if table.is_empty() {
            delta.push_literal(&source_data);
            return Ok(delta);
        }

        let mut pos = 0usize;

        // Initialize fast rolling checksum with first block
        let init_len = block_size.min(source_data.len());
        let mut rolling = FastRollingChecksum::new(&source_data[..init_len]);

        while pos + block_size <= source_data.len() {
            let weak = rolling.digest();

            // Fast path: check weak hash first before computing strong hash
            if table.has_weak_match(weak) {
                let block_data = &source_data[pos..pos + block_size];
                if let Some(sig) = table.find_match(weak, block_data) {
                    let offset = u64::from(sig.index) * block_size as u64;
                    delta.push_copy(offset, block_size as u32);
                    pos += block_size;

                    // Re-initialize rolling checksum for next window
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

        if pos < source_data.len() {
            delta.push_literal(&source_data[pos..]);
        }

        Ok(delta)
    }

    /// Apply delta to basis file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if reading/writing fails or delta is invalid.
    #[cfg(feature = "async")]
    pub async fn patch<R, W>(&self, mut basis: R, delta: &Delta, mut output: W) -> Result<()>
    where
        R: AsyncRead + AsyncSeek + Unpin,
        W: AsyncWrite + Unpin,
    {
        delta.validate()?;

        let mut hasher = blake3::Hasher::new();

        for op in &delta.ops {
            match op {
                DeltaOp::Copy { offset, len } => {
                    basis
                        .seek(std::io::SeekFrom::Start(*offset))
                        .await?;
                    let mut buffer = vec![0u8; *len as usize];
                    basis.read_exact(&mut buffer).await?;
                    output.write_all(&buffer).await?;
                    hasher.update(&buffer);
                }
                DeltaOp::Literal(data) => {
                    output.write_all(data).await?;
                    hasher.update(data);
                }
            }
        }

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

    /// Synchronize a source file to a destination file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O operation fails.
    #[cfg(feature = "async")]
    pub async fn sync_files<P1, P2>(&self, source_path: P1, dest_path: P2) -> Result<SyncResult>
    where
        P1: AsRef<Path>,
        P2: AsRef<Path>,
    {
        use crate::sync::Sync;
        use std::io::Cursor;

        let source_path = source_path.as_ref();
        let dest_path = dest_path.as_ref();

        // Check if destination exists
        let dest_exists = tokio::fs::try_exists(dest_path).await.unwrap_or(false);

        if !dest_exists {
            // No basis file - just copy source
            let source_data = tokio::fs::read(source_path).await?;
            let source_size = source_data.len() as u64;
            tokio::fs::write(dest_path, &source_data).await?;

            return Ok(SyncResult {
                bytes_matched: 0,
                bytes_literal: source_size,
                source_size,
                basis_size: 0,
            });
        }

        // Read both files into memory
        let source_data = tokio::fs::read(source_path).await?;
        let basis_data = tokio::fs::read(dest_path).await?;
        let source_size = source_data.len() as u64;
        let basis_size = basis_data.len() as u64;

        // Fast path: if files are identical, no sync needed
        if source_data == basis_data {
            return Ok(SyncResult {
                bytes_matched: source_size,
                bytes_literal: 0,
                source_size,
                basis_size,
            });
        }

        // Generate signature from basis data (using sync version for speed)
        let signature = crate::Signature::generate(&mut Cursor::new(&basis_data), self.config.block_size)?;

        // Compute delta from source - use sync version with optimizations
        let sync = crate::CopiaSync::with_block_size(self.config.block_size);
        let delta = sync.delta(Cursor::new(&source_data), &signature)?;
        let bytes_matched = delta.bytes_matched();
        let bytes_literal = delta.bytes_literal();

        // Apply patch directly to output
        let mut output = Vec::with_capacity(source_data.len());
        sync.patch(Cursor::new(&basis_data), &delta, &mut output)?;

        // Write output atomically
        let temp_path = dest_path.with_extension("copia.tmp");
        tokio::fs::write(&temp_path, &output).await?;
        tokio::fs::rename(&temp_path, dest_path).await?;

        Ok(SyncResult {
            bytes_matched,
            bytes_literal,
            source_size,
            basis_size,
        })
    }
}

impl Default for AsyncCopiaSync {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a sync operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncResult {
    /// Bytes copied from basis file.
    pub bytes_matched: u64,
    /// Literal bytes transmitted.
    pub bytes_literal: u64,
    /// Total source file size.
    pub source_size: u64,
    /// Total basis file size.
    pub basis_size: u64,
}

impl SyncResult {
    /// Calculate compression ratio.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // acceptable for ratio calculation
    pub fn compression_ratio(&self) -> f64 {
        if self.source_size == 0 {
            return 1.0;
        }
        self.bytes_matched as f64 / self.source_size as f64
    }

    /// Calculate bandwidth savings.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // acceptable for ratio calculation
    pub fn bandwidth_savings(&self) -> f64 {
        if self.source_size == 0 {
            return 0.0;
        }
        1.0 - (self.bytes_literal as f64 / self.source_size as f64)
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[tokio::test]
    async fn async_signature_empty() {
        let sync = AsyncCopiaSync::new();
        let data: &[u8] = b"";
        let cursor = Cursor::new(data);
        let sig = sync.signature(cursor).await.unwrap();

        assert!(sig.blocks.is_empty());
        assert_eq!(sig.file_size, 0);
    }

    #[tokio::test]
    async fn async_signature_single_block() {
        let sync = AsyncCopiaSync::with_block_size(512);
        let data = b"small data";
        let cursor = Cursor::new(data.as_slice());
        let sig = sync.signature(cursor).await.unwrap();

        assert_eq!(sig.blocks.len(), 1);
        assert_eq!(sig.file_size, data.len() as u64);
    }

    #[tokio::test]
    async fn async_signature_multiple_blocks() {
        let sync = AsyncCopiaSync::with_block_size(512);
        let data = vec![42u8; 2000];
        let cursor = Cursor::new(data.as_slice());
        let sig = sync.signature(cursor).await.unwrap();

        assert_eq!(sig.blocks.len(), 4);
    }

    #[tokio::test]
    async fn async_delta_identical() {
        let sync = AsyncCopiaSync::with_block_size(512);
        let data = vec![42u8; 1024];

        let sig = sync.signature(Cursor::new(&data)).await.unwrap();
        let delta = sync.delta(Cursor::new(&data), &sig).await.unwrap();

        assert_eq!(delta.bytes_matched(), 1024);
        assert_eq!(delta.bytes_literal(), 0);
    }

    #[tokio::test]
    async fn async_delta_empty_basis() {
        let sync = AsyncCopiaSync::with_block_size(512);
        let basis: &[u8] = b"";
        let source = b"new content";

        let sig = sync.signature(Cursor::new(basis)).await.unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).await.unwrap();

        assert_eq!(delta.bytes_matched(), 0);
        assert_eq!(delta.bytes_literal(), source.len() as u64);
    }

    #[tokio::test]
    async fn async_patch_roundtrip() {
        let sync = AsyncCopiaSync::with_block_size(512);
        let basis = vec![42u8; 1024];
        let source = vec![42u8; 1024];

        let sig = sync.signature(Cursor::new(&basis)).await.unwrap();
        let delta = sync.delta(Cursor::new(&source), &sig).await.unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(&basis), &delta, &mut output)
            .await
            .unwrap();

        assert_eq!(output, source);
    }

    #[tokio::test]
    async fn async_patch_modified() {
        let sync = AsyncCopiaSync::with_block_size(512);
        let basis = b"Hello, World! This is original content.".to_vec();
        let source = b"Hello, Universe! This is modified content.".to_vec();

        let sig = sync.signature(Cursor::new(&basis)).await.unwrap();
        let delta = sync.delta(Cursor::new(&source), &sig).await.unwrap();

        let mut output = Vec::new();
        sync.patch(Cursor::new(&basis), &delta, &mut output)
            .await
            .unwrap();

        assert_eq!(output, source);
    }

    #[tokio::test]
    async fn sync_result_metrics() {
        let result = SyncResult {
            bytes_matched: 800,
            bytes_literal: 200,
            source_size: 1000,
            basis_size: 900,
        };

        assert!((result.compression_ratio() - 0.8).abs() < f64::EPSILON);
        assert!((result.bandwidth_savings() - 0.8).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn sync_result_empty() {
        let result = SyncResult {
            bytes_matched: 0,
            bytes_literal: 0,
            source_size: 0,
            basis_size: 0,
        };

        assert!((result.compression_ratio() - 1.0).abs() < f64::EPSILON);
        assert!(result.bandwidth_savings().abs() < f64::EPSILON);
    }
}

#[cfg(test)]
mod sync_tests {
    use super::*;

    #[test]
    fn async_sync_new() {
        let sync = AsyncCopiaSync::new();
        assert_eq!(sync.block_size(), 2048);
    }

    #[test]
    fn async_sync_with_block_size() {
        let sync = AsyncCopiaSync::with_block_size(4096);
        assert_eq!(sync.block_size(), 4096);
    }

    #[test]
    #[should_panic(expected = "Block size must be power of 2")]
    fn async_sync_invalid_block_size() {
        let _ = AsyncCopiaSync::with_block_size(1000);
    }

    #[test]
    fn async_sync_default() {
        let sync = AsyncCopiaSync::default();
        assert_eq!(sync.block_size(), 2048);
    }

    #[test]
    fn sync_result_compression_ratio() {
        let result = SyncResult {
            bytes_matched: 500,
            bytes_literal: 500,
            source_size: 1000,
            basis_size: 800,
        };
        assert!((result.compression_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn sync_result_bandwidth_savings() {
        let result = SyncResult {
            bytes_matched: 900,
            bytes_literal: 100,
            source_size: 1000,
            basis_size: 1000,
        };
        assert!((result.bandwidth_savings() - 0.9).abs() < f64::EPSILON);
    }
}
