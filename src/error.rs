//! Error types for copia operations.

use thiserror::Error;

/// Errors that can occur during copia operations.
#[derive(Error, Debug)]
pub enum CopiaError {
    /// I/O error during read/write operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid block size specified.
    #[error("Invalid block size: {0} (must be power of 2, 512-65536)")]
    InvalidBlockSize(usize),

    /// Invalid strong hash length specified.
    #[error("Invalid hash length: {0} (must be 4-32)")]
    InvalidHashLength(usize),

    /// Delta contains invalid copy bounds.
    #[error("Invalid copy bounds: offset {offset} + len {len} exceeds basis size {basis_size}")]
    InvalidCopyBounds {
        /// Copy offset in basis file
        offset: u64,
        /// Copy length
        len: u32,
        /// Total basis file size
        basis_size: u64,
    },

    /// Checksum mismatch after patch application.
    #[error("Checksum mismatch: expected {expected:?}, got {actual:?}")]
    ChecksumMismatch {
        /// Expected checksum
        expected: [u8; 32],
        /// Actual computed checksum
        actual: [u8; 32],
    },

    /// Empty signature provided.
    #[error("Empty signature: cannot compute delta without signatures")]
    EmptySignature,

    /// Delta application failed due to corrupted data.
    #[error("Corrupted delta data")]
    CorruptedDelta,

    /// Protocol error during network operations.
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Result type for copia operations.
pub type Result<T> = std::result::Result<T, CopiaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = CopiaError::Io(io_err);
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn error_display_invalid_block_size() {
        let err = CopiaError::InvalidBlockSize(100);
        assert!(err.to_string().contains("Invalid block size: 100"));
    }

    #[test]
    fn error_display_invalid_hash_length() {
        let err = CopiaError::InvalidHashLength(2);
        assert!(err.to_string().contains("Invalid hash length: 2"));
    }

    #[test]
    fn error_display_invalid_copy_bounds() {
        let err = CopiaError::InvalidCopyBounds {
            offset: 1000,
            len: 500,
            basis_size: 1200,
        };
        let msg = err.to_string();
        assert!(msg.contains("offset 1000"));
        assert!(msg.contains("len 500"));
        assert!(msg.contains("basis size 1200"));
    }

    #[test]
    fn error_display_checksum_mismatch() {
        let err = CopiaError::ChecksumMismatch {
            expected: [1u8; 32],
            actual: [2u8; 32],
        };
        assert!(err.to_string().contains("Checksum mismatch"));
    }

    #[test]
    fn error_display_empty_signature() {
        let err = CopiaError::EmptySignature;
        assert!(err.to_string().contains("Empty signature"));
    }

    #[test]
    fn error_display_corrupted_delta() {
        let err = CopiaError::CorruptedDelta;
        assert!(err.to_string().contains("Corrupted delta"));
    }

    #[test]
    fn error_display_protocol_error() {
        let err = CopiaError::ProtocolError("invalid frame".to_string());
        assert!(err.to_string().contains("Protocol error"));
        assert!(err.to_string().contains("invalid frame"));
    }

    #[test]
    fn result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.unwrap_or(0), 42);
    }

    #[test]
    fn result_type_err() {
        let result: Result<i32> = Err(CopiaError::EmptySignature);
        assert!(result.is_err());
    }
}
