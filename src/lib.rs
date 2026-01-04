//! # Copia
//!
//! Pure Rust rsync-style synchronization for Sovereign AI.
//!
//! Copia implements the rsync delta-transfer algorithm in 100% safe Rust,
//! providing memory-safe, auditable, high-performance file synchronization.
//!
//! ## Features
//!
//! - **Rolling Checksum**: Adler-32 variant for O(1) window sliding
//! - **Strong Hash**: BLAKE3 for cryptographic block verification
//! - **Delta Encoding**: Efficient representation of file differences
//! - **Streaming**: O(1) memory for arbitrary file sizes
//!
//! ## Example
//!
//! ```rust
//! use copia::{SyncBuilder, Sync};
//! use std::io::Cursor;
//!
//! // Create sync engine with custom block size
//! let sync = SyncBuilder::new()
//!     .block_size(2048)
//!     .build();
//!
//! // Generate signature from basis file
//! let basis = b"original file content";
//! let signature = sync.signature(Cursor::new(basis.as_slice())).unwrap();
//!
//! // Compute delta from modified source
//! let source = b"modified file content";
//! let delta = sync.delta(Cursor::new(source.as_slice()), &signature).unwrap();
//!
//! // Apply delta to reconstruct
//! let mut output = Vec::new();
//! sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output).unwrap();
//! assert_eq!(output, source);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]

pub mod async_sync;
mod checksum;
mod delta;
mod error;
mod hash;
mod protocol;
mod signature;
mod sync;

pub use checksum::{FastRollingChecksum, RollingChecksum};
pub use delta::{Delta, DeltaOp, DeltaStats};
pub use error::{CopiaError, Result};
pub use hash::StrongHash;
pub use protocol::{
    Codec, FrameBuilder, FrameHeader, Message, MessageType, PROTOCOL_MAGIC, PROTOCOL_VERSION,
};
pub use signature::{BlockSignature, Signature, SignatureTable};
pub use sync::{CopiaSync, Sync, SyncBuilder, SyncConfig, SyncStats};
