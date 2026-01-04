# copia

**Pure Rust rsync-style file synchronization library**

[![Crates.io](https://img.shields.io/crates/v/copia.svg)](https://crates.io/crates/copia)
[![Documentation](https://docs.rs/copia/badge.svg)](https://docs.rs/copia)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/paiml/copia/workflows/CI/badge.svg)](https://github.com/paiml/copia/actions)

## Why copia?

- **Embeddable**: Use rsync's delta-transfer algorithm as a library, not a subprocess
- **Pure Rust**: 100% safe Rust, no unsafe code, fully auditable
- **Zero C Dependencies**: No OpenSSL, no librsync, no external binaries
- **Async Support**: First-class tokio integration for non-blocking I/O
- **Memory Safe**: No buffer overflows, no use-after-free, guaranteed by Rust

## Performance

```
┌────────────────────────────┬────────────┬────────────┬──────────────────┐
│ Scenario                   │ rsync (ms) │ copia (ms) │ Result           │
├────────────────────────────┼────────────┼────────────┼──────────────────┤
│ 1KB identical              │      43.55 │       0.05 │   Library wins   │
│ 100KB identical            │      43.23 │       0.12 │   Library wins   │
│ 1MB identical              │      43.40 │       0.33 │   Library wins   │
│ 1MB 5% changed             │      44.72 │       4.54 │   Library wins   │
│ 10MB identical             │      43.68 │       3.92 │   Library wins   │
│ 10MB 1% changed            │      46.91 │      43.05 │   Comparable     │
│ 10MB 100% different        │      52.84 │      43.88 │   Comparable     │
└────────────────────────────┴────────────┴────────────┴──────────────────┘

⚠️  IMPORTANT: rsync times include ~40ms process spawn overhead.
    This benchmark compares copia as a library vs rsync as a subprocess.
    For embedded/library use cases, copia avoids this overhead entirely.
    For CLI-to-CLI comparison, performance is comparable on large files.
```

**When copia shines:**
- Embedded in applications (no process spawn overhead)
- High-frequency sync operations (amortize startup cost)
- Small file synchronization (overhead dominates)
- When you need async I/O or Rust integration

**When rsync is fine:**
- One-off large file transfers (spawn overhead negligible)
- Shell scripts and CLI workflows
- When you need rsync's full feature set (permissions, links, etc.)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
copia = "0.1"
```

For async support:

```toml
[dependencies]
copia = { version = "0.1", features = ["async"] }
```

### CLI Installation

```bash
cargo install copia --features cli
```

## Quick Start

### Library Usage

```rust
use copia::{CopiaSync, Sync};
use std::io::Cursor;

// Create sync engine
let sync = CopiaSync::with_block_size(2048);

// Generate signature from basis (old) file
let basis = b"original file content here";
let signature = sync.signature(Cursor::new(basis.as_slice()))?;

// Compute delta from source (new) file
let source = b"modified file content here";
let delta = sync.delta(Cursor::new(source.as_slice()), &signature)?;

// Apply delta to reconstruct the new file
let mut output = Vec::new();
sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)?;

assert_eq!(output, source);
```

### Async Usage

```rust
use copia::async_sync::AsyncCopiaSync;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sync = AsyncCopiaSync::with_block_size(2048);

    // Sync source file to destination
    let result = sync.sync_files("source.txt", "dest.txt").await?;

    println!("Matched: {} bytes", result.bytes_matched);
    println!("Literal: {} bytes", result.bytes_literal);
    println!("Compression: {:.1}%", result.compression_ratio() * 100.0);

    Ok(())
}
```

### CLI Usage

```bash
# Sync a file
copia sync source.txt dest.txt

# Generate signature
copia signature file.txt -o file.sig

# Compute delta
copia delta newfile.txt file.sig -o file.delta

# Apply patch
copia patch oldfile.txt file.delta -o newfile.txt
```

## How It Works

Copia implements the rsync delta-transfer algorithm:

1. **Signature Generation**: The basis file is divided into fixed-size blocks. For each block, a rolling checksum (Adler-32 variant) and strong hash (BLAKE3) are computed.

2. **Delta Computation**: The source file is scanned with a sliding window. When the rolling checksum matches a known block, the strong hash verifies the match. Matching blocks become "copy" operations; non-matching data becomes "literal" operations.

3. **Patch Application**: The delta is applied to the basis file, copying matched blocks and inserting literal data to reconstruct the source.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Basis File │────▶│  Signature  │     │ Source File │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                          │                   │
                          ▼                   ▼
                   ┌──────────────────────────┐
                   │    Delta Computation     │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │ Delta: [Copy, Literal..] │
                   └────────────┬─────────────┘
                                │
         ┌─────────────┐        │
         │  Basis File │────────┤
         └─────────────┘        ▼
                   ┌──────────────────────────┐
                   │    Patch Application     │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │   Reconstructed Source   │
                   └──────────────────────────┘
```

## Implementation Details

| Component | Implementation |
|-----------|----------------|
| **Rolling Checksum** | Adler-32 variant with lazy modulo (normalize every 5000 rolls) |
| **Strong Hash** | BLAKE3 (32 bytes, cryptographic) |
| **Hash Table** | FxHashMap for fast u32 key lookups |
| **Parallelism** | Rayon for multi-core signature generation |

## API Reference

### Core Types

- `CopiaSync` - Main synchronization engine
- `Signature` - Block signatures for a file
- `Delta` - Difference between two files
- `RollingChecksum` - Adler-32 variant rolling checksum
- `StrongHash` - BLAKE3 cryptographic hash

### Async Types

- `AsyncCopiaSync` - Async synchronization engine
- `SyncResult` - Statistics from sync operation

## Feature Flags

| Feature | Description |
|---------|-------------|
| `async` | Enable tokio async support |
| `cli` | Build command-line interface |

## Benchmarks

Run benchmarks yourself:

```bash
# Compare against rsync (note: includes process spawn overhead)
cargo bench --bench rsync_comparison --features async

# Run criterion benchmarks (algorithm-only, no spawn overhead)
cargo bench --bench benchmarks
```

## Comparison with rsync

| Feature | copia | rsync |
|---------|-------|-------|
| Language | Pure Rust | C |
| Memory Safety | Guaranteed | Manual |
| Use as Library | Native | Subprocess only |
| Async I/O | Native | No |
| Process Overhead | None | ~40ms spawn |
| Permissions/ACLs | Not yet | Yes |
| Symbolic Links | Not yet | Yes |
| Compression | Not yet | Yes (zlib) |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs to the main branch.

## Acknowledgments

- rsync algorithm by Andrew Tridgell and Paul Mackerras
- BLAKE3 team for the fast cryptographic hash
- Rust community for excellent tooling
