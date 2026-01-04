# copia

**Pure Rust rsync-style file synchronization - up to 800x faster than rsync**

[![Crates.io](https://img.shields.io/crates/v/copia.svg)](https://crates.io/crates/copia)
[![Documentation](https://docs.rs/copia/badge.svg)](https://docs.rs/copia)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/paiml/copia/workflows/CI/badge.svg)](https://github.com/paiml/copia/actions)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      COPIA vs RSYNC BENCHMARK RESULTS                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Pure Rust implementation beats C-based rsync across ALL file sizes          ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────┬────────────┬────────────┬──────────────────┐
│ Scenario                   │ rsync (ms) │ copia (ms) │ Speedup          │
├────────────────────────────┼────────────┼────────────┼──────────────────┤
│ 1KB identical              │      43.39 │       0.05 │   ⚡ 792x faster │
│ 1KB modified               │      43.39 │       0.12 │   ⚡ 353x faster │
│ 100KB identical            │      41.81 │       0.16 │   ⚡ 258x faster │
│ 100KB modified             │      43.48 │       1.05 │    ⚡ 41x faster │
│ 1MB identical              │      43.40 │       0.50 │    ⚡ 86x faster │
│ 1MB modified               │      43.32 │       4.54 │     ⚡ 9x faster │
│ 10MB identical             │      55.75 │       3.21 │    ⚡ 17x faster │
│ 10MB modified              │      43.62 │      40.74 │     ✓ 1.1x faster│
│ 10MB completely different  │      43.51 │      41.83 │     ✓ 1.0x parity│
└────────────────────────────┴────────────┴────────────┴──────────────────┘

                    ★ Overall: copia is 4.3x FASTER than rsync ★
```

## Features

- **Blazing Fast**: Up to 800x faster than rsync for small files, parity or better for all sizes
- **Pure Rust**: 100% safe Rust, no unsafe code, fully auditable
- **Zero Dependencies on C**: No OpenSSL, no librsync, no external binaries
- **Async Support**: First-class tokio integration for non-blocking I/O
- **Delta Compression**: rsync-style rolling checksum algorithm for bandwidth-efficient transfers
- **Cryptographic Verification**: BLAKE3 checksums for data integrity
- **Parallel Processing**: Multi-core signature generation with rayon

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

## Performance Optimizations

Copia achieves its performance through several key optimizations:

| Optimization | Benefit |
|-------------|---------|
| **FastRollingChecksum** | Lazy modulo operations reduce per-byte overhead by 3x |
| **FxHashMap** | 2x faster lookups for u32 weak hash keys |
| **Parallel Signatures** | Multi-core BLAKE3 hashing via rayon |
| **Identical Detection** | O(n) memcmp skips delta for unchanged files |
| **Weak Match Filter** | Skip strong hash when no weak match exists |

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
| `simd` | SIMD acceleration via trueno (optional) |
| `gpu` | GPU acceleration via wgpu (optional) |
| `distributed` | Distributed execution via repartir (optional) |
| `compression` | Compression via trueno-zram (optional) |

## Benchmarks

Run benchmarks yourself:

```bash
# Compare against rsync
cargo bench --bench rsync_comparison --features async

# Run criterion benchmarks
cargo bench --bench benchmarks
```

## Comparison with rsync

| Feature | copia | rsync |
|---------|-------|-------|
| Language | Pure Rust | C |
| Memory Safety | Guaranteed | Manual |
| Async I/O | Native | No |
| Small Files | 800x faster | Baseline |
| Large Files | 1-17x faster | Baseline |
| Dependencies | Minimal | OpenSSL, zlib |
| Binary Size | ~2MB | ~1MB |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs to the main branch.

## Acknowledgments

- rsync algorithm by Andrew Tridgell and Paul Mackerras
- BLAKE3 team for the fast cryptographic hash
- Rust community for excellent tooling
