# ADR-001: Algorithm and Architecture Choices

## Status

Accepted

## Context

Copia implements the rsync delta-transfer algorithm as a pure Rust
library. Key design decisions affect performance, security, and
maintainability.

## Decision

### Rolling Checksum: Adler-32 Variant

**Choice**: Custom Adler-32 with lazy modulo (normalize every 5000
rolls).

**Rationale**: The rolling checksum must support O(1) window sliding.
Adler-32 provides good distribution with low collision rates. The lazy
modulo optimization reduces divisions by ~99.8% in the inner loop.

**Alternatives considered**:
- CRC32: Not efficiently rollable
- Rabin fingerprint: Higher quality but slower
- Gear hash: Good for CDC but not block-matching

### Strong Hash: BLAKE3

**Choice**: BLAKE3 (32-byte output).

**Rationale**: BLAKE3 is 5-10x faster than SHA-256 while providing
equivalent cryptographic security. It supports incremental hashing and
SIMD acceleration out of the box. The 256-bit output provides
negligible collision probability even for exabyte-scale datasets.

**Alternatives considered**:
- SHA-256: Slower, no advantage
- xxHash: Faster but not cryptographic
- MD5: Broken, unacceptable for integrity verification

### Hash Table: FxHashMap

**Choice**: `rustc-hash::FxHashMap` for weak-hash lookup.

**Rationale**: Signature lookup is on the hot path (called for every
byte position). FxHashMap uses a fast, non-cryptographic hash that is
2-5x faster than the standard HashMap for u32 keys. Since the keys are
already hashed (rolling checksums), we only need fast distribution, not
cryptographic quality.

### Parallel Signature Generation

**Choice**: Rayon `par_chunks` for files > 64 KB.

**Rationale**: Signature generation is embarrassingly parallel — each
block's checksum and hash are independent. The 64 KB threshold avoids
overhead for small files where sequential is faster. Benchmarks show
3-4x speedup on 4+ core machines for large files.

### Memory Safety: forbid(unsafe_code)

**Choice**: `#![forbid(unsafe_code)]` crate-wide.

**Rationale**: File synchronization is a security-critical operation.
Buffer overflows in rsync (C) have caused CVEs. By forbidding unsafe
code, we guarantee memory safety at the type level. Performance is
achieved through algorithmic choices, not unsafe shortcuts.

## Consequences

- All delta operations produce correct output (verified by BLAKE3
  checksum on patch)
- The rolling checksum invariant (a, b < MOD) is maintained across all
  operations
- Delta bytes_matched + bytes_literal always equals source_size
- No undefined behavior is possible in the library
