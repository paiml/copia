# Copia: Pure Rust Rsync-Style Synchronization for Sovereign AI

**Status**: Specification
**Type**: Architecture Specification
**Version**: 2.0.0
**Created**: 2026-01-04
**Updated**: 2026-01-04
**Authors**: Pragmatic AI Labs
**Document ID**: SPEC-COPIA-002
**Quality Framework**: Iron Lotus + Certeza
**Ecosystem**: trueno + repartir + trueno-zram
**License**: MIT

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
   - 2.1 [Motivation](#21-motivation)
   - 2.2 [Design Principles](#22-design-principles)
   - 2.3 [Problem Statement](#23-problem-statement)
   - 2.4 [Goals and Non-Goals](#24-goals-and-non-goals)
3. [Research Foundation](#3-research-foundation)
4. [System Architecture](#4-system-architecture)
   - 4.1 [Conceptual Model](#41-conceptual-model)
   - 4.2 [Component Overview](#42-component-overview)
   - 4.3 [Data Flow Model](#43-data-flow-model)
5. [Algorithm Specifications](#5-algorithm-specifications)
   - 5.1 [Rolling Checksum (Adler-32 Variant)](#51-rolling-checksum-adler-32-variant)
   - 5.2 [Strong Hash (BLAKE3)](#52-strong-hash-blake3)
   - 5.3 [Delta Encoding](#53-delta-encoding)
   - 5.4 [Block Matching](#54-block-matching)
6. [API Design](#6-api-design)
   - 6.1 [Core Traits](#61-core-traits)
   - 6.2 [Builder Pattern APIs](#62-builder-pattern-apis)
   - 6.3 [Async Interface](#63-async-interface)
7. [Wire Protocol](#7-wire-protocol)
   - 7.1 [Protocol Overview](#71-protocol-overview)
   - 7.2 [Message Types](#72-message-types)
   - 7.3 [Framing Format](#73-framing-format)
8. [Performance Specifications](#8-performance-specifications)
9. [Security Model](#9-security-model)
10. [Toyota Way Analysis](#10-toyota-way-analysis)
11. [Quality Gates](#11-quality-gates)
12. [Testing Strategy](#12-testing-strategy)
13. [Popperian Falsification Checklist](#13-popperian-falsification-checklist)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [PAIML Stack Integration](#15-paiml-stack-integration)
16. [Trueno Ecosystem Integration](#16-trueno-ecosystem-integration)
   - 16.1 [trueno SIMD/GPU Acceleration](#161-trueno-simdgpu-acceleration)
   - 16.2 [repartir Distributed Execution](#162-repartir-distributed-execution)
   - 16.3 [trueno-zram Compression](#163-trueno-zram-compression)
   - 16.4 [GPU/WGPU Compute Pipeline](#164-gpuwgpu-compute-pipeline)
17. [References](#17-references)

---

## 1. Executive Summary

### Vision

Copia is a pure Rust implementation of rsync-style file synchronization primitives designed for Sovereign AI workloads. It provides memory-safe, auditable, high-performance delta synchronization with zero C/C++ dependencies, enabling secure and efficient data transfer across distributed AI infrastructure.

### Key Objectives

| Objective | Rationale | Success Metric |
|-----------|-----------|----------------|
| **Pure Rust Implementation** | Digital sovereignty, full auditability | 0 lines of C/C++/unsafe FFI |
| **rsync Algorithm Compatibility** | Industry-proven delta sync | Semantic equivalence with librsync |
| **SIMD Acceleration** | Performance parity with C | ≥90% of librsync throughput |
| **Streaming Architecture** | Memory-bounded operation | O(1) memory for arbitrary file sizes |
| **Zero-Copy I/O** | Minimize data movement | ≤2 copies source-to-destination |

### Impact

- **Bandwidth Reduction**: 10-100x for incremental syncs of large model weights
- **Memory Safety**: Eliminates buffer overflow vulnerabilities endemic to C rsync
- **Auditability**: Complete AST-level inspection via pmat tooling
- **Integration**: Native compatibility with trueno tensor operations and repartir distributed execution

---

## 2. Introduction

### 2.1 Motivation

Modern AI infrastructure requires synchronization of multi-gigabyte model weights, datasets, and checkpoints across distributed systems. Traditional rsync implementations suffer from:

1. **Memory Safety Vulnerabilities**: CVE history in C-based implementations
2. **Opaque Dependencies**: Difficult to audit for supply chain security
3. **Limited Integration**: No native Rust ecosystem compatibility
4. **Single-Threaded Design**: Cannot leverage modern multi-core systems

Copia addresses these limitations through a ground-up Rust implementation following Iron Lotus Framework principles.

### 2.2 Design Principles

| Principle | Application in Copia |
|-----------|---------------------|
| **Genchi Genbutsu** | All code pure Rust; no black-box dependencies |
| **Jidoka** | Fail-fast on corruption; automatic integrity verification |
| **Kaizen** | Incremental algorithm improvements via mutation testing |
| **Muda Elimination** | Zero-copy paths; streaming without buffering |
| **Heijunka** | Load-balanced parallel block processing |
| **Poka-yoke** | Type-safe protocol encoding; invalid states unrepresentable |

### 2.3 Problem Statement

**Current State**: Sovereign AI deployments require file synchronization but face a dilemma:
- Use C-based rsync (security risk, audit burden)
- Use naive full-file transfer (bandwidth waste)
- Use proprietary solutions (vendor lock-in)

**Desired State**: A pure Rust synchronization library that:
- Implements rsync delta algorithm with semantic equivalence
- Integrates natively with PAIML stack (trueno, repartir, aprender)
- Meets or exceeds C implementation performance
- Passes rigorous Popperian falsification testing

### 2.4 Goals and Non-Goals

**Goals**:
- [ ] Implement rolling checksum algorithm (Adler-32 variant)
- [ ] Implement strong hash using BLAKE3
- [ ] Delta encoding/decoding with streaming support
- [ ] Async I/O via tokio
- [ ] SIMD acceleration via trueno primitives
- [ ] Wire protocol compatible with distributed execution
- [ ] Comprehensive test coverage (≥95%)
- [ ] Mutation testing coverage (≥80%)

**Non-Goals**:
- Full rsync CLI compatibility (use repartir for orchestration)
- SSH transport (use rustls for TLS)
- ACL/xattr synchronization (filesystem-specific)
- Compression (delegated to trueno-ublk)

---

## 3. Research Foundation

### 3.1 Foundational Algorithms

#### The rsync Algorithm (Tridgell & Mackerras, 1996) [1]

The rsync algorithm enables efficient remote file synchronization by:

1. **Signature Generation**: Receiver computes rolling checksums and strong hashes for fixed-size blocks
2. **Delta Computation**: Sender matches local content against received signatures
3. **Delta Transmission**: Only non-matching regions transmitted
4. **Reconstruction**: Receiver applies delta to reconstruct updated file

**Complexity Analysis**:
- Signature generation: O(n) where n = file size
- Block matching: O(n) expected, O(n × m) worst case where m = block count
- Space: O(m) for signature storage

#### Rolling Checksum Theory (Rabin, 1981) [2]

Rolling checksums enable O(1) update when sliding a window:

```
s(k+1, l+1) = s(k, l) - X_k + X_{l+1}
```

Where:
- s(k, l) = checksum of bytes k through l
- X_i = byte value at position i

Copia uses an Adler-32 variant optimized for SIMD execution.

#### BLAKE3 Cryptographic Hash (O'Connor et al., 2020) [3]

BLAKE3 provides:
- 256-bit security level
- SIMD-parallel Merkle tree construction
- 3-7x faster than SHA-256 on modern CPUs
- Streaming incremental hashing

### 3.2 Performance Research

#### Zero-Copy I/O (Banga et al., 1999) [4]

Eliminating memory copies through:
- Memory-mapped I/O (mmap)
- Vectored I/O (readv/writev)
- Splice/sendfile for kernel-bypass

#### SIMD Text Processing (Langdale & Lemire, 2019) [5]

Techniques applicable to checksum computation:
- Vectorized byte scanning
- Parallel reduction operations
- Cache-line aligned processing

### 3.3 Rust Safety Guarantees

#### Memory Safety Without Garbage Collection (Jung et al., 2017) [6]

RustBelt formalization proves:
- No use-after-free
- No double-free
- No data races in safe Rust
- Controlled unsafe blocks with explicit invariants

---

## 4. System Architecture

### 4.1 Conceptual Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COPIA ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   SOURCE FILE   │    │   DESTINATION   │    │   WIRE PROTO    │         │
│  │                 │    │      FILE       │    │                 │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         STREAMING LAYER                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  AsyncIO │  │  Buffer  │  │  Framing │  │  Codec   │            │   │
│  │  │  (tokio) │  │   Pool   │  │          │  │          │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ALGORITHM LAYER                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Rolling  │  │  BLAKE3  │  │  Block   │  │  Delta   │            │   │
│  │  │ Checksum │  │   Hash   │  │ Matcher  │  │ Encoder  │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SIMD ACCELERATION (trueno)                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  AVX2    │  │  AVX-512 │  │   NEON   │  │  Scalar  │            │   │
│  │  │ Backend  │  │  Backend │  │  Backend │  │ Fallback │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Overview

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| **copia-core** | Algorithm implementations | trueno (SIMD) |
| **copia-proto** | Wire protocol encoding | bincode, serde |
| **copia-io** | Async I/O operations | tokio |
| **copia-cli** | Command-line interface | clap |

### 4.3 Data Flow Model

```
SENDER SIDE                                    RECEIVER SIDE
───────────────────────────────────────────────────────────────────────────

                                               ┌─────────────────────┐
                                               │  Basis File (old)   │
                                               └──────────┬──────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │  Generate Signature │
                                               │  - Rolling checksum │
                                               │  - Strong hash      │
                                               └──────────┬──────────┘
                                                          │
┌─────────────────────┐                                   │
│   Source File       │◄──────── Signature ───────────────┘
│   (new version)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Block Matching    │
│   - Scan with       │
│     rolling hash    │
│   - Verify with     │
│     strong hash     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Delta Generation  │
│   - Copy commands   │
│   - Literal data    │
└──────────┬──────────┘
           │
           └──────────── Delta ──────────────►┌─────────────────────┐
                                              │   Apply Delta       │
                                              │   - Copy from basis │
                                              │   - Insert literals │
                                              └──────────┬──────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │   Reconstructed     │
                                              │   File (new)        │
                                              └─────────────────────┘
```

### 4.4 Atomicity & Durability

To guarantee data integrity during synchronization failures, Copia employs a strict "temp-and-swap" strategy:

1.  **Isolation**: Patches are applied to a temporary file (`.copia.tmp.XXXXXX`) in the same filesystem as the destination.
2.  **Basis Integrity**: The existing destination file (basis) is read-only during the patch operation.
3.  **Atomic Promotion**: Upon successful reconstruction and verification, the temporary file is atomically renamed over the destination using `renameat2` (Linux) or equivalent.
4.  **Cleanup**: Temporary files are tracked and cleaned up on process termination or error.

---

## 5. Algorithm Specifications

### 5.1 Rolling Checksum (Adler-32 Variant)

The rolling checksum enables O(1) window sliding for block boundary detection.

#### Definition

```rust
/// Rolling checksum state for incremental computation
#[derive(Debug, Clone, Copy)]
pub struct RollingChecksum {
    /// Sum of all bytes in window (mod 2^16)
    a: u32,
    /// Weighted sum: sum of (i+1) * byte[i] (mod 2^16)
    b: u32,
    /// Current window size
    count: usize,
}

impl RollingChecksum {
    /// Base modulus for checksum arithmetic
    const MOD: u32 = 65521; // Largest prime < 2^16

    /// Create new checksum from initial block
    pub fn new(data: &[u8]) -> Self {
        let mut a: u32 = 0;
        let mut b: u32 = 0;

        for (i, &byte) in data.iter().enumerate() {
            a = a.wrapping_add(byte as u32);
            b = b.wrapping_add((data.len() - i) as u32 * byte as u32);
        }

        Self {
            a: a % Self::MOD,
            b: b % Self::MOD,
            count: data.len(),
        }
    }

    /// Roll window by one byte: remove old_byte, add new_byte
    #[inline]
    pub fn roll(&mut self, old_byte: u8, new_byte: u8) {
        let old = old_byte as u32;
        let new = new_byte as u32;

        self.a = (self.a.wrapping_sub(old).wrapping_add(new)) % Self::MOD;
        self.b = (self.b.wrapping_sub(self.count as u32 * old).wrapping_add(self.a)) % Self::MOD;
    }

    /// Combine a and b into 32-bit digest
    #[inline]
    pub fn digest(&self) -> u32 {
        (self.b << 16) | self.a
    }
}
```

#### SIMD Optimization

```rust
/// SIMD-accelerated rolling checksum using trueno
#[cfg(feature = "simd")]
pub fn rolling_checksum_simd(data: &[u8]) -> u32 {
    use trueno::simd::{u8x32, u32x8};

    // Process 32 bytes at a time with AVX2
    let chunks = data.chunks_exact(32);
    let remainder = chunks.remainder();

    let mut sum_a = u32x8::splat(0);
    let mut sum_b = u32x8::splat(0);

    for chunk in chunks {
        let bytes = u8x32::from_slice(chunk);
        // Vectorized accumulation...
    }

    // Horizontal reduction + remainder processing
    // ...
}
```

### 5.2 Strong Hash (BLAKE3)

BLAKE3 provides cryptographic verification of block matches.

```rust
use blake3::Hasher;

/// Strong hash for block verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrongHash([u8; 32]);

impl StrongHash {
    /// Compute BLAKE3 hash of data
    pub fn compute(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Compute with streaming interface
    pub fn compute_streaming<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut hasher = Hasher::new();
        std::io::copy(reader, &mut hasher)?;
        Ok(Self(*hasher.finalize().as_bytes()))
    }

    /// Truncated hash for memory efficiency (configurable)
    pub fn truncated(&self, len: usize) -> &[u8] {
        &self.0[..len.min(32)]
    }
}
```

### 5.3 Delta Encoding

Delta instructions represent the difference between source and destination.

```rust
/// Delta instruction types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaOp {
    /// Copy `len` bytes from basis file starting at `offset`
    Copy {
        offset: u64,
        len: u32,
    },
    /// Insert literal bytes directly
    Literal(Vec<u8>),
}

/// Encoded delta stream
#[derive(Debug, Clone)]
pub struct Delta {
    /// Block size used for signature generation
    pub block_size: u32,
    /// Sequence of delta operations
    pub ops: Vec<DeltaOp>,
}

impl Delta {
    /// Estimate bandwidth savings
    pub fn compression_ratio(&self, original_size: u64) -> f64 {
        let delta_size: u64 = self.ops.iter().map(|op| match op {
            DeltaOp::Copy { .. } => 12, // offset + len encoding
            DeltaOp::Literal(data) => data.len() as u64 + 4, // length prefix
        }).sum();

        1.0 - (delta_size as f64 / original_size as f64)
    }
}
```

### 5.4 Block Matching

The block matcher finds corresponding blocks between source and signatures.

```rust
use std::collections::HashMap;

/// Block signature for matching
#[derive(Debug, Clone)]
pub struct BlockSignature {
    pub index: u32,
    pub weak_hash: u32,
    pub strong_hash: StrongHash,
}

/// Signature table with two-level lookup
pub struct SignatureTable {
    /// First level: rolling checksum -> candidates
    weak_index: HashMap<u32, Vec<usize>>,
    /// Full signature data
    signatures: Vec<BlockSignature>,
    /// Block size
    block_size: usize,
}

impl SignatureTable {
    /// Build signature table from basis file
    pub fn build<R: std::io::Read>(
        reader: &mut R,
        block_size: usize,
    ) -> std::io::Result<Self> {
        let mut signatures = Vec::new();
        let mut weak_index: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut buffer = vec![0u8; block_size];
        let mut index = 0u32;

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 { break; }

            let data = &buffer[..n];
            let weak = RollingChecksum::new(data).digest();
            let strong = StrongHash::compute(data);

            weak_index.entry(weak).or_default().push(signatures.len());
            signatures.push(BlockSignature { index, weak_hash: weak, strong_hash: strong });
            index += 1;
        }

        Ok(Self { weak_index, signatures, block_size })
    }

    /// Find matching block for given data
    pub fn find_match(&self, weak: u32, data: &[u8]) -> Option<&BlockSignature> {
        let candidates = self.weak_index.get(&weak)?;
        let strong = StrongHash::compute(data);

        candidates.iter()
            .map(|&i| &self.signatures[i])
            .find(|sig| sig.strong_hash == strong)
    }
}
```

---

## 6. API Design

### 6.1 Core Traits

```rust
/// Core synchronization operations
pub trait Sync {
    type Error: std::error::Error;

    /// Generate signature from basis file
    fn signature<R: Read>(&self, basis: R) -> Result<Signature, Self::Error>;

    /// Compute delta between source and signature
    fn delta<R: Read>(&self, source: R, signature: &Signature) -> Result<Delta, Self::Error>;

    /// Apply delta to basis file, producing output
    fn patch<R: Read, W: Write>(
        &self,
        basis: R,
        delta: &Delta,
        output: W,
    ) -> Result<(), Self::Error>;
}

/// Async variant for I/O-bound operations
#[async_trait]
pub trait AsyncSync {
    type Error: std::error::Error;

    async fn signature<R: AsyncRead + Unpin>(
        &self,
        basis: R,
    ) -> Result<Signature, Self::Error>;

    async fn delta<R: AsyncRead + Unpin>(
        &self,
        source: R,
        signature: &Signature,
    ) -> Result<Delta, Self::Error>;

    async fn patch<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
        &self,
        basis: R,
        delta: &Delta,
        output: W,
    ) -> Result<(), Self::Error>;
}
```

### 6.2 Builder Pattern APIs

```rust
/// Configuration builder for sync operations
#[derive(Debug, Clone)]
pub struct SyncBuilder {
    block_size: usize,
    strong_hash_len: usize,
    parallel: bool,
    buffer_size: usize,
}

impl SyncBuilder {
    /// Create new builder with defaults
    pub fn new() -> Self {
        Self {
            block_size: 2048,       // Default block size
            strong_hash_len: 8,     // Truncated hash for efficiency
            parallel: true,         // Enable parallel processing
            buffer_size: 64 * 1024, // 64KB I/O buffer
        }
    }

    /// Set block size (must be power of 2, 512-65536)
    pub fn block_size(mut self, size: usize) -> Self {
        assert!(size.is_power_of_two() && size >= 512 && size <= 65536);
        self.block_size = size;
        self
    }

    /// Set strong hash length (4-32 bytes)
    pub fn strong_hash_len(mut self, len: usize) -> Self {
        assert!((4..=32).contains(&len));
        self.strong_hash_len = len;
        self
    }

    /// Enable/disable parallel block processing
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Set I/O buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Build the sync engine
    pub fn build(self) -> CopiaSync {
        CopiaSync {
            config: self,
        }
    }
}

impl Default for SyncBuilder {
    fn default() -> Self {
        Self::new()
    }
}
```

### 6.3 Async Interface

```rust
use tokio::io::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};

/// High-level async synchronization API
pub struct CopiaAsync {
    inner: CopiaSync,
}

impl CopiaAsync {
    /// Synchronize source to destination (simplified flow)
    pub async fn sync<S, B, W>(
        &self,
        mut source: S,
        mut basis: B,
        mut output: W,
    ) -> Result<SyncStats, CopiaError>
    where
        S: AsyncRead + Unpin,
        B: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        // 1. Generate signature from basis
        // Note: Real impl needs to handle basis seeking if reused
        let signature = self.inner.signature_async(&mut basis).await?;

        // 2. Compute delta from source
        let delta = self.inner.delta_async(&mut source, &signature).await?;
        let source_size = delta.source_len(); // Assumed available from Delta

        // 3. Apply delta to basis, writing to output
        // Note: Logic requires basis re-reading unless buffered
        self.inner.patch_async(&mut basis, &delta, &mut output).await?;

        Ok(SyncStats {
            bytes_matched: delta.bytes_matched(),
            bytes_literal: delta.bytes_literal(),
            compression_ratio: delta.compression_ratio(source_size),
        })
    }
}
```

---

## 7. Wire Protocol

### 7.1 Protocol Overview

Copia uses a simple framed protocol for network transmission.

```
┌─────────────────────────────────────────────────────────────────┐
│                      COPIA WIRE PROTOCOL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ HEADER  │───▶│ LENGTH  │───▶│  TYPE   │───▶│ PAYLOAD │     │
│  │ (magic) │    │ (u32le) │    │  (u8)   │    │ (bytes) │     │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│   4 bytes        4 bytes        1 byte        variable         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Message Types

```rust
/// Protocol message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Signature request/response
    Signature = 0x01,
    /// Delta data
    Delta = 0x02,
    /// Sync complete acknowledgment
    Ack = 0x03,
    /// Error message
    Error = 0x04,
    /// Heartbeat/keepalive
    Ping = 0x05,
    /// Heartbeat response
    Pong = 0x06,
}

/// Protocol messages
#[derive(Debug, Clone)]
pub enum Message {
    /// Request signature generation
    SignatureRequest {
        file_id: u64,
        block_size: u32,
    },
    /// Signature response
    SignatureResponse {
        file_id: u64,
        signatures: Vec<BlockSignature>,
    },
    /// Delta transmission
    DeltaData {
        file_id: u64,
        ops: Vec<DeltaOp>,
        checksum: StrongHash,
    },
    /// Acknowledgment
    Ack {
        file_id: u64,
        success: bool,
    },
    /// Error
    Error {
        code: u32,
        message: String,
    },
}
```

### 7.3 Framing Format

```rust
/// Protocol frame header
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct FrameHeader {
    /// Magic bytes: "COPA"
    pub magic: [u8; 4],
    /// Payload length (little-endian)
    pub length: u32,
    /// Message type
    pub msg_type: u8,
    /// Protocol version
    pub version: u8,
    /// Flags (reserved)
    pub flags: u16,
}

impl FrameHeader {
    pub const MAGIC: [u8; 4] = *b"COPA";
    pub const VERSION: u8 = 1;
    pub const SIZE: usize = 12;

    pub fn new(msg_type: MessageType, payload_len: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            length: payload_len,
            msg_type: msg_type as u8,
            version: Self::VERSION,
            flags: 0,
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.magic != Self::MAGIC {
            return Err(ProtocolError::InvalidMagic);
        }
        if self.version != Self::VERSION {
            return Err(ProtocolError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}
```

---

## 8. Performance Specifications

### 8.1 Throughput Targets

| Operation | Target | Conditions |
|-----------|--------|------------|
| Signature generation | ≥2 GB/s | Sequential read, 4KB blocks |
| Delta computation | ≥1 GB/s | 50% block match rate |
| Delta application | ≥3 GB/s | Sequential write |
| Rolling checksum | ≥10 GB/s | SIMD-accelerated |
| BLAKE3 hashing | ≥5 GB/s | Multi-threaded |

### 8.2 Latency Targets

| Operation | P50 | P99 | P99.9 |
|-----------|-----|-----|-------|
| Block match lookup | <100ns | <500ns | <1μs |
| Frame encode/decode | <1μs | <5μs | <10μs |
| Signature (1MB file) | <500μs | <2ms | <5ms |

### 8.3 Memory Constraints

| Constraint | Limit | Rationale |
|------------|-------|-----------|
| Peak memory | O(signature_size) | Stream processing |
| Signature table | 20 bytes/block | Weak hash + strong hash + index |
| I/O buffers | 2 × buffer_size | Double buffering |
| Delta buffer | Configurable | Batch transmission |

### 8.4 Benchmark Comparisons

| Implementation | 1GB Signature | 1GB Delta (10% change) | Language |
|----------------|--------------|------------------------|----------|
| librsync 2.3 | 450ms | 890ms | C |
| rdiff | 520ms | 950ms | C |
| **copia (target)** | **500ms** | **900ms** | **Rust** |

---

## 9. Security Model

### 9.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| **Hash collision attack** | BLAKE3 256-bit security; collision-resistant |
| **Memory corruption** | Pure Rust; no unsafe in hot paths |
| **Protocol injection** | Strict frame validation; magic bytes |
| **Supply chain** | Zero C dependencies; reproducible builds |
| **Timing attacks** | Constant-time hash comparison |

### 9.2 Cryptographic Choices

```rust
/// Security configuration
pub struct SecurityConfig {
    /// Minimum strong hash length (bytes)
    pub min_hash_len: usize,  // Default: 8, Paranoid: 32
    /// Enable constant-time comparisons
    pub constant_time: bool,  // Default: true
    /// Verify final checksum
    pub verify_checksum: bool, // Default: true
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            min_hash_len: 8,
            constant_time: true,
            verify_checksum: true,
        }
    }
}
```

### 9.3 Memory Safety Guarantees

- **No unsafe code** in public API surface
- **Isolated unsafe** in SIMD backends with `// SAFETY:` documentation
- **Bounds checking** on all slice operations
- **Integer overflow** protection via checked arithmetic

---

## 10. Toyota Way Analysis

### 10.1 Jidoka (Autonomation with Human Touch)

| Principle | Copia Implementation |
|-----------|---------------------|
| **Stop-the-line** | Fail immediately on checksum mismatch |
| **Built-in quality** | Type-safe protocol encoding |
| **Andon cord** | Automatic corruption detection |

```rust
/// Jidoka: Stop on quality issue
impl Delta {
    pub fn apply<R: Read, W: Write>(
        &self,
        basis: &mut R,
        output: &mut W,
    ) -> Result<(), CopiaError> {
        for op in &self.ops {
            match op {
                DeltaOp::Copy { offset, len } => {
                    // JIDOKA: Verify copy bounds before proceeding
                    if *offset + *len as u64 > self.basis_size {
                        return Err(CopiaError::InvalidCopyBounds);
                    }
                    // ...
                }
                DeltaOp::Literal(data) => {
                    output.write_all(data)?;
                }
            }
        }
        Ok(())
    }
}
```

### 10.2 Kaizen (Continuous Improvement)

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Test coverage | 0% | ≥95% | cargo llvm-cov |
| Mutation score | 0% | ≥80% | cargo mutants |
| Technical debt | N/A | 0 SATD | pmat tdg |
| Complexity | N/A | <10 cyclomatic | pmat analyze |

### 10.3 Genchi Genbutsu (Go and See)

- **100% Rust**: Every line auditable
- **No black boxes**: SIMD via trueno (also pure Rust)
- **Tracing**: Built-in observability
- **pmat integration**: AST-level quality analysis

### 10.4 Muda (Waste Elimination)

| Waste Type | Elimination Strategy |
|------------|---------------------|
| **Motion** | Zero-copy I/O paths |
| **Waiting** | Async I/O with tokio |
| **Overprocessing** | Adaptive block sizing |
| **Defects** | Fail-fast validation |
| **Inventory** | Streaming (no full-file buffering) |

### 10.5 Heijunka (Level Scheduling)

```rust
/// Parallel block processing with work stealing
pub async fn compute_signatures_parallel(
    data: &[u8],
    block_size: usize,
    parallelism: usize,
) -> Vec<BlockSignature> {
    use rayon::prelude::*;

    data.par_chunks(block_size)
        .enumerate()
        .map(|(i, chunk)| BlockSignature {
            index: i as u32,
            weak_hash: RollingChecksum::new(chunk).digest(),
            strong_hash: StrongHash::compute(chunk),
        })
        .collect()
}
```

---

## 11. Quality Gates

### 11.1 Tier 1: On-Save (Sub-second)

```bash
# Instant feedback
cargo check
cargo clippy -- -D warnings
```

### 11.2 Tier 2: On-Commit (1-5 minutes)

```bash
# Pre-commit gate
cargo test
cargo llvm-cov --fail-under-lines 95
cargo fmt --check
```

### 11.3 Tier 3: On-Merge (Comprehensive)

```bash
# Exhaustive validation
cargo mutants --minimum-coverage 80
cargo fuzz run fuzz_delta -- -max_total_time=300
pmat analyze --fail-on-tdg-decrease
```

### 11.4 Quality Metrics

| Metric | Threshold | Tool |
|--------|-----------|------|
| Line coverage | ≥95% | cargo llvm-cov |
| Branch coverage | ≥90% | cargo llvm-cov |
| Mutation coverage | ≥80% | cargo mutants |
| Cyclomatic complexity | ≤15 | pmat analyze |
| Cognitive complexity | ≤10 | pmat analyze |
| SATD count | 0 | pmat tdg |
| Clippy warnings | 0 | cargo clippy |

---

## 12. Testing Strategy

### 12.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_checksum_basic() {
        let data = b"hello world";
        let checksum = RollingChecksum::new(data);
        assert_eq!(checksum.digest(), 0x1a0b045d);
    }

    #[test]
    fn rolling_checksum_roll() {
        let data1 = b"hello";
        let data2 = b"ellox";

        let mut cs = RollingChecksum::new(data1);
        cs.roll(b'h', b'x');

        let expected = RollingChecksum::new(data2);
        assert_eq!(cs.digest(), expected.digest());
    }

    #[test]
    fn strong_hash_deterministic() {
        let data = b"test data";
        let hash1 = StrongHash::compute(data);
        let hash2 = StrongHash::compute(data);
        assert_eq!(hash1, hash2);
    }
}
```

### 12.2 Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn rolling_checksum_equivalence(data in prop::collection::vec(any::<u8>(), 1..1000)) {
        // Rolling checksum should equal direct computation
        if data.len() > 1 {
            let direct = RollingChecksum::new(&data[1..]);

            let mut rolling = RollingChecksum::new(&data[..data.len()-1]);
            rolling.roll(data[0], data[data.len()-1]);

            // Note: This is a simplified example; actual implementation differs
        }
    }

    #[test]
    fn delta_roundtrip(
        basis in prop::collection::vec(any::<u8>(), 100..10000),
        modifications in prop::collection::vec((0usize..100, any::<u8>()), 0..50)
    ) {
        // Apply modifications to create source
        let mut source = basis.clone();
        for (pos, byte) in modifications {
            if pos < source.len() {
                source[pos] = byte;
            }
        }

        // Generate signature, delta, apply
        let sync = SyncBuilder::new().block_size(64).build();
        let sig = sync.signature(&basis[..]).unwrap();
        let delta = sync.delta(&source[..], &sig).unwrap();

        let mut output = Vec::new();
        sync.patch(&basis[..], &delta, &mut output).unwrap();

        assert_eq!(source, output);
    }
}
```

### 12.3 Differential Testing

```rust
/// Differential test against librsync (reference implementation)
#[test]
#[ignore] // Run with --ignored for integration tests
fn differential_vs_librsync() {
    let test_cases = [
        ("empty", vec![]),
        ("small", vec![1, 2, 3, 4, 5]),
        ("block_aligned", vec![0u8; 2048]),
        ("random", generate_random_data(100_000)),
    ];

    for (name, data) in test_cases {
        let copia_sig = copia_signature(&data);
        let librsync_sig = librsync_signature(&data);

        assert_eq!(
            copia_sig.len(),
            librsync_sig.len(),
            "Signature count mismatch for {}", name
        );

        // Verify semantic equivalence (not bit-exact)
        for (c, l) in copia_sig.iter().zip(librsync_sig.iter()) {
            assert_eq!(c.weak_hash, l.weak_hash, "Weak hash mismatch for {}", name);
        }
    }
}
```

### 12.4 Fuzz Testing

```rust
// fuzz/fuzz_targets/fuzz_delta.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use copia::{SyncBuilder, Signature};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 { return; }

    let mid = data.len() / 2;
    let basis = &data[..mid];
    let source = &data[mid..];

    let sync = SyncBuilder::new()
        .block_size(64)
        .build();

    // Should not panic
    if let Ok(sig) = sync.signature(basis) {
        let _ = sync.delta(source, &sig);
    }
});
```

---

## 13. Popperian Falsification Checklist

### 13.1 Structural Invariants (Points 1-20)

| # | Falsifiable Claim | Test Method |
|---|-------------------|-------------|
| 1 | Rolling checksum is O(1) per byte | Benchmark with varying window sizes |
| 2 | Strong hash collisions < 2^-64 | Statistical analysis over 10^9 samples |
| 3 | Block size is always power of 2 | assert!(block_size.is_power_of_two()) |
| 4 | Signature table lookup is O(1) average | Benchmark with varying table sizes |
| 5 | Delta ops are non-overlapping | Verify in delta generation |
| 6 | Copy offsets are always valid | Bounds check on every copy |
| 7 | Literal data is non-empty | assert!(!literal.is_empty()) |
| 8 | Frame magic is "COPA" | Protocol validation |
| 9 | Version byte equals 1 | Protocol validation |
| 10 | Payload length matches actual | Frame parsing verification |
| 11 | Message types are exhaustive | Enum variant coverage |
| 12 | Builder defaults are sane | Unit tests for defaults |
| 13 | Config validation rejects invalid | Test invalid inputs |
| 14 | Error types are non-overlapping | Enum analysis |
| 15 | Async/sync APIs are equivalent | Differential testing |
| 16 | SIMD results equal scalar | Cross-backend verification |
| 17 | Parallel results equal sequential | Determinism tests |
| 18 | Empty input produces empty output | Edge case tests |
| 19 | Maximum file size is 2^63 bytes | Type system enforcement |
| 20 | Block index fits in u32 | Overflow detection |

### 13.2 Memory Safety (Points 21-40)

| # | Falsifiable Claim | Test Method |
|---|-------------------|-------------|
| 21 | No buffer overflows | Miri, AddressSanitizer |
| 22 | No use-after-free | Miri analysis |
| 23 | No double-free | Miri analysis |
| 24 | No null pointer dereference | Rust type system |
| 25 | No uninitialized memory read | Miri analysis |
| 26 | No data races | ThreadSanitizer |
| 27 | All slices bounds-checked | Clippy + Miri |
| 28 | Integer overflow protected | Checked arithmetic |
| 29 | Stack usage bounded | Static analysis |
| 30 | Heap allocations bounded | Memory profiling |
| 31 | No memory leaks | Valgrind/Miri |
| 32 | Panic-free in normal operation | Fuzzing coverage |
| 33 | OOM handled gracefully | Memory limit testing |
| 34 | File handles closed | Resource tracking |
| 35 | Temp files cleaned up | Integration tests |
| 36 | No unsafe in public API | grep for pub unsafe |
| 37 | All unsafe documented | // SAFETY: audit |
| 38 | Unsafe confined to SIMD | Module boundary check |
| 39 | Drop implemented correctly | Custom drop tests |
| 40 | Send/Sync correctly derived | Compile-time verification |

### 13.3 Functional Correctness (Points 41-60)

| # | Falsifiable Claim | Test Method |
|---|-------------------|-------------|
| 41 | Identical files produce empty delta | Unit test |
| 42 | Delta application is idempotent | Property test |
| 43 | Signature is deterministic | Repeated computation |
| 44 | Delta is deterministic | Repeated computation |
| 45 | Patch output matches source exactly | Byte comparison |
| 46 | Partial matches handled correctly | Synthetic tests |
| 47 | Block boundary alignment correct | Edge case tests |
| 48 | Last partial block handled | File size % block_size != 0 |
| 49 | Empty file handled | Zero-length input test |
| 50 | Large file handled | >4GB test files |
| 51 | Binary data handled | Non-UTF8 content |
| 52 | Streaming produces same result | Compare buffered vs streaming |
| 53 | Checksum detects corruption | Bit-flip tests |
| 54 | Protocol version validated | Invalid version rejection |
| 55 | Invalid frames rejected | Malformed input tests |
| 56 | Truncated frames detected | Partial read simulation |
| 57 | Duplicate blocks handled | Repeated content tests |
| 58 | Insertion at start handled | Prepend modification |
| 59 | Insertion at end handled | Append modification |
| 60 | Middle insertion handled | Interior modification |

### 13.4 Performance (Points 61-80)

| # | Falsifiable Claim | Test Method |
|---|-------------------|-------------|
| 61 | Signature throughput ≥2 GB/s | Criterion benchmarks |
| 62 | Delta throughput ≥1 GB/s | Criterion benchmarks |
| 63 | Patch throughput ≥3 GB/s | Criterion benchmarks |
| 64 | Memory usage is O(signatures) | Memory profiling |
| 65 | No O(n²) algorithms | Complexity analysis |
| 66 | SIMD provides ≥2x speedup | Backend comparison |
| 67 | Parallel provides ≥2x speedup | Thread count scaling |
| 68 | Latency P99 < 10x P50 | Distribution analysis |
| 69 | Cold start < 10ms | Startup benchmark |
| 70 | Hash table load factor < 0.75 | Runtime monitoring |
| 71 | Cache efficiency > 80% | perf stat analysis |
| 72 | Branch prediction > 95% | perf stat analysis |
| 73 | Zero-copy path used when possible | Tracing verification |
| 74 | Buffer reuse effective | Allocation tracking |
| 75 | Async overhead < 5% | Sync vs async comparison |
| 76 | Protocol overhead < 1% | Wire efficiency test |
| 77 | Compression ratio > 90% for 1% change | Synthetic benchmark |
| 78 | Compression ratio > 50% for 10% change | Synthetic benchmark |
| 79 | Small file overhead < 100 bytes | Minimum delta size |
| 80 | Large file scales linearly | Scaling benchmark |

### 13.5 Security (Points 81-90)

| # | Falsifiable Claim | Test Method |
|---|-------------------|-------------|
| 81 | Hash comparison is constant-time | Timing analysis |
| 82 | No timing side-channels in matching | Statistical timing tests |
| 83 | Invalid input cannot crash | Fuzzing |
| 84 | Malicious delta cannot corrupt | Adversarial testing |
| 85 | Path traversal prevented | ../../../etc/passwd tests |
| 86 | Symlink attacks prevented | Symlink handling tests |
| 87 | Resource exhaustion prevented | Limits enforced |
| 88 | Integer overflow exploits prevented | Boundary value tests |
| 89 | Supply chain verifiable | cargo vet, cargo deny |
| 90 | Reproducible builds | Bit-exact rebuild |

### 13.6 Compatibility (Points 91-100)

| # | Falsifiable Claim | Test Method |
|---|-------------------|-------------|
| 91 | API stable across minor versions | Semver compliance |
| 92 | Wire protocol backward compatible | Version negotiation |
| 93 | Cross-platform byte order handled | BE/LE testing |
| 94 | Works on Linux x86_64 | CI matrix |
| 95 | Works on Linux aarch64 | CI matrix |
| 96 | Works on macOS | CI matrix |
| 97 | Works on Windows | CI matrix |
| 98 | WASM target compiles | cargo build --target wasm32 |
| 99 | Integrates with trueno | Integration tests |
| 100 | Integrates with repartir | Integration tests |

---

## 14. Implementation Roadmap

### Phase 1: Foundation

- [ ] Project scaffolding (Cargo.toml, workspace)
- [ ] Core types (RollingChecksum, StrongHash, BlockSignature)
- [ ] Basic signature generation (scalar)
- [ ] Basic delta computation
- [ ] Basic patch application
- [ ] Unit test suite (≥80% coverage)

### Phase 2: Performance

- [ ] SIMD rolling checksum (AVX2, NEON)
- [ ] Parallel signature generation
- [ ] Parallel block matching
- [ ] Zero-copy I/O paths
- [ ] Benchmark suite with criterion
- [ ] Performance validation vs targets

### Phase 3: Network

- [ ] Wire protocol implementation
- [ ] Frame encoding/decoding
- [ ] Async I/O integration (tokio)
- [ ] Connection management
- [ ] Error handling and recovery

### Phase 4: Integration

- [ ] trueno SIMD backend integration
- [ ] repartir distributed execution hooks
- [ ] CLI interface
- [ ] Documentation
- [ ] Mutation testing (≥80%)
- [ ] Fuzzing campaign

### Phase 5: Hardening

- [ ] Security audit
- [ ] Performance optimization
- [ ] API stabilization
- [ ] v1.0.0 release

---

## 15. PAIML Stack Integration

### 15.1 trueno Integration

```rust
/// SIMD-accelerated checksum using trueno primitives
#[cfg(feature = "trueno")]
pub mod simd {
    use trueno::simd::*;

    pub fn rolling_checksum_avx2(data: &[u8]) -> u32 {
        // Leverage trueno's AVX2 primitives for vectorized computation
        // ...
    }
}
```

### 15.2 repartir Integration

```rust
/// Distributed sync task for repartir execution
#[cfg(feature = "repartir")]
pub struct SyncTask {
    source: PathBuf,
    destination: PathBuf,
    config: SyncBuilder,
}

impl repartir::Task for SyncTask {
    type Output = SyncStats;

    fn execute(self) -> Self::Output {
        let sync = self.config.build();
        // ...
    }
}
```

### 15.3 Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                      PAIML STACK                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│  │ aprender│────▶│ trueno  │◀────│  copia  │              │
│  │  (ML)   │     │ (SIMD)  │     │ (sync)  │              │
│  └─────────┘     └─────────┘     └────┬────┘              │
│       │               │               │                    │
│       └───────────────┼───────────────┘                    │
│                       ▼                                    │
│               ┌─────────────┐                              │
│               │  repartir   │                              │
│               │(distributed)│                              │
│               └─────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 16. Documentation Integration Strategy

To ensure documentation remains in sync with implementation, Copia enforces the "verified-code-inclusion" pattern.

### 16.1 Source of Truth
All code examples in documentation (except simple snippets) must be sourced from compilable, tested Rust files using the `mdbook` include syntax.

```markdown
<!-- GOOD: Includes actual test code -->
{{#include ../../examples/simple_sync.rs:20:45}}

<!-- BAD: Hardcoded block that will rot -->
```rust
fn example() { ... }
```
```

### 16.2 Verification Pipeline
1.  **Extraction**: Code blocks are not extracted; instead, `mdbook` pulls from source.
2.  **Compilation**: The source files (`examples/*.rs`, `tests/*.rs`) are compiled by CI.
3.  **Validation**: `cargo run --example ...` ensures runtime correctness.

---

## 16. Trueno Ecosystem Integration

### 16.1 trueno SIMD/GPU Acceleration

Copia leverages trueno's multi-target compute primitives for high-performance operations across CPU SIMD and GPU backends.

#### Backend Selection Strategy

```rust
use trueno::{Backend, select_best_available_backend, select_backend_for_operation};
use trueno::{Vector, Matrix, OperationType};

/// Copia compute backend configuration
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Preferred CPU SIMD backend
    pub cpu_backend: Backend,
    /// Enable GPU acceleration for large workloads
    pub gpu_enabled: bool,
    /// Minimum data size to trigger GPU offload (bytes)
    pub gpu_threshold: usize,
    /// GPU device index (for multi-GPU systems)
    pub gpu_device: usize,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            cpu_backend: select_best_available_backend(),
            gpu_enabled: true,
            gpu_threshold: 10 * 1024 * 1024, // 10MB
            gpu_device: 0,
        }
    }
}
```

#### SIMD-Accelerated Rolling Checksum

```rust
use trueno::Vector;

/// Vectorized Adler-32 computation using trueno SIMD
#[cfg(feature = "trueno")]
pub fn rolling_checksum_simd(data: &[u8], backend: Backend) -> u32 {
    match backend {
        Backend::AVX2 | Backend::AVX512 => {
            // Process 32/64 bytes at a time
            let vec = Vector::<u8>::from_slice(data);
            // Vectorized reduction for sum_a and sum_b
            compute_adler32_vectorized(&vec)
        }
        Backend::NEON => {
            // ARM NEON 128-bit processing
            compute_adler32_neon(data)
        }
        Backend::WasmSIMD => {
            // WebAssembly SIMD128
            compute_adler32_wasm(data)
        }
        _ => {
            // Scalar fallback
            RollingChecksum::new(data).digest()
        }
    }
}

/// Parallel signature generation using trueno matrices
#[cfg(feature = "trueno")]
pub fn parallel_signatures(
    data: &[u8],
    block_size: usize,
) -> Vec<BlockSignature> {
    use rayon::prelude::*;

    // Create matrix view for parallel block processing
    let num_blocks = (data.len() + block_size - 1) / block_size;

    (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * block_size;
            let end = (start + block_size).min(data.len());
            let block = &data[start..end];

            BlockSignature {
                index: i as u32,
                weak_hash: rolling_checksum_simd(block, Backend::Auto),
                strong_hash: StrongHash::compute(block),
            }
        })
        .collect()
}
```

#### GPU-Accelerated Block Matching

```rust
use trueno::backends::gpu::{GpuBackend, GpuMonitor};

/// GPU-accelerated signature table for massive files
#[cfg(feature = "gpu")]
pub struct GpuSignatureTable {
    /// GPU backend handle
    gpu: GpuBackend,
    /// Weak hashes on GPU memory
    weak_hashes: wgpu::Buffer,
    /// Strong hashes on GPU memory
    strong_hashes: wgpu::Buffer,
    /// Block count
    count: u32,
    /// Block size
    block_size: usize,
}

#[cfg(feature = "gpu")]
impl GpuSignatureTable {
    /// Build signature table on GPU
    pub async fn build(data: &[u8], block_size: usize) -> Result<Self, CopiaError> {
        let gpu = GpuBackend::new().await?;
        let monitor = GpuMonitor::new(&gpu);

        // Upload data to GPU
        let data_buffer = gpu.create_buffer_init(data);

        // Execute compute shader for parallel hashing
        let (weak_buf, strong_buf) = gpu.execute_shader(
            include_str!("shaders/signature.wgsl"),
            &data_buffer,
            block_size as u32,
        ).await?;

        Ok(Self {
            gpu,
            weak_hashes: weak_buf,
            strong_hashes: strong_buf,
            count: (data.len() / block_size) as u32,
            block_size,
        })
    }

    /// Find matching blocks using GPU parallel search
    pub async fn find_matches_gpu(
        &self,
        source_data: &[u8],
    ) -> Result<Vec<BlockMatch>, CopiaError> {
        // Upload source to GPU and execute parallel matching
        let source_buf = self.gpu.create_buffer_init(source_data);

        self.gpu.execute_shader(
            include_str!("shaders/block_match.wgsl"),
            &[&source_buf, &self.weak_hashes, &self.strong_hashes],
            self.block_size as u32,
        ).await
    }
}
```

### 16.2 repartir Distributed Execution

Copia integrates with repartir for distributed file synchronization across multi-node clusters.

#### Distributed Sync Pool

```rust
use repartir::{Pool, PoolBuilder, Task, TaskId, Backend, Priority};
use repartir::executor::{Executor, CpuExecutor, GpuExecutor};

/// Distributed copia sync pool
pub struct DistributedSync {
    pool: Pool,
    config: SyncConfig,
}

impl DistributedSync {
    /// Create distributed sync pool with specified workers
    pub async fn new(worker_count: usize) -> Result<Self, CopiaError> {
        let pool = PoolBuilder::new()
            .cpu_workers(worker_count)
            .gpu_enabled(true)
            .build()
            .await?;

        Ok(Self {
            pool,
            config: SyncConfig::default(),
        })
    }

    /// Synchronize multiple files in parallel across workers
    pub async fn sync_batch(
        &self,
        jobs: Vec<SyncJob>,
    ) -> Result<Vec<SyncResult>, CopiaError> {
        let tasks: Vec<_> = jobs.into_iter()
            .map(|job| {
                CopiaTask::new(job.source, job.dest, self.config.clone())
            })
            .collect();

        let task_ids = self.pool.submit_batch(tasks).await?;

        // Wait for all tasks with progress tracking
        let results = self.pool.wait_all(task_ids).await?;

        results.into_iter()
            .map(|r| r.into())
            .collect()
    }
}

/// Copia sync task for repartir execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopiaTask {
    pub source: PathBuf,
    pub dest: PathBuf,
    pub config: SyncConfig,
}

impl repartir::Task for CopiaTask {
    type Output = SyncStats;
    type Error = CopiaError;

    fn backend_hint(&self) -> Backend {
        // Use GPU for large files
        if self.estimated_size() > 100 * 1024 * 1024 {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }

    fn priority(&self) -> Priority {
        Priority::Normal
    }

    async fn execute(self) -> Result<Self::Output, Self::Error> {
        let sync = AsyncCopiaSync::with_config(self.config);
        sync.sync_files(&self.source, &self.dest).await
    }
}
```

#### Multi-Node Cluster Synchronization

```rust
use repartir::executor::RemoteExecutor;
use repartir::messaging::TlsConfig;

/// Cluster-wide file synchronization coordinator
pub struct ClusterSync {
    /// Local pool for coordination
    coordinator: Pool,
    /// Remote worker connections
    workers: Vec<RemoteExecutor>,
}

impl ClusterSync {
    /// Connect to cluster workers
    pub async fn connect(
        worker_addrs: &[SocketAddr],
        tls_config: TlsConfig,
    ) -> Result<Self, CopiaError> {
        let workers = futures::future::try_join_all(
            worker_addrs.iter().map(|addr| {
                RemoteExecutor::connect(*addr, tls_config.clone())
            })
        ).await?;

        Ok(Self {
            coordinator: Pool::local().await?,
            workers,
        })
    }

    /// Replicate file to all cluster nodes
    pub async fn replicate(
        &self,
        source: &Path,
        remote_dest: &Path,
    ) -> Result<ReplicationStats, CopiaError> {
        // Generate signature once
        let sync = AsyncCopiaSync::new();
        let sig = sync.signature_file(source).await?;

        // Distribute delta computation to each worker
        let delta_tasks: Vec<_> = self.workers.iter()
            .map(|worker| {
                DeltaTask {
                    signature: sig.clone(),
                    dest_path: remote_dest.to_path_buf(),
                }
            })
            .collect();

        // Execute in parallel across cluster
        let results = self.coordinator
            .distribute(delta_tasks, &self.workers)
            .await?;

        Ok(ReplicationStats::aggregate(results))
    }
}
```

#### Locality-Aware Scheduling

```rust
use repartir::scheduler::{Scheduler, LocalityMetrics, DataLocationTracker};

/// Data-aware sync scheduler
pub struct LocalityAwareSync {
    scheduler: Scheduler,
    data_tracker: DataLocationTracker,
}

impl LocalityAwareSync {
    /// Schedule sync tasks with data locality optimization
    pub async fn schedule_with_locality(
        &self,
        tasks: Vec<CopiaTask>,
    ) -> Result<Vec<TaskId>, CopiaError> {
        // Track data locations
        for task in &tasks {
            self.data_tracker.register(
                task.source.clone(),
                self.detect_node(&task.source),
            );
        }

        // Schedule with locality hints
        let scheduled = self.scheduler
            .schedule_with_locality(tasks, &self.data_tracker)
            .await?;

        // Log locality metrics
        let metrics = self.scheduler.locality_metrics();
        tracing::info!(
            tasks_with_locality = metrics.tasks_with_locality,
            total_tasks = metrics.total_tasks,
            "Locality-aware scheduling complete"
        );

        Ok(scheduled)
    }
}
```

### 16.3 trueno-zram Compression

Copia uses trueno-zram for delta compression, reducing bandwidth for transmission.

#### Compressed Delta Protocol

```rust
use trueno_zram::{PageCompressor, CompressorBuilder, Algorithm, SimdBackend};
use trueno_zram::CompressedPage;

/// Delta compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: Algorithm,
    /// SIMD backend for compression
    pub simd_backend: SimdBackend,
    /// Minimum size to compress (bytes)
    pub min_compress_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Lz4, // Fast default
            simd_backend: trueno_zram::best_backend(),
            min_compress_size: 512,
        }
    }
}

/// Compressed delta for efficient transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedDelta {
    /// Original delta size
    pub original_size: u64,
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression algorithm used
    pub algorithm: Algorithm,
    /// Compression ratio achieved
    pub ratio: f32,
}

impl CompressedDelta {
    /// Compress a delta using trueno-zram
    pub fn compress(delta: &Delta, config: &CompressionConfig) -> Result<Self, CopiaError> {
        let serialized = bincode::serialize(delta)?;

        if serialized.len() < config.min_compress_size {
            // Don't compress small deltas
            return Ok(Self {
                original_size: serialized.len() as u64,
                data: serialized,
                algorithm: Algorithm::None,
                ratio: 1.0,
            });
        }

        let compressor = CompressorBuilder::new()
            .algorithm(config.algorithm)
            .backend(config.simd_backend)
            .build()?;

        let compressed = compressor.compress_buffer(&serialized)?;
        let ratio = compressed.len() as f32 / serialized.len() as f32;

        Ok(Self {
            original_size: serialized.len() as u64,
            data: compressed,
            algorithm: config.algorithm,
            ratio,
        })
    }

    /// Decompress delta
    pub fn decompress(&self) -> Result<Delta, CopiaError> {
        let decompressed = match self.algorithm {
            Algorithm::None => self.data.clone(),
            Algorithm::Lz4 | Algorithm::Lz4Hc => {
                trueno_zram::lz4_decompress(&self.data, self.original_size as usize)?
            }
            Algorithm::Zstd { .. } => {
                trueno_zram::zstd_decompress(&self.data)?
            }
            Algorithm::Adaptive => {
                trueno_zram::adaptive_decompress(&self.data)?
            }
        };

        Ok(bincode::deserialize(&decompressed)?)
    }
}
```

#### Streaming Compression Pipeline

```rust
use trueno_zram::streaming::{CompressStream, DecompressStream};
use tokio::io::{AsyncRead, AsyncWrite};

/// Streaming compressed delta transmission
pub struct CompressedStream<W> {
    inner: W,
    compressor: CompressStream,
    stats: CompressionStats,
}

impl<W: AsyncWrite + Unpin> CompressedStream<W> {
    pub fn new(writer: W, algorithm: Algorithm) -> Self {
        Self {
            inner: writer,
            compressor: CompressStream::new(algorithm),
            stats: CompressionStats::default(),
        }
    }

    /// Write delta operation with compression
    pub async fn write_op(&mut self, op: &DeltaOp) -> Result<(), CopiaError> {
        let encoded = bincode::serialize(op)?;
        let compressed = self.compressor.compress_chunk(&encoded)?;

        self.stats.bytes_in += encoded.len() as u64;
        self.stats.bytes_out += compressed.len() as u64;

        self.inner.write_all(&compressed).await?;
        Ok(())
    }

    /// Finalize stream and return stats
    pub async fn finish(mut self) -> Result<CompressionStats, CopiaError> {
        let final_chunk = self.compressor.finish()?;
        self.inner.write_all(&final_chunk).await?;
        self.inner.flush().await?;
        Ok(self.stats)
    }
}
```

### 16.4 GPU/WGPU Compute Pipeline

Copia leverages wgpu for GPU-accelerated operations on large files.

#### WGSL Compute Shaders

```wgsl
// shaders/rolling_checksum.wgsl
// Parallel Adler-32 computation on GPU

struct ChecksumOutput {
    sum_a: u32,
    sum_b: u32,
}

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<ChecksumOutput>;

const WORKGROUP_SIZE: u32 = 256u;
const MOD: u32 = 65521u;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let block_idx = global_id.x;
    let block_size = arrayLength(&data) / arrayLength(&output);
    let start = block_idx * block_size;
    let end = min(start + block_size, arrayLength(&data));

    var sum_a: u32 = 0u;
    var sum_b: u32 = 0u;

    for (var i = start; i < end; i++) {
        sum_a = (sum_a + data[i]) % MOD;
        sum_b = (sum_b + sum_a) % MOD;
    }

    output[block_idx].sum_a = sum_a;
    output[block_idx].sum_b = sum_b;
}
```

```wgsl
// shaders/block_match.wgsl
// Parallel block matching on GPU

struct BlockSignature {
    weak_hash: u32,
    strong_hash: array<u32, 8>, // 256-bit BLAKE3
}

struct MatchResult {
    source_offset: u32,
    basis_index: u32,
    matched: u32, // 0 or 1
}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read> signatures: array<BlockSignature>;
@group(0) @binding(2) var<storage, read_write> matches: array<MatchResult>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let source_block = global_id.x;
    let block_size = 2048u; // Configurable

    // Compute weak hash for this source block
    let weak = compute_weak_hash(source, source_block * block_size, block_size);

    // Search for matching signature
    for (var i = 0u; i < arrayLength(&signatures); i++) {
        if (signatures[i].weak_hash == weak) {
            // Verify with strong hash
            let strong = compute_strong_hash(source, source_block * block_size, block_size);
            if (compare_hash(strong, signatures[i].strong_hash)) {
                matches[source_block].source_offset = source_block * block_size;
                matches[source_block].basis_index = i;
                matches[source_block].matched = 1u;
                return;
            }
        }
    }

    matches[source_block].matched = 0u;
}
```

#### GPU Pipeline Orchestration

```rust
use trueno::backends::gpu::{GpuBackend, TensorView, PartitionView};

/// GPU compute pipeline for copia operations
pub struct GpuPipeline {
    backend: GpuBackend,
    checksum_shader: wgpu::ComputePipeline,
    match_shader: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuPipeline {
    /// Initialize GPU pipeline with compiled shaders
    pub async fn new() -> Result<Self, CopiaError> {
        let backend = GpuBackend::new().await?;

        let checksum_shader = backend.create_compute_pipeline(
            include_str!("shaders/rolling_checksum.wgsl"),
        )?;

        let match_shader = backend.create_compute_pipeline(
            include_str!("shaders/block_match.wgsl"),
        )?;

        Ok(Self {
            backend,
            checksum_shader,
            match_shader,
            bind_group_layout: backend.default_bind_group_layout(),
        })
    }

    /// Generate signatures on GPU
    pub async fn generate_signatures(
        &self,
        data: &[u8],
        block_size: usize,
    ) -> Result<Vec<BlockSignature>, CopiaError> {
        let num_blocks = (data.len() + block_size - 1) / block_size;

        // Upload data to GPU
        let data_buffer = self.backend.create_buffer_init(data);
        let output_buffer = self.backend.create_buffer(
            num_blocks * std::mem::size_of::<BlockSignature>()
        );

        // Create bind group
        let bind_group = self.backend.create_bind_group(
            &self.bind_group_layout,
            &[&data_buffer, &output_buffer],
        );

        // Dispatch compute shader
        let workgroups = (num_blocks as u32 + 255) / 256;
        self.backend.dispatch(
            &self.checksum_shader,
            &bind_group,
            workgroups,
            1,
            1,
        ).await?;

        // Read back results
        let results = self.backend.read_buffer(&output_buffer).await?;
        Ok(results)
    }

    /// Find block matches on GPU
    pub async fn find_matches(
        &self,
        source: &[u8],
        signatures: &[BlockSignature],
        block_size: usize,
    ) -> Result<Vec<BlockMatch>, CopiaError> {
        let num_source_blocks = (source.len() + block_size - 1) / block_size;

        // Upload to GPU
        let source_buf = self.backend.create_buffer_init(source);
        let sig_buf = self.backend.create_buffer_init(bytemuck::cast_slice(signatures));
        let match_buf = self.backend.create_buffer(
            num_source_blocks * std::mem::size_of::<MatchResult>()
        );

        // Dispatch matching shader
        let workgroups = (num_source_blocks as u32 + 255) / 256;
        self.backend.dispatch(
            &self.match_shader,
            &self.backend.create_bind_group(
                &self.bind_group_layout,
                &[&source_buf, &sig_buf, &match_buf],
            ),
            workgroups,
            1,
            1,
        ).await?;

        // Read and filter matches
        let results: Vec<MatchResult> = self.backend.read_buffer(&match_buf).await?;
        Ok(results.into_iter()
            .filter(|m| m.matched == 1)
            .map(Into::into)
            .collect())
    }
}
```

### 16.5 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COPIA + TRUENO ECOSYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           COPIA CORE                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │  Signature   │  │    Delta     │  │    Patch     │  │  Protocol  │  │   │
│  │  │  Generator   │  │  Computer    │  │  Applicator  │  │   Codec    │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │   │
│  └─────────┼─────────────────┼─────────────────┼────────────────┼─────────┘   │
│            │                 │                 │                │             │
│            ▼                 ▼                 ▼                ▼             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      ACCELERATION LAYER                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │   trueno     │  │   trueno     │  │  trueno-zram │  │  repartir  │  │   │
│  │  │    SIMD      │  │     GPU      │  │ Compression  │  │Distributed │  │   │
│  │  │  Backends    │  │   Backend    │  │   Engine     │  │   Pool     │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│            │                 │                 │                │             │
│            ▼                 ▼                 ▼                ▼             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       HARDWARE TARGETS                                   │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌───────┐ │   │
│  │  │ SSE2   │  │ AVX2   │  │AVX-512 │  │ NEON   │  │ WGPU   │  │Remote │ │   │
│  │  │ x86    │  │ x86    │  │ x86    │  │ ARM    │  │Vulkan/ │  │Workers│ │   │
│  │  │        │  │        │  │        │  │        │  │Metal/  │  │       │ │   │
│  │  │        │  │        │  │        │  │        │  │DX12    │  │       │ │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └───────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 16.6 Feature Matrix

| Feature | trueno | repartir | trueno-zram | wgpu |
|---------|--------|----------|-------------|------|
| Rolling checksum SIMD | ✓ | | | |
| BLAKE3 parallel | ✓ | | | |
| GPU signature gen | ✓ | | | ✓ |
| GPU block matching | ✓ | | | ✓ |
| Distributed tasks | | ✓ | | |
| Work-stealing | | ✓ | | |
| Remote execution | | ✓ | | |
| Delta compression | | | ✓ | |
| LZ4/Zstd SIMD | | | ✓ | |
| Adaptive algorithm | | | ✓ | |
| CUDA acceleration | ✓ | ✓ | ✓ | |
| WebGPU support | ✓ | | | ✓ |
| WASM SIMD | ✓ | | | |

### 16.7 Cargo Feature Flags

```toml
[features]
default = ["async"]

# Core features
async = ["tokio"]
simd = ["trueno"]

# GPU acceleration
gpu = ["trueno/gpu", "wgpu"]
gpu-wasm = ["trueno/gpu-wasm"]
cuda = ["trueno/cuda-monitor"]

# Distributed execution
distributed = ["repartir/cpu"]
distributed-gpu = ["repartir/gpu", "gpu"]
distributed-remote = ["repartir/remote"]
distributed-tls = ["repartir/remote-tls"]

# Compression
compression = ["trueno-zram"]
compression-cuda = ["trueno-zram/cuda"]

# Full ecosystem
full = [
    "async",
    "simd",
    "gpu",
    "distributed",
    "distributed-gpu",
    "compression",
]

# Development/testing
tui = ["trueno/tui-monitor", "repartir/tui"]
tracing = ["trueno/tracing"]
```

---

## 17. References

[1] A. Tridgell and P. Mackerras, "The rsync algorithm," Technical Report TR-CS-96-05, Australian National University, 1996.

[2] M. O. Rabin, "Fingerprinting by random polynomials," Center for Research in Computing Technology, Harvard University, Tech. Rep. TR-15-81, 1981.

[3] J. O'Connor, J.-P. Aumasson, S. Neves, and Z. Wilcox-O'Hearn, "BLAKE3: One function, fast everywhere," 2020. [Online]. Available: https://github.com/BLAKE3-team/BLAKE3-specs

[4] G. Banga, J. C. Mogul, and P. Druschel, "A scalable and explicit event delivery mechanism for UNIX," in Proceedings of the 1999 USENIX Annual Technical Conference, 1999.

[5] G. Langdale and D. Lemire, "Parsing gigabytes of JSON per second," The VLDB Journal, vol. 28, no. 6, pp. 941-960, 2019.

[6] R. Jung, J.-H. Jourdan, R. Krebbers, and D. Dreyer, "RustBelt: Securing the foundations of the Rust programming language," Proceedings of the ACM on Programming Languages, vol. 2, no. POPL, pp. 1-34, 2017.

[7] D. E. Knuth, "The Art of Computer Programming, Volume 3: Sorting and Searching," Addison-Wesley, 1973.

[8] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, "Introduction to Algorithms," MIT Press, 3rd ed., 2009.

[9] P. Valiant, "Universal Hash Functions," 2021. [Online]. Available: https://www.cs.purdue.edu/homes/ninghui/courses/Fall04/lectures/lect19.pdf

[10] NIST, "Secure Hash Standard (SHS)," FIPS PUB 180-4, 2015.

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-04 | Pragmatic AI Labs | Initial specification |
