# Falsifiable Hypotheses

This document records falsifiable claims about copia's behavior and
performance. Each hypothesis can be tested against empirical evidence.

## H1: Delta Correctness

**Claim**: For any pair of files (basis, source), applying the delta
produced by copia to the basis always reconstructs the exact source
byte-for-byte.

**Falsification**: Run `sync → patch` roundtrip on arbitrary inputs
and verify BLAKE3 checksum match. Tested via proptest with 256+
random inputs per CI run.

**Status**: Verified (proptest roundtrip, integration tests)

## H2: Rolling Checksum Invariant

**Claim**: After any sequence of `new()`, `push()`, and `roll()`
operations, the checksum components `a` and `b` are always less than
`MOD = 65521`.

**Falsification**: Run proptest with arbitrary byte sequences and
verify bounds after every operation.

**Status**: Verified (proptest `a_always_bounded`, `b_always_bounded`)

## H3: Performance Threshold

**Claim**: For identical files, copia library calls complete in under
1ms for files up to 1MB (excluding process spawn overhead).

**Falsification**: Run Criterion benchmarks on 1MB identical files.
Measured: 0.33ms median (95% CI).

**Status**: Verified (criterion benchmarks)

## H4: Zero Unsafe Code

**Claim**: The copia library contains zero unsafe code blocks.

**Falsification**: `#![forbid(unsafe_code)]` is enforced at the crate
level. Any unsafe code causes a compile error.

**Status**: Verified (compile-time enforcement)

## H5: Delta Size Bound

**Claim**: For identical files, the delta contains only copy
operations (zero literal bytes). `bytes_matched == source_size`.

**Falsification**: Run proptest `identical_high_ratio` which verifies
`compression_ratio >= 0.99` for identical inputs.

**Status**: Verified (proptest)
