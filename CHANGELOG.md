# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.3] - 2025-02-17

### Added

- Structured tracing with renacer-compatible NDJSON output
- `--trace-output` CLI flag for performance analysis
- `tracing` feature flag for zero-cost library instrumentation
- Architecture Decision Records (ADR) documentation
- Dockerfile and flake.nix for reproducible builds

### Changed

- Refactored CLI sync functions for lower cognitive complexity
- README restructured with Usage section and statistical methodology

## [0.1.2] - 2025-02-10

### Added

- Async file synchronization with tokio
- Parallel signature generation with rayon
- Wire protocol for network transfers
- CLI with recursive directory sync

## [0.1.1] - 2025-02-01

### Added

- Property-based tests with proptest
- Criterion benchmarks with rsync comparison
- Mutation testing support

## [0.1.0] - 2025-01-15

### Added

- Initial release
- Rolling checksum (Adler-32 variant)
- Strong hash (BLAKE3)
- Delta encoding and patch application
- Configurable block sizes (512-65536)
