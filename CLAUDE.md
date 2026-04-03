# Copia Project Guidelines

## Contract-First Design

This project follows contract-first development with provable-contracts.
Contracts live in `../provable-contracts/contracts/copia/`.
Run `pmat comply check` to validate contract compliance.

## Overview
Copia is a pure Rust rsync-style delta sync library providing efficient
file synchronization with content-aware chunking and bandwidth optimization.

## Code Search Policy

**CRITICAL: NEVER use grep or grep-like tools for code search. ALWAYS use `pmat query`.**

`pmat query` returns semantically ranked results with TDG grades, complexity, and quality metrics. Examples:

```bash
# Find functions by intent
pmat query "delta encoding" --min-grade A

# Find code with fault patterns (unwrap, panic, unsafe)
pmat query "chunk" --faults --exclude-tests

# Search with git history
pmat query "sync" -G

# Regex search (e.g., function patterns)
pmat query --regex "fn\s+chunk_\w+" --limit 10

# Coverage gaps (mandatory for testing work)
pmat query --coverage-gaps --limit 20 --exclude-tests
```

## Development Commands

```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo build --all-features
cargo doc --all-features --no-deps --open
```

## Quality Standards

- **Test Coverage:** 95% minimum (use `pmat query --coverage-gaps` to find gaps)
- **Code Quality:** TDG grade A required
- **Unsafe Code:** Denied (`#![deny(unsafe_code)]`)
- **Feature Flags:** `default`, `async`, `contracts`, `tracing`, `cli`

## Testing Coverage Work

1. Run: `pmat query --coverage-gaps --limit 30 --exclude-tests`
2. Pick highest impact functions first
3. View source: `pmat query "function_name" --include-source --limit 1`
4. Write tests and verify coverage improvement

## Key Principles

- No unsafe code unless absolutely justified
- All public APIs require comprehensive tests
- Use feature flags for optional functionality
- Contract verification for critical paths
