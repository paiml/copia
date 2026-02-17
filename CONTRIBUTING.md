# Contributing to Copia

## Development Setup

```bash
# Clone and build
git clone https://github.com/paiml/copia
cd copia
cargo build --all-features

# Run tests
cargo test --all-features

# Run lints
cargo clippy --all-features -- -D warnings
cargo fmt --check
```

## Quality Gates

All contributions must pass:

- `cargo clippy --all-features -- -D warnings`
- `cargo test --all-features` (283+ tests)
- `cargo llvm-cov --all-features --fail-under-lines 95`
- `cargo fmt --check`

## Commit Convention

Use conventional commits: `type: subject`

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `ci`
