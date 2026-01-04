# Copia Makefile - Iron Lotus + Certeza Quality Framework
# Three-Tier Testing Methodology

.PHONY: all build test lint fmt clean tier1 tier2 tier3 coverage bench doc

# Default target
all: tier2

# =============================================================================
# TIER 1: On-Save (Sub-second feedback)
# =============================================================================

tier1: fmt-check check clippy
	@echo "✅ Tier 1 passed"

check:
	cargo check --all-features

clippy:
	cargo clippy --all-features -- -D warnings

fmt-check:
	cargo fmt --check

# =============================================================================
# TIER 2: On-Commit (1-5 minutes)
# =============================================================================

tier2: tier1 test coverage-check
	@echo "✅ Tier 2 passed"

test:
	cargo test --all-features

# Coverage with 95% threshold (Certeza requirement)
coverage:
	cargo llvm-cov --all-features --html

coverage-check:
	cargo llvm-cov --all-features --fail-under-lines 95

# =============================================================================
# TIER 3: On-Merge (Exhaustive validation)
# =============================================================================

tier3: tier2 mutants doc
	@echo "✅ Tier 3 passed"

# Mutation testing with 80% threshold
mutants:
	cargo mutants --minimum-coverage 80 -- --all-features

# =============================================================================
# BUILD & RELEASE
# =============================================================================

build:
	cargo build --release

build-debug:
	cargo build

# =============================================================================
# DEVELOPMENT
# =============================================================================

fmt:
	cargo fmt

fix:
	cargo clippy --fix --allow-dirty --allow-staged

watch:
	cargo watch -x "check --all-features"

doc:
	cargo doc --no-deps --all-features

doc-open:
	cargo doc --no-deps --all-features --open

# =============================================================================
# BENCHMARKS
# =============================================================================

bench:
	cargo bench

bench-baseline:
	cargo bench -- --save-baseline main

bench-compare:
	cargo bench -- --baseline main

# =============================================================================
# MAINTENANCE
# =============================================================================

clean:
	cargo clean
	rm -rf target/
	rm -rf mutants.out/

update:
	cargo update

audit:
	cargo audit

tree:
	cargo tree

# =============================================================================
# QUALITY VALIDATION (pmat/certeza)
# =============================================================================

certeza:
	@echo "Running Certeza quality validation..."
	@if [ -d "../certeza" ]; then \
		cd ../certeza && cargo run -- check ../copia; \
	else \
		echo "⚠️  Certeza not found at ../certeza"; \
	fi

pmat:
	@echo "Running pmat analysis..."
	@if command -v pmat > /dev/null 2>&1; then \
		pmat rust-project-score .; \
	else \
		echo "⚠️  pmat not installed"; \
	fi

# =============================================================================
# CI/CD TARGETS
# =============================================================================

ci: tier2
	@echo "✅ CI checks passed"

ci-full: tier3
	@echo "✅ Full CI checks passed"

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "Copia - Pure Rust rsync-style synchronization"
	@echo ""
	@echo "Tier 1 (On-Save):"
	@echo "  make tier1      - Quick checks (fmt, check, clippy)"
	@echo ""
	@echo "Tier 2 (On-Commit):"
	@echo "  make tier2      - Tests + coverage (95% required)"
	@echo "  make coverage   - Generate coverage report"
	@echo ""
	@echo "Tier 3 (On-Merge):"
	@echo "  make tier3      - Mutation testing (80% required)"
	@echo "  make mutants    - Run mutation tests"
	@echo ""
	@echo "Development:"
	@echo "  make build      - Release build"
	@echo "  make test       - Run tests"
	@echo "  make bench      - Run benchmarks"
	@echo "  make doc        - Generate documentation"
	@echo "  make fmt        - Format code"
	@echo "  make clean      - Clean build artifacts"
	@echo ""
	@echo "Quality:"
	@echo "  make certeza    - Run Certeza validation"
	@echo "  make pmat       - Run pmat analysis"
