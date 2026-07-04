# Copia Makefile - Iron Lotus + Certeza Quality Framework
# Three-Tier Testing Methodology

.PHONY: contracts bench bench-baseline bench-compare coverage coverage-check
.PHONY: all build test test-fast lint fmt clean tier1 tier2 tier3 coverage bench doc

# Default target
all: tier2

# =============================================================================
# TIER 1: On-Save (Sub-second feedback)
# =============================================================================

tier1: fmt-check check clippy
	@echo "✅ Tier 1 passed"

test-fast: ## Fast lib tests only
	cargo test --lib

lint: clippy fmt-check ## Run all linters

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
	PROPTEST_CASES=256 QUICKCHECK_TESTS=256 cargo test --all-features

# Coverage with 95% threshold (Certeza requirement)
coverage:
	cargo llvm-cov --all-features --html --ignore-filename-regex 'bin/'

# Whole-crate 95% floor — matches CI exactly (--lib + --bins + --tests, cli
# feature). tests/e2e_ssh.rs drives the binary against `ssh localhost` so the
# CLI + SSH shims are covered by integration; NO bin/ exclusion.
coverage-check:
	cargo llvm-cov nextest --lib --bins --tests --features cli --no-cfg-coverage --lcov --output-path target/copia-lcov.info
	@awk -F'[:,]' '/^DA:/{t++; if($$3>0)c++} END{p=(c/t)*100; printf "DA-line coverage: %.2f%%\n", p; if(p<95){print "FAIL: below the 95% floor"; exit 1}}' target/copia-lcov.info

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

# --features async: the sync-throughput + rsync-comparison benches drive the
# async transfer engine (required-features = ["async"]).
bench:
	cargo bench --features async

bench-baseline:
	cargo bench --features async -- --save-baseline main

bench-compare:
	cargo bench --features async -- --baseline main

# =============================================================================
# PROVABLE CONTRACTS
# =============================================================================

# Validate every contract's schema, machine-check the Lean (L4) proofs, verify
# the binding registry against source and report proof levels (expect L5), then
# run the falsification tests that encode each contract's predictions (L2).
contracts:
	@for c in contracts/*-v1.yaml; do pv validate "$$c" || exit 1; done
	@echo "== Lean 4 proofs (L4) =="; for l in lean/*.lean; do lean "$$l" || exit 1; done
	@echo "== proof levels (expect L5) =="; pv proof-status contracts/ --binding contracts/binding.yaml
	cargo test --features cli --test contract_falsification --test e2e_bidir --test e2e_hub --test integration_all

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
