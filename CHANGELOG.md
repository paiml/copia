# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.5] - 2026-07-04

Copia grows from a single-file rsync delta engine into a three-layer
**distributed file-sync system** — one-way incremental mirror (L1),
two-way bidirectional sync (L2), and a hub daemon (L3) — with the core
safety invariants **machine-proved** in Lean 4.

### Added

#### L1 — Incremental one-way mirror (`copia sync -r`)

- **Recursive incremental mirror** in all three directions from one engine:
  local→local, **push** (local→remote over SSH), and **pull**
  (remote→local over SSH). The direction is inferred from whether each
  endpoint parses as a local path or `host:path`.

  ```bash
  copia sync -r ./site  web:/srv/site       # push
  copia sync -r web:/srv/site  ./site       # pull  (remote -> local)
  copia sync -r ./a ./b                      # local -> local
  ```

  (`remote→remote` is explicitly rejected.)
- **rsync-style quick check** — a file is transferred only when the
  destination lacks it or its **size or mtime differ**; identical files
  are skipped. mtime is compared at 1-second granularity (matching rsync's
  default posture). Content hashing (blake3) — never mtime alone — remains
  the oracle for what a file *is*.
- **Atomic delivery** — every file streams to a `<path>.copia-tmp` sibling
  and is `rename`d into place, so a reader never observes a half-written
  file and an interrupted run leaves no partial destination file.
- **mtime preservation** — the source mtime is restored on the delivered
  file so subsequent quick checks stay stable.
- **`--delete` (opt-in mirror)** — removes destination files no longer
  present in the (filtered) source. Off by default. Pull deletes locally;
  push batches removals through a single `xargs rm` over SSH.
- **`--exclude GLOB` (repeatable)** — gitignore-style filtering. A
  slash-free pattern (e.g. `target`, `*.tmp`) prunes any matching path
  *component* anywhere in the tree; a pattern containing `/` matches the
  whole relative path as a glob. Excluded destination paths are also
  protected from `--delete`.
- **`--dry-run` / `-n`** — prints the full `send` / `delete` plan and a
  `to transfer / unchanged / to delete` summary without modifying either
  side.
- **`-j/--jobs N`** parallel transfers (default 4), bounded by a semaphore.

  ```bash
  copia sync -r ./src web:/srv/app --delete --exclude target --exclude '*.tmp' -n
  ```

#### L2 — Bidirectional two-way sync (`copia bisync`)

- **`copia bisync A B`** — two-way sync of two local directories via a
  **blake3 three-way reconcile** against a per-pair archive of the
  last-synced state (persisted under `~/.copia/archive/`,
  epoch + format-versioned, written atomically).
- **Safe deletes** — a delete is only propagated with *positive archive
  evidence* that the file existed at the last sync. If the archive is
  missing or unreadable, `bisync` runs in a conservative no-base mode with
  deletes disabled — an absent base can never be misread as "delete it".
- **Convergent conflict-copy** — when both sides edited a file divergently,
  **both versions are kept on both sides**; the deterministic loser (chosen
  by blake3, **never** by mtime) is written to
  `<path>.conflict-<host>-<hash>`. No edit is ever silently discarded.

  ```bash
  copia bisync ~/notes ~/notes-mirror
  copia bisync ~/notes ~/notes-mirror -n -v      # show the reconcile plan
  ```

#### L3 — Hub daemon + CAS client (`copia serve`, `copia hub-sync`)

- **`copia serve ROOT`** — hub daemon speaking a **framed CBOR protocol**
  (ciborium) on stdin/stdout. Clients spawn it remotely as
  `ssh <host> copia serve <root>`. The hub commits content-addressed
  (CAS-on-blake3) writes under a brief per-tree commit lock (`fs2`), with a
  `COPIA1` handshake-magic prologue guard, bounded (1 MiB) frames, a
  path-traversal guard, and end-to-end content-integrity checks.
- **`copia hub-sync LOCAL TARGET`** — the hub client. `TARGET` is
  `host:root` (over SSH) or a local hub path (spawns a local `serve`). It
  pushes the local tree over one persistent connection using
  **compare-and-swap**: a write commits only if the hub's current hash
  matches what the client last saw. A stale CAS lands a **conflict-copy on
  the hub — never a lost update**, so N clients can safely sync into one hub
  (star topology).

  ```bash
  # server side is spawned automatically by the client:
  copia hub-sync ./work  hub-host:/srv/hub      # push to a remote hub over SSH
  copia hub-sync ./work  /srv/local-hub          # or a local hub path
  ```

### Proved (provable-contracts reach L5)

- `incremental-sync-v1`, `bidirectional-sync-v1`, and `hub-protocol-v1`
  climb the full provable-contracts (`pv`) ladder to **L5** — the top of
  the ladder and the only L5 contracts in the fleet:
  L1 equations → L2 falsification `#[test]`s → L3 **Kani** harnesses
  (exhaustive, e.g. `plan-kani-001` for the quick check) → L4 **Lean 4**
  proofs (**11 theorems** across `lean/*.lean`, **0 `sorry`**, checked by
  `lean`) → L5 all bindings verified against source
  (`contracts/binding.yaml`, checked by `pv proof-status --binding`).
- Machine-proved invariants include **`NoBaseNeverDeletes`** (a lost
  archive can never turn a create into a delete), **`DeleteNeedsEvidence`**,
  **`Blake3Oracle`** / **`ConflictNotSilentPick`** (identical content is
  never a conflict; divergent content is never silently picked),
  **`StaleCasNeverCommits`** (a stale CAS can never silently lose a
  concurrent write), **`BoundedFrame`**, and **`NoTraversal`**. The L1
  suite proves `QuickCheckCorrect`, `SkipGuarantee`, `DeleteOptIn`, and
  `ExcludeSafety`.
- Enforce the whole ladder with `make contracts`.

### Quality

- 95%+ line coverage, clippy pedantic clean, TDG gates. No `unsafe` in
  copia itself. The build binary lands under a redirected target dir.

## [0.1.4] - 2026-06-25

### Changed

- Dependency refresh (tokio 1.52, serde_json 1.0.150, thiserror 2.0.18,
  tracing-subscriber 0.3.23, and transitive updates) — `cargo deny check`
  passes clean (advisories/bans/licenses/sources all ok)
- Resolved Rust 1.93 clippy lints (`unnecessary_map_or`, `io::Error::other`,
  `needless_collect`, `redundant_closure_for_method_calls`) in tests and benches

### Contracts

- Added `qa_gate` blocks to `hash-integrity-v1` and `delta-sync-v1` provable
  contracts (now `pv validate` clean with 0 errors / 0 warnings)

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
