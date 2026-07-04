<h1 align="center">copia</h1>

<p align="center"><strong>A sovereign, pure-Rust rsync replacement and distributed file-sync tool — proved correct to L5.</strong></p>

[![Crates.io](https://img.shields.io/crates/v/copia.svg)](https://crates.io/crates/copia)
[![Documentation](https://docs.rs/copia/badge.svg)](https://docs.rs/copia)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/paiml/copia/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/copia/actions/workflows/ci.yml)

## What is copia?

`copia` is two things in one binary:

1. **A pure-Rust rsync replacement** — the classic rolling-checksum + BLAKE3 delta-transfer engine, with no C dependencies (no `librsync`, no OpenSSL), no `unsafe` in copia itself, and native async I/O.
2. **A distributed file-sync system** — one-way incremental mirroring, two-way (bidirectional) sync with conflict-safe reconciliation, and a hub daemon for star-topology sync across many clients.

### Why copia?

- **`blake3` is the sole content oracle.** `mtime` *never* decides whether a file changed, who wins a conflict, or what gets deleted. Content hashes do.
- **Data is never lost.** Bidirectional conflicts keep **both** versions on **both** sides. Deletes require positive archive evidence — a lost archive disables deletes rather than guessing.
- **Every write is atomic.** Files land in a sibling `.copia-tmp` and are `rename(2)`'d into place — a reader ever sees the old file or the new file, never a partial one.
- **Concurrency is safe.** Hub writes use compare-and-swap on the BLAKE3 hash: a stale write becomes a conflict-copy, never a lost update.
- **The safety is machine-proved.** These are not aspirations — they are checked invariants (see [Provable to L5](#provable-to-l5)).

## Feature matrix

| Subcommand | Layer | What it does |
|------------|-------|--------------|
| `sync` | **L1** — incremental one-way | Single-file rsync delta, or a recursive incremental **mirror** (`local→local`, `push local→remote`, `pull remote→local` over SSH). Quick-check skip (size **or** mtime differ), atomic temp+rename delivery, mtime preservation, opt-in `--delete`, gitignore-style `--exclude`, `--dry-run`. |
| `bisync` | **L2** — two-way sync | Local ↔ local BLAKE3 3-way reconcile against a persisted per-pair archive. Safe deletes (need archive evidence; lost archive ⇒ deletes disabled). Convergent **conflict-copy**: divergent edits keep both versions on both sides, deterministic BLAKE3 winner. |
| `serve` | **L3** — hub daemon | Serves a directory over a framed CBOR protocol on stdin/stdout. Clients spawn it as `ssh <host> copia serve <root>`. CAS-on-BLAKE3 commits under a brief tree lock, with a MAGIC prologue guard, bounded frames, and a path-traversal guard. |
| `hub-sync` | **L3** — hub client | Pushes a local tree to a hub (`host:root` over SSH, or a local hub path) over one persistent connection. CAS concurrency safety: N clients → 1 hub, a stale write lands a conflict-copy, never a lost update. |
| `signature` / `delta` / `patch` | primitives | The rsync delta primitives, usable standalone: generate a block signature, compute a delta against it, apply the delta to reconstruct a file. |

## Install

```bash
cargo install copia --features cli
```

The `cli` feature builds the `copia` binary (it pulls in `async` + `tracing`). To embed the delta-transfer engine as a **library**, add `copia = "0.1"` (with `features = ["async"]` for the tokio engine) — see [docs.rs/copia](https://docs.rs/copia).

## Quick start

### L1 — incremental one-way sync (`sync`)

```bash
# Single-file rsync delta
copia sync report.pdf backup/report.pdf

# Recursive incremental mirror, local → local
copia sync ./project ./backup --recursive

# Preview a mirror with deletes and an exclude filter (nothing is written)
copia sync ./project ./backup -r --delete --exclude target --exclude '*.tmp' --dry-run

# Push a tree to a remote host over SSH (host:path)
copia sync ./project server:/srv/project --recursive --jobs 8

# Pull a tree from a remote host
copia sync server:/srv/project ./project --recursive
```

Only new-or-changed files transfer (quick check on size + mtime); each file is delivered atomically and its mtime is preserved so the next run's quick check stays stable.

### L2 — two-way sync (`bisync`)

```bash
# Reconcile two local directories in both directions
copia bisync ~/notes ~/Dropbox/notes

# See the reconcile plan first (no changes to either side)
copia bisync ~/notes ~/Dropbox/notes --dry-run --verbose
```

Divergent edits to the same path are never clobbered: the BLAKE3 loser is written to `<path>.conflict-<host>-<hash>` on both sides, so both versions survive everywhere. Per-pair state lives under `~/.copia/archive/`.

### L3 — hub sync (`serve` + `hub-sync`)

```bash
# On the hub host: run the daemon (usually spawned for you over SSH)
copia serve /srv/hub

# From each client: push your tree to the hub over SSH
copia hub-sync ./project server:/srv/hub

# ...or against a local hub path (spawns a local serve for you)
copia hub-sync ./project /srv/hub
```

Many clients push to one hub in a star topology. Each commit is compare-and-swapped on the hub's current BLAKE3 hash — a client whose view is stale lands a conflict-copy instead of silently overwriting a concurrent write.

### Delta primitives

```bash
copia signature old.bin -o old.sig       # block signature of the basis file
copia delta new.bin old.sig -o new.delta # delta of new.bin against the signature
copia patch old.bin new.delta -o new.bin # reconstruct new.bin from basis + delta
```

## Architecture

copia is layered so that the hard decisions live in small, pure, testable cores:

- **L0 transport** — local filesystem and SSH (`cat` / `find -printf` / `cat > tmp && mv`).
- **L1 incremental** — `plan.rs` (pure planner), `meta.rs` (size + mtime + BLAKE3 fingerprints), `incremental.rs` (orchestration + atomic delivery), `single_sync.rs`, `transfer.rs` / `dir_sync.rs`.
- **L2 bidirectional** — `reconcile.rs` (pure 3-way case table), `archive.rs` (versioned per-pair state), `bidir.rs`.
- **L3 hub** — `wire.rs` (framed CBOR + `cas_decide`), `serve.rs` (daemon + CAS commit under an `fs2` lock), `hub.rs` (client).

See **[docs/architecture.md](docs/architecture.md)** for the full picture.

## Provable to L5

copia's safety guarantees are not just tested — they are **machine-proved**. Three contracts (`incremental-sync-v1`, `bidirectional-sync-v1`, `hub-protocol-v1`) reach **L5**, the top of the provable-contracts (`pv`) ladder:

| Rung | Meaning |
|------|---------|
| **L1** | Equations declared |
| **L2** | Falsification `#[test]`s |
| **L3** | Kani harnesses **verified** (`cargo kani`, exhaustive) |
| **L4** | Lean 4 proofs (**11 theorems** in `lean/*.lean`, **0 `sorry`**, checked by `lean`) |
| **L5** | Every equation bound to real source (`contracts/binding.yaml`, checked by `pv proof-status --binding`) |

Machine-proved invariants include:

- **`NoBaseNeverDeletes`** — a lost archive can never turn a *create* into a *delete*.
- **`StaleCasNeverCommits`** — a stale compare-and-swap can never silently lose a concurrent write.

Enforce it all with `make contracts`. Details in **[docs/proofs.md](docs/proofs.md)**.

## Learn more

- **[docs/usage.md](docs/usage.md)** — full command reference and worked examples.
- **[docs/specifications/rsync-copia-spec.md](docs/specifications/rsync-copia-spec.md)** — the L1 rsync-engine specification.
- **[docs/specifications/distributed-sync.md](docs/specifications/distributed-sync.md)** — the L2/L3 distributed-sync specification (quorum-vetted against Unison, Syncthing, git, Dynamo/Cassandra, etcd, rsync).

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- The rsync algorithm by Andrew Tridgell and Paul Mackerras.
- The BLAKE3 team for the fast cryptographic hash.
- The Rust community for excellent tooling.
