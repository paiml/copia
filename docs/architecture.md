# copia Architecture

`copia` is a pure-Rust "sovereign" `rsync` replacement **and** a distributed
file-sync system. It has no `unsafe` code, ships on crates.io, and — uniquely in
the fleet — carries machine-checked **provable contracts** (Kani + Lean 4) for
its safety-critical decision logic.

This document maps the layered architecture, names the module that implements
each piece, and traces the data flow (with small ASCII diagrams) for the three
sync modes. The distributed design was vetted by a 6-system quorum; see
[`docs/specifications/distributed-sync.md`](specifications/distributed-sync.md).

---

## 1. The layer cake

copia is built as four layers, each reusing the one below. Higher layers never
re-implement transport, planning, hashing, or atomic delivery — they compose the
primitives the lower layers expose.

```
┌──────────────────────────────────────────────────────────────────────┐
│ L3  HUB (star topology, N clients → 1 hub)                            │
│     copia serve <root>            copia hub-sync <local> <target>     │
│     wire.rs · serve.rs · hub.rs   — CAS-on-blake3, CBOR protocol      │
├──────────────────────────────────────────────────────────────────────┤
│ L2  BIDIRECTIONAL (local ↔ local, two-way)                           │
│     copia bisync A B                                                  │
│     reconcile.rs · archive.rs · bidir.rs — 3-way blake3 reconcile     │
├──────────────────────────────────────────────────────────────────────┤
│ L1  INCREMENTAL (one-way mirror)                                      │
│     copia sync SRC DST [-r]                                           │
│     plan.rs · meta.rs · incremental.rs · single_sync.rs              │
│                       · transfer.rs · dir_sync.rs                     │
├──────────────────────────────────────────────────────────────────────┤
│ L0  TRANSPORT + rsync delta primitives                               │
│     local FS  +  SSH (cat / find -printf / cat>tmp && mv)            │
│     library crate: signature.rs · delta.rs · checksum.rs · hash.rs   │
│                    · sync.rs · async_sync.rs · protocol.rs           │
│     copia signature | delta | patch                                  │
└──────────────────────────────────────────────────────────────────────┘
```

**How each layer reuses the one below:**

- **L1 → L0.** The recursive mirror walks the tree, then for a single changed
  file drops to the rsync signature/delta/patch engine in the library crate. All
  transport (local copy, `ssh host cat`, `find -printf`) lives in `transfer.rs` /
  `dir_sync.rs`.
- **L2 → L1.** Bidirectional sync reuses L1's **atomic temp+rename delivery**
  (`copy_atomic` mirrors `incremental::deliver_local`) and the same
  **blake3 fingerprints** produced by `meta::fingerprint_path`.
- **L3 → L2/L1.** The hub reuses `meta::discover_local_fingerprints` (same blake3
  fingerprints), the same atomic `.copia-tmp`+rename staging, and the same
  conflict-copy idea — now gated by a compare-and-swap instead of a 3-way base.

The invariant that ties the stack together: **blake3 is the sole content oracle.**
`size`+`mtime` are only ever a *fast-path* (the L1 quick check deciding whether to
re-transfer); they **never** decide a change, a conflict, or a winner. Any
content/merge decision is made on the blake3 digest.

---

## 2. Module map

### Library crate (`src/`) — L0 rsync delta engine

| Module | Responsibility |
|---|---|
| `async_sync.rs` | `AsyncCopiaSync` — the async signature/delta/patch engine used by the CLI. |
| `signature.rs` | Block signatures (rolling weak checksum + blake3 strong hash per block). |
| `delta.rs` | Delta computation: match source blocks against a signature, emit copy/literal ops. |
| `checksum.rs` | Rolling (Adler-style) weak checksum for the sliding window. |
| `hash.rs` | blake3 strong hashing. |
| `sync.rs` | Sync-engine glue over the primitives. |
| `protocol.rs` | Serialized `Signature` / `Delta` types. |
| `error.rs` | `thiserror` error taxonomy. |
| `trace_output.rs` | Optional renacer-compatible NDJSON trace layer (`--trace-output`). |

These back the standalone `copia signature | delta | patch` subcommands and the
single-file path of `copia sync`.

### CLI binary (`src/bin/copia/`) — L1/L2/L3 orchestration

| Module | Layer | Responsibility |
|---|---|---|
| `main.rs` | — | `clap` CLI, `FileLocation::parse` (`host:path` vs local), dispatch. |
| `plan.rs` | L1 | **Pure planner**: `needs_transfer`, `build_plan`, `is_excluded`, `glob_match`. No I/O. |
| `meta.rs` | L1 | Size+mtime discovery (`discover_*_with_meta`), blake3 fingerprints (`fingerprint_path`), mtime preservation. |
| `incremental.rs` | L1 | Recursive orchestration for all three directions + atomic delivery + `--delete`. |
| `single_sync.rs` | L1 | Single-file (non-recursive) sync via the L0 delta engine. |
| `transfer.rs` | L0/L1 | Transfer primitives: local file discovery, SSH push, remote dir creation, byte formatting. |
| `dir_sync.rs` | L0/L1 | Remote→local streaming, local dir creation, parallel progress counters. |
| `reconcile.rs` | L2 | **Pure 3-way case table**: `reconcile_path` / `reconcile`. No I/O. Kani-proved. |
| `archive.rs` | L2 | Per-pair last-synced state (`~/.copia/archive/<hash>.json`), epoch+format-versioned, atomic save. |
| `bidir.rs` | L2 | Bidirectional orchestration: scan, load archive, reconcile, apply, record. |
| `wire.rs` | L3 | Framed CBOR protocol (`ciborium`), `MAGIC` prologue, bounded frames, **pure `cas_decide`**. Kani-proved. |
| `serve.rs` | L3 | Hub daemon: applies CAS writes under an `fs2` tree commit-lock; path-traversal guard. |
| `hub.rs` | L3 | Hub client: one persistent connection, CAS push. |

The **pure** modules (`plan.rs`, `reconcile.rs`, and `wire::cas_decide`) are
deliberately free of I/O so their decision logic can be exhaustively
model-checked. This is what makes the L5 contracts possible.

---

## 3. Data flow: L1 one-way sync

`copia sync SRC DST` runs the rsync-style quick check and transfers only what
changed. Non-recursive syncs a single file through the delta engine; `-r`
recursively mirrors a whole tree.

```
copia sync SRC DST -r [-j N] [--delete] [--exclude GLOB] [-n] [-v]

  ┌── meta.rs ────────────┐        ┌── meta.rs ────────────┐
  │ discover SRC tree     │        │ discover DST tree     │
  │ (size + mtime)        │        │ (size + mtime)        │
  └───────────┬───────────┘        └───────────┬───────────┘
              │  MetaMap(src)                   │  MetaMap(dst)
              └──────────────┬──────────────────┘
                             ▼
                   ┌── plan.rs ──────────────┐   quick check:
                   │ build_plan(src,dst,...)  │   transfer iff dst lacks it
                   │  → transfer[]            │   OR size ≠ OR mtime ≠
                   │  → skipped               │   excluded paths pruned
                   │  → delete[] (opt-in)     │   (never transferred, never deleted)
                   └───────────┬─────────────┘
                               ▼
       ┌── incremental.rs (≤ N parallel jobs, tokio Semaphore) ──┐
       │ for each file:  stream → <dst>.copia-tmp                │
       │                 fsync/rename → <dst>   (ATOMIC)         │
       │                 set mtime (preserve)                    │
       │ then --delete:  remove dst files absent from src        │
       └─────────────────────────────────────────────────────────┘
```

**Directions** (resolved from the endpoints in `run_sync_recursive`):

- `local → local` — `run_local`, `tokio::fs::copy` into a `.copia-tmp` sibling.
- `push` (`local → host:path`) — `run_remote(Push)`, `ssh host "cat > tmp && mv"`.
- `pull` (`host:path → local`) — `run_remote(Pull)`, `ssh host cat` streamed to a
  local `.copia-tmp`, then renamed.

Remote trees are enumerated in one round-trip with
`ssh host "find . -printf '%s\t%T@\t%p\0'"` (`meta::discover_remote_with_meta`).
mtime is compared at **1-second granularity** (matching rsync's default, because
sub-second precision is unreliable across `find` vs `SystemTime` and filesystems).

**Runnable examples**

```bash
# Recursive mirror, 8 parallel jobs, delete extraneous dest files
copia sync ./site/ /var/www/site/ -r -j 8 --delete

# Push to a remote host over SSH, excluding build junk (dry run first)
copia sync ./project intel:/data/project -r --exclude target --exclude '*.tmp' -n

# Pull from remote
copia sync intel:/data/logs ./logs -r -v

# Single-file rsync delta (non-recursive): the signature/delta/patch engine
copia sync bigfile.bin host:bigfile.bin

# The delta primitives, standalone
copia signature old.bin -o old.sig
copia delta new.bin old.sig -o patch.delta
copia patch old.bin patch.delta -o reconstructed.bin
```

`--dry-run` prints the plan (`send …` / `delete …`) and modifies nothing.

---

## 4. Data flow: L2 bidirectional 3-way reconcile

`copia bisync A B` is a two-way (local ↔ local) sync. It reads a persisted
**archive** (the last-synced common state for this exact `(A, B)` pair) and
reconciles each path against it on blake3. It never loses data: divergent edits
keep **both** versions on **both** sides.

```
copia bisync A B [-n] [-v]

  fingerprint A (blake3)   fingerprint B (blake3)   load archive base
  discover_local_          discover_local_          ~/.copia/archive/<pair>.json
    fingerprints(A)          fingerprints(B)        (pair+format checked;
        │                        │                   mismatch → base = None,
        └──────────┬─────────────┘                   SAFE no-base mode, no deletes)
                   ▼
        ┌── reconcile.rs — pure per-path case table ──────────────────┐
        │  (a, b, base) →                                             │
        │    a==b==base ............... Noop                          │
        │    a==b≠base ................ ConvergeIdentical (record)     │
        │    only a changed ........... PropagateAtoB                  │
        │    only b changed ........... PropagateBtoA                  │
        │    both changed, a≠b ........ Conflict(BothChanged)          │
        │    b gone, a==base .......... DeleteB   (positive evidence)  │
        │    b gone, a≠base ........... Conflict(DeleteVsModify)       │
        │    absent + base==None ...... Propagate (CREATE, never del)  │
        └───────────────────────────┬────────────────────────────────┘
                                     ▼
        ┌── bidir.rs — apply, then commit-then-record ──────────────┐
        │ propagate: copy_atomic (temp + rename)                     │
        │ BothChanged → convergent conflict-copy:                    │
        │   winner = max(blake3)  (deterministic, clock-independent) │
        │   loser  → <path>.conflict-<host>-<hash>  on BOTH sides    │
        │ → both replicas hold an identical file set                 │
        │ finally: archive.save()  (epoch += 1)                      │
        └────────────────────────────────────────────────────────────┘
```

**Safety properties (all decided on blake3, never mtime):**

- **Deletes need positive evidence.** A delete is emitted only when the surviving
  side's content still equals the archive base. A modified survivor against a
  deleted peer is a **conflict**, not a delete.
- **Lost archive ⇒ safe no-base mode.** If the archive is missing, unparsable, or
  keyed to a different root pair, `Archive::load` returns `None`; reconcile then
  treats every base as absent, so **no deletes are ever produced** and one-sided
  absences become creates.
- **Convergent conflict-copy.** On `BothChanged`, the winner is the max-blake3
  version (both ends compute the same, independent of any clock), the loser is
  preserved as `<path>.conflict-<host>-<hash>` on **both** sides, and the pair
  still converges to an identical file set — no data is ever discarded.
- **Commit-then-record.** The archive is written (atomic tmp+fsync+rename, prior
  copy kept as `.bak`) only *after* the data it describes is on disk. There is no
  write-ahead log.

```bash
copia bisync ~/notes /mnt/usb/notes        # two-way sync
copia bisync ~/notes /mnt/usb/notes -n -v  # preview the reconcile plan + conflicts
```

---

## 5. Data flow: L3 hub CAS push

The hub is a **star topology**: N clients push to 1 authoritative hub. The hub is
the total order; concurrency safety is **compare-and-swap on blake3** — no
client-held locks, leases, fencing tokens, or vector clocks.

```
client: copia hub-sync ./local  intel:/srv/hub
                          │  spawns:  ssh -T intel copia serve /srv/hub
                          ▼
  ┌── hub.rs (client) ─────────┐         ┌── serve.rs (hub daemon) ───────────┐
  │ write MAGIC "COPIA1"       │────────▶│ read_magic  (prologue guard)       │
  │ Hello{version}             │◀───────▶│ Hello{version}                     │
  │ List                       │────────▶│ discover_local_fingerprints        │
  │                            │◀────────│ Fingerprints{path→blake3+ftype}    │
  │                            │         │                                    │
  │ for each local file whose  │         │                                    │
  │ blake3 ≠ hub's:            │         │                                    │
  │  Put{path, expected=hub's  │────────▶│ stream len bytes → .copia-tmp,     │
  │      last-seen hash, len,  │  +raw   │ hash while writing                 │
  │      hash}  then raw bytes │  bytes  │ verify streamed hash == claimed    │
  │                            │         │ ┌ with_commit_lock (fs2) ────────┐ │
  │                            │         │ │ cas_decide(current, expected): │ │
  │                            │         │ │  match → rename tmp → path     │ │
  │                            │         │ │  stale → rename tmp →          │ │
  │                            │         │ │     path.conflict-<hash>       │ │
  │                            │         │ └────────────────────────────────┘ │
  │ PutResult{committed,       │◀────────│                                    │
  │           current}         │         │                                    │
  └────────────────────────────┘         └─────────────────────────────────────┘
```

**Wire protocol (`wire.rs`):**

- `MAGIC = b"COPIA1"` prologue — a non-copia peer / injected shell banner is
  rejected before any framing.
- Each control message is a **bounded** big-endian `u32` length prefix + a CBOR
  body (`MAX_FRAME = 1 MiB`); an oversized prefix (e.g. `0xFFFFFFFF`) is refused
  **before** allocating. Bulk file content streams as raw bytes *after* the frame
  — never a giant CBOR blob.
- `cas_decide(current, expected)` is the sole gate: commit **iff** the hub's
  current hash equals what the client last observed. A stale `expected` is a
  `Conflict` → the loser is kept as a conflict-copy, never a lost update.

**Hub-side integrity (`serve.rs`):**

- `safe_join` rejects absolute paths and `..`/root escapes (path-traversal guard).
- Streamed content is re-hashed and must match the client's claimed hash before it
  can commit.
- The CAS + rename runs under a brief, local, exclusive `commit.lock` (`fs2`), so
  concurrent SSH-spawned `serve` processes commit **linearizably**.

```bash
# Hub side is spawned automatically over SSH; you only run the client:
copia hub-sync ./workspace intel:/srv/hub     # push to a remote hub
copia hub-sync ./workspace /srv/local-hub      # push to a local hub (spawns serve)

# A CAS conflict makes the run exit non-zero; re-run to reconcile:
#   CAS conflict (hub changed under us): <path> — hub kept a conflict-copy
```

---

## 6. Provable contracts — the differentiator

The pure decision logic is not just tested; the safety invariants are **machine
proved**. copia's three contracts (`contracts/incremental-sync-v1.yaml`,
`bidirectional-sync-v1.yaml`, `hub-protocol-v1.yaml`) reach **L5** — the top of
the provable-contracts (`pv`) ladder, and the only L5 contracts in the fleet.

The ladder (each rung is strictly stronger; distinguish **PROVED** from merely
**TESTED**):

| Rung | Meaning | Where |
|---|---|---|
| **L1** | Equations declared | `equations:` in each `contracts/*.yaml` |
| **L2** | Falsification `#[test]`s | `mod tests` in `plan.rs`, `reconcile.rs`, `wire.rs`, … |
| **L3** | Kani harnesses **verified** (exhaustive) | `#[cfg(kani)] mod kani_proofs` — `cargo kani` |
| **L4** | Lean 4 proofs | `lean/*.lean` — 11 theorems, **0 `sorry`**, checked by `lean` |
| **L5** | Every binding verified against source | `contracts/binding.yaml` — `pv proof-status --binding` |

`binding.yaml` maps each contract equation to the concrete function that
implements it (`needs_transfer`, `reconcile_path`, `cas_decide`, `safe_join`,
`handle_put`, …). `pv proof-status --binding` re-checks that each named function
actually exists — rename one and the contract falsifiably drops below L5.

**Machine-proved invariants include:**

- **`NoBaseNeverDeletes`** (`reconcile.rs` `no_base_never_deletes`, Kani; mirrored
  in `lean/BidirectionalReconcile.lean`): with no trustworthy base, reconcile
  **never** emits a delete — a lost/corrupt archive can never turn a create into
  destructive data loss.
- **`delete_requires_positive_evidence`** (Kani): a delete is emitted only when the
  survivor still equals the base.
- **`StaleCasNeverCommits`** (`wire.rs` `stale_cas_never_commits`, Kani; mirrored
  in `lean/HubCas.lean`): a stale CAS can **never** silently commit a concurrent
  write.
- **quick-check correctness** (`plan.rs` `needs_transfer_iff_new_or_differing`,
  Kani; `lean/IncrementalSync.lean`): a file transfers iff it is new or differs in
  size/mtime, and an identical file is never re-transferred.

Enforce the whole ladder with a single target:

```bash
make contracts
#   pv validate contracts/*-v1.yaml     (L1/L2)
#   lean lean/*.lean                    (L4: 11 theorems, 0 sorry)
#   pv proof-status --binding           (L5: bindings verified against source)
```

Run the exhaustive model checks directly with `cargo kani`.

---

## 7. Design provenance

The distributed design (L2 reconciler + L3 hub) was validated by a **6-system
design quorum** — Unison, Syncthing, git, Dynamo/Cassandra, ZooKeeper/etcd+WAL,
and rsync/gRPC/SSH — recorded in
[`docs/specifications/distributed-sync.md`](specifications/distributed-sync.md).
The contracts prove the implementation is *correct*; the quorum review argues the
*design* is world-class. The honest framing from that review is preserved in the
code: the hub is a single-master, CP, W=1 design (the hub is a SPOF by
construction) — **not** a Dynamo-style AP quorum — chosen because the single hub
gives a total order and the content-hash CAS is immune to pause/partition and ABA
without any client-side coordination.
