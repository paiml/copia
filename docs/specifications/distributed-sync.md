# copia Distributed Sync — L2 (bidirectional) + L3 (hub daemon)

Vetted 2026-07-04 by a 6-system design quorum (Unison, Syncthing, git, Dynamo/Cassandra,
ZooKeeper/etcd+WAL, rsync/gRPC/SSH). Verdict: **proceed-with-required-changes**. Built on the
shipped L1 incremental one-way engine (quick-check skip, atomic `.copia-tmp`+rename delivery,
mtime preservation).

This document is both the design rationale (the quorum's non-negotiable safety rules, which are
preserved verbatim below) **and** an honest map of what is shipped in `0.1.x` versus the
documented follow-ons. Each section carries an *Implemented in* pointer to the module that
realizes it, and every claim here has been checked against that source.

## Status legend

- ✅ **SHIPPED** — implemented, tested, and (where noted) machine-proved in `0.1.x`.
- 🔭 **FOLLOW-ON** — designed and reserved in the protocol/spec, not yet implemented. These are
  called out inline so nothing here overstates the current binary.

The three L2/L3 contracts (`bidirectional-sync-v1`, `hub-protocol-v1`) plus the L1
`incremental-sync-v1` are the only **L5** provable contracts in the fleet — see
[Provable contracts](#provable-contracts) at the end.

---

## Non-negotiable safety rules (unanimous across the lineage)

These are the design invariants the quorum required. All five govern the shipped reconciler and
hub; where a rule's *fully-general* form (e.g. a per-entry racy-clean timestamp guard) is a
follow-on, that is flagged in the relevant section rather than here.

1. **blake3 is the sole oracle.** ✅ Every `equal / changed / conflict / propagate / delete /
   winner` decision is on the content digest. size+mtime+ftype is ONLY a fast-path deciding
   whether to re-hash — it may never conclude "unchanged", "conflict", or "winner". *(Shipped: the
   reconcile case table and the hub CAS take only blake3 + entry-type; the size+mtime quick-check
   lives in the L1 planner and merely gates re-hashing.)*
2. **No mtime winner.** ✅ Divergence keeps BOTH sides (conflict-copy); any canonical pick uses a
   clock-independent total order — here `max(blake3)` — computed identically on both ends.
3. **Deletes need positive archive evidence.** ✅ A one-sided absence propagates as a delete ONLY
   if the path was in the archive AND the surviving side's current blake3 == the archived blake3.
   Archive absent / corrupt / pair-or-format-mismatch, or a scan error → CREATE-or-CONFLICT, never
   delete.
4. **Commit-then-record, NO WAL.** ✅ Data lands via tmp → `sync_all` → atomic rename BEFORE the
   archive entry is recorded; the archive is never ahead of the data. Whole-archive write is
   tmp+fsync+rename with the previous copy retained as `.bak`. Recovery = re-scan + idempotent
   re-apply (target-hash-keyed: if dest already == target blake3, skip).
5. **CAS at apply.** ✅ On the hub, the compare (read current hash) and the rename happen as one
   step under a per-tree commit-lock, so a value that changed since the client read it can never be
   clobbered — it becomes a conflict-copy.

---

## L2 — two-party bidirectional reconciler (Unison-shaped)

**CLI:** `copia bisync A B [-n/--dry-run] [-v/--verbose]` — both sides are **local** directories.
Exits non-zero when any conflict was preserved (so scripts can detect divergence).

*Implemented in: `reconcile.rs` (pure case table, Kani+Lean proved), `archive.rs` (per-pair base),
`bidir.rs` (scan → reconcile → atomic apply → record).*

### Fingerprint & archive

✅ A **Fingerprint** is `{ blake3: [u8;32], ftype: File | Symlink }` — the digest is authoritative;
`ftype` guards a file↔symlink flip from masquerading as a same-kind content change.

✅ The **archive** is the last-synced common state for one `(rootA, rootB)` pair, stored as pretty
JSON at `~/.copia/archive/<root_pair_hash>.json`. Header:
`{ format_version, root_pair_hash = blake3(canon(A) ⧺ NUL ⧺ canon(B)), epoch: u64, host_id,
entries: path → Fingerprint }`. `root_pair_hash` is order-sensitive (A-then-B). `Archive::load`
returns the archive **only** if it parses AND `format_version` and `root_pair_hash` both match;
any mismatch → `None` → safe no-base mode. Saved atomically (tmp → `sync_all` → retain old as
`.bak` → rename → fsync parent).

> 🔭 **Follow-on — richer entries.** The quorum design reserved per-entry `{ size, mtime, inode,
> symlink_target }` for a racy-clean fast path and symlink-target tracking. Shipped entries carry
> only `{ blake3, ftype }`; `bidir` conservatively re-hashes every scanned file, so correctness
> holds without the extra fields.

### 3-way reconcile

✅ `reconcile_path(A_now, B_now, base)` on blake3, the full case table:

| A vs B vs base                                   | Action                                        |
|--------------------------------------------------|-----------------------------------------------|
| both == base                                     | `Noop`                                         |
| exactly one != base                              | `PropagateAtoB` / `PropagateBtoA`              |
| both != base, A == B                             | `ConvergeIdentical` (record only, no file op)  |
| both != base, A != B                             | `Conflict(BothChanged)`                        |
| one absent, base present, survivor == base       | `DeleteA` / `DeleteB`                           |
| one absent, base present, survivor != base       | `Conflict(DeleteVsModify)` (keep the survivor)  |
| one absent, base **absent**                       | `PropagateA/BtoB/A` = a **CREATE**, never delete |

`reconcile(A, B, base, trust_base)` walks the sorted union of both trees. **Safe mode**
(`trust_base = false`) forces every base lookup to `None`, so NO deletes are ever produced and all
divergence degrades to create/conflict — this is exactly what a lost/mismatched archive triggers.

> Two of these rows are machine-proved: `NoBaseNeverDeletes` (base absent ⇒ never `DeleteA/DeleteB`)
> and `DeleteNeedsEvidence` (a `DeleteA` implies survivor == base). See
> [Provable contracts](#provable-contracts).

### Apply & convergent conflict-copy

*Implemented in: `bidir.rs::apply`.*

✅ Delivery is atomic (temp sibling `.copia-tmp` + rename, parent dirs created as needed). On
`Conflict(BothChanged)`:

1. Deterministic winner = **max(blake3)** — both ends compute the same, no clock involved.
2. The loser is preserved as `<path>.conflict-<host_id>-<short_blake3>` (first 6 bytes, 12 hex
   chars) on **BOTH** sides FIRST.
3. The winner's content is written to the real path on both sides.

Both replicas end with the identical set `{ path = winner, path.conflict-… = loser }`, so the pair
**converges without ever losing data**. `Conflict(DeleteVsModify)` keeps the modification (restores
the surviving side onto the deleted one). `host_id` comes from `$HOSTNAME` / `hostname` / `"host"`.

> 🔭 **Follow-on — `maxConflicts` cap.** The quorum recommended a bound (default 8; 0 = keep newest
> only) on how many conflict-copies accumulate per path. Not yet implemented — every conflict-copy
> is currently retained.

> 🔭 **Follow-on — networked L2.** `bisync` is local↔local only. The quorum's "exchange +
> cross-verify archive headers over the wire" is the SSH-networked L2 variant; today the archive
> match is a single local load check (`Archive::load`). Push/pull *one-way* over SSH already ships
> as `copia sync -r` (L1).

### Runnable examples

```bash
# Two-way reconcile of two local trees (creates ~/.copia/archive/<pair>.json on first run)
copia bisync ~/notes /mnt/backup/notes

# See the plan without touching either side
copia bisync -n ~/notes /mnt/backup/notes

# Verbose: list every preserved conflict path (exit code != 0 if any)
copia bisync -v ~/notes /mnt/backup/notes
```

First run prints `SAFE no-base mode (no deletes; divergence kept as conflicts)` because there is no
trusted archive yet — deletes are disabled until a common base exists.

---

## L3 — N clients → 1 hub (star; single serializer, no consensus)

**CLI:**
- `copia serve ROOT` — the hub side. Reads framed CBOR requests on **stdin**, writes responses on
  **stdout**. Clients spawn it as `ssh <host> copia serve <root>`.
- `copia hub-sync LOCAL TARGET` — the client. `TARGET` is `host:root` (SSH) or a bare local path
  (spawns a local `serve`). Exits non-zero if any CAS conflict occurred.

*Implemented in: `wire.rs` (framing + CAS gate, Kani+Lean proved), `serve.rs` (hub daemon + commit
lock), `hub.rs` (persistent-connection client).*

### Wire protocol

✅ One persistent pipe per client. Every control message is a **length-prefixed CBOR frame**
(`ciborium`): a big-endian `u32` length + the CBOR body. Bulk file content streams as **raw bytes
after** the `Put`/`Content` frame — never inside a giant CBOR blob.

- **`MAGIC = b"COPIA1"`, `VERSION = 1`.** The hub reads and verifies the 6-byte magic prologue
  before parsing any frame; a non-copia peer (ssh banner, rc-file output) is rejected immediately.
- **`MAX_FRAME = 1 MiB`.** A length prefix exceeding this is rejected BEFORE allocating, so a
  `0xFFFFFFFF` prefix can never trigger a multi-GiB allocation. (Streamed content is not bounded by
  this.) Clean EOF at a frame boundary returns `None`, not an error.
- **Requests:** `Hello{version}`, `List`, `Get{path}`, `Put{path, expected: Option<Hash>, len,
  hash}`, `Delete{path, expected: Option<Hash>}`, `Bye`.
- **Responses:** `Hello{version}`, `Fingerprints(map)`, `Content{len, hash}`, `PutResult{committed,
  current}`, `DeleteResult{deleted, current}`, `Error(String)`.

> 🔭 **Follow-on — async duplex + credit-based flow control + incremental file list.** The quorum
> design calls for a full-duplex, credit-windowed stream and a streamed file list. Shipped `wire.rs`
> is a **synchronous request/response** loop over buffered stdin/stdout, and `List` returns the whole
> fingerprint map in one frame. This is correct and bounded; pipelining/backpressure is future work.

### Concurrency — CAS-on-blake3

✅ NO client-held locks, leases, fencing tokens, or vector clocks. The client carries, per path, the
blake3 it last saw for that file (`expected`) as its causal baseline. To write it sends
`Put{expected, len, hash}` + the content bytes. The hub:

1. streams the content to a temp file while hashing it, and **rejects it if
   `blake3(streamed) != hash`** (content-integrity guard);
2. under a brief exclusive **commit-lock** (`fs2` flock on `<root>/.copia/commit.lock`), reads the
   file's CURRENT hash and runs `cas_decide(current, expected)`:
   - **`Commit`** (current == expected) → atomic rename temp into place;
   - **`Conflict`** (stale) → rename temp to `<path>.conflict-<short_hash>` — **never** overwriting
     the live value.

Because compare + rename are one step reading the CURRENT content hash, the CAS is immune to
pause/partition and ABA — the state *is* the content. `cas_decide` is a pure function; its
lost-update safety (`Commit ⇒ current == expected`) is Kani- and Lean-proved.

The commit-lock is what makes "single serializer, no consensus" true even though each client
SSH-spawns its **own** `serve` process: the OS file lock serializes commits across those processes
into one linearizable order per tree.

> Note: the "single daemon" is logical, not a long-lived socket process. There is no pidfile
> start-guard; the flock at commit time is the serialization point. 🔭 A persistent **socket daemon**
> (one resident process instead of per-connection SSH spawns) is a documented follow-on.

### Deletes

✅ `Delete{expected}` is a **conditional CAS delete**: under the commit-lock, remove the file iff
`current == expected`; a stale `expected` is refused (`DeleteResult{deleted:false}`), never removing
a file another writer just changed.

> 🔭 **Follow-on — tombstones.** The quorum design specifies versioned **tombstones** with a
> `gc_grace` window (default 30d) so that a create against a live tombstone from a stale baseline is
> rejected/conflicted. Shipped `handle_delete` does a plain CAS `remove_file` with no tombstone and
> no GC. The CAS still prevents a stale delete; resurrection-race protection awaits tombstones.

### Path safety & integrity

✅ `safe_join(root, rel)` rejects absolute paths and any `..`/root-escape component, so a client can
never read or write outside the served root. Every `Put` is content-integrity checked (re-hash ==
claimed hash) before it is eligible to commit.

### Honest framing (from the quorum)

Single-master **CP**, hub = SPOF, W=1 — this is explicitly **NOT** Dynamo-style AP, and that is the
point: the star's single total order is what makes "no vector clocks" sound. Default per-file
atomicity.

> 🔭 **Follow-on — tree-staging.** For snapshot readers that need a whole-tree atomic swap
> (staging dir → fsync → single rename), the quorum reserved an opt-in tree-staging mode. Shipped
> delivery is per-file atomic only.

### Runnable examples

```bash
# Push a local tree to a remote hub over SSH (spawns `copia serve /srv/hub` there)
copia hub-sync ~/project intel:/srv/hub

# Push to a hub on this machine (spawns a local `copia serve`)
copia hub-sync ~/project /srv/local-hub

# Run the hub directly against a pipe (normally clients spawn this for you)
copia serve /srv/hub
```

A second identical `hub-sync` transfers 0 files (blake3 quick-check skip). On a CAS conflict the
client prints `CAS conflict (hub changed under us): <path> — hub kept a conflict-copy` and exits
non-zero; re-run to reconcile against the hub's new state.

---

## Adopted defaults (quorum recommendations)

| Default                          | Status | Where                                   |
|----------------------------------|--------|-----------------------------------------|
| no-auto-winner (`max(blake3)`)   | ✅     | `bidir.rs`, `serve.rs`                   |
| per-file atomicity               | ✅     | `bidir.rs`, `serve.rs` (tmp+rename)     |
| CBOR encoding                    | ✅     | `wire.rs`                               |
| star topology (N → 1 hub)        | ✅     | `serve.rs` + commit-lock                |
| commit-then-record, no WAL       | ✅     | `archive.rs`, `bidir.rs`                |
| tree-staging (snapshot readers)  | 🔭     | opt-in follow-on                        |
| `gc_grace = 30d` tombstones      | 🔭     | follow-on                               |
| `maxConflicts = 8`               | 🔭     | follow-on                               |
| credit-based flow control        | 🔭     | follow-on                               |
| chunk-resume                     | 🔭     | follow-on (Put streams whole file)      |

The star total-order is what makes "no vector clocks" sound; the content-hash CAS is what makes
"no leases/fencing" sound.

---

## Provable contracts

The differentiator: L2/L3 correctness is not just tested, it is **proved**. copia's
`incremental-sync-v1`, `bidirectional-sync-v1`, and `hub-protocol-v1` reach **L5** — the top of the
provable-contracts (`pv`) ladder and the only L5 contracts in the fleet.

The ladder, all enforced by `make contracts`:

1. **L1** — equations declared in `contracts/*.yaml`.
2. **L2** — falsification `#[test]`s (`FALSIFY-BIDIR-00x`, `FALSIFY-HUB-00x`) that fail if the rule
   is violated.
3. **L3** — Kani harnesses, verified exhaustively by `cargo kani`:
   - `reconcile::kani_proofs::no_base_never_deletes`, `delete_requires_positive_evidence`;
   - `wire::kani_proofs::stale_cas_never_commits`.
4. **L4** — Lean 4 proofs: **11 theorems** across `lean/IncrementalSync.lean` (4),
   `lean/BidirectionalReconcile.lean` (4), `lean/HubCas.lean` (3), **0 `sorry`**, verified by
   `lean`. Includes `NoBaseNeverDeletes`, `DeleteNeedsEvidence`, `Blake3Oracle`,
   `ConflictNotSilentPick`, `StaleCasNeverCommits`, `BoundedFrame`, `NoTraversal`.
5. **L5** — every equation bound to the concrete function that implements it in
   `contracts/binding.yaml`, checked against source by `pv proof-status --binding`. A binding only
   counts if `fn <function>` actually exists — rename a function and the contract drops below L5.

Machine-proved invariants of note:

- **`NoBaseNeverDeletes`** — a lost/corrupt archive can never turn a create into a delete.
- **`StaleCasNeverCommits`** — a stale CAS can never silently lose a concurrent hub write.

```bash
make contracts   # pv validate → lean proofs → pv proof-status (expect L5) → falsification tests
```

---

## Implementation order (as shipped, then follow-ons)

**L2 (shipped):** (1) fingerprint + archive schema → (2) scan/fingerprint → (3) 3-way reconcile →
(4) safe no-base mode → (5) atomic apply + commit-then-record → (6) convergent conflict-copy +
contract.

**L3 (shipped):** (7) CBOR framing + magic + bounds → (8) `serve` daemon + commit-lock + CAS +
content integrity → (9) persistent client `hub-sync` + CAS push → (10) conditional CAS delete →
(11) contract (Kani CAS + path-safety + framing).

**Follow-ons (🔭):** async duplex + credit-based flow control; tombstoned deletes with `gc_grace`;
tree-staging for snapshot readers; chunk-resume; a resident socket daemon; richer archive entries;
`maxConflicts` cap; networked (SSH) L2 header exchange.
