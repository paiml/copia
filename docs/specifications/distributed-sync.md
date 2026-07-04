# copia Distributed Sync — L2 (bidirectional) + L3 (hub daemon)

Vetted 2026-07-04 by a 6-system design quorum (Unison, Syncthing, git, Dynamo/Cassandra,
ZooKeeper/etcd+WAL, rsync/gRPC/SSH). Verdict: **proceed-with-required-changes**. Built on the
shipped L1 (incremental one-way: quick-check skip, atomic .copia-tmp+rename, mtime preservation).

## Non-negotiable safety rules (unanimous across the lineage)

1. **blake3 is the sole oracle.** Every equal / changed / conflict / propagate / noop / winner
   decision is on the content digest. size+mtime+inode+ftype is ONLY a fast-path deciding whether
   to re-hash — it may never conclude "unchanged", "conflict", or "winner". Racy-clean guard: if
   `mtime >= archive.recorded_time` or within fs granularity → force a hash.
2. **No mtime winner.** Divergence → conflict-copy BOTH sides, no automatic winner. Any canonical
   pick uses a clock-independent total order `(blake3, host_id)` computed identically on both ends.
3. **Deletes need positive archive evidence.** A one-sided absence propagates as a delete ONLY if
   the path was in the archive AND the surviving side's current blake3 == the archived blake3.
   Archive absent/corrupt/epoch-mismatch, or a scan error / partial listing → CREATE-or-CONFLICT,
   never delete. "Confirmed absent" (complete successful scan) ≠ "scan error".
4. **Commit-then-record, NO WAL.** Per path: tmp → fsync(tmp) → atomic rename (commit) →
   fsync(parent dir) → THEN update that path's archive entry. Archive never gets ahead of data.
   Whole-archive write = tmp+rename+fsync, previous retained. Recovery = re-scan + idempotent
   target-hash-keyed re-apply ("if dest already == target blake3, skip").
5. **TOCTOU CAS at apply.** Under a per-path critical section re-stat + re-hash source and dest and
   compare to the plan snapshot; if either changed → skip+report, never clobber.

## L2 — two-party bidirectional reconciler (Unison-shaped)

- **Archive** = last-synced common state, one per `(rootA, rootB)` pair, stored on both sides.
  Header `{format_version, root_pair_hash=H(canon(rootA)+canon(rootB)), epoch:u64, host_id}`.
  Entry per path `{blake3, size, mtime, inode, ftype, symlink_target?}`.
- **3-way reconcile** (A-now, B-now, base=archive) on blake3, full case table:
  both==base→noop; one!=base→propagate; both!=base & A==B→converge; both!=base & A!=B→CONFLICT;
  absent + positive-evidence→delete; absent + surviving-changed→delete-vs-modify CONFLICT;
  base-absent + present-one→CREATE.
- **Safe mode**: exchange+cross-verify archive headers; on mismatch/missing/epoch-disagree →
  no-base mode (propagate creates, conflict-copy differing, DISABLE deletes; explicit opt-in to
  establish a fresh common archive).
- **Conflict-copy**: loser → `<path>.conflict-<origin_host>-<short_blake3>`; inert (never a
  reconcile source, never re-conflicted); capped `maxConflicts=8` (0 = keep newest only).

## L3 — N clients → 1 hub daemon (star; single serializer, no consensus)

- One persistent SSH connection; `copia --server` daemon; framed **CBOR** wire protocol
  (MAGIC+version+capability handshake, `-T` no PTY, stderr separate, bounded ~1 MB frames,
  oversize-prefix rejected pre-alloc, async duplex with **credit-based flow control**, incremental
  file list).
- **Concurrency**: NO client-held locks/leases/fencing. Client carries its L2 archive per-path
  blake3 as causal baseline; to write it sends `(Hexpected, Hnew, bytes)`; hub, inside a short
  in-process per-path mutex, does `read Hcur; if Hcur==Hexpected → atomic rename+fsync; else →
  conflict-copy`. Compare+rename is one atomic step reading CURRENT hash → immune to
  pause/partition and ABA (state IS the content).
- **Deletes** = conditional CAS `(Hexpected → tombstone)`; bounded versioned tombstones with
  `gc_grace=30d`; create-against-live-tombstone from a stale baseline → reject/conflict.
- **Single-daemon guard**: `flock()` a pidfile at the tree root; refuse start if held; epoch-stamp
  index writes. Only split-brain defense (no quorum).
- **Honest framing**: single-master CP, hub = SPOF (W=1), NOT Dynamo-AP. Default per-file
  atomicity; optional tree-staging (staging dir → fsync → single atomic swap) for snapshot readers.

## Adopted defaults (quorum recommendations; user-overridable)
no-auto-winner · per-file atomicity (tree-staging opt-in) · gc_grace=30d · maxConflicts=8 ·
CBOR encoding · star topology (N→1 hub) — the star total-order is what makes "no vector clocks" sound.

## Implementation order
L2: (1) fingerprint+archive schema → (2) change-detect+racy-clean → (3) 3-way reconcile →
(4) safe-mode → (5) apply/CAS/commit-then-record → (6) conflict-copy + contract.
L3: (7) wire skeleton → (8) async duplex+credit flow → (9) daemon+flock+per-path-mutex+CAS →
(10) conditional deletes+tombstones → (11) resume+tree-staging → (12) contract (kani CAS linearizability).
