# copia — CLI Usage Cookbook

`copia` is a pure-Rust rsync replacement **and** a distributed file-sync system.
It has three sync layers you reach through six subcommands, plus the three rsync
delta primitives as standalone tools. This page is the copy-pasteable reference:
every subcommand, its exact flags, runnable examples, and the exit-code / conflict
semantics — all verified against `src/bin/copia/main.rs` and the module code.

- **L1 — one-way incremental sync** (`sync`): rsync-style quick check, atomic
  delivery, mirror deletes, excludes. Local, push (over SSH), and pull.
- **L2 — bidirectional sync** (`bisync`): two-way, blake3 3-way reconcile, safe
  deletes, convergent conflict-copy that never loses data.
- **L3 — the hub** (`serve` + `hub-sync`): N clients push to 1 hub with
  CAS-on-blake3 concurrency safety (star topology).
- **Primitives** (`signature` / `delta` / `patch`): the rsync rolling-checksum +
  blake3 delta engine, usable on their own.

> **The differentiator:** the L1/L2/L3 designs are pinned by machine-checked
> *provable contracts* (`contracts/*.yaml`). Three of them —
> `incremental-sync-v1`, `bidirectional-sync-v1`, `hub-protocol-v1` — reach **L5**,
> the top of the `pv` proof ladder, and are the only L5 contracts in the fleet.
> See [Provable safety](#provable-safety-what-is-proved-vs-tested).

---

## Install & global options

```bash
cargo install copia            # crates.io (current: 0.1.5)
# or, from a checkout:
cargo build --release --features cli   # binary at target/release/copia
copia --help
copia --version
```

Every subcommand accepts one global flag:

| Global flag | Effect |
|-------------|--------|
| `--trace-output <FILE>` | Write renacer-compatible NDJSON trace events to `<FILE>` for the run. |

Logging verbosity is controlled by `RUST_LOG` (the env filter defaults to
`copia=info`), e.g. `RUST_LOG=copia=debug copia sync ...`.

### Path syntax

A path argument is **local** unless it looks like `host:path`, in which case it is
a **remote** location reached over SSH. The parser treats a string as `host:path`
only when the text before the first `:` is longer than one character and contains
no `/` or `\` (so a bare `C:` Windows drive and any `/abs/path` stay local).

```
/data/models          # local
intel:/srv/hub        # remote: host "intel", path "/srv/hub"
intel:~/corpus        # remote with ~ expanded by the remote shell
```

SSH transport shells out to your system `ssh` — configure hosts in
`~/.ssh/config` as usual (copia does no key management of its own).

---

## `copia sync` — one-way incremental (L1)

```
copia sync <SOURCE> <DEST> [flags]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-r`, `--recursive` | off | Sync a whole directory tree (mirror). Without it, `sync` runs the **single-file** rsync delta engine. |
| `-j`, `--jobs <N>` | `4` | Parallel transfer workers (recursive only). |
| `-b`, `--block-size <N>` | `2048` | Delta block size, bytes. Power of two, 512–65536 (single-file / primitives only). |
| `-n`, `--dry-run` | off | Show the plan; transfer and delete nothing. |
| `--delete` | off | **Mirror**: remove `DEST` files that no longer exist in `SOURCE`. |
| `--exclude <GLOB>` | — | Skip matching paths. Repeatable. |
| `-v`, `--verbose` | off | Extra output (prints the `src -> dst` line at the end). |

### Two modes

**Non-recursive (single file)** runs the rsync signature→delta→patch pipeline
(rolling checksum + blake3) to update one destination file from one source file:

```bash
copia sync ./model.bin ./backup/model.bin           # local single-file delta
copia sync ./model.bin intel:/srv/model.bin         # push a single file
copia sync -b 8192 ./huge.iso ./mirror/huge.iso     # larger delta block size
```

**Recursive (`-r`)** is an incremental *directory mirror* across three
directions, chosen automatically from the endpoints:

| SOURCE | DEST | Direction |
|--------|------|-----------|
| local | local | local → local copy |
| local | `host:path` | **push** (local → remote over SSH) |
| `host:path` | local | **pull** (remote → local over SSH) |
| `host:path` | `host:path` | **not supported** (errors out) |

```bash
# local mirror
copia sync -r ./src-tree ./dst-tree

# push a tree to a hub box
copia sync -r ./corpus intel:/srv/corpus

# pull a tree down
copia sync -r intel:/srv/corpus ./corpus
```

### The quick check (incremental skip)

A source file is transferred **only** when the destination lacks it, or its
**size OR mtime differ** (compared at 1-second granularity, matching rsync's
default). Identical files are skipped — never re-sent. mtime is preserved on
delivered files, so a second run of an unchanged tree transfers nothing:

```bash
$ copia sync -r ./corpus intel:/srv/corpus
Plan: 3 to transfer, 41 unchanged (skipped), 0 to delete
...
Complete: 3 sent, 41 skipped, 0 deleted, 0 failed
$ copia sync -r ./corpus intel:/srv/corpus
Already up to date (44 files).
```

Delivery is **atomic**: each file streams to a `<name>.copia-tmp` sibling and is
`rename`d into place, so a reader never sees a half-written file and a crash
never leaves a torn destination.

### `--dry-run`

Prints exactly what would happen, then exits without touching anything:

```bash
$ copia sync -r --delete ./corpus intel:/srv/corpus -n
Plan: 2 to transfer, 40 unchanged (skipped), 1 to delete
send   new/chapter-04.md
send   updated/index.json
delete stale/removed.txt
(dry run) nothing was modified
```

### `--delete` (mirror) and `--exclude`

`--delete` removes destination files absent from the (filtered) source — opt-in,
never the default. `--exclude` uses gitignore-style globs:

- A pattern with **no `/`** matches any single path component anywhere in the
  tree (`target`, `*.tmp`, or the trailing-slash form `target/`).
- A pattern **containing `/`** matches the whole relative path as a glob
  (`a/*/c.log`).

Excluded paths are dropped entirely: never transferred, and **never counted
toward the delete set** — so an excluded file already on the destination is left
untouched rather than mirror-deleted.

```bash
# mirror a build tree but never ship or delete build artifacts / temp files
copia sync -r --delete \
  --exclude target --exclude '*.tmp' --exclude node_modules/ \
  ./project intel:/srv/project
```

`--jobs` bounds concurrency (default 4); a first sync to a missing destination
treats every source file as new (empty destination map).

---

## `copia bisync` — bidirectional two-way sync (L2)

```
copia bisync <A> <B> [-n|--dry-run] [-v|--verbose]
```

Two-way sync of **two local directories**. copia fingerprints both trees with
blake3, loads a persisted per-pair *archive* (the last-synced common state), and
runs a 3-way reconcile. blake3 is the **sole** oracle for what changed, converged,
conflicts, or should propagate/delete — **mtime never decides**.

| Flag | Meaning |
|------|---------|
| `-n`, `--dry-run` | Print the reconcile plan; modify neither side. |
| `-v`, `--verbose` | List each preserved conflict path at the end. |

The archive lives at `~/.copia/archive/<root_pair_hash>.json` (keyed by a blake3
of both canonical roots + a format version + a monotonic epoch), written
atomically with the prior copy kept as `.bak`.

### Seed, then propagate

The **first** run for a pair has no trusted archive, so copia runs in **safe
no-base mode**: no deletes are possible and any one-sided file is treated as a
*create*, never a delete. This seeds the archive:

```bash
$ copia bisync ~/notes-laptop ~/notes-desktop
No trusted archive for this pair — SAFE no-base mode (no deletes; divergence kept as conflicts).
Bidirectional plan: 12 action(s), 0 conflict(s) [safe no-base]
Bidirectional sync complete: 12 applied, 0 conflict(s) preserved.
```

Subsequent runs use the archive as the common base and can propagate edits and
**safe deletes** in both directions:

```bash
$ copia bisync ~/notes-laptop ~/notes-desktop
Bidirectional plan: 3 action(s), 0 conflict(s)
Bidirectional sync complete: 3 applied, 0 conflict(s) preserved.
```

### Safe deletes

A delete is propagated **only** with positive archive evidence — i.e. the
surviving side's content still equals the recorded base. If the surviving side
was *also* modified, it is **not** deleted; it becomes a delete-vs-modify
conflict and the modification is kept. If the archive is lost or mismatched,
copia falls back to safe no-base mode and disables deletes entirely, so a
missing archive can **never** turn a create into a delete (this is machine-proved
— `NoBaseNeverDeletes`).

### Convergent conflict-copy

When both sides changed the same path to **different** content, copia picks **no
semantic winner** and loses no data. Both versions survive on **both** sides:

- The real path deterministically takes the **max-blake3** version (clock-
  independent, identical on both replicas — never mtime).
- The losing version lands at `<path>.conflict-<host>-<short_blake3>` on both
  sides.

Both replicas end up with an identical file set and the pair converges.

```bash
$ copia bisync -v ~/notes-laptop ~/notes-desktop
Bidirectional plan: 1 action(s), 1 conflict(s)
Bidirectional sync complete: 1 applied, 1 conflict(s) preserved.
  conflict: journal/2026-07-04.md
# both sides now contain:
#   journal/2026-07-04.md                              (deterministic winner)
#   journal/2026-07-04.md.conflict-hostA-1f3a9c0b2e77  (the other version)
```

> **Exit code:** `bisync` exits **non-zero when conflicts were preserved** — a
> signal to review, **not** a data-loss error. Both versions are on disk on both
> sides. Resolve by deleting the `.conflict-*` copy you don't want, then re-run.

Dry-run prints the raw reconcile actions:

```bash
$ copia bisync -n ~/notes-laptop ~/notes-desktop
PropagateAtoB          new/idea.md
DeleteB                stale/old.md
Conflict(BothChanged)  journal/2026-07-04.md
(dry run) nothing was modified
```

---

## The hub — `copia serve` + `copia hub-sync` (L3)

The hub is a **star topology**: one long-lived `copia serve <root>` process is the
source of truth, and any number of clients push their local trees into it with
CAS-on-blake3 so concurrent writers can never silently clobber each other.

### `copia serve` — the hub daemon

```
copia serve <ROOT>
```

`serve` reads a framed CBOR request/response protocol on **stdin** and writes
responses on **stdout**. You almost never run it by hand — clients spawn it for
you, remotely as `ssh <host> copia serve <root>` or locally as a child process.
Its guarantees:

- **MAGIC prologue guard** — aborts unless the channel opens with the exact
  `COPIA1` bytes, so an SSH banner / motd / rc-file output can't desync the
  stream.
- **Bounded frames** — a length prefix over 1 MiB is rejected *before* any
  allocation (a `0xFFFFFFFF` prefix can't trigger a 4 GiB alloc). Bulk file
  content streams as raw bytes after the frame, not inside CBOR.
- **Path-traversal guard** — a client-supplied path that is absolute or contains
  `..` is refused; a client can never read or write outside `ROOT`.
- **Content integrity** — a `Put` is rejected unless `blake3(streamed bytes)`
  equals the hash the client claimed.
- **CAS commit under a brief tree commit-lock** (`fs2` file lock on
  `<root>/.copia/commit.lock`), so commits from concurrent SSH-spawned server
  processes are linearizable.

### `copia hub-sync` — the client

```
copia hub-sync <LOCAL> <TARGET>
```

`TARGET` is either `host:root` (spawns `ssh host copia serve root`) or a **local
hub path** (spawns a local `copia serve`). One persistent connection is opened;
the client lists the hub, then for each local file:

- if the hub already has the identical blake3 → **skipped**;
- otherwise **CAS-push**: the write carries the hash the client last saw for that
  path. It commits **only if** the hub's current hash still equals that
  `expected`. A stale CAS (someone changed the file since the client read it)
  does **not** overwrite — the hub lands the incoming bytes as a
  `<path>.conflict-<hash>` copy, never a lost update.

```bash
# push a local tree to a remote hub over SSH
copia hub-sync ./corpus intel:/srv/hub

# local hub (no host: prefix) — spawns a local serve child
copia hub-sync ./corpus /srv/local-hub
```

A clean run:

```bash
$ copia hub-sync ./corpus intel:/srv/hub
Hub push complete: 7 sent, 12 unchanged, 0 conflict(s).
```

### What a CAS conflict looks like

Two clients push the same path concurrently. The first commits; the second's
`expected` is now stale, so its write becomes a conflict-copy on the hub and the
client reports it and **exits non-zero**:

```bash
$ copia hub-sync ./corpus intel:/srv/hub
  CAS conflict (hub changed under us): index/manifest.json — hub kept a conflict-copy
Hub push complete: 6 sent, 12 unchanged, 1 conflict(s).
# exit status 1 — "1 CAS conflict(s) — re-run to reconcile"
```

The hub now holds `index/manifest.json` (the committed value) **and**
`index/manifest.json.conflict-<hash>` (the loser). Re-running the client
re-reads the hub's current hash and CAS-pushes cleanly.

> **No client-held locks, leases, fencing tokens, or vector clocks.** The single
> hub is the total order and the content-hash CAS is immune to pause/partition
> and ABA — the design was quorum-validated against etcd/Dynamo/Unison/Syncthing.

---

## Primitives — `signature` / `delta` / `patch`

The rsync three-step delta engine, usable standalone. Round-trip:
`signature(old) → delta(new, sig) → patch(old, delta) == new`.

### `copia signature`

```
copia signature <FILE> [-o|--output <FILE>] [-b|--block-size <N>]
```

Computes the block signature (rolling checksum + blake3 per block) of `FILE`.
Default output is `<FILE>.sig`; default block size `2048` (power of two, 512–65536).

```bash
$ copia signature old.bin
Generated signature: old.bin.sig (512 blocks, 1048576 bytes)
```

### `copia delta`

```
copia delta <SOURCE> <SIGNATURE> [-o|--output <FILE>]
```

Computes the delta of the new `SOURCE` against the old file's `SIGNATURE`.
Default output is `<SOURCE>.delta`. The block size is read from the signature.

```bash
$ copia delta new.bin old.bin.sig
Generated delta: new.bin.delta (37 ops, 96.4% matched)
```

### `copia patch`

```
copia patch <BASIS> <DELTA> [-o|--output <FILE>]
```

Reconstructs the new file by applying `DELTA` to the old `BASIS`. Default output
is `<BASIS>.patched`.

```bash
$ copia patch old.bin new.bin.delta -o rebuilt.bin
Applied patch: rebuilt.bin (1050000 bytes)
```

Full round-trip:

```bash
copia signature old.bin -o old.sig
copia delta     new.bin old.sig -o patch.delta
copia patch     old.bin patch.delta -o rebuilt.bin   # rebuilt.bin == new.bin
```

---

## Fleet patterns

### A shared RAG-index hub (N clients → 1 hub)

Keep a single authoritative index tree on a hub box; every producer pushes into
it with CAS so concurrent producers never clobber one another.

```bash
# On each producer machine (cron / timer):
copia hub-sync /local/rag-index intel:/srv/rag-hub
```

The first client to push a changed shard commits; a racing client that read the
old shard gets a conflict-copy on the hub and exits non-zero, so your scheduler
can retry — the failure is loud, and no shard is silently lost. Consumers pull
the assembled tree with a one-way sync:

```bash
copia sync -r intel:/srv/rag-hub /local/rag-index      # pull latest
```

### Two-box bidirectional mirror (laptop ↔ desktop, both writable)

When both replicas are edited, use `bisync` and treat conflict-copies as review
items. Seed once (safe no-base), then run on a timer:

```bash
# seed (first run: safe no-base mode, no deletes)
copia bisync ~/work ~/work-mirror

# steady state (archive-backed: propagates edits + safe deletes both ways)
copia bisync -v ~/work ~/work-mirror
# non-zero exit ⇒ conflicts preserved as <path>.conflict-<host>-<hash>; review & re-run
```

Because blake3 (never mtime) is the oracle, a touch that doesn't change bytes is
a no-op, and a genuine two-sided edit always keeps **both** versions on **both**
boxes.

### One-way publish / backup mirror

A read-only downstream (mirror, backup, CDN origin) is a plain `sync -r`. Add
`--delete` to make it a strict mirror; exclude volatile dirs so they are never
shipped or deleted:

```bash
copia sync -r --delete --exclude target --exclude '.git' \
  ./site intel:/srv/www
```

---

## Exit codes & report semantics

`copia` returns `0` on success and `1` on any error (`ExitCode::SUCCESS` /
`FAILURE`). What counts as an error is command-specific:

| Command | Exit `0` | Exit `1` |
|---------|----------|----------|
| `sync` | all planned files delivered (`0 failed`) | one or more files failed to transfer; unsupported direction (remote→remote); bad block size (not a power of two / outside 512–65536) |
| `bisync` | reconcile applied, **no** conflicts preserved | **conflicts were preserved** (both versions saved — a review signal, *not* data loss); I/O error scanning a root |
| `serve` | client sent `Bye` / clean EOF | bad protocol prologue; I/O error |
| `hub-sync` | pushed with **no** CAS conflicts | one or more CAS conflicts (hub kept conflict-copies — re-run to reconcile); connection/handshake failure |
| `signature`/`delta`/`patch` | file written | I/O error; bad block size |

Report lines you'll see:

- **`sync`** — a plan line to stderr (`Plan: N to transfer, M unchanged
  (skipped), K to delete`) and a completion line
  (`Complete: sent, skipped, deleted, failed` + throughput). `Already up to date
  (N files).` means the quick check skipped everything.
- **`bisync`** — `Bidirectional plan: N action(s), K conflict(s)` and
  `Bidirectional sync complete: N applied, K conflict(s) preserved.` The
  `[safe no-base]` suffix flags a missing/mismatched archive (deletes disabled).
- **`hub-sync`** — a per-conflict `CAS conflict (hub changed under us): ...` line
  plus `Hub push complete: sent, unchanged, conflict(s).`

**Conflicts are never data loss.** A `bisync` `.conflict-<host>-<hash>` copy and a
hub `.conflict-<hash>` copy both preserve the losing bytes on disk; the non-zero
exit is your prompt to review and re-run.

---

## Provable safety — what is *proved* vs *tested*

copia's differentiator is that these behaviors aren't just documented, they're
pinned by machine-checked contracts in `contracts/*.yaml`. The `pv` proof ladder:

1. **L1** — equations declared in the contract.
2. **L2** — falsification `#[test]`s that encode each prediction.
3. **L3** — Kani harnesses model-checked exhaustively (`cargo kani`).
4. **L4** — Lean 4 proofs — **11 theorems** across `lean/*.lean`, **0 `sorry`**,
   checked by `lean`.
5. **L5** — every contract equation *bound* to the function that implements it
   (`contracts/binding.yaml`), verified against source by
   `pv proof-status --binding` (rename the function and the contract drops below
   L5). `incremental-sync-v1`, `bidirectional-sync-v1`, and `hub-protocol-v1` all
   reach **L5**.

Invariants that are **machine-proved** (not merely tested), with their Lean
theorems:

| Invariant | Meaning | Lean theorem |
|-----------|---------|--------------|
| Quick-check correctness | transfer iff new or size/mtime differ; identical files never re-sent | `QuickCheckCorrect`, `SkipGuarantee` |
| No-base never deletes | a lost/mismatched archive can never turn a create into a delete | `NoBaseNeverDeletes` |
| Delete needs evidence | a delete requires the survivor still equal the base; else conflict | `DeleteNeedsEvidence` |
| Conflict is never a silent pick | divergent edits become a conflict, never an arbitrary winner | `ConflictNotSilentPick` |
| Stale CAS never commits | a concurrent hub write can never silently lose a prior update | `StaleCasNeverCommits` |
| Bounded frame / no traversal | oversized frames and path escapes are rejected | `BoundedFrame`, `NoTraversal` |

Run the whole ladder — schema validation, the Lean proofs, the binding/proof-level
report (expect L5), and the falsification tests:

```bash
make contracts
```

Content integrity (blake3 re-hash on every `Put`) and the MAGIC prologue guard
are enforced in code and covered by the `FALSIFY-HUB` tests, not modelled as Lean
obligations.
