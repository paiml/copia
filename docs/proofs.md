# Proofs: copia's provable-contracts trust story

Most file-sync tools ask you to trust their tests. copia asks you to check its
**proofs**. Every safety-critical decision in copia — *does this file transfer?
should this absence become a delete? can this concurrent write silently win?* —
is a pure function pinned by a machine-checked contract. The dangerous ones are
proved exhaustively by a model checker (Kani) **and** by a theorem prover
(Lean 4), then the proof is bound back to the exact function in the source tree.

This document is the map. It is scrupulously honest about the line between
**PROVED** (Kani + Lean, mathematically exhaustive) and **TESTED**
(falsification `#[test]`s that exercise real I/O behaviour). Read it, then run
`make contracts` and reproduce every claim yourself.

---

## The `pv` ladder (L1 → L5)

copia's contracts are graded on the provable-contracts (`pv`) ladder. A contract
climbs one rung at a time; you cannot skip a rung.

| Level | Meaning | Tooling |
|-------|---------|---------|
| **L1** | Equations declared — the behaviour is written down as a formula with domain, codomain, and invariants. | `contracts/*.yaml` |
| **L2** | Falsification `#[test]`s — each prediction the contract makes has a test that *fails loudly* if the behaviour regresses. | `cargo test` |
| **L3** | Kani harnesses **VERIFIED** — the pure decision function is model-checked exhaustively over its whole input domain. | `cargo kani` |
| **L4** | Lean 4 proofs — the same invariant is proved as a theorem, `0 sorry`, machine-checked. | `lean lean/*.lean` |
| **L5** | Bindings verified — every equation is bound to a real function in the source (`contracts/binding.yaml`), and `pv` confirms that function still exists. | `pv proof-status --binding` |

**L5 is the top of the ladder and the only place a contract is both *proved* and
*provably wired to the code that ships*.** A binding is falsifiable: rename the
implementing function and the contract drops below L5 on the next
`pv proof-status` run.

---

## What `pv proof-status` actually reports

This is the live output of `pv proof-status contracts/ --binding contracts/binding.yaml`
against the tree (verified while writing this doc, `pv 0.49.0`):

```
Proof Status (11 contracts)

  Contract                            Level Obligs Tests Kani Lean  Bindings
  ────────────────────────────────────────────────────────────────────────
  bidirectional-sync-v1               L5      4     5    2    4    5/5
  hub-protocol-v1                     L5      3     5    1    3    5/5
  incremental-sync-v1                 L5      4     6    1    4    5/5
  delta-sync-v1                       L3      3     3    1    0    0/3
  hash-integrity-v1                   L3      4     4    1    0    0/4
  local-to-local-recursive-v1         L3      3     4    1    0    0/3
  path-safety-v1                      L3      3     3    1    0    0/3
  recursive-push-v1                   L3      3     4    1    0    0/3
  recursive-remote-pull-v1            L3      3     4    1    0    0/3
  single-file-sync-roundtrip-v1       L3      3     4    1    0    0/3
  streaming-transfer-v1               L3      3     3    1    0    0/2

  Totals: 36 obligations, 45 tests, 12 kani, 11 lean proved, 15/39 bound
```

Two honest takeaways:

1. **Three contracts reach L5** — `incremental-sync-v1`, `bidirectional-sync-v1`,
   and `hub-protocol-v1`. These are the three architectural layers of copia
   (L1 one-way incremental, L2 bidirectional, L3 hub). They are, at time of
   writing, the only L5 contracts in the Sovereign AI Stack fleet: equations →
   falsification tests → Kani → Lean → verified source bindings, all rungs
   climbed.
2. **The other eight are L3** — the rsync delta primitives and the recursive /
   transport variants. They have declared equations (L1), falsification tests
   (L2), and a verified Kani harness each (L3), but no Lean theorem and no source
   binding yet (`0` in the Lean and Bindings columns). They are honestly *not*
   proved to the same depth as the three flagship contracts, and this doc does
   not pretend otherwise.

> The `pv proof-status` output also prints a generic "Kernel Classes A–E" table.
> That is boilerplate for LLM-kernel contracts and is **not** relevant to copia —
> ignore it. copia's contracts are all `kind: pattern` / distributed-systems
> contracts.

---

## The three L5 contracts in detail

Each L5 contract declares a handful of **equations**. Some equations are proved
(Kani + Lean); others describe I/O behaviour that can only be *tested*, not
proved as a pure function. The tables below make that split explicit.

### 1. `incremental-sync-v1` — the rsync-parity core (L5)

The one-way recursive mirror: transfer only new-or-changed files (quick check on
`size` + `mtime`), deliver atomically (`.copia-tmp` + `rename`), preserve mtime,
honour `--exclude`, mirror with `--delete`, preview with `--dry-run`. Applies to
all three directions: `local → local`, push `local → remote`, pull
`remote → local`.

| Equation | Guarantee | Status |
|----------|-----------|--------|
| `quick_check` | transfer(p) ⇔ absent, or size differs, or mtime differs | **PROVED** — Kani `plan-kani-001` + Lean `QuickCheckCorrect`, `SkipGuarantee` |
| `exclude_filtering` | excluded(p) ⇒ p not transferred and not deleted | **PROVED** — Lean `ExcludeSafety` |
| `delete_mirror` | without `--delete` the delete set is empty | **PROVED** — Lean `DeleteOptIn` |
| `atomic_delivery` | a reader observes only the *old* or the *new* file, never a torn write; no `.copia-tmp` survives | **TESTED** — `FALSIFY-INCR-003` (I/O behaviour, temp+rename) |
| `mtime_preservation` | after transfer, `mtime(dst) == mtime(src)` so the next quick check is stable | **TESTED** — `FALSIFY-INCR-001` (I/O behaviour, `touch -d @secs` / `set_modified`) |

Implementing functions (from `binding.yaml`): `plan::needs_transfer`,
`plan::is_excluded`, `plan::build_plan`, `incremental::deliver_local`,
`meta::set_local_mtime`.

Six falsification tests (`FALSIFY-INCR-001..006`) cover idempotence, quick-check
correctness, atomic delivery, exclude safety, delete mirror + opt-in, and
dry-run purity.

### 2. `bidirectional-sync-v1` — the L2 two-way reconciler (L5)

Unison-shaped two-party sync. **blake3 is the sole oracle** for
equal/changed/conflict/propagate/delete — `mtime` never decides anything. A
one-sided absence becomes a delete **only with positive archive evidence**;
with no trusted base it is a create. Divergence becomes a convergent
conflict-copy that preserves BOTH versions on BOTH sides, with a deterministic
blake3 winner (never a clock winner).

| Equation | Guarantee | Status |
|----------|-----------|--------|
| `positive_evidence_delete` | base=None ⇒ never a delete; DeleteA ⇒ survivor's hash == base hash | **PROVED** — Kani `reconcile-kani-001` + `reconcile-kani-002`; Lean `NoBaseNeverDeletes` + `DeleteNeedsEvidence` |
| `blake3_oracle` | identical content is never a conflict, regardless of base | **PROVED** — Lean `Blake3Oracle` |
| `three_way_reconcile` | divergent edits (both differ from base and each other) ⇒ Conflict, never a silent pick | **PROVED** — Lean `ConflictNotSilentPick` |
| `convergent_conflict_copy` | both replicas end identical with both versions preserved | **TESTED** — `FALSIFY-BIDIR-004` (I/O apply behaviour) |
| `commit_then_record` | data is committed (tmp+rename) *before* the archive entry is recorded; archive is never ahead of data; no WAL | **TESTED** — encoded in the e2e bisync tests |

Implementing functions: `reconcile::reconcile_path`, `reconcile::reconcile`,
`meta::fingerprint_path`, `bidir::apply`, `archive::save`.

Five falsification tests (`FALSIFY-BIDIR-001..005`). This is the
**data-loss-critical** contract: the two Kani harnesses and two Lean theorems on
`positive_evidence_delete` are the machine-checked reason a lost or corrupt
archive can never turn a create into a destructive delete.

### 3. `hub-protocol-v1` — the L3 distributed hub (L5)

N clients → 1 hub, star topology. `copia serve <root>` speaks a framed, versioned,
CBOR request/response protocol over one persistent stdin/stdout pipe.
Concurrency safety is **CAS-on-blake3** under a brief tree commit-lock: a write
commits only when the hub's current content hash equals what the client last
observed. A stale CAS never overwrites — it lands a conflict-copy. No leases, no
fencing tokens, no vector clocks: the single hub is the total order and the
content-hash CAS is immune to pause/partition and ABA.

| Equation | Guarantee | Status |
|----------|-----------|--------|
| `cas_safety` | commit ⇔ hub current hash == client expected hash; a stale CAS never commits | **PROVED** — Kani `hub-kani-001` + Lean `StaleCasNeverCommits` |
| `bounded_framing` | a length prefix > `MAX_FRAME` is rejected *before* allocating (DoS guard) | **PROVED** — Lean `BoundedFrame` |
| `path_traversal_guard` | an accepted path has no `..` and no root/absolute component — a client cannot escape the served root | **PROVED** — Lean `NoTraversal` |
| `magic_prologue` | the hub aborts unless the first bytes are exactly `MAGIC` (an ssh banner / motd cannot desync the stream) | **TESTED** — `FALSIFY-HUB-004` |
| `content_integrity` | a Put is rejected unless `blake3(streamed bytes)` == the client's claimed hash | **TESTED** — enforced in `serve::handle_put`, exercised by the e2e hub tests |

Implementing functions: `wire::cas_decide`, `wire::read_frame`, `wire::read_magic`,
`serve::safe_join`, `serve::handle_put`.

Five falsification tests (`FALSIFY-HUB-001..005`).

---

## The Kani harnesses (L3 — exhaustive model checking)

Kani proves the pure decision functions over their **entire** input domain
(`bound: 0`, `strategy: exhaustive`) — not sampled, not fuzzed. The four
flagship harnesses live under `#[cfg(kani)] mod kani_proofs` in the source:

| Harness | Function | File | Proves |
|---------|----------|------|--------|
| `plan-kani-001` | `needs_transfer_iff_new_or_differing` | `src/bin/copia/plan.rs` | quick check transfers iff new/size/mtime differ |
| `reconcile-kani-001` | `no_base_never_deletes` | `src/bin/copia/reconcile.rs` | no trusted base ⇒ never a delete |
| `reconcile-kani-002` | `delete_requires_positive_evidence` | `src/bin/copia/reconcile.rs` | a delete implies survivor hash == base hash |
| `hub-kani-001` | `stale_cas_never_commits` | `src/bin/copia/wire.rs` | a stale CAS expected never commits |

The eight L3 contracts each carry one additional Kani harness (the delta / hash /
recursive / streaming primitives). `pv proof-status` totals **12 Kani harnesses**
across all contracts.

Run them (Kani must be installed — `cargo install --locked kani-verifier && cargo kani setup`):

```bash
cargo kani --features cli --harness needs_transfer_iff_new_or_differing
cargo kani --features cli --harness no_base_never_deletes
cargo kani --features cli --harness delete_requires_positive_evidence
cargo kani --features cli --harness stale_cas_never_commits
# …or run every harness in the crate:
cargo kani --features cli
```

Each prints `VERIFICATION:- SUCCESSFUL`.

---

## The 11 Lean 4 theorems (L4 — machine-checked proof)

The Kani harnesses check the Rust functions; the Lean proofs check the *same
invariants* as mathematics, independently, in a second tool. All 11 theorems are
`0 sorry` and machine-verified. They live in `lean/*.lean` and model content
hashes as `Nat`, mirroring the pure Rust decision functions.

### `lean/IncrementalSync.lean` (mirrors `plan::needs_transfer` / `build_plan`)

| Theorem | Guarantees |
|---------|-----------|
| `QuickCheckCorrect` | a present file transfers iff its size or mtime differs from the destination. |
| `SkipGuarantee` | an identical `(size, mtime)` file is *never* re-transferred. |
| `DeleteOptIn` | without `--delete`, the computed delete set is empty (`[]`). |
| `ExcludeSafety` | an excluded path is never transferred and never deleted. |

### `lean/BidirectionalReconcile.lean` (mirrors `reconcile::reconcile_path`)

| Theorem | Guarantees |
|---------|-----------|
| `NoBaseNeverDeletes` | with no trustworthy base, reconcile never emits `DeleteA` or `DeleteB` — a lost/corrupt archive cannot turn a create into data loss. |
| `DeleteNeedsEvidence` | a `DeleteA` is emitted *only* with positive evidence: the surviving side's content equals the archived base. |
| `Blake3Oracle` | identical content on both sides is never a conflict, regardless of the base — content is the sole equality oracle. |
| `ConflictNotSilentPick` | genuinely divergent content (both differ from base and from each other) yields a `Conflict`, never a silent propagate or delete. |

### `lean/HubCas.lean` (mirrors `wire::cas_decide` / `read_frame` / `serve::safe_join`)

| Theorem | Guarantees |
|---------|-----------|
| `StaleCasNeverCommits` | a stale `expected` never commits — the lost-update fence for concurrent hub writers. |
| `BoundedFrame` | a length over `MAX_FRAME` is rejected before allocating — the DoS guard. |
| `NoTraversal` | an accepted path contains no `..` and no root/absolute component, so a client can never escape the served root. |

Verify all of them:

```bash
for l in lean/*.lean; do lean "$l"; done   # each prints nothing on success (0 errors, no sorry)
```

`pv proof-status` counts **11 lean proved** — exactly these theorems.

---

## The binding registry (`contracts/binding.yaml`) — L5

L4 proves the maths. **L5 proves the maths is wired to the code that ships.**
`contracts/binding.yaml` maps every equation of the three L5 contracts to the
exact function that implements it, with its module path and signature, e.g.:

```yaml
- contract: bidirectional-sync-v1.yaml
  equation: positive_evidence_delete
  module_path: copia::reconcile
  function: reconcile
  signature: 'fn reconcile(a: &FpMap, b: &FpMap, base: &FpMap, trust_base: bool) -> Vec<(PathBuf, Action)>'
  status: implemented
```

`pv proof-status --binding` **verifies each binding against the source** — a
binding only counts if `fn <function>` actually exists in `module_path`. This is
what makes L5 falsifiable rather than aspirational: the three L5 contracts each
report `5/5` bound because all fifteen of their equations resolve to live
functions. Rename one and the count drops, dragging the contract below L5 on the
next run. The eight L3 contracts report `0/N` — they are honestly unbound.

---

## `make contracts` — reproduce everything

The single command that runs the whole ladder is `make contracts`
(from the `Makefile`):

```make
contracts:
	@for c in contracts/*-v1.yaml; do pv validate "$$c" || exit 1; done
	@echo "== Lean 4 proofs (L4) =="; for l in lean/*.lean; do lean "$$l" || exit 1; done
	@echo "== proof levels (expect L5) =="; pv proof-status contracts/ --binding contracts/binding.yaml
	cargo test --features cli --test contract_falsification --test e2e_bidir --test e2e_hub --test integration_all
```

It performs, in order:

1. **Schema-validate** every `contracts/*-v1.yaml` (`pv validate`).
2. **Machine-check the Lean proofs** — `lean` on each `lean/*.lean` (L4).
3. **Report proof levels** — `pv proof-status … --binding` (verifies the L5
   bindings; expect three `L5` rows).
4. **Run the falsification tests** (L2) — `contract_falsification`, `e2e_bidir`,
   `e2e_hub`, `integration_all`.

```bash
make contracts
```

Prerequisites on `PATH`: `pv` (provable-contracts CLI, `0.49.0`+) and `lean`
(Lean 4 / elan). The Kani harnesses are run separately with `cargo kani` (see
above) because Kani requires its own toolchain setup.

---

## The honest summary

- **PROVED (Kani + Lean, exhaustive):** the *decisions that can lose your data* —
  the quick-check transfer/skip rule; exclude and `--delete` opt-in safety; that a
  lost archive never deletes; that a delete needs positive blake3 evidence; that
  blake3 (not mtime) is the sole conflict oracle; that divergence is never a
  silent pick; that a stale hub CAS never commits; that frames are bounded; and
  that no client path escapes the served root. Eleven Lean theorems, four flagship
  Kani harnesses, `0 sorry`.
- **TESTED (falsification `#[test]`s):** the *I/O behaviours* that are not pure
  functions — atomic temp+rename delivery, mtime preservation across a run,
  content-integrity re-hashing, the magic-prologue guard, streaming, and the
  full end-to-end round trips. Real code paths, real files, real `ssh localhost`,
  failing loudly on regression — but exercised, not exhaustively proved.
- **Levels:** three contracts at **L5** (bound + proved), eight rsync/transport
  primitives at **L3** (Kani-verified, not yet Lean-proved or bound).

That distinction is the whole point. copia does not claim its file I/O is
theorem-proved; it claims the *dangerous algebra underneath* is — and gives you
`make contracts`, `cargo kani`, and `lean lean/*.lean` to check it for yourself.
