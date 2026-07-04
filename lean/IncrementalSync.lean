/-
  L4 Lean 4 proofs for copia `incremental-sync-v1`.
  Self-contained (no Mathlib): the quick-check + delete-opt-in facts are
  decidable boolean identities. Mirrors `plan::needs_transfer` / `build_plan`.
  Verify: `lean lean/IncrementalSync.lean` (0 errors, no `sorry`).
-/
namespace ProvableContracts.Copia

/-- The per-file quick-check decision, mirroring `plan::needs_transfer`:
    transfer iff the destination is absent or differs in size or mtime. -/
def needsTransfer (ssize dsize : Nat) (smtime dmtime : Int) (present : Bool) : Bool :=
  if present then (ssize != dsize || smtime != dmtime) else true

/-- Theorems.QuickCheckCorrect — a present file transfers iff size or mtime differ. -/
theorem QuickCheckCorrect (ssize dsize : Nat) (smtime dmtime : Int) :
    needsTransfer ssize dsize smtime dmtime true
      = (ssize != dsize || smtime != dmtime) := by
  simp [needsTransfer]

/-- Theorems.SkipGuarantee — an identical (size, mtime) file is never re-transferred. -/
theorem SkipGuarantee (size : Nat) (mtime : Int) :
    needsTransfer size size mtime mtime true = false := by
  simp [needsTransfer]

/-- Theorems.DeleteOptIn — without `--delete` the delete set is empty. -/
def deleteSet (withDelete : Bool) (candidates : List String) : List String :=
  if withDelete then candidates else []

theorem DeleteOptIn (candidates : List String) :
    deleteSet false candidates = [] := by
  rfl

end ProvableContracts.Copia
