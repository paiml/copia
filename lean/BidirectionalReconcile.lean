/-
  L4 Lean 4 proofs for copia `bidirectional-sync-v1` — the data-loss-critical
  reconcile safety invariants (mirrors reconcile::reconcile_path). Content hashes
  modelled as Nat. Verify: `lean lean/BidirectionalReconcile.lean` (0 errors, no sorry).
-/
namespace ProvableContracts.Copia.Bidir

inductive Action
  | Noop | PropA | PropB | Converge | DeleteA | DeleteB | Conflict
  deriving DecidableEq

/-- Reconcile one path from the two live hashes and the archive base hash. -/
def reconcile : Option Nat → Option Nat → Option Nat → Action
  | none,    none,    _       => Action.Noop
  | some _,  none,    none    => Action.PropA
  | some av, none,    some z  => if av = z then Action.DeleteA else Action.Conflict
  | none,    some _,  none    => Action.PropB
  | none,    some bv, some z  => if bv = z then Action.DeleteB else Action.Conflict
  | some av, some bv, none    => if av = bv then Action.Converge else Action.Conflict
  | some av, some bv, some z  =>
      if av = bv then (if av = z then Action.Noop else Action.Converge)
      else if av = z then Action.PropB
      else if bv = z then Action.PropA
      else Action.Conflict

/-- Theorems.NoBaseNeverDeletes — with no trustworthy base, reconcile NEVER emits
    a delete, so a lost/corrupt archive cannot turn a create into data loss. -/
theorem NoBaseNeverDeletes (a b : Option Nat) :
    reconcile a b none ≠ Action.DeleteA ∧ reconcile a b none ≠ Action.DeleteB := by
  cases a <;> cases b <;> simp only [reconcile] <;>
    first
      | exact ⟨by decide, by decide⟩
      | (split <;> exact ⟨by decide, by decide⟩)

/-- Theorems.DeleteNeedsEvidence — a DeleteA is emitted only with positive
    evidence: the surviving side's content equals the base. -/
theorem DeleteNeedsEvidence (av base : Nat) :
    reconcile (some av) none (some base) = Action.DeleteA → av = base := by
  simp only [reconcile]
  split
  next h => intro _; exact h
  next => intro h; exact absurd h (by decide)

end ProvableContracts.Copia.Bidir
