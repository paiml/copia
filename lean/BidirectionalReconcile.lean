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

/-- Theorems.Blake3Oracle — identical content on both sides is NEVER a conflict
    (content is the sole equality oracle), regardless of the base. -/
theorem Blake3Oracle (h : Nat) (base : Option Nat) :
    reconcile (some h) (some h) base ≠ Action.Conflict := by
  cases base with
  | none => simp [reconcile]
  | some z =>
      by_cases hz : h = z
      · simp [reconcile, hz]
      · simp [reconcile, hz]

/-- Theorems.ConflictNotSilentPick — genuinely divergent content (both differ
    from the base and from each other) yields a Conflict, never a silent
    propagate or delete. -/
theorem ConflictNotSilentPick (a b z : Nat) :
    a ≠ b → a ≠ z → b ≠ z →
    reconcile (some a) (some b) (some z) = Action.Conflict := by
  intro hab haz hbz
  simp only [reconcile]
  rw [if_neg hab, if_neg haz, if_neg hbz]

end ProvableContracts.Copia.Bidir
