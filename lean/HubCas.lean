/-
  L4/L5 Lean 4 proofs for copia `hub-protocol-v1`. Pure decisions modelled
  faithfully. Verify: `lean lean/HubCas.lean` (0 errors, no sorry).
-/
namespace ProvableContracts.Copia.Hub

/-- casCommit: commit iff current == expected (wire::cas_decide). -/
def casCommit (current expected : Option Nat) : Bool := current = expected

/-- Theorems.StaleCasNeverCommits — a stale expected never commits (lost-update fence). -/
theorem StaleCasNeverCommits (current expected : Option Nat) :
    casCommit current expected = true → current = expected := by
  intro h; simpa [casCommit] using h

/-- Frame acceptance gate (wire::read_frame). -/
def frameOk (maxFrame len : Nat) : Bool := len ≤ maxFrame

/-- Theorems.BoundedFrame — a length over MAX_FRAME is rejected before allocating. -/
theorem BoundedFrame (maxFrame len : Nat) :
    frameOk maxFrame len = true → len ≤ maxFrame := by
  intro h; simpa [frameOk] using h

/-- A path component (serve::safe_join sees these). -/
inductive Comp | normal (s : String) | parentDir | rootDir
  deriving DecidableEq

/-- safe_join accepts a path iff NO component is `..` or a root/absolute marker. -/
def accepted : List Comp → Bool
  | [] => true
  | Comp.parentDir :: _ => false
  | Comp.rootDir :: _ => false
  | (Comp.normal _) :: rest => accepted rest

/-- Theorems.NoTraversal — an accepted path contains no `..` and no root/absolute
    component, so a client can never escape the served root. -/
theorem NoTraversal (p : List Comp) :
    accepted p = true → Comp.parentDir ∉ p ∧ Comp.rootDir ∉ p := by
  induction p with
  | nil => simp
  | cons c rest ih =>
      cases c with
      | parentDir => simp [accepted]
      | rootDir => simp [accepted]
      | normal s =>
          intro h
          simp only [accepted] at h
          have hr := ih h
          simp only [List.mem_cons, not_or]
          exact ⟨⟨by simp, hr.1⟩, ⟨by simp, hr.2⟩⟩

end ProvableContracts.Copia.Hub
