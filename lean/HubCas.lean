/-
  L4 Lean 4 proof for copia `hub-protocol-v1` — the CAS linearizability/safety
  invariant (mirrors wire::cas_decide). Hashes modelled as Option Nat (None=absent).
  Verify: `lean lean/HubCas.lean` (0 errors, no sorry).
-/
namespace ProvableContracts.Copia.Hub

/-- The hub CAS decision: commit iff the current state equals what the client
    last observed (`expected`). -/
def casCommit (current expected : Option Nat) : Bool := current = expected

/-- Theorems.StaleCasNeverCommits — a client whose `expected` differs from the
    hub's CURRENT state never commits, so a concurrent write is never silently
    lost (the linearizability fence). -/
theorem StaleCasNeverCommits (current expected : Option Nat) :
    casCommit current expected = true → current = expected := by
  intro h
  simpa [casCommit] using h

end ProvableContracts.Copia.Hub
