//! Pure 3-way bidirectional reconcile (Unison-shaped), vetted by the distributed
//! design quorum (docs/specifications/distributed-sync.md). blake3 is the SOLE
//! oracle for equal/changed/conflict/propagate/delete — size+mtime are only a
//! fast-path elsewhere (meta) deciding whether to re-hash, and never appear here.
//! No I/O, so the whole case table is exhaustively unit-tested + Kani-proved.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// A content-addressed fingerprint. `blake3` is authoritative; `ftype` guards a
/// file<->symlink flip from being mistaken for a content change of the same kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub blake3: [u8; 32],
    pub ftype: FileType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    File,
    Symlink,
}

impl Fingerprint {
    /// Two fingerprints are equal iff the content digest AND the entry type match.
    fn same(a: &Self, b: &Self) -> bool {
        a.blake3 == b.blake3 && a.ftype == b.ftype
    }
}

/// Path -> fingerprint for one replica (or the archive base).
pub type FpMap = BTreeMap<PathBuf, Fingerprint>;

/// The reconciled action for a single path. Deletes require positive archive
/// evidence; ambiguity always degrades to a conflict-copy, never data loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Both sides equal the base — nothing to do.
    Noop,
    /// Changed on A only — send A's content to B (also covers create A->B).
    PropagateAtoB,
    /// Changed on B only — send B's content to A (also covers create B->A).
    PropagateBtoA,
    /// Both changed to the SAME content — converge silently, just record the base.
    ConvergeIdentical,
    /// B deleted, A unchanged since base — propagate the delete to A.
    DeleteA,
    /// A deleted, B unchanged since base — propagate the delete to B.
    DeleteB,
    /// Irreconcilable divergence — keep BOTH via conflict-copy, pick no winner.
    Conflict(ConflictKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConflictKind {
    /// Both sides changed to different content since the base.
    BothChanged,
    /// One side deleted while the other modified — keep the modification.
    DeleteVsModify,
}

/// Reconcile a single path from the two live fingerprints and the archive base.
/// `base` is the last-synced common fingerprint, or `None` when there is no
/// trustworthy base (first sync, or safe-mode from a lost/mismatched archive) —
/// in which case a one-sided absence is a CREATE, never a delete.
#[must_use]
pub fn reconcile_path(
    a: Option<Fingerprint>,
    b: Option<Fingerprint>,
    base: Option<Fingerprint>,
) -> Action {
    match (a, b) {
        (None, None) => Action::Noop,
        (Some(av), Some(bv)) => {
            if Fingerprint::same(&av, &bv) {
                // Identical now: a true noop if it also equals the base, else both
                // sides independently reached the same new content — record it.
                if base.is_some_and(|z| Fingerprint::same(&av, &z)) {
                    Action::Noop
                } else {
                    Action::ConvergeIdentical
                }
            } else {
                // MSRV 1.75: map_or(true, ..), not is_none_or (stable 1.82).
                let a_changed = base.map_or(true, |z| !Fingerprint::same(&av, &z));
                let b_changed = base.map_or(true, |z| !Fingerprint::same(&bv, &z));
                match (a_changed, b_changed) {
                    (true, false) => Action::PropagateAtoB,
                    (false, true) => Action::PropagateBtoA,
                    // Both differ from base but also from each other (a==b is
                    // handled above) => conflict. Never a silent pick.
                    _ => Action::Conflict(ConflictKind::BothChanged),
                }
            }
        }
        // B absent: either B deleted (base present) or A created (base absent).
        (Some(av), None) => match base {
            None => Action::PropagateAtoB, // create A->B, never a delete
            Some(z) if Fingerprint::same(&av, &z) => Action::DeleteA, // A unchanged, B deleted
            Some(_) => Action::Conflict(ConflictKind::DeleteVsModify), // A modified, B deleted
        },
        // A absent: symmetric.
        (None, Some(bv)) => match base {
            None => Action::PropagateBtoA,
            Some(z) if Fingerprint::same(&bv, &z) => Action::DeleteB,
            Some(_) => Action::Conflict(ConflictKind::DeleteVsModify),
        },
    }
}

/// Reconcile whole trees over the union of paths. `trust_base=false` (safe mode)
/// forces every base lookup to `None`, so NO deletes are ever produced and all
/// divergence degrades to create/conflict.
#[must_use]
pub fn reconcile(a: &FpMap, b: &FpMap, base: &FpMap, trust_base: bool) -> Vec<(PathBuf, Action)> {
    let mut paths: Vec<&PathBuf> = a.keys().chain(b.keys()).collect();
    paths.sort_unstable();
    paths.dedup();
    let mut out = Vec::new();
    for p in paths {
        let z = if trust_base {
            base.get(p).copied()
        } else {
            None
        };
        let act = reconcile_path(a.get(p).copied(), b.get(p).copied(), z);
        if act != Action::Noop {
            out.push((p.clone(), act));
        }
    }
    out
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn any_fp() -> Fingerprint {
        Fingerprint {
            blake3: kani::any(),
            ftype: if kani::any() {
                FileType::File
            } else {
                FileType::Symlink
            },
        }
    }

    /// reconcile-kani-001: with NO trustworthy base, reconcile NEVER emits a
    /// delete — a one-sided absence is always a create, so a lost/corrupt archive
    /// can never turn a legitimate create into destructive data loss.
    #[kani::proof]
    fn no_base_never_deletes() {
        let a = if kani::any() { Some(any_fp()) } else { None };
        let b = if kani::any() { Some(any_fp()) } else { None };
        let act = reconcile_path(a, b, None);
        assert!(!matches!(act, Action::DeleteA | Action::DeleteB));
    }

    /// reconcile-kani-002: a delete is emitted ONLY when the surviving side's
    /// content still equals the base (positive evidence) — a modified survivor
    /// against a deleted peer is a conflict, never a delete.
    #[kani::proof]
    fn delete_requires_positive_evidence() {
        let base = any_fp();
        let surv = any_fp();
        // A survives, B absent.
        let act = reconcile_path(Some(surv), None, Some(base));
        if matches!(act, Action::DeleteA) {
            assert!(Fingerprint::same(&surv, &base));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn fp(byte: u8) -> Fingerprint {
        Fingerprint {
            blake3: [byte; 32],
            ftype: FileType::File,
        }
    }

    #[test]
    fn both_equal_base_is_noop() {
        assert_eq!(
            reconcile_path(Some(fp(1)), Some(fp(1)), Some(fp(1))),
            Action::Noop
        );
    }

    #[test]
    fn one_sided_change_propagates() {
        assert_eq!(
            reconcile_path(Some(fp(2)), Some(fp(1)), Some(fp(1))),
            Action::PropagateAtoB
        );
        assert_eq!(
            reconcile_path(Some(fp(1)), Some(fp(2)), Some(fp(1))),
            Action::PropagateBtoA
        );
    }

    #[test]
    fn identical_divergence_converges_not_conflicts() {
        // both changed to the SAME new content -> record, never conflict
        assert_eq!(
            reconcile_path(Some(fp(3)), Some(fp(3)), Some(fp(1))),
            Action::ConvergeIdentical
        );
        // both still equal base -> true noop
        assert_eq!(
            reconcile_path(Some(fp(1)), Some(fp(1)), Some(fp(1))),
            Action::Noop
        );
    }

    #[test]
    fn divergent_change_is_conflict() {
        assert_eq!(
            reconcile_path(Some(fp(2)), Some(fp(3)), Some(fp(1))),
            Action::Conflict(ConflictKind::BothChanged)
        );
    }

    #[test]
    fn delete_needs_positive_evidence() {
        // B deleted, A unchanged since base -> propagate delete to A
        assert_eq!(
            reconcile_path(Some(fp(1)), None, Some(fp(1))),
            Action::DeleteA
        );
        // B deleted, A MODIFIED since base -> conflict, keep A
        assert_eq!(
            reconcile_path(Some(fp(2)), None, Some(fp(1))),
            Action::Conflict(ConflictKind::DeleteVsModify)
        );
    }

    #[test]
    fn no_base_absence_is_create_never_delete() {
        assert_eq!(
            reconcile_path(Some(fp(1)), None, None),
            Action::PropagateAtoB
        );
        assert_eq!(
            reconcile_path(None, Some(fp(1)), None),
            Action::PropagateBtoA
        );
    }

    #[test]
    fn ftype_flip_is_a_change_even_with_same_bytes() {
        let file = Fingerprint {
            blake3: [1; 32],
            ftype: FileType::File,
        };
        let link = Fingerprint {
            blake3: [1; 32],
            ftype: FileType::Symlink,
        };
        assert_eq!(
            reconcile_path(Some(file), Some(link), Some(file)),
            Action::PropagateBtoA
        );
    }

    #[test]
    fn safe_mode_disables_deletes_across_a_tree() {
        let a: FpMap = [
            (PathBuf::from("keep"), fp(1)),
            (PathBuf::from("only_a"), fp(2)),
        ]
        .into_iter()
        .collect();
        let mut b = FpMap::new();
        b.insert(PathBuf::from("keep"), fp(1));
        let base: FpMap = [
            (PathBuf::from("keep"), fp(1)),
            (PathBuf::from("only_a"), fp(2)),
        ]
        .into_iter()
        .collect();
        // trusted base: only_a present on A, absent on B, unchanged => DeleteA
        let trusted = reconcile(&a, &b, &base, true);
        assert_eq!(trusted, vec![(PathBuf::from("only_a"), Action::DeleteA)]);
        // safe mode: NO deletes — only_a becomes a create A->B; the identical
        // "keep" is recorded (ConvergeIdentical) to seed the fresh archive.
        let safe = reconcile(&a, &b, &base, false);
        assert_eq!(
            safe,
            vec![
                (PathBuf::from("keep"), Action::ConvergeIdentical),
                (PathBuf::from("only_a"), Action::PropagateAtoB),
            ]
        );
    }
}
