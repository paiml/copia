//! `copia verify` — prove two trees are byte-identical, changing nothing.
//!
//! # Why this exists
//!
//! copia is the Sovereign AI Stack's rsync replacement. Until this command it
//! was that for *transfer* and not for *archival*, so the one operation where
//! being wrong destroys data — move a tree, then delete the original — was
//! still delegated to rsync. forjar's `nas_archive` moves 755 GB to the NAS and
//! deletes the source, and it stayed on
//! `rsync -a --checksum --dry-run --itemize-changes` purely because copia could
//! not express that pass (copia#46).
//!
//! # The four guarantees, and why each is load-bearing
//!
//! 1. **Compares by CONTENT.** Never size+mtime. A file truncated in place with
//!    its mtime preserved is exactly the corruption a quick-check misses, and it
//!    is indistinguishable from a healthy file until you read the bytes.
//!
//! 2. **Provably read-only.** This runs against the destination that is about to
//!    justify deleting the source. It opens files for reading and holds no write
//!    handle anywhere; `verify_read_only_never_writes` asserts that by
//!    fingerprinting both trees before and after a run and requiring both to be
//!    unchanged, which catches a write this module does not even know it makes.
//!
//! 3. **Per-path outcome, machine-readable.** Three outcomes that must never
//!    collapse: `identical`, `differs`, and `unreadable`. The third is the one
//!    that matters — `discover_local_fingerprints` SKIPS a file it cannot read
//!    ("never guessed"), which is right for sync and catastrophic here, because
//!    a skipped file is absent from both maps and therefore reads as agreement.
//!    A verifier that cannot tell "same" from "I could not look" is worse than
//!    none: it produces a confident answer with no evidence behind it. That
//!    failure has already happened downstream — `nas_archive`'s predecessor
//!    printed `verified: 0 files differ` when the comparison itself had failed.
//!
//! 4. **Distinct exit codes.** A caller deleting 755 GB branches on these:
//!
//!    | code | meaning |
//!    |---|---|
//!    | 0 | identical — safe to delete the source |
//!    | 1 | differences found — do not delete |
//!    | 2 | could not compare — do not delete. NOT the same as 1: the tree may be perfect, and we have no evidence either way |
//!    | 3 | usage or IO error before comparison began |

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::meta::fingerprint_path;
use super::reconcile::Fingerprint;
use super::transfer::discover_local_files;

/// Exit codes. Public contract — a caller branches on these before deleting.
pub const EXIT_IDENTICAL: i32 = 0;
pub const EXIT_DIFFERS: i32 = 1;
pub const EXIT_UNREADABLE: i32 = 2;
pub const EXIT_ERROR: i32 = 3;

/// What we concluded about one path.
///
/// `Identical` is never constructed internally — identical paths are counted,
/// not listed, because printing 400,000 "identical" lines on a 755 GB tree
/// buries the four that matter. It stays in the enum because the token set is a
/// PUBLIC contract a caller may match on, and a variant that exists only when
/// something happens to emit it is not a contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Outcome {
    Identical,
    /// Present in both, content or entry type differs.
    Differs,
    /// Present in source, absent from destination.
    Missing,
    /// Present in destination, absent from source.
    Extra,
    /// Could not be read on one side or the other. NOT a difference, and
    /// emphatically not agreement.
    Unreadable,
}

impl Outcome {
    /// The stable token emitted per path. Machine-readable; never localised.
    pub const fn token(self) -> &'static str {
        match self {
            Self::Identical => "identical",
            Self::Differs => "differs",
            Self::Missing => "missing",
            Self::Extra => "extra",
            Self::Unreadable => "unreadable",
        }
    }
}

/// The verdict over a whole tree.
#[derive(Debug, Default)]
pub struct Report {
    pub identical: usize,
    pub differs: Vec<PathBuf>,
    pub missing: Vec<PathBuf>,
    pub extra: Vec<PathBuf>,
    pub unreadable: Vec<PathBuf>,
}

impl Report {
    /// Exit code for this report.
    ///
    /// `Unreadable` outranks `Differs` deliberately. Both refuse the delete, but
    /// they mean different things and a caller may want to retry one and not the
    /// other — and collapsing them would let "I could not look" be reported as
    /// "I looked and found a difference", which is a claim we did not earn.
    pub fn exit_code(&self) -> i32 {
        exit_code_for(
            self.differs.len(),
            self.missing.len(),
            self.extra.len(),
            self.unreadable.len(),
        )
    }

    /// True only when the source may safely be deleted.
    pub fn safe_to_delete_source(&self) -> bool {
        safe_to_delete_for(
            self.identical,
            self.differs.len(),
            self.missing.len(),
            self.extra.len(),
            self.unreadable.len(),
        )
    }
}

/// The exit-code decision, as a pure function of the counts.
///
/// Extracted from `Report` so it can be PROVED rather than sampled: `Report`
/// carries `Vec`s, and a Kani harness that constructs one drags the allocator
/// and `core::fmt` into the model — measured elsewhere in this fleet at 117
/// minutes for a 216-case space. Over four `usize` counts the same property is
/// exhaustive and instant.
#[must_use]
pub const fn exit_code_for(differs: usize, missing: usize, extra: usize, unreadable: usize) -> i32 {
    if unreadable > 0 {
        EXIT_UNREADABLE
    } else if differs > 0 || missing > 0 || extra > 0 {
        EXIT_DIFFERS
    } else {
        EXIT_IDENTICAL
    }
}

/// Whether the source may be deleted, as a pure function of the counts.
///
/// `identical > 0` is not pedantry. Two empty trees produce all-zero counts and
/// would otherwise authorise deleting a source on a comparison that examined
/// nothing — a conclusion with no evidence under it, which is the same shape as
/// a green test over an empty set.
#[must_use]
pub const fn safe_to_delete_for(
    identical: usize,
    differs: usize,
    missing: usize,
    extra: usize,
    unreadable: usize,
) -> bool {
    identical > 0 && exit_code_for(differs, missing, extra, unreadable) == EXIT_IDENTICAL
}

/// One side's scan: fingerprints we got, and paths we could not read.
struct Scan {
    fps: std::collections::BTreeMap<PathBuf, Fingerprint>,
    unreadable: BTreeSet<PathBuf>,
}

/// Fingerprint a tree, RECORDING failures instead of dropping them.
///
/// This is the one place this command deliberately does not reuse
/// `discover_local_fingerprints`: that function skips what it cannot read, which
/// is correct for sync (retry next run) and wrong here (a skipped file is absent
/// from both maps and therefore looks like agreement).
fn scan(root: &Path) -> Result<Scan, Box<dyn std::error::Error>> {
    let mut fps = std::collections::BTreeMap::new();
    let mut unreadable = BTreeSet::new();
    for rel in discover_local_files(root)? {
        match fingerprint_path(&root.join(&rel)) {
            Ok(fp) => {
                fps.insert(rel, fp);
            }
            Err(_) => {
                unreadable.insert(rel);
            }
        }
    }
    Ok(Scan { fps, unreadable })
}

/// Compare two trees by content. Reads only.
pub fn compare(src: &Path, dest: &Path) -> Result<Report, Box<dyn std::error::Error>> {
    let source = scan(src)?;
    let destination = scan(dest)?;
    let mut report = Report::default();

    // Anything unreadable on EITHER side taints that path. We do not know what
    // is there, so we do not get to say.
    for path in source
        .unreadable
        .iter()
        .chain(destination.unreadable.iter())
    {
        if !report.unreadable.contains(path) {
            report.unreadable.push(path.clone());
        }
    }

    let all: BTreeSet<&PathBuf> = source.fps.keys().chain(destination.fps.keys()).collect();
    for path in all {
        if report.unreadable.contains(path) {
            continue;
        }
        match (source.fps.get(path), destination.fps.get(path)) {
            (Some(from), Some(to)) => {
                if from == to {
                    report.identical += 1;
                } else {
                    report.differs.push(path.clone());
                }
            }
            (Some(_), None) => report.missing.push(path.clone()),
            (None, Some(_)) => report.extra.push(path.clone()),
            // Cannot occur — `all` is built from the two maps' keys. Treated as
            // UNREADABLE rather than aborting: if the impossible happens in a
            // verifier that authorises deleting 755 GB, the safe answer is
            // "no evidence", not a panic and not silence.
            (None, None) => report.unreadable.push(path.clone()),
        }
    }
    report.differs.sort();
    report.missing.sort();
    report.extra.sort();
    report.unreadable.sort();
    Ok(report)
}

/// Run the command. Returns the process exit code.
pub fn verify(src: &Path, dest: &Path, quiet: bool) -> i32 {
    let report = match compare(src, dest) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("copia verify: cannot compare: {e}");
            return EXIT_ERROR;
        }
    };

    if !quiet {
        // One line per non-identical path, token first so it is greppable and
        // cut-able. Identical paths are not printed: on a 755 GB tree that is
        // noise that hides the four lines that matter.
        for (list, o) in [
            (&report.differs, Outcome::Differs),
            (&report.missing, Outcome::Missing),
            (&report.extra, Outcome::Extra),
            (&report.unreadable, Outcome::Unreadable),
        ] {
            for p in list {
                println!("{}\t{}", o.token(), p.display());
            }
        }
    }

    eprintln!(
        "copia verify: {} identical, {} differ, {} missing, {} extra, {} unreadable",
        report.identical,
        report.differs.len(),
        report.missing.len(),
        report.extra.len(),
        report.unreadable.len()
    );
    // State the verdict the caller actually needs, in words, rather than making
    // them re-derive it from five counts and an exit code. `safe_to_delete_source`
    // is the single predicate that authorises destroying data; it belongs on the
    // output, not only in a test.
    if report.safe_to_delete_source() {
        eprintln!("copia verify: trees are identical — the source may safely be deleted");
    } else {
        eprintln!("copia verify: DO NOT DELETE the source");
    }
    if report.exit_code() == EXIT_UNREADABLE {
        eprintln!(
            "copia verify: REFUSING to certify — {} path(s) could not be read. \
             This is not a difference; it is an absence of evidence, and deleting \
             a source on the strength of it is how data is lost.",
            report.unreadable.len()
        );
    }
    report.exit_code()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "copia-verify-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(root: &Path, rel: &str, body: &[u8]) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    #[test]
    fn identical_trees_are_safe_to_delete() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        for r in [&a, &b] {
            write(r, "x.txt", b"hello");
            write(r, "deep/y.bin", &[0u8; 4096]);
        }
        let rep = compare(&a, &b).unwrap();
        assert_eq!(rep.exit_code(), EXIT_IDENTICAL);
        assert!(rep.safe_to_delete_source());
        assert_eq!(rep.identical, 2);
        fs::remove_dir_all(&t).ok();
    }

    /// THE case a size+mtime check misses: same length, same mtime, different
    /// bytes. If this passes, the verifier is not comparing content.
    #[test]
    fn same_size_same_mtime_different_bytes_is_caught() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "f", b"AAAAAAAA");
        write(&b, "f", b"BBBBBBBB");
        // Force identical mtimes so ONLY content can distinguish the two files.
        // `touch -r` rather than a `filetime` dev-dependency: adding a crate to
        // a sovereign-stack tool for one test assertion is a poor trade, and
        // this is a test-only shell-out with a checked outcome.
        let _ = std::process::Command::new("touch")
            .arg("-r")
            .arg(a.join("f"))
            .arg(b.join("f"))
            .status();
        assert_eq!(
            fs::metadata(a.join("f")).unwrap().len(),
            fs::metadata(b.join("f")).unwrap().len(),
            "fixture must have equal sizes or it proves nothing"
        );
        // If `touch -r` was unavailable the mtimes may differ, and the test then
        // proves the weaker (still necessary) claim: equal size, different bytes
        // is caught. Say which was proved rather than implying the stronger one.
        let same_mtime = fs::metadata(a.join("f")).unwrap().modified().ok()
            == fs::metadata(b.join("f")).unwrap().modified().ok();
        if !same_mtime {
            eprintln!("note: mtimes differ; this run proves equal-size/different-bytes only");
        }

        let rep = compare(&a, &b).unwrap();
        assert_eq!(rep.exit_code(), EXIT_DIFFERS);
        assert!(!rep.safe_to_delete_source());
        assert_eq!(rep.differs.len(), 1);
        fs::remove_dir_all(&t).ok();
    }

    #[test]
    fn a_missing_file_is_not_identical() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "keep", b"1");
        write(&a, "lost", b"2");
        write(&b, "keep", b"1");
        let rep = compare(&a, &b).unwrap();
        assert_eq!(rep.exit_code(), EXIT_DIFFERS);
        assert_eq!(rep.missing, vec![PathBuf::from("lost")]);
        fs::remove_dir_all(&t).ok();
    }

    /// Guarantee 4. "I could not look" must not be reported as "I looked and
    /// found a difference", and must never be reported as agreement.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_file_refuses_certification_with_its_own_code() {
        use std::os::unix::fs::PermissionsExt;
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "secret", b"same");
        write(&b, "secret", b"same");
        // Unreadable on the DESTINATION side — the side about to justify a delete.
        fs::set_permissions(b.join("secret"), fs::Permissions::from_mode(0o000)).unwrap();

        let rep = compare(&a, &b).unwrap();
        // Running as root defeats chmod; skip rather than assert a false thing.
        if rep.unreadable.is_empty() {
            fs::set_permissions(b.join("secret"), fs::Permissions::from_mode(0o644)).ok();
            fs::remove_dir_all(&t).ok();
            return;
        }
        assert_eq!(
            rep.exit_code(),
            EXIT_UNREADABLE,
            "must not be 0 and must not be 1"
        );
        assert!(!rep.safe_to_delete_source());
        assert!(rep.differs.is_empty(), "unreadable is not a difference");
        assert_eq!(
            rep.identical, 0,
            "an unreadable path must not be counted identical"
        );

        fs::set_permissions(b.join("secret"), fs::Permissions::from_mode(0o644)).ok();
        fs::remove_dir_all(&t).ok();
    }

    /// Guarantee 2, asserted rather than asserted-about: fingerprint both trees
    /// before and after a run and require both unchanged. This catches a write
    /// the module does not know it makes, which a code review cannot.
    #[test]
    fn verify_read_only_never_writes() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "x", b"one");
        write(&a, "y/z", b"two");
        write(&b, "x", b"one");
        write(&b, "y/z", b"DIFFERENT");

        let before_a = scan(&a).unwrap().fps;
        let before_b = scan(&b).unwrap().fps;
        let code = verify(&a, &b, true);
        assert_eq!(code, EXIT_DIFFERS);
        let after_a = scan(&a).unwrap().fps;
        let after_b = scan(&b).unwrap().fps;

        assert_eq!(before_a, after_a, "verify modified the SOURCE tree");
        assert_eq!(before_b, after_b, "verify modified the DESTINATION tree");
        fs::remove_dir_all(&t).ok();
    }

    /// An empty comparison must not read as success. Two empty trees are
    /// "identical" in the trivial sense, and deleting a source on that basis is
    /// the vacuous-green shape: nothing was compared, so nothing was proved.
    #[test]
    fn two_empty_trees_are_not_safe_to_delete() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        let rep = compare(&a, &b).unwrap();
        assert_eq!(rep.identical, 0);
        assert!(
            !rep.safe_to_delete_source(),
            "a comparison that examined nothing must not authorise a deletion"
        );
        fs::remove_dir_all(&t).ok();
    }

    /// The non-quiet path prints one line per non-identical path. Untested
    /// until now, which meant the ONLY output an operator actually reads was
    /// the part with no test behind it.
    #[test]
    fn the_per_path_report_is_emitted_and_the_verdict_refuses() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "gone", b"1");
        write(&a, "changed", b"one");
        write(&b, "changed", b"two");
        write(&b, "unexpected", b"3");

        let rep = compare(&a, &b).unwrap();
        assert_eq!(rep.missing, vec![PathBuf::from("gone")]);
        assert_eq!(rep.differs, vec![PathBuf::from("changed")]);
        assert_eq!(rep.extra, vec![PathBuf::from("unexpected")]);
        assert!(!rep.safe_to_delete_source());

        // The printing path itself must not panic and must not change the code.
        assert_eq!(verify(&a, &b, false), EXIT_DIFFERS);
        fs::remove_dir_all(&t).ok();
    }

    /// A path present only at the DESTINATION is not harmless. It means the
    /// destination is not a faithful copy, and a caller about to delete the
    /// source is entitled to know before it does.
    #[test]
    fn an_extra_file_at_the_destination_refuses_the_delete() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "shared", b"x");
        write(&b, "shared", b"x");
        write(&b, "stowaway", b"y");
        let rep = compare(&a, &b).unwrap();
        assert_eq!(rep.exit_code(), EXIT_DIFFERS);
        assert_eq!(rep.extra, vec![PathBuf::from("stowaway")]);
        assert!(!rep.safe_to_delete_source());
        fs::remove_dir_all(&t).ok();
    }

    /// A source that cannot be walked is an ERROR, distinct from every verdict.
    /// Returning 1 here would tell a caller "I compared them and they differ",
    /// which is a claim about a comparison that never happened.
    #[test]
    fn an_unwalkable_source_is_exit_3_not_a_verdict() {
        let t = tmp();
        let missing = t.join("does-not-exist");
        let dest = t.join("dest");
        fs::create_dir_all(&dest).unwrap();
        assert_eq!(verify(&missing, &dest, true), EXIT_ERROR);
        fs::remove_dir_all(&t).ok();
    }

    /// A file replaced by a symlink whose target string happens to hash to the
    /// same bytes must still be a difference. Type is part of identity.
    #[cfg(unix)]
    #[test]
    fn a_file_to_symlink_flip_is_a_difference() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        write(&a, "thing", b"target.txt");
        write(&a, "target.txt", b"payload");
        write(&b, "target.txt", b"payload");
        std::os::unix::fs::symlink("target.txt", b.join("thing")).unwrap();

        let rep = compare(&a, &b).unwrap();
        assert_eq!(
            rep.differs,
            vec![PathBuf::from("thing")],
            "a regular file and a symlink are different entries even when the \
             hashed bytes coincide"
        );
        fs::remove_dir_all(&t).ok();
    }

    /// The pure decision, exercised over the whole shape space the Kani harness
    /// proves. Kani proves it holds; this pins the exact codes so a renumbering
    /// is a test failure and not a silent contract change.
    #[test]
    fn the_pure_decision_matches_the_published_codes() {
        assert_eq!(exit_code_for(0, 0, 0, 0), EXIT_IDENTICAL);
        assert_eq!(exit_code_for(1, 0, 0, 0), EXIT_DIFFERS);
        assert_eq!(exit_code_for(0, 1, 0, 0), EXIT_DIFFERS);
        assert_eq!(exit_code_for(0, 0, 1, 0), EXIT_DIFFERS);
        assert_eq!(exit_code_for(0, 0, 0, 1), EXIT_UNREADABLE);
        // unreadable outranks every other failure, together or alone
        assert_eq!(exit_code_for(9, 9, 9, 1), EXIT_UNREADABLE);
        assert!(safe_to_delete_for(1, 0, 0, 0, 0));
        assert!(!safe_to_delete_for(0, 0, 0, 0, 0), "examined nothing");
        assert!(
            !safe_to_delete_for(99, 0, 0, 0, 1),
            "one unreadable path is enough"
        );
    }

    /// A FIFO replaced by a regular file is a DIFFERENCE, not agreement.
    ///
    /// This is the integrity property the `Other` fingerprint kind exists for.
    /// Before it, neither entry was in the walk at all, so the comparison saw
    /// nothing on either side and reported identical — the shape that made
    /// `copia verify` certify a lost FIFO as safe to delete.
    #[cfg(unix)]
    #[test]
    fn an_entry_kind_swap_is_a_difference() {
        let t = tmp();
        let (a, b) = (t.join("a"), t.join("b"));
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(a.join("thing"))
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !made {
            fs::remove_dir_all(&t).ok();
            return;
        }
        // Same NAME, different KIND.
        write(&b, "thing", b"i am a regular file");

        let rep = compare(&a, &b).unwrap();
        assert_eq!(
            rep.differs,
            vec![PathBuf::from("thing")],
            "a FIFO and a regular file at the same path must not compare equal"
        );
        assert!(!rep.safe_to_delete_source());
        fs::remove_dir_all(&t).ok();
    }

    #[test]
    fn outcome_tokens_are_stable() {
        assert_eq!(Outcome::Identical.token(), "identical");
        assert_eq!(Outcome::Differs.token(), "differs");
        assert_eq!(Outcome::Missing.token(), "missing");
        assert_eq!(Outcome::Extra.token(), "extra");
        assert_eq!(Outcome::Unreadable.token(), "unreadable");
    }
}

// ── Kani: the decision that authorises destroying data ───────────────────
//
// These target `exit_code_for` / `safe_to_delete_for`, never `Report`. That is
// the allocation-free boundary: `Report` carries `Vec`s, and modelling the
// allocator to prove arithmetic over four counts buys nothing and costs hours.

/// Any count, bounded so the model stays small. The properties below do not
/// depend on magnitude — only on zero versus non-zero — so a bound of 3 covers
/// every distinct case rather than sampling a large space.
#[cfg(kani)]
fn any_count() -> usize {
    let n: usize = kani::any();
    kani::assume(n <= 3);
    n
}

/// A delete is NEVER authorised without positive evidence.
///
/// This is the safety property the whole command exists for. If it can be
/// violated, copia can tell a caller to delete 755 GB on the strength of a
/// comparison that found a difference, could not look, or examined nothing.
#[cfg(kani)]
#[kani::proof]
fn verify_never_authorises_a_delete_without_positive_evidence() {
    let (identical, differs, missing, extra, unreadable) = (
        any_count(),
        any_count(),
        any_count(),
        any_count(),
        any_count(),
    );
    if safe_to_delete_for(identical, differs, missing, extra, unreadable) {
        assert!(identical > 0, "authorised a delete having compared nothing");
        assert!(differs == 0, "authorised a delete with known differences");
        assert!(missing == 0, "authorised a delete with a missing path");
        assert!(extra == 0, "authorised a delete with an unexpected path");
        assert!(unreadable == 0, "authorised a delete over unreadable paths");
    }
}

/// `unreadable` outranks `differs`, always.
///
/// Both refuse the delete, so collapsing them is tempting. They must not
/// collapse: "I could not look" reported as "I looked and found a difference"
/// is a claim about evidence we do not have.
#[cfg(kani)]
#[kani::proof]
fn unreadable_is_never_reported_as_a_difference() {
    let (differs, missing, extra, unreadable) =
        (any_count(), any_count(), any_count(), any_count());
    let code = exit_code_for(differs, missing, extra, unreadable);
    if unreadable > 0 {
        assert!(code == EXIT_UNREADABLE);
    }
    assert!(code == EXIT_IDENTICAL || code == EXIT_DIFFERS || code == EXIT_UNREADABLE);
}
