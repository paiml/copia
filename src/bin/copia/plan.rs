//! Pure sync-planning — the rsync-style quick check, exclude globbing, and
//! delete-set computation. Deliberately free of I/O so it is exhaustively
//! unit-testable; `incremental.rs` supplies the metadata and executes the plan.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

/// Minimal per-file metadata driving the quick check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileMeta {
    pub size: u64,
    /// mtime as whole epoch seconds. Sub-second precision is unreliable across
    /// `find -printf %T@` vs `SystemTime` and across filesystems, so the quick
    /// check compares at 1-second granularity — exactly rsync's default posture.
    pub mtime: i64,
}

/// A tree of relative paths to their metadata (sorted, for deterministic plans).
pub type MetaMap = BTreeMap<PathBuf, FileMeta>;

/// The diff of source vs destination that drives one sync run.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SyncPlan {
    /// New or changed files to transfer (sorted).
    pub transfer: Vec<PathBuf>,
    /// Count of files skipped because size AND mtime matched the destination.
    pub skipped: usize,
    /// Destination files to remove (present on dest, absent from filtered
    /// source) — populated only when delete is requested. Sorted.
    pub delete: Vec<PathBuf>,
}

/// Quick check: a source file needs transfer when the destination lacks it, or
/// its size or mtime differ. Excluded source paths are dropped entirely — never
/// transferred, and (critically) never counted toward the delete set, so an
/// `--exclude`d path on the destination is left untouched rather than deleted.
pub fn build_plan(
    src: &MetaMap,
    dst: &MetaMap,
    excludes: &[String],
    with_delete: bool,
) -> SyncPlan {
    let mut plan = SyncPlan::default();
    for (path, smeta) in src {
        if is_excluded(path, excludes) {
            continue;
        }
        if needs_transfer(*smeta, dst.get(path).copied()) {
            plan.transfer.push(path.clone());
        } else {
            plan.skipped += 1;
        }
    }
    if with_delete {
        for path in dst.keys() {
            if !src.contains_key(path) && !is_excluded(path, excludes) {
                plan.delete.push(path.clone());
            }
        }
    }
    plan.transfer.sort();
    plan.delete.sort();
    plan
}

/// The per-file quick-check decision, factored out so it can be exhaustively
/// model-checked (Kani, `plan-kani-001`): a source file must be transferred iff
/// the destination lacks it, or its size or mtime differ. This is the single
/// invariant the whole incremental engine rests on.
#[must_use]
pub fn needs_transfer(src: FileMeta, dst: Option<FileMeta>) -> bool {
    dst.map_or(true, |d| src.size != d.size || src.mtime != d.mtime)
}

/// True if any exclude pattern matches the relative path. gitignore-style: a
/// pattern with no `/` matches any single path *component* (so `target` or
/// `*.tmp` prune anywhere in the tree, and a trailing slash like `target/` is
/// accepted as the same); a pattern containing `/` matches the whole relative
/// path as a glob.
pub fn is_excluded(rel: &Path, excludes: &[String]) -> bool {
    for pat in excludes {
        let pat = pat.trim_end_matches('/');
        if pat.is_empty() {
            continue;
        }
        if pat.contains('/') {
            if glob_match(pat, &rel.to_string_lossy()) {
                return true;
            }
        } else {
            for comp in rel.components() {
                if let Component::Normal(c) = comp {
                    if glob_match(pat, &c.to_string_lossy()) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Classic wildcard match with backtracking: `*` matches any run of characters,
/// `?` matches exactly one. O(n·m) worst case, which is fine for path-length
/// inputs. Slash-awareness is handled by the caller (per-component vs whole-path).
pub fn glob_match(pat: &str, text: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// L3 proof harnesses (Kani) for the pure quick-check decision. Run with
/// `cargo kani`. These bound the `incremental-sync-v1` proof obligations to
/// machine-checked proofs rather than tests alone.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// plan-kani-001: a file is transferred iff it is new (no dest) or differs
    /// in size or mtime; an identical file is NEVER re-transferred. Exhaustive
    /// over all (size, mtime) pairs — the core skip guarantee.
    #[kani::proof]
    fn needs_transfer_iff_new_or_differing() {
        let s = FileMeta {
            size: kani::any(),
            mtime: kani::any(),
        };
        // New destination (absent) always transfers.
        assert!(needs_transfer(s, None));
        // An identical destination is never re-transferred (the skip guarantee).
        assert!(!needs_transfer(s, Some(s)));
        // A present destination transfers iff size or mtime differ.
        let d = FileMeta {
            size: kani::any(),
            mtime: kani::any(),
        };
        assert_eq!(
            needs_transfer(s, Some(d)),
            s.size != d.size || s.mtime != d.mtime
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn m(size: u64, mtime: i64) -> FileMeta {
        FileMeta { size, mtime }
    }

    #[test]
    fn needs_transfer_matches_the_quick_check_rule() {
        assert!(needs_transfer(m(1, 1), None)); // new
        assert!(!needs_transfer(m(1, 1), Some(m(1, 1)))); // identical -> skip
        assert!(needs_transfer(m(2, 1), Some(m(1, 1)))); // size differs
        assert!(needs_transfer(m(1, 2), Some(m(1, 1)))); // mtime differs
    }
    fn map(entries: &[(&str, u64, i64)]) -> MetaMap {
        entries
            .iter()
            .map(|(p, s, t)| (PathBuf::from(p), m(*s, *t)))
            .collect()
    }

    #[test]
    fn glob_star_question_and_literal() {
        assert!(glob_match("*.tmp", "foo.tmp"));
        assert!(glob_match("*.tmp", ".tmp"));
        assert!(!glob_match("*.tmp", "foo.txt"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
        assert!(glob_match("target", "target"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a*b*c", "axxbyyc"));
        assert!(!glob_match("a*b*c", "axxbyy"));
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
    }

    #[test]
    fn exclude_matches_any_component_when_slashless() {
        let ex = vec!["target".to_string(), "*.tmp".to_string()];
        assert!(is_excluded(Path::new("target/debug/app"), &ex));
        assert!(is_excluded(Path::new("crate/target/x"), &ex));
        assert!(is_excluded(Path::new("build/out.tmp"), &ex));
        assert!(!is_excluded(Path::new("src/main.rs"), &ex));
    }

    #[test]
    fn exclude_trailing_slash_and_full_path_pattern() {
        assert!(is_excluded(
            Path::new("node_modules/x/y"),
            &["node_modules/".to_string()]
        ));
        // slash-bearing pattern matches the whole relative path as a glob
        assert!(is_excluded(
            Path::new("a/b/c.log"),
            &["a/*/c.log".to_string()]
        ));
        assert!(!is_excluded(
            Path::new("a/b/c.log"),
            &["a/c.log".to_string()]
        ));
        // empty patterns are ignored
        assert!(!is_excluded(
            Path::new("x"),
            &[String::new(), "/".to_string()]
        ));
    }

    #[test]
    fn quick_check_skips_identical_transfers_changed_and_new() {
        let src = map(&[
            ("same", 10, 100),
            ("changed_size", 20, 100),
            ("changed_mtime", 10, 200),
            ("new", 5, 100),
        ]);
        let dst = map(&[
            ("same", 10, 100),
            ("changed_size", 10, 100),
            ("changed_mtime", 10, 100),
        ]);
        let plan = build_plan(&src, &dst, &[], false);
        assert_eq!(plan.skipped, 1); // "same"
        assert_eq!(
            plan.transfer,
            vec![
                PathBuf::from("changed_mtime"),
                PathBuf::from("changed_size"),
                PathBuf::from("new")
            ]
        );
        assert!(plan.delete.is_empty()); // delete not requested
    }

    #[test]
    fn delete_set_is_dest_minus_source_and_honors_excludes() {
        let src = map(&[("keep", 1, 1)]);
        let dst = map(&[("keep", 1, 1), ("stale", 1, 1), ("target", 1, 1)]);
        // without delete: nothing removed
        assert!(build_plan(&src, &dst, &[], false).delete.is_empty());
        // with delete: "stale" removed, but "target" is excluded => NOT deleted
        let plan = build_plan(&src, &dst, &["target".to_string()], true);
        assert_eq!(plan.delete, vec![PathBuf::from("stale")]);
        assert_eq!(plan.skipped, 1); // "keep"
    }

    #[test]
    fn excluded_source_file_is_never_transferred() {
        let src = map(&[("keep.rs", 1, 1), ("junk.tmp", 1, 1)]);
        let plan = build_plan(&src, &MetaMap::new(), &["*.tmp".to_string()], false);
        assert_eq!(plan.transfer, vec![PathBuf::from("keep.rs")]);
    }
}
