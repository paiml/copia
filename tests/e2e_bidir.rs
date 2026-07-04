//! End-to-end tests for `copia bisync` — the L2 bidirectional reconciler.
//! FALSIFY-BIDIR-001/002 are covered by unit + Kani + Lean (reconcile.rs); these
//! drive the CLI binary for the observable behaviours (propagation, convergent
//! conflict-copy, safe delete). Each test uses an isolated HOME so per-pair
//! archives don't collide.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::similar_names
)]
use std::path::{Path, PathBuf};
use std::process::Command;

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("copia-bidir-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn bisync(home: &Path, a: &Path, b: &Path, extra: &[&str]) -> (bool, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_copia"));
    c.env("HOME", home).arg("bisync");
    for e in extra {
        c.arg(e);
    }
    c.arg(a).arg(b);
    let o = c.output().unwrap();
    (
        o.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
    )
}

fn seed_pair(a: &Path, b: &Path) {
    std::fs::create_dir_all(a).unwrap();
    std::fs::create_dir_all(b).unwrap();
    std::fs::write(a.join("f"), b"seed").unwrap();
    std::fs::write(b.join("f"), b"seed").unwrap();
}

/// FALSIFY-BIDIR-003: a change on A reaches B, and a change on B reaches A.
#[test]
fn falsify_bidir_003_two_way_propagation() {
    let base = tmp("prop");
    let (home, a, b) = (base.join("home"), base.join("A"), base.join("B"));
    seed_pair(&a, &b);
    bisync(&home, &a, &b, &[]); // seed archive
    std::fs::write(a.join("f"), b"fromA").unwrap();
    assert!(bisync(&home, &a, &b, &[]).0);
    assert_eq!(
        std::fs::read(b.join("f")).unwrap(),
        b"fromA",
        "A->B propagation"
    );
    std::fs::write(b.join("f"), b"fromB").unwrap();
    assert!(bisync(&home, &a, &b, &[]).0);
    assert_eq!(
        std::fs::read(a.join("f")).unwrap(),
        b"fromB",
        "B->A propagation"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-BIDIR-004: divergent edits converge to an identical set, both preserved.
#[test]
fn falsify_bidir_004_conflict_converges_no_data_loss() {
    let base = tmp("conf");
    let (home, a, b) = (base.join("home"), base.join("A"), base.join("B"));
    seed_pair(&a, &b);
    bisync(&home, &a, &b, &[]);
    std::fs::write(a.join("f"), b"AAA").unwrap();
    std::fs::write(b.join("f"), b"BBB").unwrap();
    let (_ok, log) = bisync(&home, &a, &b, &["-v"]); // non-zero exit on conflict is expected
    assert!(log.contains("conflict"), "must report a conflict: {log}");
    let mut fa: Vec<_> = std::fs::read_dir(&a)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    let mut fb: Vec<_> = std::fs::read_dir(&b)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    fa.sort();
    fb.sort();
    assert_eq!(
        fa, fb,
        "FALSIFY-BIDIR-004: replicas must converge to an identical file set"
    );
    let all: String = fa
        .iter()
        .map(|n| String::from_utf8_lossy(&std::fs::read(a.join(n)).unwrap()).into_owned())
        .collect();
    assert!(
        all.contains("AAA") && all.contains("BBB"),
        "both versions must survive: {all}"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-BIDIR-005: create propagates; delete of a COMMON file mirrors safely;
/// and a dry-run mutates nothing.
#[test]
fn falsify_bidir_005_safe_delete_create_and_dry_run() {
    let base = tmp("del");
    let (home, a, b) = (base.join("home"), base.join("A"), base.join("B"));
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    std::fs::write(a.join("keep"), b"k").unwrap();
    std::fs::write(b.join("keep"), b"k").unwrap();
    bisync(&home, &a, &b, &[]);
    // dry-run of a new file changes nothing
    std::fs::write(a.join("g"), b"new").unwrap();
    let (_ok, log) = bisync(&home, &a, &b, &["-n"]);
    assert!(log.contains("dry run"), "dry-run must report: {log}");
    assert!(!b.join("g").exists(), "dry-run must not propagate");
    // real run propagates
    assert!(bisync(&home, &a, &b, &[]).0);
    assert!(b.join("g").exists(), "create must propagate");
    // g is common now; delete on A mirrors to B (positive-evidence delete)
    std::fs::remove_file(a.join("g")).unwrap();
    assert!(bisync(&home, &a, &b, &[]).0);
    assert!(
        !b.join("g").exists(),
        "FALSIFY-BIDIR-005: delete of a common file must mirror"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// Modify-vs-delete is a conflict that KEEPS the modification (never data loss).
#[test]
fn bidir_modify_vs_delete_keeps_modification() {
    let base = tmp("modvsdel");
    let (home, a, b) = (base.join("home"), base.join("A"), base.join("B"));
    seed_pair(&a, &b);
    bisync(&home, &a, &b, &[]); // f is common
    std::fs::remove_file(a.join("f")).unwrap(); // deleted on A
    std::fs::write(b.join("f"), b"edited").unwrap(); // modified on B
    bisync(&home, &a, &b, &[]);
    // modification survives on both sides
    assert_eq!(
        std::fs::read(a.join("f")).unwrap(),
        b"edited",
        "modification restored to A"
    );
    assert_eq!(
        std::fs::read(b.join("f")).unwrap(),
        b"edited",
        "modification kept on B"
    );
    std::fs::remove_dir_all(&base).ok();
}
