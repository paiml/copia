//! Differential harness: copia against rsync, on the shapes that matter.
//!
//! copia claims to replace rsync for archival — move a tree, prove it landed,
//! delete the original. A claim that large is not settled by copia agreeing with
//! itself. So every case here runs BOTH tools over the same fixture and requires
//! the results to agree, with rsync as the oracle:
//!
//! * `rsync -a src/ dst/` vs `copia sync -r src dst` — the destinations must be
//!   indistinguishable in structure, content, symlink-ness, and link targets.
//! * `rsync -a --checksum --dry-run --itemize-changes` vs `copia verify` — the
//!   two must agree on whether the trees match, because that verdict is what
//!   authorises deleting 755 GB.
//!
//! # Why rsync must be PRESENT, not skipped around
//!
//! An absent oracle makes every case here vacuously green. That failure has
//! already been paid for downstream: forjar's `nas_archive` suite reported PASS
//! on seven tests while none of them ran, because rsync was missing and each
//! test skipped with an `eprintln!` that nextest hides for passing tests. So
//! `the_oracle_is_present` FAILS rather than skips, and CI must install rsync.
//!
//! # Shapes chosen because they have broken something
//!
//! Symlink-to-directory is first for a reason: `copia sync -r` silently DROPPED
//! it (the walk excluded it from directories and it was not a file), and because
//! `copia verify` shared that walk, the path was absent from both scans and the
//! trees compared IDENTICAL. copia reported success over data it had just lost.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::similar_names
)]
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn copia_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_copia"))
}

fn have_rsync() -> bool {
    Command::new("rsync")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "copia-diff-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&p).expect("mkdir");
    p
}

fn write(root: &Path, rel: &str, body: &[u8]) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("mkdir -p");
    }
    fs::write(p, body).expect("write");
}

/// A comparable description of one tree: every entry, its kind, and either its
/// content hash or its link target. Deliberately NOT a byte-for-byte metadata
/// dump — mtimes and owners are a separate concern and would make every case
/// fail for reasons unrelated to what it is testing.
fn describe(root: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        let mut items: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        items.sort();
        for p in items {
            let rel = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string();
            let Ok(md) = fs::symlink_metadata(&p) else {
                continue;
            };
            if md.file_type().is_symlink() {
                let target = fs::read_link(&p).unwrap_or_default();
                out.push(format!("symlink\t{rel}\t{}", target.display()));
            } else if md.is_dir() {
                out.push(format!("dir\t{rel}"));
                walk(root, &p, out);
            } else {
                let body = fs::read(&p).unwrap_or_default();
                out.push(format!("file\t{rel}\t{}", blake3::hash(&body).to_hex()));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out
}

fn rsync_to(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("mkdir dst");
    let out = Command::new("rsync")
        .arg("-a")
        .arg(format!("{}/", src.display()))
        .arg(format!("{}/", dst.display()))
        .output()
        .expect("spawn rsync");
    assert!(
        out.status.success(),
        "rsync failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn copia_to(src: &Path, dst: &Path) {
    let out = Command::new(copia_bin())
        .args(["sync", "-r"])
        .arg(src)
        .arg(dst)
        .output()
        .expect("spawn copia");
    assert!(
        out.status.success(),
        "copia sync failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// rsync's verdict on whether two trees already match, by CONTENT.
/// Empty itemised output means nothing would change — the trees agree.
fn rsync_says_identical(src: &Path, dst: &Path) -> bool {
    let out = Command::new("rsync")
        .args(["-a", "--checksum", "--dry-run", "--itemize-changes"])
        .arg(format!("{}/", src.display()))
        .arg(format!("{}/", dst.display()))
        .output()
        .expect("spawn rsync");
    String::from_utf8_lossy(&out.stdout).trim().is_empty()
}

fn copia_says_identical(src: &Path, dst: &Path) -> bool {
    let out = Command::new(copia_bin())
        .args(["verify", "-q"])
        .arg(src)
        .arg(dst)
        .output()
        .expect("spawn copia verify");
    out.status.code() == Some(0)
}

/// Build one fixture shape into `root`.
fn build_fixture(shape: &str, root: &Path) {
    match shape {
        "plain-files" => {
            write(root, "a.txt", b"alpha");
            write(root, "b.bin", &[7u8; 10_000]);
        }
        "nested-dirs" => {
            write(root, "one/two/three/deep.txt", b"deep");
            write(root, "one/sibling.txt", b"sib");
        }
        "symlink-to-file" => {
            write(root, "real.txt", b"real");
            #[cfg(unix)]
            std::os::unix::fs::symlink("real.txt", root.join("link.txt")).expect("symlink");
        }
        // FIRST because it is the one that silently lost data.
        "symlink-to-dir" => {
            write(root, "realdir/inside.txt", b"inside");
            #[cfg(unix)]
            std::os::unix::fs::symlink("realdir", root.join("dirlink")).expect("symlink");
        }
        "dangling-symlink" => {
            #[cfg(unix)]
            std::os::unix::fs::symlink("nowhere.txt", root.join("dangling")).expect("symlink");
            write(root, "present.txt", b"present");
        }
        "empty-dir" => {
            fs::create_dir_all(root.join("hollow")).expect("mkdir");
            write(root, "beside.txt", b"beside");
        }
        "awkward-names" => {
            write(root, "with space.txt", b"space");
            write(root, "unicode-\u{00e9}\u{4e2d}.txt", b"unicode");
            write(root, "dash-leading/-weird.txt", b"dash");
        }
        "large-file" => {
            write(root, "big.bin", &vec![3u8; 3 * 1024 * 1024]);
        }
        other => panic!("unknown fixture shape: {other}"),
    }
}

const SHAPES: &[&str] = &[
    "symlink-to-dir",
    "symlink-to-file",
    "dangling-symlink",
    "plain-files",
    "nested-dirs",
    "empty-dir",
    "awkward-names",
    "large-file",
];

/// The oracle must exist. An absent rsync makes every case below vacuous, and
/// this suite would report PASS while proving nothing — the exact failure that
/// has already happened in this fleet.
#[test]
fn the_oracle_is_present() {
    assert!(
        have_rsync(),
        "rsync is not installed, so every differential case in this file would \
         compare copia against nothing and pass. Install rsync in CI; do not \
         convert these into skips."
    );
}

/// copia sync must produce what rsync -a produces.
#[test]
fn copia_sync_matches_rsync_on_every_shape() {
    if !have_rsync() {
        return; // the_oracle_is_present already fails; do not double-report
    }
    let mut failures = Vec::new();
    for shape in SHAPES {
        let base = tmp(shape);
        let (src, r, c) = (base.join("src"), base.join("rsync"), base.join("copia"));
        fs::create_dir_all(&src).expect("mkdir src");
        build_fixture(shape, &src);

        rsync_to(&src, &r);
        copia_to(&src, &c);

        let (dr, dc) = (describe(&r), describe(&c));
        if dr != dc {
            failures.push(format!(
                "shape `{shape}`:\n     rsync -> {dr:?}\n     copia -> {dc:?}"
            ));
        }
        fs::remove_dir_all(&base).ok();
    }
    assert!(
        failures.is_empty(),
        "copia sync diverged from rsync -a on {} shape(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// copia verify must agree with rsync's own content comparison — in BOTH
/// directions. Agreeing only when trees match would be satisfied by a verifier
/// that always says "identical".
#[test]
fn copia_verify_agrees_with_rsync_checksum_both_ways() {
    if !have_rsync() {
        return;
    }
    let mut failures = Vec::new();
    for shape in SHAPES {
        let base = tmp(&format!("v-{shape}"));
        let (src, dst) = (base.join("src"), base.join("dst"));
        fs::create_dir_all(&src).expect("mkdir src");
        build_fixture(shape, &src);
        rsync_to(&src, &dst);

        // 1. After a faithful copy, both must say identical.
        let (r_same, c_same) = (
            rsync_says_identical(&src, &dst),
            copia_says_identical(&src, &dst),
        );
        if !(r_same && c_same) {
            failures.push(format!(
                "shape `{shape}` after rsync -a: rsync_identical={r_same} copia_identical={c_same}"
            ));
        }

        // 2. Perturb one byte WITHOUT changing size, and both must say differ.
        //    Equal size is the point: a size+mtime check would miss it.
        let victim = fs::read_dir(&dst)
            .expect("read dst")
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                fs::symlink_metadata(p)
                    .map(|m| m.is_file())
                    .unwrap_or(false)
                    && fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false)
            });
        if let Some(v) = victim {
            let mut body = fs::read(&v).expect("read victim");
            let before = body.len();
            body[0] ^= 0xFF;
            fs::write(&v, &body).expect("write victim");
            assert_eq!(
                before as u64,
                fs::metadata(&v).expect("stat").len(),
                "the edit must preserve size or it proves nothing about content comparison"
            );

            let (r_diff, c_diff) = (
                !rsync_says_identical(&src, &dst),
                !copia_says_identical(&src, &dst),
            );
            if !(r_diff && c_diff) {
                failures.push(format!(
                    "shape `{shape}` after a 1-byte same-size edit: rsync_differs={r_diff} \
                     copia_differs={c_diff} — a verifier that misses this authorises \
                     deleting a corrupted source"
                ));
            }
        }
        fs::remove_dir_all(&base).ok();
    }
    assert!(
        failures.is_empty(),
        "copia verify disagreed with rsync on {} case(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Falsification: the harness must be able to SEE a divergence. If `describe`
/// cannot distinguish a symlink from a regular file, `copia_sync_matches_rsync`
/// passes over the exact defect this file was written for.
#[test]
fn the_harness_can_see_a_symlink_divergence() {
    let base = tmp("falsify");
    let (a, b) = (base.join("a"), base.join("b"));
    write(&a, "real.txt", b"same");
    write(&b, "real.txt", b"same");
    #[cfg(unix)]
    std::os::unix::fs::symlink("real.txt", a.join("link")).expect("symlink");
    // b gets a COPY where a has a link — byte-identical content, different tree.
    write(&b, "link", b"same");

    let (da, db) = (describe(&a), describe(&b));
    assert_ne!(
        da, db,
        "describe() cannot tell a symlink from a regular file with the same \
         content, so every comparison in this file is blind to the defect it exists to catch"
    );
    fs::remove_dir_all(&base).ok();
}
