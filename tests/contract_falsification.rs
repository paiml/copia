//! Falsification tests wiring the provable contracts under `contracts/` to real
//! executable checks. Each test mechanically falsifies one prediction from a
//! contract's `falsification_tests:` list.
//!
//! Scope: only the UNIT-TESTABLE predictions live here. SSH-only predictions
//! (recursive push over the wire, remote parse dispatch) are covered by
//! `tests/e2e_ssh.rs` and are intentionally NOT duplicated here.
//!
//! Contracts exercised:
//!   - path-safety-v1                  → FALSIFY-PATH-001/002/003
//!   - local-to-local-recursive-v1     → FALSIFY-L2L-001/002/003/004
//!   - streaming-transfer-v1           → FALSIFY-STREAM-001/003
//!   - single-file-sync-roundtrip-v1   → FALSIFY-SINGLE-001/003/004
//!   - recursive-push-v1               → FALSIFY-PUSH-003 (empty-tree short-circuit)
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreadable_literal,
    clippy::doc_markdown
)]

use std::path::{Path, PathBuf};
use std::process::Command;

// ── Helpers ───────────────────────────────────────────────────────────────

fn copia() -> Command {
    Command::new(env!("CARGO_BIN_EXE_copia"))
}

/// Fresh, empty temp dir unique to this process + tag.
fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "copia-falsify-{tag}-{}-{}",
        std::process::id(),
        tag
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Deterministic pseudo-random bytes (no dev-dependency needed).
fn prng_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            // xorshift64
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x & 0xff) as u8
        })
        .collect()
}

/// Recursively collect (relpath, bytes) for every file under `root`.
fn read_tree(root: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                let rel = path.strip_prefix(root).unwrap().to_path_buf();
                out.insert(rel, std::fs::read(&path).unwrap());
            }
        }
    }
    out
}

/// The EXACT escape production uses before wrapping a path in `$'...'`
/// (transfer.rs / dir_sync.rs / main.rs): double backslashes, then escape quotes.
fn ansi_c_escape(p: &str) -> String {
    p.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Resolve `$'<escape(p)>'` through a real bash to see which filename the remote
/// shell would actually name. `cwd` isolates any accidental side effects.
fn bash_resolves_to(p: &str, cwd: &Path) -> Vec<u8> {
    let snippet = format!("printf '%s' $'{}'", ansi_c_escape(p));
    let out = Command::new("bash")
        .arg("-c")
        .arg(&snippet)
        .current_dir(cwd)
        .output()
        .expect("bash must be available");
    assert!(
        out.status.success(),
        "bash failed for path {p:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

// ── path-safety-v1 ─────────────────────────────────────────────────────────

/// FALSIFY-PATH-001: the $'...'-quoted escape resolves back to the original for
/// a battery of special-character paths.
#[test]
fn falsify_path_001_escape_roundtrip() {
    let dir = tmp("path001");
    let cases = [
        "plain.txt",
        "with space.txt",
        "it's.txt",
        "a'b'c.txt",
        "back\\slash.txt",
        "quote'and\\back.txt",
        "weird $VAR name.txt",
        "angle<brackets>.txt",
        "tab\tinside.txt",
        "sub dir/deep's/file.bin",
    ];
    for p in cases {
        let resolved = bash_resolves_to(p, &dir);
        assert_eq!(
            resolved,
            p.as_bytes(),
            "FALSIFY-PATH-001: $'...' escape of {p:?} resolved to {:?}",
            String::from_utf8_lossy(&resolved)
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// FALSIFY-PATH-002: a single quote in the path does not close the ANSI-C literal.
#[test]
fn falsify_path_002_single_quote_containment() {
    let dir = tmp("path002");
    let p = "it's a file.txt";
    assert_eq!(
        bash_resolves_to(p, &dir),
        p.as_bytes(),
        "FALSIFY-PATH-002: apostrophe must not split the filename"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// FALSIFY-PATH-003: `$` / command substitution inside the path is literal — no
/// expansion runs, no side-effect file is created.
#[test]
fn falsify_path_003_no_injection() {
    let dir = tmp("path003");
    let p = "$(touch pwned).txt";
    let resolved = bash_resolves_to(p, &dir);
    assert_eq!(
        resolved,
        p.as_bytes(),
        "FALSIFY-PATH-003: command substitution must stay literal"
    );
    assert!(
        !dir.join("pwned").exists(),
        "FALSIFY-PATH-003: injected command executed — 'pwned' was created"
    );
    // Also prove escape() alone never emits a bare (unbackslashed) quote.
    let esc = ansi_c_escape("a'b");
    assert_eq!(esc, "a\\'b");
    std::fs::remove_dir_all(&dir).ok();
}

// ── local-to-local-recursive-v1  +  streaming-transfer-v1 ──────────────────

/// FALSIFY-L2L-001/002 and FALSIFY-STREAM-001/003: a nested tree containing a
/// multi-chunk, non-256KiB-aligned binary syncs local->local byte-for-byte,
/// with the destination path set exactly matching the source.
#[test]
fn falsify_l2l_001_002_stream_001_003_tree_byte_fidelity() {
    let base = tmp("l2l-tree");
    let src = base.join("src");
    let dst = base.join("dst");
    std::fs::create_dir_all(src.join("sub/deep")).unwrap();
    std::fs::write(src.join("a.txt"), b"alpha\n").unwrap();
    std::fs::write(src.join("sub/b.txt"), b"beta beta\n").unwrap();
    // > 256 KiB and NOT a multiple of the 256 KiB chunk → exercises chunk
    // boundaries + a short final read (streaming-transfer-v1).
    let big = prng_bytes(600 * 1024 + 123, 0x9E3779B97F4A7C15);
    std::fs::write(src.join("sub/deep/big.bin"), &big).unwrap();

    let status = copia()
        .args(["sync", "--recursive"])
        .arg(&src)
        .arg(&dst)
        .status()
        .unwrap();
    assert!(status.success(), "recursive local->local sync must succeed");

    let src_tree = read_tree(&src);
    let dst_tree = read_tree(&dst);

    // FALSIFY-L2L-002: identical relative path sets, no missing / no extra.
    assert_eq!(
        src_tree.keys().collect::<Vec<_>>(),
        dst_tree.keys().collect::<Vec<_>>(),
        "FALSIFY-L2L-002: destination path set must equal source path set"
    );
    // FALSIFY-L2L-001 + FALSIFY-STREAM-001/003: every file byte-for-byte equal.
    for (rel, bytes) in &src_tree {
        assert_eq!(
            dst_tree.get(rel).unwrap(),
            bytes,
            "FALSIFY-L2L-001: {rel:?} differs after recursive sync"
        );
    }
    // Explicit multi-chunk assertion for the streaming contract.
    assert_eq!(
        std::fs::read(dst.join("sub/deep/big.bin")).unwrap(),
        big,
        "FALSIFY-STREAM-001: multi-chunk non-aligned binary must reconstruct exactly"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-L2L-004: re-syncing over a pre-existing, divergent destination
/// reconstructs the source exactly (delta-against-basis leaves no stale bytes).
#[test]
fn falsify_l2l_004_overwrite_exactness() {
    let base = tmp("l2l-overwrite");
    let src = base.join("src");
    let dst = base.join("dst");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(&dst).unwrap();
    // Divergent, longer pre-existing destination content.
    std::fs::write(src.join("f.bin"), prng_bytes(40_000, 11)).unwrap();
    std::fs::write(dst.join("f.bin"), prng_bytes(90_000, 22)).unwrap();

    let status = copia()
        .args(["sync", "--recursive"])
        .arg(&src)
        .arg(&dst)
        .status()
        .unwrap();
    assert!(status.success());

    assert_eq!(
        std::fs::read(dst.join("f.bin")).unwrap(),
        std::fs::read(src.join("f.bin")).unwrap(),
        "FALSIFY-L2L-004: overwrite must reconstruct source exactly (no stale tail)"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-L2L-003 / FALSIFY-PUSH-003: an empty source tree yields no files and
/// exits 0 (the shared 'No files found.' short-circuit).
#[test]
fn falsify_l2l_003_push_003_empty_tree() {
    let base = tmp("empty-tree");
    let src = base.join("src");
    let dst = base.join("dst");
    std::fs::create_dir_all(&src).unwrap();

    let out = copia()
        .args(["sync", "--recursive"])
        .arg(&src)
        .arg(&dst)
        .output()
        .unwrap();
    assert!(out.status.success(), "empty-tree sync must exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("No files found"),
        "FALSIFY-L2L-003: empty tree must short-circuit with 'No files found.', got: {stderr}"
    );
    // No files should have been created under dst.
    if dst.exists() {
        assert!(
            read_tree(&dst).is_empty(),
            "FALSIFY-L2L-003: empty source must not create destination files"
        );
    }
    std::fs::remove_dir_all(&base).ok();
}

// ── single-file-sync-roundtrip-v1 ──────────────────────────────────────────

/// FALSIFY-SINGLE-001 / FALSIFY-SINGLE-003: a non-recursive local->local sync of
/// an absolute path round-trips byte-for-byte and never touches SSH (proving the
/// FileLocation::parse Local-guard chose the local branch, offline).
#[test]
fn falsify_single_001_003_local_roundtrip_offline() {
    let base = tmp("single-rt");
    let src = base.join("source.bin");
    let dst = base.join("dest.bin");
    let data = prng_bytes(20_000, 0xDEADBEEF);
    std::fs::write(&src, &data).unwrap();

    // Force PATH to a location without `ssh` so any accidental SSH dispatch fails
    // loudly instead of silently succeeding — this makes the "offline" claim real.
    let status = copia()
        .args(["sync"])
        .arg(&src)
        .arg(&dst)
        .env("PATH", base.as_os_str())
        .status()
        .unwrap();
    assert!(
        status.success(),
        "FALSIFY-SINGLE-003: absolute local paths must sync offline (no SSH)"
    );
    assert_eq!(
        std::fs::read(&dst).unwrap(),
        data,
        "FALSIFY-SINGLE-001: single-file sync must be byte-identical"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-SINGLE-004: remote->remote is rejected with an explicit error and a
/// non-zero exit — never silently mis-dispatched. Reaches the reject branch
/// before any SSH, so it is safe offline.
#[test]
fn falsify_single_004_remote_to_remote_rejected() {
    let out = copia()
        .args(["sync", "h1:/a/path", "h2:/b/path"])
        .env("PATH", "/nonexistent-copia-path")
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "FALSIFY-SINGLE-004: remote->remote must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not yet supported"),
        "FALSIFY-SINGLE-004: expected an explicit rejection, got: {stderr}"
    );
}

// ── incremental-sync-v1 (L1-L4: tests here are L2; L3 Kani plan-kani-001; L4 Lean) ──

fn seed(src: &std::path::Path) {
    std::fs::create_dir_all(src.join("d")).unwrap();
    std::fs::write(src.join("a.txt"), b"alpha").unwrap();
    std::fs::write(src.join("d/b.bin"), vec![7u8; 4096]).unwrap();
}
fn out_of(cmd: &mut Command) -> (bool, String) {
    let o = cmd.output().unwrap();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    (o.status.success(), combined)
}

/// FALSIFY-INCR-001: a 2nd sync of an unchanged tree transfers 0 files.
#[test]
fn falsify_incr_001_idempotent_second_run_skips_all() {
    let base = tmp("incr-idem");
    let (src, dst) = (base.join("src"), base.join("dst"));
    seed(&src);
    let (s1, _) = out_of(copia().args(["sync", "-r"]).arg(&src).arg(&dst));
    assert!(s1, "first sync must succeed");
    let (s2, log) = out_of(copia().args(["sync", "-r"]).arg(&src).arg(&dst));
    assert!(
        s2 && (log.contains("0 to transfer") || log.contains("up to date")),
        "FALSIFY-INCR-001: unchanged re-sync must skip all — got: {log}"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-INCR-003: atomic delivery — content identical, no `.copia-tmp` survives.
#[test]
fn falsify_incr_003_atomic_no_tmp_survives() {
    let base = tmp("incr-atomic");
    let (src, dst) = (base.join("src"), base.join("dst"));
    seed(&src);
    assert!(out_of(copia().args(["sync", "-r"]).arg(&src).arg(&dst)).0);
    assert_eq!(std::fs::read(dst.join("a.txt")).unwrap(), b"alpha");
    assert_eq!(std::fs::read(dst.join("d/b.bin")).unwrap(), vec![7u8; 4096]);
    let leftover = walk_find(&dst, "copia-tmp");
    assert!(
        leftover.is_none(),
        "FALSIFY-INCR-003: staging file survived: {leftover:?}"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-INCR-004: `--exclude '*.tmp'` keeps matching files out of the destination.
#[test]
fn falsify_incr_004_exclude_omits_matches() {
    let base = tmp("incr-excl");
    let (src, dst) = (base.join("src"), base.join("dst"));
    seed(&src);
    std::fs::write(src.join("junk.tmp"), b"junk").unwrap();
    assert!(
        out_of(
            copia()
                .args(["sync", "-r", "--exclude", "*.tmp"])
                .arg(&src)
                .arg(&dst)
        )
        .0
    );
    assert!(dst.join("a.txt").exists(), "wanted file must sync");
    assert!(
        !dst.join("junk.tmp").exists(),
        "FALSIFY-INCR-004: excluded file must NOT sync"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-INCR-005: `--delete` mirrors away a stale dest file; without it, it stays.
#[test]
fn falsify_incr_005_delete_is_mirror_and_opt_in() {
    let base = tmp("incr-del");
    let (src, dst) = (base.join("src"), base.join("dst"));
    seed(&src);
    assert!(out_of(copia().args(["sync", "-r"]).arg(&src).arg(&dst)).0);
    std::fs::remove_file(src.join("a.txt")).unwrap(); // a.txt now stale on dst
                                                      // no --delete: stale file is retained
    assert!(out_of(copia().args(["sync", "-r"]).arg(&src).arg(&dst)).0);
    assert!(
        dst.join("a.txt").exists(),
        "without --delete, stale file must remain"
    );
    // --delete: stale file is mirrored away
    assert!(out_of(copia().args(["sync", "-r", "--delete"]).arg(&src).arg(&dst)).0);
    assert!(
        !dst.join("a.txt").exists(),
        "FALSIFY-INCR-005: --delete must remove the stale file"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// FALSIFY-INCR-006: `--dry-run` mutates nothing.
#[test]
fn falsify_incr_006_dry_run_mutates_nothing() {
    let base = tmp("incr-dry");
    let (src, dst) = (base.join("src"), base.join("dst"));
    seed(&src);
    let (ok, log) = out_of(copia().args(["sync", "-r", "-n"]).arg(&src).arg(&dst));
    assert!(
        ok && log.contains("dry run"),
        "dry-run must report a plan: {log}"
    );
    assert!(
        !dst.exists() || !dst.join("a.txt").exists(),
        "FALSIFY-INCR-006: dry-run must not create destination files"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// Recursively find a file whose name contains `needle` (for temp-file leak checks).
fn walk_find(root: &std::path::Path, needle: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(needle))
        {
            return Some(p);
        }
        if p.is_dir() {
            if let Some(f) = walk_find(&p, needle) {
                return Some(f);
            }
        }
    }
    None
}

/// FALSIFY-INCR-006b: `--dry-run --delete` lists the delete plan but removes nothing.
#[test]
fn falsify_incr_006b_dry_run_delete_lists_but_keeps() {
    let base = tmp("incr-drydel");
    let (src, dst) = (base.join("src"), base.join("dst"));
    seed(&src);
    assert!(out_of(copia().args(["sync", "-r"]).arg(&src).arg(&dst)).0);
    std::fs::remove_file(src.join("a.txt")).unwrap();
    let (ok, log) = out_of(
        copia()
            .args(["sync", "-r", "-n", "--delete"])
            .arg(&src)
            .arg(&dst),
    );
    assert!(
        ok && log.contains("delete a.txt"),
        "dry-run --delete must LIST the delete: {log}"
    );
    assert!(
        dst.join("a.txt").exists(),
        "FALSIFY-INCR-006b: dry-run must not actually delete"
    );
    std::fs::remove_dir_all(&base).ok();
}
