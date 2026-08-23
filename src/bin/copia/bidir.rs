//! L2 bidirectional sync (local<->local): scan + content-fingerprint both trees,
//! load the per-pair archive (or degrade to safe no-base mode), reconcile on
//! blake3 (reconcile.rs — Kani-proved), apply with atomic delivery + convergent
//! conflict-copy, then record the new common state (commit-then-record).
//!
//! Convergent conflict-copy: on divergence NO semantic winner is chosen — BOTH
//! versions survive on BOTH sides. The real path deterministically takes the
//! max-(blake3) version (clock-independent, identical on both ends); the loser
//! lands at `<path>.conflict-<host>-<short_blake3>`. So both replicas end up with
//! an identical file set and the pair converges without ever losing data.

use super::archive::{archive_path, root_pair_hash, Archive};
use super::meta::discover_local_fingerprints;
use super::reconcile::{reconcile, Action, ConflictKind, Fingerprint, FpMap};
use std::path::{Path, PathBuf};

pub struct BidirOptions {
    pub dry_run: bool,
    pub verbose: bool,
}

fn host_id() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "host".to_string())
}

fn short_hex(h: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(12);
    for b in &h[..6] {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Atomic local copy: temp sibling + rename (never a torn destination).
fn copy_atomic(src: &Path, dst: &Path) -> std::io::Result<()> {
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut tmp = dst.as_os_str().to_owned();
    tmp.push(".copia-tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::copy(src, &tmp)?;
    std::fs::rename(&tmp, dst)
}

pub fn run_bisync(
    root_a: &Path,
    root_b: &Path,
    opts: &BidirOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let a = discover_local_fingerprints(root_a)?;
    let b = discover_local_fingerprints(root_b)?;
    let pair = root_pair_hash(root_a, root_b);
    let apath = archive_path(&pair);
    let loaded = Archive::load(&apath, &pair);
    let trust_base = loaded.is_some();
    let base: FpMap = loaded
        .as_ref()
        .map_or_else(FpMap::new, |z| z.entries.clone());
    if !trust_base {
        eprintln!("No trusted archive for this pair — SAFE no-base mode (no deletes; divergence kept as conflicts).");
    }

    let plan = reconcile(&a, &b, &base, trust_base);
    let conflicts = plan
        .iter()
        .filter(|(_, act)| matches!(act, Action::Conflict(_)))
        .count();
    eprintln!(
        "Bidirectional plan: {} action(s), {} conflict(s){}",
        plan.len(),
        conflicts,
        if trust_base { "" } else { " [safe no-base]" }
    );

    if opts.dry_run {
        for (p, act) in &plan {
            println!("{:<22} {}", format!("{act:?}"), p.display());
        }
        println!("(dry run) nothing was modified");
        return Ok(());
    }

    let host = host_id();
    // Start from the trusted base and mutate to the new common state as we apply.
    let mut common = base;
    let mut conflict_paths: Vec<PathBuf> = Vec::new();
    for (path, act) in &plan {
        apply(
            root_a,
            root_b,
            path,
            *act,
            &a,
            &b,
            &host,
            &mut common,
            &mut conflict_paths,
        )?;
    }

    // Commit-then-record: data is on disk; now persist the new common archive.
    let mut arc = loaded.unwrap_or_else(|| Archive::fresh(pair.clone(), host.clone()));
    arc.entries = common;
    arc.epoch += 1;
    arc.host_id = host;
    arc.save(&apath)?;

    println!(
        "Bidirectional sync complete: {} applied, {} conflict(s) preserved.",
        plan.len(),
        conflict_paths.len()
    );
    if opts.verbose && !conflict_paths.is_empty() {
        for p in &conflict_paths {
            eprintln!("  conflict: {}", p.display());
        }
    }
    if conflict_paths.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} path(s) had conflicts (both versions preserved)",
            conflict_paths.len()
        )
        .into())
    }
}

/// Record `rel`'s fingerprint, as seen on `side`, into the new common base.
///
/// A no-op when `side` does not hold `rel` — the caller's base entry (if any) is
/// then left exactly as it was, which is what the inlined `if let` did.
fn record_common(common: &mut FpMap, side: &FpMap, rel: &Path) {
    if let Some(fp) = side.get(rel) {
        common.insert(rel.to_path_buf(), *fp);
    }
}

/// Delete-vs-modify: keep the modification by restoring the surviving side onto
/// the deleted one.
///
/// A is probed FIRST and `b` is consulted only when A does not hold `rel`; that
/// order is the arm's own, and it is load-bearing — with both sides present the
/// A->B copy is the one that must run.
fn apply_delete_vs_modify(
    pa: &Path,
    pb: &Path,
    rel: &Path,
    a: &FpMap,
    b: &FpMap,
    common: &mut FpMap,
) -> std::io::Result<()> {
    if a.contains_key(rel) {
        copy_atomic(pa, pb)?;
        record_common(common, a, rel);
    } else if b.contains_key(rel) {
        copy_atomic(pb, pa)?;
        record_common(common, b, rel);
    }
    Ok(())
}

/// Deterministic, clock-independent winner: max blake3, so both ends pick the
/// same side without talking. Returns `(win_root, win_fp, lose_root, lose_fp)`.
///
/// Ties (`fa == fb`) go to A, matching the `>=` this was lifted from.
fn winner_loser<'r, 'f>(
    root_a: &'r Path,
    fa: &'f Fingerprint,
    root_b: &'r Path,
    fb: &'f Fingerprint,
) -> (&'r Path, &'f Fingerprint, &'r Path, &'f Fingerprint) {
    if fa.blake3 >= fb.blake3 {
        (root_a, fa, root_b, fb)
    } else {
        (root_b, fb, root_a, fa)
    }
}

/// Convergent conflict-copy: winner takes `rel`; the loser is preserved as
/// `.conflict-<host>-<short_blake3>` on BOTH sides so the replicas end up with
/// an identical file set and no version is lost.
#[allow(clippy::too_many_arguments)]
fn apply_both_changed(
    root_a: &Path,
    root_b: &Path,
    rel: &Path,
    a: &FpMap,
    b: &FpMap,
    host: &str,
    common: &mut FpMap,
    conflicts: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    let (Some(fa), Some(fb)) = (a.get(rel), b.get(rel)) else {
        return Ok(());
    };
    let (win_root, win_fp, lose_root, lose_fp) = winner_loser(root_a, fa, root_b, fb);
    let loser_name = {
        let mut n = rel.as_os_str().to_owned();
        n.push(format!(".conflict-{host}-{}", short_hex(&lose_fp.blake3)));
        PathBuf::from(n)
    };
    let win_full = win_root.join(rel); // winner content
    let lose_full = lose_root.join(rel); // loser content (about to be overwritten)
                                         // 1. Preserve the loser as a conflict-copy on BOTH sides FIRST.
    copy_atomic(&lose_full, &lose_root.join(&loser_name))?;
    copy_atomic(&lose_full, &win_root.join(&loser_name))?;
    // 2. Put the winner's content on both real paths (winner side already has it).
    copy_atomic(&win_full, &lose_full)?;
    // Both replicas now hold {rel = winner, loser_name = loser} — identical.
    common.insert(rel.to_path_buf(), *win_fp);
    common.insert(loser_name, *lose_fp);
    conflicts.push(rel.to_path_buf());
    Ok(())
}

/// Apply one reconciler decision. Every arm is dispatched in the same order and
/// with the same short-circuiting as before; the per-arm bodies live in the
/// helpers above so the safety-critical ordering inside each is stated once.
#[allow(clippy::too_many_arguments)]
fn apply(
    root_a: &Path,
    root_b: &Path,
    rel: &Path,
    act: Action,
    a: &FpMap,
    b: &FpMap,
    host: &str,
    common: &mut FpMap,
    conflicts: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    let pa = root_a.join(rel);
    let pb = root_b.join(rel);
    match act {
        Action::Noop => {}
        Action::ConvergeIdentical => {
            // Both sides already hold identical new content — no file op, just
            // record it as the new common base.
            record_common(common, a, rel);
        }
        Action::PropagateAtoB => {
            copy_atomic(&pa, &pb)?;
            record_common(common, a, rel);
        }
        Action::PropagateBtoA => {
            copy_atomic(&pb, &pa)?;
            record_common(common, b, rel);
        }
        Action::DeleteA => {
            let _ = std::fs::remove_file(&pa);
            common.remove(rel);
        }
        Action::DeleteB => {
            let _ = std::fs::remove_file(&pb);
            common.remove(rel);
        }
        Action::Conflict(ConflictKind::DeleteVsModify) => {
            apply_delete_vs_modify(&pa, &pb, rel, a, b, common)?;
        }
        Action::Conflict(ConflictKind::BothChanged) => {
            apply_both_changed(root_a, root_b, rel, a, b, host, common, conflicts)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::reconcile::FileType;
    use super::*;

    #[test]
    fn short_hex_is_12_hex_chars() {
        assert_eq!(short_hex(&[0xab; 32]), "abababababab");
    }

    fn fp(first: u8) -> Fingerprint {
        let mut h = [0u8; 32];
        h[0] = first;
        Fingerprint {
            blake3: h,
            ftype: FileType::File,
        }
    }

    /// The conflict winner is max-blake3 and a TIE goes to A — this is what makes
    /// both replicas pick the same side without talking. Extracting the `>=` into
    /// `winner_loser` must not have flipped either case.
    #[test]
    fn winner_is_max_blake3_and_ties_go_to_a() {
        let (ra, rb) = (Path::new("/a"), Path::new("/b"));
        let (lo, hi) = (fp(1), fp(2));

        let (wr, wf, lr, lf) = winner_loser(ra, &hi, rb, &lo);
        assert_eq!((wr, lr), (ra, rb), "A holds the larger hash, so A wins");
        assert_eq!((*wf, *lf), (hi, lo));

        let (wr, wf, lr, lf) = winner_loser(ra, &lo, rb, &hi);
        assert_eq!((wr, lr), (rb, ra), "B holds the larger hash, so B wins");
        assert_eq!((*wf, *lf), (hi, lo));

        let (wr, _, lr, _) = winner_loser(ra, &hi, rb, &hi);
        assert_eq!((wr, lr), (ra, rb), "equal hashes must tie-break to A (>=)");
    }

    /// `record_common` replaced an inlined `if let`: when the side does not hold
    /// the path it must do NOTHING — in particular it must not remove or clobber
    /// an entry already in the base.
    #[test]
    fn record_common_is_a_noop_when_the_side_lacks_the_path() {
        let rel = Path::new("f");
        let mut side = FpMap::new();
        side.insert(rel.to_path_buf(), fp(7));

        let mut common = FpMap::new();
        record_common(&mut common, &side, rel);
        assert_eq!(common.get(rel), Some(&fp(7)), "present -> inserted");

        // Absent from `side`, but already present in `common`: left untouched.
        let mut common = FpMap::new();
        common.insert(rel.to_path_buf(), fp(9));
        record_common(&mut common, &FpMap::new(), rel);
        assert_eq!(common.get(rel), Some(&fp(9)), "absent -> base unchanged");
        assert_eq!(common.len(), 1);
    }

    #[test]
    fn copy_atomic_replaces_and_leaves_no_tmp() {
        let dir = std::env::temp_dir().join(format!("copia-bidir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("s");
        let d = dir.join("sub/d");
        std::fs::write(&s, b"payload").unwrap();
        copy_atomic(&s, &d).unwrap();
        assert_eq!(std::fs::read(&d).unwrap(), b"payload");
        let mut tmp = d.as_os_str().to_owned();
        tmp.push(".copia-tmp");
        assert!(!PathBuf::from(tmp).exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
