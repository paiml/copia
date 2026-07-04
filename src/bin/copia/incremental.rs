//! Incremental recursive sync — the rsync-style quick check (skip files whose
//! size AND mtime match), atomic temp-file+rename delivery, mtime preservation,
//! `--delete` mirroring, `--exclude` filtering, and `--dry-run` — across all
//! three directions (push local->remote, pull remote->local, local->local).
//!
//! Built on `plan` (pure diff) + `meta` (both-sides metadata) + the `dir_sync`
//! and `transfer` primitives. This is the L1 foundation the bidirectional (L2)
//! reconciler and the hub daemon (L3) build on: both start from the same
//! `MetaMap` diff and reuse atomic delivery for concurrency-safe writes.

use super::dir_sync::{create_local_dirs, transfer_file_from_remote, TransferProgress};
use super::meta::{discover_local_with_meta, discover_remote_with_meta, set_local_mtime};
use super::plan::{build_plan, MetaMap, SyncPlan};
use super::transfer::{
    collect_dirs, create_remote_dirs, format_bytes, join_handles, transfer_file_to_remote,
    transfer_speed,
};
use super::FileLocation;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

/// Knobs for one incremental sync run.
pub struct SyncOptions {
    pub jobs: usize,
    pub verbose: bool,
    pub dry_run: bool,
    pub delete: bool,
    pub excludes: Vec<String>,
}

#[derive(Clone, Copy)]
enum Dir {
    Push,
    Pull,
}

/// Entry point: resolve the direction from the endpoints and run the plan.
pub async fn run_sync_recursive(
    source: FileLocation,
    dest: FileLocation,
    opts: SyncOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    match (source, dest) {
        (FileLocation::Local(local), FileLocation::Remote { host, path }) => {
            run_remote(Dir::Push, &host, &path, &local, &opts).await
        }
        (FileLocation::Remote { host, path }, FileLocation::Local(local)) => {
            run_remote(Dir::Pull, &host, &path, &local, &opts).await
        }
        (FileLocation::Local(from), FileLocation::Local(to)) => run_local(&from, &to, &opts).await,
        _ => Err("Recursive sync supports local->remote, local->local, and remote->local (not remote->remote)".into()),
    }
}

/// Append `.copia-tmp` to a path (unique per destination; never collides the way
/// `with_extension` would). The atomic-delivery staging name.
fn tmp_path(dst: &Path) -> PathBuf {
    let mut s = dst.as_os_str().to_owned();
    s.push(".copia-tmp");
    PathBuf::from(s)
}

/// Emit the quick-check summary; in dry-run also list every planned action.
fn print_plan(plan: &SyncPlan, dry_run: bool) {
    eprintln!(
        "Plan: {} to transfer, {} unchanged (skipped), {} to delete",
        plan.transfer.len(),
        plan.skipped,
        plan.delete.len()
    );
    if dry_run {
        for p in &plan.transfer {
            println!("send   {}", p.display());
        }
        for p in &plan.delete {
            println!("delete {}", p.display());
        }
        println!("(dry run) nothing was modified");
    }
}

/// Final report; errors if any file failed to transfer (non-zero exit).
fn report(
    start: Instant,
    progress: &TransferProgress,
    plan: &SyncPlan,
    src_desc: &str,
    dst_desc: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let elapsed = start.elapsed();
    let tx = progress.bytes();
    println!(
        "\nComplete: {} sent, {} skipped, {} deleted, {} failed",
        progress.done(),
        plan.skipped,
        plan.delete.len(),
        progress.failed()
    );
    println!(
        "Transferred {} in {:.1}s ({}/s)",
        format_bytes(tx),
        elapsed.as_secs_f64(),
        format_bytes(transfer_speed(tx, elapsed))
    );
    if verbose {
        eprintln!("{src_desc} -> {dst_desc}");
    }
    if progress.failed() > 0 {
        return Err(format!("{} file(s) failed to transfer", progress.failed()).into());
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
async fn run_remote(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    opts: &SyncOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (src_desc, dst_desc) = match dir {
        Dir::Push => (
            local_root.display().to_string(),
            format!("{host}:{remote_root}"),
        ),
        Dir::Pull => (
            format!("{host}:{remote_root}"),
            local_root.display().to_string(),
        ),
    };
    eprintln!("Scanning {src_desc} and {dst_desc}...");

    // Source drives transfers; destination drives skip + delete. A missing dest
    // (first sync) yields an empty map, so every source file is new.
    // Resolve each `?` into a binding BEFORE any await so no error-carrying
    // temporary is held across it (keeps the future `Send`).
    let (src_meta, dst_meta): (MetaMap, MetaMap) = match dir {
        Dir::Push => {
            let local = discover_local_with_meta(local_root)?;
            let remote = discover_remote_with_meta(host, remote_root)
                .await
                .unwrap_or_default();
            (local, remote)
        }
        Dir::Pull => {
            let remote = discover_remote_with_meta(host, remote_root).await?;
            let local = discover_local_with_meta(local_root).unwrap_or_default();
            (remote, local)
        }
    };
    // Empty source short-circuits (unless mirroring, where empty source is a
    // legitimate "delete everything on the destination").
    if src_meta.is_empty() && !opts.delete {
        eprintln!("No files found.");
        return Ok(());
    }
    let plan = build_plan(&src_meta, &dst_meta, &opts.excludes, opts.delete);
    print_plan(&plan, opts.dry_run);
    if opts.dry_run {
        return Ok(());
    }
    if plan.transfer.is_empty() && plan.delete.is_empty() {
        println!("Already up to date ({} files).", src_meta.len());
        return Ok(());
    }

    let dirs = collect_dirs(&plan.transfer);
    match dir {
        Dir::Push => create_remote_dirs(host, remote_root, &dirs).await?,
        Dir::Pull => create_local_dirs(local_root, &dirs)?,
    }

    let semaphore = Arc::new(Semaphore::new(opts.jobs));
    let progress = TransferProgress::new(plan.transfer.len() as u64);
    let mut handles = Vec::with_capacity(plan.transfer.len());
    for rel in &plan.transfer {
        let mtime = src_meta.get(rel).map(|m| m.mtime);
        let remote_file = format!("{}/{}", remote_root, rel.display());
        let local_file = local_root.join(rel);
        let host = host.to_string();
        let sem = Arc::clone(&semaphore);
        let prog = progress.clone();
        let rel_disp = rel.display().to_string();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await;
            let res = match dir {
                Dir::Push => transfer_file_to_remote(&local_file, &host, &remote_file, mtime).await,
                Dir::Pull => deliver_pull(&host, &remote_file, &local_file, mtime).await,
            };
            match res {
                Ok(size) => prog.record_ok(size),
                Err(e) => prog.record_err(&rel_disp, &e),
            }
        }));
    }
    join_handles(handles).await;

    if !plan.delete.is_empty() {
        apply_remote_deletes(dir, host, remote_root, local_root, &plan.delete).await;
    }
    report(start, &progress, &plan, &src_desc, &dst_desc, opts.verbose)
}

/// Atomic pull: stream to a `.copia-tmp` sibling, rename into place, set mtime.
async fn deliver_pull(
    host: &str,
    remote_file: &str,
    local_dest: &Path,
    mtime: Option<i64>,
) -> Result<u64, String> {
    let tmp = tmp_path(local_dest);
    let size = transfer_file_from_remote(host, remote_file, &tmp).await?;
    tokio::fs::rename(&tmp, local_dest)
        .await
        .map_err(|e| format!("rename {}: {e}", local_dest.display()))?;
    if let Some(t) = mtime {
        let _ = set_local_mtime(local_dest, t);
    }
    Ok(size)
}

/// Delete the mirror's stale files. Pull deletes locally; push batches one
/// `xargs rm` over SSH.
async fn apply_remote_deletes(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    dels: &[PathBuf],
) {
    match dir {
        Dir::Pull => {
            for rel in dels {
                let _ = std::fs::remove_file(local_root.join(rel));
            }
        }
        Dir::Push => {
            use std::fmt::Write as _;
            use tokio::io::AsyncWriteExt;
            let mut list = String::new();
            for rel in dels {
                let _ = writeln!(list, "{}/{}", remote_root, rel.display());
            }
            if let Ok(mut child) = tokio::process::Command::new("ssh")
                .arg(host)
                .arg("xargs -d '\\n' rm -f --")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(list.as_bytes()).await;
                    drop(stdin);
                }
                let _ = child.wait_with_output().await;
            }
        }
    }
    eprintln!("Deleted {} stale file(s) on the destination", dels.len());
}

#[allow(clippy::cast_possible_truncation)]
async fn run_local(
    src: &Path,
    dst: &Path,
    opts: &SyncOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    eprintln!("Scanning {} and {}...", src.display(), dst.display());
    let src_meta = discover_local_with_meta(src)?;
    if src_meta.is_empty() && !opts.delete {
        eprintln!("No files found.");
        return Ok(());
    }
    let dst_meta = discover_local_with_meta(dst).unwrap_or_default();
    let plan = build_plan(&src_meta, &dst_meta, &opts.excludes, opts.delete);
    print_plan(&plan, opts.dry_run);
    if opts.dry_run {
        return Ok(());
    }
    if plan.transfer.is_empty() && plan.delete.is_empty() {
        println!("Already up to date ({} files).", src_meta.len());
        return Ok(());
    }

    create_local_dirs(dst, &collect_dirs(&plan.transfer))?;
    let semaphore = Arc::new(Semaphore::new(opts.jobs));
    let progress = TransferProgress::new(plan.transfer.len() as u64);
    let mut handles = Vec::with_capacity(plan.transfer.len());
    for rel in &plan.transfer {
        let mtime = src_meta.get(rel).map(|m| m.mtime);
        let s = src.join(rel);
        let d = dst.join(rel);
        let sem = Arc::clone(&semaphore);
        let prog = progress.clone();
        let rel_disp = rel.display().to_string();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await;
            match deliver_local(&s, &d, mtime).await {
                Ok(size) => prog.record_ok(size),
                Err(e) => prog.record_err(&rel_disp, &e),
            }
        }));
    }
    join_handles(handles).await;

    if !plan.delete.is_empty() {
        for rel in &plan.delete {
            let _ = std::fs::remove_file(dst.join(rel));
        }
        eprintln!("Deleted {} stale file(s)", plan.delete.len());
    }
    report(
        start,
        &progress,
        &plan,
        &src.display().to_string(),
        &dst.display().to_string(),
        opts.verbose,
    )
}

/// Atomic local copy: copy to a `.copia-tmp` sibling, rename, set mtime.
async fn deliver_local(src: &Path, dst: &Path, mtime: Option<i64>) -> Result<u64, String> {
    let tmp = tmp_path(dst);
    let size = tokio::fs::copy(src, &tmp)
        .await
        .map_err(|e| format!("copy {}: {e}", src.display()))?;
    tokio::fs::rename(&tmp, dst)
        .await
        .map_err(|e| format!("rename {}: {e}", dst.display()))?;
    if let Some(t) = mtime {
        let _ = set_local_mtime(dst, t);
    }
    Ok(size)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn tmp_path_appends_suffix_without_colliding() {
        assert_eq!(
            tmp_path(Path::new("a/b.json")),
            PathBuf::from("a/b.json.copia-tmp")
        );
        // distinct sources -> distinct temp names (with_extension would collide)
        assert_ne!(
            tmp_path(Path::new("a/x.json")),
            tmp_path(Path::new("a/x.txt"))
        );
    }

    #[tokio::test]
    async fn deliver_local_is_atomic_and_preserves_mtime() {
        let base = std::env::temp_dir().join(format!("copia-incr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let src = base.join("s");
        let dst = base.join("d");
        std::fs::write(&src, b"payload").unwrap();
        let n = deliver_local(&src, &dst, Some(1_600_000_000))
            .await
            .unwrap();
        assert_eq!(n, 7);
        assert_eq!(std::fs::read(&dst).unwrap(), b"payload");
        // no temp file left behind
        assert!(!tmp_path(&dst).exists());
        let m = discover_local_with_meta(&base).unwrap();
        assert_eq!(m[&PathBuf::from("d")].mtime, 1_600_000_000);
        std::fs::remove_dir_all(&base).ok();
    }
}
