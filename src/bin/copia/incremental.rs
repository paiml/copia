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
use std::path::{Component, Path, PathBuf};
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
    deleted: usize,
    omitted: &[String],
    src_desc: &str,
    dst_desc: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let elapsed = start.elapsed();
    let tx = progress.bytes();
    println!(
        "\nComplete: {} sent, {} skipped, {} deleted, {} failed, {} omitted",
        progress.done(),
        plan.skipped,
        deleted,
        progress.failed(),
        omitted.len()
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
    omission_error(omitted)
}

/// The `(source, destination)` labels for this direction, as an operator sees
/// them in the scan line and in the final report.
fn endpoint_descriptions(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
) -> (String, String) {
    match dir {
        Dir::Push => (
            local_root.display().to_string(),
            format!("{host}:{remote_root}"),
        ),
        Dir::Pull => (
            format!("{host}:{remote_root}"),
            local_root.display().to_string(),
        ),
    }
}

/// Both sides' metadata, as `(source, destination)`.
///
/// Source drives transfers; destination drives skip + delete. A missing dest
/// (first sync) yields an empty map, so every source file is new.
/// Resolve each `?` into a binding BEFORE any await so no error-carrying
/// temporary is held across it (keeps the future `Send`).
/// Both listings. Every omission from either side lands in `omitted`: an
/// incomplete listing is never read as an empty tree (issue #50).
async fn discover_both(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    omitted: &mut Vec<String>,
) -> (MetaMap, MetaMap) {
    match dir {
        Dir::Push => {
            let mut local = discover_local_with_meta(local_root, omitted);
            // A name that is not UTF-8 has no spelling on the wire: it cannot be
            // sent under its own name, so it is an omission, never renamed.
            local.retain(|rel, _| {
                if rel.to_str().is_some() {
                    return true;
                }
                omitted.push(format!(
                    "non-UTF-8 name cannot be sent to {host}:{remote_root}: {}",
                    rel.display()
                ));
                false
            });
            // A symlink cannot be carried, whatever its target: copia sends bytes,
            // and a link's target is not the link. It is checked here, not left to
            // the transfer, because the quick check can skip an entry whose size
            // and mtime match, and then no transfer runs to fail. The transfer
            // refuses a link too, as a second line of defence.
            local.retain(|rel, _| {
                let full = local_root.join(rel);
                if !std::fs::symlink_metadata(&full).is_ok_and(|m| m.file_type().is_symlink()) {
                    return true;
                }
                omitted.push(format!("cannot carry symlink {}", full.display()));
                false
            });
            let remote = discover_remote_with_meta(host, remote_root, true, omitted).await;
            (local, remote)
        }
        Dir::Pull => {
            let remote = discover_remote_with_meta(host, remote_root, false, omitted).await;
            // The local root is the DESTINATION: an absent one is a first sync,
            // as it is for a push and for a local sync (`run_local`).
            let local = if is_absent(local_root) {
                MetaMap::new()
            } else {
                discover_local_with_meta(local_root, omitted)
            };
            (remote, local)
        }
    }
}

/// The veto, stated once. An incomplete source (any omission) may not delete.
/// The delete pass checks this flag itself, so the guarantee does not depend on
/// how the plan was built.
fn omission_error(omitted: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if omitted.is_empty() {
        return Ok(());
    }
    Err(format!(
        "run incomplete: {} source entr{} omitted (see OMITTED above)",
        omitted.len(),
        if omitted.len() == 1 { "y" } else { "ies" }
    )
    .into())
}

/// A transfer that failed leaves the destination as incomplete as an omitted
/// source entry does. It sets the omitted flag, so `--delete` is vetoed by the
/// same gate. Before this, the delete pass checked only the listing, and a
/// file that failed in `cat` or `ssh` still let the delete run.
fn veto_failed_transfers(progress: &TransferProgress, plan: &SyncPlan, omitted: &mut Vec<String>) {
    let failed = progress.failed();
    let unfinished = progress.unfinished();
    if failed == 0 && unfinished == 0 {
        return;
    }
    if failed > 0 {
        omitted.push(format!("{failed} transfer(s) failed"));
    }
    // A task that panicked or was cancelled never reports to the counters. It is
    // neither done nor failed, and the run is no more complete for it.
    if unfinished > 0 {
        omitted.push(format!("{unfinished} transfer(s) did not finish"));
    }
    if !plan.delete.is_empty() {
        eprintln!(
            "VETOED: --delete skipped {} destination file(s) because transfers failed",
            plan.delete.len()
        );
    }
}

/// Print every omission, and say plainly when `--delete` is vetoed by them.
fn print_omissions(omitted: &[String]) {
    if omitted.is_empty() {
        return;
    }
    eprintln!(
        "OMITTED {} source entr(y/ies); the run is incomplete:",
        omitted.len()
    );
    for line in omitted {
        eprintln!("  {line}");
    }
}

/// The veto line for `--delete`, or `None` when nothing is vetoed. A run with no
/// omissions vetoes nothing, and a run with nothing to delete has nothing to veto.
fn veto_line(omitted: &[String], delete_count: usize) -> Option<String> {
    if omitted.is_empty() || delete_count == 0 {
        return None;
    }
    Some(format!(
        "VETOED: --delete skipped {delete_count} destination file(s) because the source is incomplete"
    ))
}

fn veto_notice(omitted: &[String], plan: &SyncPlan) {
    print_omissions(omitted);
    if let Some(line) = veto_line(omitted, plan.delete.len()) {
        eprintln!("{line}");
    }
}

/// Create the destination-side directories this transfer needs.
async fn create_transfer_dirs(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    match dir {
        Dir::Push => create_remote_dirs(host, remote_root, dirs).await,
        Dir::Pull => create_local_dirs(local_root, dirs),
    }
}

/// Deliver one file in this direction: push streams it up over SSH, pull
/// streams it down into a temp sibling and renames it into place.
async fn deliver_one(
    dir: Dir,
    host: &str,
    remote_file: &str,
    local_file: &Path,
    mtime: Option<i64>,
) -> Result<u64, String> {
    match dir {
        Dir::Push => transfer_file_to_remote(local_file, host, remote_file, mtime).await,
        Dir::Pull => deliver_pull(host, remote_file, local_file, mtime).await,
    }
}

/// Spawn one task per planned file, `jobs`-limited by a shared semaphore.
/// Returns the handles in plan order plus the counters every task writes into.
#[allow(clippy::cast_possible_truncation)]
fn spawn_transfers(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    transfer: &[PathBuf],
    src_meta: &MetaMap,
    jobs: usize,
) -> (Vec<tokio::task::JoinHandle<()>>, TransferProgress) {
    let semaphore = Arc::new(Semaphore::new(jobs));
    let progress = TransferProgress::new(transfer.len() as u64);
    let mut handles = Vec::with_capacity(transfer.len());
    for rel in transfer {
        let mtime = src_meta.get(rel).map(|m| m.mtime);
        let remote_file = format!("{}/{}", remote_root, rel.display());
        let local_file = local_root.join(rel);
        let host = host.to_string();
        let sem = Arc::clone(&semaphore);
        let prog = progress.clone();
        let rel_disp = rel.display().to_string();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await;
            match deliver_one(dir, &host, &remote_file, &local_file, mtime).await {
                Ok(size) => prog.record_ok(size),
                Err(e) => prog.record_err(&rel_disp, &e),
            }
        }));
    }
    (handles, progress)
}

async fn run_remote(
    dir: Dir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    opts: &SyncOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (src_desc, dst_desc) = endpoint_descriptions(dir, host, remote_root, local_root);
    eprintln!("Scanning {src_desc} and {dst_desc}...");

    let mut omitted = Vec::new();
    let (src_meta, dst_meta) =
        discover_both(dir, host, remote_root, local_root, &mut omitted).await;
    // Empty source short-circuits (unless mirroring, where empty source is a
    // legitimate "delete everything on the destination").
    if src_meta.is_empty() && !opts.delete {
        eprintln!("No files found.");
        print_omissions(&omitted);
        return omission_error(&omitted);
    }
    let plan = build_plan(&src_meta, &dst_meta, &opts.excludes, opts.delete);
    print_plan(&plan, opts.dry_run);
    veto_notice(&omitted, &plan);
    if opts.dry_run {
        return omission_error(&omitted);
    }
    if plan.transfer.is_empty() && plan.delete.is_empty() {
        println!("Already up to date ({} files).", src_meta.len());
        return omission_error(&omitted);
    }

    let dirs = collect_dirs(&plan.transfer);
    create_transfer_dirs(dir, host, remote_root, local_root, &dirs).await?;

    let (handles, progress) = spawn_transfers(
        dir,
        host,
        remote_root,
        local_root,
        &plan.transfer,
        &src_meta,
        opts.jobs,
    );
    join_handles(handles).await;
    veto_failed_transfers(&progress, &plan, &mut omitted);

    // The veto, enforced at the delete pass itself.
    let deleted = if omitted.is_empty() && !plan.delete.is_empty() {
        apply_remote_deletes(dir, host, remote_root, local_root, &plan.delete).await;
        plan.delete.len()
    } else {
        0
    };
    report(
        start,
        &progress,
        &plan,
        deleted,
        &omitted,
        &src_desc,
        &dst_desc,
        opts.verbose,
    )
}

/// Atomic pull: stream to a `.copia-tmp` sibling, rename into place, set mtime.
async fn deliver_pull(
    host: &str,
    remote_file: &str,
    local_dest: &Path,
    mtime: Option<i64>,
) -> Result<u64, String> {
    let tmp = tmp_path(local_dest);
    check_staging_slot(&tmp)?;
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
            // NUL-terminated: a newline is a legal file-name byte, and a
            // newline-delimited list deleted the wrong names.
            let mut list = String::new();
            for rel in dels {
                let _ = write!(list, "{}/{}\0", remote_root, rel.display());
            }
            if let Ok(mut child) = tokio::process::Command::new("ssh")
                .arg(host)
                .arg("xargs -0 rm -f --")
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

/// True when `path` does not exist at all. A dangling symlink exists. A trailing
/// slash is trimmed first: `lstat("link/")` follows the link, so a dangling one
/// would read as absent.
fn is_absent(path: &Path) -> bool {
    // `..` is not resolved lexically, so a path through it is never read as absent.
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return false;
    }
    // Rebuilt from components, which drops a trailing slash and `.` segments on
    // any platform. The old code trimmed only UTF-8 names, so a non-UTF-8 link
    // with a trailing slash was lstat-ed through the link and read as absent.
    let normal: PathBuf = path.components().collect();
    // A DANGLING link on the way is not absence. `lstat("dangling/child")` fails
    // with ENOENT because the link's target is missing, not because `child` is.
    let dangling_ancestor = normal.ancestors().skip(1).any(|a| {
        !a.as_os_str().is_empty()
            && std::fs::symlink_metadata(a).is_ok_and(|m| m.file_type().is_symlink())
            && std::fs::metadata(a).is_err()
    });
    if dangling_ancestor {
        return false;
    }
    matches!(
        std::fs::symlink_metadata(&normal),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound
    )
}

/// The staging name `.copia-tmp` must be absent or a regular file. `File::create`
/// and `copy` follow a symlink and block on a FIFO, so a leftover of either kind
/// is refused by name instead of hanging the transfer.
fn check_staging_slot(tmp: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(tmp) {
        Ok(md) if !md.file_type().is_file() => Err(format!(
            "{}: staging path is not a regular file (fifo, symlink or other); remove it and retry",
            tmp.display()
        )),
        _ => Ok(()),
    }
}

#[allow(clippy::cast_possible_truncation)]
async fn run_local(
    src: &Path,
    dst: &Path,
    opts: &SyncOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    eprintln!("Scanning {} and {}...", src.display(), dst.display());
    // Omissions from EITHER side land here. Only the source's omissions can
    // matter to a delete, but an incomplete destination listing is equally an
    // incomplete run, so it fails the exit code too.
    let mut omitted = Vec::new();
    let src_meta = discover_local_with_meta(src, &mut omitted);
    if src_meta.is_empty() && !opts.delete {
        eprintln!("No files found.");
        print_omissions(&omitted);
        return omission_error(&omitted);
    }
    // A missing destination root is a first sync, not an omission: there is
    // nothing on it to delete. The source keeps its omission (above), so a
    // missing source under --delete still vetoes.
    let dst_meta = if is_absent(dst) {
        MetaMap::new()
    } else {
        discover_local_with_meta(dst, &mut omitted)
    };
    let plan = build_plan(&src_meta, &dst_meta, &opts.excludes, opts.delete);
    print_plan(&plan, opts.dry_run);
    veto_notice(&omitted, &plan);
    if opts.dry_run {
        return omission_error(&omitted);
    }
    if plan.transfer.is_empty() && plan.delete.is_empty() {
        println!("Already up to date ({} files).", src_meta.len());
        return omission_error(&omitted);
    }

    // Create EVERY directory in the source, not only the ones that happen to
    // contain a file in this transfer. An empty directory is structure, and an
    // archive that drops it has not faithfully copied the tree even though
    // every byte arrived (infra rsync-differential, `empty-dir` shape).
    // Falls back to the file-derived set if the source cannot be walked, so a
    // discovery failure degrades to the previous behaviour rather than
    // aborting a transfer that would otherwise succeed.
    let all_dirs =
        crate::transfer::discover_local_dirs(src).unwrap_or_else(|_| collect_dirs(&plan.transfer));
    create_local_dirs(dst, &all_dirs)?;
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
    veto_failed_transfers(&progress, &plan, &mut omitted);

    // The veto, enforced at the delete pass itself.
    let deleted = if omitted.is_empty() && !plan.delete.is_empty() {
        for rel in &plan.delete {
            let _ = std::fs::remove_file(dst.join(rel));
        }
        eprintln!("Deleted {} stale file(s)", plan.delete.len());
        plan.delete.len()
    } else {
        0
    };
    report(
        start,
        &progress,
        &plan,
        deleted,
        &omitted,
        &src.display().to_string(),
        &dst.display().to_string(),
        opts.verbose,
    )
}

/// Atomic local delivery: write a `.copia-tmp` sibling, rename, set mtime.
///
/// A SYMLINK is recreated as a symlink, never followed. `tokio::fs::copy`
/// dereferences, so the previous version replaced `link -> target` with a second
/// copy of the target's bytes — the link was gone and the tree silently grew.
/// Paired with the walk fix in `transfer::discover_local_files`; either alone
/// leaves `copia verify` disagreeing with what `copia sync` actually wrote.
async fn deliver_local(src: &Path, dst: &Path, mtime: Option<i64>) -> Result<u64, String> {
    if let Ok(md) = tokio::fs::symlink_metadata(src).await {
        let ft = md.file_type();
        // REFUSE what cannot be copied, LOUDLY. Never open it.
        //
        // Making the walk record every entry kind — so `verify` can SEE a
        // FIFO instead of silently agreeing it is absent — means this
        // function now receives paths it must not read. `tokio::fs::copy`
        // on a FIFO BLOCKS until a writer appears. Measured: it hung a sync
        // for ten minutes until the process was killed. A hang is not better
        // than a silent drop; it is the same failure with worse manners.
        //
        // So the entry stays VISIBLE to the walk and this transfer FAILS with
        // a named reason, counted in the run summary — which leaves `verify`
        // reporting the destination as `missing`, the correct refusal.
        if !ft.is_symlink() && !ft.is_file() {
            return Err(format!(
                "{}: copia cannot carry this entry kind (fifo/socket/device); it is \
                 recorded, not skipped, so verify refuses to certify a destination \
                 without it",
                src.display()
            ));
        }
        if md.file_type().is_symlink() {
            let target = tokio::fs::read_link(src)
                .await
                .map_err(|e| format!("read_link {}: {e}", src.display()))?;
            // Replace whatever is there. A stale regular file at this path is
            // exactly what the old dereferencing behaviour left behind.
            let _ = tokio::fs::remove_file(dst).await;
            #[cfg(unix)]
            tokio::fs::symlink(&target, dst)
                .await
                .map_err(|e| format!("symlink {}: {e}", dst.display()))?;
            #[cfg(not(unix))]
            return Err(format!(
                "cannot recreate symlink {} on this platform",
                dst.display()
            ));
            // mtime is deliberately NOT set: on Linux `utimes` follows the link
            // and would stamp the TARGET, mutating a file we were only asked to
            // link to. rsync needs -h/--no-dereference for the same reason.
            return Ok(0);
        }
    }
    let tmp = tmp_path(dst);
    check_staging_slot(&tmp)?;
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

    /// A FIFO must be REFUSED with a named reason — never opened, never hung on.
    ///
    /// Making the walk record every entry kind means this function now receives
    /// paths it must not read. `tokio::fs::copy` on a FIFO blocks until a writer
    /// appears; measured, it hung a real sync for ten minutes. The timeout below
    /// is the assertion: without it a regression hangs the suite instead of
    /// failing it.
    #[cfg(unix)]
    #[tokio::test]
    async fn deliver_local_refuses_a_fifo_instead_of_blocking_on_it() {
        let base = std::env::temp_dir().join(format!(
            "copia-dl-fifo-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&base).unwrap();
        let fifo = base.join("pipe");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !made {
            let _ = std::fs::remove_dir_all(&base);
            return;
        }
        let dst = base.join("delivered");

        // A thread + channel rather than tokio::time::timeout: this crate does
        // not enable tokio's `time` feature, and adding a feature for one test
        // assertion is a worse trade than four lines here.
        let (tx, rx) = std::sync::mpsc::channel();
        let (f, d) = (fifo, dst.clone());
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            let _ = tx.send(rt.block_on(deliver_local(&f, &d, None)));
        });
        let res = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("deliver_local BLOCKED on a FIFO — it must refuse, not read");

        let err = res.expect_err("a FIFO must not deliver successfully");
        assert!(
            err.contains("cannot carry this entry kind"),
            "the refusal must NAME why, so an operator knows the entry was seen \
             and declined rather than lost: {err}"
        );
        assert!(
            !dst.exists(),
            "a refused entry must leave nothing behind at the destination"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// `deliver_local` must recreate a symlink, not copy what it points at.
    ///
    /// `tokio::fs::copy` dereferences, so the previous version replaced
    /// `link -> target` with a second copy of the target's bytes: the link was
    /// gone and the tree silently grew. Caught by the rsync differential
    /// harness; pinned here so it cannot regress without a unit failure.
    #[cfg(unix)]
    #[tokio::test]
    async fn deliver_local_recreates_a_symlink_rather_than_following_it() {
        let base = std::env::temp_dir().join(format!("copia-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let target = base.join("target.txt");
        std::fs::write(&target, b"payload").unwrap();
        let link = base.join("link");
        std::os::unix::fs::symlink("target.txt", &link).unwrap();
        let dst = base.join("delivered");

        deliver_local(&link, &dst, Some(1_000_000)).await.unwrap();

        let md = std::fs::symlink_metadata(&dst).unwrap();
        assert!(
            md.file_type().is_symlink(),
            "delivered a regular file where the source was a symlink"
        );
        assert_eq!(
            std::fs::read_link(&dst).unwrap(),
            std::path::PathBuf::from("target.txt")
        );

        // The TARGET must be untouched. `utimes` follows a link, so stamping the
        // delivered path would have mutated a file we were only asked to link to.
        let tmeta = std::fs::metadata(&target).unwrap();
        assert_eq!(tmeta.len(), 7, "the link's target was rewritten");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Delivering over an existing regular file must leave a link, not fail.
    /// That is exactly the state the old dereferencing behaviour left behind,
    /// so the first corrected run has to be able to repair it.
    #[cfg(unix)]
    #[tokio::test]
    async fn deliver_local_replaces_a_stale_regular_file_with_the_link() {
        let base = std::env::temp_dir().join(format!("copia-dl2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("t.txt"), b"x").unwrap();
        let link = base.join("l");
        std::os::unix::fs::symlink("t.txt", &link).unwrap();
        let dst = base.join("stale");
        std::fs::write(&dst, b"left over from the dereferencing bug").unwrap();

        deliver_local(&link, &dst, None).await.unwrap();

        assert!(std::fs::symlink_metadata(&dst)
            .unwrap()
            .file_type()
            .is_symlink());
        let _ = std::fs::remove_dir_all(&base);
    }

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
        let m = discover_local_with_meta(&base, &mut Vec::new());
        assert_eq!(m[&PathBuf::from("d")].mtime, 1_600_000_000);
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 RED: a source entry whose stat fails is silently dropped from the
    /// source map, so `--delete` removes its destination copy. A `0o444` directory
    /// is listable but its entries cannot be lstat-ed (search bit missing).
    #[cfg(unix)]
    #[tokio::test]
    async fn sync_delete_is_vetoed_when_a_source_entry_cannot_be_statted() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("copia-50-stat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("s");
        let dst = base.join("d");
        std::fs::create_dir_all(src.join("locked")).unwrap();
        std::fs::create_dir_all(dst.join("locked")).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(src.join("locked/b.txt"), b"b").unwrap();
        std::fs::write(dst.join("locked/b.txt"), b"b").unwrap();
        std::fs::write(dst.join("stale.txt"), b"stale").unwrap();
        std::fs::set_permissions(src.join("locked"), std::fs::Permissions::from_mode(0o444))
            .unwrap();
        let opts = SyncOptions {
            jobs: 2,
            verbose: false,
            dry_run: false,
            delete: true,
            excludes: vec![],
        };
        let result = run_local(&src, &dst, &opts).await;
        std::fs::set_permissions(src.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert!(result.is_err(), "an omitted source entry must fail the run");
        assert!(
            dst.join("stale.txt").exists(),
            "delete ran on an incomplete source"
        );
        assert!(
            dst.join("locked/b.txt").exists(),
            "omitted source entry's destination copy was deleted"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 acceptance: an unreadable subdirectory with `--delete` deletes
    /// nothing and exits non-zero.
    #[cfg(unix)]
    #[tokio::test]
    async fn sync_delete_is_vetoed_by_an_unreadable_subdirectory() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("copia-50-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("s");
        let dst = base.join("d");
        std::fs::create_dir_all(src.join("locked")).unwrap();
        std::fs::create_dir_all(dst.join("locked")).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(src.join("locked/b.txt"), b"b").unwrap();
        std::fs::write(dst.join("locked/b.txt"), b"b").unwrap();
        std::fs::write(dst.join("stale.txt"), b"stale").unwrap();
        std::fs::set_permissions(src.join("locked"), std::fs::Permissions::from_mode(0o000))
            .unwrap();
        let opts = SyncOptions {
            jobs: 2,
            verbose: false,
            dry_run: false,
            delete: true,
            excludes: vec![],
        };
        let result = run_local(&src, &dst, &opts).await;
        std::fs::set_permissions(src.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert!(
            result.is_err(),
            "an unreadable subdirectory must exit non-zero"
        );
        assert!(
            dst.join("stale.txt").exists(),
            "delete ran on an incomplete source"
        );
        assert!(
            dst.join("locked/b.txt").exists(),
            "omitted entry was deleted"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    pub(super) fn delete_opts() -> SyncOptions {
        SyncOptions {
            jobs: 2,
            verbose: false,
            dry_run: false,
            delete: true,
            excludes: vec![],
        }
    }

    /// Put an `ssh` shim first on PATH. It runs its command under bash on this
    /// machine, so `run_remote` can be driven against a temp tree.
    #[cfg(unix)]
    pub(super) fn fake_ssh_on_path() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            use std::os::unix::fs::PermissionsExt;
            let dir = std::env::temp_dir().join(format!("copia-fake-ssh-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let shim = dir.join("ssh");
            std::fs::write(&shim, "#!/bin/sh\nshift\nexec bash -c \"$1\"\n").unwrap();
            std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
            let mut paths = vec![dir];
            paths.extend(std::env::split_paths(
                &std::env::var_os("PATH").unwrap_or_default(),
            ));
            std::env::set_var("PATH", std::env::join_paths(paths).unwrap());
        });
    }

    /// Run `run_remote` on its own thread with a deadline. A FIFO that reaches
    /// `cat` blocks forever, so a regression must fail instead of hanging the suite.
    #[cfg(unix)]
    pub(super) fn run_remote_with_deadline(
        dir: Dir,
        remote_root: &Path,
        local_root: &Path,
        opts: SyncOptions,
    ) -> Result<(), String> {
        let (tx, rx) = std::sync::mpsc::channel();
        let remote = remote_root.to_string_lossy().into_owned();
        let local = local_root.to_path_buf();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            let res = rt
                .block_on(run_remote(dir, "shim.invalid", &remote, &local, &opts))
                .map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("run_remote BLOCKED: it hung instead of vetoing or refusing")
    }

    /// Issue #50 review finding A RED: a remote FIFO on a pull under --delete must
    /// set the omitted flag and veto the delete. The remote listing carried no
    /// entry kind, so the FIFO was planned for transfer and the stale local file
    /// was deleted.
    #[cfg(unix)]
    #[test]
    fn remote_pull_fifo_entry_vetoes_delete() {
        fake_ssh_on_path();
        let base =
            std::env::temp_dir().join(format!("copia-50-remote-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(remote.join("pipe"))
            .status()
            .expect("mkfifo")
            .success();
        assert!(made, "mkfifo failed");
        std::fs::write(local.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        let err = res.expect_err("a FIFO on the remote must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert!(
            local.join("stale.txt").exists(),
            "delete ran past a remote FIFO the listing could not carry"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding A RED: a local FIFO in a push source under
    /// --delete must set the omitted flag and veto the delete. The FIFO was
    /// planned for transfer, and `File::open` on it blocked until a writer
    /// appeared, so the run hung instead of failing.
    #[cfg(unix)]
    #[test]
    fn local_push_fifo_entry_vetoes_delete() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-push-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(src.join("pipe"))
            .status()
            .expect("mkfifo")
            .success();
        assert!(made, "mkfifo failed");
        std::fs::write(remote.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());

        let err = res.expect_err("a FIFO in the local source must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert!(
            remote.join("stale.txt").exists(),
            "delete ran past a local FIFO the source could not carry"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding B: a nonexistent destination root with --delete is
    /// a first sync. Nothing on the destination can be deleted, so the run must
    /// deliver the source and succeed. The missing root was an omission, which
    /// failed the run.
    #[cfg(unix)]
    #[test]
    fn push_to_an_absent_remote_root_is_a_first_sync() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-first-push-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote-absent");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(src.join("sub/b.txt"), b"b").unwrap();

        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());

        assert!(res.is_ok(), "first sync to an absent root failed: {res:?}");
        assert_eq!(std::fs::read(remote.join("a.txt")).unwrap(), b"a");
        assert_eq!(std::fs::read(remote.join("sub/b.txt")).unwrap(), b"b");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding B, local side: an absent destination root with
    /// --delete is a first sync and must succeed.
    #[tokio::test]
    async fn sync_to_an_absent_local_root_is_a_first_sync() {
        let base =
            std::env::temp_dir().join(format!("copia-50-first-local-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let dst = base.join("dst-absent");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();

        let res = run_local(&src, &dst, &delete_opts()).await;

        assert!(res.is_ok(), "first sync to an absent root failed: {res:?}");
        assert_eq!(std::fs::read(dst.join("a.txt")).unwrap(), b"a");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding C: `run_remote`'s delete veto. A remote listing
    /// that exits non-zero reads as an empty source. With --delete that would plan
    /// every local file for deletion, so the run must delete nothing and fail.
    #[cfg(unix)]
    #[test]
    fn remote_listing_failure_vetoes_run_remote_delete() {
        use std::os::unix::fs::PermissionsExt;
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-veto-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(remote.join("locked")).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        std::fs::write(remote.join("locked/b.txt"), b"b").unwrap();
        std::fs::write(local.join("stale.txt"), b"stale").unwrap();
        std::fs::set_permissions(
            remote.join("locked"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        std::fs::set_permissions(
            remote.join("locked"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        assert!(res.is_err(), "a failed remote listing must fail the run");
        assert!(
            local.join("stale.txt").exists(),
            "run_remote deleted on an incomplete remote listing"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding 1 RED: a remote symlink whose target is a FIFO
    /// must be an omission on a pull. The listing carried it as a symlink, `cat`
    /// followed it into the FIFO, and the run blocked forever.
    #[cfg(unix)]
    #[test]
    fn remote_symlink_to_fifo_vetoes_pull_delete() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-link-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        // The FIFO sits OUTSIDE the listed root, so the link is the only entry
        // that points at it.
        let made = std::process::Command::new("mkfifo")
            .arg(base.join("pipe"))
            .status()
            .expect("mkfifo")
            .success();
        assert!(made, "mkfifo failed");
        std::os::unix::fs::symlink(base.join("pipe"), remote.join("link")).unwrap();
        std::fs::write(local.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        let err = res.expect_err("a symlink to a FIFO on the remote must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert_eq!(std::fs::read(local.join("a.txt")).unwrap(), b"a");
        assert!(
            local.join("stale.txt").exists(),
            "delete ran past a remote symlink the listing could not carry"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding 1 RED: a dangling remote symlink on a pull must
    /// veto --delete. Its transfer failed, and the delete ran anyway.
    #[cfg(unix)]
    #[test]
    fn remote_dangling_symlink_vetoes_pull_delete() {
        fake_ssh_on_path();
        let base =
            std::env::temp_dir().join(format!("copia-50-dangling-pull-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        std::os::unix::fs::symlink("nowhere", remote.join("dangling")).unwrap();
        std::fs::write(local.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        let err = res.expect_err("a dangling remote symlink must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert_eq!(std::fs::read(local.join("a.txt")).unwrap(), b"a");
        assert!(
            local.join("stale.txt").exists(),
            "delete ran past a dangling remote symlink"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding 2 RED: a pull into a destination root that does
    /// not exist is a first sync, as it is for a push and for a local sync.
    #[cfg(unix)]
    #[test]
    fn pull_to_an_absent_local_root_is_a_first_sync() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-first-pull-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local-absent");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        assert!(res.is_ok(), "first pull to an absent root failed: {res:?}");
        assert_eq!(std::fs::read(local.join("a.txt")).unwrap(), b"a");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Issue #50 review finding 4 RED: a file that fails to transfer must veto
    /// --delete. The unreadable remote file failed in `cat`, the omitted flag
    /// stayed clear, and the stale local file was deleted.
    #[cfg(unix)]
    #[test]
    fn failed_pull_transfer_vetoes_delete() {
        use std::os::unix::fs::PermissionsExt;
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-xfer-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        std::fs::write(remote.join("secret.txt"), b"s").unwrap();
        std::fs::set_permissions(
            remote.join("secret.txt"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();
        std::fs::write(local.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        std::fs::set_permissions(
            remote.join("secret.txt"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        assert!(res.is_err(), "a failed transfer must fail the run");
        assert_eq!(
            std::fs::read(local.join("a.txt")).unwrap(),
            b"a",
            "the healthy sibling must still transfer"
        );
        assert!(
            local.join("stale.txt").exists(),
            "--delete ran although a transfer failed"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Review finding (both lanes) RED: a local source name that is not UTF-8 has
    /// no spelling on the wire. Lossy naming wrote `caf\u{FFFD}` and let --delete
    /// remove the stale remote file.
    #[cfg(unix)]
    #[test]
    fn push_of_a_non_utf8_source_name_is_an_omission() {
        use std::os::unix::ffi::OsStrExt;
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-50-nonutf8-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::write(src.join(std::ffi::OsStr::from_bytes(b"caf\xe9")), b"one").unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(remote.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());

        let err = res.expect_err("a non-UTF-8 source name must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert_eq!(
            std::fs::read(remote.join("a.txt")).unwrap(),
            b"a",
            "the healthy sibling must still transfer"
        );
        assert!(
            !remote.join("caf\u{FFFD}").exists(),
            "the non-UTF-8 name was lossy-renamed and written"
        );
        assert!(
            remote.join("stale.txt").exists(),
            "--delete ran on a source whose non-UTF-8 name could not be carried"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Review finding (both lanes) RED: a dangling local symlink on a push must veto
    /// --delete even when the destination already holds a regular file with the
    /// link's lstat size and mtime. The quick check then skips the entry, so the
    /// transfer that would have failed never runs.
    #[cfg(unix)]
    #[test]
    fn push_of_a_dangling_symlink_vetoes_delete_even_when_quick_check_matches() {
        fake_ssh_on_path();
        let base =
            std::env::temp_dir().join(format!("copia-50-dangling-push-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::os::unix::fs::symlink("nowhere", src.join("dangling")).unwrap();
        let link = std::fs::symlink_metadata(src.join("dangling")).unwrap();
        let mtime = link
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        std::fs::write(remote.join("dangling"), b"nowhere").unwrap();
        set_local_mtime(&remote.join("dangling"), i64::try_from(mtime).unwrap()).unwrap();
        std::fs::write(remote.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());

        let err = res.expect_err("a dangling local symlink must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert_eq!(
            std::fs::read(remote.join("a.txt")).unwrap(),
            b"a",
            "the healthy sibling must still transfer"
        );
        assert!(
            remote.join("stale.txt").exists(),
            "--delete ran past a local symlink that cannot be carried"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Review finding 4, local side RED: a failed local transfer must veto --delete.
    /// `run_local` had no test for this path.
    #[cfg(unix)]
    #[tokio::test]
    async fn failed_local_transfer_vetoes_delete() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("copia-50-local-xfer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(src.join("secret.txt"), b"s").unwrap();
        std::fs::set_permissions(
            src.join("secret.txt"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();
        std::fs::write(dst.join("stale.txt"), b"stale").unwrap();

        let res = run_local(&src, &dst, &delete_opts()).await;

        std::fs::set_permissions(
            src.join("secret.txt"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        assert!(res.is_err(), "a failed local transfer must fail the run");
        assert_eq!(
            std::fs::read(dst.join("a.txt")).unwrap(),
            b"a",
            "the healthy sibling must still transfer"
        );
        assert!(
            dst.join("stale.txt").exists(),
            "--delete ran although a local transfer failed"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Review finding (round 3, minor) RED: a dangling symlink named with a trailing
    /// slash exists. `lstat("link/")` follows the link, so it read as absent and a
    /// local destination was treated as a first sync.
    #[cfg(unix)]
    #[test]
    fn dangling_symlink_with_trailing_slash_is_not_absent() {
        let base = std::env::temp_dir().join(format!("copia-50-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::os::unix::fs::symlink("nowhere", base.join("dangling")).unwrap();
        assert!(!is_absent(&base.join("dangling/")));
        assert!(!is_absent(&base.join("dangling")));
        assert!(is_absent(&base.join("missing/")));
        std::fs::remove_dir_all(&base).ok();
    }

    /// Review finding 3 RED: a transfer that never reports back (a panicked or
    /// cancelled task) is neither done nor failed. It must still veto --delete.
    #[test]
    fn a_transfer_that_never_reports_vetoes_delete() {
        let progress = TransferProgress::new(2);
        progress.record_ok(1);
        let plan = SyncPlan {
            transfer: vec![PathBuf::from("a"), PathBuf::from("b")],
            skipped: 0,
            delete: vec![PathBuf::from("stale")],
        };
        let mut omitted = Vec::new();
        veto_failed_transfers(&progress, &plan, &mut omitted);
        assert!(
            !omitted.is_empty(),
            "an unfinished transfer did not veto the delete"
        );
    }
}

// Review regressions for copia #50 round 4 (findings 2, 4, 6, 7, lane-2 FIFO).
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod r93_incremental {
    use super::tests::{delete_opts, fake_ssh_on_path, run_remote_with_deadline};
    use super::*;
    use crate::transfer::stream_to_remote;

    /// Finding 2 (HIGH): a remote file whose name holds a newline is deleted by its
    /// own name. A newline-delimited xargs split it into two wrong names.
    #[cfg(unix)]
    #[test]
    fn remote_delete_removes_a_file_whose_name_contains_a_newline() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-nl-del-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(remote.join("stale\nname.txt"), b"stale").unwrap();
        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());
        assert!(res.is_ok(), "{res:?}");
        assert!(
            !remote.join("stale\nname.txt").exists(),
            "the newline-named stale file survived --delete"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 2 (HIGH): a remote directory whose name holds a newline is created
    /// under its own name. The dir name ends in a newline so a split never creates a
    /// stray directory outside the test tree.
    #[cfg(unix)]
    #[test]
    fn push_creates_a_directory_whose_name_contains_a_newline() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-nl-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote");
        std::fs::create_dir_all(src.join("ab\n")).unwrap();
        std::fs::write(src.join("ab\n/f.txt"), b"f").unwrap();
        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());
        assert!(res.is_ok(), "{res:?}");
        assert_eq!(std::fs::read(remote.join("ab\n/f.txt")).unwrap(), b"f");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 4 (MEDIUM-HIGH): a push to a multi-level absent root is a first sync.
    #[cfg(unix)]
    #[test]
    fn push_to_a_multi_level_absent_remote_root_is_a_first_sync() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-deep-push-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("a/b/remote");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());
        assert!(res.is_ok(), "{res:?}");
        assert_eq!(std::fs::read(remote.join("a.txt")).unwrap(), b"a");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Lane-2 (LOW-MEDIUM): a leading-dash remote root is a path, not an option, to
    /// `cd`. The root is relative and lives in the test's working directory.
    #[cfg(unix)]
    #[tokio::test]
    async fn remote_listing_accepts_a_root_that_begins_with_a_dash() {
        fake_ssh_on_path();
        let root = format!("-copia-93-dash-{}", std::process::id());
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(format!("{root}/f.txt"), b"x").unwrap();
        let mut omitted = Vec::new();
        let m = discover_remote_with_meta("shim.invalid", &root, false, &mut omitted).await;
        std::fs::remove_dir_all(&root).ok();
        assert!(omitted.is_empty(), "{omitted:?}");
        assert!(m.contains_key(&PathBuf::from("f.txt")), "{m:?}");
    }

    /// Lane-2 (LOW): a non-UTF-8 dangling link with a trailing slash is not absent.
    #[cfg(unix)]
    #[test]
    fn is_absent_does_not_read_a_non_utf8_dangling_link_as_absent() {
        use std::os::unix::ffi::OsStrExt;
        let base = std::env::temp_dir().join(format!("copia-93-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let name = std::ffi::OsStr::from_bytes(b"link\xff");
        std::os::unix::fs::symlink("nowhere", base.join(name)).unwrap();
        let mut with_slash = base.join(name).into_os_string();
        with_slash.push("/");
        assert!(
            !is_absent(Path::new(&with_slash)),
            "dangling link read as absent"
        );
        assert!(!is_absent(&base.join(name)));
        std::fs::remove_dir_all(&base).ok();
    }

    /// Lane-2 (hang): a pre-existing `.copia-tmp` FIFO at the local pull destination
    /// is refused by name. `File::create` on it would block with no reader.
    #[cfg(unix)]
    #[tokio::test]
    async fn pull_refuses_a_fifo_at_the_staging_name() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-pull-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(local.join("a.txt.copia-tmp"))
            .status()
            .unwrap()
            .success();
        assert!(made, "mkfifo failed");
        let dst = local.join("a.txt");
        let res = deliver_pull(
            "shim.invalid",
            &remote.join("a.txt").to_string_lossy(),
            &dst,
            None,
        )
        .await;
        assert!(res.is_err(), "a FIFO staging path was used: {res:?}");
        assert!(!dst.exists());
        std::fs::remove_dir_all(&base).ok();
    }

    /// Lane-2 (hang): the same refusal on a local-to-local delivery.
    #[cfg(unix)]
    #[tokio::test]
    async fn local_delivery_refuses_a_fifo_at_the_staging_name() {
        let base = std::env::temp_dir().join(format!("copia-93-local-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let src = base.join("s");
        std::fs::write(&src, b"payload").unwrap();
        let dst = base.join("d");
        let made = std::process::Command::new("mkfifo")
            .arg(tmp_path(&dst))
            .status()
            .unwrap()
            .success();
        assert!(made, "mkfifo failed");
        let res = deliver_local(&src, &dst, None).await;
        assert!(res.is_err(), "a FIFO staging path was used: {res:?}");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Lane-2 (hang): a push whose remote staging name is a FIFO is refused on the
    /// remote, before `cat` can block on it.
    #[cfg(unix)]
    #[tokio::test]
    async fn push_refuses_a_fifo_at_the_remote_staging_name() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-push-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("remote")).unwrap();
        let local = base.join("a.txt");
        std::fs::write(&local, b"a").unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(base.join("remote/a.txt.copia-tmp"))
            .status()
            .unwrap()
            .success();
        assert!(made, "mkfifo failed");
        let remote_file = base.join("remote/a.txt").to_string_lossy().into_owned();
        let res = transfer_file_to_remote(&local, "shim.invalid", &remote_file, None).await;
        assert!(res.is_err(), "a FIFO staging path was written to: {res:?}");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 6 (MEDIUM): VETOED is printed only when omissions veto deletes.
    #[test]
    fn veto_line_is_printed_only_when_something_is_vetoed() {
        assert_eq!(
            veto_line(&[], 3),
            None,
            "a healthy --delete run printed VETOED"
        );
        assert_eq!(veto_line(&["x".to_owned()], 0), None);
        let line = veto_line(&["x".to_owned()], 3).expect("an omission with deletes must veto");
        assert!(line.starts_with("VETOED: --delete skipped 3"), "{line}");
    }

    /// A reader that yields `good` bytes, then fails: a local read that dies
    /// mid-stream, after the remote command has already started.
    struct FailsMidStream {
        given: usize,
        good: usize,
    }

    impl tokio::io::AsyncRead for FailsMidStream {
        fn poll_read(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let this = self.get_mut();
            if this.given >= this.good {
                return std::task::Poll::Ready(Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "injected read failure",
                )));
            }
            let n = (this.good - this.given).min(buf.remaining());
            buf.put_slice(&vec![b'N'; n]);
            this.given += n;
            std::task::Poll::Ready(Ok(()))
        }
    }

    /// Residual RED: a local read that fails after the stream has started must
    /// never commit the bytes already sent. The remote `mv` ran on a partial file.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_mid_stream_local_read_failure_never_commits_a_partial_file() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-midstream-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("remote")).unwrap();
        let remote_file = base.join("remote/data.bin");
        std::fs::write(&remote_file, b"remote-original").unwrap();

        let res = stream_to_remote(
            FailsMidStream {
                given: 0,
                good: 1000,
            },
            4000,
            "shim.invalid",
            &remote_file.to_string_lossy(),
            None,
        )
        .await;
        // Give a detached `ssh` shim time to finish before the remote is read.
        std::thread::sleep(std::time::Duration::from_millis(500));

        assert!(
            res.is_err(),
            "an injected read failure reported success: {res:?}"
        );
        assert_eq!(
            std::fs::read(&remote_file).unwrap(),
            b"remote-original",
            "a partial file was committed after a mid-stream local read failure"
        );
        assert!(
            !base.join("remote/data.bin.copia-tmp").exists(),
            "the staging file was left behind"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 2 RED: a local source that cannot be read must fail BEFORE the
    /// remote target is opened. `cat >` truncates on start, so the remote file
    /// was emptied by a push whose local open had already failed.
    #[cfg(unix)]
    #[tokio::test]
    async fn push_refuses_an_unreadable_local_source_without_touching_the_remote() {
        use std::os::unix::fs::PermissionsExt;
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-eacces-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("remote")).unwrap();
        let local = base.join("secret.txt");
        std::fs::write(&local, b"new").unwrap();
        std::fs::set_permissions(&local, std::fs::Permissions::from_mode(0o000)).unwrap();
        let remote_file = base.join("remote/secret.txt");
        std::fs::write(&remote_file, b"remote-original").unwrap();

        let res =
            transfer_file_to_remote(&local, "shim.invalid", &remote_file.to_string_lossy(), None)
                .await;
        // The remote write is detached once spawned; give a spawned `cat >` time
        // to truncate before the remote file is read.
        std::thread::sleep(std::time::Duration::from_millis(500));
        std::fs::set_permissions(&local, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(res.is_err(), "an unreadable source was sent: {res:?}");
        assert_eq!(
            std::fs::read(&remote_file).unwrap(),
            b"remote-original",
            "the remote file was truncated by a push that never read its source"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 3 (push) RED: a symlink to a regular file must not be streamed as
    /// its target's bytes. `File::open` followed the link.
    #[cfg(unix)]
    #[tokio::test]
    async fn push_refuses_a_symlink_source_and_never_sends_its_target() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-link-push-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("remote")).unwrap();
        std::fs::write(base.join("real.txt"), b"secret").unwrap();
        std::os::unix::fs::symlink("real.txt", base.join("link")).unwrap();
        let remote_file = base.join("remote/link");

        let res = transfer_file_to_remote(
            &base.join("link"),
            "shim.invalid",
            &remote_file.to_string_lossy(),
            None,
        )
        .await;

        assert!(res.is_err(), "a symlink source was sent: {res:?}");
        assert!(
            !remote_file.exists(),
            "the target's bytes were written under the link's name"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 3 (push, run level) RED: a symlink to a regular file in the source
    /// is an omission, so `--delete` is vetoed, and no bytes reach the remote.
    #[cfg(unix)]
    #[test]
    fn push_omits_a_symlink_source_and_vetoes_delete() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-link-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let src = base.join("src");
        let remote = base.join("remote");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::write(src.join("a.txt"), b"a").unwrap();
        std::fs::write(src.join("real.txt"), b"secret").unwrap();
        std::os::unix::fs::symlink("real.txt", src.join("link")).unwrap();
        std::fs::write(remote.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Push, &remote, &src, delete_opts());

        let err = res.expect_err("a symlink source must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert!(
            remote.join("stale.txt").exists(),
            "--delete ran past a symlink the source could not carry"
        );
        assert!(
            !remote.join("link").exists(),
            "the symlink's target was sent under the link's name"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 3 (pull, run level) RED: a remote symlink to a regular file is an
    /// omission. `cat` used to follow it and the link arrived as a regular file.
    #[cfg(unix)]
    #[test]
    fn remote_symlink_to_file_is_an_omission_not_carried_bytes() {
        fake_ssh_on_path();
        let base = std::env::temp_dir().join(format!("copia-93-link-pull-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let remote = base.join("remote");
        let local = base.join("local");
        std::fs::create_dir_all(&remote).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(remote.join("a.txt"), b"a").unwrap();
        std::os::unix::fs::symlink("a.txt", remote.join("link")).unwrap();
        std::fs::write(local.join("stale.txt"), b"stale").unwrap();

        let res = run_remote_with_deadline(Dir::Pull, &remote, &local, delete_opts());

        let err = res.expect_err("a remote symlink to a file must fail the run");
        assert!(err.contains("omitted"), "{err}");
        assert!(
            local.join("stale.txt").exists(),
            "--delete ran past a remote symlink"
        );
        assert!(
            !local.join("link").exists(),
            "the remote link's target was written under the link's name"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 7 RED: a path UNDER a dangling symlink is not absent. Its ancestor
    /// is a link that exists, and `lstat` reports ENOENT only through the dangle.
    #[cfg(unix)]
    #[test]
    fn a_path_under_a_dangling_symlink_is_not_absent() {
        let base = std::env::temp_dir().join(format!("copia-93-dangle-anc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::os::unix::fs::symlink("nowhere", base.join("dangling")).unwrap();
        assert!(
            !is_absent(&base.join("dangling/child")),
            "a path under a dangling symlink was read as absent"
        );
        assert!(is_absent(&base.join("missing/child")));
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 6, local side: ENAMETOOLONG is not NotFound, so a local path with an
    /// over-long component is never classified absent. (Already true; kept as a
    /// regression guard.)
    #[cfg(unix)]
    #[test]
    fn an_enametoolong_local_path_is_not_absent() {
        let base = std::env::temp_dir().join(format!("copia-93-long-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let long = base.join("a".repeat(300));
        assert!(!is_absent(&long), "ENAMETOOLONG was read as absent");
        std::fs::remove_dir_all(&base).ok();
    }
}
