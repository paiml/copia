//! Recursive directory sync orchestration (push local->remote, pull remote->local).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::instrument;

use super::transfer::{
    collect_dirs, compute_total_size, create_remote_dirs, discover_local_files, format_bytes,
    join_handles, transfer_file_to_remote, transfer_speed,
};
use super::FileLocation;
use copia::async_sync::AsyncCopiaSync;

#[instrument(skip(source, dest, block_size))]
pub async fn run_sync_recursive(
    source: FileLocation,
    dest: FileLocation,
    block_size: usize,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match (&source, &dest) {
        (FileLocation::Local(local_src), FileLocation::Remote { host, path }) => {
            // GH-17: Warn that --block-size is not yet used for remote recursive sync
            if block_size != 4096 {
                eprintln!("Warning: --block-size is not yet implemented for remote recursive sync. Using default.");
            }
            run_sync_dir(TransferDir::Push, host, path, local_src, jobs, verbose).await
        }
        (FileLocation::Local(local_src), FileLocation::Local(local_dest)) => {
            run_sync_dir_local_to_local(local_src, local_dest, block_size, jobs, verbose).await
        }
        (FileLocation::Remote { host, path }, FileLocation::Local(local_dest)) => {
            // GH-17: --block-size not yet used for remote recursive sync.
            if block_size != 4096 {
                eprintln!("Warning: --block-size is not yet implemented for remote recursive sync. Using default.");
            }
            run_sync_dir(TransferDir::Pull, host, path, local_dest, jobs, verbose).await
        }
        _ => Err("Recursive sync supports local->remote, local->local, and remote->local (not remote->remote)".into()),
    }
}

/// Discover all files under a remote directory tree via SSH `find`.
/// Returns paths relative to `remote_root`, NUL-delimited so filenames with
/// spaces/newlines are handled. Mirror of `discover_local_files` over SSH.
async fn discover_remote_files(
    host: &str,
    remote_root: &str,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let escaped = remote_root.replace('\\', "\\\\").replace('\'', "\\'");
    let output = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!("cd $'{escaped}' && find . -type f -print0"))
        .output()
        .await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(
            format!("Failed to list remote files under {host}:{remote_root}: {stderr}").into(),
        );
    }
    Ok(parse_remote_find_output(&output.stdout))
}

/// Parse `find . -type f -print0` output (NUL-delimited, "./"-prefixed) into
/// sorted relative paths. Factored out for unit testing.
fn parse_remote_find_output(stdout: &[u8]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in stdout.split(|&b| b == 0) {
        if entry.is_empty() {
            continue;
        }
        let s = String::from_utf8_lossy(entry);
        // `find .` prints "./rel/path" — strip the leading "./".
        let rel = s.strip_prefix("./").unwrap_or(&s);
        if !rel.is_empty() {
            files.push(PathBuf::from(rel));
        }
    }
    files.sort();
    files
}

/// Create the local directory structure for a pulled tree.
fn create_local_dirs(
    local_root: &Path,
    dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(local_root)?;
    for dir in dirs {
        std::fs::create_dir_all(local_root.join(dir))?;
    }
    Ok(())
}

/// Transfer a single file remote->local via SSH streaming.
/// Streams `ssh host "cat file"` straight to disk — never buffers the whole
/// file in memory (the RAG index.json is >1 GiB). Returns bytes written.
async fn transfer_file_from_remote(
    host: &str,
    remote_path: &str,
    local_path: &Path,
) -> Result<u64, String> {
    use tokio::io::AsyncWriteExt;

    let escaped = remote_path.replace('\\', "\\\\").replace('\'', "\\'");
    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!("cat $'{escaped}'"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("ssh spawn: {e}"))?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to open SSH stdout".to_string())?;
    let mut file = tokio::fs::File::create(local_path)
        .await
        .map_err(|e| format!("create {}: {e}", local_path.display()))?;
    let written = tokio::io::copy(&mut stdout, &mut file)
        .await
        .map_err(|e| format!("stream {}: {e}", local_path.display()))?;
    file.flush().await.map_err(|e| format!("flush: {e}"))?;
    drop(stdout);

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| format!("ssh wait: {e}"))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!("SSH failed for {remote_path}: {stderr}"));
    }
    Ok(written)
}

/// Direction of a recursive directory transfer.
#[derive(Clone, Copy, Debug)]
enum TransferDir {
    /// local -> remote (push)
    Push,
    /// remote -> local (pull)
    Pull,
}

/// Spawn parallel per-file transfers for a directory tree, in either direction.
/// Both directions compute the same local/remote paths and share identical
/// progress/error handling — only the per-file transfer call differs.
fn spawn_dir_transfers(
    files: &[PathBuf],
    dir: TransferDir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    semaphore: &Arc<Semaphore>,
    progress: &TransferProgress,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(files.len());

    for rel_path in files {
        let remote_file = format!("{}/{}", remote_root, rel_path.display());
        let local_file = local_root.join(rel_path);
        let host = host.to_string();
        let sem = Arc::clone(semaphore);
        let prog = progress.clone();
        let rel_display = rel_path.display().to_string();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await;
            match transfer_by_dir(dir, &host, &remote_file, &local_file).await {
                Ok(size) => prog.record_ok(size),
                Err(e) => prog.record_err(&rel_display, &e),
            }
        });
        handles.push(handle);
    }
    handles
}

/// Recursively sync a directory tree in either direction: push (local->remote)
/// or pull (remote->local). The direction selects discovery, destination-dir
/// creation, and the per-file transfer; discover/collect/spawn/report are shared.
#[allow(clippy::cast_possible_truncation)]
#[instrument(skip(local_root), fields(host, remote_root))]
async fn run_sync_dir(
    dir: TransferDir,
    host: &str,
    remote_root: &str,
    local_root: &Path,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let (src_desc, dst_desc) = match dir {
        TransferDir::Push => (
            local_root.display().to_string(),
            format!("{host}:{remote_root}"),
        ),
        TransferDir::Pull => (
            format!("{host}:{remote_root}"),
            local_root.display().to_string(),
        ),
    };

    eprintln!("Discovering files in {src_desc}...");
    let files = match dir {
        TransferDir::Push => discover_local_files(local_root)?,
        TransferDir::Pull => discover_remote_files(host, remote_root).await?,
    };
    if files.is_empty() {
        eprintln!("No files found.");
        return Ok(());
    }

    let dirs = collect_dirs(&files);
    match dir {
        // Only the local side knows file sizes up front.
        TransferDir::Push => eprintln!(
            "Found {} files ({}) across {} directories",
            files.len(),
            format_bytes(compute_total_size(local_root, &files)),
            dirs.len()
        ),
        TransferDir::Pull => {
            eprintln!(
                "Found {} files across {} directories",
                files.len(),
                dirs.len()
            );
        }
    }

    match dir {
        TransferDir::Push => {
            eprintln!("Creating remote directory structure...");
            create_remote_dirs(host, remote_root, &dirs).await?;
        }
        TransferDir::Pull => {
            eprintln!("Creating local directory structure...");
            create_local_dirs(local_root, &dirs)?;
        }
    }

    eprintln!("Transferring with {jobs} parallel jobs...");
    let semaphore = Arc::new(Semaphore::new(jobs));
    let progress = TransferProgress::new(files.len() as u64);

    let handles = spawn_dir_transfers(
        &files,
        dir,
        host,
        remote_root,
        local_root,
        &semaphore,
        &progress,
    );
    join_handles(handles).await;

    report_parallel_transfer(start, &progress, &src_desc, &dst_desc, jobs, verbose)
}

/// Shared completion report + result for a parallel directory transfer
/// (both push and pull). Deduplicates the identical tail of the
/// local->remote and remote->local paths.
fn report_parallel_transfer(
    start: Instant,
    progress: &TransferProgress,
    src_desc: &str,
    dst_desc: &str,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let elapsed = start.elapsed();
    let total_tx = progress.bytes_transferred.load(Ordering::Relaxed);
    let done_count = progress.files_done.load(Ordering::Relaxed);
    let fail_count = progress.files_failed.load(Ordering::Relaxed);

    println!("\nComplete: {done_count} files synced, {fail_count} failed");
    println!(
        "Transferred {} in {:.1}s ({}/s)",
        format_bytes(total_tx),
        elapsed.as_secs_f64(),
        format_bytes(transfer_speed(total_tx, elapsed))
    );

    if verbose {
        eprintln!("Source: {src_desc}");
        eprintln!("Destination: {dst_desc}");
        eprintln!("Jobs: {jobs}");
    }

    if fail_count > 0 {
        Err(format!("{fail_count} files failed to transfer").into())
    } else {
        Ok(())
    }
}

/// Shared progress counters for parallel transfer operations.
#[derive(Clone)]
struct TransferProgress {
    bytes_transferred: Arc<AtomicU64>,
    files_done: Arc<AtomicU64>,
    files_failed: Arc<AtomicU64>,
    total_files: u64,
}

impl TransferProgress {
    fn new(total_files: u64) -> Self {
        Self {
            bytes_transferred: Arc::new(AtomicU64::new(0)),
            files_done: Arc::new(AtomicU64::new(0)),
            files_failed: Arc::new(AtomicU64::new(0)),
            total_files,
        }
    }

    /// Record a successful transfer + emit a periodic progress line.
    fn record_ok(&self, size: u64) {
        self.bytes_transferred.fetch_add(size, Ordering::Relaxed);
        let n = self.files_done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 50 == 0 || n == self.total_files {
            let transferred = self.bytes_transferred.load(Ordering::Relaxed);
            eprintln!(
                "  [{n}/{}] {} transferred",
                self.total_files,
                format_bytes(transferred)
            );
        }
    }

    /// Record a failed transfer.
    fn record_err(&self, rel_display: &str, err: &str) {
        self.files_failed.fetch_add(1, Ordering::Relaxed);
        eprintln!("  FAILED {rel_display}: {err}");
    }
}

/// Transfer one file in the given direction (push local->remote, pull remote->local).
async fn transfer_by_dir(
    dir: TransferDir,
    host: &str,
    remote_file: &str,
    local_file: &Path,
) -> Result<u64, String> {
    match dir {
        TransferDir::Push => transfer_file_to_remote(local_file, host, remote_file).await,
        TransferDir::Pull => transfer_file_from_remote(host, remote_file, local_file).await,
    }
}

#[allow(clippy::cast_possible_truncation)]
#[instrument(skip(local_src, local_dest))]
async fn run_sync_dir_local_to_local(
    local_src: &Path,
    local_dest: &Path,
    block_size: usize,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    eprintln!("Discovering files in {}...", local_src.display());
    let files = discover_local_files(local_src)?;
    if files.is_empty() {
        eprintln!("No files found.");
        return Ok(());
    }

    let total_size = compute_total_size(local_src, &files);
    eprintln!("Found {} files ({})", files.len(), format_bytes(total_size));

    // Create directory structure
    std::fs::create_dir_all(local_dest)?;
    for dir in collect_dirs(&files) {
        std::fs::create_dir_all(local_dest.join(dir))?;
    }

    // GH-17: Use user-provided block_size instead of hardcoded 2048
    let sync = Arc::new(AsyncCopiaSync::with_block_size(block_size));
    let semaphore = Arc::new(Semaphore::new(jobs));
    let progress = TransferProgress::new(files.len() as u64);

    let handles =
        spawn_local_transfers(&files, local_src, local_dest, &sync, &semaphore, &progress);

    join_handles(handles).await;

    let elapsed = start.elapsed();
    let total_tx = progress.bytes_transferred.load(Ordering::Relaxed);

    println!(
        "\nComplete: {} files synced",
        progress.files_done.load(Ordering::Relaxed)
    );
    println!(
        "Transferred {} in {:.1}s ({}/s)",
        format_bytes(total_tx),
        elapsed.as_secs_f64(),
        format_bytes(transfer_speed(total_tx, elapsed))
    );

    if verbose {
        eprintln!("Source: {}", local_src.display());
        eprintln!("Destination: {}", local_dest.display());
    }

    Ok(())
}

/// Spawn parallel local sync tasks for each file.
fn spawn_local_transfers(
    files: &[PathBuf],
    local_src: &Path,
    local_dest: &Path,
    sync: &Arc<AsyncCopiaSync>,
    semaphore: &Arc<Semaphore>,
    progress: &TransferProgress,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(files.len());
    let total_files = progress.total_files;

    for rel_path in files {
        let src_file = local_src.join(rel_path);
        let dst_file = local_dest.join(rel_path);
        let sem = Arc::clone(semaphore);
        let bytes_tx = Arc::clone(&progress.bytes_transferred);
        let done = Arc::clone(&progress.files_done);
        let sync = Arc::clone(sync);
        let rel_display = rel_path.display().to_string();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await;
            match sync.sync_files(&src_file, &dst_file).await {
                Ok(result) => {
                    bytes_tx.fetch_add(result.source_size, Ordering::Relaxed);
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 100 == 0 || n == total_files {
                        eprintln!("  [{n}/{total_files}] files synced");
                    }
                }
                Err(e) => {
                    eprintln!("  FAILED {rel_display}: {e}");
                }
            }
        });
        handles.push(handle);
    }

    handles
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod remote_pull_tests {
    use super::*;

    #[test]
    fn parse_find_strips_dot_slash_and_sorts() {
        // `find . -type f -print0` output: NUL-delimited, "./"-prefixed.
        let out = b"./a.txt\0./sub/b.txt\0./sub/deep/big.bin\0";
        let got = parse_remote_find_output(out);
        assert_eq!(
            got,
            vec![
                PathBuf::from("a.txt"),
                PathBuf::from("sub/b.txt"),
                PathBuf::from("sub/deep/big.bin"),
            ],
            "must strip ./ and return sorted relative paths"
        );
    }

    #[test]
    fn parse_find_handles_empty_and_spaces() {
        // Trailing NUL → empty final entry (must be skipped); spaces preserved.
        let out = b"./with space.txt\0";
        assert_eq!(
            parse_remote_find_output(out),
            vec![PathBuf::from("with space.txt")]
        );
        assert!(
            parse_remote_find_output(b"").is_empty(),
            "empty listing → no files"
        );
        assert!(
            parse_remote_find_output(b"\0\0").is_empty(),
            "only NULs → no files"
        );
    }

    #[test]
    fn create_local_dirs_builds_nested_tree() {
        let tmp = std::env::temp_dir().join(format!("copia-pull-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dirs = vec![PathBuf::from("sub"), PathBuf::from("sub/deep")];
        create_local_dirs(&tmp, &dirs).expect("create dirs");
        assert!(tmp.join("sub/deep").is_dir(), "nested dir must exist");
        // Idempotent: a second call must not fail.
        create_local_dirs(&tmp, &dirs).expect("idempotent re-create");
        std::fs::remove_dir_all(&tmp).ok();
    }
}
