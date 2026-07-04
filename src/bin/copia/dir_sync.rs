//! Low-level directory-transfer primitives shared by the incremental sync
//! orchestrator (`incremental.rs`): remote->local streaming, local dir creation,
//! and the parallel-transfer progress counters. The orchestration itself (quick
//! check, planning, atomic delivery, delete, dry-run) lives in `incremental.rs`.

use super::transfer::format_bytes;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::instrument;

/// Create the local directory structure for a synced tree (idempotent).
pub fn create_local_dirs(
    local_root: &Path,
    dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(local_root)?;
    for dir in dirs {
        std::fs::create_dir_all(local_root.join(dir))?;
    }
    Ok(())
}

/// Stream `ssh host "cat remote_path"` straight to `local_path` — never buffers
/// the whole file (the RAG index.json is >1 GiB). The caller is responsible for
/// atomic delivery (transfer to a temp path, then rename). Returns bytes written.
#[instrument(skip(local_path), fields(host, remote_path))]
pub async fn transfer_file_from_remote(
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

/// Shared progress counters for parallel transfers (push/pull/local).
#[derive(Clone)]
pub struct TransferProgress {
    bytes_transferred: Arc<AtomicU64>,
    files_done: Arc<AtomicU64>,
    files_failed: Arc<AtomicU64>,
    total_files: u64,
}

impl TransferProgress {
    pub fn new(total_files: u64) -> Self {
        Self {
            bytes_transferred: Arc::new(AtomicU64::new(0)),
            files_done: Arc::new(AtomicU64::new(0)),
            files_failed: Arc::new(AtomicU64::new(0)),
            total_files,
        }
    }

    /// Record a successful transfer + emit a periodic progress line.
    pub fn record_ok(&self, size: u64) {
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
    pub fn record_err(&self, rel_display: &str, err: &str) {
        self.files_failed.fetch_add(1, Ordering::Relaxed);
        eprintln!("  FAILED {rel_display}: {err}");
    }

    pub fn bytes(&self) -> u64 {
        self.bytes_transferred.load(Ordering::Relaxed)
    }
    pub fn done(&self) -> u64 {
        self.files_done.load(Ordering::Relaxed)
    }
    pub fn failed(&self) -> u64 {
        self.files_failed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn create_local_dirs_builds_nested_tree_idempotently() {
        let tmp = std::env::temp_dir().join(format!("copia-pull-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dirs = vec![PathBuf::from("sub"), PathBuf::from("sub/deep")];
        create_local_dirs(&tmp, &dirs).expect("create dirs");
        assert!(tmp.join("sub/deep").is_dir(), "nested dir must exist");
        create_local_dirs(&tmp, &dirs).expect("idempotent re-create");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn progress_counters_track_ok_err_and_bytes() {
        let p = TransferProgress::new(3);
        p.record_ok(100);
        p.record_ok(50);
        p.record_err("bad", "boom");
        assert_eq!(p.done(), 2);
        assert_eq!(p.failed(), 1);
        assert_eq!(p.bytes(), 150);
    }
}
