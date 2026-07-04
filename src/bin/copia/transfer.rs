//! Low-level transfer primitives + helpers shared by the sync orchestration.
//! Split out of `dir_sync` to keep each module within the size budget.

use std::path::{Path, PathBuf};
use tracing::instrument;

/// Discover all files under a directory recursively, returning paths relative to root.
pub fn discover_local_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let entries = std::fs::read_dir(&dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && !path.is_symlink() {
                dirs.push(path);
            } else if path.is_file() {
                let rel = path.strip_prefix(root)?.to_path_buf();
                files.push(rel);
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Collect unique directory paths from a list of relative file paths.
pub fn collect_dirs(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = std::collections::BTreeSet::new();
    for file in files {
        let mut cur = file.as_path();
        while let Some(parent) = cur.parent() {
            if parent.as_os_str().is_empty() {
                break;
            }
            dirs.insert(parent.to_path_buf());
            cur = parent;
        }
    }
    dirs.into_iter().collect()
}

/// Create directories on a remote host via SSH.
/// Pipes directory list through stdin to avoid shell quoting issues with
/// special characters (apostrophes, angle brackets, etc.) in directory names.
pub async fn create_remote_dirs(
    host: &str,
    remote_root: &str,
    dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write;
    use tokio::io::AsyncWriteExt;

    // Build a newline-delimited list of full paths
    let mut dir_list = format!("{remote_root}\n");
    for dir in dirs {
        // GH-23: writeln! to String is infallible in practice, but log if it fails
        if writeln!(dir_list, "{}/{}", remote_root, dir.display()).is_err() {
            eprintln!(
                "Warning: failed to format directory path: {}",
                dir.display()
            );
        }
    }

    // Pipe directory list via stdin, read line-by-line and mkdir each
    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg("xargs -d '\\n' mkdir -p")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to open SSH stdin".to_string())?;

    // Write directory list in chunks to avoid buffering issues
    for chunk in dir_list.as_bytes().chunks(64 * 1024) {
        stdin.write_all(chunk).await?;
    }
    drop(stdin);

    let result = child.wait_with_output().await?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!("Failed to create remote directories: {stderr}").into());
    }

    Ok(())
}

/// Transfer a single file from local to remote via SSH streaming.
/// Uses streaming I/O — does not read entire file into memory.
#[instrument(skip(local_path), fields(host, remote_path))]
pub async fn transfer_file_to_remote(
    local_path: &Path,
    host: &str,
    remote_path: &str,
) -> Result<u64, String> {
    use tokio::io::AsyncReadExt;

    let metadata = tokio::fs::metadata(local_path)
        .await
        .map_err(|e| format!("{}: {e}", local_path.display()))?;
    let file_size = metadata.len();

    // Use $'...' quoting with backslash escapes for paths with special chars
    let escaped = remote_path.replace('\\', "\\\\").replace('\'', "\\'");
    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!("cat > $'{escaped}'"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("ssh spawn: {e}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to open SSH stdin".to_string())?;

    // Stream file in chunks instead of reading all into memory
    let mut file = tokio::fs::File::open(local_path)
        .await
        .map_err(|e| format!("open {}: {e}", local_path.display()))?;
    let mut buf = vec![0u8; 256 * 1024]; // 256KB chunks
    loop {
        let n = file
            .read(&mut buf)
            .await
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        tokio::io::AsyncWriteExt::write_all(&mut stdin, &buf[..n])
            .await
            .map_err(|e| format!("write: {e}"))?;
    }
    drop(stdin);

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| format!("ssh wait: {e}"))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!("SSH failed for {}: {stderr}", local_path.display()));
    }

    Ok(file_size)
}

/// Compute the total size of files relative to a root directory.
pub fn compute_total_size(root: &Path, files: &[PathBuf]) -> u64 {
    let mut total: u64 = 0;
    let mut skipped = 0usize;
    for f in files {
        match std::fs::metadata(root.join(f)) {
            Ok(m) => total += m.len(),
            Err(_) => skipped += 1,
        }
    }
    // GH-23: Warn if metadata errors caused files to be excluded from total
    if skipped > 0 {
        eprintln!(
            "Warning: {skipped} file(s) excluded from size calculation (metadata unavailable)"
        );
    }
    total
}

/// Calculate transfer speed in bytes per second.
#[allow(clippy::cast_precision_loss)]
pub fn transfer_speed(bytes: u64, elapsed: std::time::Duration) -> u64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let speed = (bytes as f64 / secs).max(0.0) as u64;
        speed
    } else {
        0
    }
}

/// Wait for all spawned task handles, logging errors.
pub async fn join_handles(handles: Vec<tokio::task::JoinHandle<()>>) {
    for handle in handles {
        if let Err(e) = handle.await {
            eprintln!("Task error: {e}");
        }
    }
}

/// Format bytes into a human-readable string.
#[allow(clippy::cast_precision_loss)]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[(u64, &str, usize)] = &[
        (1024 * 1024 * 1024 * 1024, "TiB", 2),
        (1024 * 1024 * 1024, "GiB", 2),
        (1024 * 1024, "MiB", 1),
        (1024, "KiB", 1),
    ];

    for &(threshold, unit, precision) in UNITS {
        if bytes >= threshold {
            return format!(
                "{:.prec$} {unit}",
                bytes as f64 / threshold as f64,
                prec = precision
            );
        }
    }
    format!("{bytes} B")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod transfer_tests {
    use super::*;

    #[test]
    fn format_bytes_covers_all_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MiB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GiB");
        assert_eq!(format_bytes(1024u64.pow(4)), "1.00 TiB");
    }

    #[test]
    fn transfer_speed_rate_and_zero_guard() {
        use std::time::Duration;
        assert_eq!(transfer_speed(1000, Duration::from_secs(1)), 1000);
        assert_eq!(transfer_speed(2000, Duration::from_secs(2)), 1000);
        // zero elapsed must not divide-by-zero
        assert_eq!(transfer_speed(1000, Duration::from_secs(0)), 0);
    }

    #[test]
    fn collect_dirs_builds_unique_sorted_parent_set() {
        let files = vec![
            PathBuf::from("a.txt"),
            PathBuf::from("sub/b.txt"),
            PathBuf::from("sub/deep/c.txt"),
        ];
        let dirs = collect_dirs(&files);
        assert_eq!(
            dirs,
            vec![PathBuf::from("sub"), PathBuf::from("sub/deep")],
            "top-level file contributes no dir; nested parents are unique + sorted"
        );
        assert!(collect_dirs(&[PathBuf::from("flat")]).is_empty());
    }

    #[test]
    fn discover_and_total_size_over_a_temp_tree() {
        let tmp = std::env::temp_dir().join(format!("copia-disc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("a.txt"), b"12345").unwrap(); // 5 bytes
        std::fs::write(tmp.join("sub/b.txt"), b"678").unwrap(); // 3 bytes
        let files = discover_local_files(&tmp).unwrap();
        assert_eq!(
            files,
            vec![PathBuf::from("a.txt"), PathBuf::from("sub/b.txt")]
        );
        assert_eq!(compute_total_size(&tmp, &files), 8);
        std::fs::remove_dir_all(&tmp).ok();
    }
}
