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
            // ORDER MATTERS: a symlink is an ENTRY, never a door to walk
            // through, so it is classified before is_dir/is_file — both of
            // which FOLLOW links.
            //
            // The previous order lost data twice over. `is_file()` follows, so
            // a symlink-to-file was copied as its target's bytes and the link
            // was gone. Worse, a symlink-to-DIRECTORY matched `is_dir()` but
            // was excluded by `!is_symlink()`, and then failed `is_file()` —
            // so it was dropped from the walk entirely. Measured 2026-08-22:
            // `copia sync -r` silently discarded `dirlink -> realdir`, and
            // because `copia verify` shares this walk the path was absent from
            // BOTH scans and the trees compared IDENTICAL. A verifier that
            // authorises deleting a source reported success over data it had
            // just lost.
            if path.is_symlink() {
                let rel = path.strip_prefix(root)?.to_path_buf();
                files.push(rel);
            } else if path.is_dir() {
                dirs.push(path);
            } else {
                // EVERYTHING else, not just is_file(). A filesystem has seven
                // entry kinds and this arm used to admit one of them, so a FIFO,
                // socket or device was in neither list and left the walk — and
                // `copia verify`, built on this walk, then called a tree that had
                // lost one IDENTICAL. Recording it may mean the transfer fails
                // loudly on an entry copia cannot carry; that is the correct
                // outcome and strictly better than a silent omission.
                let rel = path.strip_prefix(root)?.to_path_buf();
                files.push(rel);
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Every directory under `root`, relative — INCLUDING ones that hold no files.
///
/// `collect_dirs` derives directories from file paths, so a directory with no
/// files in it does not exist as far as the sync is concerned and is never
/// created at the destination. rsync -a preserves it. Caught by the rsync
/// differential harness on the `empty-dir` shape.
///
/// Structure is part of what an ARCHIVE is: a course tree whose empty chapter
/// directories vanish on the way to the NAS has not been faithfully archived,
/// even though every byte arrived.
pub fn discover_local_dirs(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            // Never walk THROUGH a symlink: it is an entry, and following one
            // can also loop forever.
            if path.is_symlink() {
                continue;
            }
            if path.is_dir() {
                out.push(path.strip_prefix(root)?.to_path_buf());
                stack.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
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
/// Stream a local file to `host:remote_path` over SSH. Atomic + optionally
/// mtime-preserving: the bytes land in a `.copia-tmp` sibling and are renamed
/// into place (`mv -f`) only after a clean transfer, so an interrupted push
/// never leaves a truncated/corrupt destination a reader could observe. When
/// `mtime` is Some, the destination's mtime is set to it (epoch seconds) so the
/// next run's quick check can skip the unchanged file.
#[instrument(skip(local_path), fields(host, remote_path))]
pub async fn transfer_file_to_remote(
    local_path: &Path,
    host: &str,
    remote_path: &str,
    mtime: Option<i64>,
) -> Result<u64, String> {
    use tokio::io::AsyncReadExt;

    let metadata = tokio::fs::metadata(local_path)
        .await
        .map_err(|e| format!("{}: {e}", local_path.display()))?;
    let file_size = metadata.len();

    // Use $'...' quoting with backslash escapes for paths with special chars.
    let escaped = remote_path.replace('\\', "\\\\").replace('\'', "\\'");
    let tmp_escaped = format!("{escaped}.copia-tmp");
    let touch = mtime.map_or(String::new(), |t| format!(" && touch -d @{t} $'{escaped}'"));
    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!(
            "cat > $'{tmp_escaped}' && mv -f $'{tmp_escaped}' $'{escaped}'{touch}"
        ))
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
    fn discover_local_dirs_finds_directories_with_no_files() {
        let tmp = std::env::temp_dir().join(format!("copia-dld-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("a/b/c")).unwrap();
        std::fs::create_dir_all(tmp.join("hollow")).unwrap();
        std::fs::write(tmp.join("a/f.txt"), b"x").unwrap();

        let dirs = discover_local_dirs(&tmp).unwrap();
        // `hollow` holds no files, so collect_dirs cannot see it at all — that
        // is the empty-directory loss this function exists to prevent.
        assert!(
            dirs.contains(&PathBuf::from("hollow")),
            "an empty directory was not discovered: {dirs:?}"
        );
        assert!(dirs.contains(&PathBuf::from("a/b/c")), "{dirs:?}");
        assert!(
            !collect_dirs(&[PathBuf::from("a/f.txt")]).contains(&PathBuf::from("hollow")),
            "fixture is wrong: collect_dirs must NOT find the empty dir, or this \
             test is not measuring the difference"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[cfg(unix)]
    fn discover_local_dirs_does_not_walk_through_a_symlink() {
        let tmp = std::env::temp_dir().join(format!("copia-dld2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("real/inner")).unwrap();
        std::os::unix::fs::symlink("real", tmp.join("link")).unwrap();

        let dirs = discover_local_dirs(&tmp).unwrap();
        assert!(dirs.contains(&PathBuf::from("real")), "{dirs:?}");
        assert!(
            !dirs.iter().any(|d| d.starts_with("link")),
            "walked THROUGH a symlink — that duplicates the tree and can loop \
             forever on a cycle: {dirs:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// EXHAUSTIVE over entry kinds, not over the kinds someone remembered.
    ///
    /// The symlink fix in #47 added `symlink` to a list and left FIFOs, sockets
    /// and devices dropped — so `copia sync -r` discarded a FIFO and
    /// `copia verify`, sharing this walk, reported "trees are identical — the
    /// source may safely be deleted". The defect was not fixed, it was moved to
    /// a shape nobody had enumerated.
    ///
    /// So this test does not list shapes. It creates one of EVERY entry kind a
    /// test can create without root and requires the walk to surface all of
    /// them. Block and character devices need root and are named in the
    /// assertion message rather than silently omitted — an untested kind that
    /// nobody mentions is how this bug happened twice.
    #[test]
    #[cfg(unix)]
    fn the_walk_surfaces_every_entry_kind_it_can_be_given() {
        use std::os::unix::net::UnixListener;
        // PID alone is not unique: a leftover from an earlier failed run in the
        // same process collides, and the failure (`AlreadyExists` on a symlink)
        // looks nothing like the thing under test.
        let tmp = std::env::temp_dir().join(format!(
            "copia-kinds-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // regular file
        std::fs::write(tmp.join("regular"), b"x").unwrap();
        // directory containing a file (a bare dir is not an ENTRY in the file list)
        std::fs::create_dir_all(tmp.join("dir")).unwrap();
        std::fs::write(tmp.join("dir/inner"), b"y").unwrap();
        // symlink to file, symlink to dir, dangling symlink
        std::os::unix::fs::symlink("regular", tmp.join("link-file")).unwrap();
        std::os::unix::fs::symlink("dir", tmp.join("link-dir")).unwrap();
        std::os::unix::fs::symlink("nowhere", tmp.join("link-dangling")).unwrap();
        // socket
        let sock = UnixListener::bind(tmp.join("sock")).ok();
        // FIFO — the kind that was silently dropped
        let fifo = tmp.join("fifo");
        let made_fifo = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        let files = discover_local_files(&tmp).unwrap();
        let has = |n: &str| files.contains(&PathBuf::from(n));

        let mut missing = Vec::new();
        for name in [
            "regular",
            "dir/inner",
            "link-file",
            "link-dir",
            "link-dangling",
        ] {
            if !has(name) {
                missing.push(name);
            }
        }
        if sock.is_some() && !has("sock") {
            missing.push("sock");
        }
        if made_fifo && !has("fifo") {
            missing.push("fifo");
        }

        assert!(
            missing.is_empty(),
            "the walk dropped {missing:?}. An entry it cannot REPRESENT must still be \
             VISIBLE — omission reads as agreement to every comparison built on this \
             walk, which is how a FIFO was lost and then certified as identical. \
             (Block and character devices need root and are not covered here.)"
        );
        assert!(
            !has("link-dir/inner"),
            "followed a symlinked directory and duplicated its contents: {files:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[cfg(unix)]
    fn the_walk_surfaces_symlinks_instead_of_following_or_dropping_them() {
        let tmp = std::env::temp_dir().join(format!("copia-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("realdir")).unwrap();
        std::fs::write(tmp.join("realdir/inside.txt"), b"x").unwrap();
        std::os::unix::fs::symlink("realdir", tmp.join("dirlink")).unwrap();
        std::os::unix::fs::symlink("nowhere", tmp.join("dangling")).unwrap();

        let files = discover_local_files(&tmp).unwrap();
        // Both were invisible before: a symlinked DIRECTORY matched is_dir() but
        // was excluded by !is_symlink() and then failed is_file(), so it left the
        // walk entirely — and `copia verify` shares this walk, so the trees
        // compared IDENTICAL over the loss.
        assert!(files.contains(&PathBuf::from("dirlink")), "{files:?}");
        assert!(files.contains(&PathBuf::from("dangling")), "{files:?}");
        assert!(
            files.contains(&PathBuf::from("realdir/inside.txt")),
            "{files:?}"
        );
        assert!(
            !files.contains(&PathBuf::from("dirlink/inside.txt")),
            "followed the symlink and duplicated its contents: {files:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
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
    fn discover_local_files_over_a_temp_tree() {
        let tmp = std::env::temp_dir().join(format!("copia-disc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("a.txt"), b"12345").unwrap();
        std::fs::write(tmp.join("sub/b.txt"), b"678").unwrap();
        let files = discover_local_files(&tmp).unwrap();
        assert_eq!(
            files,
            vec![PathBuf::from("a.txt"), PathBuf::from("sub/b.txt")]
        );
        std::fs::remove_dir_all(&tmp).ok();
    }
}
