//! Low-level transfer primitives + helpers shared by the sync orchestration.
//! Split out of `dir_sync` to keep each module within the size budget.

use super::meta::posix_quote;
use std::path::{Path, PathBuf};
use tracing::instrument;

/// Discover all files under a directory recursively, returning paths relative to root.
///
/// Fails on the first walk error. `copia sync` uses [`walk_local`], which records
/// the error as an omission and keeps walking.
pub fn discover_local_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut omitted = Vec::new();
    let files = walk_local(root, &mut omitted);
    match omitted.into_iter().next() {
        Some(first) => Err(first.into()),
        None => Ok(files),
    }
}

/// Walk `root` like [`discover_local_files`], but never drop an entry silently:
/// every directory or entry the walk could not read is pushed to `omitted` and
/// the rest of the tree is still returned (issue #50). The caller decides what an
/// omission means; `sync` treats any omission as a veto on `--delete`.
pub fn walk_local(root: &Path, omitted: &mut Vec<String>) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) => {
                omitted.push(format!("unreadable directory {}: {e}", dir.display()));
                continue;
            }
        };
        for entry in entries {
            let path = match entry {
                Ok(entry) => entry.path(),
                Err(e) => {
                    omitted.push(format!("unreadable entry in {}: {e}", dir.display()));
                    continue;
                }
            };
            let Ok(rel) = path.strip_prefix(root).map(Path::to_path_buf) else {
                omitted.push(format!("entry outside walk root: {}", path.display()));
                continue;
            };
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
                files.push(rel);
            }
        }
    }

    files.sort();
    files
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
/// Per-directory `mkdir -p`, run by `sh` with the remote root as `$1`. Every path
/// prefix strictly below the root is refused if it is a symlink (`mkdir -p` would
/// follow it and create outside the tree). A refusal exits before that path is
/// created, so nothing is written through the link.
const MKDIR_GUARDED_SH: &str = r#"root=$1; shift; for p; do s=$p; while [ "$s" != "$root" ] && [ -n "$s" ]; do if [ -L "$s" ]; then echo "refusing to create directories through symlink $s" >&2; exit 3; fi; case $s in */*) s=${s%/*};; *) break;; esac; done; mkdir -p -- "$p" || exit 1; done"#;

pub async fn create_remote_dirs(
    host: &str,
    remote_root: &str,
    dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write;
    use tokio::io::AsyncWriteExt;

    // NUL-terminated list of full paths. A newline is a legal file-name byte, so
    // a newline-delimited list split one directory into two wrong ones.
    let mut dir_list = format!("{remote_root}\0");
    for dir in dirs {
        // GH-23: write! to String is infallible in practice, but log if it fails
        if write!(dir_list, "{}/{}\0", remote_root, dir.display()).is_err() {
            eprintln!(
                "Warning: failed to format directory path: {}",
                dir.display()
            );
        }
    }

    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!(
            "xargs -0 sh -c {} sh {}",
            posix_quote(MKDIR_GUARDED_SH),
            posix_quote(remote_root)
        ))
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
    // `symlink_metadata`, not `metadata`: the latter FOLLOWS a link, so a symlink
    // to a regular file passed the regular-file test and its target's bytes were
    // sent under the link's name. A link is refused, loudly.
    let metadata = tokio::fs::symlink_metadata(local_path)
        .await
        .map_err(|e| format!("{}: {e}", local_path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "cannot carry symlink {}: copia sends bytes, not links",
            local_path.display()
        ));
    }
    // Refuse before `ssh` spawns or the file is opened: `File::open` on a FIFO,
    // socket or device blocks (or is meaningless), and copia cannot carry it.
    if !metadata.is_file() {
        return Err(format!(
            "cannot carry {}: not a regular file",
            local_path.display()
        ));
    }
    let file_size = metadata.len();

    // Open the source BEFORE `ssh` runs. `cat >` truncates the remote target as
    // soon as it starts, so a source that cannot be read (EACCES) must fail here,
    // while the remote file is still untouched.
    let file = tokio::fs::File::open(local_path)
        .await
        .map_err(|e| format!("open {}: {e}", local_path.display()))?;

    stream_to_remote(file, file_size, host, remote_path, mtime)
        .await
        .map_err(|e| match e {
            StreamError::Read(msg) | StreamError::Remote(msg) => {
                format!("{}: {msg}", local_path.display())
            }
        })
}

/// Why a stream to the remote failed. `Read` is a local read error; the remote
/// command was already told to abort, so nothing was committed.
#[derive(Debug)]
pub(crate) enum StreamError {
    Read(String),
    Remote(String),
}

/// Stream `reader` (exactly `size` bytes expected) to `host:remote_path`.
pub(crate) async fn stream_to_remote<R>(
    mut reader: R,
    size: u64,
    host: &str,
    remote_path: &str,
    mtime: Option<i64>,
) -> Result<u64, StreamError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;

    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(push_command(remote_path, size, mtime))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| StreamError::Remote(format!("ssh spawn: {e}")))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| StreamError::Remote("Failed to open SSH stdin".to_string()))?;

    // Stream in chunks instead of reading all into memory. A failure stops the
    // loop, but the child is still waited on, so the remote side has finished
    // deciding before this returns.
    let mut buf = vec![0u8; 256 * 1024]; // 256KB chunks
    let mut sent: u64 = 0;
    let mut local_err: Option<StreamError> = None;
    loop {
        let n = match reader.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                // Closing stdin after exactly `size` bytes would look complete to
                // the remote count check, so one extra byte forces a mismatch.
                if sent == size {
                    let _ = tokio::io::AsyncWriteExt::write_all(&mut stdin, b"\0").await;
                }
                local_err = Some(StreamError::Read(format!("read: {e}")));
                break;
            }
        };
        if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut stdin, &buf[..n]).await {
            local_err = Some(StreamError::Remote(format!("write: {e}")));
            break;
        }
        sent += n as u64;
    }
    drop(stdin);

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| StreamError::Remote(format!("ssh wait: {e}")))?;
    if let Some(e) = local_err {
        return Err(e);
    }
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(StreamError::Remote(format!("SSH failed: {stderr}")));
    }

    Ok(size)
}

/// The remote POSIX shell command that streams stdin into `remote_path`
/// atomically: `cat` into a `.copia-tmp` sibling, then `mv` over the target.
///
/// The `mv` runs only if the staged file holds exactly `expected` bytes, the size
/// recorded before the stream. A short stream (a failed local read) or a long one
/// (the file grew) is removed and never committed.
///
/// Plain POSIX quoting only (`'...'` with `'\''`), so it runs under dash as well
/// as bash. `--` ends the options of `mv`, `rm` and `touch`, so a path that begins
/// with `-` is a path.
pub(crate) fn push_command(remote_path: &str, expected: u64, mtime: Option<i64>) -> String {
    let tmp_q = posix_quote(&format!("{remote_path}.copia-tmp"));
    let dst_q = posix_quote(remote_path);
    // The staging name must not exist AT ALL. Anything there, including a regular
    // file, may be synced data with that name, so it is refused, never overwritten.
    // `-e` follows links, so `-L` is checked too: a dangling link passes `! -e`.
    let staging_check = format!(
        "{{ [ ! -e {tmp_q} ] && [ ! -L {tmp_q} ]; }} || \
         {{ echo 'copia: staging path already exists; not overwriting it' >&2; exit 1; }}; "
    );
    let touch = mtime.map_or(String::new(), |t| format!(" && touch -d @{t} -- {dst_q}"));
    let verify = format!(
        "[ \"$(wc -c < {tmp_q} | tr -d ' ')\" = \"{expected}\" ] || \
         {{ rm -f -- {tmp_q}; echo 'copia: received bytes differ from the source size; not committed' >&2; exit 1; }}"
    );
    // `mv` into a directory moves the staged file INTO it, and `mv` onto a symlink
    // to a directory writes through the link. Both are refused, before the rename.
    let dst_check = format!(
        "[ ! -d {dst_q} ] && [ ! -L {dst_q} ] || \
         {{ rm -f -- {tmp_q}; echo 'copia: destination is a directory or a symlink; not committed' >&2; exit 1; }}"
    );
    let ancestors = ancestor_guard(remote_path);
    format!(
        "{ancestors}{staging_check}cat > {tmp_q}; rc=$?; if [ $rc -ne 0 ]; then rm -f -- {tmp_q}; exit $rc; fi; {{ {verify}; }} && {{ {dst_check}; }} && mv -f -- {tmp_q} {dst_q}{touch}"
    )
}

/// A remote guard that refuses a path whose DIRECTORY ancestors include a symlink
/// (or that does not resolve). Staging through a symlinked ancestor writes outside
/// the synced tree. Each ancestor is entered with `cd` only after `-L` rules out a
/// link, so every component appears once and the guard is linear in the path.
fn ancestor_guard(remote_path: &str) -> String {
    let mut parts: Vec<&str> = remote_path.split('/').filter(|c| !c.is_empty()).collect();
    parts.pop(); // the final name is written, not entered
    if parts.is_empty() {
        return String::new();
    }
    let base = if remote_path.starts_with('/') {
        "/"
    } else {
        "."
    };
    let mut checks = format!("cd -- {} || exit 1; ", posix_quote(base));
    for part in parts {
        let q = posix_quote(&format!("./{part}"));
        checks.push_str(&format!("[ ! -L {q} ] && cd -- {q} || exit 1; "));
    }
    checks.push_str("exit 0; ");
    format!(
        "( {checks}) || {{ echo 'copia: staging path runs through a symlink or a missing directory; not written' >&2; exit 1; }}; "
    )
}

/// The remote POSIX shell command that prints `remote_path`'s bytes.
///
/// A symlink is refused, never followed. The listing already omits links, but a
/// path can be replaced by a link between that listing and this read, and `cat`
/// would then print the target's bytes under the link's name.
pub(crate) fn pull_command(remote_path: &str) -> String {
    let q = posix_quote(remote_path);
    // `-f` is false for a FIFO, socket or device, so those are refused by a stat
    // alone, before `cat` could open and block on them. `-f` follows a link, so
    // `! -L` refuses a symlink to a regular file.
    format!(
        "{{ [ -f {q} ] && [ ! -L {q} ]; }} || {{ echo 'copia: source is not a regular file' >&2; exit 1; }}; cat -- {q}"
    )
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

/// Review regressions for copia #50 round 5: the remote commands must run under
/// a POSIX shell (dash), and a dangling staging link must not be written through.
#[cfg(test)]
#[cfg(unix)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod r93_transfer {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// dash where it exists (Debian and Ubuntu remotes run it as `sh`), else sh.
    fn posix_sh() -> &'static str {
        if Path::new("/bin/dash").exists() {
            "/bin/dash"
        } else {
            "/bin/sh"
        }
    }

    /// Run a remote command under `shell` in `cwd`, feeding `stdin`.
    fn run_in(shell: &str, command: &str, cwd: &Path, stdin: &[u8]) -> std::process::Output {
        let mut child = Command::new(shell)
            .arg("-c")
            .arg(command)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn posix sh");
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(stdin)
            .expect("write stdin");
        child.wait_with_output().expect("wait")
    }

    /// dash: the shell that must run the remote commands on Debian/Ubuntu hosts.
    fn run_posix(command: &str, cwd: &Path, stdin: &[u8]) -> std::process::Output {
        run_in(posix_sh(), command, cwd, stdin)
    }

    /// bash: where the pre-fix `$'...'` form WORKS, so a test on it fails for the
    /// defect itself and not because the shell rejected the syntax.
    fn run_bash(command: &str, cwd: &Path, stdin: &[u8]) -> std::process::Output {
        run_in("/bin/bash", command, cwd, stdin)
    }

    fn scratch(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("copia-93-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    /// Finding 4 RED: a name with a single quote and a newline must round-trip
    /// under dash. `$'...'` is bash-only, so dash writes a literal `$'` path.
    #[test]
    fn push_and_pull_round_trip_a_quote_and_newline_name_under_dash() {
        let base = scratch("posix-roundtrip");
        let name = "it's\nname.txt";
        let out = run_posix(
            &push_command(name, 7, Some(1_600_000_000)),
            &base,
            b"payload",
        );
        assert!(
            out.status.success(),
            "push under dash failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(std::fs::read(base.join(name)).unwrap(), b"payload");

        let out = run_posix(&pull_command(name), &base, b"");
        assert!(
            out.status.success(),
            "pull under dash failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, b"payload");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 1 RED: a DANGLING symlink at the staging name passes `! -e`, and
    /// `cat >` then writes through it, creating the link's target. It must be refused.
    #[test]
    fn push_refuses_a_dangling_staging_symlink_without_writing_through_it() {
        let base = scratch("dangling-stage");
        std::os::unix::fs::symlink("nowhere", base.join("a.txt.copia-tmp")).unwrap();
        let out = run_bash(&push_command("a.txt", 7, None), &base, b"payload");
        let created = base.join("nowhere").exists();
        assert!(
            !out.status.success(),
            "a dangling staging symlink was accepted"
        );
        assert!(!created, "the write followed the staging symlink");
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 3 (remote side): a symlink source must not be streamed as its
    /// target's bytes, even when the listing raced with a replacement.
    #[test]
    fn pull_refuses_a_symlink_source_instead_of_printing_the_target() {
        let base = scratch("pull-link");
        std::fs::write(base.join("real.txt"), b"secret").unwrap();
        std::os::unix::fs::symlink("real.txt", base.join("link")).unwrap();
        let out = run_bash(&pull_command("link"), &base, b"");
        assert!(
            !out.status.success() || out.stdout.is_empty(),
            "a symlink's target was printed: {:?}",
            out.stdout
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Finding 5: a leading-dash path must reach `mv`/`touch` after `--`, or it is
    /// parsed as an option and the push fails.
    #[test]
    fn push_survives_a_leading_dash_path() {
        let base = scratch("dash-path");
        std::fs::create_dir_all(base.join("-root")).unwrap();
        let out = run_bash(
            &push_command("-root/a.txt", 7, Some(1_600_000_000)),
            &base,
            b"payload",
        );
        assert!(
            out.status.success(),
            "a leading-dash path was parsed as an option: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(std::fs::read(base.join("-root/a.txt")).unwrap(), b"payload");
        std::fs::remove_dir_all(&base).ok();
    }
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

#[cfg(test)]
mod r93_round3 {
    use super::*;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    fn posix_sh() -> &'static str {
        if Path::new("/bin/dash").exists() {
            "/bin/dash"
        } else {
            "/bin/sh"
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("copia-r3-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    fn push_in(base: &Path, remote: &str, size: u64) -> std::process::Output {
        let mut child = Command::new(posix_sh())
            .arg("-c")
            .arg(push_command(remote, size, None))
            .current_dir(base)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        use std::io::Write;
        child.stdin.take().unwrap().write_all(b"payload").unwrap();
        child.wait_with_output().unwrap()
    }

    fn entries(dir: &Path) -> usize {
        std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
    }

    /// Round 3, item 1 RED: `mv -f` into a symlink-to-directory moves the staged
    /// file OUTSIDE the synced tree, with no veto.
    #[cfg(unix)]
    #[test]
    fn push_refuses_a_symlink_destination_and_writes_nothing_outside() {
        let base = scratch("link-dst");
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, base.join("link")).unwrap();
        let out = push_in(&base, "link", 7);
        assert!(!out.status.success(), "a symlink destination was accepted");
        assert_eq!(
            entries(&outside),
            0,
            "a file landed outside the synced tree"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Round 3, item 1 RED: `mv -f` into a directory moves the file INTO it.
    #[cfg(unix)]
    #[test]
    fn push_refuses_a_directory_destination() {
        let base = scratch("dir-dst");
        std::fs::create_dir_all(base.join("dir")).unwrap();
        let out = push_in(&base, "dir", 7);
        assert!(
            !out.status.success(),
            "a directory destination was accepted"
        );
        assert_eq!(
            entries(&base.join("dir")),
            0,
            "the file was moved into the directory"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Round 3, item 2 RED: staging through an ancestor symlink writes outside.
    #[cfg(unix)]
    #[test]
    fn push_refuses_a_symlink_ancestor_and_writes_nothing_outside() {
        let base = scratch("anc-link");
        let outside = base.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, base.join("link")).unwrap();
        let out = push_in(&base, "link/a.txt", 7);
        assert!(!out.status.success(), "a symlink ancestor was accepted");
        assert_eq!(
            entries(&outside),
            0,
            "the staged file was written through the link"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Round 3, item 5 RED: an existing regular file at the staging name is synced
    /// data; a push must refuse, not overwrite it.
    #[cfg(unix)]
    #[test]
    fn push_refuses_to_overwrite_an_existing_staging_file() {
        let base = scratch("stage-reuse");
        std::fs::write(base.join("a.txt.copia-tmp"), b"synced").unwrap();
        let out = push_in(&base, "a.txt", 7);
        assert!(
            !out.status.success(),
            "an existing staging file was accepted"
        );
        assert_eq!(
            std::fs::read(base.join("a.txt.copia-tmp")).unwrap(),
            b"synced",
            "the synced staging-named file was overwritten"
        );
        std::fs::remove_dir_all(&base).ok();
    }

    /// Round 3, item 4 RED: `cat` on a remote FIFO blocks until a writer appears.
    /// The source must be refused before it is read.
    #[cfg(unix)]
    #[test]
    fn pull_refuses_a_fifo_source_without_blocking() {
        let base = scratch("fifo-src");
        let fifo = base.join("pipe");
        assert!(Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        let mut child = Command::new(posix_sh())
            .arg("-c")
            .arg(pull_command("pipe"))
            .current_dir(&base)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let start = Instant::now();
        let status = loop {
            match child.try_wait().unwrap() {
                Some(st) => break st,
                None if start.elapsed() > Duration::from_secs(5) => {
                    child.kill().ok();
                    child.wait().ok();
                    std::fs::remove_dir_all(&base).ok();
                    panic!("pull hung on a FIFO source instead of refusing it");
                }
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        };
        assert!(!status.success(), "a FIFO source was accepted");
        std::fs::remove_dir_all(&base).ok();
    }

    /// B5 RED: a failed `cat` must not leave the staging file. Measured: `cat`
    /// exit 1 left a 4096-byte `.copia-tmp`, and the retry's staging check then
    /// refused. A directory as stdin makes `cat` create the staging file, then fail
    /// reading (EISDIR), which is the same failure with no timing dependence.
    #[test]
    fn push_removes_the_staging_file_when_cat_fails() {
        let base = scratch("push-cat-fails");
        let dir_stdin = std::fs::File::open(&base).unwrap();
        let out = Command::new(posix_sh())
            .arg("-c")
            .arg(push_command("p.bin", 0, None))
            .current_dir(&base)
            .stdin(Stdio::from(dir_stdin))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("spawn posix sh");
        assert!(!out.status.success(), "a failed cat must fail the push");
        assert!(
            !base.join("p.bin.copia-tmp").exists(),
            "a failed cat must not leave the staging file behind"
        );
        assert!(!base.join("p.bin").exists(), "nothing may be committed");
        std::fs::remove_dir_all(&base).ok();
    }
}
