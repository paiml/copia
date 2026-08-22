//! Filesystem + remote metadata discovery for the quick check, plus the mtime
//! preservation helpers that keep it stable across runs.

use super::plan::{FileMeta, MetaMap};
use super::reconcile::{FileType, Fingerprint, FpMap};
use super::transfer::discover_local_files;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

/// blake3-fingerprint one local path (streaming — never loads a >1GiB file into
/// memory). A symlink hashes its target string; a regular file hashes its bytes.
pub fn fingerprint_path(full: &Path) -> std::io::Result<Fingerprint> {
    let meta = std::fs::symlink_metadata(full)?;
    if meta.file_type().is_symlink() {
        let target = std::fs::read_link(full)?;
        let h = blake3::hash(target.as_os_str().as_encoded_bytes());
        Ok(Fingerprint {
            blake3: *h.as_bytes(),
            ftype: FileType::Symlink,
        })
    } else if meta.file_type().is_file() {
        let mut hasher = blake3::Hasher::new();
        let mut f = std::fs::File::open(full)?;
        std::io::copy(&mut f, &mut hasher)?;
        Ok(Fingerprint {
            blake3: *hasher.finalize().as_bytes(),
            ftype: FileType::File,
        })
    } else {
        // A FIFO, socket, or device. NEVER OPEN IT: opening a FIFO for reading
        // BLOCKS until a writer appears, which would hang the walk indefinitely
        // on an ordinary tree. That hazard is why these were skipped, and
        // skipping them is what made them invisible to every comparison.
        //
        // Fingerprint the KIND instead. It cannot detect a change in what flows
        // through a pipe — nothing can — but it makes the entry EXIST, so a
        // destination missing it reads as `missing` rather than as agreement.
        let tag = kind_tag(meta.file_type());
        Ok(Fingerprint {
            blake3: *blake3::hash(tag.as_bytes()).as_bytes(),
            ftype: FileType::Other,
        })
    }
}

/// A stable name for a non-regular, non-symlink entry kind.
fn kind_tag(ft: std::fs::FileType) -> &'static str {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        if ft.is_fifo() {
            return "fifo";
        }
        if ft.is_socket() {
            return "socket";
        }
        if ft.is_block_device() {
            return "block-device";
        }
        if ft.is_char_device() {
            return "char-device";
        }
    }
    "unknown"
}

/// Walk a local tree and content-fingerprint every file into an `FpMap`. Files
/// whose fingerprint can't be read (permission/race) are skipped, never guessed.
pub fn discover_local_fingerprints(root: &Path) -> Result<FpMap, Box<dyn std::error::Error>> {
    let mut out = FpMap::new();
    for rel in discover_local_files(root)? {
        if let Ok(fp) = fingerprint_path(&root.join(&rel)) {
            out.insert(rel, fp);
        }
    }
    Ok(out)
}

/// A std file mtime as whole epoch seconds (the quick-check granularity).
fn mtime_secs(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
}

/// Walk a local tree and stat every file into a `MetaMap` of relative paths.
pub fn discover_local_with_meta(root: &Path) -> Result<MetaMap, Box<dyn std::error::Error>> {
    let mut out = MetaMap::new();
    for rel in discover_local_files(root)? {
        // symlink_metadata, not metadata: `metadata` FOLLOWS the link, so a
        // DANGLING symlink errors and the `if let Ok` silently drops it from
        // the plan — copia then omits a path rsync -a faithfully recreates.
        // Caught by the rsync differential harness on the `dangling-symlink`
        // shape. A symlink's own size/mtime is the right quick-check value
        // anyway; `deliver_local` recreates it as a link regardless.
        if let Ok(meta) = std::fs::symlink_metadata(root.join(&rel)) {
            out.insert(
                rel,
                FileMeta {
                    size: meta.len(),
                    mtime: mtime_secs(&meta),
                },
            );
        }
    }
    Ok(out)
}

/// List a remote tree with size+mtime in one `find -printf` over SSH.
pub async fn discover_remote_with_meta(
    host: &str,
    remote_root: &str,
) -> Result<MetaMap, Box<dyn std::error::Error>> {
    let escaped = remote_root.replace('\\', "\\\\").replace('\'', "\\'");
    // %s size, %T@ mtime (float secs), %p path — TAB-separated, NUL-terminated.
    let output = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!(
            // `! -type d`, NOT `-type f`. The previous form enumerated regular
            // files ONLY, so a symlink, FIFO, socket or device on a remote tree
            // was invisible to every plan built from this listing — and a
            // destination missing one read as agreement, the same omission that
            // made `copia verify` certify a lost FIFO locally.
            //
            // Directories are excluded because they are handled separately;
            // everything else is an ENTRY and must be visible even when copia
            // cannot carry it. `find` does not follow symlinks by default, so
            // %s/%T@ describe the link itself rather than its target.
            "cd $'{escaped}' && find . ! -type d -printf '%s\\t%T@\\t%p\\0'"
        ))
        .output()
        .await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to list {host}:{remote_root}: {stderr}").into());
    }
    Ok(parse_remote_meta_output(&output.stdout))
}

/// Parse NUL-terminated `size\tmtime\t./path` records into a `MetaMap`.
pub fn parse_remote_meta_output(stdout: &[u8]) -> MetaMap {
    let mut out = MetaMap::new();
    for entry in stdout.split(|&b| b == 0) {
        if entry.is_empty() {
            continue;
        }
        let s = String::from_utf8_lossy(entry);
        let mut parts = s.splitn(3, '\t');
        let (Some(size), Some(mtime), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let Ok(size) = size.parse::<u64>() else {
            continue;
        };
        // %T@ is "secs.nanos" — truncate to whole seconds.
        let mtime = mtime
            .split('.')
            .next()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);
        let rel = path.strip_prefix("./").unwrap_or(path);
        if !rel.is_empty() {
            out.insert(PathBuf::from(rel), FileMeta { size, mtime });
        }
    }
    out
}

/// Set a local file's mtime to `secs` epoch seconds (best-effort).
pub fn set_local_mtime(path: &Path, secs: i64) -> std::io::Result<()> {
    let t = UNIX_EPOCH + Duration::from_secs(u64::try_from(secs.max(0)).unwrap_or(0));
    std::fs::File::options()
        .write(true)
        .open(path)?
        .set_modified(t)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_meta_reads_size_mtime_path_and_truncates_subsecond() {
        let raw = b"1024\t1700000000.500\t./a.txt\x00\t\t\x00512\t1699999999\t./sub/b.bin\0";
        let m = parse_remote_meta_output(raw);
        assert_eq!(m.len(), 2);
        assert_eq!(
            m[&PathBuf::from("a.txt")],
            FileMeta {
                size: 1024,
                mtime: 1_700_000_000
            }
        );
        assert_eq!(
            m[&PathBuf::from("sub/b.bin")],
            FileMeta {
                size: 512,
                mtime: 1_699_999_999
            }
        );
    }

    #[test]
    fn parse_meta_skips_malformed_records() {
        // missing fields / non-numeric size are dropped, not panicked on
        let m = parse_remote_meta_output(b"notanumber\t123\t./x\0\0only-one-field\0");
        assert!(m.is_empty());
    }

    #[test]
    fn discover_local_meta_and_set_mtime_roundtrip() {
        let base = std::env::temp_dir().join(format!("copia-meta-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("d")).unwrap();
        std::fs::write(base.join("d/f"), b"hello").unwrap();
        let m = discover_local_with_meta(&base).unwrap();
        assert_eq!(m[&PathBuf::from("d/f")].size, 5);
        // set + read back the mtime at 1s granularity
        set_local_mtime(&base.join("d/f"), 1_600_000_000).unwrap();
        let m2 = discover_local_with_meta(&base).unwrap();
        assert_eq!(m2[&PathBuf::from("d/f")].mtime, 1_600_000_000);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A FIFO must fingerprint as `Other` — and must NOT be opened.
    ///
    /// This is the test that would have caught the original defect. Opening a
    /// FIFO for reading BLOCKS until a writer appears, which is exactly why the
    /// walk skipped these kinds, and skipping them is what made a lost FIFO read
    /// as agreement. The `timeout` here is load-bearing: without it a regression
    /// that opens the pipe hangs the SUITE rather than failing it, and a hung
    /// test is indistinguishable from a slow one.
    #[test]
    #[cfg(unix)]
    fn a_fifo_fingerprints_by_kind_and_never_blocks() {
        let base = std::env::temp_dir().join(format!(
            "copia-fifo-fp-{}-{}",
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

        let (tx, rx) = std::sync::mpsc::channel();
        let probe = fifo;
        std::thread::spawn(move || {
            let _ = tx.send(fingerprint_path(&probe).map(|f| f.ftype));
        });
        let got = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("fingerprint_path BLOCKED on a FIFO — it must classify without opening");

        assert_eq!(got.unwrap(), FileType::Other);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Each kind gets its own tag, so two different kinds never fingerprint
    /// alike. Collapsing them would make a socket-where-a-FIFO-was read as
    /// identical — the same omission defect one level down.
    #[test]
    #[cfg(unix)]
    fn distinct_entry_kinds_get_distinct_tags() {
        use std::os::unix::net::UnixListener;
        let base = std::env::temp_dir().join(format!(
            "copia-kindtag-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&base).unwrap();
        let fifo = base.join("f");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let sock = UnixListener::bind(base.join("s")).ok();

        if made && sock.is_some() {
            let ft_fifo = std::fs::symlink_metadata(&fifo).unwrap().file_type();
            let ft_sock = std::fs::symlink_metadata(base.join("s"))
                .unwrap()
                .file_type();
            assert_eq!(kind_tag(ft_fifo), "fifo");
            assert_eq!(kind_tag(ft_sock), "socket");
            assert_ne!(
                fingerprint_path(&fifo).unwrap().blake3,
                fingerprint_path(&base.join("s")).unwrap().blake3,
                "a FIFO and a socket fingerprint alike — one could replace the other silently"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
