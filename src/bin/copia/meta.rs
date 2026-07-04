//! Filesystem + remote metadata discovery for the quick check, plus the mtime
//! preservation helpers that keep it stable across runs.

use super::plan::{FileMeta, MetaMap};
use super::transfer::discover_local_files;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

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
        if let Ok(meta) = std::fs::metadata(root.join(&rel)) {
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
            "cd $'{escaped}' && find . -type f -printf '%s\\t%T@\\t%p\\0'"
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
}
