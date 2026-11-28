//! Filesystem + remote metadata discovery for the quick check, plus the mtime
//! preservation helpers that keep it stable across runs.

use super::plan::{FileMeta, MetaMap};
use super::reconcile::{FileType, Fingerprint, FpMap};
use super::transfer::{discover_local_files, walk_local};
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
///
/// Every entry that cannot be listed, stat-ed, or carried is pushed to
/// `omitted` (issue #50). An entry is never silently left out of the map: an
/// omitted source entry is what makes `--delete` unsafe.
pub fn discover_local_with_meta(root: &Path, omitted: &mut Vec<String>) -> MetaMap {
    let mut out = MetaMap::new();
    for rel in walk_local(root, omitted) {
        let full = root.join(&rel);
        // symlink_metadata, not metadata: `metadata` FOLLOWS the link, so a
        // DANGLING symlink errors and the entry would be dropped — copia then
        // omits a path rsync -a faithfully recreates. Caught by the rsync
        // differential harness on the `dangling-symlink` shape. A symlink's own
        // size/mtime is the right quick-check value anyway; `deliver_local`
        // recreates it as a link regardless.
        match std::fs::symlink_metadata(&full) {
            Ok(meta) => {
                let ft = meta.file_type();
                if !(ft.is_file() || ft.is_symlink()) {
                    // Not planned: `File::open` on a FIFO blocks until a writer
                    // appears, so the entry must never reach the transfer.
                    omitted.push(format!("cannot carry {} {}", kind_tag(ft), full.display()));
                    continue;
                }
                out.insert(
                    rel,
                    FileMeta {
                        size: meta.len(),
                        mtime: mtime_secs(&meta),
                    },
                );
            }
            Err(e) => omitted.push(format!("cannot stat {}: {e}", full.display())),
        }
    }
    out
}

/// List a remote tree with size+mtime in one `find -printf` over SSH.
///
/// A listing that cannot run or exits non-zero yields an empty map and an entry
/// in `omitted`. It never reads as "the remote is empty".
pub async fn discover_remote_with_meta(
    host: &str,
    remote_root: &str,
    absent_root_is_empty: bool,
    omitted: &mut Vec<String>,
) -> MetaMap {
    // A destination root that does not exist is a first sync: an empty listing,
    // with nothing to delete. Only the destination passes `absent_root_is_empty`.
    // The source must still read an absent root as an omission, or a mistyped
    // source path under --delete would empty the destination.
    let absent_guard = if absent_root_is_empty {
        absent_root_guard(remote_root)
    } else {
        String::new()
    };
    let output = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(remote_listing_command(&absent_guard, remote_root))
        .output()
        .await;
    match output {
        Ok(output) if output.status.success() => parse_remote_meta_output(&output.stdout, omitted),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            omitted.push(format!(
                "failed to list {host}:{remote_root} (exit {}): {}",
                output.status,
                stderr.trim()
            ));
            MetaMap::new()
        }
        Err(e) => {
            omitted.push(format!(
                "failed to start listing of {host}:{remote_root}: {e}"
            ));
            MetaMap::new()
        }
    }
}

/// Quote a path for POSIX sh (dash and bash alike). `$'...'` is bash-only: dash
/// reads it as a literal string, which made an existing root look absent.
pub(crate) fn posix_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The full remote command that lists a tree: the optional absent-root guard,
/// then `find` under `remote_root`. `cd --` so a root that begins with `-` is a
/// path, not an option.
pub(crate) fn remote_listing_command(absent_guard: &str, remote_root: &str) -> String {
    let root_q = posix_quote(remote_root);
    // `! -type d`, NOT `-type f`. The previous form enumerated regular files
    // ONLY, so a symlink, FIFO, socket or device on a remote tree was invisible
    // to every plan built from this listing. Directories are handled separately;
    // everything else is an ENTRY and must be visible even when copia cannot
    // carry it. `find` does not follow symlinks, so %s/%T@ describe the link.
    // Kind = %y then %Y, e.g. `ff`, `lf` (symlink to a file), `lN` (dangling).
    format!("{absent_guard}cd -- {root_q} && find . ! -type d -printf '%y%Y\\t%s\\t%T@\\t%p\\0'")
}

/// Shell prefix that exits 0 only for a destination root that is definitively
/// absent. Every component is tested only after its parent is confirmed a
/// searchable directory, so `-e` can fail only with ENOENT; an EACCES, ENOTDIR
/// or ELOOP anywhere on the way falls through to the listing, which then fails
/// and is an omission. A root with a `.` or `..` component is never classified:
/// `x/.` on an unreadable or non-directory `x` fails `-e` for another reason.
fn absent_root_guard(remote_root: &str) -> String {
    let trimmed = remote_root.trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    let components: Vec<&str> = trimmed.split('/').filter(|c| !c.is_empty()).collect();
    if components.iter().any(|c| *c == "." || *c == "..") {
        return String::new();
    }
    let base = if trimmed.starts_with('/') { "/" } else { "." };
    // Runs in a subshell: `exit 0` there means absent, `exit 1` means "not
    // classified", and only the former short-circuits the listing.
    let base_q = posix_quote(base);
    let mut checks = format!("[ -d {base_q} ] && [ -x {base_q} ] || exit 1; ");
    let mut prefix = base.to_owned();
    for component in components {
        let next = if prefix.ends_with('/') {
            format!("{prefix}{component}")
        } else {
            format!("{prefix}/{component}")
        };
        let q = posix_quote(&next);
        checks.push_str(&format!(
            "[ -e {q} ] || [ -L {q} ] || exit 0; [ -d {q} ] && [ -x {q} ] || exit 1; "
        ));
        prefix = next;
    }
    checks.push_str("exit 1; ");
    format!("( {checks}) && exit 0; ")
}

/// Whether a remote record's kind (`%y%Y`) is one copia carries: a regular file,
/// or a symlink that resolves to a regular file (its bytes are streamed by `cat`).
/// A symlink to a FIFO, a directory, a socket or a device, and a dangling
/// symlink (`N`) or loop (`L`), is not carried: `cat` would block on a FIFO or
/// fail, and that failure must not be the only thing standing between a run and
/// `--delete`.
fn remote_kind_carried(kind: &str) -> bool {
    matches!(kind, "ff" | "lf")
}

/// A stable name for a remote record kind (`%y%Y`), for the omission message.
fn remote_entry_name(kind: &str) -> String {
    let mut chars = kind.chars();
    let (Some(entry), Some(target), None) = (chars.next(), chars.next(), chars.next()) else {
        return "malformed-kind".to_owned();
    };
    let name = |k: char| match k {
        'f' => "file",
        'd' => "directory",
        'p' => "fifo",
        's' => "socket",
        'c' => "char-device",
        'b' => "block-device",
        'D' => "door",
        _ => "unknown",
    };
    match (entry, target) {
        ('l', 'N') => "dangling-symlink".to_owned(),
        ('l', 'L') => "symlink-loop".to_owned(),
        ('l', t) => format!("symlink-to-{}", name(t)),
        (e, _) => name(e).to_owned(),
    }
}

/// Parse NUL-terminated `kind\tsize\tmtime\t./path` records into a `MetaMap`.
/// A record that does not parse is pushed to `omitted`, never dropped silently.
///
/// An entry whose kind copia cannot carry (anything but a regular file or a
/// symlink) is an omission and is NOT entered in the map. It must not be planned,
/// because `cat` on a remote FIFO blocks forever.
pub fn parse_remote_meta_output(stdout: &[u8], omitted: &mut Vec<String>) -> MetaMap {
    let mut out = MetaMap::new();
    if stdout.is_empty() {
        return out;
    }
    // Every record ends in a NUL. Bytes after the last NUL are a record the
    // listing never finished: an omission, never parsed. An empty record between
    // two NULs is malformed too. Neither is dropped silently.
    let Some(last) = stdout.iter().rposition(|&b| b == 0) else {
        omitted.push(format!(
            "truncated remote record (no terminating NUL): {}",
            String::from_utf8_lossy(stdout)
        ));
        return out;
    };
    if last + 1 < stdout.len() {
        omitted.push(format!(
            "truncated remote record (no terminating NUL): {}",
            String::from_utf8_lossy(&stdout[last + 1..])
        ));
    }
    for entry in stdout[..last].split(|&b| b == 0) {
        if entry.is_empty() {
            omitted.push("empty remote record (between NUL terminators)".to_owned());
            continue;
        }
        match parse_remote_record(entry) {
            Ok((rel, meta)) => {
                out.insert(rel, meta);
            }
            Err(reason) => omitted.push(reason),
        }
    }
    out
}

/// Parse one `kind\tsize\tmtime\t./path` record. Every failure is returned as
/// the omission message: nothing is dropped or defaulted.
fn parse_remote_record(entry: &[u8]) -> Result<(PathBuf, FileMeta), String> {
    // A name that is not UTF-8 cannot be carried faithfully: lossy decoding
    // renamed it, so the plan transferred a different file and deleted the
    // real one. It is an omission, never a guess.
    let s = std::str::from_utf8(entry).map_err(|_| {
        format!(
            "malformed remote record (path is not UTF-8): {}",
            String::from_utf8_lossy(entry)
        )
    })?;
    let mut parts = s.splitn(4, '\t');
    let (Some(kind), Some(size), Some(mtime), Some(path)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(format!("malformed remote record: {s}"));
    };
    let size = size
        .parse::<u64>()
        .map_err(|_| format!("malformed remote size: {s}"))?;
    let mtime = parse_remote_mtime(mtime).ok_or_else(|| format!("malformed remote mtime: {s}"))?;
    let rel = path.strip_prefix("./").unwrap_or(path);
    if rel.is_empty() || rel == "." {
        return Err(format!("malformed remote record (empty path): {s}"));
    }
    // The path is a name under the remote root. An absolute path, or any empty,
    // `.` or `..` segment, would escape local_root on a pull. Refused, not carried.
    if rel
        .split('/')
        .any(|seg| seg.is_empty() || seg == "." || seg == "..")
    {
        return Err(format!(
            "unsafe remote path (absolute or contains . or ..): {s}"
        ));
    }
    if !remote_kind_carried(kind) {
        return Err(format!("cannot carry {} {rel}", remote_entry_name(kind)));
    }
    Ok((PathBuf::from(rel), FileMeta { size, mtime }))
}

/// `%T@` is "secs.nanos". Truncate to whole seconds. Anything else after the
/// dot, or a non-integer before it, is malformed: a default of 0 would read as
/// "changed" and put a fabricated timestamp in the plan.
fn parse_remote_mtime(raw: &str) -> Option<i64> {
    let (whole, frac) = raw.split_once('.').unwrap_or((raw, "0"));
    let frac_ok = !frac.is_empty() && frac.bytes().all(|b| b.is_ascii_digit());
    whole.parse::<i64>().ok().filter(|_| frac_ok)
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
        let raw = b"ff\t1024\t1700000000.500\t./a.txt\x00ff\t512\t1699999999\t./sub/b.bin\0";
        let m = parse_remote_meta_output(raw, &mut Vec::new());
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

    /// Issue #50 review finding 3 RED: a malformed record is an omission, never
    /// dropped or defaulted. Three shapes: an empty path, a non-numeric mtime
    /// (defaulted to 0 before), and a non-UTF-8 name (lossy-decoded before, so
    /// the plan deleted the real file and transferred a renamed copy).
    #[test]
    fn malformed_remote_records_are_omissions() {
        let cases: [&[u8]; 5] = [
            b"ff\t3\t1700000000.0\t./\x00",
            b"ff\t3\tabc\t./x\x00",
            b"ff\t1\t1700000000.0\t./caf\xe9\x00",
            b"ff\t3\t1700000000.abc\t./x\x00",
            b"ff\t1\t1700000000.0\t.\x00",
        ];
        for raw in cases {
            let mut omitted = Vec::new();
            let m = parse_remote_meta_output(raw, &mut omitted);
            assert!(m.is_empty(), "{raw:?} was planned: {m:?}");
            assert_eq!(omitted.len(), 1, "{raw:?} was not omitted: {omitted:?}");
            assert!(omitted[0].contains("malformed"), "{omitted:?}");
        }
    }

    /// Review finding (lane 2) RED: a destination root written with a trailing
    /// slash that names an existing regular file is not absent. The guard exited 0.
    #[cfg(unix)]
    #[test]
    fn absent_root_guard_never_reads_an_existing_file_as_absent() {
        let base = std::env::temp_dir().join(format!("copia-50-guard-file-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let file = base.join("regfile");
        std::fs::write(&file, b"x").unwrap();
        let root = format!("{}/", file.display());
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("{}echo LISTING", absent_root_guard(&root)))
            .output()
            .expect("sh");
        std::fs::remove_dir_all(&base).ok();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("LISTING"),
            "guard exited as absent: {stdout:?}"
        );
    }

    #[test]
    fn parse_meta_skips_malformed_records() {
        // missing fields / non-numeric size are dropped, not panicked on
        let mut omitted = Vec::new();
        let m = parse_remote_meta_output(
            b"ff\tnotanumber\t123\t./x\0\0only-one-field\0",
            &mut omitted,
        );
        assert!(m.is_empty());
        // ...and each one is an omission, so a sync can never delete on its strength.
        // Three: the bad size, the empty record between the NULs (finding 5), and the one-field record.
        assert_eq!(omitted.len(), 3, "{omitted:?}");
    }

    /// Review finding A: a FIFO on a remote tree must name its kind and become an
    /// omission. It is not planned for transfer, because `cat` on a FIFO blocks.
    #[test]
    fn remote_fifo_entry_is_an_omission_and_not_planned() {
        let mut omitted = Vec::new();
        let m = parse_remote_meta_output(
            b"pp\t0\t1700000000.0\t./pipe\x00ff\t3\t1700000000.0\t./a.txt\x00",
            &mut omitted,
        );
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m.contains_key(&PathBuf::from("a.txt")));
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(omitted[0].contains("fifo"), "{omitted:?}");
    }

    /// Issue #50: a remote listing that exits non-zero is an omission, and it
    /// reads as an empty map, never as "the remote has nothing".
    #[tokio::test]
    async fn remote_listing_failure_is_an_omission() {
        // A root that cannot exist. The test may run with the ssh shim on PATH
        // (see incremental.rs), which runs the listing here, so a real path such
        // as /nonexistent would be listed instead of failing.
        let absent = format!("/copia-no-such-root-{}", std::process::id());
        let mut omitted = Vec::new();
        let m =
            discover_remote_with_meta("no-such-host.invalid", &absent, false, &mut omitted).await;
        assert!(m.is_empty());
        assert_eq!(omitted.len(), 1, "{omitted:?}");
    }

    /// Issue #50: a directory whose entries cannot be lstat-ed is an omission,
    /// and the entry is not silently absent from the map.
    #[cfg(unix)]
    #[test]
    fn unstatable_entry_is_an_omission() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("copia-50-meta-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("locked")).unwrap();
        std::fs::write(base.join("locked/f"), b"x").unwrap();
        std::fs::write(base.join("ok"), b"y").unwrap();
        std::fs::set_permissions(base.join("locked"), std::fs::Permissions::from_mode(0o444))
            .unwrap();
        let mut omitted = Vec::new();
        let m = discover_local_with_meta(&base, &mut omitted);
        std::fs::set_permissions(base.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert!(m.contains_key(&PathBuf::from("ok")));
        assert!(!m.contains_key(&PathBuf::from("locked/f")));
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(omitted[0].contains("cannot stat"), "{omitted:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Issue #50 review finding B RED: the absent-root guard runs under the remote
    /// login shell, which is POSIX sh (dash) on many hosts. There `$'...'` is a
    /// literal string, so an existing root read as absent and the listing exited 0.
    #[cfg(unix)]
    #[test]
    fn absent_root_guard_is_posix_sh_safe() {
        let base = std::env::temp_dir().join(format!("copia-50-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let present = base.join("present");
        let absent = base.join("absent");
        std::fs::create_dir_all(&present).unwrap();
        let run = |root: &Path| {
            let script = format!("{}echo REACHED", absent_root_guard(&root.to_string_lossy()));
            std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(script)
                .output()
                .unwrap()
        };
        // An existing root must not take the early exit.
        let out = run(&present);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "REACHED",
            "{out:?}"
        );
        // A definitively absent root takes it: exit 0, nothing else runs.
        let out = run(&absent);
        assert!(out.status.success(), "{out:?}");
        assert!(out.stdout.is_empty(), "{out:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Issue #50 review finding B: a stat failure is not "absent". A root whose
    /// parent cannot be searched must not take the early exit.
    #[cfg(unix)]
    #[test]
    fn absent_root_guard_does_not_exit_on_a_stat_failure() {
        use std::os::unix::fs::PermissionsExt;
        let base =
            std::env::temp_dir().join(format!("copia-50-guard-eacces-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let locked = base.join("locked");
        std::fs::create_dir_all(locked.join("child")).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let script = format!(
            "{}echo REACHED",
            absent_root_guard(&locked.join("child").to_string_lossy())
        );
        let out = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(script)
            .output()
            .unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        // Not absent, only unreadable: the guard must fall through to the listing
        // (which then fails and is an omission), never take the exit 0 path.
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "REACHED",
            "{out:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Issue #50: an entry kind copia cannot carry (a FIFO) is an omission.
    #[cfg(unix)]
    #[test]
    fn uncarriable_entry_kind_is_an_omission() {
        let base = std::env::temp_dir().join(format!("copia-50-fifo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(base.join("pipe"))
            .status()
            .unwrap();
        assert!(status.success());
        let mut omitted = Vec::new();
        let _ = discover_local_with_meta(&base, &mut omitted);
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(omitted[0].starts_with("cannot carry fifo"), "{omitted:?}");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Issue #50: an unreadable subdirectory is named as an omission, and the
    /// rest of the tree is still walked and returned.
    #[cfg(unix)]
    #[test]
    fn walk_names_an_unreadable_subdirectory_and_keeps_the_rest() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("copia-50-metawalk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("locked")).unwrap();
        std::fs::write(base.join("locked/f"), b"x").unwrap();
        std::fs::write(base.join("ok"), b"y").unwrap();
        std::fs::set_permissions(base.join("locked"), std::fs::Permissions::from_mode(0o000))
            .unwrap();
        let mut omitted = Vec::new();
        let files = walk_local(&base, &mut omitted);
        std::fs::set_permissions(base.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert_eq!(files, vec![PathBuf::from("ok")]);
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(
            omitted[0].starts_with("unreadable directory"),
            "{omitted:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn discover_local_meta_and_set_mtime_roundtrip() {
        let base = std::env::temp_dir().join(format!("copia-meta-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("d")).unwrap();
        std::fs::write(base.join("d/f"), b"hello").unwrap();
        let m = discover_local_with_meta(&base, &mut Vec::new());
        assert_eq!(m[&PathBuf::from("d/f")].size, 5);
        // set + read back the mtime at 1s granularity
        set_local_mtime(&base.join("d/f"), 1_600_000_000).unwrap();
        let m2 = discover_local_with_meta(&base, &mut Vec::new());
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

// Review regressions for copia #50 round 4 (findings 1, 3, 4, 5).
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod r93_meta {
    use super::*;

    /// Finding 1 (CRITICAL): a remote path that escapes the root is refused.
    #[test]
    fn remote_paths_that_escape_the_root_are_refused() {
        let cases: [&[u8]; 5] = [
            b"ff\t1\t1700000000.0\t/etc/passwd\x00",
            b"ff\t1\t1700000000.0\t./../x\x00",
            b"ff\t1\t1700000000.0\t./a/../../x\x00",
            b"ff\t1\t1700000000.0\t./a/./b\x00",
            b"ff\t1\t1700000000.0\t.//x\x00",
        ];
        for raw in cases {
            let mut omitted = Vec::new();
            let m = parse_remote_meta_output(raw, &mut omitted);
            assert!(m.is_empty(), "{raw:?} was planned: {m:?}");
            assert_eq!(omitted.len(), 1, "{raw:?} was not refused: {omitted:?}");
            assert!(omitted[0].contains("unsafe"), "{omitted:?}");
        }
    }

    /// Finding 5 (MEDIUM): a truncated trailing record and an empty NUL record are
    /// omissions, never silently dropped.
    #[test]
    fn truncated_and_empty_records_are_omissions() {
        let mut omitted = Vec::new();
        let m = parse_remote_meta_output(b"ff\t1\t1700000000.0\t./a", &mut omitted);
        assert!(m.is_empty(), "{m:?}");
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(omitted[0].contains("truncated"), "{omitted:?}");

        let mut omitted = Vec::new();
        let m = parse_remote_meta_output(
            b"ff\t1\t1700000000.0\t./a\x00ff\t1\t1700000000.0\t./b",
            &mut omitted,
        );
        assert_eq!(m.len(), 1, "{m:?}");
        assert!(m.contains_key(&PathBuf::from("a")));
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(omitted[0].contains("truncated"), "{omitted:?}");

        let mut omitted = Vec::new();
        let m = parse_remote_meta_output(b"\x00ff\t1\t1700000000.0\t./b\x00", &mut omitted);
        assert_eq!(m.len(), 1, "{m:?}");
        assert_eq!(omitted.len(), 1, "{omitted:?}");
        assert!(omitted[0].contains("empty"), "{omitted:?}");
    }

    #[cfg(unix)]
    fn run_guard(root: &str) -> std::process::Output {
        std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("{}echo REACHED", absent_root_guard(root)))
            .output()
            .unwrap()
    }

    /// Finding 3 (HIGH): a root ending in `/.`, or a dangling link, an unsearchable
    /// dir or a regular file under it, must never take the absent exit. Only a true
    /// ENOENT may. Each case must fall through to the listing.
    #[cfg(unix)]
    #[test]
    fn dot_suffixed_roots_are_never_read_as_absent() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let base = std::env::temp_dir().join(format!("copia-93-dot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let child = base.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::set_permissions(&child, std::fs::Permissions::from_mode(0o000)).unwrap();
        symlink("nowhere", base.join("dangling")).unwrap();
        std::fs::write(base.join("file"), b"x").unwrap();
        let roots = [
            format!("{}/.", child.display()),
            format!("{}/./", child.display()),
            format!("{}/dangling/.", base.display()),
            format!("{}/file/.", base.display()),
        ];
        let results: Vec<_> = roots.iter().map(|r| (r.clone(), run_guard(r))).collect();
        std::fs::set_permissions(&child, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::remove_dir_all(&base).ok();
        for (root, out) in results {
            assert_eq!(
                String::from_utf8_lossy(&out.stdout).trim(),
                "REACHED",
                "{root} was read as absent: {out:?}"
            );
        }
    }

    /// Finding 4 (MEDIUM-HIGH): a multi-level absent destination is a first sync.
    /// A missing level under an existing directory is absent; a file in the middle is not.
    #[cfg(unix)]
    #[test]
    fn multi_level_absent_root_takes_the_early_exit() {
        let base = std::env::temp_dir().join(format!("copia-93-deep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("present")).unwrap();
        let root = base.join("present/missing-a/missing-b/remote");
        let out = run_guard(&root.to_string_lossy());
        assert!(out.status.success(), "{out:?}");
        assert!(
            out.stdout.is_empty(),
            "absent deep root fell through: {out:?}"
        );

        std::fs::write(base.join("present/file"), b"x").unwrap();
        let out = run_guard(&base.join("present/file/x/y").to_string_lossy());
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "REACHED",
            "{out:?}"
        );
        std::fs::remove_dir_all(&base).ok();
    }
}
