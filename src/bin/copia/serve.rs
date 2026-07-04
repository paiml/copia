//! `copia serve <root>` — the L3 hub side. Reads framed requests from stdin,
//! applies CAS-on-blake3 writes under a brief tree commit-lock (so concurrent
//! SSH-spawned server processes commit linearizably), streams responses to
//! stdout. A stale CAS never overwrites — it lands a conflict-copy instead
//! (docs/specifications/distributed-sync.md).

use super::meta::discover_local_fingerprints;
use super::wire::{cas_decide, read_frame, write_frame, Cas, Hash, Request, Response, VERSION};
use fs2::FileExt;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

/// Join a client-supplied relative path under `root`, rejecting absolute paths
/// and any `..`/root escape (path-traversal guard).
fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    let p = Path::new(rel);
    if p.is_absolute() {
        return None;
    }
    for c in p.components() {
        if matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return None;
        }
    }
    Some(root.join(p))
}

fn tmp_of(dst: &Path) -> PathBuf {
    let mut s = dst.as_os_str().to_owned();
    s.push(".copia-tmp");
    PathBuf::from(s)
}

/// The hub's current blake3 for `dst`, or `None` if absent.
fn current_hash(dst: &Path) -> Option<Hash> {
    super::meta::fingerprint_path(dst).ok().map(|f| f.blake3)
}

/// Run `f` while holding the tree's exclusive commit lock (brief, local — never
/// across a client round-trip), giving linearizable commits across processes.
fn with_commit_lock<T>(lockdir: &Path, f: impl FnOnce() -> T) -> std::io::Result<T> {
    let lf = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lockdir.join("commit.lock"))?;
    lf.lock_exclusive()?;
    let out = f();
    let _ = fs2::FileExt::unlock(&lf);
    Ok(out)
}

pub fn serve(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(root)?;
    let lockdir = root.join(".copia");
    std::fs::create_dir_all(&lockdir)?;
    let mut r = BufReader::new(std::io::stdin().lock());
    let mut w = BufWriter::new(std::io::stdout().lock());
    if !super::wire::read_magic(&mut r)? {
        return Err("client sent a bad protocol prologue (not copia)".into());
    }
    while let Some(req) = read_frame::<_, Request>(&mut r)? {
        match req {
            Request::Hello { .. } => write_frame(&mut w, &Response::Hello { version: VERSION })?,
            Request::List => {
                let fps = discover_local_fingerprints(root).unwrap_or_default();
                let map = fps
                    .into_iter()
                    .filter(|(p, _)| !p.starts_with(".copia"))
                    .map(|(p, f)| (p.to_string_lossy().into_owned(), f))
                    .collect();
                write_frame(&mut w, &Response::Fingerprints(map))?;
            }
            Request::Get { path } => handle_get(root, &path, &mut w)?,
            Request::Put {
                path,
                expected,
                len,
                hash,
            } => handle_put(root, &lockdir, &path, expected, len, hash, &mut r, &mut w)?,
            Request::Delete { path, expected } => {
                handle_delete(root, &lockdir, &path, expected, &mut w)?;
            }
            Request::Bye => break,
        }
    }
    Ok(())
}

fn handle_get<W: Write>(root: &Path, path: &str, w: &mut W) -> std::io::Result<()> {
    let Some(dst) = safe_join(root, path) else {
        return write_frame(w, &Response::Error("bad path".into()));
    };
    match (std::fs::metadata(&dst), current_hash(&dst)) {
        (Ok(m), Some(hash)) => {
            write_frame(w, &Response::Content { len: m.len(), hash })?;
            let mut f = std::fs::File::open(&dst)?;
            std::io::copy(&mut f, w)?;
            w.flush()
        }
        _ => write_frame(w, &Response::Error("not found".into())),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_put<R: Read, W: Write>(
    root: &Path,
    lockdir: &Path,
    path: &str,
    expected: Option<Hash>,
    len: u64,
    hash: Hash,
    r: &mut R,
    w: &mut W,
) -> std::io::Result<()> {
    let Some(dst) = safe_join(root, path) else {
        // Still must drain the content bytes to keep the stream framed.
        std::io::copy(&mut r.take(len), &mut std::io::sink())?;
        return write_frame(w, &Response::Error("bad path".into()));
    };
    if let Some(p) = dst.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = tmp_of(&dst);
    // Stream exactly `len` bytes to the temp file + hash them (never buffer whole).
    let mut hasher = blake3::Hasher::new();
    {
        let mut tf = std::fs::File::create(&tmp)?;
        let mut limited = r.take(len);
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            let n = limited.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            tf.write_all(&buf[..n])?;
        }
        tf.sync_all()?;
    }
    // Integrity: the streamed content must match the hash the client claimed.
    if *hasher.finalize().as_bytes() != hash {
        let _ = std::fs::remove_file(&tmp);
        return write_frame(w, &Response::Error("content hash mismatch".into()));
    }
    let resp = with_commit_lock(lockdir, || {
        let current = current_hash(&dst);
        match cas_decide(current, expected) {
            Cas::Commit => {
                let _ = std::fs::rename(&tmp, &dst);
                Response::PutResult {
                    committed: true,
                    current: Some(hash),
                }
            }
            Cas::Conflict => {
                // Never overwrite on a stale CAS — land a conflict-copy.
                let mut cn = dst.as_os_str().to_owned();
                cn.push(format!(".conflict-{}", super::wire::short_hash(&hash)));
                let _ = std::fs::rename(&tmp, PathBuf::from(cn));
                Response::PutResult {
                    committed: false,
                    current,
                }
            }
        }
    })?;
    write_frame(w, &resp)
}

fn handle_delete<W: Write>(
    root: &Path,
    lockdir: &Path,
    path: &str,
    expected: Option<Hash>,
    w: &mut W,
) -> std::io::Result<()> {
    let Some(dst) = safe_join(root, path) else {
        return write_frame(w, &Response::Error("bad path".into()));
    };
    let resp = with_commit_lock(lockdir, || {
        let current = current_hash(&dst);
        match cas_decide(current, expected) {
            Cas::Commit => {
                let _ = std::fs::remove_file(&dst);
                Response::DeleteResult {
                    deleted: true,
                    current: None,
                }
            }
            Cas::Conflict => Response::DeleteResult {
                deleted: false,
                current,
            },
        }
    })?;
    write_frame(w, &resp)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn safe_join_blocks_traversal_and_absolute() {
        let root = Path::new("/srv/hub");
        assert!(safe_join(root, "a/b.txt").is_some());
        assert!(safe_join(root, "../etc/passwd").is_none());
        assert!(safe_join(root, "a/../../x").is_none());
        assert!(safe_join(root, "/etc/passwd").is_none());
    }

    fn h(b: &[u8]) -> Hash {
        *blake3::hash(b).as_bytes()
    }
    fn put(root: &Path, lockdir: &Path, path: &str, expected: Option<Hash>, content: &[u8]) {
        let mut w = Vec::new();
        handle_put(
            root,
            lockdir,
            path,
            expected,
            content.len() as u64,
            h(content),
            &mut std::io::Cursor::new(content.to_vec()),
            &mut w,
        )
        .unwrap();
    }

    #[test]
    fn cas_commit_then_stale_expected_conflicts_never_overwrites() {
        let root = std::env::temp_dir().join(format!("copia-serve-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let lockdir = root.join(".copia");
        std::fs::create_dir_all(&lockdir).unwrap();
        // 1. create (expected None) -> commit
        put(&root, &lockdir, "f", None, b"X");
        assert_eq!(std::fs::read(root.join("f")).unwrap(), b"X");
        // 2. correct CAS (expected = hash of X) -> commit to Y
        put(&root, &lockdir, "f", Some(h(b"X")), b"YY");
        assert_eq!(std::fs::read(root.join("f")).unwrap(), b"YY");
        // 3. STALE CAS (expected = hash of X, but hub is now YY) -> conflict, NOT overwrite
        put(&root, &lockdir, "f", Some(h(b"X")), b"ZZZ");
        assert_eq!(
            std::fs::read(root.join("f")).unwrap(),
            b"YY",
            "stale CAS must NOT overwrite"
        );
        // the losing write is preserved as a conflict-copy
        let conflict = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .find(|e| e.file_name().to_string_lossy().contains("conflict"));
        assert!(conflict.is_some(), "stale CAS must leave a conflict-copy");
        assert_eq!(
            std::fs::read(conflict.unwrap().path()).unwrap(),
            b"ZZZ",
            "conflict-copy keeps the loser"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn handle_get_streams_content_and_reports_missing() {
        let root = std::env::temp_dir().join(format!("copia-get-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("f"), b"payload").unwrap();
        // present file: Content header frame then the raw bytes
        let mut w = Vec::new();
        handle_get(&root, "f", &mut w).unwrap();
        assert!(
            w.ends_with(b"payload"),
            "content must be streamed after the header frame"
        );
        // missing file: an Error frame, no panic
        let mut w2 = Vec::new();
        handle_get(&root, "nope", &mut w2).unwrap();
        assert!(!w2.is_empty());
        // traversal is refused
        let mut w3 = Vec::new();
        handle_get(&root, "../escape", &mut w3).unwrap();
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn cas_delete_only_with_matching_expected() {
        let root = std::env::temp_dir().join(format!("copia-serve-del-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let lockdir = root.join(".copia");
        std::fs::create_dir_all(&lockdir).unwrap();
        put(&root, &lockdir, "f", None, b"data");
        // stale expected -> not deleted
        let mut w = Vec::new();
        handle_delete(&root, &lockdir, "f", Some(h(b"wrong")), &mut w).unwrap();
        assert!(root.join("f").exists(), "stale CAS delete must be refused");
        // correct expected -> deleted
        let mut w2 = Vec::new();
        handle_delete(&root, &lockdir, "f", Some(h(b"data")), &mut w2).unwrap();
        assert!(
            !root.join("f").exists(),
            "matching CAS delete removes the file"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
