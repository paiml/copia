//! The bidirectional-sync archive: the last-synced common state for one
//! `(rootA, rootB)` pair. Keyed by a hash of both canonical roots + a format
//! version + a monotonic epoch, so a lost/mismatched archive degrades to safe
//! no-base mode rather than being blindly trusted. Written atomically
//! (tmp+rename+fsync) with the previous copy retained as `.bak`.

use super::reconcile::FpMap;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct Archive {
    pub format_version: u32,
    /// blake3(canonical(rootA) + NUL + canonical(rootB)) — identifies the pair.
    pub root_pair_hash: String,
    /// Monotonic generation; bumped on every successful commit.
    pub epoch: u64,
    pub host_id: String,
    pub entries: FpMap,
}

impl Archive {
    pub fn fresh(root_pair_hash: String, host_id: String) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            root_pair_hash,
            epoch: 0,
            host_id,
            entries: FpMap::new(),
        }
    }

    /// Load an archive iff it exists, parses, and matches the expected pair +
    /// format version. Any mismatch or error returns `None` — the caller then
    /// runs in safe no-base mode (no deletes). Never trusts a foreign archive.
    pub fn load(path: &Path, expected_pair: &str) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let a: Self = serde_json::from_slice(&bytes).ok()?;
        if a.format_version == FORMAT_VERSION && a.root_pair_hash == expected_pair {
            Some(a)
        } else {
            None
        }
    }

    /// Persist atomically: write tmp, fsync, retain the current file as `.bak`,
    /// then rename tmp into place and fsync the parent dir. The archive is only
    /// ever written AFTER the data it describes has committed (commit-then-record).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            PathBuf::from(s)
        };
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
        }
        if path.exists() {
            let mut bak = path.as_os_str().to_owned();
            bak.push(".bak");
            let _ = std::fs::rename(path, PathBuf::from(bak));
        }
        std::fs::rename(&tmp, path)?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }
}

/// Deterministic identifier for a root pair (order-sensitive: A then B).
pub fn root_pair_hash(a: &Path, b: &Path) -> String {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let mut h = blake3::Hasher::new();
    h.update(canon(a).as_os_str().as_encoded_bytes());
    h.update(b"\0");
    h.update(canon(b).as_os_str().as_encoded_bytes());
    h.finalize().to_hex().to_string()
}

/// Archive location: `~/.copia/archive/<root_pair_hash>.json`.
pub fn archive_path(pair_hash: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".copia")
        .join("archive")
        .join(format!("{pair_hash}.json"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::reconcile::{FileType, Fingerprint};

    #[test]
    fn save_load_roundtrip_and_rejects_wrong_pair() {
        let dir = std::env::temp_dir().join(format!("copia-arch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("a.json");
        let mut arc = Archive::fresh("PAIR123".into(), "hostA".into());
        arc.epoch = 5;
        arc.entries.insert(
            PathBuf::from("f"),
            Fingerprint {
                blake3: [9; 32],
                ftype: FileType::File,
            },
        );
        arc.save(&path).unwrap();
        // correct pair loads
        let got = Archive::load(&path, "PAIR123").unwrap();
        assert_eq!(got.epoch, 5);
        assert_eq!(got.entries[&PathBuf::from("f")].blake3, [9; 32]);
        // wrong pair -> None (safe no-base mode)
        assert!(Archive::load(&path, "OTHER").is_none());
        // a second save retains the previous as .bak
        arc.save(&path).unwrap();
        let mut bak = path.as_os_str().to_owned();
        bak.push(".bak");
        assert!(
            PathBuf::from(bak).exists(),
            "previous archive retained as .bak"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn root_pair_hash_is_order_sensitive_and_stable() {
        let a = Path::new("/tmp/x");
        let b = Path::new("/tmp/y");
        assert_eq!(root_pair_hash(a, b), root_pair_hash(a, b));
        assert_ne!(root_pair_hash(a, b), root_pair_hash(b, a));
    }
}
