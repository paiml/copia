//! `copia hub-sync <local> <target>` — the L3 client. Opens ONE persistent
//! connection to a hub (`ssh host copia serve <root>`, or a local `copia serve`
//! when the target has no `host:`), then pushes the local tree with CAS-on-blake3:
//! each write carries the hash the client last saw for that path, so a concurrent
//! writer's change is detected and the stale write becomes a hub conflict-copy
//! instead of a silent lost update (docs/specifications/distributed-sync.md).

use super::meta::discover_local_fingerprints;
use super::reconcile::Fingerprint;
use super::wire::{read_frame, write_frame, Hash, Request, Response, VERSION};
use std::collections::BTreeMap;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Split `host:root` into (host, root); `None` if there's no `host:` (a local
/// hub path). A single leading segment with no `/` before the first `:` is a host.
fn split_target(t: &str) -> Option<(&str, &str)> {
    let idx = t.find(':')?;
    let host = &t[..idx];
    if host.is_empty() || host.contains('/') {
        None
    } else {
        Some((host, &t[idx + 1..]))
    }
}

pub struct HubClient {
    child: Child,
    w: BufWriter<ChildStdin>,
    r: BufReader<ChildStdout>,
}

impl HubClient {
    /// Spawn the hub process and perform the version handshake.
    pub fn connect(target: &str) -> std::io::Result<Self> {
        let mut cmd = if let Some((host, root)) = split_target(target) {
            let mut c = Command::new("ssh");
            c.arg("-T").arg(host).arg("copia").arg("serve").arg(root);
            c
        } else {
            let exe = std::env::current_exe()?;
            let mut c = Command::new(exe);
            c.arg("serve").arg(target);
            c
        };
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = cmd.spawn()?;
        let w = BufWriter::new(child.stdin.take().ok_or_else(broken)?);
        let r = BufReader::new(child.stdout.take().ok_or_else(broken)?);
        let mut me = Self { child, w, r };
        super::wire::write_magic(&mut me.w)?;
        me.send(&Request::Hello { version: VERSION })?;
        match me.recv()? {
            Response::Hello { version } if version >= 1 => Ok(me),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad hub handshake: {other:?}"),
            )),
        }
    }

    fn send(&mut self, req: &Request) -> std::io::Result<()> {
        write_frame(&mut self.w, req)
    }
    fn recv(&mut self) -> std::io::Result<Response> {
        read_frame(&mut self.r)?
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "hub closed"))
    }

    /// The hub tree's current fingerprints.
    pub fn list(&mut self) -> std::io::Result<BTreeMap<String, Fingerprint>> {
        self.send(&Request::List)?;
        match self.recv()? {
            Response::Fingerprints(m) => Ok(m),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected Fingerprints, got {other:?}"),
            )),
        }
    }

    /// CAS-push one file's content. Returns `true` on commit, `false` if the CAS
    /// lost (a conflict-copy was stored on the hub).
    pub fn put(
        &mut self,
        rel: &str,
        expected: Option<Hash>,
        local: &Path,
        hash: Hash,
    ) -> std::io::Result<bool> {
        let len = std::fs::metadata(local)?.len();
        self.send(&Request::Put {
            path: rel.to_string(),
            expected,
            len,
            hash,
        })?;
        // Stream the content bytes right after the frame.
        let mut f = std::fs::File::open(local)?;
        std::io::copy(&mut f, &mut self.w)?;
        self.w.flush()?;
        match self.recv()? {
            Response::PutResult { committed, .. } => Ok(committed),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected PutResult, got {other:?}"),
            )),
        }
    }

    pub fn bye(mut self) {
        let _ = self.send(&Request::Bye);
        let _ = self.w.flush();
        let _ = self.child.wait();
    }
}

fn broken() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "hub pipe unavailable")
}

/// Push a local tree to the hub with CAS. Files already identical on the hub are
/// skipped; changed/new files are CAS-pushed; a CAS loss is reported (the hub
/// kept a conflict-copy) and makes the run exit non-zero.
pub fn hub_sync(local_root: &Path, target: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = HubClient::connect(target)?;
    let hub = client.list()?;
    let local = discover_local_fingerprints(local_root)?;

    let (mut sent, mut skipped, mut conflicts) = (0u64, 0u64, 0u64);
    for (rel, fp) in &local {
        let rel_s = rel.to_string_lossy().into_owned();
        let expected = hub.get(&rel_s).map(|f| f.blake3);
        if expected == Some(fp.blake3) {
            skipped += 1;
            continue;
        }
        let committed = client.put(&rel_s, expected, &local_root.join(rel), fp.blake3)?;
        if committed {
            sent += 1;
        } else {
            conflicts += 1;
            eprintln!("  CAS conflict (hub changed under us): {rel_s} — hub kept a conflict-copy");
        }
    }
    client.bye();
    println!("Hub push complete: {sent} sent, {skipped} unchanged, {conflicts} conflict(s).");
    if conflicts == 0 {
        Ok(())
    } else {
        Err(format!("{conflicts} CAS conflict(s) — re-run to reconcile").into())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn split_target_distinguishes_remote_and_local() {
        assert_eq!(
            split_target("intel:/data/hub"),
            Some(("intel", "/data/hub"))
        );
        assert_eq!(split_target("host:rel/path"), Some(("host", "rel/path")));
        assert!(split_target("/local/abs/path").is_none());
        assert!(split_target("rel/local").is_none());
    }
}
