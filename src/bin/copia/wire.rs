//! L3 hub wire protocol — a framed, versioned, self-describing (CBOR) request/
//! response protocol over one persistent stdin/stdout pipe (`ssh host copia
//! serve <root>`). Framing is a bounded `u32` length prefix + a CBOR message;
//! bulk file content streams as raw bytes AFTER the message frame (never a giant
//! CBOR blob). See docs/specifications/distributed-sync.md.

use ciborium::{de::from_reader, ser::into_writer};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Handshake magic + protocol version (min-version negotiated at Hello).
pub const MAGIC: &[u8; 6] = b"COPIA1";
pub const VERSION: u32 = 1;
/// Reject any control frame whose length prefix exceeds this BEFORE allocating
/// (a 0xFFFFFFFF prefix must never trigger a 4 GiB allocation). File content is
/// streamed separately and is not bounded by this.
pub const MAX_FRAME: u32 = 1 << 20; // 1 MiB

pub type Hash = [u8; 32];

/// First 6 bytes of a hash as hex (for conflict-copy names).
#[must_use]
pub fn short_hash(h: &Hash) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(12);
    for b in &h[..6] {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Client -> hub. Content-bearing requests (`Put`) carry a `len`; the raw bytes
/// follow the frame on the stream.
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    Hello {
        version: u32,
    },
    /// List the hub tree's fingerprints (path -> blake3+ftype).
    List,
    /// Fetch a file; the hub replies `Content{len,hash}` then `len` raw bytes.
    Get {
        path: String,
    },
    /// CAS write: commit iff the hub's current hash == `expected`; the `len`
    /// content bytes follow this frame. A stale `expected` becomes a conflict-copy.
    Put {
        path: String,
        expected: Option<Hash>,
        len: u64,
        hash: Hash,
    },
    /// CAS delete: tombstone iff current == `expected`, else conflict.
    Delete {
        path: String,
        expected: Option<Hash>,
    },
    Bye,
}

/// Hub -> client.
#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    /// Hello ack with the hub's version.
    Hello {
        version: u32,
    },
    Fingerprints(std::collections::BTreeMap<String, super::reconcile::Fingerprint>),
    /// Header for a `Get`; `len` raw content bytes follow this frame.
    Content {
        len: u64,
        hash: Hash,
    },
    /// `committed=false` means the CAS lost — the hub stored a conflict-copy and
    /// the caller must re-reconcile against `current`.
    PutResult {
        committed: bool,
        current: Option<Hash>,
    },
    DeleteResult {
        deleted: bool,
        current: Option<Hash>,
    },
    Error(String),
}

/// Write the fixed protocol magic (the very first bytes on a clean channel).
pub fn write_magic<W: Write>(w: &mut W) -> std::io::Result<()> {
    w.write_all(MAGIC)?;
    w.flush()
}

/// Read + verify the magic prologue. `false` => a non-copia peer / injected
/// banner / rc-file output — the caller must abort (the quorum's prologue guard).
pub fn read_magic<R: Read>(r: &mut R) -> std::io::Result<bool> {
    let mut m = [0u8; 6];
    r.read_exact(&mut m)?;
    Ok(&m == MAGIC)
}

/// Write one length-prefixed CBOR frame.
pub fn write_frame<W: Write, T: Serialize>(w: &mut W, msg: &T) -> std::io::Result<()> {
    let mut buf = Vec::new();
    into_writer(msg, &mut buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let len = u32::try_from(buf.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "frame too large"))?;
    if len > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds MAX_FRAME",
        ));
    }
    w.write_all(&len.to_be_bytes())?;
    w.write_all(&buf)?;
    w.flush()
}

/// Read one length-prefixed CBOR frame, rejecting an oversized prefix BEFORE
/// allocating. Returns `Ok(None)` on a clean EOF at a frame boundary.
pub fn read_frame<R: Read, T: for<'de> Deserialize<'de>>(r: &mut R) -> std::io::Result<Option<T>> {
    let mut lenb = [0u8; 4];
    match r.read_exact(&mut lenb) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_be_bytes(lenb);
    if len > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds MAX_FRAME",
        ));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    from_reader(&buf[..])
        .map(Some)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

/// The CAS outcome — the sole gate on a hub write. Kept pure so the lost-update
/// safety is Kani-proved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cas {
    Commit,
    Conflict,
}

/// Compare-and-swap decision: commit ONLY when the hub's CURRENT hash matches
/// what the client last observed (`expected`). A stale `expected` (a concurrent
/// writer changed the file since the client read it) can NEVER commit — it is a
/// conflict, preventing the silent lost update. `None` means "absent".
#[must_use]
pub fn cas_decide(current: Option<Hash>, expected: Option<Hash>) -> Cas {
    if current == expected {
        Cas::Commit
    } else {
        Cas::Conflict
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// hub-kani-001: a client whose `expected` does not match the hub's CURRENT
    /// state NEVER commits — the CAS is the fence that stops the lost update.
    #[kani::proof]
    fn stale_cas_never_commits() {
        let cur: Option<Hash> = if kani::any() { Some(kani::any()) } else { None };
        let exp: Option<Hash> = if kani::any() { Some(kani::any()) } else { None };
        if cas_decide(cur, exp) == Cas::Commit {
            assert!(cur == exp);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cas_commits_only_on_match() {
        assert_eq!(cas_decide(Some([1; 32]), Some([1; 32])), Cas::Commit);
        assert_eq!(cas_decide(None, None), Cas::Commit); // create: both absent
        assert_eq!(cas_decide(Some([2; 32]), Some([1; 32])), Cas::Conflict); // stale
        assert_eq!(cas_decide(Some([1; 32]), None), Cas::Conflict); // client thought absent
        assert_eq!(cas_decide(None, Some([1; 32])), Cas::Conflict); // client thought present
    }

    #[test]
    fn frame_roundtrip_and_bounds() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &Request::Hello { version: VERSION }).unwrap();
        let got: Option<Request> = read_frame(&mut &buf[..]).unwrap();
        assert!(matches!(got, Some(Request::Hello { version: 1 })));
        // clean EOF at a boundary -> None
        let empty: Option<Request> = read_frame(&mut &b""[..]).unwrap();
        assert!(empty.is_none());
        // an oversized length prefix is rejected before allocating
        let huge = [0xFF, 0xFF, 0xFF, 0xFF];
        assert!(read_frame::<_, Request>(&mut &huge[..]).is_err());
    }
}
