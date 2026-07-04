//! End-to-end tests for the L3 hub: `copia hub-sync` pushes to `copia serve` over
//! the framed CBOR protocol (server spawned as a local subprocess — no ssh).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn copia() -> Command {
    Command::new(env!("CARGO_BIN_EXE_copia"))
}
fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("copia-hub-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

#[test]
fn hub_push_lands_files_then_second_push_skips() {
    let base = tmp("push");
    let (local, hub) = (base.join("local"), base.join("hub"));
    std::fs::create_dir_all(local.join("sub")).unwrap();
    std::fs::write(local.join("a.txt"), b"alpha").unwrap();
    std::fs::write(local.join("sub/b.bin"), vec![7u8; 20000]).unwrap();
    let out = copia()
        .arg("hub-sync")
        .arg(&local)
        .arg(&hub)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "push: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(hub.join("a.txt")).unwrap(), b"alpha");
    assert_eq!(
        std::fs::read(hub.join("sub/b.bin")).unwrap(),
        vec![7u8; 20000]
    );
    // second push: everything unchanged -> CAS skips
    let out2 = copia()
        .arg("hub-sync")
        .arg(&local)
        .arg(&hub)
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out2.stdout);
    assert!(
        log.contains("0 sent") && log.contains("unchanged"),
        "2nd push must skip: {log}"
    );
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn hub_rejects_bad_protocol_prologue() {
    let base = tmp("prologue");
    let hub = base.join("hub");
    std::fs::create_dir_all(&hub).unwrap();
    let mut child = copia()
        .arg("serve")
        .arg(&hub)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Feed a non-copia prologue (as an ssh banner / wrong peer would).
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"NOTCOPIA-garbage")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success(), "hub must reject a bad prologue");
    std::fs::remove_dir_all(&base).ok();
}
