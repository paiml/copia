//! Cross-feature integration test: L1 (incremental) + L2 (bidirectional) + L3
//! (hub) exercised together in one realistic fleet scenario — two edge nodes and
//! a central hub moving a shared dataset. Drives the real CLI binary; a local
//! `serve` subprocess stands in for the SSH hub.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
use std::path::{Path, PathBuf};
use std::process::Command;

fn copia() -> Command {
    Command::new(env!("CARGO_BIN_EXE_copia"))
}
fn run(args: &[&str]) -> (bool, String) {
    let o = copia().args(args).output().unwrap();
    (
        o.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
    )
}
fn workspace() -> PathBuf {
    let p = std::env::temp_dir().join(format!("copia-integ-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}
fn write(p: &Path, name: &str, body: &[u8]) {
    let f = p.join(name);
    std::fs::create_dir_all(f.parent().unwrap()).unwrap();
    std::fs::write(f, body).unwrap();
}
fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

#[test]
fn fleet_scenario_l1_l2_l3_end_to_end() {
    let ws = workspace();
    let (node1, node2, hub, staging) = (
        ws.join("node1"),
        ws.join("node2"),
        ws.join("hub"),
        ws.join("staging"),
    );

    // ── L1: node1 builds a dataset and incrementally syncs it to a staging dir ──
    write(&node1, "data/a.txt", b"alpha");
    write(&node1, "data/sub/b.bin", &vec![7u8; 30000]);
    write(&node1, "data/scratch.tmp", b"junk");
    let src = format!("{}/", node1.join("data").display());
    let dst = format!("{}/", staging.display());
    let (ok, log) = run(&["sync", "-r", "--exclude", "*.tmp", &src, &dst]);
    assert!(ok, "L1 sync failed: {log}");
    assert!(staging.join("a.txt").exists() && staging.join("sub/b.bin").exists());
    assert!(
        !staging.join("scratch.tmp").exists(),
        "L1 --exclude must drop *.tmp"
    );
    // re-sync is a no-op (quick-check skip)
    let (ok2, log2) = run(&["sync", "-r", "--exclude", "*.tmp", &src, &dst]);
    assert!(
        ok2 && (log2.contains("0 to transfer") || log2.contains("up to date")),
        "L1 not incremental: {log2}"
    );

    // ── L3: two nodes push to ONE hub over the framed protocol (CAS-safe) ──
    write(&node1, "up/shared.txt", b"from-node1");
    write(&node1, "up/n1only.txt", b"n1");
    write(&node2, "up/n2only.txt", b"n2");
    let hub_s = hub.display().to_string();
    assert!(
        run(&["hub-sync", &node1.join("up").display().to_string(), &hub_s]).0,
        "node1 hub push"
    );
    assert!(
        run(&["hub-sync", &node2.join("up").display().to_string(), &hub_s]).0,
        "node2 hub push"
    );
    // hub holds every distinct file from both nodes
    assert_eq!(read(&hub.join("shared.txt")), b"from-node1");
    assert!(
        hub.join("n1only.txt").exists() && hub.join("n2only.txt").exists(),
        "hub must hold both nodes' files"
    );

    // node2 pushes a CHANGED shared file it fetched (fresh baseline) -> commits
    write(&node2, "up2/shared.txt", b"from-node2-v2");
    let (ok3, _l3) = run(&["hub-sync", &node2.join("up2").display().to_string(), &hub_s]);
    assert!(ok3, "node2 second push");
    assert_eq!(
        read(&hub.join("shared.txt")),
        b"from-node2-v2",
        "L3 CAS commit updated the shared file"
    );

    // ── L2: node1 and node2 bidirectionally reconcile a shared config dir ──
    let (cfg1, cfg2, home) = (ws.join("cfg1"), ws.join("cfg2"), ws.join("home"));
    write(&cfg1, "settings", b"base");
    write(&cfg2, "settings", b"base");
    let bisync = |extra: &[&str]| {
        let mut a = vec!["bisync"];
        a.extend_from_slice(extra);
        let (c1, c2) = (cfg1.display().to_string(), cfg2.display().to_string());
        a.push(&c1);
        a.push(&c2);
        let o = copia().env("HOME", &home).args(&a).output().unwrap();
        (
            o.status.success(),
            format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
        )
    };
    bisync(&[]); // seed archive
                 // one-sided edit propagates
    std::fs::write(cfg1.join("settings"), b"edited-on-1").unwrap();
    assert!(bisync(&[]).0);
    assert_eq!(
        read(&cfg2.join("settings")),
        b"edited-on-1",
        "L2 one-sided edit must propagate"
    );
    // divergent edit -> convergent conflict-copy, both versions survive
    std::fs::write(cfg1.join("settings"), b"CONFLICT-1").unwrap();
    std::fs::write(cfg2.join("settings"), b"CONFLICT-2").unwrap();
    let (_c, clog) = bisync(&["-v"]);
    assert!(
        clog.contains("conflict"),
        "L2 must report a conflict: {clog}"
    );
    let files1: Vec<_> = std::fs::read_dir(&cfg1)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    let files2: Vec<_> = std::fs::read_dir(&cfg2)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(
        files1.len(),
        files2.len(),
        "L2 replicas must converge to the same file set"
    );
    let all: String = files1
        .iter()
        .map(|n| String::from_utf8_lossy(&read(&cfg1.join(n))).into_owned())
        .collect();
    assert!(
        all.contains("CONFLICT-1") && all.contains("CONFLICT-2"),
        "L2 must lose no data: {all}"
    );

    std::fs::remove_dir_all(&ws).ok();
}
