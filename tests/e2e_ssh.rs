//! End-to-end tests that drive the `copia` CLI binary against real SSH
//! (localhost), so main.rs + the SSH transfer shims get covered by integration
//! (they are not unit-testable). Skipped only if localhost SSH is unavailable.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::unnecessary_literal_unwrap
)]
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

fn copia() -> Command {
    Command::new(env!("CARGO_BIN_EXE_copia"))
}

fn ssh_probe() -> bool {
    Command::new("ssh")
        .args([
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=5",
            "localhost",
            "true",
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// True if passwordless `ssh localhost` works — provisioning it once if needed.
/// On dev boxes it already works (early return). In the CI clean-room container
/// openssh is installed but `sshd` isn't running and no key exists, so we set it
/// up. Three subtleties, all verified against the sovereign-ci container: keys and
/// `authorized_keys` must live in the login user's passwd home (from `getent`),
/// not `$HOME` — GitHub Actions sets `$HOME` to `/github/home` for container jobs
/// but `sshd` and bare `ssh` resolve `~` from the passwd database (`/root`); the
/// stock image may ship `PermitRootLogin no`, so a drop-in enables key-only root
/// login; and a system-wide client config under `/etc/ssh/ssh_config.d` relaxes
/// host-key checking and pins the identity for `localhost`, so copia's bare
/// `ssh localhost` runs non-interactively. Returns false (SSH tests skip) on failure.
fn ssh_localhost_ok() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| {
        if ssh_probe() {
            return true;
        }
        let script = r#"
RH=$(getent passwd "$(id -un)" | cut -d: -f6); [ -n "$RH" ] || RH="${HOME:-/root}"
mkdir -p /etc/ssh/sshd_config.d /etc/ssh/ssh_config.d 2>/dev/null
printf 'PermitRootLogin prohibit-password\nPubkeyAuthentication yes\n' \
  > /etc/ssh/sshd_config.d/99-ci-localhost.conf 2>/dev/null
printf 'Host localhost\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  IdentityFile %s/.ssh/id_ed25519\n  LogLevel ERROR\n' "$RH" \
  > /etc/ssh/ssh_config.d/99-ci-localhost.conf 2>/dev/null
grep -q 'ssh_config.d/' /etc/ssh/ssh_config 2>/dev/null \
  || echo 'Include /etc/ssh/ssh_config.d/*.conf' >> /etc/ssh/ssh_config 2>/dev/null
ssh-keygen -A 2>/dev/null
mkdir -p /run/sshd 2>/dev/null
pgrep -x sshd >/dev/null 2>&1 || /usr/sbin/sshd 2>/dev/null
mkdir -p "$RH/.ssh" && chmod 700 "$RH/.ssh"
[ -f "$RH/.ssh/id_ed25519" ] || ssh-keygen -t ed25519 -N '' -f "$RH/.ssh/id_ed25519" 2>/dev/null
cat "$RH/.ssh/id_ed25519.pub" >> "$RH/.ssh/authorized_keys" 2>/dev/null
chmod 600 "$RH/.ssh/authorized_keys" 2>/dev/null
"#;
        let _ = Command::new("sh").arg("-c").arg(script).status();
        std::thread::sleep(std::time::Duration::from_millis(800));
        ssh_probe()
    })
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("copia-e2e-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn seed_tree(root: &Path) {
    std::fs::create_dir_all(root.join("sub/deep")).unwrap();
    std::fs::write(root.join("a.txt"), b"alpha\n").unwrap();
    std::fs::write(root.join("sub/b.txt"), b"beta\n").unwrap();
    std::fs::write(root.join("sub/deep/c.bin"), vec![0xABu8; 8192]).unwrap();
}

fn assert_tree(root: &Path) {
    assert_eq!(std::fs::read(root.join("a.txt")).unwrap(), b"alpha\n");
    assert_eq!(std::fs::read(root.join("sub/b.txt")).unwrap(), b"beta\n");
    assert_eq!(
        std::fs::read(root.join("sub/deep/c.bin")).unwrap(),
        vec![0xABu8; 8192]
    );
}

#[test]
fn e2e_cli_version_and_help() {
    assert!(copia().arg("--version").output().unwrap().status.success());
    assert!(copia().arg("--help").output().unwrap().status.success());
    // unknown args exit non-zero (covers clap error path)
    assert!(!copia().arg("--nope").output().unwrap().status.success());
}

#[test]
fn e2e_recursive_pull_via_ssh_localhost() {
    if !ssh_localhost_ok() {
        eprintln!("skip e2e_pull: localhost ssh unavailable");
        return;
    }
    let base = tmp("pull");
    let src = base.join("src");
    let dst = base.join("dst");
    seed_tree(&src);
    let out = copia()
        .args([
            "sync",
            "-r",
            "-j",
            "4",
            &format!("localhost:{}/", src.display()),
            &format!("{}/", dst.display()),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "pull: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_tree(&dst);
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_recursive_push_via_ssh_localhost() {
    if !ssh_localhost_ok() {
        eprintln!("skip e2e_push: localhost ssh unavailable");
        return;
    }
    let base = tmp("push");
    let src = base.join("src");
    let dst = base.join("dst");
    seed_tree(&src);
    let out = copia()
        .args([
            "sync",
            "-r",
            "-j",
            "4",
            &format!("{}/", src.display()),
            &format!("localhost:{}/", dst.display()),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "push: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_tree(&dst);
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_recursive_local_to_local() {
    let base = tmp("l2l");
    let src = base.join("src");
    let dst = base.join("dst");
    seed_tree(&src);
    let out = copia()
        .args([
            "sync",
            "-r",
            &format!("{}/", src.display()),
            &format!("{}/", dst.display()),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "l2l: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_tree(&dst);
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_single_file_local_and_ssh() {
    let base = tmp("single");
    std::fs::create_dir_all(&base).unwrap();
    let a = base.join("a");
    let b = base.join("b");
    std::fs::write(&a, b"single-file-body").unwrap();
    // local -> local single file
    assert!(copia()
        .args(["sync", a.to_str().unwrap(), b.to_str().unwrap()])
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(std::fs::read(&b).unwrap(), b"single-file-body");
    // remote -> local single file (covers run_sync_remote_to_local)
    if ssh_localhost_ok() {
        let c = base.join("c");
        let ok = copia()
            .args([
                "sync",
                &format!("localhost:{}", a.display()),
                c.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            ok.status.success(),
            "ssh single: {}",
            String::from_utf8_lossy(&ok.stderr)
        );
        assert_eq!(std::fs::read(&c).unwrap(), b"single-file-body");
    }
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_signature_delta_patch_cli_roundtrip() {
    let base = tmp("sdp");
    std::fs::create_dir_all(&base).unwrap();
    let basis = base.join("basis");
    let source = base.join("source");
    let sig = base.join("sig");
    let delta = base.join("delta");
    let out = base.join("out");
    std::fs::write(
        &basis,
        b"the quick brown fox jumps over the lazy dog".repeat(50),
    )
    .unwrap();
    std::fs::write(
        &source,
        b"the quick brown cat jumps over the lazy dog".repeat(50),
    )
    .unwrap();
    assert!(copia()
        .args([
            "signature",
            basis.to_str().unwrap(),
            "-o",
            sig.to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(copia()
        .args([
            "delta",
            source.to_str().unwrap(),
            sig.to_str().unwrap(),
            "-o",
            delta.to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(copia()
        .args([
            "patch",
            basis.to_str().unwrap(),
            delta.to_str().unwrap(),
            "-o",
            out.to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(
        std::fs::read(&out).unwrap(),
        std::fs::read(&source).unwrap(),
        "patch reconstructs source"
    );
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_empty_and_error_paths() {
    let base = tmp("err");
    std::fs::create_dir_all(base.join("empty")).unwrap();
    // empty recursive source -> Ok, no-op
    assert!(copia()
        .args([
            "sync",
            "-r",
            &format!("{}/", base.join("empty").display()),
            &format!("{}/", base.join("out").display())
        ])
        .output()
        .unwrap()
        .status
        .success());
    // nonexistent source -> error exit
    assert!(!copia()
        .args(["sync", "/no/such/path/x", base.join("y").to_str().unwrap()])
        .output()
        .unwrap()
        .status
        .success());
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_single_file_local_to_remote_ssh() {
    if !ssh_localhost_ok() {
        eprintln!("skip: localhost ssh unavailable");
        return;
    }
    let base = tmp("l2r");
    std::fs::create_dir_all(&base).unwrap();
    let a = base.join("a");
    let dest = base.join("remote_dest");
    std::fs::write(&a, b"push-one-file").unwrap();
    // covers run_sync_local_to_remote (single file, local -> remote)
    let out = copia()
        .args([
            "sync",
            a.to_str().unwrap(),
            &format!("localhost:{}", dest.display()),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "l2r single: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&dest).unwrap(), b"push-one-file");
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn e2e_signature_delta_patch_default_output_paths() {
    let base = tmp("defout");
    std::fs::create_dir_all(&base).unwrap();
    let basis = base.join("basis");
    let source = base.join("source");
    std::fs::write(
        &basis,
        b"content for default-output derivation test".repeat(30),
    )
    .unwrap();
    std::fs::write(
        &source,
        b"content for DEFAULT-output derivation test".repeat(30),
    )
    .unwrap();
    // No -o: signature -> basis.sig, delta -> source.delta (covers the default paths)
    assert!(copia()
        .args(["signature", basis.to_str().unwrap()])
        .output()
        .unwrap()
        .status
        .success());
    assert!(base.join("basis.sig").exists(), "default .sig path");
    assert!(copia()
        .args([
            "delta",
            source.to_str().unwrap(),
            base.join("basis.sig").to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(base.join("source.delta").exists(), "default .delta path");
    let out = base.join("recon");
    assert!(copia()
        .args([
            "patch",
            basis.to_str().unwrap(),
            base.join("source.delta").to_str().unwrap(),
            "-o",
            out.to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(
        std::fs::read(&out).unwrap(),
        std::fs::read(&source).unwrap()
    );
    std::fs::remove_dir_all(&base).ok();
}

// CI note: these SSH e2e tests self-provision sshd + localhost keys in the
// sovereign-ci container (openssh baked in) so the SSH-transport paths are
// covered rather than skipped.
