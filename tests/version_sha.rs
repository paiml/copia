//! `copia --version` names the build commit (paiml/copia#74).
#![cfg(feature = "cli")]

#[test]
fn version_carries_seven_hex_commit_sha() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_copia"))
        .arg("--version")
        .output()
        .expect("run copia --version");
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).expect("utf8");
    let sha = env!("COPIA_GIT_SHA");
    assert!(
        text.contains(&format!("({sha})")),
        "version output lacks commit sha: {text}"
    );
    if sha != "unknown" {
        assert_eq!(sha.len(), 7);
        assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
