//! paiml/copia#74: `copia --version` must name the build's commit, so a fleet host can
//! tell which build it runs. RED on main: the version string carries no commit.
#![cfg(feature = "cli")]

use std::process::Command;

#[test]
fn version_names_the_build_commit() {
    let out = Command::new(env!("CARGO_BIN_EXE_copia"))
        .arg("--version")
        .output()
        .expect("run copia --version");
    assert!(out.status.success(), "copia --version failed");
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.trim();

    // Expected shape: `copia <semver> (<7 hex chars of the commit>)`.
    let open = line
        .find('(')
        .unwrap_or_else(|| panic!("no commit in version line: {line:?}"));
    let rest = &line[open + 1..];
    let sha = rest
        .strip_suffix(')')
        .unwrap_or_else(|| panic!("commit not closed in version line: {line:?}"));
    assert_eq!(sha.len(), 7, "commit must be 7 hex chars: {line:?}");
    assert!(
        sha.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "commit must be lowercase hex: {line:?}"
    );
}
