//! paiml/copia#74: `copia --version` must name the build's commit, so a fleet host can
//! tell which build it runs. RED on main: the version string carries no commit.
#![cfg(feature = "cli")]
#![allow(clippy::expect_used, clippy::panic)]

use std::process::Command;

/// The 7-char commit of the checkout this test runs in, or `unknown` without git.
fn expected_sha() -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .trim()
            .chars()
            .take(7)
            .collect(),
        _ => "unknown".to_string(),
    }
}

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
    let prefix = concat!("copia ", env!("CARGO_PKG_VERSION"), " (");
    assert!(
        line.starts_with(prefix),
        "version line must start with {prefix:?}: {line:?}"
    );
    let open = line
        .find('(')
        .unwrap_or_else(|| panic!("no commit in version line: {line:?}"));
    let rest = &line[open + 1..];
    let sha = rest
        .strip_suffix(')')
        .unwrap_or_else(|| panic!("commit not closed in version line: {line:?}"));
    assert_eq!(
        sha,
        expected_sha(),
        "--version must name the checkout's HEAD: {line:?}"
    );
    assert_eq!(sha.len(), 7, "commit must be 7 hex chars: {line:?}");
    assert!(
        sha.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "commit must be lowercase hex: {line:?}"
    );
}
