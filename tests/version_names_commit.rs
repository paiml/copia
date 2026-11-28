//! paiml/copia#74: `copia --version` must name the build's commit, so a fleet host can
//! tell which build it runs. RED on main: the version string carries no commit.
#![cfg(feature = "cli")]
#![allow(clippy::expect_used, clippy::panic)]

use std::process::Command;

/// A usable hex SHA trimmed, lowercased and cut to 7 chars, as build.rs normalizes it.
fn normalize(raw: &str) -> Option<String> {
    let t = raw.trim();
    (t.len() >= 7 && t.chars().all(|c| c.is_ascii_hexdigit())).then(|| t[..7].to_ascii_lowercase())
}

/// Mirrors build.rs `resolve_sha`: a usable env value first, then the git HEAD, else
/// `unknown` (the documented tarball fallback when there is neither).
fn expected_sha_from(env: Option<&str>, head: Option<&str>) -> String {
    env.and_then(normalize)
        .or_else(|| head.and_then(normalize))
        .unwrap_or_else(|| "unknown".to_string())
}

/// The commit this checkout's build embeds, read the same way build.rs reads it.
fn expected_sha() -> String {
    let env = std::env::var("COPIA_GIT_SHA").ok();
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    expected_sha_from(env.as_deref(), head.as_deref())
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
    let expected = expected_sha();
    assert_eq!(
        sha, expected,
        "--version must name the build's commit: {line:?}"
    );
    // No env and no git (a source tarball): the documented fallback is `unknown`,
    // which is not hex by design. Only a real commit gets the hex checks.
    if expected == "unknown" {
        return;
    }
    assert_eq!(sha.len(), 7, "commit must be 7 hex chars: {line:?}");
    assert!(
        sha.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "commit must be lowercase hex: {line:?}"
    );
}

/// The expected commit follows a usable `COPIA_GIT_SHA` as build.rs does, and falls
/// back to HEAD for an unset, empty or non-hex value. Pure inputs, so it does not
/// depend on the CI environment. Fails if the env value is ignored.
#[test]
fn expected_sha_honors_copia_git_sha() {
    let head = "fedcba9876543210fedcba9876543210fedcba98";
    let env_sha = "0123456789abcdef0123456789abcdef01234567";
    assert_eq!(expected_sha_from(Some(env_sha), Some(head)), "0123456");
    assert_eq!(
        expected_sha_from(Some("  0123456789ABCDEF  "), Some(head)),
        "0123456"
    );
    assert_eq!(expected_sha_from(Some("not-a-sha"), Some(head)), "fedcba9");
    assert_eq!(expected_sha_from(Some(""), Some(head)), "fedcba9");
    assert_eq!(expected_sha_from(None, Some(head)), "fedcba9");
    assert_eq!(expected_sha_from(None, None), "unknown");
}
