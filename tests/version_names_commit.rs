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

/// The 7-char commit the build embeds: `COPIA_GIT_SHA` when usable (same precedence as
/// build.rs), else the checkout's HEAD, else `unknown` without git.
fn expected_sha() -> String {
    if let Some(s) = std::env::var("COPIA_GIT_SHA")
        .ok()
        .as_deref()
        .and_then(normalize)
    {
        return s;
    }
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

/// The expected commit follows `COPIA_GIT_SHA` when it is set, as build.rs does, and
/// the checkout's HEAD otherwise. Run with `COPIA_GIT_SHA` set to a 40-char sha that
/// differs from HEAD: the build embeds that sha, so the expected value must be it too.
#[test]
fn expected_sha_honors_copia_git_sha() {
    match std::env::var("COPIA_GIT_SHA") {
        Ok(env_sha) => {
            let want: String = env_sha
                .trim()
                .to_ascii_lowercase()
                .chars()
                .take(7)
                .collect();
            assert_eq!(
                expected_sha(),
                want,
                "expected sha must follow COPIA_GIT_SHA"
            );
        }
        Err(_) => {
            let head = expected_sha();
            assert_eq!(
                head.len(),
                7,
                "without COPIA_GIT_SHA the expected sha is HEAD"
            );
        }
    }
}
