//! Embed the build's commit so `copia --version` names it (paiml/copia#74).
//!
//! `COPIA_GIT_SHA` wins when set (CI pins it to the tagged commit). Otherwise the
//! HEAD of the git checkout is used. Either value is trimmed, lowercased and cut to
//! the first 7 hex chars. A tarball with neither reads `unknown`.

use std::path::Path;
use std::process::Command;

/// The 7-char commit to print, or `None` when the input is not a usable hex SHA.
fn normalize(raw: &str) -> Option<String> {
    let t = raw.trim();
    if t.len() >= 7 && t.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(t[..7].to_ascii_lowercase())
    } else {
        None
    }
}

/// Pick the SHA: a usable env value first, then the git value, else `unknown`.
pub fn resolve_sha(env: Option<&str>, git: Option<&str>) -> String {
    env.and_then(normalize)
        .or_else(|| git.and_then(normalize))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Files whose change moves the checked-out commit: HEAD, the ref HEAD points at
/// (when attached), the refs directory, and packed-refs. The refs directory is
/// watched because a branch packed by `git pack-refs` has no loose ref file, and
/// the next commit creates one without touching HEAD or packed-refs.
/// `git_path` maps a git path name to its location.
pub fn rerun_paths(
    head_ref: Option<&str>,
    git_path: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    let mut names = vec!["HEAD".to_string()];
    if let Some(r) = head_ref {
        names.push(r.to_string());
    }
    names.push("refs".to_string());
    names.push("packed-refs".to_string());
    names.iter().filter_map(|n| git_path(n)).collect()
}

fn git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

#[allow(dead_code)]
fn main() {
    println!("cargo:rerun-if-env-changed=COPIA_GIT_SHA");

    let head_ref = git(&["symbolic-ref", "-q", "HEAD"]);
    for path in rerun_paths(head_ref.as_deref(), |name| {
        git(&["rev-parse", "--git-path", name])
    }) {
        if Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    let env = std::env::var("COPIA_GIT_SHA").ok();
    let head = git(&["rev-parse", "HEAD"]);
    let sha = resolve_sha(env.as_deref(), head.as_deref());

    println!("cargo:rustc-env=COPIA_GIT_SHA={sha}");
}
