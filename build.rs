// Embed the git commit in the binary so `copia --version` names the build
// (paiml/copia#74). Order: GIT_SHA env (set by CI or a packager), then
// `git rev-parse` on the checkout, then "unknown" for a source tarball.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GIT_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
    println!("cargo:rerun-if-changed=.git/packed-refs");

    let from_env = std::env::var("GIT_SHA")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let from_git = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let sha = from_env
        .map(|s| s.chars().take(7).collect::<String>())
        .or(from_git)
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=COPIA_GIT_SHA={sha}");
}
