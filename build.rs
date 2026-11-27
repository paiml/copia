//! Embed the build's commit so `copia --version` names it (paiml/copia#74).
//!
//! `COPIA_GIT_SHA` wins when set (CI pins it to the tagged commit). Otherwise the
//! short HEAD of the git checkout is used. A tarball with neither reads `unknown`.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=COPIA_GIT_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");

    let sha = std::env::var("COPIA_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short=7", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=COPIA_GIT_SHA={sha}");
}
