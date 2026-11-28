//! paiml/copia#74 review fixes: the build-script SHA resolution and rerun paths.
//! `build.rs` is included as a module so its pure functions are tested directly.
#![allow(dead_code, clippy::all, clippy::pedantic, clippy::expect_used)]

#[path = "../build.rs"]
mod build;

use build::{rerun_paths, resolve_sha};

const FULL: &str = "ABCDEF0123456789abcdef0123456789ABCDEF01";

#[test]
fn full_length_env_sha_is_truncated_to_seven_lowercase() {
    // CI passes the full 40-char commit; the acceptance is 7 hex chars.
    assert_eq!(resolve_sha(Some(FULL), None), "abcdef0");
}

#[test]
fn env_sha_with_surrounding_whitespace_is_trimmed() {
    assert_eq!(
        resolve_sha(Some("  abcdef1234\n"), Some("1111111")),
        "abcdef1"
    );
}

#[test]
fn whitespace_only_env_falls_back_to_git() {
    assert_eq!(resolve_sha(Some(" \n\t"), Some("1234567")), "1234567");
}

#[test]
fn empty_env_falls_back_to_git() {
    assert_eq!(resolve_sha(Some(""), Some("1234567")), "1234567");
}

#[test]
fn non_hex_env_falls_back_to_git() {
    assert_eq!(resolve_sha(Some("not-a-sha"), Some("7654321")), "7654321");
}

#[test]
fn git_value_longer_than_seven_is_truncated() {
    // `git rev-parse --short=7` is a minimum length, not a cap.
    assert_eq!(resolve_sha(None, Some("1234567890abcdef")), "1234567");
}

#[test]
fn no_env_and_no_git_reads_unknown() {
    assert_eq!(resolve_sha(None, None), "unknown");
    assert_eq!(resolve_sha(Some(" "), Some("")), "unknown");
}

fn resolver(map: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |name: &str| {
        map.iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| (*v).to_string())
    }
}

#[test]
fn attached_head_watches_head_its_ref_and_packed_refs() {
    let git = resolver(&[
        ("HEAD", ".git/HEAD"),
        ("refs/heads/fix/74", ".git/refs/heads/fix/74"),
        ("packed-refs", ".git/packed-refs"),
    ]);
    let paths = rerun_paths(Some("refs/heads/fix/74"), git);
    assert!(paths.contains(&".git/HEAD".to_string()), "{paths:?}");
    assert!(
        paths.contains(&".git/refs/heads/fix/74".to_string()),
        "{paths:?}"
    );
    assert!(paths.contains(&".git/packed-refs".to_string()), "{paths:?}");
}

#[test]
fn detached_head_watches_head_and_packed_refs() {
    let git = resolver(&[("HEAD", ".git/HEAD"), ("packed-refs", ".git/packed-refs")]);
    let paths = rerun_paths(None, git);
    assert_eq!(
        paths,
        vec![".git/HEAD".to_string(), ".git/packed-refs".to_string()]
    );
}

#[test]
fn tarball_without_git_watches_nothing() {
    let git = resolver(&[]);
    assert!(rerun_paths(None, git).is_empty());
}

#[test]
fn six_char_env_sha_falls_back_to_git() {
    // A short hex value is not a commit: it must not be printed as one.
    assert_eq!(resolve_sha(Some("abc123"), Some("1234567")), "1234567");
}

/// Compile `build.rs` as a standalone program and run it inside `dir`, returning
/// the `cargo:` directives it prints. This exercises `main()`, not only the helpers.
fn build_script_output(dir: &std::path::Path, env_sha: Option<&str>) -> String {
    let exe = dir.join("build_script_bin");
    let status = std::process::Command::new("rustc")
        .args(["--edition", "2021", "-o"])
        .arg(&exe)
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"))
        .status()
        .expect("run rustc on build.rs");
    assert!(status.success(), "rustc failed on build.rs");
    let mut cmd = std::process::Command::new(&exe);
    cmd.current_dir(dir).env_remove("COPIA_GIT_SHA");
    if let Some(sha) = env_sha {
        cmd.env("COPIA_GIT_SHA", sha);
    }
    let out = cmd.output().expect("run build script");
    assert!(out.status.success(), "build script failed");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn git_in(dir: &std::path::Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@example.invalid")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@example.invalid")
        .status()
        .expect("run git");
    assert!(status.success(), "git {args:?} failed");
}

#[test]
fn packed_branch_ref_change_reruns_build_script() {
    // After `git pack-refs --all` the branch has no loose ref file. A later commit
    // creates one without touching HEAD or packed-refs, so the build script must
    // watch the refs directory or Cargo never reruns it and --version goes stale.
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    git_in(dir, &["init", "-q", "-b", "main"]);
    git_in(dir, &["commit", "-q", "--allow-empty", "-m", "init"]);
    git_in(dir, &["pack-refs", "--all"]);
    assert!(!dir.join(".git/refs/heads/main").exists());

    let stdout = build_script_output(dir, None);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        lines.contains(&"cargo:rerun-if-changed=.git/refs"),
        "refs directory not watched: {stdout}"
    );
    assert!(
        lines.contains(&"cargo:rerun-if-changed=.git/packed-refs"),
        "packed-refs not watched: {stdout}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("cargo:rustc-env=COPIA_GIT_SHA=")),
        "no COPIA_GIT_SHA emitted: {stdout}"
    );
}

fn init_repo_with_commit(dir: &std::path::Path) {
    git_in(dir, &["init", "-q", "-b", "main"]);
    git_in(dir, &["commit", "-q", "--allow-empty", "-m", "init"]);
}

/// The 7-char HEAD of the repo at `dir`, as git reports it.
fn git_head_short(dir: &std::path::Path) -> String {
    let out = std::process::Command::new("git")
        .current_dir(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("run git rev-parse");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .chars()
        .take(7)
        .collect()
}

/// The SHA the build script embedded via `cargo:rustc-env=COPIA_GIT_SHA=`.
fn embedded_sha(stdout: &str) -> String {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("cargo:rustc-env=COPIA_GIT_SHA="))
        .map(str::to_string)
        .expect("build script embeds COPIA_GIT_SHA")
}

const ENV_FULL: &str = "0123456789abcdef0123456789abcdef01234567";
const ENV_FULL_B: &str = "fedcba9876543210fedcba9876543210fedcba98";

#[test]
fn build_script_main_embeds_valid_env_sha_over_git() {
    // Test-first: main() must read COPIA_GIT_SHA. Against a build script that
    // ignored the env value, this would embed the checkout's HEAD instead.
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    init_repo_with_commit(dir);
    let git_sha = git_head_short(dir);
    let stdout = build_script_output(dir, Some(ENV_FULL));
    let embedded = embedded_sha(&stdout);
    assert_eq!(embedded, "0123456", "env sha must win: {stdout}");
    assert_ne!(git_sha, embedded, "test must distinguish env from git");
}

#[test]
fn build_script_main_falls_back_to_git_for_six_char_env() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    init_repo_with_commit(dir);
    let stdout = build_script_output(dir, Some("abc123"));
    assert_eq!(embedded_sha(&stdout), git_head_short(dir), "{stdout}");
}

#[test]
fn build_script_emits_env_rerun_line_and_changing_env_changes_sha() {
    // Changing COPIA_GIT_SHA must both trigger a rerun (the rerun-if-env-changed
    // line) and change the embedded version (the env lookup in main()).
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    init_repo_with_commit(dir);
    let first = build_script_output(dir, Some(ENV_FULL));
    let second = build_script_output(dir, Some(ENV_FULL_B));
    for stdout in [&first, &second] {
        assert!(
            stdout
                .lines()
                .any(|l| l == "cargo:rerun-if-env-changed=COPIA_GIT_SHA"),
            "rerun-if-env-changed line missing: {stdout}"
        );
    }
    assert_eq!(embedded_sha(&first), "0123456");
    assert_eq!(embedded_sha(&second), "fedcba9");
}
