//! Every bash `run: |` block under `.github/workflows` must parse.
//!
//! The nightly's glibc-floor step (#54) was committed double-spaced: a blank
//! line after a `\` continuation ends the command, and the next line, which
//! starts with `|`, is a syntax error. The 24h activity gate skipped every
//! nightly for 33 days, so the step first ran on 2026-09-25 (#72) and failed
//! both Linux builds with `syntax error near unexpected token '|'`. Nothing
//! parsed the step before a runner did. This does, for every block.
//!
//! `bash -n` alone cannot see that shape: bash 5.1 defers parsing a `$(...)`
//! until it runs, and `bash -n` accepts the #54 block verbatim. So the
//! continuation form is checked directly, and `bash -n` covers the rest.

#![cfg(unix)]
#![allow(clippy::expect_used)]

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The `shell:` a step declares, read from the step's own keys.
fn step_shell<'a>(lines: &[&'a str], run_idx: usize, sib_indent: usize) -> Option<&'a str> {
    let step_start = (0..=run_idx).rev().find(|&j| {
        let l = lines[j];
        indent_of(l) + 2 == sib_indent && l.trim_start().starts_with("- ")
    })?;
    let mut j = step_start;
    while j < lines.len() {
        let l = lines[j];
        let t = l.trim_start();
        if j > step_start && !t.is_empty() && indent_of(l) < sib_indent {
            break;
        }
        let key = if j == step_start {
            t.trim_start_matches("- ")
        } else {
            t
        };
        if (j == step_start || indent_of(l) == sib_indent) && key.starts_with("shell:") {
            return Some(key["shell:".len()..].trim());
        }
        j += 1;
    }
    None
}

/// `(first line number, dedented script)` for each literal bash `run: |` block.
fn run_blocks(src: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let body = line.trim_start();
        let dashed = body.starts_with("- ");
        let key = body.trim_start_matches("- ").trim_end();
        let run_idx = i;
        i += 1;
        if !(key == "run: |" || key == "run: |-") {
            continue;
        }
        let key_indent = indent_of(line);
        // `- run: |` sits at the dash; its sibling keys sit two columns right.
        let sib_indent = if dashed { key_indent + 2 } else { key_indent };
        let start = i;
        let mut script = Vec::new();
        let mut indent = None;
        while i < lines.len() {
            let l = lines[i];
            let t = l.trim_start();
            if !t.is_empty() && indent_of(l) <= sib_indent {
                break;
            }
            let ind = *indent.get_or_insert_with(|| indent_of(l));
            script.push(if t.is_empty() {
                ""
            } else {
                l.get(ind..).unwrap_or(t)
            });
            i += 1;
        }
        if matches!(
            step_shell(&lines, run_idx, sib_indent),
            None | Some("bash" | "sh")
        ) {
            blocks.push((start + 1, script.join("\n")));
        }
    }
    blocks
}

/// GitHub expressions are substituted before bash sees the script.
fn strip_expressions(script: &str) -> String {
    let mut out = String::with_capacity(script.len());
    let mut rest = script;
    while let Some(open) = rest.find("${{") {
        out.push_str(&rest[..open]);
        let Some(close) = rest[open..].find("}}") else {
            rest = &rest[open..];
            break;
        };
        out.push('X');
        rest = &rest[open + close + 2..];
    }
    out.push_str(rest);
    out
}

/// A `\` continuation whose next line is blank: the command ends there.
///
/// An odd run of trailing backslashes continues; an even run is literal. Not
/// quote- or heredoc-aware: a `\`-then-blank inside one is a loud false
/// positive, never a silent pass.
fn broken_continuation(script: &str) -> Option<usize> {
    let lines: Vec<&str> = script.lines().collect();
    lines.windows(2).position(|w| {
        let tail = w[0].len() - w[0].trim_end_matches('\\').len();
        tail % 2 == 1 && w[1].trim().is_empty()
    })
}

fn bash_rejects(script: &str) -> Option<String> {
    let mut child = Command::new("bash")
        .arg("-n")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("bash must be on PATH to check workflow run blocks");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(script.as_bytes())
        .expect("write script to bash -n");
    let out = child.wait_with_output().expect("bash -n exits");
    (!out.status.success()).then(|| String::from_utf8_lossy(&out.stderr).into_owned())
}

fn block_defect(script: &str) -> Option<String> {
    let script = strip_expressions(script);
    if let Some(n) = broken_continuation(&script) {
        return Some(format!(
            "line {}: `\\` continuation followed by a blank line",
            n + 1
        ));
    }
    bash_rejects(&script)
}

#[test]
fn every_workflow_run_block_parses_as_bash() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    let mut checked = 0;
    let mut failures = Vec::new();
    for entry in fs::read_dir(&dir).expect("read .github/workflows") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        let src = fs::read_to_string(&path).expect("read workflow");
        for (line, script) in run_blocks(&src) {
            checked += 1;
            if let Some(err) = block_defect(&script) {
                failures.push(format!("{}:{line}: {}", path.display(), err.trim()));
            }
        }
    }
    // Zero blocks checked would pass vacuously; the workflows hold dozens.
    assert!(
        checked >= 10,
        "only {checked} run blocks found — the extractor is blind"
    );
    assert!(
        failures.is_empty(),
        "run blocks bash cannot parse:\n{}",
        failures.join("\n")
    );
}

#[test]
fn a_blank_line_after_a_continuation_is_caught() {
    let yaml = "    steps:\n      - name: x\n        run: |\n          a=$(echo 1 \\\n\n                 | cat)\n";
    let blocks = run_blocks(yaml);
    assert_eq!(blocks.len(), 1);
    assert!(
        block_defect(&blocks[0].1).is_some(),
        "the #54 shape must not pass"
    );
    let spaces = yaml.replace("\\\n\n", "\\\n   \n");
    assert!(
        block_defect(&run_blocks(&spaces)[0].1).is_some(),
        "a whitespace-only line too"
    );
    let fixed = yaml.replace("\\\n\n", "\\\n");
    assert!(block_defect(&run_blocks(&fixed)[0].1).is_none());
}

#[test]
fn a_plain_syntax_error_is_caught() {
    let yaml = "      - run: |\n          if true; then\n            echo x\n";
    assert!(
        block_defect(&run_blocks(yaml)[0].1).is_some(),
        "unterminated if must not pass"
    );
}

#[test]
fn a_pwsh_step_is_not_read_as_bash() {
    let before = "      - name: w\n        shell: pwsh\n        run: |\n          foreach ($f in @(1)) { }\n";
    let after = "      - name: w\n        run: |\n          foreach ($f in @(1)) { }\n        shell: pwsh\n      - run: |\n          echo ok\n";
    assert!(run_blocks(before).is_empty());
    assert_eq!(run_blocks(after).len(), 1, "only the second, bash, step");
}

#[test]
fn expressions_do_not_break_the_parse() {
    let s = strip_expressions("echo \"${{ matrix.target }}\" ${{ github.sha }}");
    assert_eq!(s, "echo \"X\" X");
}

#[test]
fn a_dashed_run_does_not_absorb_its_sibling_keys() {
    let yaml = "      - run: |\n          echo ok\n        shell: pwsh\n      - run: |\n          echo a\n        env:\n          X: 1\n";
    let blocks = run_blocks(yaml);
    assert_eq!(blocks.len(), 1, "the pwsh step is skipped");
    assert_eq!(blocks[0].1.trim_end(), "echo a", "env: is not script");
}

#[test]
fn an_odd_run_of_backslashes_is_a_continuation() {
    assert_eq!(broken_continuation("echo a \\\\\\\n\necho b"), Some(0));
    assert_eq!(broken_continuation("echo a \\\\\n\necho b"), None);
}
