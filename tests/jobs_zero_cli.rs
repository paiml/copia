//! `copia sync --recursive --jobs 0` must not hang. A zero-permit semaphore
//! blocks every transfer forever, so zero is either rejected at parse time with
//! a message naming `--jobs`, or the run completes. A hang fails the test.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn jobs_zero_is_rejected_or_completes_never_hangs() {
    let base = tempfile::tempdir().unwrap();
    let src = base.path().join("src");
    let dst = base.path().join("dst");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("a.txt"), b"alpha").unwrap();
    let out_path = base.path().join("stdout.txt");
    let err_path = base.path().join("stderr.txt");

    let mut child = Command::new(env!("CARGO_BIN_EXE_copia"))
        .arg("sync")
        .arg("--recursive")
        .arg("--jobs")
        .arg("0")
        .arg(&src)
        .arg(&dst)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&out_path).unwrap())
        .stderr(std::fs::File::create(&err_path).unwrap())
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("copia sync --jobs 0 hung: no exit within 20 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    };

    if status.success() {
        return;
    }
    let stderr = std::fs::read_to_string(&err_path).unwrap();
    assert!(
        stderr.contains("--jobs"),
        "non-zero exit for --jobs 0 must name --jobs in the error; stderr was: {stderr}"
    );
}
