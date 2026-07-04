#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
//! End-to-end throughput benchmarks for the recursive/local sync engine.
//!
//! The existing `benchmarks.rs` covers the pure delta/signature/patch
//! primitives, and `rsync_comparison.rs` shells out to the `rsync` binary.
//! Neither exercises [`AsyncCopiaSync::sync_files`] — the real
//! file-to-file transfer path (read basis + source, generate signature,
//! compute delta, patch, atomic write) that the recursive directory-sync
//! orchestration in `src/bin/copia/` drives one file at a time.
//!
//! This bench builds a deterministic temp-directory tree on disk and
//! benchmarks a local -> local file sync end-to-end on a Tokio runtime,
//! across several file sizes and source/basis similarity ratios. That
//! covers the recursive engine's per-file hot path without needing SSH.
//!
//! # Statistical Methodology
//!
//! - **Sample size**: 10 iterations per scenario (disk I/O + atomic
//!   rename have higher variance and are slower, so fewer samples keep CI
//!   fast while remaining stable).
//! - **Reset**: the destination ("basis") file is rewritten to its
//!   original content before every timed iteration
//!   ([`BatchSize::PerIteration`]) so each measurement exercises the full
//!   delta path rather than the identical-file fast path.
//! - **Throughput**: reported in bytes of the source file synced.

use std::fs;
use std::path::{Path, PathBuf};

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use tempfile::TempDir;

use copia::async_sync::AsyncCopiaSync;

/// Block size used across all scenarios (crate default, valid 512..=65536).
const BLOCK_SIZE: usize = 2048;

/// Generate deterministic pseudo-random bytes for a basis file.
fn basis_bytes(size: usize, seed: u8) -> Vec<u8> {
    let mut data = vec![0u8; size];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i.wrapping_mul(31).wrapping_add(seed as usize)) % 256) as u8;
    }
    data
}

/// Derive a source file from a basis by mutating `change_percent` of its
/// bytes in a contiguous region (models a localized edit), matching the
/// approach used by `rsync_comparison.rs`.
fn source_bytes(basis: &[u8], change_percent: u8) -> Vec<u8> {
    let mut data = basis.to_vec();

    if change_percent == 0 {
        return data;
    }

    if change_percent >= 100 {
        for byte in &mut data {
            *byte = byte.wrapping_add(128);
        }
        return data;
    }

    let modify_bytes = data.len() * change_percent as usize / 100;
    let start = data.len() / 4;
    let end = start.saturating_add(modify_bytes).min(data.len());
    for byte in &mut data[start..end] {
        *byte = byte.wrapping_add(1);
    }
    data
}

/// One benchmark scenario: a file size + a source/basis similarity.
struct Scenario {
    label: &'static str,
    size: usize,
    change_percent: u8,
}

fn bench_sync_files(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("failed to build tokio runtime");
    let sync = AsyncCopiaSync::with_block_size(BLOCK_SIZE);

    // Scratch tree lives for the whole bench; each scenario gets its own
    // source + dest file inside it.
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let root: &Path = temp_dir.path();

    let scenarios = [
        // 1 KiB
        Scenario {
            label: "identical",
            size: 1024,
            change_percent: 0,
        },
        Scenario {
            label: "50pct_changed",
            size: 1024,
            change_percent: 50,
        },
        Scenario {
            label: "100pct_different",
            size: 1024,
            change_percent: 100,
        },
        // 64 KiB
        Scenario {
            label: "identical",
            size: 64 * 1024,
            change_percent: 0,
        },
        Scenario {
            label: "50pct_changed",
            size: 64 * 1024,
            change_percent: 50,
        },
        Scenario {
            label: "100pct_different",
            size: 64 * 1024,
            change_percent: 100,
        },
        // 1 MiB
        Scenario {
            label: "identical",
            size: 1024 * 1024,
            change_percent: 0,
        },
        Scenario {
            label: "50pct_changed",
            size: 1024 * 1024,
            change_percent: 50,
        },
        Scenario {
            label: "100pct_different",
            size: 1024 * 1024,
            change_percent: 100,
        },
    ];

    let mut group = c.benchmark_group("sync_files");
    // Small sample size: disk I/O bound and slower than the pure-CPU
    // primitives, so cap iterations to keep CI fast (task requirement).
    group.sample_size(10);

    for (idx, scenario) in scenarios.iter().enumerate() {
        let basis = basis_bytes(scenario.size, 42);
        let source = source_bytes(&basis, scenario.change_percent);

        // Distinct paths per scenario so parallel-nothing but clean state.
        let source_path: PathBuf = root.join(format!("source_{idx}.dat"));
        let dest_path: PathBuf = root.join(format!("dest_{idx}.dat"));
        fs::write(&source_path, &source).expect("failed to write source file");

        group.throughput(Throughput::Bytes(scenario.size as u64));

        let bench_id = BenchmarkId::new(scenario.label, format_size(scenario.size));

        group.bench_function(bench_id, |b| {
            b.iter_batched(
                // Setup (untimed): reset the destination to the original
                // basis content so every timed iteration runs the full
                // signature -> delta -> patch path.
                || {
                    fs::write(&dest_path, &basis).expect("failed to reset dest file");
                },
                // Routine (timed): drive the async end-to-end sync.
                |()| {
                    runtime
                        .block_on(sync.sync_files(black_box(&source_path), black_box(&dest_path)))
                        .expect("sync_files failed")
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

/// Human-readable size label for benchmark ids (e.g. `1KiB`, `1MiB`).
fn format_size(size: usize) -> String {
    if size >= 1024 * 1024 {
        format!("{}MiB", size / (1024 * 1024))
    } else {
        format!("{}KiB", size / 1024)
    }
}

criterion_group!(benches, bench_sync_files);
criterion_main!(benches);
