//! Benchmark comparing copia vs rsync performance.
//!
//! Run with: cargo bench --bench `rsync_comparison` --features async

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use copia::async_sync::AsyncCopiaSync;

/// Test scenario configuration
struct Scenario {
    name: &'static str,
    size: usize,
    change_percent: u8,
}

struct BenchResult {
    scenario: String,
    size: usize,
    rsync_time: Duration,
    copia_time: Duration,
    copia_bytes_literal: u64,
    copia_bytes_matched: u64,
}

fn create_test_file(path: &PathBuf, size: usize, seed: u8) {
    let mut file = File::create(path).expect("Failed to create file");
    let mut data = vec![0u8; size];

    // Generate pseudo-random but deterministic data
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = ((i.wrapping_mul(31).wrapping_add(seed as usize)) % 256) as u8;
    }

    file.write_all(&data).expect("Failed to write file");
}

fn create_modified_copy(src: &PathBuf, dest: &PathBuf, change_percent: u8) {
    let mut data = fs::read(src).expect("Failed to read source");

    if change_percent == 0 {
        // Identical - just copy
        fs::write(dest, &data).expect("Failed to write");
        return;
    }

    if change_percent >= 100 {
        // Completely different
        for byte in &mut data {
            *byte = byte.wrapping_add(128);
        }
        fs::write(dest, &data).expect("Failed to write");
        return;
    }

    // Modify a contiguous region (change_percent of the file)
    // This simulates realistic edits where changes are localized
    let modify_bytes = data.len() * change_percent as usize / 100;
    let start = data.len() / 4; // Start at 25% into file

    for i in start..start.saturating_add(modify_bytes).min(data.len()) {
        data[i] = data[i].wrapping_add(1);
    }

    fs::write(dest, &data).expect("Failed to write");
}

#[tokio::main]
async fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                      COPIA vs RSYNC BENCHMARK COMPARISON                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Testing rsync-style delta synchronization performance                        ║");
    println!("║  Block size: 2048 bytes                                                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let base_path = temp_dir.path();

    let scenarios = vec![
        // Small files
        Scenario { name: "1KB identical", size: 1024, change_percent: 0 },
        Scenario { name: "1KB 10% changed", size: 1024, change_percent: 10 },
        Scenario { name: "1KB 50% changed", size: 1024, change_percent: 50 },
        // Medium files
        Scenario { name: "100KB identical", size: 100 * 1024, change_percent: 0 },
        Scenario { name: "100KB 10% changed", size: 100 * 1024, change_percent: 10 },
        Scenario { name: "100KB 50% changed", size: 100 * 1024, change_percent: 50 },
        // Large files
        Scenario { name: "1MB identical", size: 1024 * 1024, change_percent: 0 },
        Scenario { name: "1MB 5% changed", size: 1024 * 1024, change_percent: 5 },
        Scenario { name: "1MB 10% changed", size: 1024 * 1024, change_percent: 10 },
        // Very large files
        Scenario { name: "10MB identical", size: 10 * 1024 * 1024, change_percent: 0 },
        Scenario { name: "10MB 1% changed", size: 10 * 1024 * 1024, change_percent: 1 },
        Scenario { name: "10MB 5% changed", size: 10 * 1024 * 1024, change_percent: 5 },
        Scenario { name: "10MB 100% different", size: 10 * 1024 * 1024, change_percent: 100 },
    ];

    let mut results = Vec::new();
    let sync = AsyncCopiaSync::with_block_size(2048);

    for scenario in &scenarios {
        print!("Running: {:<25}", scenario.name);
        std::io::stdout().flush().ok();

        // Create original file (the "basis" that exists at destination)
        let original_path = base_path.join("original.dat");
        create_test_file(&original_path, scenario.size, 42);

        // Create modified file (the "source" we want to sync)
        let modified_path = base_path.join("modified.dat");
        create_modified_copy(&original_path, &modified_path, scenario.change_percent);

        // Paths for rsync and copia destinations
        let rsync_dest = base_path.join("rsync_dest.dat");
        let copia_dest = base_path.join("copia_dest.dat");

        // Determine iterations based on file size
        let iterations = if scenario.size < 100_000 { 20 } else if scenario.size < 1_000_000 { 10 } else { 5 };

        // Benchmark rsync (using delta algorithm)
        let mut rsync_total = Duration::ZERO;
        for _ in 0..iterations {
            // Reset destination to original state
            fs::copy(&original_path, &rsync_dest).expect("Failed to copy");

            let start = Instant::now();
            // Use rsync to sync modified -> dest (which has original content)
            // --no-whole-file forces delta algorithm
            let _ = Command::new("rsync")
                .arg("--no-whole-file")
                .arg(&modified_path)
                .arg(&rsync_dest)
                .output()
                .expect("Failed to run rsync");
            rsync_total += start.elapsed();
        }
        let rsync_avg = rsync_total / iterations as u32;

        // Benchmark copia
        let mut copia_total = Duration::ZERO;
        let mut copia_bytes_literal = 0u64;
        let mut copia_bytes_matched = 0u64;

        for _ in 0..iterations {
            // Reset destination to original state
            fs::copy(&original_path, &copia_dest).expect("Failed to copy");

            let start = Instant::now();
            let result = sync.sync_files(&modified_path, &copia_dest).await.expect("Sync failed");
            copia_total += start.elapsed();
            copia_bytes_literal = result.bytes_literal;
            copia_bytes_matched = result.bytes_matched;
        }
        let copia_avg = copia_total / iterations as u32;

        // Verify copia produced correct output
        let modified_content = fs::read(&modified_path).expect("Failed to read modified");
        let copia_content = fs::read(&copia_dest).expect("Failed to read copia result");
        assert_eq!(copia_content, modified_content, "copia output mismatch for {}", scenario.name);

        let speedup = rsync_avg.as_secs_f64() / copia_avg.as_secs_f64();
        let speedup_str = if speedup >= 1.0 {
            format!("\x1b[32m{speedup:.1}x faster\x1b[0m")
        } else {
            format!("\x1b[31m{:.1}x slower\x1b[0m", 1.0 / speedup)
        };

        println!(" rsync: {:>8.2}ms  copia: {:>8.2}ms  {}",
            rsync_avg.as_secs_f64() * 1000.0,
            copia_avg.as_secs_f64() * 1000.0,
            speedup_str
        );

        results.push(BenchResult {
            scenario: scenario.name.to_string(),
            size: scenario.size,
            rsync_time: rsync_avg,
            copia_time: copia_avg,
            copia_bytes_literal,
            copia_bytes_matched,
        });
    }

    // Print summary
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                   SUMMARY                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_rsync: Duration = results.iter().map(|r| r.rsync_time).sum();
    let total_copia: Duration = results.iter().map(|r| r.copia_time).sum();
    let overall_speedup = total_rsync.as_secs_f64() / total_copia.as_secs_f64();

    println!("  Total rsync time:  {:>10.2}ms", total_rsync.as_secs_f64() * 1000.0);
    println!("  Total copia time:  {:>10.2}ms", total_copia.as_secs_f64() * 1000.0);
    println!();

    if overall_speedup >= 1.0 {
        println!("  \x1b[32;1m★ Overall: copia is {overall_speedup:.2}x FASTER than rsync\x1b[0m");
    } else {
        println!("  Overall: copia is {:.2}x slower than rsync", 1.0 / overall_speedup);
    }
    println!();

    // Detailed table
    println!("┌────────────────────────────┬──────────┬────────────┬────────────┬──────────────────┐");
    println!("│ Scenario                   │ Size     │ rsync (ms) │ copia (ms) │ Speedup          │");
    println!("├────────────────────────────┼──────────┼────────────┼────────────┼──────────────────┤");

    for result in &results {
        let speedup = result.rsync_time.as_secs_f64() / result.copia_time.as_secs_f64();
        let speedup_str = if speedup >= 1.0 {
            format!("{speedup:.1}x faster")
        } else {
            format!("{:.1}x slower", 1.0 / speedup)
        };

        let size_str = if result.size >= 1024 * 1024 {
            format!("{}MB", result.size / (1024 * 1024))
        } else {
            format!("{}KB", result.size / 1024)
        };

        println!("│ {:<26} │ {:>8} │ {:>10.2} │ {:>10.2} │ {:>16} │",
            result.scenario,
            size_str,
            result.rsync_time.as_secs_f64() * 1000.0,
            result.copia_time.as_secs_f64() * 1000.0,
            speedup_str
        );
    }

    println!("└────────────────────────────┴──────────┴────────────┴────────────┴──────────────────┘");
    println!();

    // Delta efficiency
    println!("┌────────────────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Scenario                   │ Bytes Matched│ Bytes Literal│ Efficiency   │");
    println!("├────────────────────────────┼──────────────┼──────────────┼──────────────┤");

    for result in &results {
        let total = result.copia_bytes_matched + result.copia_bytes_literal;
        let efficiency = if total > 0 {
            result.copia_bytes_matched as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        println!("│ {:<26} │ {:>12} │ {:>12} │ {:>10.1}% │",
            result.scenario,
            format_bytes(result.copia_bytes_matched),
            format_bytes(result.copia_bytes_literal),
            efficiency
        );
    }

    println!("└────────────────────────────┴──────────────┴──────────────┴──────────────┘");
    println!();

    // Note about rsync
    println!("Note: rsync times include process spawn overhead (~40ms on this system).");
    println!("      For a more fair comparison of the algorithm itself, see the criterion benches.");
    println!();
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}
