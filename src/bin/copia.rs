//! Copia CLI - rsync-style file synchronization.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tokio::sync::Semaphore;

use copia::async_sync::AsyncCopiaSync;

/// Represents a file location - either local or remote via SSH
#[derive(Debug, Clone)]
enum FileLocation {
    Local(PathBuf),
    Remote { host: String, path: String },
}

impl FileLocation {
    /// Parse a path string into a `FileLocation`
    /// Supports formats: `/local/path`, `host:path`, `host:~/path`
    fn parse(s: &str) -> Self {
        // Check for host:path format (but not Windows drive letters like C:)
        if let Some(colon_pos) = s.find(':') {
            let before_colon = &s[..colon_pos];
            // If it's a single letter, it might be a Windows drive
            if before_colon.len() > 1 && !before_colon.contains('/') && !before_colon.contains('\\')
            {
                return Self::Remote {
                    host: before_colon.to_string(),
                    path: s[colon_pos + 1..].to_string(),
                };
            }
        }
        Self::Local(PathBuf::from(s))
    }
}

/// Copia - Pure Rust rsync-style synchronization for Sovereign AI
#[derive(Parser)]
#[command(name = "copia")]
#[command(author = "Pragmatic AI Labs")]
#[command(version)]
#[command(about = "rsync-style file synchronization in pure Rust")]
#[command(long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Synchronize source to destination (supports host:path for SSH)
    Sync {
        /// Source path (local path or host:path for SSH)
        #[arg(required = true)]
        source: String,

        /// Destination path (local path or host:path for SSH)
        #[arg(required = true)]
        dest: String,

        /// Block size for signature generation (512-65536, power of 2)
        #[arg(short, long, default_value = "2048")]
        block_size: usize,

        /// Recursively sync directories
        #[arg(short, long)]
        recursive: bool,

        /// Number of parallel transfer jobs
        #[arg(short, long, default_value = "4")]
        jobs: usize,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate signature for a file
    Signature {
        /// File to generate signature for
        #[arg(required = true)]
        file: PathBuf,

        /// Output signature file (default: <file>.sig)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Block size (512-65536, power of 2)
        #[arg(short, long, default_value = "2048")]
        block_size: usize,
    },

    /// Compute delta between source and signature
    Delta {
        /// Source file (new version)
        #[arg(required = true)]
        source: PathBuf,

        /// Signature file (of old version)
        #[arg(required = true)]
        signature: PathBuf,

        /// Output delta file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Apply delta to basis file
    Patch {
        /// Basis file (old version)
        #[arg(required = true)]
        basis: PathBuf,

        /// Delta file
        #[arg(required = true)]
        delta: PathBuf,

        /// Output file (reconstructed new version)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Sync {
            source,
            dest,
            block_size,
            recursive,
            jobs,
            verbose,
        } => {
            let src_loc = FileLocation::parse(&source);
            let dest_loc = FileLocation::parse(&dest);
            if recursive {
                run_sync_recursive(src_loc, dest_loc, block_size, jobs, verbose).await
            } else {
                run_sync(src_loc, dest_loc, block_size, verbose).await
            }
        }
        Commands::Signature {
            file,
            output,
            block_size,
        } => run_signature(&file, output, block_size).await,
        Commands::Delta {
            source,
            signature,
            output,
        } => run_delta(&source, &signature, output).await,
        Commands::Patch {
            basis,
            delta,
            output,
        } => run_patch(&basis, &delta, output).await,
    }
}

// ── Recursive directory sync ──────────────────────────────────────────

/// Discover all files under a directory recursively, returning paths relative to root.
fn discover_local_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let entries = std::fs::read_dir(&dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.is_file() {
                let rel = path.strip_prefix(root)?.to_path_buf();
                files.push(rel);
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Collect unique directory paths from a list of relative file paths.
fn collect_dirs(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = std::collections::BTreeSet::new();
    for file in files {
        let mut cur = file.as_path();
        while let Some(parent) = cur.parent() {
            if parent.as_os_str().is_empty() {
                break;
            }
            dirs.insert(parent.to_path_buf());
            cur = parent;
        }
    }
    dirs.into_iter().collect()
}

/// Create directories on a remote host via batched SSH calls.
/// Batches into chunks of 200 to avoid "Argument list too long" errors.
async fn create_remote_dirs(
    host: &str,
    remote_root: &str,
    dirs: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write;

    // Always ensure root exists
    let output = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!("mkdir -p '{remote_root}'"))
        .output()
        .await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to create remote root: {stderr}").into());
    }

    if dirs.is_empty() {
        return Ok(());
    }

    // Batch directories into chunks to avoid exceeding arg length limits
    for (i, chunk) in dirs.chunks(200).enumerate() {
        let mut mkdir_cmd = String::from("mkdir -p");
        for dir in chunk {
            let full = format!("{}/{}", remote_root, dir.display());
            let _ = write!(mkdir_cmd, " '{full}'");
        }

        let output = tokio::process::Command::new("ssh")
            .arg(host)
            .arg(&mkdir_cmd)
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "Failed to create remote directories (batch {}): {stderr}",
                i + 1
            )
            .into());
        }
    }

    Ok(())
}

/// Transfer a single file from local to remote via SSH streaming.
/// Uses streaming I/O — does not read entire file into memory.
async fn transfer_file_to_remote(
    local_path: &Path,
    host: &str,
    remote_path: &str,
) -> Result<u64, String> {
    use tokio::io::AsyncReadExt;

    let metadata = tokio::fs::metadata(local_path)
        .await
        .map_err(|e| format!("{}: {e}", local_path.display()))?;
    let file_size = metadata.len();

    let mut child = tokio::process::Command::new("ssh")
        .arg(host)
        .arg(format!("cat > '{}'", remote_path.replace('\'', "'\\''")))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("ssh spawn: {e}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to open SSH stdin".to_string())?;

    // Stream file in chunks instead of reading all into memory
    let mut file = tokio::fs::File::open(local_path)
        .await
        .map_err(|e| format!("open {}: {e}", local_path.display()))?;
    let mut buf = vec![0u8; 256 * 1024]; // 256KB chunks
    loop {
        let n = file
            .read(&mut buf)
            .await
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        tokio::io::AsyncWriteExt::write_all(&mut stdin, &buf[..n])
            .await
            .map_err(|e| format!("write: {e}"))?;
    }
    drop(stdin);

    let result = child
        .wait_with_output()
        .await
        .map_err(|e| format!("ssh wait: {e}"))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!("SSH failed for {}: {stderr}", local_path.display()));
    }

    Ok(file_size)
}

/// Format bytes into a human-readable string.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * 1024 * 1024 * 1024;

    if bytes >= TIB {
        format!("{:.2} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

async fn run_sync_recursive(
    source: FileLocation,
    dest: FileLocation,
    _block_size: usize,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match (&source, &dest) {
        (FileLocation::Local(local_src), FileLocation::Remote { host, path }) => {
            run_sync_dir_local_to_remote(local_src, host, path, jobs, verbose).await
        }
        (FileLocation::Local(local_src), FileLocation::Local(local_dest)) => {
            run_sync_dir_local_to_local(local_src, local_dest, jobs, verbose).await
        }
        _ => Err("Recursive sync currently supports local->remote and local->local".into()),
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn run_sync_dir_local_to_remote(
    local_root: &Path,
    host: &str,
    remote_root: &str,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    // 1. Discover files
    eprintln!("Discovering files in {}...", local_root.display());
    let files = discover_local_files(local_root)?;
    if files.is_empty() {
        eprintln!("No files found.");
        return Ok(());
    }

    // Compute total size
    let mut total_size: u64 = 0;
    for f in &files {
        let full = local_root.join(f);
        if let Ok(m) = std::fs::metadata(&full) {
            total_size += m.len();
        }
    }

    eprintln!(
        "Found {} files ({}) across {} directories",
        files.len(),
        format_bytes(total_size),
        collect_dirs(&files).len()
    );

    // 2. Create directory structure on remote
    eprintln!("Creating remote directory structure...");
    let dirs = collect_dirs(&files);
    create_remote_dirs(host, remote_root, &dirs).await?;

    // 3. Transfer files with bounded concurrency
    eprintln!("Transferring with {jobs} parallel jobs...");

    let semaphore = Arc::new(Semaphore::new(jobs));
    let bytes_transferred = Arc::new(AtomicU64::new(0));
    let files_done = Arc::new(AtomicU64::new(0));
    let files_failed = Arc::new(AtomicU64::new(0));
    let total_files = files.len() as u64;

    let mut handles = Vec::with_capacity(files.len());

    for rel_path in files {
        let local_file = local_root.join(&rel_path);
        let remote_file = format!("{}/{}", remote_root, rel_path.display());
        let host = host.to_string();
        let sem = Arc::clone(&semaphore);
        let bytes_tx = Arc::clone(&bytes_transferred);
        let done = Arc::clone(&files_done);
        let failed = Arc::clone(&files_failed);

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await;
            match transfer_file_to_remote(&local_file, &host, &remote_file).await {
                Ok(size) => {
                    bytes_tx.fetch_add(size, Ordering::Relaxed);
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 50 == 0 || n == total_files {
                        let transferred = bytes_tx.load(Ordering::Relaxed);
                        eprintln!(
                            "  [{}/{}] {} transferred",
                            n,
                            total_files,
                            format_bytes(transferred)
                        );
                    }
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    eprintln!("  FAILED {}: {e}", rel_path.display());
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all transfers
    for handle in handles {
        if let Err(e) = handle.await {
            eprintln!("Task error: {e}");
        }
    }

    let elapsed = start.elapsed();
    let total_tx = bytes_transferred.load(Ordering::Relaxed);
    let done_count = files_done.load(Ordering::Relaxed);
    let fail_count = files_failed.load(Ordering::Relaxed);
    let speed = if elapsed.as_secs_f64() > 0.0 {
        total_tx as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("\nComplete: {done_count} files synced, {fail_count} failed");
    println!(
        "Transferred {} in {:.1}s ({}/s)",
        format_bytes(total_tx),
        elapsed.as_secs_f64(),
        format_bytes(speed.max(0.0) as u64)
    );

    if verbose {
        eprintln!("Source: {}", local_root.display());
        eprintln!("Destination: {host}:{remote_root}");
        eprintln!("Jobs: {jobs}");
    }

    if fail_count > 0 {
        Err(format!("{fail_count} files failed to transfer").into())
    } else {
        Ok(())
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn run_sync_dir_local_to_local(
    local_src: &Path,
    local_dest: &Path,
    jobs: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    eprintln!("Discovering files in {}...", local_src.display());
    let files = discover_local_files(local_src)?;
    if files.is_empty() {
        eprintln!("No files found.");
        return Ok(());
    }

    let mut total_size: u64 = 0;
    for f in &files {
        let full = local_src.join(f);
        if let Ok(m) = std::fs::metadata(&full) {
            total_size += m.len();
        }
    }

    eprintln!("Found {} files ({})", files.len(), format_bytes(total_size));

    // Create directory structure
    let dirs = collect_dirs(&files);
    for dir in &dirs {
        let dest_dir = local_dest.join(dir);
        std::fs::create_dir_all(&dest_dir)?;
    }
    std::fs::create_dir_all(local_dest)?;

    let sync = AsyncCopiaSync::with_block_size(2048);
    let semaphore = Arc::new(Semaphore::new(jobs));
    let bytes_transferred = Arc::new(AtomicU64::new(0));
    let files_done = Arc::new(AtomicU64::new(0));
    let total_files = files.len() as u64;

    let mut handles = Vec::with_capacity(files.len());
    let sync = Arc::new(sync);

    for rel_path in files {
        let src_file = local_src.join(&rel_path);
        let dst_file = local_dest.join(&rel_path);
        let sem = Arc::clone(&semaphore);
        let bytes_tx = Arc::clone(&bytes_transferred);
        let done = Arc::clone(&files_done);
        let sync = Arc::clone(&sync);

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await;
            match sync.sync_files(&src_file, &dst_file).await {
                Ok(result) => {
                    bytes_tx.fetch_add(result.source_size, Ordering::Relaxed);
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 100 == 0 || n == total_files {
                        eprintln!("  [{n}/{total_files}] files synced");
                    }
                }
                Err(e) => {
                    eprintln!("  FAILED {}: {e}", rel_path.display());
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        if let Err(e) = handle.await {
            eprintln!("Task error: {e}");
        }
    }

    let elapsed = start.elapsed();
    let total_tx = bytes_transferred.load(Ordering::Relaxed);
    let speed = if elapsed.as_secs_f64() > 0.0 {
        total_tx as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "\nComplete: {} files synced",
        files_done.load(Ordering::Relaxed)
    );
    println!(
        "Transferred {} in {:.1}s ({}/s)",
        format_bytes(total_tx),
        elapsed.as_secs_f64(),
        format_bytes(speed.max(0.0) as u64)
    );

    if verbose {
        eprintln!("Source: {}", local_src.display());
        eprintln!("Destination: {}", local_dest.display());
    }

    Ok(())
}

// ── Single-file sync ──────────────────────────────────────────────────

async fn run_sync(
    source: FileLocation,
    dest: FileLocation,
    block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_block_size(block_size)?;

    match (&source, &dest) {
        (FileLocation::Local(local_src), FileLocation::Local(local_dest)) => {
            run_sync_local_to_local(local_src, local_dest, block_size, verbose).await
        }
        (FileLocation::Local(local_src), FileLocation::Remote { host, path }) => {
            run_sync_local_to_remote(local_src, host, path, block_size, verbose).await
        }
        (FileLocation::Remote { host, path }, FileLocation::Local(local_dest)) => {
            run_sync_remote_to_local(host, path, local_dest, block_size, verbose).await
        }
        (
            FileLocation::Remote {
                host: src_host,
                path: src_path,
            },
            FileLocation::Remote {
                host: dst_host,
                path: dst_path,
            },
        ) => {
            Err(format!(
                "Remote-to-remote sync not yet supported: {src_host}:{src_path} -> {dst_host}:{dst_path}"
            )
            .into())
        }
    }
}

async fn run_sync_local_to_local(
    source: &PathBuf,
    dest: &PathBuf,
    block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let sync = AsyncCopiaSync::with_block_size(block_size);

    if verbose {
        eprintln!("Syncing {} -> {}", source.display(), dest.display());
        eprintln!("Block size: {block_size}");
    }

    let result = sync.sync_files(source, dest).await?;

    if verbose {
        eprintln!("Source size: {} bytes", result.source_size);
        eprintln!("Basis size: {} bytes", result.basis_size);
        eprintln!("Bytes matched: {} bytes", result.bytes_matched);
        eprintln!("Bytes literal: {} bytes", result.bytes_literal);
        eprintln!(
            "Compression ratio: {:.1}%",
            result.compression_ratio() * 100.0
        );
        eprintln!(
            "Bandwidth savings: {:.1}%",
            result.bandwidth_savings() * 100.0
        );
    }

    println!(
        "Synced {} ({} bytes matched, {} bytes literal)",
        dest.display(),
        result.bytes_matched,
        result.bytes_literal
    );

    Ok(())
}

async fn run_sync_local_to_remote(
    source: &Path,
    host: &str,
    remote_path: &str,
    _block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if verbose {
        eprintln!("Syncing {} -> {}:{}", source.display(), host, remote_path);
    }

    let size = transfer_file_to_remote(source, host, remote_path)
        .await
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    if verbose {
        eprintln!("Transferred: {size} bytes");
    }

    println!(
        "Synced {host}:{remote_path} ({} transferred)",
        format_bytes(size)
    );

    Ok(())
}

async fn run_sync_remote_to_local(
    host: &str,
    remote_path: &str,
    dest: &PathBuf,
    _block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::process::Command;

    if verbose {
        eprintln!("Syncing {}:{} -> {}", host, remote_path, dest.display());
    }

    let output = Command::new("ssh")
        .arg(host)
        .arg(format!("cat '{}'", remote_path.replace('\'', "'\\''")))
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("SSH transfer failed: {stderr}").into());
    }

    let remote_data = output.stdout;
    let remote_size = remote_data.len();

    tokio::fs::write(dest, &remote_data).await?;

    if verbose {
        eprintln!("Remote size: {remote_size} bytes");
    }

    println!(
        "Synced {} ({} transferred)",
        dest.display(),
        format_bytes(remote_size as u64)
    );

    Ok(())
}

// ── Signature / Delta / Patch ─────────────────────────────────────────

async fn run_signature(
    file: &PathBuf,
    output: Option<PathBuf>,
    block_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_block_size(block_size)?;

    let output = output.unwrap_or_else(|| {
        let mut p = file.clone();
        p.set_extension("sig");
        p
    });

    let sync = AsyncCopiaSync::with_block_size(block_size);

    let file_handle = tokio::fs::File::open(file).await?;
    let reader = tokio::io::BufReader::new(file_handle);
    let signature = sync.signature(reader).await?;

    let serialized = bincode::serialize(&signature)?;
    tokio::fs::write(&output, serialized).await?;

    println!(
        "Generated signature: {} ({} blocks, {} bytes)",
        output.display(),
        signature.blocks.len(),
        signature.file_size
    );

    Ok(())
}

async fn run_delta(
    source: &PathBuf,
    signature: &PathBuf,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = output.unwrap_or_else(|| {
        let mut p = source.clone();
        p.set_extension("delta");
        p
    });

    let sig_data = tokio::fs::read(signature).await?;
    let sig: copia::Signature = bincode::deserialize(&sig_data)?;

    let sync = AsyncCopiaSync::with_block_size(sig.block_size);

    let file_handle = tokio::fs::File::open(source).await?;
    let reader = tokio::io::BufReader::new(file_handle);
    let delta = sync.delta(reader, &sig).await?;

    let serialized = bincode::serialize(&delta)?;
    tokio::fs::write(&output, serialized).await?;

    println!(
        "Generated delta: {} ({} ops, {:.1}% matched)",
        output.display(),
        delta.ops.len(),
        delta.compression_ratio(delta.source_size) * 100.0
    );

    Ok(())
}

async fn run_patch(
    basis: &PathBuf,
    delta: &PathBuf,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = output.unwrap_or_else(|| {
        let mut p = basis.clone();
        p.set_extension("patched");
        p
    });

    let delta_data = tokio::fs::read(delta).await?;
    let delta: copia::Delta = bincode::deserialize(&delta_data)?;

    let sync = AsyncCopiaSync::with_block_size(delta.block_size as usize);

    let basis_file = tokio::fs::File::open(basis).await?;
    let output_file = tokio::fs::File::create(&output).await?;

    sync.patch(basis_file, &delta, output_file).await?;

    println!(
        "Applied patch: {} ({} bytes)",
        output.display(),
        delta.source_size
    );

    Ok(())
}

fn validate_block_size(size: usize) -> Result<(), String> {
    if !size.is_power_of_two() {
        return Err(format!("Block size must be a power of 2, got {size}"));
    }
    if !(512..=65536).contains(&size) {
        return Err(format!("Block size must be 512-65536, got {size}"));
    }
    Ok(())
}
