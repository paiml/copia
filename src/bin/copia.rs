//! Copia CLI - rsync-style file synchronization.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use copia::async_sync::AsyncCopiaSync;

/// Represents a file location - either local or remote via SSH
#[derive(Debug, Clone)]
enum FileLocation {
    Local(PathBuf),
    Remote { host: String, path: String },
}

impl FileLocation {
    /// Parse a path string into a FileLocation
    /// Supports formats: `/local/path`, `host:path`, `host:~/path`
    fn parse(s: &str) -> Self {
        // Check for host:path format (but not Windows drive letters like C:)
        if let Some(colon_pos) = s.find(':') {
            let before_colon = &s[..colon_pos];
            // If it's a single letter, it might be a Windows drive
            if before_colon.len() > 1 && !before_colon.contains('/') && !before_colon.contains('\\')
            {
                return FileLocation::Remote {
                    host: before_colon.to_string(),
                    path: s[colon_pos + 1..].to_string(),
                };
            }
        }
        FileLocation::Local(PathBuf::from(s))
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
    /// Synchronize source file to destination (supports host:path for SSH)
    Sync {
        /// Source file path (local path or host:path for SSH)
        #[arg(required = true)]
        source: String,

        /// Destination file path (local path or host:path for SSH)
        #[arg(required = true)]
        dest: String,

        /// Block size for signature generation (512-65536, power of 2)
        #[arg(short, long, default_value = "2048")]
        block_size: usize,

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
            verbose,
        } => {
            let src_loc = FileLocation::parse(&source);
            let dest_loc = FileLocation::parse(&dest);
            run_sync(src_loc, dest_loc, block_size, verbose).await
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

async fn run_sync(
    source: FileLocation,
    dest: FileLocation,
    block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_block_size(block_size)?;

    match (&source, &dest) {
        (FileLocation::Local(src), FileLocation::Local(dst)) => {
            run_sync_local_to_local(src, dst, block_size, verbose).await
        }
        (FileLocation::Local(src), FileLocation::Remote { host, path }) => {
            run_sync_local_to_remote(src, host, path, block_size, verbose).await
        }
        (FileLocation::Remote { host, path }, FileLocation::Local(dst)) => {
            run_sync_remote_to_local(host, path, dst, block_size, verbose).await
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
            // Remote-to-remote: pull to temp, then push
            Err(format!(
                "Remote-to-remote sync not yet supported: {}:{} -> {}:{}",
                src_host, src_path, dst_host, dst_path
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
    source: &PathBuf,
    host: &str,
    remote_path: &str,
    _block_size: usize, // TODO: Use for delta transfer optimization
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::process::Command;

    if verbose {
        eprintln!(
            "Syncing {} -> {}:{}",
            source.display(),
            host,
            remote_path
        );
    }

    // Read source file
    let source_data = tokio::fs::read(source).await?;
    let source_size = source_data.len();

    // For now, use a simple scp-style approach: transfer whole file via SSH
    // Future optimization: implement delta transfer over SSH
    let mut child = Command::new("ssh")
        .arg(host)
        .arg(format!("cat > '{}'", remote_path.replace('\'', "'\\''")))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    tokio::io::AsyncWriteExt::write_all(&mut stdin, &source_data).await?;
    drop(stdin);

    let result = child.wait_with_output().await?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!("SSH transfer failed: {}", stderr).into());
    }

    if verbose {
        eprintln!("Source size: {} bytes", source_size);
        eprintln!("Transferred: {} bytes (full file)", source_size);
    }

    println!(
        "Synced {}:{} ({} bytes transferred)",
        host, remote_path, source_size
    );

    Ok(())
}

async fn run_sync_remote_to_local(
    host: &str,
    remote_path: &str,
    dest: &PathBuf,
    _block_size: usize, // TODO: Use for delta transfer optimization
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::process::Command;

    if verbose {
        eprintln!(
            "Syncing {}:{} -> {}",
            host,
            remote_path,
            dest.display()
        );
    }

    // Fetch remote file via SSH
    let output = Command::new("ssh")
        .arg(host)
        .arg(format!("cat '{}'", remote_path.replace('\'', "'\\''")))
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("SSH transfer failed: {}", stderr).into());
    }

    let remote_data = output.stdout;
    let remote_size = remote_data.len();

    // Write to destination
    tokio::fs::write(dest, &remote_data).await?;

    if verbose {
        eprintln!("Remote size: {} bytes", remote_size);
        eprintln!("Transferred: {} bytes (full file)", remote_size);
    }

    println!(
        "Synced {} ({} bytes transferred)",
        dest.display(),
        remote_size
    );

    Ok(())
}

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

    // Load signature
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

    // Load delta
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
