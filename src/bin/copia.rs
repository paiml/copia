//! Copia CLI - rsync-style file synchronization.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use copia::async_sync::AsyncCopiaSync;

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
    /// Synchronize source file to destination
    Sync {
        /// Source file path
        #[arg(required = true)]
        source: PathBuf,

        /// Destination file path
        #[arg(required = true)]
        dest: PathBuf,

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
        } => run_sync(&source, &dest, block_size, verbose).await,
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
    source: &PathBuf,
    dest: &PathBuf,
    block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_block_size(block_size)?;

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
