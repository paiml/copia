//! Copia CLI - rsync-style file synchronization.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tracing::instrument;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use copia::async_sync::AsyncCopiaSync;
use copia::trace_output::CopiaTraceLayer;

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
    /// Write renacer-compatible NDJSON trace output to file
    #[arg(long, global = true)]
    trace_output: Option<PathBuf>,

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

        /// Show what would change without transferring or deleting anything
        #[arg(short = 'n', long)]
        dry_run: bool,

        /// Mirror: delete destination files that no longer exist in the source
        #[arg(long)]
        delete: bool,

        /// Skip paths matching GLOB (repeatable). Slash-free patterns match any
        /// path component (e.g. `target`, `*.tmp`); patterns with `/` match the
        /// whole relative path.
        #[arg(long = "exclude", value_name = "GLOB")]
        excludes: Vec<String>,
    },

    /// Bidirectional sync of two local directories (L2: blake3 3-way reconcile,
    /// safe deletes, convergent conflict-copy — never loses data).
    Bisync {
        /// First directory
        #[arg(required = true)]
        a: PathBuf,

        /// Second directory
        #[arg(required = true)]
        b: PathBuf,

        /// Show the reconcile plan without modifying either side
        #[arg(short = 'n', long)]
        dry_run: bool,

        /// Show verbose output (per-conflict listing)
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

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_invalid_env| EnvFilter::new("copia=info"));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_timer(tracing_subscriber::fmt::time::uptime());

    if let Some(ref path) = cli.trace_output {
        match std::fs::File::create(path) {
            Ok(file) => {
                tracing_subscriber::registry()
                    .with(fmt_layer)
                    .with(env_filter)
                    .with(CopiaTraceLayer::new(file))
                    .init();
            }
            Err(e) => {
                eprintln!("Error: cannot create trace output file: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(env_filter)
            .init();
    }

    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[instrument(skip(cli))]
async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Sync {
            source,
            dest,
            block_size,
            recursive,
            jobs,
            verbose,
            dry_run,
            delete,
            excludes,
        } => {
            let src_loc = FileLocation::parse(&source);
            let dest_loc = FileLocation::parse(&dest);
            if recursive {
                let opts = SyncOptions {
                    jobs,
                    verbose,
                    dry_run,
                    delete,
                    excludes,
                };
                run_sync_recursive(src_loc, dest_loc, opts).await
            } else {
                run_sync(src_loc, dest_loc, block_size, verbose).await
            }
        }
        Commands::Bisync {
            a,
            b,
            dry_run,
            verbose,
        } => bidir::run_bisync(&a, &b, &bidir::BidirOptions { dry_run, verbose }),
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

mod archive;
mod bidir;
mod dir_sync;
mod incremental;
mod meta;
mod plan;
mod reconcile;
mod single_sync;
mod transfer;
use incremental::{run_sync_recursive, SyncOptions};
use single_sync::run_sync;

// ── Signature / Delta / Patch ─────────────────────────────────────────

#[instrument(skip(file, output))]
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

#[instrument(skip(source, signature, output))]
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

#[instrument(skip(basis, delta, output))]
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

pub(crate) fn validate_block_size(size: usize) -> Result<(), String> {
    if !size.is_power_of_two() {
        return Err(format!("Block size must be a power of 2, got {size}"));
    }
    if !(512..=65536).contains(&size) {
        return Err(format!("Block size must be 512-65536, got {size}"));
    }
    Ok(())
}
