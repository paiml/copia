//! Single-file sync (non-recursive) across the three directions. The recursive
//! path lives in `incremental.rs`; this handles one file at a time.

use super::transfer::{format_bytes, transfer_file_to_remote};
use super::{validate_block_size, FileLocation};
use copia::async_sync::AsyncCopiaSync;
use std::path::{Path, PathBuf};
use tracing::instrument;

#[instrument(skip(source, dest))]
pub async fn run_sync(
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
        ) => Err(format!(
            "Remote-to-remote sync not yet supported: {src_host}:{src_path} -> {dst_host}:{dst_path}"
        )
        .into()),
    }
}

#[instrument(skip(source, dest))]
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

#[instrument(skip(source, block_size), fields(host, remote_path))]
async fn run_sync_local_to_remote(
    source: &Path,
    host: &str,
    remote_path: &str,
    block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if block_size != 4096 {
        eprintln!("Warning: --block-size is not yet implemented for remote transfers. Using SSH streaming.");
    }
    if verbose {
        eprintln!("Syncing {} -> {}:{}", source.display(), host, remote_path);
    }

    let size = transfer_file_to_remote(source, host, remote_path, None)
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

#[instrument(skip(dest, block_size), fields(host, remote_path))]
async fn run_sync_remote_to_local(
    host: &str,
    remote_path: &str,
    dest: &PathBuf,
    block_size: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if block_size != 4096 {
        eprintln!("Warning: --block-size is not yet implemented for remote transfers. Using SSH streaming.");
    }
    use tokio::process::Command;

    if verbose {
        eprintln!("Syncing {}:{} -> {}", host, remote_path, dest.display());
    }

    let output = Command::new("ssh")
        .arg(host)
        .arg(format!(
            "cat $'{}'",
            remote_path.replace('\\', "\\\\").replace('\'', "\\'")
        ))
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
