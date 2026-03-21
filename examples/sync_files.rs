//! Demonstrates the copia signature/delta/patch synchronization flow.
//!
//! This example creates two in-memory "files" (basis and source),
//! computes a delta, and patches the basis to reconstruct the source.
//! No filesystem access is required.
//!
//! Run with: `cargo run --example sync_files`

use copia::Sync;
use std::io::Cursor;

fn main() {
    println!("=== Copia Delta-Sync Demo ===\n");

    // The basis (old) file content.
    let basis = b"The quick brown fox jumps over the lazy dog. \
                  This is the original content that both sides share. \
                  Only a small portion of this file will change.";

    // The source (new) file content — a few edits.
    let source = b"The quick brown fox leaps over the lazy cat. \
                   This is the original content that both sides share. \
                   A new paragraph has been appended at the end!";

    println!("Basis  size: {} bytes", basis.len());
    println!("Source size: {} bytes", source.len());

    // Build a sync engine with a 512-byte block size (minimum allowed).
    let sync = copia::SyncBuilder::new().block_size(512).build();

    // Step 1: Generate a signature from the basis file.
    let signature = sync
        .signature(Cursor::new(basis.as_slice()))
        .expect("signature generation failed");
    println!(
        "\nSignature: {} blocks (block_size={})",
        signature.blocks.len(),
        signature.block_size
    );

    // Step 2: Compute the delta between the source and the signature.
    let delta = sync
        .delta(Cursor::new(source.as_slice()), &signature)
        .expect("delta computation failed");
    println!(
        "Delta: {} operations, source_size={}",
        delta.ops.len(),
        delta.source_size
    );

    // Print delta stats.
    println!(
        "  Matched: {} bytes  |  Literal: {} bytes",
        delta.bytes_matched(),
        delta.bytes_literal()
    );

    // Step 3: Apply the delta to the basis to reconstruct the source.
    let mut output = Vec::new();
    sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
        .expect("patch application failed");

    assert_eq!(output, source, "Reconstructed output must equal source");
    println!("\nPatch applied successfully — output matches source.");
    println!("Reconstructed: {} bytes", output.len());

    // Show the reconstructed text.
    println!(
        "\nReconstructed content:\n  \"{}\"",
        String::from_utf8_lossy(&output)
    );

    println!("\nDone.");
}
