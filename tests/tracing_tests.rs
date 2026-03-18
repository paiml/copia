//! Integration tests for structured tracing instrumentation.
//!
//! Verifies that sync operations emit expected spans with correct fields
//! and that NDJSON output deserializes to CopiaSpanRecord.
#![allow(clippy::unwrap_used, clippy::doc_markdown)]
#![cfg(feature = "cli")]

use std::io::{Cursor, Write};
use std::sync::{Arc, Mutex};

use copia::trace_output::CopiaTraceLayer;
use copia::{CopiaSync, Sync, SyncBuilder};
use tracing_subscriber::layer::SubscriberExt;

/// Shared buffer for capturing trace output in tests.
#[derive(Clone)]
struct TestBuf(Arc<Mutex<Vec<u8>>>);

impl TestBuf {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    fn to_string(&self) -> String {
        let guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        String::from_utf8_lossy(&guard).to_string()
    }

    fn records(&self) -> Vec<serde_json::Value> {
        self.to_string()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

impl Write for TestBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run a closure with a test tracing subscriber, return captured NDJSON records.
fn with_tracing<F: FnOnce()>(f: F) -> Vec<serde_json::Value> {
    let buf = TestBuf::new();
    let layer = CopiaTraceLayer::new(buf.clone());

    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, f);

    buf.records()
}

#[test]
fn signature_emits_span_with_fields() {
    let records = with_tracing(|| {
        let sync = CopiaSync::with_block_size(512);
        let data = vec![42u8; 2000];
        let _sig = sync.signature(Cursor::new(data.as_slice())).unwrap();
    });

    // Should have spans for both CopiaSync::signature and Signature::generate
    let sig_spans: Vec<_> = records
        .iter()
        .filter(|r| r["span_name"] == "signature" || r["span_name"] == "generate")
        .collect();
    assert!(
        !sig_spans.is_empty(),
        "Expected signature/generate span, got: {records:?}"
    );

    // Find the generate span (inner) and check fields
    let generate_span = records.iter().find(|r| r["span_name"] == "generate");

    if let Some(span) = generate_span {
        assert_eq!(span["attributes"]["block_size"], 512);
        assert_eq!(span["attributes"]["file_size"], 2000);
        assert_eq!(span["attributes"]["block_count"], 4);
    }
}

#[test]
fn delta_emits_span_with_fields() {
    let records = with_tracing(|| {
        let sync = CopiaSync::with_block_size(512);
        let basis = vec![42u8; 1024];
        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let _delta = sync.delta(Cursor::new(basis.as_slice()), &sig).unwrap();
    });

    let delta_span = records.iter().find(|r| r["span_name"] == "delta");

    assert!(
        delta_span.is_some(),
        "Expected delta span, got: {records:?}"
    );
    let span = delta_span.unwrap();

    assert_eq!(span["attributes"]["block_size"], 512);
    assert_eq!(span["attributes"]["source_size"], 1024);
    assert_eq!(span["attributes"]["bytes_matched"], 1024);
    assert_eq!(span["attributes"]["bytes_literal"], 0);
}

#[test]
fn patch_emits_span_with_fields() {
    let records = with_tracing(|| {
        let sync = SyncBuilder::new()
            .block_size(512)
            .verify_checksum(true)
            .build();
        let basis = vec![42u8; 1024];
        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(basis.as_slice()), &sig).unwrap();
        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();
    });

    let patch_span = records.iter().find(|r| r["span_name"] == "patch");

    assert!(
        patch_span.is_some(),
        "Expected patch span, got: {records:?}"
    );
    let span = patch_span.unwrap();

    assert_eq!(span["attributes"]["verify_checksum"], true);
    assert!(span["attributes"]["op_count"].as_u64().unwrap_or(0) > 0);
}

#[test]
fn full_roundtrip_emits_all_spans() {
    let records = with_tracing(|| {
        let sync = CopiaSync::with_block_size(512);
        let basis = b"Hello, World! This is a test file for rsync.".to_vec();
        let source = b"Hello, Universe! This is a test file for rsync.".to_vec();

        let sig = sync.signature(Cursor::new(basis.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(source.as_slice()), &sig).unwrap();
        let mut output = Vec::new();
        sync.patch(Cursor::new(basis.as_slice()), &delta, &mut output)
            .unwrap();
    });

    let span_names: Vec<&str> = records
        .iter()
        .filter_map(|r| r["span_name"].as_str())
        .collect();

    // Must have at least signature, delta, patch
    assert!(
        span_names.contains(&"signature") || span_names.contains(&"generate"),
        "Missing signature/generate span in {span_names:?}"
    );
    assert!(
        span_names.contains(&"delta"),
        "Missing delta span in {span_names:?}"
    );
    assert!(
        span_names.contains(&"patch"),
        "Missing patch span in {span_names:?}"
    );
}

#[test]
fn ndjson_output_deserializes_to_valid_records() {
    let buf = TestBuf::new();
    let layer = CopiaTraceLayer::new(buf.clone());

    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::with_default(subscriber, || {
        let sync = CopiaSync::with_block_size(512);
        let data = vec![0u8; 1024];
        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();
        let _delta = sync.delta(Cursor::new(data.as_slice()), &sig).unwrap();
    });

    let output = buf.to_string();
    for line in output.trim().lines() {
        // Verify each line is valid JSON with expected fields
        let record: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Invalid JSON: {e}\nLine: {line}"));

        // Validate required CopiaSpanRecord fields
        assert!(record["trace_id"].is_string(), "Missing trace_id");
        assert!(record["span_id"].is_number(), "Missing span_id");
        assert!(
            record["parent_span_id"].is_number(),
            "Missing parent_span_id"
        );
        assert!(record["span_name"].is_string(), "Missing span_name");
        assert!(record["start_nanos"].is_number(), "Missing start_nanos");
        assert!(record["end_nanos"].is_number(), "Missing end_nanos");
        assert!(
            record["duration_nanos"].is_number(),
            "Missing duration_nanos"
        );
        assert!(record["logical_clock"].is_number(), "Missing logical_clock");
        assert!(record["status_code"].is_string(), "Missing status_code");
        assert!(record["attributes"].is_object(), "Missing attributes");
        assert!(record["process_id"].is_number(), "Missing process_id");
        assert!(record["thread_id"].is_number(), "Missing thread_id");

        // Validate trace_id format
        assert!(
            record["trace_id"]
                .as_str()
                .unwrap_or("")
                .starts_with("copia-"),
            "trace_id should start with copia-"
        );
    }
}

#[test]
fn empty_file_signature_records_zero_fields() {
    let records = with_tracing(|| {
        let sync = CopiaSync::new();
        let _sig = sync.signature(Cursor::new(&[] as &[u8])).unwrap();
    });

    let generate_span = records.iter().find(|r| r["span_name"] == "generate");

    if let Some(span) = generate_span {
        assert_eq!(span["attributes"]["file_size"], 0);
        assert_eq!(span["attributes"]["block_count"], 0);
        assert_eq!(span["attributes"]["parallel"], false);
    }
}

#[test]
fn parallel_signature_records_parallel_field() {
    let records = with_tracing(|| {
        let sync = CopiaSync::with_block_size(512);
        // >64KB triggers parallel mode
        let data = vec![1u8; 100_000];
        let _sig = sync.signature(Cursor::new(data.as_slice())).unwrap();
    });

    let generate_span = records.iter().find(|r| r["span_name"] == "generate");

    if let Some(span) = generate_span {
        assert_eq!(span["attributes"]["parallel"], true);
        assert_eq!(span["attributes"]["file_size"], 100_000);
    }
}

#[test]
fn logical_clock_increments_across_operations() {
    let records = with_tracing(|| {
        let sync = CopiaSync::with_block_size(512);
        let data = vec![42u8; 1024];

        let sig = sync.signature(Cursor::new(data.as_slice())).unwrap();
        let delta = sync.delta(Cursor::new(data.as_slice()), &sig).unwrap();
        let mut output = Vec::new();
        sync.patch(Cursor::new(data.as_slice()), &delta, &mut output)
            .unwrap();
    });

    let clocks: Vec<u64> = records
        .iter()
        .filter_map(|r| r["logical_clock"].as_u64())
        .collect();

    // Verify monotonic increase
    for window in clocks.windows(2) {
        assert!(
            window[1] > window[0],
            "Logical clock should be strictly increasing: {clocks:?}"
        );
    }
}
