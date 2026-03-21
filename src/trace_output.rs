//! Renacer-compatible structured trace output.
//!
//! Produces NDJSON `CopiaSpanRecord` entries matching renacer's `SpanRecord`
//! schema for unified trace analysis across the sovereign stack.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use tracing::span;
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Renacer-compatible span record for NDJSON output.
///
/// Matches renacer's `SpanRecord` JSON schema for cross-tool trace fusion.
#[derive(Debug, Clone, Serialize)]
pub struct CopiaSpanRecord {
    /// Trace identifier (process-scoped).
    pub trace_id: String,
    /// Unique span identifier.
    pub span_id: u64,
    /// Parent span identifier (0 if root).
    pub parent_span_id: u64,
    /// Span name (e.g. "signature", "delta", "patch").
    pub span_name: String,
    /// Start time in nanoseconds since process start.
    pub start_nanos: u64,
    /// End time in nanoseconds since process start.
    pub end_nanos: u64,
    /// Duration in nanoseconds.
    pub duration_nanos: u64,
    /// Monotonic logical clock (Lamport-compatible).
    pub logical_clock: u64,
    /// Status: "ok" or "error".
    pub status_code: String,
    /// Span attributes as key-value pairs.
    pub attributes: HashMap<String, serde_json::Value>,
    /// Process ID.
    pub process_id: u32,
    /// Thread ID.
    pub thread_id: u64,
}

/// Per-span data stored in the tracing registry extensions.
struct SpanData {
    start: Instant,
    start_nanos: u64,
    attributes: HashMap<String, serde_json::Value>,
}

/// Custom tracing layer that emits renacer-compatible NDJSON.
pub struct CopiaTraceLayer<W: Write + Send + 'static> {
    writer: Arc<Mutex<W>>,
    logical_clock: Arc<AtomicU64>,
    epoch: Instant,
    trace_id: String,
}

impl<W: Write + Send + 'static> CopiaTraceLayer<W> {
    /// Create a new trace layer writing to the given output.
    #[provable_contracts_macros::contract("copia-trace-v1", equation = "new")]
    pub fn new(writer: W) -> Self {
        let pid = std::process::id();
        Self {
            writer: Arc::new(Mutex::new(writer)),
            logical_clock: Arc::new(AtomicU64::new(0)),
            epoch: Instant::now(),
            trace_id: format!("copia-{pid}-{}", Self::timestamp_id()),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn timestamp_id() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    }
}

impl<S, W> Layer<S> for CopiaTraceLayer<W>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    W: Write + Send + 'static,
{
    #[allow(clippy::cast_possible_truncation)]
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let now = Instant::now();
        let start_nanos = now.duration_since(self.epoch).as_nanos() as u64;

        let mut attributes = HashMap::new();
        let mut visitor = JsonFieldVisitor(&mut attributes);
        attrs.record(&mut visitor);

        if let Some(span) = ctx.span(id) {
            let mut extensions = span.extensions_mut();
            extensions.insert(SpanData {
                start: now,
                start_nanos,
                attributes,
            });
        }
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut extensions = span.extensions_mut();
            if let Some(data) = extensions.get_mut::<SpanData>() {
                let mut visitor = JsonFieldVisitor(&mut data.attributes);
                values.record(&mut visitor);
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::significant_drop_tightening)]
    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let end = Instant::now();

        let Some(span) = ctx.span(&id) else {
            return;
        };

        let extensions = span.extensions();
        let Some(data) = extensions.get::<SpanData>() else {
            return;
        };

        let duration_nanos = end.duration_since(data.start).as_nanos() as u64;
        let end_nanos = data.start_nanos + duration_nanos;
        let attributes = data.attributes.clone();
        let start_nanos = data.start_nanos;
        drop(extensions);

        let logical_clock = self.logical_clock.fetch_add(1, Ordering::Relaxed);

        let parent_span_id = span.parent().map_or(0, |p| p.id().into_u64());

        let record = CopiaSpanRecord {
            trace_id: self.trace_id.clone(),
            span_id: id.into_u64(),
            parent_span_id,
            span_name: span.name().to_string(),
            start_nanos,
            end_nanos,
            duration_nanos,
            logical_clock,
            status_code: "ok".to_string(),
            attributes,
            process_id: std::process::id(),
            thread_id: thread_id(),
        };

        match serde_json::to_string(&record) {
            Ok(json) => {
                if let Ok(mut w) = self.writer.lock() {
                    // GH-23: Log trace write failures instead of silently discarding
                    if writeln!(w, "{json}").is_err() {
                        eprintln!("Warning: failed to write trace record");
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: failed to serialize trace record: {e}");
            }
        }
    }
}

/// Visitor that records span fields as JSON values.
struct JsonFieldVisitor<'a>(&'a mut HashMap<String, serde_json::Value>);

impl tracing::field::Visit for JsonFieldVisitor<'_> {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(
            field.name().to_string(),
            serde_json::json!(format!("{value:?}")),
        );
    }
}

/// Get a stable thread identifier.
fn thread_id() -> u64 {
    // Use a simple hash of the thread name/id for a stable u64
    let id = std::thread::current().id();
    let debug = format!("{id:?}");
    // Extract numeric portion from "ThreadId(N)"
    debug
        .trim_start_matches("ThreadId(")
        .trim_end_matches(')')
        .parse()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::layer::SubscriberExt;

    /// A Write impl backed by a shared buffer for testing.
    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<Vec<u8>>>);

    impl SharedBuf {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn contents(&self) -> Vec<u8> {
            self.0.lock().map_or_else(|_| Vec::new(), |g| g.clone())
        }
    }

    impl Write for SharedBuf {
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

    #[test]
    fn span_record_serialization() {
        let record = CopiaSpanRecord {
            trace_id: "test-trace-1".to_string(),
            span_id: 1,
            parent_span_id: 0,
            span_name: "signature".to_string(),
            start_nanos: 1000,
            end_nanos: 2000,
            duration_nanos: 1000,
            logical_clock: 0,
            status_code: "ok".to_string(),
            attributes: {
                let mut m = HashMap::new();
                m.insert("block_size".to_string(), serde_json::json!(2048));
                m.insert("file_size".to_string(), serde_json::json!(10000));
                m
            },
            process_id: 42,
            thread_id: 1,
        };

        let json = serde_json::to_string(&record).unwrap_or_default();
        assert!(json.contains("\"span_name\":\"signature\""));
        assert!(json.contains("\"trace_id\":\"test-trace-1\""));
        assert!(json.contains("\"block_size\":2048"));
        assert!(json.contains("\"duration_nanos\":1000"));

        // Verify it deserializes back
        let parsed: serde_json::Value =
            serde_json::from_str(&json).unwrap_or(serde_json::Value::Null);
        assert_eq!(parsed["span_id"], 1);
        assert_eq!(parsed["parent_span_id"], 0);
        assert_eq!(parsed["status_code"], "ok");
    }

    #[test]
    fn layer_emits_ndjson_on_span_close() {
        let buf = SharedBuf::new();
        let layer = CopiaTraceLayer::new(buf.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("test_op", block_size = 1024);
            let _guard = span.enter();
            // span closes when guard drops
        });

        let output = String::from_utf8(buf.contents()).unwrap_or_default();
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 1, "Expected exactly one NDJSON line");

        let parsed: serde_json::Value =
            serde_json::from_str(lines[0]).unwrap_or(serde_json::Value::Null);
        assert_eq!(parsed["span_name"], "test_op");
        assert_eq!(parsed["attributes"]["block_size"], 1024);
        assert_eq!(parsed["status_code"], "ok");
        assert!(parsed["duration_nanos"].as_u64().unwrap_or(0) > 0);
        assert!(parsed["process_id"].as_u64().is_some());
    }

    #[test]
    fn layer_records_parent_span_id() {
        let buf = SharedBuf::new();
        let layer = CopiaTraceLayer::new(buf.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let parent = tracing::info_span!("parent_op");
            let _parent_guard = parent.enter();
            {
                let child = tracing::info_span!("child_op");
                let _child_guard = child.enter();
            }
        });

        let output = String::from_utf8(buf.contents()).unwrap_or_default();
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2, "Expected two NDJSON lines");

        // Child closes first
        let child_record: serde_json::Value =
            serde_json::from_str(lines[0]).unwrap_or(serde_json::Value::Null);
        assert_eq!(child_record["span_name"], "child_op");
        assert!(child_record["parent_span_id"].as_u64().unwrap_or(0) > 0);

        // Parent closes second
        let parent_record: serde_json::Value =
            serde_json::from_str(lines[1]).unwrap_or(serde_json::Value::Null);
        assert_eq!(parent_record["span_name"], "parent_op");
        assert_eq!(parent_record["parent_span_id"], 0);
    }

    #[test]
    fn layer_records_dynamic_fields() {
        let buf = SharedBuf::new();
        let layer = CopiaTraceLayer::new(buf.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "delta",
                block_size = 2048,
                file_size = tracing::field::Empty,
                bytes_matched = tracing::field::Empty,
            );
            let _guard = span.enter();
            tracing::Span::current().record("file_size", 50000_u64);
            tracing::Span::current().record("bytes_matched", 48000_u64);
        });

        let output = String::from_utf8(buf.contents()).unwrap_or_default();
        let line = output.trim();
        let parsed: serde_json::Value =
            serde_json::from_str(line).unwrap_or(serde_json::Value::Null);

        assert_eq!(parsed["span_name"], "delta");
        assert_eq!(parsed["attributes"]["block_size"], 2048);
        assert_eq!(parsed["attributes"]["file_size"], 50000);
        assert_eq!(parsed["attributes"]["bytes_matched"], 48000);
    }

    #[test]
    fn logical_clock_increments() {
        let buf = SharedBuf::new();
        let layer = CopiaTraceLayer::new(buf.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..3 {
                let span = tracing::info_span!("tick");
                let _guard = span.enter();
            }
        });

        let output = String::from_utf8(buf.contents()).unwrap_or_default();
        let clocks: Vec<u64> = output
            .trim()
            .lines()
            .filter_map(|line| {
                let v: serde_json::Value = serde_json::from_str(line).ok()?;
                v["logical_clock"].as_u64()
            })
            .collect();

        assert_eq!(clocks, vec![0, 1, 2]);
    }

    #[test]
    fn thread_id_is_nonzero() {
        let tid = thread_id();
        assert!(tid > 0);
    }

    #[test]
    fn json_field_visitor_records_all_types() {
        // Test all field types through actual span recording
        let buf = SharedBuf::new();
        let layer = CopiaTraceLayer::new(buf.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(
                "typed_fields",
                int_val = 42_i64,
                uint_val = 100_u64,
                bool_val = true,
                str_val = "hello",
            );
            let _guard = span.enter();
        });

        let output = String::from_utf8(buf.contents()).unwrap_or_default();
        let parsed: serde_json::Value =
            serde_json::from_str(output.trim()).unwrap_or(serde_json::Value::Null);

        assert_eq!(parsed["attributes"]["int_val"], 42);
        assert_eq!(parsed["attributes"]["uint_val"], 100);
        assert_eq!(parsed["attributes"]["bool_val"], true);
        assert_eq!(parsed["attributes"]["str_val"], "hello");
    }

    #[test]
    fn empty_span_produces_valid_record() {
        let buf = SharedBuf::new();
        let layer = CopiaTraceLayer::new(buf.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("empty_span");
            let _guard = span.enter();
        });

        let output = String::from_utf8(buf.contents()).unwrap_or_default();
        let parsed: serde_json::Value =
            serde_json::from_str(output.trim()).unwrap_or(serde_json::Value::Null);

        assert_eq!(parsed["span_name"], "empty_span");
        assert!(parsed["attributes"]
            .as_object()
            .map_or(false, |m| m.is_empty()));
        assert!(parsed["trace_id"]
            .as_str()
            .map_or(false, |s| s.starts_with("copia-")));
    }
}
