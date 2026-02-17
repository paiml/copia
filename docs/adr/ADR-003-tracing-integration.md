# ADR-003: Structured Tracing with Renacer Compatibility

## Status

Accepted

## Context

Performance analysis and debugging require structured observability.
Renacer provides a unified trace model for the sovereign stack, but
has heavy dependencies (30+ crates). Copia needs lightweight tracing
that produces renacer-compatible output.

## Decision

Two-layer design:
- Library: `tracing` crate with `#[instrument]` behind feature flag
- CLI: Custom `tracing::Layer` producing renacer SpanRecord NDJSON

**Rationale**: The `tracing` crate facade is zero-cost when disabled
(empty macros). Feature-gating ensures no runtime overhead for library
users who don't need tracing. The custom Layer avoids a dependency on
renacer itself while producing wire-compatible output.

## Consequences

- Library builds without tracing have zero instrumentation overhead
- CLI builds include structured span output with `--trace-output`
- Hot-path functions (rolling checksum, hash, block matching) are
  deliberately NOT instrumented to avoid per-byte overhead
- Trace output is compatible with renacer's unified trace viewer
