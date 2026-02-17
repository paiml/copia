# ADR-002: Wire Protocol Design

## Status

Accepted

## Context

Copia needs a wire protocol for network-based file synchronization.
The protocol must efficiently frame signatures, deltas, and control
messages over TCP connections.

## Decision

Custom binary framing protocol with:
- 12-byte fixed header (magic, length, type, version, flags)
- Maximum 16 MB payload per frame
- Version negotiation for forward compatibility
- Ping/pong heartbeat for connection health

**Rationale**: Protocol buffers and similar frameworks add 10+ MB of
dependencies. A custom protocol for 7 message types is simpler,
faster, and fully auditable.

## Consequences

- Frame headers validate magic bytes and version on every decode
- Payload size is bounded to prevent memory exhaustion
- Protocol is extensible via version field and reserved flags
