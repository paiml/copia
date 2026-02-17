FROM rust:1.75 AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
COPY tests/ tests/
RUN cargo build --release --features cli

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/copia /usr/local/bin/
ENTRYPOINT ["copia"]
