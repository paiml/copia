//! Benchmarks for copia operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::io::Cursor;

use copia::{CopiaSync, RollingChecksum, StrongHash, Sync, SyncBuilder};

fn bench_rolling_checksum(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_checksum");

    for size in [64, 512, 2048, 8192].iter() {
        let data = vec![42u8; *size];

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("new", size), &data, |b, data| {
            b.iter(|| RollingChecksum::new(black_box(data)));
        });
    }

    group.finish();
}

fn bench_rolling_checksum_roll(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_checksum_roll");

    let data = vec![42u8; 2048];
    let mut checksum = RollingChecksum::new(&data);

    group.bench_function("roll", |b| {
        b.iter(|| {
            checksum.roll(black_box(42), black_box(43));
        });
    });

    group.finish();
}

fn bench_strong_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("strong_hash");

    for size in [64, 512, 2048, 8192, 65536].iter() {
        let data = vec![42u8; *size];

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("compute", size), &data, |b, data| {
            b.iter(|| StrongHash::compute(black_box(data)));
        });
    }

    group.finish();
}

fn bench_signature(c: &mut Criterion) {
    let mut group = c.benchmark_group("signature");
    let sync = CopiaSync::with_block_size(2048);

    for size in [1024, 10240, 102400, 1024000].iter() {
        let data = vec![42u8; *size];

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("generate", size), &data, |b, data| {
            b.iter(|| sync.signature(Cursor::new(black_box(data))));
        });
    }

    group.finish();
}

fn bench_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta");
    let sync = CopiaSync::with_block_size(2048);

    for size in [1024, 10240, 102400].iter() {
        let basis = vec![42u8; *size];
        let source = vec![42u8; *size]; // Identical for best case

        let sig = sync.signature(Cursor::new(&basis)).unwrap();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("identical", size), &source, |b, source| {
            b.iter(|| sync.delta(Cursor::new(black_box(source)), &sig));
        });
    }

    // Also benchmark worst case (completely different)
    for size in [1024, 10240].iter() {
        let basis = vec![0u8; *size];
        let source = vec![1u8; *size];

        let sig = sync.signature(Cursor::new(&basis)).unwrap();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("different", size), &source, |b, source| {
            b.iter(|| sync.delta(Cursor::new(black_box(source)), &sig));
        });
    }

    group.finish();
}

fn bench_patch(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch");
    let sync = SyncBuilder::new()
        .block_size(2048)
        .verify_checksum(false) // Skip verification for raw speed
        .build();

    for size in [1024, 10240, 102400].iter() {
        let basis = vec![42u8; *size];
        let source = vec![42u8; *size];

        let sig = sync.signature(Cursor::new(&basis)).unwrap();
        let delta = sync.delta(Cursor::new(&source), &sig).unwrap();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("apply", size), &delta, |b, delta| {
            let mut output = Vec::with_capacity(*size);
            b.iter(|| {
                output.clear();
                sync.patch(Cursor::new(black_box(&basis)), delta, &mut output)
            });
        });
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    let sync = SyncBuilder::new()
        .block_size(2048)
        .verify_checksum(true)
        .build();

    for size in [1024, 10240, 102400].iter() {
        let basis = vec![42u8; *size];

        // Source with 10% modification
        let mut source = basis.clone();
        for i in (0..source.len()).step_by(10) {
            source[i] = 0xFF;
        }

        group.throughput(Throughput::Bytes(*size as u64 * 2)); // basis + source
        group.bench_with_input(
            BenchmarkId::new("10pct_change", size),
            &(&basis, &source),
            |b, (basis, source)| {
                b.iter(|| {
                    let sig = sync.signature(Cursor::new(black_box(*basis))).unwrap();
                    let delta = sync.delta(Cursor::new(black_box(*source)), &sig).unwrap();
                    let mut output = Vec::with_capacity(source.len());
                    sync.patch(Cursor::new(black_box(*basis)), &delta, &mut output)
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rolling_checksum,
    bench_rolling_checksum_roll,
    bench_strong_hash,
    bench_signature,
    bench_delta,
    bench_patch,
    bench_roundtrip,
);

criterion_main!(benches);
