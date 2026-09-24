//! The shipped `x86_64` binary must run on every fleet CPU (paiml/copia#65).
//!
//! `target-cpu=native` on an AVX-512 build host produced a v0.3.0 release that
//! died with SIGILL on an AVX2-only Intel Core Ultra 9 185H. This asserts on
//! what the compiler enabled, not on the config text, so any flag, env var or
//! profile that leaks AVX-512 into the build turns it red.

#[test]
#[cfg(target_arch = "x86_64")]
fn build_does_not_require_avx512() {
    assert!(
        !std::hint::black_box(cfg!(target_feature = "avx512f")),
        "built with AVX-512 enabled: the artifact will SIGILL on AVX2-only CPUs (copia#65)"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn build_keeps_the_avx2_baseline() {
    assert!(
        std::hint::black_box(cfg!(target_feature = "avx2")),
        "built below x86-64-v3: the portable baseline is AVX2 (copia#65)"
    );
}
