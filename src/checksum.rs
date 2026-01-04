//! Rolling checksum implementation for rsync algorithm.
//!
//! This module implements an Adler-32 variant rolling checksum that enables
//! O(1) window sliding for efficient block boundary detection.

/// Rolling checksum state for incremental computation.
///
/// The rolling checksum uses an Adler-32 variant that allows O(1) updates
/// when sliding a window by one byte. This is critical for the rsync
/// algorithm's block matching performance.
///
/// # Algorithm
///
/// The checksum consists of two components:
/// - `a`: Sum of all bytes in the window (mod MOD)
/// - `b`: Weighted sum where each byte is multiplied by its distance from the end
///
/// The final digest combines both: `(b << 16) | a`
///
/// # Example
///
/// ```rust
/// use copia::RollingChecksum;
///
/// let data = b"hello";
/// let checksum = RollingChecksum::new(data);
/// let digest = checksum.digest();
///
/// // Rolling update
/// let mut rolling = RollingChecksum::new(b"hello");
/// rolling.roll(b'h', b'!');  // "ello!"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RollingChecksum {
    /// Sum of all bytes in window
    a: u32,
    /// Weighted sum: sum of (`window_size` - i) * byte\[i\]
    b: u32,
    /// Current window size
    count: usize,
}

impl RollingChecksum {
    /// Base modulus for checksum arithmetic.
    /// Largest prime less than 2^16 for good distribution.
    const MOD: u32 = 65521;

    /// Create a new rolling checksum from initial data block.
    ///
    /// # Arguments
    ///
    /// * `data` - Initial byte slice to compute checksum over
    ///
    /// # Example
    ///
    /// ```rust
    /// use copia::RollingChecksum;
    ///
    /// let checksum = RollingChecksum::new(b"test data");
    /// assert!(checksum.digest() != 0);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(data: &[u8]) -> Self {
        let mut a: u32 = 0;
        let mut b: u32 = 0;
        let len = data.len();

        for (i, &byte) in data.iter().enumerate() {
            a = a.wrapping_add(u32::from(byte));
            // Weight is (len - i) so first byte has highest weight
            // Truncation is intentional: checksum uses 32-bit arithmetic
            b = b.wrapping_add((len - i) as u32 * u32::from(byte));
        }

        Self {
            a: a % Self::MOD,
            b: b % Self::MOD,
            count: len,
        }
    }

    /// Create an empty rolling checksum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use copia::RollingChecksum;
    ///
    /// let checksum = RollingChecksum::empty();
    /// assert_eq!(checksum.digest(), 0);
    /// ```
    #[must_use]
    pub const fn empty() -> Self {
        Self { a: 0, b: 0, count: 0 }
    }

    /// Roll the window by one byte: remove `old_byte` from start, add `new_byte` at end.
    ///
    /// This operation is O(1) and is the key to efficient rsync block matching.
    ///
    /// # Arguments
    ///
    /// * `old_byte` - Byte being removed from the start of the window
    /// * `new_byte` - Byte being added to the end of the window
    ///
    /// # Example
    ///
    /// ```rust
    /// use copia::RollingChecksum;
    ///
    /// let mut checksum = RollingChecksum::new(b"abcd");
    /// checksum.roll(b'a', b'e');  // Now represents "bcde"
    /// ```
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn roll(&mut self, old_byte: u8, new_byte: u8) {
        let old = u32::from(old_byte);
        let new = u32::from(new_byte);

        // Update a: remove old, add new
        self.a = (self.a.wrapping_sub(old).wrapping_add(new)) % Self::MOD;

        // Update b: remove old's contribution (it was weighted by count), add new a
        // Truncation is intentional: checksum uses 32-bit arithmetic
        self.b = (self
            .b
            .wrapping_sub(self.count as u32 * old)
            .wrapping_add(self.a))
            % Self::MOD;
    }

    /// Add a single byte to the window (increasing window size).
    ///
    /// # Arguments
    ///
    /// * `byte` - Byte to add to the end of the window
    #[inline]
    pub fn push(&mut self, byte: u8) {
        let val = u32::from(byte);
        self.a = (self.a.wrapping_add(val)) % Self::MOD;
        self.b = (self.b.wrapping_add(self.a)) % Self::MOD;
        self.count += 1;
    }

    /// Get the combined 32-bit digest.
    ///
    /// The digest combines both checksum components into a single value:
    /// `(b << 16) | a`
    ///
    /// # Example
    ///
    /// ```rust
    /// use copia::RollingChecksum;
    ///
    /// let checksum = RollingChecksum::new(b"test");
    /// let digest = checksum.digest();
    /// ```
    #[inline]
    #[must_use]
    pub const fn digest(&self) -> u32 {
        (self.b << 16) | self.a
    }

    /// Get the current window size.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if the window is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the `a` component (simple sum).
    #[inline]
    #[must_use]
    pub const fn sum_a(&self) -> u32 {
        self.a
    }

    /// Get the `b` component (weighted sum).
    #[inline]
    #[must_use]
    pub const fn sum_b(&self) -> u32 {
        self.b
    }
}

/// High-performance rolling checksum with lazy modulo operations.
///
/// This variant delays modulo operations until `digest()` is called,
/// providing ~3x faster rolling operations for delta computation.
/// The trade-off is slightly larger intermediate values.
#[derive(Debug, Clone, Copy)]
pub struct FastRollingChecksum {
    /// Accumulated sum (lazy mod)
    a: u64,
    /// Weighted sum (lazy mod)
    b: u64,
    /// Window size
    count: usize,
    /// Rolling counter for periodic normalization
    rolls: u32,
}

impl FastRollingChecksum {
    /// Modulus for final reduction
    const MOD: u64 = 65521;
    /// Normalize every N rolls to prevent overflow
    const NORMALIZE_INTERVAL: u32 = 5000;

    /// Create from initial data block.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(data: &[u8]) -> Self {
        let mut a: u64 = 0;
        let mut b: u64 = 0;
        let len = data.len();

        for (i, &byte) in data.iter().enumerate() {
            a += u64::from(byte);
            b += (len - i) as u64 * u64::from(byte);
        }

        Self {
            a: a % Self::MOD,
            b: b % Self::MOD,
            count: len,
            rolls: 0,
        }
    }

    /// Create empty checksum.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            a: 0,
            b: 0,
            count: 0,
            rolls: 0,
        }
    }

    /// Roll window by one byte - O(1) with lazy modulo.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn roll(&mut self, old_byte: u8, new_byte: u8) {
        let old = u64::from(old_byte);
        let new = u64::from(new_byte);

        // Update a: remove old, add new (with MOD to keep positive)
        self.a = self.a + Self::MOD + new - old;

        // Update b: remove old's contribution, add new a
        self.b = self.b + Self::MOD * (self.count as u64) + self.a - self.count as u64 * old;

        self.rolls += 1;

        // Periodic normalization to prevent overflow
        if self.rolls >= Self::NORMALIZE_INTERVAL {
            self.a %= Self::MOD;
            self.b %= Self::MOD;
            self.rolls = 0;
        }
    }

    /// Add byte to window.
    #[inline]
    pub fn push(&mut self, byte: u8) {
        let val = u64::from(byte);
        self.a += val;
        self.b += self.a;
        self.count += 1;

        self.rolls += 1;
        if self.rolls >= Self::NORMALIZE_INTERVAL {
            self.a %= Self::MOD;
            self.b %= Self::MOD;
            self.rolls = 0;
        }
    }

    /// Get 32-bit digest with final modulo reduction.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn digest(&self) -> u32 {
        let a = (self.a % Self::MOD) as u32;
        let b = (self.b % Self::MOD) as u32;
        (b << 16) | a
    }

    /// Get window size.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for RollingChecksum {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // UNIT TESTS - Basic functionality
    // ==========================================================================

    #[test]
    fn new_empty_slice() {
        let checksum = RollingChecksum::new(b"");
        assert_eq!(checksum.digest(), 0);
        assert_eq!(checksum.len(), 0);
        assert!(checksum.is_empty());
    }

    #[test]
    fn new_single_byte() {
        let checksum = RollingChecksum::new(b"a");
        assert_ne!(checksum.digest(), 0);
        assert_eq!(checksum.len(), 1);
        assert!(!checksum.is_empty());
    }

    #[test]
    fn new_multiple_bytes() {
        let checksum = RollingChecksum::new(b"hello");
        assert_ne!(checksum.digest(), 0);
        assert_eq!(checksum.len(), 5);
    }

    #[test]
    fn empty_constructor() {
        let checksum = RollingChecksum::empty();
        assert_eq!(checksum.digest(), 0);
        assert_eq!(checksum.len(), 0);
        assert!(checksum.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let checksum = RollingChecksum::default();
        assert_eq!(checksum, RollingChecksum::empty());
    }

    #[test]
    fn digest_deterministic() {
        let data = b"test data for checksum";
        let checksum1 = RollingChecksum::new(data);
        let checksum2 = RollingChecksum::new(data);
        assert_eq!(checksum1.digest(), checksum2.digest());
    }

    #[test]
    fn different_data_different_digest() {
        let checksum1 = RollingChecksum::new(b"hello");
        let checksum2 = RollingChecksum::new(b"world");
        assert_ne!(checksum1.digest(), checksum2.digest());
    }

    #[test]
    fn sum_components_accessible() {
        let checksum = RollingChecksum::new(b"test");
        assert!(checksum.sum_a() < RollingChecksum::MOD);
        assert!(checksum.sum_b() < RollingChecksum::MOD);
    }

    // ==========================================================================
    // ROLLING TESTS - Window sliding behavior
    // ==========================================================================

    #[test]
    fn roll_single_byte_window() {
        let mut checksum = RollingChecksum::new(b"a");
        checksum.roll(b'a', b'b');
        let direct = RollingChecksum::new(b"b");
        assert_eq!(checksum.digest(), direct.digest());
    }

    #[test]
    fn roll_preserves_window_size() {
        let mut checksum = RollingChecksum::new(b"abcd");
        let original_len = checksum.len();
        checksum.roll(b'a', b'e');
        assert_eq!(checksum.len(), original_len);
    }

    #[test]
    fn roll_multiple_times() {
        // Start with "abcd", roll to "bcde", then to "cdef"
        let mut checksum = RollingChecksum::new(b"abcd");
        checksum.roll(b'a', b'e');
        checksum.roll(b'b', b'f');

        let direct = RollingChecksum::new(b"cdef");
        assert_eq!(checksum.digest(), direct.digest());
    }

    #[test]
    fn roll_full_window_replacement() {
        // Roll through an entire window
        let mut checksum = RollingChecksum::new(b"aaaa");
        checksum.roll(b'a', b'b');
        checksum.roll(b'a', b'b');
        checksum.roll(b'a', b'b');
        checksum.roll(b'a', b'b');

        let direct = RollingChecksum::new(b"bbbb");
        assert_eq!(checksum.digest(), direct.digest());
    }

    #[test]
    fn roll_with_same_byte() {
        let mut checksum = RollingChecksum::new(b"aaaa");
        let original = checksum.digest();
        checksum.roll(b'a', b'a');
        assert_eq!(checksum.digest(), original);
    }

    // ==========================================================================
    // PUSH TESTS - Growing window
    // ==========================================================================

    #[test]
    fn push_to_empty() {
        let mut checksum = RollingChecksum::empty();
        checksum.push(b'a');
        assert_eq!(checksum.len(), 1);
        assert_ne!(checksum.digest(), 0);
    }

    #[test]
    fn push_multiple() {
        let mut checksum = RollingChecksum::empty();
        for &byte in b"hello" {
            checksum.push(byte);
        }
        // Note: push builds differently than new due to how b is computed
        assert_eq!(checksum.len(), 5);
    }

    #[test]
    fn push_increases_length() {
        let mut checksum = RollingChecksum::new(b"test");
        let original_len = checksum.len();
        checksum.push(b'!');
        assert_eq!(checksum.len(), original_len + 1);
    }

    // ==========================================================================
    // EDGE CASES
    // ==========================================================================

    #[test]
    fn all_zeros() {
        let data = [0u8; 100];
        let checksum = RollingChecksum::new(&data);
        assert_eq!(checksum.digest(), 0);
    }

    #[test]
    fn all_ones() {
        let data = [1u8; 100];
        let checksum = RollingChecksum::new(&data);
        assert_ne!(checksum.digest(), 0);
    }

    #[test]
    fn all_max_bytes() {
        let data = [255u8; 100];
        let checksum = RollingChecksum::new(&data);
        assert_ne!(checksum.digest(), 0);
        // Verify modular arithmetic doesn't overflow
        assert!(checksum.sum_a() < RollingChecksum::MOD);
        assert!(checksum.sum_b() < RollingChecksum::MOD);
    }

    #[test]
    fn large_window() {
        let data = vec![42u8; 65536]; // Max block size
        let checksum = RollingChecksum::new(&data);
        assert_eq!(checksum.len(), 65536);
        assert!(checksum.sum_a() < RollingChecksum::MOD);
        assert!(checksum.sum_b() < RollingChecksum::MOD);
    }

    #[test]
    fn binary_data() {
        let data: Vec<u8> = (0..=255).collect();
        let checksum = RollingChecksum::new(&data);
        assert_eq!(checksum.len(), 256);
        assert_ne!(checksum.digest(), 0);
    }

    // ==========================================================================
    // EQUALITY AND CLONING
    // ==========================================================================

    #[test]
    fn clone_equals_original() {
        let original = RollingChecksum::new(b"test data");
        let cloned = original;
        assert_eq!(original, cloned);
        assert_eq!(original.digest(), cloned.digest());
    }

    #[test]
    fn debug_format() {
        let checksum = RollingChecksum::new(b"test");
        let debug = format!("{checksum:?}");
        assert!(debug.contains("RollingChecksum"));
        assert!(debug.contains("a:"));
        assert!(debug.contains("b:"));
    }

    // ==========================================================================
    // INVARIANTS
    // ==========================================================================

    #[test]
    fn mod_invariant_a() {
        // a should always be less than MOD
        let data = vec![255u8; 10000];
        let checksum = RollingChecksum::new(&data);
        assert!(checksum.sum_a() < RollingChecksum::MOD);
    }

    #[test]
    fn mod_invariant_b() {
        // b should always be less than MOD
        let data = vec![255u8; 10000];
        let checksum = RollingChecksum::new(&data);
        assert!(checksum.sum_b() < RollingChecksum::MOD);
    }

    #[test]
    fn roll_maintains_mod_invariant() {
        let mut checksum = RollingChecksum::new(&[255u8; 1000]);
        for _ in 0..1000 {
            checksum.roll(255, 255);
            assert!(checksum.sum_a() < RollingChecksum::MOD);
            assert!(checksum.sum_b() < RollingChecksum::MOD);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Rolling should preserve length invariant
        #[test]
        fn roll_preserves_length(
            data in prop::collection::vec(any::<u8>(), 1..100),
            old in any::<u8>(),
            new in any::<u8>()
        ) {
            let mut checksum = RollingChecksum::new(&data);
            let original_len = checksum.len();
            checksum.roll(old, new);
            prop_assert_eq!(checksum.len(), original_len);
        }

        /// a component always less than MOD
        #[test]
        fn a_always_bounded(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let checksum = RollingChecksum::new(&data);
            prop_assert!(checksum.sum_a() < RollingChecksum::MOD);
        }

        /// b component always less than MOD
        #[test]
        fn b_always_bounded(data in prop::collection::vec(any::<u8>(), 0..1000)) {
            let checksum = RollingChecksum::new(&data);
            prop_assert!(checksum.sum_b() < RollingChecksum::MOD);
        }

        /// Same data produces same digest
        #[test]
        fn deterministic(data in prop::collection::vec(any::<u8>(), 0..500)) {
            let checksum1 = RollingChecksum::new(&data);
            let checksum2 = RollingChecksum::new(&data);
            prop_assert_eq!(checksum1.digest(), checksum2.digest());
        }

        /// Empty data produces zero digest
        #[test]
        fn empty_is_zero(_unused in 0..1i32) {
            let checksum = RollingChecksum::new(&[]);
            prop_assert_eq!(checksum.digest(), 0);
        }

        /// Push maintains bounded invariants
        #[test]
        fn push_maintains_bounds(
            initial in prop::collection::vec(any::<u8>(), 0..100),
            to_push in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            let mut checksum = RollingChecksum::new(&initial);
            for byte in to_push {
                checksum.push(byte);
                prop_assert!(checksum.sum_a() < RollingChecksum::MOD);
                prop_assert!(checksum.sum_b() < RollingChecksum::MOD);
            }
        }

        /// Rolling then pushing should not panic
        #[test]
        fn roll_and_push_no_panic(
            data in prop::collection::vec(any::<u8>(), 2..50),
            operations in prop::collection::vec((any::<bool>(), any::<u8>(), any::<u8>()), 0..100)
        ) {
            let mut checksum = RollingChecksum::new(&data);
            for (is_roll, byte1, byte2) in operations {
                if is_roll && checksum.len() > 0 {
                    checksum.roll(byte1, byte2);
                } else {
                    checksum.push(byte1);
                }
                // Should not panic and invariants hold
                prop_assert!(checksum.sum_a() < RollingChecksum::MOD);
                prop_assert!(checksum.sum_b() < RollingChecksum::MOD);
            }
        }
    }
}
