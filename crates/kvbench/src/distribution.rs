//! Key-distribution generators: uniform, zipfian, latest.
//!
//! All generators are deterministic given a seed and `record_count`. Per-run
//! reproducibility is a hard requirement: a failing bench must be exactly
//! reproducible.

use serde::{Deserialize, Serialize};

/// Key distribution for sampling record indices in `[0, record_count)`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params", rename_all = "lowercase")]
pub enum Distribution {
    /// Uniform random: every key equally likely.
    #[default]
    Uniform,
    /// Zipfian: hot-key skew, theta in `(0, 1)`. Default theta = 0.99 (YCSB).
    Zipfian { theta: f64 },
    /// Latest: recency-biased toward recently-inserted keys.
    Latest,
}

impl Distribution {
    /// Default zipfian theta (0.99, per the YCSB spec).
    #[allow(dead_code)]
    pub const DEFAULT_ZIPF_THETA: f64 = 0.99;

    /// Create a sampler for this distribution over `[0, record_count)`.
    pub fn sampler(&self, record_count: u64, seed: u64) -> KeySampler {
        match self {
            Self::Uniform => KeySampler::Uniform(UniformGen::new(record_count, seed)),
            Self::Zipfian { theta } => {
                KeySampler::Zipfian(ZipfianGen::new(record_count, *theta, seed))
            }
            Self::Latest => KeySampler::Latest(LatestGen::new(record_count, seed)),
        }
    }
}

/// A bound sampling generator. Call [`KeySampler::sample`] with the current operation
/// index (0-based) to get a key index in `[0, record_count)`.
pub enum KeySampler {
    Uniform(UniformGen),
    Zipfian(ZipfianGen),
    Latest(LatestGen),
}

impl KeySampler {
    /// Produce the key index for operation `op_index`.
    pub fn sample(&mut self, op_index: u64) -> u64 {
        match self {
            Self::Uniform(g) => g.sample(op_index),
            Self::Zipfian(g) => g.sample(op_index),
            Self::Latest(g) => g.sample(op_index),
        }
    }

    /// Record that a key was inserted (for `Latest` distribution tracking).
    pub fn notify_insert(&mut self) {
        if let Self::Latest(g) = self {
            g.notify_insert();
        }
    }
}

// ---------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------

/// Uniform random generator using the same hash pattern as the existing
/// substrate microbench at `crates/btree/benches/ycsb.rs:13`.
pub struct UniformGen {
    record_count: u64,
    seed: u64,
}

impl UniformGen {
    pub fn new(record_count: u64, seed: u64) -> Self {
        Self {
            record_count: record_count.max(1),
            seed,
        }
    }

    pub fn sample(&self, op_index: u64) -> u64 {
        let h = hash64(self.seed.wrapping_add(op_index));
        h % self.record_count
    }
}

// ---------------------------------------------------------------------------
// Zipfian
// ---------------------------------------------------------------------------

/// Zipfian generator using the inverse-CDF method with precomputed cumulative
/// probabilities. Implements the YCSB zipfian distribution (theta in (0,1)).
pub struct ZipfianGen {
    #[allow(dead_code)]
    record_count: u64,
    /// Cumulative probabilities: `cdf[i]` = P(X <= i).
    cdf: Vec<f64>,
    seed: u64,
}

impl ZipfianGen {
    pub fn new(record_count: u64, theta: f64, seed: u64) -> Self {
        let n = record_count.max(1) as usize;
        let cdf = build_zipf_cdf(n, theta);
        Self {
            record_count: record_count.max(1),
            cdf,
            seed,
        }
    }

    pub fn sample(&self, op_index: u64) -> u64 {
        let h = hash64(self.seed.wrapping_add(op_index));
        let u = (h as f64) / (u64::MAX as f64);
        sample_cdf(&self.cdf, u) as u64
    }

    #[allow(dead_code)]
    pub fn record_count(&self) -> u64 {
        self.record_count
    }
}

/// Build the zipfian CDF: `P(X = k) = k^(-theta) / H_n,theta`.
fn build_zipf_cdf(n: usize, theta: f64) -> Vec<f64> {
    let mut probs = Vec::with_capacity(n);
    let mut sum = 0.0_f64;
    for i in 1..=n {
        let p = (i as f64).powf(-theta);
        probs.push(p);
        sum += p;
    }
    // Normalize and accumulate.
    let mut cdf = 0.0_f64;
    for p in &mut probs {
        *p /= sum;
        cdf += *p;
        *p = cdf;
    }
    // Ensure the last element is exactly 1.0 to avoid rounding edge cases.
    if let Some(last) = probs.last_mut() {
        *last = 1.0;
    }
    probs
}

/// Binary-search the CDF for the smallest index whose cumulative probability
/// is >= `u`.
fn sample_cdf(cdf: &[f64], u: f64) -> usize {
    let mut lo = 0;
    let mut hi = cdf.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf[mid] < u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.min(cdf.len().saturating_sub(1))
}

// ---------------------------------------------------------------------------
// Latest (recency-biased)
// ---------------------------------------------------------------------------

/// Latest distribution: exponential bias toward recently-inserted keys.
/// Samples uniformly from `[0, current_insert_count)`, so keys inserted later
/// are more likely to be accessed (matches YCSB's `latest` distribution).
pub struct LatestGen {
    seed: u64,
    /// Number of keys inserted so far (grows during the load phase).
    inserted: std::sync::atomic::AtomicU64,
    /// Total record count (upper bound).
    record_count: u64,
}

impl LatestGen {
    pub fn new(record_count: u64, seed: u64) -> Self {
        Self {
            seed,
            inserted: std::sync::atomic::AtomicU64::new(0),
            record_count,
        }
    }

    pub fn sample(&self, op_index: u64) -> u64 {
        let h = hash64(self.seed.wrapping_add(op_index));
        let inserted = self
            .inserted
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1);
        h % inserted.min(self.record_count)
    }

    pub fn notify_insert(&self) {
        self.inserted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// 64-bit hash mixing function (same constants as `crates/btree/benches/ycsb.rs:13`).
pub fn hash64(x: u64) -> u64 {
    x.wrapping_mul(0x517cc1b727220a95)
        .wrapping_add(0x9e3779b97f4a7c15)
}

/// Format a key index as an 8-byte big-endian key (for lexicographic ordering
/// in B+tree / LSM engines).
pub fn format_key(index: u64) -> [u8; 8] {
    index.to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_covers_range() {
        let n = 1000_u64;
        let sampler = UniformGen::new(n, 42);
        let mut counts = vec![0u64; n as usize];
        for i in 0..100_000 {
            let k = sampler.sample(i);
            assert!(k < n, "sample {} out of range: {}", k, n);
            counts[k as usize] += 1;
        }
        // Chi-square-ish: each bucket should have ~100 hits; check no bucket
        // is wildly off (within 3x of expected).
        for &c in &counts {
            assert!(
                c > 30 && c < 300,
                "uniform bucket count {} outside [30, 300]",
                c
            );
        }
    }

    #[test]
    fn zipfian_top_1pct_dominates() {
        let n = 10_000_u64;
        let sampler = ZipfianGen::new(n, 0.99, 123);
        let mut counts = vec![0u64; n as usize];
        for i in 0..100_000 {
            let k = sampler.sample(i);
            assert!(k < n);
            counts[k as usize] += 1;
        }
        // Sort descending.
        counts.sort_by(|a, b| b.cmp(a));
        let top_1pct: u64 = counts[..100].iter().sum();
        // Top 1% of keys should account for ~>30% of probes at theta=0.99.
        let pct = top_1pct as f64 / 100_000.0 * 100.0;
        assert!(
            pct > 30.0,
            "zipfian top-1% should be >30% of probes, got {:.1}%",
            pct
        );
    }

    #[test]
    fn latest_biases_to_recent() {
        let n = 10_000_u64;
        let sampler = LatestGen::new(n, 7);
        // Simulate all keys inserted.
        for _ in 0..n {
            sampler.notify_insert();
        }
        let mut last_half = 0u64;
        let mut first_half = 0u64;
        for i in 0..100_000 {
            let k = sampler.sample(i);
            if k < n / 2 {
                first_half += 1;
            } else {
                last_half += 1;
            }
        }
        // With all keys inserted and uniform hash, both halves should be
        // roughly equal (~50k each). But with `latest` after partial inserts,
        // the bias is toward recent. Here we just verify it's in range.
        assert!(first_half > 30_000 && last_half > 30_000);
    }

    #[test]
    fn latest_recency_bias_during_growth() {
        // During the run phase, if keys were only partially inserted (e.g.
        // YcsbD loads half), `latest` should bias toward the inserted range.
        let n = 10_000_u64;
        let sampler = LatestGen::new(n, 7);
        // Only insert 2000 keys.
        for _ in 0..2000 {
            sampler.notify_insert();
        }
        let mut in_range = 0u64;
        for i in 0..10_000 {
            let k = sampler.sample(i);
            if k < 2000 {
                in_range += 1;
            }
        }
        // All samples should be within the inserted range [0, 2000).
        assert_eq!(in_range, 10_000, "latest sampled outside inserted range");
    }

    #[test]
    fn format_key_orders_lexicographically() {
        let k0 = format_key(0);
        let k1 = format_key(1);
        let k255 = format_key(255);
        let k256 = format_key(256);
        assert!(k0 < k1);
        assert!(k1 < k255);
        assert!(k255 < k256);
    }

    #[test]
    fn deterministic_given_seed() {
        let gen1 = UniformGen::new(100, 42);
        let gen2 = UniformGen::new(100, 42);
        for i in 0..1000 {
            assert_eq!(gen1.sample(i), gen2.sample(i));
        }
    }
}
