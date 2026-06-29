//! No-op stand-ins for `fast-telemetry` metric types.
//!
//! Active when the `metrics` feature is off. Call sites stay unchanged: the
//! shim types expose the same method surface as the real types but discard
//! every value.

#![allow(dead_code)]

use std::marker::PhantomData;

pub trait MetricVisitor {}

pub struct Counter;

impl Counter {
    pub fn new(_shards: usize) -> Self {
        Self
    }

    #[inline]
    pub fn inc(&self) {}

    #[inline]
    pub fn add(&self, _delta: isize) {}
}

pub struct Gauge;

impl Gauge {
    pub fn new() -> Self {
        Self
    }

    #[inline]
    pub fn set(&self, _value: i64) {}
}

pub struct Histogram;

impl Histogram {
    pub fn new(_bounds: &[u64], _shards: usize) -> Self {
        Self
    }

    #[inline]
    pub fn record(&self, _value: u64) {}
}

pub struct LabeledCounter<L>(PhantomData<L>);

impl<L> LabeledCounter<L> {
    pub fn new(_shards: usize) -> Self {
        Self(PhantomData)
    }

    #[inline]
    pub fn inc(&self, _label: L) {}

    #[inline]
    pub fn add(&self, _label: L, _delta: isize) {}

    #[inline]
    pub fn get(&self, _label: L) -> u64 {
        0
    }
}

pub struct LabeledGauge<L>(PhantomData<L>);

impl<L> LabeledGauge<L> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    #[inline]
    pub fn set(&self, _label: L, _value: i64) {}
}
