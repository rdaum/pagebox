//! No-op stand-ins for `fast-telemetry` metric types.
//!
//! Active when the `metrics` feature is off. Call sites stay unchanged: the
//! shim types expose the same method surface as the real types but discard
//! every value.

#![allow(dead_code)]

use std::marker::PhantomData;

pub trait MetricVisitor {}

pub struct LabeledCounter<L>(PhantomData<L>);

impl<L> LabeledCounter<L> {
    pub fn new(_shards: usize) -> Self {
        Self(PhantomData)
    }

    #[inline]
    pub fn inc(&self, _label: L) {}

    #[inline]
    pub fn add(&self, _label: L, _delta: isize) {}
}

pub struct LabeledHistogram<L>(PhantomData<L>);

impl<L> LabeledHistogram<L> {
    pub fn new(_bounds: &[u64], _shards: usize) -> Self {
        Self(PhantomData)
    }

    #[inline]
    pub fn get(&self, _label: L) -> HistogramHandle {
        HistogramHandle
    }
}

pub struct HistogramHandle;

impl HistogramHandle {
    #[inline]
    pub fn record(&self, _value: u64) {}
}
