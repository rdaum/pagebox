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
}

pub struct LabeledCounter<L>(PhantomData<L>);

impl<L> LabeledCounter<L> {
    pub fn new(_shards: usize) -> Self {
        Self(PhantomData)
    }

    #[inline]
    pub fn inc(&self, _label: L) {}
}
