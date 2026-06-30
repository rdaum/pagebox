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
    pub fn add(&self, _value: isize) {}
}

pub struct LabeledCounter<L>(PhantomData<L>);

impl<L> LabeledCounter<L> {
    pub fn new(_shards: usize) -> Self {
        Self(PhantomData)
    }

    #[inline]
    pub fn inc(&self, _label: L) {}

    #[inline]
    pub fn add(&self, _label: L, _value: isize) {}
}
