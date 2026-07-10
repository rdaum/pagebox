//! Engine adapters.
//!
//! Each submodule wraps a KV engine in the [`KvEngine`](crate::engine::KvEngine) trait.

pub mod kvstore_adapter;

#[cfg(feature = "fjall")]
pub mod fjall;

#[cfg(feature = "redb")]
pub mod redb;

#[cfg(feature = "lmdb")]
pub mod lmdb;

#[cfg(feature = "rocksdb")]
pub mod rocksdb;
