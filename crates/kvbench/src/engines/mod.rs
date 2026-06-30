//! Engine adapters.
//!
//! Each submodule wraps an external KV engine in the [`KvEngine`] trait.

pub mod kvstore_adapter;

#[cfg(feature = "fjall")]
pub mod fjall;

#[cfg(feature = "redb")]
pub mod redb;

#[cfg(feature = "sled")]
pub mod sled;
