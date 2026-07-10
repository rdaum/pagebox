//! Comparison contracts carried by benchmark specs and reports.
//!
//! A contract defines which engines may appear in one comparison and what the
//! configured memory budget means. Keeping this in the report prevents the
//! summarizer from ranking runs that were collected under different rules.

use serde::{Deserialize, Serialize};

use crate::engine::{CacheControl, EngineOpts, KvEngine, SyncMode};
use crate::workload::WorkloadSpec;

/// How the benchmark constrains the measured working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRegime {
    /// The configured cache is deliberately larger than the data set.
    Resident,
    /// Only the engine's application-managed cache is bounded.
    ///
    /// The operating-system page cache is not controlled by kvbench, so this
    /// must not be described as a physical-I/O or total-RSS comparison.
    ApplicationCache,
    /// The application cache is bounded and data-file reads bypass the OS
    /// page cache. Only engines with an explicit direct-read mode participate.
    DirectIoApplicationCache,
}

/// What completion of a measured operation means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationContract {
    /// Point reads against a fully loaded and durably synchronized data set.
    PointRead,
    /// A mutation is visible to subsequent operations when its call returns;
    /// crash durability may lag until the post-run durability drain.
    VisiblePointMutation,
    /// Every mutation is crash-durable when its call returns.
    DurablePointMutation,
}

/// Rules shared by every engine in one scenario.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonContract {
    /// Stable scenario identifier used to group repeated measurements.
    pub id: String,
    /// Engines that are valid participants in this scenario.
    pub engines: Vec<String>,
    /// Meaning of the configured cache budget.
    pub memory_regime: MemoryRegime,
    /// Completion semantics of timed operations.
    pub operation_contract: OperationContract,
}

impl ComparisonContract {
    /// Validate engine-independent contract and workload relationships.
    pub fn validate_workload(&self, spec: &WorkloadSpec, opts: &EngineOpts) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("comparison.id must not be empty".to_string());
        }
        if self.engines.is_empty() {
            return Err(format!(
                "comparison '{}' must declare at least one engine",
                self.id
            ));
        }
        for (index, engine) in self.engines.iter().enumerate() {
            if self.engines[..index].contains(engine) {
                return Err(format!(
                    "comparison '{}' lists engine '{}' more than once",
                    self.id, engine
                ));
            }
        }
        if opts.cache_budget_bytes == 0 {
            return Err("cache_budget_bytes must be greater than zero".to_string());
        }
        if self.memory_regime == MemoryRegime::DirectIoApplicationCache && !opts.direct_io {
            return Err(format!(
                "comparison '{}' requires engine_opts.direct_io = true",
                self.id
            ));
        }
        if self.memory_regime != MemoryRegime::DirectIoApplicationCache && opts.direct_io {
            return Err(format!(
                "comparison '{}' enables direct I/O without the direct_io_application_cache memory regime",
                self.id
            ));
        }

        let has_mutations = spec.workload.has_mutations();
        match self.operation_contract {
            OperationContract::PointRead if has_mutations => Err(format!(
                "comparison '{}' declares point reads but workload '{}' mutates data",
                self.id, spec.workload
            )),
            OperationContract::VisiblePointMutation
                if !has_mutations || opts.sync_mode != SyncMode::Relaxed =>
            {
                Err(format!(
                    "comparison '{}' requires a mutating workload with relaxed sync",
                    self.id
                ))
            }
            OperationContract::DurablePointMutation
                if !has_mutations || opts.sync_mode != SyncMode::Strict =>
            {
                Err(format!(
                    "comparison '{}' requires a mutating workload with strict sync",
                    self.id
                ))
            }
            _ => Ok(()),
        }
    }

    /// Validate that a selected engine and workload satisfy this contract.
    pub fn validate<E: KvEngine>(
        &self,
        spec: &WorkloadSpec,
        opts: &EngineOpts,
    ) -> Result<(), String> {
        self.validate_workload(spec, opts)?;
        if !self.engines.iter().any(|engine| engine == E::NAME) {
            return Err(format!(
                "engine '{}' is not in comparison cohort '{}' ({})",
                E::NAME,
                self.id,
                self.engines.join(", ")
            ));
        }
        if matches!(
            self.memory_regime,
            MemoryRegime::ApplicationCache | MemoryRegime::DirectIoApplicationCache
        ) && E::CACHE_CONTROL != CacheControl::Application
        {
            return Err(format!(
                "engine '{}' has OS-managed residency and cannot participate in application-cache comparison '{}'",
                E::NAME,
                self.id
            ));
        }
        if self.memory_regime == MemoryRegime::DirectIoApplicationCache && !E::SUPPORTS_DIRECT_IO {
            return Err(format!(
                "engine '{}' does not expose direct data-file reads for comparison '{}'",
                E::NAME,
                self.id
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::engine::EngineStats;
    use crate::workload::Workload;

    struct OsManagedEngine;

    impl KvEngine for OsManagedEngine {
        const NAME: &'static str = "os-managed";
        const CACHE_CONTROL: CacheControl = CacheControl::OsManaged;

        fn open(_dir: &Path, _opts: &EngineOpts) -> std::io::Result<Self> {
            Ok(Self)
        }

        fn put(&self, _key: &[u8], _value: &[u8]) {}

        fn get(&self, _key: &[u8]) -> Option<Vec<u8>> {
            None
        }

        fn del(&self, _key: &[u8]) {}

        fn scan_range(&self, _start: &[u8], _end: &[u8], _f: &mut dyn FnMut(&[u8], &[u8])) {}

        fn sync(&self) -> std::io::Result<()> {
            Ok(())
        }

        fn stats(&self) -> EngineStats {
            EngineStats::default()
        }
    }

    fn contract(
        memory_regime: MemoryRegime,
        operation_contract: OperationContract,
    ) -> ComparisonContract {
        ComparisonContract {
            id: "test".to_string(),
            engines: vec![OsManagedEngine::NAME.to_string()],
            memory_regime,
            operation_contract,
        }
    }

    #[test]
    fn application_cache_contract_rejects_os_managed_engine() {
        let spec = WorkloadSpec {
            workload: Workload::YcsbC,
            ..WorkloadSpec::default()
        };
        let result = contract(MemoryRegime::ApplicationCache, OperationContract::PointRead)
            .validate::<OsManagedEngine>(&spec, &EngineOpts::default());
        assert!(
            result.is_err(),
            "OS-managed engines must not enter application-cache comparisons"
        );
    }

    #[test]
    fn point_read_contract_rejects_mutating_workload() {
        let spec = WorkloadSpec {
            workload: Workload::YcsbA,
            ..WorkloadSpec::default()
        };
        let result = contract(MemoryRegime::Resident, OperationContract::PointRead)
            .validate::<OsManagedEngine>(&spec, &EngineOpts::default());
        assert!(
            result.is_err(),
            "read-only comparison must reject mutation workloads"
        );
    }

    #[test]
    fn relaxed_mutation_contract_accepts_visible_point_updates() {
        let spec = WorkloadSpec {
            workload: Workload::YcsbA,
            ..WorkloadSpec::default()
        };
        contract(
            MemoryRegime::Resident,
            OperationContract::VisiblePointMutation,
        )
        .validate::<OsManagedEngine>(&spec, &EngineOpts::default())
        .expect("relaxed point mutation should satisfy the visible contract");
    }
}
