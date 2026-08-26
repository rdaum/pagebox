// Keep this benchmark-only module in sync across the four measured crates. Per-crate copies keep published crate
// packages self-contained without adding a production dependency.

use std::time::Duration;

use micromeasure::bench::BenchmarkGroup;
use micromeasure::{
    BenchContext, BenchmarkCaseOrder, BenchmarkRunner, ConcurrentBenchContext,
    ConcurrentBenchmarkGroup, MeasurementBackend,
};

#[cfg(not(target_os = "linux"))]
use micromeasure::WallClockBackend;
#[cfg(target_os = "linux")]
use micromeasure::{LinuxPerfBackend, PmuCounterProfile};

#[cfg(target_os = "linux")]
const PMU_ENV: &str = "PAGEBOX_BENCH_PMU";
#[cfg(target_os = "linux")]
const RAPL_ENV: &str = "PAGEBOX_BENCH_RAPL";
#[cfg(target_os = "linux")]
const MEMORY_BANDWIDTH_ENV: &str = "PAGEBOX_BENCH_MEMORY_BANDWIDTH";

/// A benchmark runner that applies Pagebox's measurement policy to every group.
pub struct PageboxBenchmarkRunner<'a> {
    inner: &'a mut BenchmarkRunner,
}

#[allow(
    dead_code,
    reason = "each benchmark target uses a subset of the shared runner API"
)]
impl<'a> PageboxBenchmarkRunner<'a> {
    pub fn new(inner: &'a mut BenchmarkRunner) -> Self {
        Self { inner }
    }

    pub fn group<T: BenchContext>(
        &self,
        name: &'static str,
        register: impl FnOnce(&BenchmarkGroup<'_, T>),
    ) {
        self.inner.group::<T>(name, |group| {
            let group = group.backend(cpu_measurement_backend);
            register(&group);
        });
    }

    pub fn concurrent_group<T: ConcurrentBenchContext + Send + Sync>(
        &self,
        name: &'static str,
        register: impl FnOnce(&ConcurrentBenchmarkGroup<'_, T>),
    ) {
        self.inner.concurrent_group::<T>(name, |group| {
            let group = group.backend(cpu_measurement_backend);
            register(&group);
        });
    }

    pub fn set_case_cooldown(&mut self, cooldown: Duration) {
        self.inner.set_case_cooldown(cooldown);
    }

    pub fn ordered_case_indices(&self, case_count: usize, order: BenchmarkCaseOrder) -> Vec<usize> {
        self.inner.ordered_case_indices(case_count, order)
    }
}

#[cfg(target_os = "linux")]
fn environment_value(name: &str, default: &'static str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

#[cfg(target_os = "linux")]
fn cpu_measurement_backend() -> Box<dyn MeasurementBackend> {
    let profile = match environment_value(PMU_ENV, "full").as_str() {
        "full" => PmuCounterProfile::Full,
        "compact" => PmuCounterProfile::Compact,
        "none" => PmuCounterProfile::None,
        value => panic!("{PMU_ENV} must be one of full, compact, or none; got {value:?}"),
    };

    let mut backend = LinuxPerfBackend::new().with_counter_profile(profile);
    backend = match environment_value(RAPL_ENV, "off").as_str() {
        "off" => backend,
        "package" => backend.with_rapl_energy(),
        "package-core" => backend.with_rapl_core_energy(),
        value => {
            panic!("{RAPL_ENV} must be one of off, package, or package-core; got {value:?}")
        }
    };
    backend = match environment_value(MEMORY_BANDWIDTH_ENV, "auto").as_str() {
        "auto" => backend,
        "requested" => backend.with_memory_bandwidth(),
        "off" => backend.without_memory_bandwidth(),
        value => {
            panic!("{MEMORY_BANDWIDTH_ENV} must be one of auto, requested, or off; got {value:?}")
        }
    };

    Box::new(backend)
}

#[cfg(not(target_os = "linux"))]
fn cpu_measurement_backend() -> Box<dyn MeasurementBackend> {
    Box::new(WallClockBackend::new())
}
