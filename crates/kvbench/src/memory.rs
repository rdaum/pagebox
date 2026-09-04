//! Process and containing-cgroup memory evidence for one measured phase.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Memory evidence kept separate from engine-managed cache and WAL capacity.
/// Missing kernel fields are serialized as `null`, never as zero.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProcessMemoryStats {
    pub phase_peak_rss_bytes: Option<u64>,
    pub phase_peak_rss_anon_bytes: Option<u64>,
    pub phase_peak_rss_file_bytes: Option<u64>,
    pub phase_peak_pss_bytes: Option<u64>,
    /// Linux `VmHWM`: a process-lifetime value that can include the load phase.
    pub process_lifetime_peak_rss_bytes: Option<u64>,
    /// Memory attributed to the process's containing cgroup, which may be
    /// shared with other processes and is therefore not engine RSS.
    pub cgroup_current_bytes: Option<u64>,
    pub cgroup_peak_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProcessMemorySample {
    rss_bytes: Option<u64>,
    rss_anon_bytes: Option<u64>,
    rss_file_bytes: Option<u64>,
    pss_bytes: Option<u64>,
    lifetime_peak_rss_bytes: Option<u64>,
}

impl ProcessMemorySample {
    fn merge_max(&mut self, other: Self) {
        self.rss_bytes = option_max(self.rss_bytes, other.rss_bytes);
        self.rss_anon_bytes = option_max(self.rss_anon_bytes, other.rss_anon_bytes);
        self.rss_file_bytes = option_max(self.rss_file_bytes, other.rss_file_bytes);
        self.pss_bytes = option_max(self.pss_bytes, other.pss_bytes);
        self.lifetime_peak_rss_bytes =
            option_max(self.lifetime_peak_rss_bytes, other.lifetime_peak_rss_bytes);
    }
}

fn option_max(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    }
}

/// Samples Linux process memory while a measured phase is running.
pub struct ProcessMemorySampler {
    stop: Arc<AtomicBool>,
    peak: Arc<Mutex<ProcessMemorySample>>,
    worker: Option<JoinHandle<()>>,
}

impl ProcessMemorySampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(Mutex::new(read_process_memory()));
        let worker_stop = Arc::clone(&stop);
        let worker_peak = Arc::clone(&peak);
        let worker = thread::Builder::new()
            .name("kvbench-memory".to_string())
            .spawn(move || {
                while !worker_stop.load(Ordering::Acquire) {
                    let sample = read_process_memory();
                    worker_peak.lock().unwrap().merge_max(sample);
                    thread::park_timeout(Duration::from_millis(50));
                }
            })
            .ok();
        Self { stop, peak, worker }
    }

    pub fn finish(mut self) -> ProcessMemoryStats {
        self.stop_worker();
        let mut peak = *self.peak.lock().unwrap();
        peak.merge_max(read_process_memory());
        let (cgroup_current_bytes, cgroup_peak_bytes) = read_cgroup_memory();
        ProcessMemoryStats {
            phase_peak_rss_bytes: peak.rss_bytes,
            phase_peak_rss_anon_bytes: peak.rss_anon_bytes,
            phase_peak_rss_file_bytes: peak.rss_file_bytes,
            phase_peak_pss_bytes: peak.pss_bytes,
            process_lifetime_peak_rss_bytes: peak.lifetime_peak_rss_bytes,
            cgroup_current_bytes,
            cgroup_peak_bytes,
        }
    }

    fn stop_worker(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = &self.worker {
            worker.thread().unpark();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for ProcessMemorySampler {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

#[cfg(target_os = "linux")]
fn read_process_memory() -> ProcessMemorySample {
    let status = std::fs::read_to_string("/proc/self/status").ok();
    let smaps = std::fs::read_to_string("/proc/self/smaps_rollup").ok();
    let mut sample = status.as_deref().map(parse_proc_status).unwrap_or_default();
    sample.pss_bytes = smaps.as_deref().and_then(parse_smaps_rollup_pss);
    sample
}

#[cfg(not(target_os = "linux"))]
fn read_process_memory() -> ProcessMemorySample {
    ProcessMemorySample::default()
}

#[cfg(target_os = "linux")]
fn parse_proc_status(status: &str) -> ProcessMemorySample {
    ProcessMemorySample {
        rss_bytes: parse_kib_field(status, "VmRSS:"),
        rss_anon_bytes: parse_kib_field(status, "RssAnon:"),
        rss_file_bytes: parse_kib_field(status, "RssFile:"),
        lifetime_peak_rss_bytes: parse_kib_field(status, "VmHWM:"),
        ..ProcessMemorySample::default()
    }
}

#[cfg(target_os = "linux")]
fn parse_smaps_rollup_pss(smaps: &str) -> Option<u64> {
    parse_kib_field(smaps, "Pss:")
}

#[cfg(target_os = "linux")]
fn parse_kib_field(input: &str, name: &str) -> Option<u64> {
    let line = input.lines().find(|line| line.starts_with(name))?;
    let mut parts = line[name.len()..].split_whitespace();
    let value = parts.next()?.parse::<u64>().ok()?;
    if parts.next()? != "kB" {
        return None;
    }
    value.checked_mul(1024)
}

#[cfg(target_os = "linux")]
fn read_cgroup_memory() -> (Option<u64>, Option<u64>) {
    let Some(relative_path) =
        std::fs::read_to_string("/proc/self/cgroup")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find_map(|line| line.strip_prefix("0::"))
                    .map(str::to_owned)
            })
    else {
        return (None, None);
    };
    let relative_path = relative_path.trim_start_matches('/');
    let base = std::path::Path::new("/sys/fs/cgroup").join(relative_path);
    (
        read_byte_value(&base.join("memory.current")),
        read_byte_value(&base.join("memory.peak")),
    )
}

#[cfg(not(target_os = "linux"))]
fn read_cgroup_memory() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(target_os = "linux")]
fn read_byte_value(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;

    #[test]
    fn proc_parsers_preserve_missing_values_as_unavailable() {
        let status = "VmHWM:\t99 kB\nVmRSS:\t42 kB\nRssAnon:\t17 kB\nRssFile:\t25 kB\n";
        let parsed = parse_proc_status(status);
        assert_eq!(parsed.rss_bytes, Some(42 * 1024));
        assert_eq!(parsed.rss_anon_bytes, Some(17 * 1024));
        assert_eq!(parsed.rss_file_bytes, Some(25 * 1024));
        assert_eq!(parsed.lifetime_peak_rss_bytes, Some(99 * 1024));
        assert_eq!(parse_smaps_rollup_pss("Pss:\t31 kB\n"), Some(31 * 1024));
        assert_eq!(parse_kib_field(status, "RssShmem:"), None);
        assert_eq!(parse_kib_field("VmRSS: 42 MB\n", "VmRSS:"), None);
    }
}
