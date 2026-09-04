//! Background dirty-page cleaner — flushes dirty pages before workers need
//! them as eviction candidates.
//!
//! The provider runs a continuous loop:
//! 1. If budget is sufficient, sleep until woken by a worker.
//! 2. Flush a bounded batch of dirty pages already covered by durable WAL so
//!    the workers' regular eviction path has clean candidates before the pool
//!    is exhausted.
//! 3. Sleep briefly to bound writeback bandwidth; workers retain ownership of
//!    eviction and the associated parent-unswizzle protocol.
//!
//! Embedders can supply [`DirtyCleanerConfig`] explicitly. Standalone
//! experiments can enable the default policy with
//! `PAGEBOX_ENABLE_BACKGROUND_PAGE_PROVIDER=1` and override its
//! `PAGEBOX_DIRTY_CLEANER_TARGET_FREE_FRAMES`,
//! `PAGEBOX_DIRTY_CLEANER_BATCH_SIZE`,
//! `PAGEBOX_DIRTY_CLEANER_MIN_IDLE_WAIT_US`, and
//! `PAGEBOX_DIRTY_CLEANER_MAX_IDLE_WAIT_US` values before constructing a pool.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use pagebox_threading as threading;

use crate::buffer_pool::BufferPool;

const MIN_IDLE_WAIT: Duration = Duration::from_millis(1);
const MAX_IDLE_WAIT: Duration = Duration::from_millis(100);
const DIRTY_BATCH_SIZE: usize = 64;

fn target_free_frames(num_frames: usize) -> usize {
    ((num_frames / 10).max(16)).min(num_frames)
}

/// Bounded policy for the background dirty-page cleaner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyCleanerConfig {
    /// Start cleaning once fewer than this many resident-budget tokens remain.
    pub target_free_frames: usize,
    /// Maximum dirty pages copied and written by one pass.
    pub batch_size: usize,
    /// Pause after a productive pass and initial retry delay after a null pass.
    pub min_idle_wait: Duration,
    /// Maximum exponential retry delay while no durable dirty page is found.
    pub max_idle_wait: Duration,
}

impl DirtyCleanerConfig {
    /// Default policy scaled to the supplied buffer-pool capacity.
    pub fn for_pool(num_frames: usize) -> Self {
        Self {
            target_free_frames: target_free_frames(num_frames),
            batch_size: DIRTY_BATCH_SIZE,
            min_idle_wait: MIN_IDLE_WAIT,
            max_idle_wait: MAX_IDLE_WAIT,
        }
    }

    fn validate_for_pool(mut self, num_frames: usize) -> Self {
        assert!(num_frames > 0, "dirty cleaner requires a non-empty pool");
        assert!(
            self.target_free_frames > 0,
            "dirty cleaner target_free_frames must be greater than zero"
        );
        assert!(
            self.batch_size > 0,
            "dirty cleaner batch_size must be greater than zero"
        );
        assert!(
            !self.min_idle_wait.is_zero(),
            "dirty cleaner min_idle_wait must be greater than zero"
        );
        assert!(
            self.max_idle_wait >= self.min_idle_wait,
            "dirty cleaner max_idle_wait must not be shorter than min_idle_wait"
        );
        self.target_free_frames = self.target_free_frames.min(num_frames);
        self
    }
}

pub struct PageProviderHandle {
    thread: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    pub(crate) need_frames: Arc<(Mutex<()>, Condvar)>,
    pub(crate) frames_available: Arc<(Mutex<()>, Condvar)>,
}

impl Default for PageProviderHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl PageProviderHandle {
    pub fn new() -> Self {
        Self {
            thread: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            need_frames: Arc::new((Mutex::new(()), Condvar::new())),
            frames_available: Arc::new((Mutex::new(()), Condvar::new())),
        }
    }

    pub fn start(&mut self, pool: Weak<BufferPool>) {
        let Some(strong) = pool.upgrade() else {
            return;
        };
        let config = DirtyCleanerConfig::for_pool(strong.num_frames());
        drop(strong);
        self.start_with_config(pool, config);
    }

    pub fn start_with_config(&mut self, pool: Weak<BufferPool>, config: DirtyCleanerConfig) {
        if self.is_running() {
            return;
        }
        let Some(strong) = pool.upgrade() else {
            return;
        };
        let config = config.validate_for_pool(strong.num_frames());
        drop(strong);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.shutdown.store(false, Ordering::Release);
        let shutdown = self.shutdown.clone();
        let need_frames = self.need_frames.clone();
        let frames_available = self.frames_available.clone();
        self.thread = Some(
            threading::spawn_efficient("page-provider", move || {
                run(pool, config, &shutdown, &need_frames, &frames_available);
            })
            .expect("failed to spawn page provider"),
        );
    }

    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.need_frames.1.notify_one();
        if let Some(h) = self.thread.take()
            && h.thread().id() != std::thread::current().id()
        {
            let _ = h.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
    }

    pub fn frames_available_notify(&self) {
        self.frames_available.1.notify_all();
    }

    pub fn need_frames_notify(&self) {
        self.need_frames.1.notify_one();
    }
}

impl Drop for PageProviderHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(
    pool: Weak<BufferPool>,
    config: DirtyCleanerConfig,
    shutdown: &AtomicBool,
    need_frames: &(Mutex<()>, Condvar),
    _frames_available: &(Mutex<()>, Condvar),
) {
    let mut idle_wait = config.min_idle_wait;
    while !shutdown.load(Ordering::Relaxed) {
        let Some(pool) = pool.upgrade() else {
            return;
        };
        let available = pool.approx_available_budget();
        if available >= config.target_free_frames {
            // Budget sufficient — sleep until a worker signals need.
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, Duration::from_millis(10));
            idle_wait = config.min_idle_wait;
            continue;
        }

        if !pool.has_dirty_resident_pages_for_provider() {
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, idle_wait);
            idle_wait = idle_wait.saturating_mul(2).min(config.max_idle_wait);
            continue;
        }

        let cleaned = pool
            .try_flush_dirty_batch_for_provider(config.batch_size)
            .unwrap_or_else(|error| panic!("background page provider dirty flush failed: {error}"));
        if cleaned == 0 {
            // Cleaning does not replenish resident-budget tokens. Once the
            // worker eviction path has caught up, wait for a new pressure
            // signal instead of repeatedly rescanning an all-clean arena.
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, idle_wait);
            idle_wait = idle_wait.saturating_mul(2).min(config.max_idle_wait);
        } else {
            idle_wait = config.min_idle_wait;
            std::thread::sleep(config.min_idle_wait);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use super::{DirtyCleanerConfig, PageProviderHandle, target_free_frames};
    use crate::buffer_pool::BufferPool;

    #[test]
    fn target_free_frames_never_exceeds_pool_capacity() {
        assert_eq!(target_free_frames(1), 1);
        assert_eq!(target_free_frames(15), 15);
        assert_eq!(target_free_frames(16), 16);
        assert_eq!(target_free_frames(160), 16);
        assert_eq!(target_free_frames(1_000), 100);
    }

    #[test]
    fn cleaner_config_scales_with_pool_capacity() {
        let small = DirtyCleanerConfig::for_pool(8);
        assert_eq!(small.target_free_frames, 8);
        assert_eq!(small.batch_size, 64);

        let large = DirtyCleanerConfig::for_pool(1_000);
        assert_eq!(large.target_free_frames, 100);
        assert_eq!(large.min_idle_wait, Duration::from_millis(1));
        assert_eq!(large.max_idle_wait, Duration::from_millis(100));

        let clamped = DirtyCleanerConfig {
            target_free_frames: 100,
            ..small
        }
        .validate_for_pool(8);
        assert_eq!(clamped.target_free_frames, 8);
    }

    #[test]
    #[should_panic(expected = "dirty cleaner batch_size must be greater than zero")]
    fn cleaner_rejects_a_zero_batch() {
        let pool = Arc::new(BufferPool::new(8));
        let mut cleaner = PageProviderHandle::new();
        let config = DirtyCleanerConfig {
            batch_size: 0,
            ..DirtyCleanerConfig::for_pool(8)
        };
        cleaner.start_with_config(Arc::downgrade(&pool), config);
    }

    #[test]
    fn finished_cleaner_is_reaped_before_restart() {
        let pool = Arc::new(BufferPool::new(32));
        let mut cleaner = PageProviderHandle::new();
        cleaner.thread = Some(std::thread::spawn(|| {}));

        let deadline = Instant::now() + Duration::from_secs(1);
        while cleaner
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
            && Instant::now() < deadline
        {
            std::thread::yield_now();
        }

        assert!(
            !cleaner.is_running(),
            "a finished cleaner thread must not be reported as running"
        );
        cleaner.start(Arc::downgrade(&pool));
        assert!(
            cleaner.is_running(),
            "starting after a finished cleaner must replace the stale handle"
        );
        cleaner.stop();
    }

    #[test]
    fn cleaner_shutdown_does_not_join_its_own_thread() {
        let cleaner = Arc::new(Mutex::new(PageProviderHandle::new()));
        let start = Arc::new(Barrier::new(2));
        let (done_tx, done_rx) = mpsc::channel();
        let worker_cleaner = Arc::clone(&cleaner);
        let worker_start = Arc::clone(&start);
        let thread = std::thread::spawn(move || {
            worker_start.wait();
            worker_cleaner.lock().unwrap().stop();
            done_tx.send(()).unwrap();
        });
        cleaner.lock().unwrap().thread = Some(thread);

        start.wait();
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cleaner shutdown from its own thread must return without joining itself");
    }
}
