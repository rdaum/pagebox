//! Background dirty-page cleaner — flushes dirty pages before workers need
//! them as eviction candidates.
//!
//! The provider runs a continuous loop:
//! 1. If budget is sufficient, sleep until woken by a worker.
//! 2. Flush a bounded batch of dirty pages so the workers' regular eviction
//!    path has clean candidates before the pool is exhausted.
//! 3. Sleep briefly to bound writeback bandwidth; workers retain ownership of
//!    eviction and the associated parent-unswizzle protocol.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use pagebox_threading as threading;

use crate::buffer_pool::BufferPool;

const MIN_IDLE_WAIT: Duration = Duration::from_millis(1);
const MAX_IDLE_WAIT: Duration = Duration::from_millis(100);

fn target_free_frames(num_frames: usize) -> usize {
    ((num_frames / 10).max(16)).min(num_frames)
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
        if self.is_running() {
            return;
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.shutdown.store(false, Ordering::Release);
        let shutdown = self.shutdown.clone();
        let need_frames = self.need_frames.clone();
        let frames_available = self.frames_available.clone();
        self.thread = Some(
            threading::spawn_efficient("page-provider", move || {
                run(pool, &shutdown, &need_frames, &frames_available);
            })
            .expect("failed to spawn page provider"),
        );
    }

    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.need_frames.1.notify_one();
        if let Some(h) = self.thread.take() {
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
    shutdown: &AtomicBool,
    need_frames: &(Mutex<()>, Condvar),
    _frames_available: &(Mutex<()>, Condvar),
) {
    let mut idle_wait = MIN_IDLE_WAIT;
    let mut flush_epoch_active = false;
    while !shutdown.load(Ordering::Relaxed) {
        let Some(pool) = pool.upgrade() else {
            return;
        };
        let target_free = target_free_frames(pool.num_frames());
        let available = pool.approx_available_budget();
        if available >= target_free {
            // Budget sufficient — sleep until a worker signals need.
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, Duration::from_millis(10));
            idle_wait = MIN_IDLE_WAIT;
            flush_epoch_active = false;
            continue;
        }

        if !pool.has_dirty_resident_pages_for_provider() {
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, idle_wait);
            idle_wait = idle_wait.saturating_mul(2).min(MAX_IDLE_WAIT);
            flush_epoch_active = false;
            continue;
        }

        if !flush_epoch_active {
            pool.begin_dirty_flush_epoch_for_provider();
            flush_epoch_active = true;
        }

        let cleaned = pool
            .try_flush_dirty_batch_for_provider(64)
            .unwrap_or_else(|error| panic!("background page provider dirty flush failed: {error}"));
        if cleaned == 0 {
            // Cleaning does not replenish resident-budget tokens. Once the
            // worker eviction path has caught up, wait for a new pressure
            // signal instead of repeatedly rescanning an all-clean arena.
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, idle_wait);
            idle_wait = idle_wait.saturating_mul(2).min(MAX_IDLE_WAIT);
            flush_epoch_active = false;
        } else {
            idle_wait = MIN_IDLE_WAIT;
            std::thread::sleep(MIN_IDLE_WAIT);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::{PageProviderHandle, target_free_frames};
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
}
