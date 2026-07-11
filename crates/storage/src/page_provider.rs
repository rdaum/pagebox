//! Background page provider thread — proactively evicts clean pages and
//! flushes dirty pages so they become evictable.
//!
//! The provider runs a continuous loop:
//! 1. If budget is sufficient, sleep until woken by a worker.
//! 2. Try to evict clean pages (non-blocking `try_evict_policy`).
//! 3. If eviction found only dirty pages (no-steal), flush them via
//!    `try_flush_dirty_batch` so the next pass can evict them.
//! 4. Notify waiting workers when frames become available.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use pagebox_threading as threading;

use crate::buffer_pool::BufferPool;

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
        self.thread.is_some()
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
    frames_available: &(Mutex<()>, Condvar),
) {
    while !shutdown.load(Ordering::Relaxed) {
        let Some(pool) = pool.upgrade() else {
            return;
        };
        let target_free = (pool.num_frames() / 10).max(16);
        let available = pool.approx_available_budget();
        if available >= target_free {
            // Budget sufficient — sleep until a worker signals need.
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, Duration::from_millis(10));
            continue;
        }

        let want = (target_free - available).min(64);
        let evicted = pool.try_evict_any_policy_for_provider(want);

        if evicted > 0 {
            frames_available.1.notify_all();
        } else {
            // Eviction couldn't find clean victims — likely all resident
            // pages are dirty (no-steal). Flush dirty pages so they become
            // evictable on the next pass.
            pool.try_flush_dirty_batch_for_provider(64)
                .unwrap_or_else(|error| {
                    panic!("background page provider dirty flush failed: {error}")
                });
            std::thread::yield_now();
        }
    }
}
