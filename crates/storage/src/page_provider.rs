//! Background page provider thread — replenishes resident-budget
//! availability by proactively evicting unpinned pages.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use pagebox_threading as threading;

use crate::buffer_frame::PageClass;
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
            let guard = need_frames.0.lock().unwrap();
            let _ = need_frames.1.wait_timeout(guard, Duration::from_millis(10));
            continue;
        }

        let want = (target_free - available).min(64);
        let evicted = PageClass::ALL
            .iter()
            .map(|&class| pool.try_evict_policy(class, want))
            .sum::<usize>();

        if evicted > 0 {
            frames_available.1.notify_all();
        } else {
            std::thread::yield_now();
        }
    }
}
