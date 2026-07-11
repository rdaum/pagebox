#![cfg(feature = "page-4k")]

use std::process::Command;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use kvstore::{KvStore, KvStoreOptions, SyncMode};

const CHILD_ENV: &str = "PAGEBOX_PAGE4_EVICTION_CHILD";

#[test]
fn concurrent_growth_progresses_through_eviction() {
    if std::env::var_os(CHILD_ENV).is_some() {
        run_child_workload();
        return;
    }

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "concurrent_growth_progresses_through_eviction",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);

    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "eviction-pressure child failed: {status}");
            return;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("concurrent 4 KiB growth did not complete within 20 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn run_child_workload() {
    const KEYS: u64 = 8_192;
    const THREADS: usize = 8;
    const POOL_FRAMES: usize = 256;

    let dir = tempfile::TempDir::new().unwrap();
    let store = Arc::new(
        KvStore::open_with(
            dir.path(),
            &KvStoreOptions::default()
                .pool_frames(POOL_FRAMES)
                .sync_mode(SyncMode::Relaxed),
        )
        .unwrap(),
    );
    let round = Arc::new(Barrier::new(THREADS));
    let handles = (0..THREADS)
        .map(|worker| {
            let store = Arc::clone(&store);
            let round = Arc::clone(&round);
            std::thread::spawn(move || {
                let value = [0xa5; 512];
                for key in (worker as u64..KEYS).step_by(THREADS) {
                    assert!(store.put(&key.to_be_bytes(), &value));
                    round.wait();
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }

    assert!(
        store.cache_evictions() > 0,
        "workload must exceed the resident frame budget"
    );

    let mut count = 0_u64;
    store.scan_all(|_, _| count += 1);
    assert_eq!(count, KEYS, "concurrent growth lost inserted records");
}
