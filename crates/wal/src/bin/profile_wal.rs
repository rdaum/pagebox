use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use pagebox_frame_kernel::PAGE_SIZE;
use pagebox_wal::{CommitMode, Wal};

fn page_data(seed: u64) -> [u8; PAGE_SIZE] {
    let mut buf = [0u8; PAGE_SIZE];
    let bytes = seed.to_le_bytes();
    for chunk in buf.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bytes);
    }
    buf
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find_map(|w| (w[0] == flag).then(|| w[1].parse::<T>().ok()).flatten())
        .unwrap_or(default)
}

fn parse_mode(args: &[String]) -> String {
    args.windows(2)
        .find_map(|w| (w[0] == "--mode").then(|| w[1].clone()))
        .unwrap_or_else(|| "append".to_string())
}

fn parse_path(args: &[String]) -> PathBuf {
    args.windows(2)
        .find_map(|w| (w[0] == "--path").then(|| PathBuf::from(&w[1])))
        .unwrap_or_else(|| {
            let mut path = std::env::temp_dir();
            path.push(format!("pagebox_profile_wal_{}", std::process::id()));
            path
        })
}

fn parse_commit_mode(args: &[String]) -> CommitMode {
    match args
        .windows(2)
        .find_map(|w| (w[0] == "--commit-mode").then_some(w[1].as_str()))
        .unwrap_or("strict")
    {
        "relaxed" => CommitMode::Relaxed,
        _ => CommitMode::Strict,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = parse_mode(&args);
    let is_commit = mode == "commit";
    let threads = parse_arg(&args, "--threads", 8usize);
    let duration_secs = parse_arg(&args, "--duration-secs", 20u64);
    let page_count = parse_arg(&args, "--page-count", (threads.max(1) * 256) as u64);
    let path = parse_path(&args);
    let commit_mode = parse_commit_mode(&args);

    let wal = Arc::new(Wal::open_opts(&path).expect("open WAL"));
    wal.set_commit_mode(commit_mode);
    let pages = Arc::new((0..page_count).map(page_data).collect::<Vec<_>>());
    let stop = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..threads)
        .map(|thread_idx| {
            let wal = Arc::clone(&wal);
            let pages = Arc::clone(&pages);
            let stop = Arc::clone(&stop);
            let total_ops = Arc::clone(&total_ops);
            std::thread::spawn(move || {
                let mut ops = 0u64;
                let base = (thread_idx as u64) << 32;
                while !stop.load(Ordering::Relaxed) {
                    let idx = (ops as usize) % pages.len();
                    let pid = base.wrapping_add(ops);
                    let lsn = wal.append_page_image(pid, &pages[idx]).expect("append");
                    if is_commit {
                        let _ = wal.flush_at_least(lsn);
                    }
                    ops = ops.wrapping_add(1);
                }
                total_ops.fetch_add(ops, Ordering::Relaxed);
            })
        })
        .collect();

    let start = Instant::now();
    std::thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("worker join");
    }
    if !is_commit {
        let _ = wal.flush();
    }
    let elapsed = start.elapsed();
    let ops = total_ops.load(Ordering::Relaxed);
    eprintln!(
        "mode={mode} commit_mode={:?} threads={threads} ops={} elapsed={:.3}s mops={:.3}",
        commit_mode,
        ops,
        elapsed.as_secs_f64(),
        ops as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
}
