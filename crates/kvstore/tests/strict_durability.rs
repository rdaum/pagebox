use std::path::Path;
use std::process::Command;

use kvstore::{KvStore, KvStoreOptions, SyncMode};

const CHILD_MODE: &str = "PAGEBOX_STRICT_DURABILITY_CHILD";
const STORE_PATH: &str = "PAGEBOX_STRICT_DURABILITY_STORE";
const KEY: &[u8] = b"strict-durability-key";
const VALUE: &[u8] = b"strict-durability-value";

fn strict_store(path: &Path) -> KvStore {
    KvStore::open_with(path, &KvStoreOptions::default().sync_mode(SyncMode::Strict)).unwrap()
}

fn run_abort_child(test_name: &str, mode: &str, store_path: &Path) {
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--test-threads=1"])
        .env(CHILD_MODE, mode)
        .env(STORE_PATH, store_path)
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "durability child must abort without running destructors"
    );
}

#[test]
fn strict_put_survives_abort_after_return() {
    let dir = tempfile::TempDir::new().unwrap();

    run_abort_child("strict_put_abort_child", "put", dir.path());

    let reopened = KvStore::open(dir.path()).unwrap();
    assert_eq!(
        reopened.get(KEY).as_deref(),
        Some(VALUE),
        "strict put must be recoverable immediately after the method returns"
    );
}

#[test]
fn strict_delete_survives_abort_after_return() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = KvStore::open(dir.path()).unwrap();
    assert!(store.put(KEY, VALUE));
    store.checkpoint().unwrap();
    drop(store);

    run_abort_child("strict_delete_abort_child", "delete", dir.path());

    let reopened = KvStore::open(dir.path()).unwrap();
    assert_eq!(
        reopened.get(KEY),
        None,
        "strict delete must be recoverable immediately after the method returns"
    );
}

#[test]
fn strict_put_abort_child() {
    if std::env::var(CHILD_MODE).as_deref() != Ok("put") {
        return;
    }
    let path = std::env::var_os(STORE_PATH).expect("child store path");
    let store = strict_store(Path::new(&path));
    assert!(store.put(KEY, VALUE));
    std::process::abort();
}

#[test]
fn strict_delete_abort_child() {
    if std::env::var(CHILD_MODE).as_deref() != Ok("delete") {
        return;
    }
    let path = std::env::var_os(STORE_PATH).expect("child store path");
    let store = strict_store(Path::new(&path));
    assert!(store.del(KEY));
    std::process::abort();
}
