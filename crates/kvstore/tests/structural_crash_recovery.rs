use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kvstore::{KvStore, KvStoreOptions, PAGE_SIZE};
use pagebox_wal::{Wal, WalReplayRecord};

const MAX_GROWTH_KEYS: u64 = 20_000;

#[derive(Clone)]
enum OwnedRecord {
    PageImage {
        lsn: u64,
        page_id: u64,
        data: Vec<u8>,
    },
    Logical {
        lsn: u64,
        kind: u64,
        payload: Vec<u8>,
    },
}

impl OwnedRecord {
    fn append_to(&self, wal: &Wal) {
        match self {
            Self::PageImage { lsn, page_id, data } => {
                let page: &[u8; PAGE_SIZE] = data
                    .as_slice()
                    .try_into()
                    .expect("captured page image must match the build page size");
                wal.append_page_image_with_lsn(*lsn, *page_id, |_, target| {
                    target.copy_from_slice(page);
                })
                .unwrap();
            }
            Self::Logical { lsn, kind, payload } => {
                wal.append_logical_with_lsn(*lsn, *kind, payload).unwrap()
            }
        }
    }
}

fn structural_key(index: u64) -> Vec<u8> {
    let mut key = vec![0x42; PAGE_SIZE / 16];
    let suffix = key.len() - 8;
    key[suffix..].copy_from_slice(&index.to_be_bytes());
    key
}

fn value(index: u64) -> Vec<u8> {
    index.to_be_bytes().to_vec()
}

fn model_entries(model: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<(Vec<u8>, Vec<u8>)> {
    model
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn scan(store: &KvStore) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut entries = Vec::new();
    store.scan_all(|key, value| entries.push((key.to_vec(), value.to_vec())));
    entries
}

fn discover_growth_boundaries() -> (u64, u64) {
    let dir = tempfile::TempDir::new().unwrap();
    let store = KvStore::open(dir.path()).unwrap();
    let mut leaf_root_split = None;

    for index in 0..MAX_GROWTH_KEYS {
        assert!(store.put(&structural_key(index), &value(index)));
        match store.height() {
            1 if leaf_root_split.is_none() => leaf_root_split = Some(index),
            2 => return (leaf_root_split.expect("height one boundary"), index),
            _ => {}
        }
    }
    panic!("test workload did not reach an inner-root split");
}

fn capture_records(path: &Path) -> Vec<OwnedRecord> {
    let wal = Wal::open(path).unwrap();
    let mut records = Vec::new();
    wal.replay_records(|record| match record {
        WalReplayRecord::PageImage { lsn, page_id, data } => {
            records.push(OwnedRecord::PageImage {
                lsn,
                page_id,
                data: data.to_vec(),
            });
        }
        WalReplayRecord::Logical { lsn, kind, payload } => {
            records.push(OwnedRecord::Logical {
                lsn,
                kind,
                payload: payload.to_vec(),
            });
        }
    })
    .unwrap();
    records
}

fn exercise_every_durable_prefix(
    baseline_data: &Path,
    records: &[OwnedRecord],
    stable_root: u64,
    before_height: u32,
    after_height: u32,
    before: &BTreeMap<Vec<u8>, Vec<u8>>,
    after: &BTreeMap<Vec<u8>, Vec<u8>>,
) {
    assert!(
        !records.is_empty(),
        "structural operation must emit WAL records"
    );
    let before = model_entries(before);
    let after = model_entries(after);

    for prefix_len in 0..=records.len() {
        let case = tempfile::TempDir::new().unwrap();
        std::fs::copy(baseline_data, case.path().join("kvstore.data")).unwrap();
        let wal = Wal::open(&case.path().join("kvstore.wal")).unwrap();
        for record in &records[..prefix_len] {
            record.append_to(&wal);
        }
        wal.flush();
        drop(wal);

        let store =
            KvStore::open_with(case.path(), &KvStoreOptions::default().pool_frames(8)).unwrap();
        if before_height >= 2 || after_height >= 2 {
            assert!(
                store.persisted_pages() > store.cache_capacity_pages(),
                "multi-level prefix must reopen with less cache than persistent data"
            );
        }
        assert_eq!(
            store.root_page_id(),
            stable_root,
            "prefix {prefix_len} moved the physical root"
        );
        let actual = scan(&store);
        assert!(
            actual == before || actual == after,
            "prefix {prefix_len} recovered a hybrid structural state"
        );
        assert!(
            store.height() == before_height || store.height() == after_height,
            "prefix {prefix_len} recovered an impossible effective height {}",
            store.height()
        );

        let continuation_key = structural_key(u64::MAX);
        assert!(
            store.put(&continuation_key, b"continued"),
            "prefix {prefix_len} must permit continued insertion"
        );
        assert_eq!(
            store.get(&continuation_key).as_deref(),
            Some(b"continued".as_slice()),
            "prefix {prefix_len} lost a continued insertion"
        );
        assert!(store.del(&continuation_key));
    }
}

struct StructuralCase {
    _dir: tempfile::TempDir,
    baseline_data: PathBuf,
    wal_path: PathBuf,
    stable_root: u64,
    before_height: u32,
    after_height: u32,
    before: BTreeMap<Vec<u8>, Vec<u8>>,
    after: BTreeMap<Vec<u8>, Vec<u8>>,
}

fn root_split_case(trigger: u64) -> StructuralCase {
    let dir = tempfile::TempDir::new().unwrap();
    let baseline_data = dir.path().join("baseline.data");
    let mut model = BTreeMap::new();
    let stable_root;
    let before_height;
    let after_height;
    {
        let store = KvStore::open(dir.path()).unwrap();
        stable_root = store.root_page_id();
        for index in 0..trigger {
            let key = structural_key(index);
            let value = value(index);
            assert!(store.put(&key, &value));
            model.insert(key, value);
        }
        before_height = store.height();
        store.checkpoint().unwrap();
        std::fs::copy(dir.path().join("kvstore.data"), &baseline_data).unwrap();

        let key = structural_key(trigger);
        let value = value(trigger);
        assert!(store.put(&key, &value));
        model.insert(key, value);
        after_height = store.height();
    }

    let after = model;
    let mut before = after.clone();
    before.remove(&structural_key(trigger));
    StructuralCase {
        wal_path: dir.path().join("kvstore.wal"),
        _dir: dir,
        baseline_data,
        stable_root,
        before_height,
        after_height,
        before,
        after,
    }
}

fn root_collapse_case(key_count: u64) -> StructuralCase {
    let discovery = tempfile::TempDir::new().unwrap();
    let store = KvStore::open(discovery.path()).unwrap();
    for index in 0..key_count {
        assert!(store.put(&structural_key(index), &value(index)));
    }
    let mut previous_height = store.height();
    let mut collapse_at = None;
    for index in 0..key_count {
        assert!(store.del(&structural_key(index)));
        let height = store.height();
        if height < previous_height {
            collapse_at = Some((index, previous_height, height));
            break;
        }
        previous_height = height;
    }
    let (collapse_at, before_height, after_height) =
        collapse_at.expect("delete workload must collapse the root");
    drop(store);

    let dir = tempfile::TempDir::new().unwrap();
    let baseline_data = dir.path().join("baseline.data");
    let mut model = BTreeMap::new();
    let stable_root;
    {
        let store = KvStore::open(dir.path()).unwrap();
        stable_root = store.root_page_id();
        for index in 0..key_count {
            let key = structural_key(index);
            let value = value(index);
            assert!(store.put(&key, &value));
            model.insert(key, value);
        }
        for index in 0..collapse_at {
            let key = structural_key(index);
            assert!(store.del(&key));
            model.remove(&key);
        }
        assert_eq!(store.height(), before_height);
        store.checkpoint().unwrap();
        std::fs::copy(dir.path().join("kvstore.data"), &baseline_data).unwrap();

        let key = structural_key(collapse_at);
        assert!(store.del(&key));
        model.remove(&key);
        assert_eq!(store.height(), after_height);
    }

    let after = model;
    let mut before = after.clone();
    before.insert(structural_key(collapse_at), value(collapse_at));
    StructuralCase {
        wal_path: dir.path().join("kvstore.wal"),
        _dir: dir,
        baseline_data,
        stable_root,
        before_height,
        after_height,
        before,
        after,
    }
}

fn run_case(case: StructuralCase) {
    let records = capture_records(&case.wal_path);
    let minimum_records = if case.after_height > case.before_height {
        3
    } else {
        2
    };
    assert!(
        records.len() >= minimum_records,
        "structural boundary must emit at least {minimum_records} records, got {}",
        records.len()
    );
    exercise_every_durable_prefix(
        &case.baseline_data,
        &records,
        case.stable_root,
        case.before_height,
        case.after_height,
        &case.before,
        &case.after,
    );
}

#[test]
fn every_leaf_and_inner_root_split_prefix_reopens_to_a_complete_state() {
    let (leaf_split_at, inner_split_at) = discover_growth_boundaries();
    let leaf_case = root_split_case(leaf_split_at);
    assert_eq!((leaf_case.before_height, leaf_case.after_height), (0, 1));
    run_case(leaf_case);

    let inner_case = root_split_case(inner_split_at);
    assert_eq!((inner_case.before_height, inner_case.after_height), (1, 2));
    run_case(inner_case);
}

#[test]
fn every_root_collapse_prefix_reopens_to_a_complete_state() {
    let (_, inner_split_at) = discover_growth_boundaries();
    let case = root_collapse_case(inner_split_at + 32);
    assert!(case.after_height < case.before_height);
    run_case(case);
}
