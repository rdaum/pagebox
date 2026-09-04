use std::collections::BTreeMap;

use kvstore::{KvStore, PAGE_SIZE};

const INITIAL_KEYS: u64 = 2_000;
const CONTINUED_KEYS: u64 = 1_000;

fn key(index: u64) -> [u8; 8] {
    index.to_be_bytes()
}

fn value(index: u64) -> Vec<u8> {
    let value_len = if PAGE_SIZE == 4 * 1024 { 256 } else { 2 * 1024 };
    let mut value = vec![(index % 251) as u8; value_len];
    value[..8].copy_from_slice(&index.to_be_bytes());
    value
}

fn assert_matches_model(store: &KvStore, model: &BTreeMap<Vec<u8>, Vec<u8>>) {
    let mut actual = Vec::new();
    store.scan_all(|key, value| actual.push((key.to_vec(), value.to_vec())));
    let expected: Vec<_> = model
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    assert_eq!(actual, expected, "reopened tree must match the model");
}

#[test]
fn uncheckpointed_root_split_reopens_for_continued_mutation() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut model = BTreeMap::new();

    let initial_root;
    let split_height;
    {
        let store = KvStore::open(dir.path()).unwrap();
        initial_root = store.root_page_id();
        for index in 0..INITIAL_KEYS {
            let key = key(index);
            let value = value(index);
            assert!(store.put(&key, &value));
            model.insert(key.to_vec(), value);
        }
        split_height = store.height();
        assert!(split_height > 0, "test data must split the root");
    }

    let store = KvStore::open(dir.path()).unwrap();
    assert_eq!(
        store.root_page_id(),
        initial_root,
        "a tree's physical root page must remain stable across splits"
    );
    assert_eq!(
        store.height(),
        split_height,
        "reopen must derive the recovered tree height from its root"
    );
    assert_matches_model(&store, &model);

    for index in INITIAL_KEYS..INITIAL_KEYS + CONTINUED_KEYS {
        let key = key(index);
        let value = value(index);
        assert!(store.put(&key, &value));
        model.insert(key.to_vec(), value);
    }
    assert_matches_model(&store, &model);
}
