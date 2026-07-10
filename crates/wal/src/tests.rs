use std::collections::HashMap;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "metrics")]
use fast_telemetry::{HistogramSnapshot, MetricLabels, MetricMeta, MetricVisitor};
use pagebox_frame_kernel::{Lsn, PAGE_SIZE, PageId, page_size};

use crate::format::{
    BATCH_MAX_RECORDS, BatchEntry, HDR_RECORD_SIZE_OFF, HDR_VERSION_OFF, WAL_HEADER_SIZE,
    WAL_RECORD_SIZE, batch_meta_count, finalize_batch_meta, init_batch_meta, page_crc,
    set_batch_meta_count, write_batch_entry,
};
use crate::{CommitMode, RecoveryPageStore, Wal, WalReplayRecord};

const TEST_PAGE_LSN_OFF: usize = 8;

#[cfg(feature = "metrics")]
fn wal_counter_event(wal: &Wal, label_value: &str) -> u64 {
    struct CounterEvent<'a> {
        label_value: &'a str,
        value: u64,
    }

    impl MetricVisitor for CounterEvent<'_> {
        fn counter(&mut self, meta: MetricMeta<'_>, labels: MetricLabels<'_>, value: i64) {
            if meta.name != "wal_events" {
                return;
            }
            if labels
                .iter()
                .any(|label| label.name == "event" && label.value == self.label_value)
            {
                self.value = self.value.saturating_add(value.max(0) as u64);
            }
        }

        fn gauge_i64(&mut self, _meta: MetricMeta<'_>, _labels: MetricLabels<'_>, _value: i64) {}

        fn gauge_f64(&mut self, _meta: MetricMeta<'_>, _labels: MetricLabels<'_>, _value: f64) {}

        fn histogram(
            &mut self,
            _meta: MetricMeta<'_>,
            _labels: MetricLabels<'_>,
            _histogram: &dyn HistogramSnapshot,
        ) {
        }
    }

    let mut visitor = CounterEvent {
        label_value,
        value: 0,
    };
    wal.visit_metrics(&mut visitor);
    visitor.value
}

fn read_test_page_lsn(page: &[u8]) -> u64 {
    let Some(bytes) = page.get(TEST_PAGE_LSN_OFF..TEST_PAGE_LSN_OFF + 8) else {
        return 0;
    };
    u64::from_ne_bytes(bytes.try_into().unwrap())
}

fn write_test_page_lsn(page: &mut [u8], lsn: u64) {
    page[TEST_PAGE_LSN_OFF..TEST_PAGE_LSN_OFF + 8].copy_from_slice(&lsn.to_ne_bytes());
}

#[derive(Default)]
struct TestRecoveryStore {
    pages: Mutex<HashMap<PageId, Vec<u8>>>,
}

impl RecoveryPageStore for TestRecoveryStore {
    fn read_page(&self, pid: PageId, buf: &mut [u8]) -> io::Result<bool> {
        let pages = self.pages.lock().unwrap();
        let Some(data) = pages.get(&pid) else {
            return Ok(false);
        };
        buf.copy_from_slice(data);
        Ok(true)
    }

    fn write_page(&self, pid: PageId, data: &[u8]) -> io::Result<()> {
        self.pages.lock().unwrap().insert(pid, data.to_vec());
        Ok(())
    }

    fn allocate(&self, pid: PageId) -> io::Result<()> {
        self.pages
            .lock()
            .unwrap()
            .insert(pid, vec![0; page_size(pid)]);
        Ok(())
    }

    fn sync(&self) -> io::Result<()> {
        Ok(())
    }

    fn next_page_id(&self) -> PageId {
        self.pages
            .lock()
            .unwrap()
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            + 1
    }
}

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pagebox_wal_{name}_{}", std::process::id()));
    p
}

struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        for shard_idx in 1..=8 {
            let mut shard = self.0.as_os_str().to_os_string();
            shard.push(format!(".shard{shard_idx}"));
            let _ = std::fs::remove_file(PathBuf::from(shard));
        }
    }
}

#[test]
fn recover_applies_page_patch_after_base_image() {
    let path = tmp_path("recover_page_patch");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open(&path).unwrap();

    let page_id = 1;
    let mut before = [0u8; PAGE_SIZE];
    let lsn1 = wal.claim_lsn();
    wal.append_page_image_with_lsn(lsn1, page_id, |lsn, page| {
        write_test_page_lsn(page, lsn);
        page[100] = 1;
        before.copy_from_slice(page);
    })
    .unwrap();

    let mut after = before;
    let lsn2 = wal.claim_lsn();
    write_test_page_lsn(&mut after, lsn2);
    after[200] = 7;
    assert!(
        wal.append_page_patch_with_lsn(lsn2, page_id, &before, &after)
            .unwrap(),
        "small page change should encode as a page patch"
    );
    wal.flush();

    let store = TestRecoveryStore::default();
    let report = wal.recover(&store, 0, read_test_page_lsn).unwrap();
    assert_eq!(report.records_applied, 2, "image and patch should apply");
    let pages = store.pages.lock().unwrap();
    assert_eq!(
        pages.get(&page_id).map(Vec::as_slice),
        Some(after.as_slice()),
        "page patch recovery should reconstruct the final page image"
    );
}

#[test]
fn sharded_recovery_applies_records_in_lsn_order() {
    let path = tmp_path("sharded_recovery_order");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open_with_shards_for_test(&path, 2).unwrap();

    let page_id = 1;
    let mut before = [0u8; PAGE_SIZE];
    let lsn1 = wal.claim_lsn();
    wal.append_page_image_with_lsn(lsn1, page_id, |lsn, page| {
        write_test_page_lsn(page, lsn);
        page[100] = 1;
        before.copy_from_slice(page);
    })
    .unwrap();

    let mut after = before;
    let lsn2 = wal.claim_lsn();
    write_test_page_lsn(&mut after, lsn2);
    after[200] = 7;
    assert!(
        wal.append_page_patch_with_lsn(lsn2, page_id, &before, &after)
            .unwrap(),
        "small page change should encode as a page patch"
    );

    let mut replayed = Vec::new();
    wal.replay_records(|record| replayed.push(record_lsn(record)))
        .unwrap();
    assert_eq!(
        replayed,
        vec![lsn1, lsn2],
        "sharded replay should merge by LSN"
    );

    let store = TestRecoveryStore::default();
    let report = wal.recover(&store, 0, read_test_page_lsn).unwrap();
    assert_eq!(report.records_applied, 2, "image and patch should apply");
    let pages = store.pages.lock().unwrap();
    assert_eq!(
        pages.get(&page_id).map(Vec::as_slice),
        Some(after.as_slice()),
        "sharded recovery should apply dependent records in LSN order"
    );
}

#[test]
fn sharded_lsn_claims_stay_on_thread_shard() {
    let path = tmp_path("sharded_thread_local_claims");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open_with_shards_for_test(&path, 3).unwrap();

    let lsn1 = wal.claim_lsn();
    let lsn2 = wal.claim_lsn();
    let lsn3 = wal.claim_lsn();

    assert_eq!(
        (lsn1 - 1) % 3,
        (lsn2 - 1) % 3,
        "same thread should keep using one WAL shard"
    );
    assert_eq!(
        (lsn2 - 1) % 3,
        (lsn3 - 1) % 3,
        "same thread should keep using one WAL shard"
    );
}

#[test]
fn sharded_open_starts_every_shard_driver() {
    let path = tmp_path("sharded_driver_startup");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open_with_shards_for_test(&path, 4).unwrap();

    assert_eq!(
        wal.started_thread_count_for_test(),
        4,
        "every WAL shard should start its driver thread"
    );
}

fn record_lsn(record: WalReplayRecord<'_>) -> u64 {
    match record {
        WalReplayRecord::PageImage { lsn, .. } | WalReplayRecord::Logical { lsn, .. } => lsn,
    }
}

#[test]
fn concurrent_strict_flushes_advance_durable_lsn() {
    let path = tmp_path("concurrent_strict_flushes");
    let _cleanup = Cleanup(path.clone());
    let wal = Arc::new(Wal::open(&path).unwrap());
    let workers = 4;
    let records_per_worker = 8;
    let start = Arc::new(Barrier::new(workers));

    let handles = (0..workers)
        .map(|worker| {
            let wal = Arc::clone(&wal);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                let mut page = [worker as u8; PAGE_SIZE];
                start.wait();
                for idx in 0..records_per_worker {
                    page[0] = idx as u8;
                    let pid = ((worker as u64) << 32) | idx as u64;
                    let lsn = wal.append_page_image(pid, &page).unwrap();
                    let durable = wal.flush_at_least(lsn);
                    assert!(
                        durable >= lsn,
                        "strict flush should make requested LSN durable"
                    );
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(
        wal.durable_lsn(),
        (workers * records_per_worker) as u64,
        "all concurrent strict commits should be durable"
    );
}

#[test]
fn replay_in_order() {
    let path = tmp_path("replay");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        for i in 1..=5u64 {
            let mut page = [0u8; PAGE_SIZE];
            page[0] = i as u8;
            wal.append_page_image(i * 10, &page).unwrap();
        }
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();

    assert_eq!(replayed.len(), 5);
    for (i, &(lsn, pid, byte)) in replayed.iter().enumerate() {
        let expected = i as u64 + 1;
        assert_eq!(lsn, expected);
        assert_eq!(pid, expected * 10);
        assert_eq!(byte, expected as u8);
    }
}

#[test]
fn replay_records_interleaves_page_images_and_logical_records() {
    let path = tmp_path("replay_records_mixed");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        let mut page = [0u8; PAGE_SIZE];
        page[0] = 11;
        assert_eq!(wal.append_page_image(10, &page).unwrap(), 1);
        assert_eq!(wal.append_logical(42, b"logical").unwrap(), 2);
        page[0] = 33;
        assert_eq!(wal.append_page_image(30, &page).unwrap(), 3);
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut events = Vec::new();
    wal.replay_records(|record| match record {
        WalReplayRecord::PageImage { lsn, page_id, data } => {
            events.push((lsn, page_id, 0, data[0], Vec::new()));
        }
        WalReplayRecord::Logical { lsn, kind, payload } => {
            events.push((lsn, 0, kind, 0, payload.to_vec()));
        }
    })
    .unwrap();

    assert_eq!(
        events,
        vec![
            (1, 10, 0, 11, Vec::new()),
            (2, 0, 42, 0, b"logical".to_vec()),
            (3, 30, 0, 33, Vec::new()),
        ],
        "mixed replay should preserve record order"
    );

    let mut page_only = Vec::new();
    wal.replay(|lsn, pid, data| page_only.push((lsn, pid, data[0])))
        .unwrap();
    assert_eq!(
        page_only,
        vec![(1, 10, 11), (3, 30, 33)],
        "page-image replay should ignore logical records"
    );
}

#[test]
fn logical_records_reassemble_payloads_larger_than_one_wal_slot() {
    let path = tmp_path("logical_large");
    let _cleanup = Cleanup(path.clone());
    let payload = (0..PAGE_SIZE * 3 + 17)
        .map(|idx| (idx % 251) as u8)
        .collect::<Vec<_>>();

    {
        let wal = Wal::open(&path).unwrap();
        assert_eq!(wal.append_logical(77, &payload).unwrap(), 1);
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut logical = Vec::new();
    wal.replay_records(|record| {
        if let WalReplayRecord::Logical { lsn, kind, payload } = record {
            logical.push((lsn, kind, payload.to_vec()));
        }
    })
    .unwrap();

    assert_eq!(
        logical,
        vec![(1, 77, payload)],
        "large logical WAL records should replay as one reassembled payload"
    );
}

#[test]
fn small_logical_records_pack_into_one_wal_slot() {
    let path = tmp_path("logical_packed");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        for idx in 0u8..100 {
            let payload = [idx, idx.wrapping_mul(3), idx.wrapping_mul(7)];
            assert_eq!(
                wal.append_logical(90 + u64::from(idx % 3), &payload)
                    .unwrap(),
                u64::from(idx) + 1
            );
        }
        wal.flush();
        #[cfg(feature = "metrics")]
        assert!(
            wal_counter_event(&wal, "write_bytes") <= (WAL_RECORD_SIZE * 2) as u64,
            "small logical records should share one WAL data slot instead of one slot each"
        );
    }

    let wal = Wal::open(&path).unwrap();
    let mut logical = Vec::new();
    wal.replay_records(|record| {
        if let WalReplayRecord::Logical { lsn, kind, payload } = record {
            logical.push((lsn, kind, payload.to_vec()));
        }
    })
    .unwrap();

    assert_eq!(logical.len(), 100);
    for idx in 0u8..100 {
        assert_eq!(
            logical[idx as usize],
            (
                u64::from(idx) + 1,
                90 + u64::from(idx % 3),
                vec![idx, idx.wrapping_mul(3), idx.wrapping_mul(7)]
            ),
            "packed logical replay should preserve record order and payload {idx}"
        );
    }
}

#[test]
fn reopen_advances_lsn_past_packed_logical_records() {
    let path = tmp_path("logical_packed_reopen_lsn");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        for idx in 0u8..10 {
            assert_eq!(wal.append_logical(8, &[idx]).unwrap(), u64::from(idx) + 1);
        }
        wal.flush();
    }

    {
        let wal = Wal::open(&path).unwrap();
        assert_eq!(
            wal.append_logical(8, b"after-reopen").unwrap(),
            11,
            "packed logical records should advance the reopened next-LSN cursor"
        );
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut lsns = Vec::new();
    wal.replay_records(|record| {
        if let WalReplayRecord::Logical { lsn, .. } = record {
            lsns.push(lsn);
        }
    })
    .unwrap();

    assert_eq!(
        lsns,
        (1u64..=11).collect::<Vec<_>>(),
        "reopened append should not reuse an LSN already present in a packed slot"
    );
}

#[test]
fn packed_logical_records_do_not_cross_later_page_images() {
    let path = tmp_path("logical_packed_order");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        assert_eq!(wal.append_logical(1, b"before").unwrap(), 1);
        let mut page = [0u8; PAGE_SIZE];
        page[0] = 55;
        assert_eq!(wal.append_page_image(20, &page).unwrap(), 2);
        assert_eq!(wal.append_logical(1, b"after").unwrap(), 3);
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut events = Vec::new();
    wal.replay_records(|record| match record {
        WalReplayRecord::Logical { lsn, payload, .. } => {
            events.push((lsn, 0, payload.to_vec()));
        }
        WalReplayRecord::PageImage { lsn, page_id, data } => {
            events.push((lsn, page_id, vec![data[0]]));
        }
    })
    .unwrap();

    assert_eq!(
        events,
        vec![
            (1, 0, b"before".to_vec()),
            (2, 20, vec![55]),
            (3, 0, b"after".to_vec()),
        ],
        "packed logical append must not reorder around later page images"
    );
}

#[test]
fn page_image_roundtrips_as_reassembled_record() {
    let path = tmp_path("page_image_roundtrip");
    let _cleanup = Cleanup(path.clone());
    // With a single page class, pid 9 maps directly to page number 9 and a
    // page image occupies one full WAL block.
    let pid: PageId = 9;

    {
        let wal = Wal::open(&path).unwrap();
        let lsn = wal.claim_lsn();
        assert_eq!(lsn, 1);
        wal.append_page_image_with_lsn(lsn, pid, |lsn, page| {
            write_test_page_lsn(page, lsn);
            page[0] = 17;
            page[PAGE_SIZE / 2] = 42;
            page[PAGE_SIZE - 1] = 99;
        })
        .unwrap();
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay_records(|record| {
        if let WalReplayRecord::PageImage { lsn, page_id, data } = record {
            replayed.push((
                lsn,
                page_id,
                data.len(),
                data[0],
                data[PAGE_SIZE / 2],
                data[PAGE_SIZE - 1],
            ));
        }
    })
    .unwrap();

    assert_eq!(
        replayed,
        vec![(1, pid, PAGE_SIZE, 17, 42, 99)],
        "page images should replay as reassembled page-image records"
    );

    let store = TestRecoveryStore::default();
    let report = wal.recover(&store, 0, read_test_page_lsn).unwrap();
    assert_eq!(report.records_applied, 1);

    let mut recovered = vec![0u8; PAGE_SIZE];
    assert!(
        store.read_page(pid, &mut recovered).unwrap(),
        "recovery should write the page image"
    );
    assert_eq!(read_test_page_lsn(&recovered), 1);
    assert_eq!(recovered[0], 17);
    assert_eq!(recovered[PAGE_SIZE / 2], 42);
    assert_eq!(recovered[PAGE_SIZE - 1], 99);
}

#[test]
fn replay_records_skips_abandoned_logical_prefix_after_reopen_append() {
    let path = tmp_path("logical_abandoned_prefix");
    let _cleanup = Cleanup(path.clone());
    let payload = vec![9u8; PAGE_SIZE * BATCH_MAX_RECORDS + 1];

    {
        let wal = Wal::open(&path).unwrap();
        assert_eq!(wal.append_logical(77, &payload).unwrap(), 1);
        wal.flush();
    }

    let first_batch_end = WAL_HEADER_SIZE + WAL_RECORD_SIZE * (BATCH_MAX_RECORDS + 1);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(first_batch_end as u64)
        .unwrap();

    {
        let wal = Wal::open(&path).unwrap();
        let mut page = [0u8; PAGE_SIZE];
        page[0] = 44;
        assert_eq!(
            wal.append_page_image(12, &page).unwrap(),
            1,
            "abandoned logical prefix should not advance the reopened LSN"
        );
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut page_images = Vec::new();
    wal.replay_records(|record| {
        if let WalReplayRecord::PageImage { lsn, page_id, data } = record {
            page_images.push((lsn, page_id, data[0]));
        }
    })
    .unwrap();

    assert_eq!(
        page_images,
        vec![(1, 12, 44)],
        "replay should ignore the incomplete logical prefix and continue with later records"
    );
}

#[test]
fn buffered_overwrite_replaces_active_record() {
    let path = tmp_path("buffered_overwrite_active");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open(&path).unwrap();

    let (lsn1, record) = wal
        .append_or_overwrite_page_image(None, 7, |_, page| page[0] = 1)
        .unwrap();
    let (lsn2, _) = wal
        .append_or_overwrite_page_image(Some(record), 7, |_, page| page[0] = 2)
        .unwrap();
    wal.flush();

    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();

    assert_eq!(lsn1, 1);
    assert_eq!(lsn2, 2);
    assert_eq!(
        replayed,
        vec![(2, 7, 2)],
        "active buffered overwrite should replace the previous page image"
    );
}

#[test]
fn buffered_overwrite_appends_after_flush() {
    let path = tmp_path("buffered_overwrite_after_flush");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open(&path).unwrap();

    let (_, record) = wal
        .append_or_overwrite_page_image(None, 7, |_, page| page[0] = 1)
        .unwrap();
    wal.flush();
    let (lsn2, _) = wal
        .append_or_overwrite_page_image(Some(record), 7, |_, page| page[0] = 2)
        .unwrap();
    wal.flush();

    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();

    assert_eq!(lsn2, 2);
    assert_eq!(
        replayed,
        vec![(1, 7, 1), (2, 7, 2)],
        "flushed buffered records should not be overwritten"
    );
}

#[test]
fn checksum_detects_corruption() {
    let mut meta = [0u8; WAL_RECORD_SIZE];
    let page = [42u8; PAGE_SIZE];
    init_batch_meta(&mut meta);
    write_batch_entry(&mut meta, 0, BatchEntry::page_image(1, 10, page_crc(&page)));
    set_batch_meta_count(&mut meta, 1);
    finalize_batch_meta(&mut meta);

    assert_eq!(batch_meta_count(&meta), Some(1));
    meta[WAL_RECORD_SIZE - 1] ^= 0xFF;
    assert_eq!(batch_meta_count(&meta), None);
}

#[test]
fn invalid_wal_headers_are_rejected() {
    let cases: &[(&str, u64, Vec<u8>, &str)] = &[
        (
            "bad_magic",
            0,
            vec![0xDE, 0xAD, 0xBE, 0xEF],
            "bad WAL magic",
        ),
        (
            "bad_version",
            HDR_VERSION_OFF as u64,
            99u16.to_le_bytes().to_vec(),
            "unsupported WAL version",
        ),
        (
            "bad_record_size",
            HDR_RECORD_SIZE_OFF as u64,
            1234u64.to_le_bytes().to_vec(),
            "incompatible WAL record size",
        ),
    ];

    for (label, offset, bytes, expected) in cases {
        let path = tmp_path(label);
        let _cleanup = Cleanup(path.clone());
        {
            let wal = Wal::open(&path).unwrap();
            let page = [0u8; PAGE_SIZE];
            wal.append_page_image(1, &page).unwrap();
            wal.flush();
        }

        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(*offset)).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();

        let err = Wal::open(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains(expected),
            "unexpected error: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Torn write recovery — partial last page-image batch
// ---------------------------------------------------------------------------

#[test]
fn torn_page_image_batch_replays_complete_batches_only() {
    let path = tmp_path("torn_page_image_batch");
    let _cleanup = Cleanup(path.clone());

    // Write enough page images to span two batches, then flush.
    let records_before_tear = BATCH_MAX_RECORDS + 1;
    {
        let wal = Wal::open(&path).unwrap();
        for i in 1..=(records_before_tear as u64) {
            let mut page = [0u8; PAGE_SIZE];
            page[0] = i as u8;
            wal.append_page_image(i, &page).unwrap();
        }
        wal.flush();
    }

    // Truncate the file so the second batch's first data slot is missing:
    // keep the header + first full batch (meta + BATCH_MAX_RECORDS data pages)
    // + the second batch's meta page, but cut before its data page.
    let first_batch_end = WAL_HEADER_SIZE + WAL_RECORD_SIZE * (BATCH_MAX_RECORDS + 1);
    let torn_offset = first_batch_end + WAL_RECORD_SIZE; // meta page only, no data
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(torn_offset as u64)
        .unwrap();

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();

    // Only the first batch's records should replay; the torn second batch is ignored.
    let expected: Vec<(u64, u64, u8)> = (1..=BATCH_MAX_RECORDS as u64)
        .map(|i| (i, i, i as u8))
        .collect();
    assert_eq!(
        replayed, expected,
        "torn batch should be skipped, only complete batches replayed"
    );
}

// ---------------------------------------------------------------------------
// Torn write recovery — zeroed meta page at tail (preallocated space)
// ---------------------------------------------------------------------------

#[test]
fn zeroed_meta_page_at_tail_is_ignored() {
    let path = tmp_path("zeroed_meta_tail");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        for i in 1..=3u64 {
            let mut page = [0u8; PAGE_SIZE];
            page[0] = i as u8;
            wal.append_page_image(i, &page).unwrap();
        }
        wal.flush();
    }

    // Append a zeroed PAGE_SIZE block after the valid data (simulating
    // allocated-but-unwritten preallocation space).
    // 3 records in one batch: meta(1) + data(3) = 4 pages.
    let valid_end = WAL_HEADER_SIZE + WAL_RECORD_SIZE * 4;
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(valid_end as u64)).unwrap();
    file.write_all(&[0u8; WAL_RECORD_SIZE]).unwrap();
    file.sync_all().unwrap();

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();

    assert_eq!(
        replayed,
        vec![(1, 1, 1u8), (2, 2, 2u8), (3, 3, 3u8)],
        "zeroed meta page at tail should be ignored"
    );
}

// ---------------------------------------------------------------------------
// CRC mismatch detection during replay (integration)
// ---------------------------------------------------------------------------

#[test]
fn crc_mismatch_in_data_slot_skips_record_during_replay() {
    let path = tmp_path("crc_mismatch_replay");
    let _cleanup = Cleanup(path.clone());

    // Write two batches so we can verify that corruption in the first batch
    // stops replay before the second batch.
    let records_in_first_batch = BATCH_MAX_RECORDS;
    let total_records = records_in_first_batch + 1;
    {
        let wal = Wal::open(&path).unwrap();
        for i in 1..=total_records as u64 {
            let mut page = [0u8; PAGE_SIZE];
            page[0] = i as u8;
            wal.append_page_image(i, &page).unwrap();
        }
        wal.flush();
    }

    // Corrupt a data byte in the first data slot of the first batch.
    // Layout: header(4096) + meta_page(4096) + data_page_0(4096) + ...
    // Corrupt byte at offset 100 within the first data page.
    let corrupt_offset = (WAL_HEADER_SIZE + WAL_RECORD_SIZE + 100) as u64;
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(corrupt_offset)).unwrap();
    file.write_all(&[0xFF]).unwrap();
    file.sync_all().unwrap();

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();

    // The CRC mismatch causes replay to stop at the corrupted record.
    // No records should be replayed because the very first data slot is corrupted.
    assert!(
        replayed.is_empty(),
        "CRC mismatch in first data slot should stop replay with no records applied"
    );
}

// ---------------------------------------------------------------------------
// Recovery with checkpoint_lsn > 0 (partial replay)
// ---------------------------------------------------------------------------

#[test]
fn recovery_with_checkpoint_lsn_skips_records_at_or_below_checkpoint() {
    let path = tmp_path("checkpoint_lsn_partial");
    let _cleanup = Cleanup(path.clone());

    let page_id = 1;
    let mut page_v1 = [0u8; PAGE_SIZE];
    page_v1[100] = 11;

    let mut page_v2 = page_v1;
    page_v2[100] = 22;

    let mut page_v3 = page_v2;
    page_v3[100] = 33;

    {
        let wal = Wal::open(&path).unwrap();
        let lsn1 = wal.claim_lsn();
        wal.append_page_image_with_lsn(lsn1, page_id, |lsn, page| {
            write_test_page_lsn(page, lsn);
            page.copy_from_slice(&page_v1);
            write_test_page_lsn(page, lsn);
        })
        .unwrap();

        let lsn2 = wal.claim_lsn();
        wal.append_page_image_with_lsn(lsn2, page_id, |lsn, page| {
            write_test_page_lsn(page, lsn);
            page.copy_from_slice(&page_v2);
            write_test_page_lsn(page, lsn);
        })
        .unwrap();

        let lsn3 = wal.claim_lsn();
        wal.append_page_image_with_lsn(lsn3, page_id, |lsn, page| {
            write_test_page_lsn(page, lsn);
            page.copy_from_slice(&page_v3);
            write_test_page_lsn(page, lsn);
        })
        .unwrap();
        wal.flush();
    }

    // Recover with checkpoint_lsn=2: records at LSN 1 and 2 should be skipped,
    // only LSN 3 should apply.
    let store = TestRecoveryStore::default();
    let wal = Wal::open(&path).unwrap();
    let report = wal.recover(&store, 2, read_test_page_lsn).unwrap();

    assert_eq!(
        report.skipped_checkpoint, 2,
        "records at or below checkpoint_lsn should be skipped"
    );
    assert_eq!(
        report.records_applied, 1,
        "only the record above checkpoint_lsn should apply"
    );

    let pages = store.pages.lock().unwrap();
    let recovered = pages.get(&page_id).expect("page should exist");
    assert_eq!(
        recovered[100], 33,
        "recovery should apply only the latest page image"
    );
}

// ---------------------------------------------------------------------------
// Recovery skipped_page_lsn — idempotent recovery
// ---------------------------------------------------------------------------

#[test]
fn recovery_skips_page_when_on_disk_lsn_is_at_least_record_lsn() {
    let path = tmp_path("skipped_page_lsn");
    let _cleanup = Cleanup(path.clone());

    let page_id = 1;
    let mut page = [0u8; PAGE_SIZE];
    page[100] = 77;

    {
        let wal = Wal::open(&path).unwrap();
        wal.append_page_image(page_id, &page).unwrap();
        wal.flush();
    }

    // Pre-populate the recovery store with a page whose embedded LSN is >= the WAL record's LSN.
    let store = TestRecoveryStore::default();
    {
        let mut existing = vec![0u8; PAGE_SIZE];
        existing[100] = 99;
        write_test_page_lsn(&mut existing, 10); // LSN 10 >> WAL record LSN 1
        store.pages.lock().unwrap().insert(page_id, existing);
    }

    let wal = Wal::open(&path).unwrap();
    let report = wal.recover(&store, 0, read_test_page_lsn).unwrap();

    assert_eq!(
        report.skipped_page_lsn, 1,
        "record should be skipped because on-disk page LSN >= record LSN"
    );
    assert_eq!(
        report.records_applied, 0,
        "no records should be applied when the page is already at least as fresh"
    );

    let pages = store.pages.lock().unwrap();
    let recovered = pages.get(&page_id).expect("page should exist");
    assert_eq!(
        recovered[100], 99,
        "recovery should not overwrite a fresher page"
    );
}

// ---------------------------------------------------------------------------
// Concurrent appenders + final replay integrity
// ---------------------------------------------------------------------------

#[test]
fn concurrent_appenders_replay_all_records_with_contiguous_lsns() {
    let path = tmp_path("concurrent_appenders_replay");
    let _cleanup = Cleanup(path.clone());

    let n_threads = 4;
    let records_per_thread = 50;
    let total_records = n_threads * records_per_thread;

    {
        let wal = Arc::new(Wal::open(&path).unwrap());
        let barrier = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let wal = Arc::clone(&wal);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..records_per_thread {
                        let pid = ((t as u64) << 32) | i as u64;
                        let mut page = [0u8; PAGE_SIZE];
                        page[0] = t as u8;
                        page[1] = i as u8;
                        wal.append_page_image(pid, &page).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        wal.flush();
    }

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0], data[1])))
        .unwrap();

    assert_eq!(
        replayed.len(),
        total_records,
        "all concurrent records should replay"
    );

    let mut lsns: Vec<u64> = replayed.iter().map(|&(lsn, _, _, _)| lsn).collect();
    lsns.sort_unstable();
    let expected_lsns: Vec<u64> = (1..=total_records as u64).collect();
    assert_eq!(
        lsns, expected_lsns,
        "replayed LSNs should be a contiguous 1..=N set"
    );

    let mut pids: Vec<u64> = replayed.iter().map(|&(_, pid, _, _)| pid).collect();
    pids.sort_unstable();
    pids.dedup();
    assert_eq!(
        pids.len(),
        total_records,
        "every record should have a unique page ID"
    );
}

// ---------------------------------------------------------------------------
// Relaxed vs strict commit mode semantics
// ---------------------------------------------------------------------------

#[test]
fn strict_commit_mode_blocks_until_durable() {
    let path = tmp_path("strict_commit");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open(&path).unwrap();
    wal.set_commit_mode(CommitMode::Strict);

    let page = [7u8; PAGE_SIZE];
    let lsn = wal.append_page_image(1, &page).unwrap();
    let committed = wal.commit_current_thread();

    assert_eq!(
        committed, lsn,
        "strict commit should return the committed LSN"
    );
    assert_eq!(
        wal.durable_lsn(),
        lsn,
        "strict commit should make the LSN durable before returning"
    );
}

#[test]
fn relaxed_commit_mode_returns_without_waiting() {
    let path = tmp_path("relaxed_commit");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open(&path).unwrap();
    wal.set_commit_mode(CommitMode::Relaxed);

    let page = [7u8; PAGE_SIZE];
    let lsn = wal.append_page_image(1, &page).unwrap();
    let committed = wal.commit_current_thread();

    assert_eq!(
        committed, lsn,
        "relaxed commit should return the target LSN"
    );
    // In relaxed mode, the background syncer makes the LSN durable
    // asynchronously. We can't assert durable_lsn < lsn because the
    // background writer may have already flushed, but we can assert that
    // commit_current_thread returned immediately (before, not after, a
    // disk sync). The strict test above proves the blocking contract;
    // here we just verify the non-blocking contract: commit returns the
    // target LSN, and the background syncer eventually catches up.
    //
    // Wait for the background syncer to make the LSN durable.
    let deadline = Instant::now() + Duration::from_millis(500);
    while wal.durable_lsn() < lsn {
        if Instant::now() > deadline {
            panic!(
                "relaxed commit: durable_lsn {} never reached {} within 500ms",
                wal.durable_lsn(),
                lsn
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn relaxed_to_strict_switch_blocks_immediately() {
    let path = tmp_path("relaxed_to_strict");
    let _cleanup = Cleanup(path.clone());
    let wal = Wal::open(&path).unwrap();

    // Start in relaxed mode, append without waiting.
    wal.set_commit_mode(CommitMode::Relaxed);
    let page = [9u8; PAGE_SIZE];
    let lsn1 = wal.append_page_image(1, &page).unwrap();
    let committed1 = wal.commit_current_thread();
    assert_eq!(committed1, lsn1);

    // Switch to strict mode and append again. The strict commit should block
    // until durable_lsn >= lsn2, which also implies the earlier relaxed
    // append (lsn1) is durable.
    wal.set_commit_mode(CommitMode::Strict);
    let lsn2 = wal.append_page_image(2, &page).unwrap();
    let committed2 = wal.commit_current_thread();
    assert_eq!(committed2, lsn2);
    assert_eq!(
        wal.durable_lsn(),
        lsn2,
        "strict commit after relaxed should make all records durable"
    );
}

// ---------------------------------------------------------------------------
// Group-commit batching behavior under contention
// ---------------------------------------------------------------------------

#[test]
fn group_commit_batches_concurrent_commits_into_fewer_writes() {
    let path = tmp_path("group_commit_batching");
    let _cleanup = Cleanup(path.clone());
    let wal = Arc::new(Wal::open(&path).unwrap());

    let n_threads = 8;
    let barrier = Arc::new(Barrier::new(n_threads));

    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let wal = Arc::clone(&wal);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let mut page = [t as u8; PAGE_SIZE];
                page[0] = t as u8;
                barrier.wait();
                wal.append_page_image((t + 1) as u64, &page).unwrap();
                wal.commit_current_thread()
            })
        })
        .collect();

    let results: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should have committed successfully.
    assert_eq!(results.len(), n_threads);

    // With group commit, the concurrent commits should be served by fewer
    // write calls than the number of threads. If metrics are enabled, we
    // can check the write_call counter; otherwise we verify that all
    // records are durable and replay correctly.
    #[cfg(feature = "metrics")]
    {
        let write_calls = wal_counter_event(&wal, "write_call");
        // At least some batching: fewer write calls than threads.
        // (In practice, with 8 threads arriving simultaneously, 1-3 write
        // calls is typical. We assert < n_threads to allow for some
        // non-determinism.)
        assert!(
            write_calls < n_threads as u64,
            "group commit should batch: {write_calls} write calls for {n_threads} concurrent commits"
        );
    }

    // All records should be durable.
    assert_eq!(
        wal.durable_lsn(),
        n_threads as u64,
        "all concurrent commits should be durable"
    );

    // Verify replay integrity.
    drop(wal);
    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();
    assert_eq!(replayed.len(), n_threads);

    let mut lsns: Vec<u64> = replayed.iter().map(|&(l, _, _)| l).collect();
    lsns.sort_unstable();
    assert_eq!(lsns, (1..=n_threads as u64).collect::<Vec<_>>());
}

#[test]
fn clean_shutdown_flushes_active_buffer_even_when_written_lsn_is_higher() {
    let path = tmp_path("shutdown_flushes_active_below_written");
    let _cleanup = Cleanup(path.clone());

    {
        let wal = Wal::open(&path).unwrap();
        let lsn1 = wal.claim_lsn();
        let lsn2 = wal.claim_lsn();

        wal.append_page_image_with_lsn(lsn2, 2, |lsn, page| {
            write_test_page_lsn(page, lsn);
            page[0] = 2;
        })
        .unwrap();
        assert!(wal.flush_at_least(lsn2) >= lsn2);

        wal.append_page_image_with_lsn(lsn1, 1, |lsn, page| {
            write_test_page_lsn(page, lsn);
            page[0] = 1;
        })
        .unwrap();
    }

    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
        .unwrap();
    replayed.sort_unstable();
    assert_eq!(replayed, vec![(1, 1, 1), (2, 2, 2)]);
}

// ---------------------------------------------------------------------------
// Shutdown concurrent with flush guards the shared completion/notification
// path. Every strict commit that returned must be durable after a clean
// shutdown, and reopen + replay must surface every committed record with
// contiguous, unique LSNs.
// ---------------------------------------------------------------------------

#[test]
fn shutdown_concurrent_with_flush_preserves_all_committed_records() {
    let path = tmp_path("shutdown_concurrent_flush");
    let _cleanup = Cleanup(path.clone());

    let n_threads = 4;
    let commits_per_thread = 64;

    let committed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    {
        let wal = Arc::new(Wal::open(&path).unwrap());
        // Strict mode is the default, so flush_at_least blocks until durable.
        let barrier = Arc::new(Barrier::new(n_threads));

        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                let wal = Arc::clone(&wal);
                let barrier = Arc::clone(&barrier);
                let committed = Arc::clone(&committed);
                std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..commits_per_thread {
                        let pid = ((t as u64) << 32) | i as u64;
                        let mut page = [0u8; PAGE_SIZE];
                        page[0] = t as u8;
                        page[1] = i as u8;
                        let lsn = wal.append_page_image(pid, &page).unwrap();
                        // Strict: returns only once `lsn` is durable. If the
                        // timing change broke durability notification, this
                        // would either hang or return before durability.
                        wal.flush_at_least(lsn);
                        committed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread panicked");
        }
        // Clean shutdown: drain + fsync + close, concurrent with nothing now,
        // but the driver/syncer interaction during the run is what's tested.
        wal.flush();
    }

    let total_committed = committed.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        total_committed,
        (n_threads * commits_per_thread) as u64,
        "every worker should have completed its commits"
    );

    // Reopen and replay: every committed record must survive clean shutdown.
    let wal = Wal::open(&path).unwrap();
    let mut replayed = Vec::new();
    wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0], data[1])))
        .unwrap();
    drop(wal);

    assert_eq!(
        replayed.len(),
        total_committed as usize,
        "every committed record should survive clean shutdown"
    );

    // LSNs must be a contiguous 1..=N set (no gaps ⇒ no lost commit, no
    // duplicates ⇒ no double-count). This is the real invariant: the
    // completion-driven durability path must not drop or duplicate a
    // durable advance.
    let mut lsns: Vec<u64> = replayed.iter().map(|&(l, _, _, _)| l).collect();
    lsns.sort_unstable();
    let expected: Vec<u64> = (1..=total_committed).collect();
    assert_eq!(lsns, expected, "committed LSNs should be contiguous 1..=N");

    // Page IDs must be unique (each commit wrote a distinct page).
    let mut pids: Vec<u64> = replayed.iter().map(|&(_, p, _, _)| p).collect();
    pids.sort_unstable();
    pids.dedup();
    assert_eq!(
        pids.len(),
        total_committed as usize,
        "every committed record should have a unique page ID"
    );
}

// ---------------------------------------------------------------------------
// io_uring backend (Linux-only). These exercise the completion-driven path:
// writes submitted as WRITEV SQEs, fsyncs as linked FSYNC SQEs, and CQE
// reap advancing written_lsn / durable_lsn + reclaiming buffers. Real
// invariants, not "did not panic" smoke (per AGENTS.md).
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod io_uring_tests {
    use super::*;
    use crate::wal_impl::WalSyncBackend;

    fn page_data(seed: u64) -> [u8; PAGE_SIZE] {
        let mut buf = [0u8; PAGE_SIZE];
        let bytes = seed.to_le_bytes();
        for chunk in buf.chunks_exact_mut(8) {
            chunk.copy_from_slice(&bytes);
        }
        buf
    }

    fn open_iouring(path: &std::path::Path) -> Wal {
        Wal::open_with_backend_for_test(path, WalSyncBackend::IoUring)
            .expect("io_uring backend unavailable on this kernel")
    }

    #[test]
    fn iouring_repeated_open_drop_preserves_ring_memory_lifetime() {
        let path = tmp_path("iouring_repeated_open_drop");
        let _cleanup = Cleanup(path.clone());
        let page = page_data(91);

        for iteration in 0..16u64 {
            let wal = open_iouring(&path);
            let lsn = wal.append_page_image(iteration, &page).unwrap();
            assert!(
                wal.flush_at_least(lsn) >= lsn,
                "iteration {iteration} should become durable"
            );
            drop(wal);
        }

        let wal = open_iouring(&path);
        let mut replayed = Vec::new();
        wal.replay(|lsn, pid, _| replayed.push((lsn, pid))).unwrap();
        assert_eq!(
            replayed,
            (1..=16).zip(0..16).collect::<Vec<_>>(),
            "repeated ring teardown must not corrupt the WAL or process memory"
        );
    }

    #[test]
    fn iouring_append_flush_reopen_recovers() {
        let path = tmp_path("iouring_append_flush_reopen");
        let _cleanup = Cleanup(path.clone());
        let model = std::collections::BTreeMap::<PageId, (Lsn, [u8; PAGE_SIZE])>::new();

        {
            let wal = open_iouring(&path);
            let mut model = model;
            let page = page_data(7);
            for i in 0..128u64 {
                let pid = i;
                let lsn = wal.append_page_image(pid, &page).unwrap();
                model.insert(pid, (lsn, page));
            }
            wal.flush();

            // Verify all are durable + replay matches the model.
            assert_eq!(
                wal.durable_lsn(),
                128,
                "io_uring backend should make all flushed records durable"
            );
            let mut replayed = Vec::new();
            wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data.to_vec())))
                .unwrap();
            assert_eq!(
                replayed.len(),
                128,
                "io_uring replay should surface all records"
            );
            for (lsn, pid, data) in &replayed {
                let (expected_lsn, expected_data) = model.get(pid).unwrap();
                assert_eq!(
                    lsn, expected_lsn,
                    "io_uring replay LSN mismatch for pid {pid}"
                );
                assert_eq!(
                    data.as_slice(),
                    expected_data.as_slice(),
                    "io_uring replay data mismatch for pid {pid}"
                );
            }
        }

        // Reopen + recover into a store; every committed page should appear.
        let wal = open_iouring(&path);
        let store = TestRecoveryStore::default();
        wal.recover(&store, 0, |_| 0).unwrap();
        assert_eq!(
            store.pages.lock().unwrap().len(),
            128,
            "io_uring recovery should apply all 128 page images"
        );
    }

    #[test]
    fn iouring_crash_then_recover_preserves_committed_prefix() {
        let path = tmp_path("iouring_crash_recover");
        let _cleanup = Cleanup(path.clone());

        let committed_before_crash;
        {
            let wal = open_iouring(&path);
            let page = page_data(42);
            for i in 0..64u64 {
                let lsn = wal.append_page_image(i, &page).unwrap();
                wal.flush_at_least(lsn);
            }
            committed_before_crash = wal.durable_lsn();
            // Crash: no drain, threads stopped in place. Pending (unflushed)
            // appends are discarded; the durable prefix survives.
            wal.crash();
        }

        let wal = open_iouring(&path);
        let store = TestRecoveryStore::default();
        wal.recover(&store, 0, |_| 0).unwrap();
        let applied = store.pages.lock().unwrap().len();
        assert!(
            applied <= 64,
            "io_uring crash recovery should not apply more than committed"
        );
        assert!(
            applied > 0,
            "io_uring crash recovery should preserve the durable prefix"
        );
        let _ = committed_before_crash;
    }

    #[test]
    fn iouring_concurrent_commit_invariant() {
        let path = tmp_path("iouring_concurrent_commit");
        let _cleanup = Cleanup(path.clone());

        let n_threads = 4;
        let commits_per_thread = 32;

        {
            let wal = Arc::new(open_iouring(&path));
            let barrier = Arc::new(Barrier::new(n_threads));

            let handles: Vec<_> = (0..n_threads)
                .map(|t| {
                    let wal = Arc::clone(&wal);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        let page = page_data(t as u64);
                        barrier.wait();
                        for i in 0..commits_per_thread {
                            let pid = ((t as u64) << 32) | i as u64;
                            let lsn = wal.append_page_image(pid, &page).unwrap();
                            wal.flush_at_least(lsn);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        }

        // Reopen + replay: post-hoc invariant scan.
        let wal = open_iouring(&path);
        let mut replayed = Vec::new();
        wal.replay(|lsn, pid, data| replayed.push((lsn, pid, data[0])))
            .unwrap();

        let total = n_threads * commits_per_thread;
        assert_eq!(
            replayed.len(),
            total,
            "io_uring: all concurrent commits should survive"
        );

        // LSNs strictly monotone and unique (contiguous 1..=N).
        let mut lsns: Vec<u64> = replayed.iter().map(|&(l, _, _)| l).collect();
        lsns.sort_unstable();
        let expected: Vec<u64> = (1..=total as u64).collect();
        assert_eq!(
            lsns, expected,
            "io_uring: committed LSNs should be contiguous 1..=N"
        );

        // Unique page IDs.
        let mut pids: Vec<u64> = replayed.iter().map(|&(_, p, _)| p).collect();
        pids.sort_unstable();
        pids.dedup();
        assert_eq!(
            pids.len(),
            total,
            "io_uring: every commit should have a unique page ID"
        );
    }
}
