use micromeasure::{BenchContext, Throughput, benchmark_main, black_box};
use pagebox_storage::buffer_frame::PAGE_SIZE;
use pagebox_storage::slotted_page::SlottedPage;

struct SlottedPageInsertCtx {
    page: [u8; PAGE_SIZE],
    insert_key: [u8; 8],
    insert_value: [u8; 24],
}

impl BenchContext for SlottedPageInsertCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("slotted-page insert bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(100_000)
    }
}

struct SlottedPageResetInsertCtx {
    baseline: [u8; PAGE_SIZE],
    working: [u8; PAGE_SIZE],
    insert_value: [u8; 24],
    key_base: u64,
    key_stride: u64,
    insert_mode: InsertMode,
}

impl BenchContext for SlottedPageResetInsertCtx {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("slotted-page reset insert bench must use factory-backed setup");
    }

    fn chunk_size() -> Option<usize> {
        Some(64)
    }
}

#[derive(Clone, Copy)]
enum InsertMode {
    Append,
    Middle,
}

fn init_slotted_page(fill_entries: usize) -> [u8; PAGE_SIZE] {
    let mut page = [0u8; PAGE_SIZE];
    let sp = SlottedPage::init(&mut page);
    let value = [0xCDu8; 24];
    for i in 0..fill_entries {
        let key = (i as u64).to_be_bytes();
        assert!(
            sp.can_insert(key.len(), value.len()),
            "test page fill exceeded capacity"
        );
        sp.insert(sp.num_slots(), &key, &value);
    }
    page
}

fn init_dense_slotted_page() -> [u8; PAGE_SIZE] {
    let mut page = [0u8; PAGE_SIZE];
    let sp = SlottedPage::init(&mut page);
    let value = [0xCDu8; 24];
    let needed = 12 + 8 + value.len();
    let mut next_key = 0u64;
    while sp.free_space_after_compaction() > needed * 2 {
        let key = next_key.to_be_bytes();
        sp.insert(sp.num_slots(), &key, &value);
        next_key += 1;
    }
    page
}

fn init_dense_even_slotted_page() -> [u8; PAGE_SIZE] {
    let mut page = [0u8; PAGE_SIZE];
    let sp = SlottedPage::init(&mut page);
    let value = [0xCDu8; 24];
    let needed = 12 + 8 + value.len();
    let mut next_key = 0u64;
    while sp.free_space_after_compaction() > needed * 2 {
        let key = (next_key * 2).to_be_bytes();
        sp.insert(sp.num_slots(), &key, &value);
        next_key += 1;
    }
    page
}

fn append_remove_leaf_local(ctx: &mut SlottedPageInsertCtx, chunk_size: usize, _chunk_num: usize) {
    for _ in 0..chunk_size {
        let sp = SlottedPage::from_page_mut(&mut ctx.page);
        let slot = sp.num_slots();
        sp.insert(slot, &ctx.insert_key, &ctx.insert_value);
        sp.remove(slot);
        sp.compactify();
        black_box(sp.free_space_after_compaction());
    }
}

fn can_reserve_insertions(
    page: &[u8; PAGE_SIZE],
    reserved_inserts: usize,
    key_base: u64,
    key_stride: u64,
    insert_mode: InsertMode,
    insert_value: &[u8; 24],
) -> bool {
    let mut working = *page;
    let sp = SlottedPage::from_page_mut(&mut working);
    for i in 0..reserved_inserts {
        let key = (key_base + i as u64 * key_stride).to_be_bytes();
        if !sp.can_insert(key.len(), insert_value.len()) {
            return false;
        }
        let insert_pos = match insert_mode {
            InsertMode::Append => sp.num_slots(),
            InsertMode::Middle => sp.lower_bound(&key).0,
        };
        sp.insert(insert_pos, &key, insert_value);
    }
    true
}

fn init_insertable_page(fill_entries: usize, reserved_inserts: usize) -> [u8; PAGE_SIZE] {
    let mut page = init_slotted_page(fill_entries);
    let sp = SlottedPage::from_page_mut(&mut page);
    let needed = reserved_inserts * (12 + 8 + 24);
    while sp.free_space_after_compaction() <= needed {
        let last = sp.num_slots() - 1;
        sp.remove(last);
        sp.compactify();
    }
    page
}

fn init_dense_insertable_page(
    reserved_inserts: usize,
    key_base: u64,
    key_stride: u64,
    insert_mode: InsertMode,
    insert_value: &[u8; 24],
) -> [u8; PAGE_SIZE] {
    let mut page = match insert_mode {
        InsertMode::Append => init_dense_slotted_page(),
        InsertMode::Middle => init_dense_even_slotted_page(),
    };
    while !can_reserve_insertions(
        &page,
        reserved_inserts,
        key_base,
        key_stride,
        insert_mode,
        insert_value,
    ) {
        let sp = SlottedPage::from_page_mut(&mut page);
        assert!(
            sp.num_slots() > 0,
            "could not reserve insertions in dense page"
        );
        let last = sp.num_slots() - 1;
        sp.remove(last);
        sp.compactify();
    }
    page
}

fn reset_insert_once(ctx: &mut SlottedPageResetInsertCtx, chunk_size: usize, _chunk_num: usize) {
    ctx.working.copy_from_slice(&ctx.baseline);
    let sp = SlottedPage::from_page_mut(&mut ctx.working);
    for i in 0..chunk_size {
        let key = (ctx.key_base + i as u64 * ctx.key_stride).to_be_bytes();
        assert!(
            sp.can_insert(key.len(), ctx.insert_value.len()),
            "reset benchmark baseline did not reserve enough free space"
        );
        let insert_pos = match ctx.insert_mode {
            InsertMode::Append => sp.num_slots(),
            InsertMode::Middle => sp.lower_bound(&key).0,
        };
        sp.insert(insert_pos, &key, &ctx.insert_value);
    }
    black_box(sp.free_space_after_compaction());
}

benchmark_main!(|runner| {
    runner.group::<SlottedPageInsertCtx>("slotted_page_leaf_local", |g| {
        g.throughput(Throughput::per_operation(100_000, "operations"))
            .factory(&|| SlottedPageInsertCtx {
                page: init_slotted_page(16),
                insert_key: u64::MAX.to_be_bytes(),
                insert_value: [0xEFu8; 24],
            })
            .bench("append_remove_sparse", append_remove_leaf_local);
        g.throughput(Throughput::per_operation(100_000, "operations"))
            .factory(&|| SlottedPageInsertCtx {
                page: init_dense_slotted_page(),
                insert_key: u64::MAX.to_be_bytes(),
                insert_value: [0xEFu8; 24],
            })
            .bench("append_remove_dense", append_remove_leaf_local);
    });

    runner.group::<SlottedPageResetInsertCtx>("slotted_page_insert_only", |g| {
        g.throughput(Throughput::per_operation(64, "inserts"))
            .factory(&|| {
                let baseline = init_insertable_page(16, 64);
                let insert_value = [0xEFu8; 24];
                SlottedPageResetInsertCtx {
                    baseline,
                    working: [0u8; PAGE_SIZE],
                    insert_value,
                    key_base: u64::MAX - 1024,
                    key_stride: 1,
                    insert_mode: InsertMode::Append,
                }
            })
            .bench("append_sparse_reset", reset_insert_once);
        g.throughput(Throughput::per_operation(64, "inserts"))
            .factory(&|| {
                let insert_value = [0xEFu8; 24];
                let baseline = init_dense_insertable_page(
                    64,
                    u64::MAX - 1024,
                    1,
                    InsertMode::Append,
                    &insert_value,
                );
                SlottedPageResetInsertCtx {
                    baseline,
                    working: [0u8; PAGE_SIZE],
                    insert_value,
                    key_base: u64::MAX - 1024,
                    key_stride: 1,
                    insert_mode: InsertMode::Append,
                }
            })
            .bench("append_dense_reset", reset_insert_once);
        g.throughput(Throughput::per_operation(64, "inserts"))
            .factory(&|| {
                let insert_value = [0xEFu8; 24];
                let baseline =
                    init_dense_insertable_page(64, 65, 2, InsertMode::Middle, &insert_value);
                SlottedPageResetInsertCtx {
                    baseline,
                    working: [0u8; PAGE_SIZE],
                    insert_value,
                    key_base: 65,
                    key_stride: 2,
                    insert_mode: InsertMode::Middle,
                }
            })
            .bench("middle_dense_reset", reset_insert_once);
    });
});
