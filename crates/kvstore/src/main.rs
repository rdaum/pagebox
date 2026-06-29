use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use pagebox_btree::BTree;
use pagebox_storage::buffer_pool::{BufferPool, BufferPoolHandle};
use pagebox_storage::page_header::read_page_lsn;
use pagebox_storage::page_store::{FilePageStore, PageStore};
use pagebox_wal::Wal;

const POOL_FRAMES: usize = 1024;
const DT_ID: u16 = 1;

pub struct KvStore {
    pool: BufferPoolHandle,
    tree: BTree,
    wal: Arc<Wal>,
    store: Arc<FilePageStore>,
}

impl KvStore {
    pub fn open<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let data_path = dir.join("kvstore.data");
        let wal_path = dir.join("kvstore.wal");

        let store = Arc::new(FilePageStore::open(&data_path)?);
        let wal = Wal::open_opts(&wal_path)?;

        let checkpoint_lsn = store.checkpoint_lsn();
        let report = wal.recover(&*store, checkpoint_lsn, read_page_lsn)?;
        if report.max_lsn > checkpoint_lsn {
            store.sync()?;
            store.set_checkpoint_lsn(report.max_lsn);
            store.sync()?;
            wal.reset()?;
        }
        let effective_checkpoint = store.checkpoint_lsn();
        wal.advance_lsn_past(effective_checkpoint);

        let wal = Arc::new(wal);
        let mut pool = BufferPool::with_store(POOL_FRAMES, Box::new(store.clone()));
        pool.set_wal(wal.clone());
        let pool: BufferPoolHandle = Arc::new(pool).into();

        let root = store.user_meta_0();
        let height = store.user_meta_1() as u32;
        let tree = if root == 0 {
            let t = BTree::new(pool.clone(), DT_ID);
            store.set_user_meta_0(t.root_page_id());
            store.set_user_meta_1(0);
            store.sync()?;
            t
        } else {
            BTree::open(pool.clone(), root, height, DT_ID)
        };

        Ok(Self {
            pool,
            tree,
            wal,
            store,
        })
    }

    pub fn put(&self, key: &[u8], value: &[u8]) -> bool {
        self.tree.upsert(key, value)
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.tree.lookup(key)
    }

    pub fn del(&self, key: &[u8]) -> bool {
        self.tree.remove(key)
    }

    pub fn scan<F: FnMut(&[u8], &[u8])>(&self, f: F) {
        self.tree.scan(f);
    }

    pub fn checkpoint(&self) -> std::io::Result<()> {
        let checkpoint_lsn = self.wal.flush();
        self.pool.flush()?;
        self.store.set_user_meta_0(self.tree.root_page_id());
        self.store.set_user_meta_1(self.tree.height() as u64);
        self.store.set_checkpoint_lsn(checkpoint_lsn);
        self.store.sync()?;
        self.wal.reset()?;
        Ok(())
    }
}

#[derive(Parser)]
#[command(name = "kvstore")]
#[command(about = "A durable KV store built on the pagebox substrate")]
struct Cli {
    /// Data directory.
    #[arg(long, default_value = "./kvstore-data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Insert or update a key-value pair.
    Put {
        key: String,
        value: String,
        /// Flush WAL after this write for strict durability.
        #[arg(long)]
        sync: bool,
    },
    /// Look up a key.
    Get { key: String },
    /// Delete a key.
    Del { key: String },
    /// Scan all key-value pairs.
    Scan,
    /// Flush dirty pages and checkpoint the WAL.
    Checkpoint,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let kv = KvStore::open(&cli.data_dir)?;

    match &cli.command {
        Command::Put { key, value, sync } => {
            let inserted = kv.put(key.as_bytes(), value.as_bytes());
            if *sync {
                kv.wal.flush();
            }
            if inserted {
                println!("PUT {} -> {}", key, value);
            } else {
                println!("UPDATED {} -> {}", key, value);
            }
        }
        Command::Get { key } => match kv.get(key.as_bytes()) {
            Some(val) => {
                let val_str = String::from_utf8_lossy(&val);
                println!("{}", val_str);
            }
            None => {
                println!("(none)");
            }
        },
        Command::Del { key } => {
            let removed = kv.del(key.as_bytes());
            if removed {
                println!("DELETED {}", key);
            } else {
                println!("NOT FOUND {}", key);
            }
        }
        Command::Scan => {
            let mut count = 0;
            kv.scan(|k, v| {
                let ks = String::from_utf8_lossy(k);
                let vs = String::from_utf8_lossy(v);
                println!("{} -> {}", ks, vs);
                count += 1;
            });
            println!("({} keys)", count);
        }
        Command::Checkpoint => {
            kv.checkpoint()?;
            println!("CHECKPOINT done");
        }
    }

    Ok(())
}
