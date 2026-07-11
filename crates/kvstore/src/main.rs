use std::path::PathBuf;

use clap::{Parser, Subcommand};
use kvstore::{KvStore, KvStoreOptions, SyncMode};

#[derive(Parser)]
#[command(name = "kvstore")]
#[command(about = "A durable KV store built on the pagebox substrate")]
struct Cli {
    /// Data directory.
    #[arg(long, default_value = "./kvstore-data")]
    data_dir: PathBuf,

    /// Use strict durability (fsync WAL after every write).
    #[arg(long)]
    sync: bool,

    /// Buffer-pool frame count (defaults to a 64 MiB data-page budget).
    #[arg(long, default_value_t = kvstore::DEFAULT_POOL_FRAMES)]
    pool_frames: usize,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Insert or update a key-value pair.
    Put { key: String, value: String },
    /// Look up a key.
    Get { key: String },
    /// Delete a key.
    Del { key: String },
    /// Scan all key-value pairs.
    Scan,
    /// Range scan over [start, end).
    Range { start: String, end: String },
    /// Flush dirty pages and checkpoint the WAL.
    Checkpoint,
    /// Flush WAL + dirty pages without checkpointing.
    Sync,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let opts = KvStoreOptions::default()
        .pool_frames(cli.pool_frames)
        .sync_mode(if cli.sync {
            SyncMode::Strict
        } else {
            SyncMode::Relaxed
        });
    let kv = KvStore::open_with(&cli.data_dir, &opts)?;

    match &cli.command {
        Command::Put { key, value } => {
            let inserted = kv.put(key.as_bytes(), value.as_bytes());
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
            kv.scan_all(|k, v| {
                let ks = String::from_utf8_lossy(k);
                let vs = String::from_utf8_lossy(v);
                println!("{} -> {}", ks, vs);
                count += 1;
            });
            println!("({} keys)", count);
        }
        Command::Range { start, end } => {
            let mut count = 0;
            kv.scan_range(start.as_bytes(), end.as_bytes(), |k, v| {
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
        Command::Sync => {
            kv.sync()?;
            println!("SYNC done");
        }
    }

    Ok(())
}
