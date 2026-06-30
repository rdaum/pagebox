use std::cmp::Ordering;

pub type Timestamp = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CowBeTreeMessage {
    PutVersion {
        pk: Vec<u8>,
        commit_ts: Timestamp,
        row: Vec<u8>,
    },
    DeleteVersion {
        pk: Vec<u8>,
        commit_ts: Timestamp,
    },
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
        commit_ts: Timestamp,
    },
    Delete {
        key: Vec<u8>,
        commit_ts: Timestamp,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BufferedMessage {
    pub(crate) key: Vec<u8>,
    pub(crate) version: VersionRecord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VersionRecord {
    pub(crate) commit_ts: Timestamp,
    pub(crate) value: Vec<u8>,
    pub(crate) deleted: bool,
}

impl CowBeTreeMessage {
    pub fn put_version(
        pk: impl Into<Vec<u8>>,
        commit_ts: Timestamp,
        row: impl Into<Vec<u8>>,
    ) -> Self {
        Self::PutVersion {
            pk: pk.into(),
            commit_ts,
            row: row.into(),
        }
    }

    pub fn delete_version(pk: impl Into<Vec<u8>>, commit_ts: Timestamp) -> Self {
        Self::DeleteVersion {
            pk: pk.into(),
            commit_ts,
        }
    }

    pub fn put(key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>, commit_ts: Timestamp) -> Self {
        Self::Put {
            key: key.into(),
            value: value.into(),
            commit_ts,
        }
    }

    pub fn delete(key: impl Into<Vec<u8>>, commit_ts: Timestamp) -> Self {
        Self::Delete {
            key: key.into(),
            commit_ts,
        }
    }

    pub(crate) fn into_buffered(self) -> BufferedMessage {
        match self {
            Self::PutVersion { pk, commit_ts, row } => BufferedMessage {
                key: pk,
                version: VersionRecord {
                    commit_ts,
                    value: row,
                    deleted: false,
                },
            },
            Self::DeleteVersion { pk, commit_ts } => BufferedMessage {
                key: pk,
                version: VersionRecord {
                    commit_ts,
                    value: Vec::new(),
                    deleted: true,
                },
            },
            Self::Put {
                key,
                value,
                commit_ts,
            } => BufferedMessage {
                key,
                version: VersionRecord {
                    commit_ts,
                    value,
                    deleted: false,
                },
            },
            Self::Delete { key, commit_ts } => BufferedMessage {
                key,
                version: VersionRecord {
                    commit_ts,
                    value: Vec::new(),
                    deleted: true,
                },
            },
        }
    }
}

impl BufferedMessage {
    pub(crate) fn put(key: &[u8], commit_ts: Timestamp, value: &[u8]) -> Self {
        Self {
            key: key.to_vec(),
            version: VersionRecord {
                commit_ts,
                value: value.to_vec(),
                deleted: false,
            },
        }
    }

    pub(crate) fn delete(key: &[u8], commit_ts: Timestamp) -> Self {
        Self {
            key: key.to_vec(),
            version: VersionRecord {
                commit_ts,
                value: Vec::new(),
                deleted: true,
            },
        }
    }

    pub(crate) fn encoded_len(&self) -> usize {
        1 + 8 + 2 + 4 + self.key.len() + self.version.value.len()
    }

    pub(crate) fn cmp_buffer_order(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| other.version.commit_ts.cmp(&self.version.commit_ts))
    }
}

pub(crate) fn sort_buffer_messages(buffer: &mut [BufferedMessage]) {
    buffer.sort_by(BufferedMessage::cmp_buffer_order);
}
