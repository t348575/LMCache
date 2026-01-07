use std::{
    cell::RefCell,
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use hashbrown::HashMap;
use log::{debug, error};
use pyo3::prelude::*;

use crate::shard::{ShardRequest, ShardResponse};

mod container;
mod shard;

#[pyclass(subclass)]
pub struct StorageBackend {
    shards: Arc<Vec<crate::shard::ShardHandle>>,
    mask: usize,
    id: u64,
}

#[pymethods]
impl StorageBackend {
    #[new]
    fn new(
        _py: Python<'_>,
        config: Py<PyAny>,
        _local_cpu_backend: Py<PyAny>,
        _dst_device: &str,
        _lmcache_worker: Option<Py<PyAny>>,
    ) -> PyResult<StorageBackend> {
        env_logger::init();
        debug!("Rust storage backend initializing");
        let (cache_path, num_shards, max_container_size, chunk_size) =
            Python::attach(|py| -> PyResult<(String, usize, u64, usize)> {
                let cfg = config.as_ref();
                let cache_path = cfg.getattr(py, "local_disk")?.extract(py)?;
                let chunk_size = cfg.getattr(py, "chunk_size")?.extract(py)?;
                let num_shards = cfg
                    .getattr(py, "num_shards")
                    .and_then(|v| v.extract(py))
                    .unwrap_or(num_cpus::get());
                let max_container_size = cfg
                    .getattr(py, "max_container_size")
                    .and_then(|v| v.extract(py))
                    .unwrap_or(1024); // 1GB
                Ok((
                    cache_path,
                    num_shards,
                    max_container_size * 1024 * 1024,
                    chunk_size,
                ))
            })?;
        debug!("{cache_path}, {num_shards}, {max_container_size}, {chunk_size}");

        let root_dir = PathBuf::from(cache_path);
        let mut shards = Vec::with_capacity(num_shards);
        for shard_id in 0..num_shards {
            let shard_dir = root_dir.join(format!("shard_{shard_id}"));
            let handle = crate::shard::ShardHandle::spawn(shard_id, shard_dir, max_container_size)?;
            shards.push(handle);
        }

        Ok(StorageBackend {
            shards: Arc::new(shards),
            mask: num_shards - 1,
            id: 0,
        })
    }

    fn contains_key(&mut self, py: Python<'_>, key: String, _pin: bool) -> bool {
        py.detach(|| {
            let shard = self.shard_for_key(&key);
            let id = self.next_id();
            self.shards[shard]
                .tx
                .send(ShardRequest::Contains { id, key })
                .unwrap();
            self.wait_for_response(shard, id, |resp| match resp {
                ShardResponse::Contains { id: resp_id, res } if resp_id == id => {
                    ExtractResult::Match(res)
                }
                other => ExtractResult::NoMatch(other),
            })
        })
    }

    fn remove_key(&self, py: Python<'_>, key: String) -> bool {
        py.detach(|| {
            self.shards[self.shard_for_key(&key)]
                .tx
                .send(ShardRequest::Delete { key })
                .unwrap();
            true
        })
    }

    fn batched_put(&mut self, py: Python<'_>, keys: Vec<String>, data_ptrs: Vec<(usize, usize)>) {
        py.detach(|| {
            assert!(keys.len() == data_ptrs.len());
            let start = Instant::now();

            let mut per_shard: HashMap<usize, Vec<(u64, String, usize, usize)>> = HashMap::new();
            let mut pending: Vec<(usize, u64)> = Vec::with_capacity(keys.len());

            let mut total_size = 0;
            for (key, (data_ptr, size)) in keys.into_iter().zip(data_ptrs) {
                let shard = self.shard_for_key(&key);
                let id = self.next_id();
                pending.push((shard, id));
                total_size += size;

                per_shard
                    .entry(shard)
                    .or_default()
                    .push((id, key, data_ptr, size));
            }

            for (shard, items) in per_shard {
                self.shards[shard]
                    .tx
                    .send(ShardRequest::PutBatch { items })
                    .unwrap();
            }

            for (shard, id) in pending {
                let res = self.wait_for_response(shard, id, |resp| match resp {
                    ShardResponse::Put { id: resp_id, res } if resp_id == id => {
                        ExtractResult::Match(res)
                    }
                    other => ExtractResult::NoMatch(other),
                });

                if let Err(err) = res {
                    error!("put key failed: {}", err);
                }
            }
            let time = start.elapsed().as_secs_f64();
            let mb = total_size as f64 / 1e6;
            debug!(
                "Took {} s for {} MB, Write bandwidth: {:.2} MB/s",
                time,
                mb,
                mb / time,
            );
        })
    }

    fn get(&mut self, py: Python<'_>, key: String, data_ptr: usize, data_len: usize) -> bool {
        py.detach(|| {
            let start = Instant::now();
            let shard = self.shard_for_key(&key);
            let id = self.next_id();
            self.shards[shard]
                .tx
                .send(ShardRequest::Get {
                    id,
                    key,
                    data_ptr,
                    size: data_len,
                })
                .unwrap();
            let res = self.wait_for_response(shard, id, |resp| match resp {
                ShardResponse::Get { id: resp_id, res } if resp_id == id => {
                    ExtractResult::Match(res.unwrap_or(false))
                }
                other => ExtractResult::NoMatch(other),
            });
            let time = start.elapsed().as_secs_f64();
            let mb = data_len as f64 / 1e6;
            debug!(
                "Took {} s for {} MB, Read bandwidth: {:.2} MB/s",
                time,
                mb,
                mb / time,
            );
            res
        })
    }

    fn batched_get(&mut self, py: Python<'_>, keys: Vec<String>, data_ptrs: Vec<(usize, usize)>) {
        py.detach(|| {
            let start = Instant::now();
            assert!(keys.len() == data_ptrs.len());
            use hashbrown::HashMap;

            let mut per_shard: HashMap<usize, Vec<(u64, String, usize, usize)>> = HashMap::new();
            let mut pending: Vec<(usize, u64)> = Vec::with_capacity(keys.len());

            let mut total_size = 0;
            for (key, (data_ptr, size)) in keys.into_iter().zip(data_ptrs) {
                let shard = self.shard_for_key(&key);
                let id = self.next_id();
                pending.push((shard, id));

                total_size += size;
                per_shard
                    .entry(shard)
                    .or_default()
                    .push((id, key, data_ptr, size));
            }

            for (shard, items) in per_shard {
                self.shards[shard]
                    .tx
                    .send(ShardRequest::GetBatch { items })
                    .unwrap();
            }

            for (shard, id) in pending {
                let res = self.wait_for_response(shard, id, |resp| match resp {
                    ShardResponse::Get { id: resp_id, res } if resp_id == id => {
                        ExtractResult::Match(res)
                    }
                    other => ExtractResult::NoMatch(other),
                });

                if let Err(err) = res {
                    error!("get key failed: {}", err);
                }
            }

            let time = start.elapsed().as_secs_f64();
            let mb = total_size as f64 / 1e6;
            debug!(
                "Took {} s for {} MB, Read bandwidth: {:.2} MB/s",
                time,
                mb,
                mb / time,
            );
        })
    }

    fn batched_contains_key(&mut self, py: Python<'_>, keys: Vec<String>) -> usize {
        py.detach(|| {
            let mut pending: Vec<(usize, u64)> = Vec::with_capacity(keys.len());
            for key in keys {
                let shard = self.shard_for_key(&key);
                let id = self.next_id();
                pending.push((shard, id));

                self.shards[shard]
                    .tx
                    .send(ShardRequest::Contains { id, key })
                    .unwrap();
            }

            let mut items = Vec::new();
            for (shard, id) in pending {
                let res = self.wait_for_response(shard, id, |resp| match resp {
                    ShardResponse::Contains { id: resp_id, res } if resp_id == id => {
                        ExtractResult::Match((id, res))
                    }
                    other => ExtractResult::NoMatch(other),
                });
                items.push(res);
            }
            items.sort_by_key(|x| x.0);
            let mut count = 0;
            for (_, exists) in items {
                if exists {
                    count += 1;
                } else {
                    return count;
                }
            }
            count
        })
    }
}

enum ExtractResult<T> {
    Match(T),
    NoMatch(ShardResponse),
}

thread_local! {
    static STASH: RefCell<HashMap<(usize, u64), ShardResponse>> =
        RefCell::new(HashMap::new());
}

impl StorageBackend {
    fn shard_for_key(&self, key: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) & self.mask
    }

    fn next_id(&mut self) -> u64 {
        self.id += 1;
        self.id
    }

    fn wait_for_response<F, T>(&self, shard_idx: usize, id: u64, extract: F) -> T
    where
        F: Fn(ShardResponse) -> ExtractResult<T>,
    {
        if let Some(val) = STASH.with(|stash| {
            let mut stash = stash.borrow_mut();
            stash
                .remove(&(shard_idx, id))
                .map(|resp| match extract(resp) {
                    ExtractResult::Match(v) => v,
                    ExtractResult::NoMatch(resp) => {
                        stash.insert((shard_idx, id), resp);
                        panic!("extract() returned NoMatch for stashed response");
                    }
                })
        }) {
            return val;
        }

        loop {
            let resp = self.shards[shard_idx].rx.recv().unwrap();
            let key = match &resp {
                ShardResponse::Put { id, .. } => (shard_idx, *id),
                ShardResponse::Get { id, .. } => (shard_idx, *id),
                ShardResponse::Contains { id, .. } => (shard_idx, *id),
            };

            match extract(resp) {
                ExtractResult::Match(v) => {
                    return v;
                }
                ExtractResult::NoMatch(resp) => {
                    STASH.with(|stash| {
                        let mut stash = stash.borrow_mut();
                        stash.insert(key, resp);
                    });
                }
            }
        }
    }
}

#[pymodule]
#[pyo3(name = "_rust_storage_backend")]
fn native_backend(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StorageBackend>()?;
    Ok(())
}
