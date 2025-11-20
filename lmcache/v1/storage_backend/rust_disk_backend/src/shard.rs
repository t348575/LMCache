use std::{io, os::fd::AsRawFd, path::PathBuf, thread::spawn};

use crossbeam_channel::{Receiver, Sender, unbounded};
use hashbrown::HashMap;
use io_uring::{IoUring, opcode, types};
use log::debug;

use crate::container::{Container, Location};

struct ShardState {
    root_dir: PathBuf,
    max_container_size: u64,
    key_index: HashMap<String, Location>,
    containers: HashMap<u32, Container>,
    next_container_id: u32,
    shard_id: usize,
    ring: IoUring,
}

impl ShardState {
    fn new(shard_id: usize, root_dir: PathBuf, max_container_size: u64) -> io::Result<Self> {
        std::fs::create_dir_all(&root_dir)?;
        let ring = IoUring::new(256)?;
        Ok(Self {
            shard_id,
            root_dir,
            max_container_size,
            key_index: HashMap::new(),
            containers: HashMap::new(),
            next_container_id: 0,
            ring,
        })
    }

    fn create_new_container(&mut self) -> io::Result<u32> {
        let id = self.next_container_id;
        self.next_container_id += 1;
        let container = Container::new(&self.root_dir, id, self.max_container_size)?;
        self.containers.insert(id, container);
        Ok(id)
    }

    fn ensure_container_for_write(&mut self, datalen: u64) -> io::Result<u32> {
        if let Some(id) = self.containers.iter().find_map(|(&id, c)| {
            (datalen <= c.max_size && c.find_free_slot(datalen).is_some()).then_some(id)
        }) {
            return Ok(id);
        }

        self.create_new_container()
    }

    fn put(&mut self, key: String, data_ptr: usize, size: usize) -> io::Result<()> {
        let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, size) };
        let datalen = data.len() as u64;
        let file_id = self.ensure_container_for_write(datalen).expect(&format!(
            "Shard {} Could not get container for write",
            self.shard_id
        ));
        let (fd, offset) = {
            let container = self.containers.get_mut(&file_id).unwrap();
            let offset = container.allocate(datalen).unwrap();
            let file = container.file.as_raw_fd();
            (file, offset)
        };

        self.write_at_io_uring(fd, data, offset)?;
        let loc = Location {
            file_id,
            file_size: datalen,
            offset,
        };
        self.key_index.insert(key, loc);
        Ok(())
    }

    fn put_batch(&mut self, items: Vec<(u64, String, usize, usize)>) -> Vec<(u64, io::Result<()>)> {
        use std::slice;

        struct Pending<'a> {
            id: u64,
            data: &'a [u8],
        }

        let mut pendings: Vec<Pending<'_>> = Vec::with_capacity(items.len());

        for (idx, (id, key, data_ptr, size)) in items.into_iter().enumerate() {
            let data = unsafe { slice::from_raw_parts(data_ptr as *const u8, size) };
            let datalen = data.len() as u64;

            let file_id = self.ensure_container_for_write(datalen).expect(&format!(
                "Shard {} Could not get container for write",
                self.shard_id
            ));

            let (fd, offset) = {
                let container = self.containers.get_mut(&file_id).unwrap();
                let offset = container.allocate(datalen).unwrap();
                let file = container.file.as_raw_fd();
                (file, offset)
            };

            let entry = opcode::Write::new(types::Fd(fd), data.as_ptr(), data.len() as u32)
                .offset(offset)
                .build()
                .user_data(idx as u64);

            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .expect("submission queue is full");
            }

            let loc = Location {
                file_id,
                file_size: datalen,
                offset,
            };
            self.key_index.insert(key, loc);

            pendings.push(Pending { id, data });
        }

        if pendings.is_empty() {
            return Vec::new();
        }

        let submitted = pendings.len();
        self.ring
            .submit_and_wait(submitted)
            .expect("submit_and_wait failed");

        let mut results: Vec<io::Result<()>> = Vec::with_capacity(submitted);
        (0..submitted).for_each(|_| results.push(Ok(())));

        for _ in 0..submitted {
            let cqe = self
                .ring
                .completion()
                .next()
                .expect("completion queue empty");

            let idx = cqe.user_data() as usize;
            let res = cqe.result();
            if res < 0 {
                results[idx] = Err(io::Error::from_raw_os_error(-res));
            } else if res as usize != pendings[idx].data.len() {
                results[idx] = Err(io::Error::new(io::ErrorKind::Other, "short write in batch"));
            }
        }

        pendings
            .into_iter()
            .zip(results.into_iter())
            .map(|(p, r)| (p.id, r))
            .collect()
    }

    fn get(&mut self, key: String, data_ptr: usize, size: usize) -> io::Result<bool> {
        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr as *mut u8, size) };
        let loc = match self.key_index.get(&key).cloned() {
            Some(loc) => loc,
            None => return Ok(false),
        };

        let container = match self.containers.get(&loc.file_id) {
            Some(c) => c,
            None => return Ok(false),
        };

        assert!(loc.file_size <= size as u64);
        self.read_at_io_uring(container.file.as_raw_fd(), data, loc.offset)?;
        Ok(true)
    }

    fn get_batch(
        &mut self,
        items: Vec<(u64, String, usize, usize)>,
    ) -> Vec<(u64, io::Result<bool>)> {
        use std::slice;

        struct Pending {
            res_index: usize,
            len: usize,
        }

        let mut pendings: Vec<Pending> = Vec::new();
        let mut result_ids: Vec<u64> = Vec::with_capacity(items.len());
        let mut results: Vec<io::Result<bool>> = Vec::with_capacity(items.len());

        for (id, key, data_ptr, size) in items {
            result_ids.push(id);
            results.push(Ok(false));
            let res_index = results.len() - 1;

            let buf = unsafe { slice::from_raw_parts_mut(data_ptr as *mut u8, size) };

            let loc = match self.key_index.get(&key).cloned() {
                Some(loc) => loc,
                None => {
                    continue;
                }
            };

            let container = match self.containers.get(&loc.file_id) {
                Some(c) => c,
                None => {
                    continue;
                }
            };

            assert!(loc.file_size <= size as u64);
            let read_len = loc.file_size as usize;

            let entry = opcode::Read::new(
                types::Fd(container.file.as_raw_fd()),
                buf.as_mut_ptr(),
                read_len as u32,
            )
            .offset(loc.offset)
            .build()
            .user_data(pendings.len() as u64);

            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .map_err(|_| io::Error::new(io::ErrorKind::Other, "submission queue is full")).unwrap();
            }

            pendings.push(Pending {
                res_index,
                len: read_len,
            });
        }

        if !pendings.is_empty() {
            let submitted = pendings.len();
            self.ring
                .submit_and_wait(submitted)
                .expect("submit_and_wait failed");

            for _ in 0..submitted {
                let cqe = self
                    .ring
                    .completion()
                    .next()
                    .expect("completion queue empty");
                let idx = cqe.user_data() as usize;
                let pending = &pendings[idx];

                let res = cqe.result();
                if res < 0 {
                    results[pending.res_index] = Err(io::Error::from_raw_os_error(-res));
                } else if res as usize != pending.len {
                    results[pending.res_index] = Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "short read in batch",
                    ));
                } else {
                    results[pending.res_index] = Ok(true);
                }
            }
        }

        result_ids.into_iter().zip(results.into_iter()).collect()
    }

    fn delete(&mut self, key: String) {
        if let Some(loc) = self.key_index.remove(&key) {
            if let Some(container) = self.containers.get_mut(&loc.file_id) {
                container.free(loc.offset, loc.file_size);
            }
        }
    }

    fn contains_key(&self, key: String) -> bool {
        self.key_index.contains_key(&key)
    }

    fn write_at_io_uring(&mut self, fd: i32, data: &[u8], offset: u64) -> io::Result<()> {
        let entry = opcode::Write::new(types::Fd(fd), data.as_ptr(), data.len() as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "submission queue is full"))?;
        }

        self.ring.submit_and_wait(1)?;

        let cqe = self
            .ring
            .completion()
            .next()
            .expect("completion queue empty");
        let res = cqe.result();
        if res < 0 {
            Err(io::Error::from_raw_os_error(-res))
        } else if res as usize != data.len() {
            Err(io::Error::new(io::ErrorKind::Other, "short write"))
        } else {
            Ok(())
        }
    }

    fn read_at_io_uring(&mut self, fd: i32, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let entry = opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), buf.len() as u32)
            .offset(offset)
            .build()
            .user_data(0);

        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "submission queue is full"))?;
        }

        self.ring.submit_and_wait(1)?;

        let cqe = self
            .ring
            .completion()
            .next()
            .expect("completion queue empty");
        let res = cqe.result();
        if res < 0 {
            Err(io::Error::from_raw_os_error(-res))
        } else if res as usize != buf.len() {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub enum ShardRequest {
    Put {
        id: u64,
        key: String,
        data_ptr: usize,
        size: usize,
    },
    PutBatch {
        items: Vec<(u64, String, usize, usize)>, // (id, key, data_ptr, size)
    },
    Get {
        id: u64,
        key: String,
        data_ptr: usize,
        size: usize,
    },
    GetBatch {
        items: Vec<(u64, String, usize, usize)>, // (id, key, data_ptr, size)
    },
    Delete {
        key: String,
    },
    Contains {
        id: u64,
        key: String,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum ShardResponse {
    Put { id: u64, res: io::Result<()> },
    Get { id: u64, res: io::Result<bool> },
    Contains { id: u64, res: bool },
}

pub struct ShardHandle {
    pub tx: Sender<ShardRequest>,
    pub rx: Receiver<ShardResponse>,
}

impl ShardHandle {
    pub fn spawn(shard_id: usize, root_dir: PathBuf, max_container_size: u64) -> io::Result<Self> {
        let (tx, rx) = unbounded::<ShardRequest>();
        let (tx_resp, rx_resp) = unbounded::<ShardResponse>();
        spawn(move || {
            let mut state = match ShardState::new(shard_id, root_dir, max_container_size) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to init shard: {e}");
                    return;
                }
            };
            shard_worker_loop(&mut state, rx, tx_resp);
        });

        Ok(Self { tx, rx: rx_resp })
    }
}

fn shard_worker_loop(
    state: &mut ShardState,
    rx: Receiver<ShardRequest>,
    tx: Sender<ShardResponse>,
) {
    while let Ok(req) = rx.recv() {
        match req {
            ShardRequest::Put {
                id,
                key,
                data_ptr,
                size,
            } => {
                let res = state.put(key, data_ptr, size);
                let _ = tx.send(ShardResponse::Put { id, res });
            }
            ShardRequest::PutBatch { items } => {
                let results = state.put_batch(items);
                for (id, res) in results {
                    let _ = tx.send(ShardResponse::Put { id, res });
                }
            }
            ShardRequest::Get {
                id,
                key,
                data_ptr,
                size,
            } => {
                let res = state.get(key, data_ptr, size);
                let _ = tx.send(ShardResponse::Get { id, res });
            }
            ShardRequest::GetBatch { items } => {
                let results = state.get_batch(items);
                for (id, res) in results {
                    let _ = tx.send(ShardResponse::Get { id, res });
                }
            }
            ShardRequest::Delete { key } => state.delete(key),
            ShardRequest::Contains { id, key } => {
                let res = state.contains_key(key);
                let _ = tx.send(ShardResponse::Contains { id, res });
            }
            ShardRequest::Shutdown => {
                break;
            }
        }
    }
}
