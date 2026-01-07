use std::{
    fs::{File, OpenOptions},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::Path,
};

#[derive(Debug)]
pub struct Container {
    pub used: u64,
    pub entries: Vec<(u64, u64)>,
    pub file: File,
    pub max_size: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Location {
    pub file_id: u32,
    pub file_size: u64,
    pub offset: u64,
}

fn posix_fallocate_file(file: &File, len: u64) -> io::Result<()> {
    let fd = file.as_raw_fd();
    let res = unsafe { libc::posix_fallocate(fd, 0, len as libc::off_t) };
    if res == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(res))
    }
}

const BLOCK_SIZE: u64 = 4096;

#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

impl Container {
    pub fn new(root_dir: &Path, id: u32, max_size: u64) -> io::Result<Self> {
        let path = root_dir.join(format!("{id}.bin"));
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .truncate(true)
            .write(true)
            .custom_flags(0o0040000)
            .open(&path)?;

        let max_size_aligned = align_up(max_size, BLOCK_SIZE);
        posix_fallocate_file(&file, max_size_aligned)?;

        Ok(Self {
            used: 0,
            entries: Vec::new(),
            file,
            max_size: max_size_aligned,
        })
    }

    pub fn find_free_slot(&self, datalen: u64) -> Option<u64> {
        if self.entries.is_empty() {
            if datalen <= self.max_size {
                return Some(0);
            } else {
                return None;
            }
        }

        let (first_offset, first_size) = self.entries[0];
        if first_offset >= datalen {
            return Some(0);
        }

        let mut prev_end = first_offset + first_size;

        for &(offset, sz) in &self.entries[1..] {
            if offset > prev_end {
                let gap = offset - prev_end;
                if gap >= datalen {
                    return Some(prev_end);
                }
            }
            prev_end = prev_end.max(offset + sz);
        }

        if prev_end + datalen <= self.max_size {
            Some(prev_end)
        } else {
            None
        }
    }

    pub fn allocate(&mut self, datalen: u64) -> Option<u64> {
        let datalen = align_up(datalen, BLOCK_SIZE);
        let offset = self.find_free_slot(datalen)?;
        let pos = self
            .entries
            .binary_search_by_key(&offset, |(o, _)| *o)
            .unwrap_or_else(|e| e);
        self.entries.insert(pos, (offset, datalen));
        self.used += datalen;
        Some(offset)
    }

    pub fn free(&mut self, offset: u64, size: u64) {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|&(o, s)| o == offset && s == size)
        {
            self.entries.remove(pos);
            self.used = self.used.saturating_sub(size);
        }
    }
}
