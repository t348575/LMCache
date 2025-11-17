import os
import threading
import time
import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from lmcache.v1.storage_backend.hash_index import HashIndex, Location

@dataclass
class Container:
    id: int
    used: int
    fd: Optional[int]
    # (offset, size)
    entries: List[Tuple[int, int]] = field(default_factory=list)

    def find_free_slot(self, datalen: int) -> Optional[int]:
        if not self.entries:
            return 0

        first_offset, first_size = self.entries[0]
        if first_offset >= datalen:
            return 0

        prev_end = first_offset + first_size
        for offset, sz in self.entries[1:]:
            if offset > prev_end:
                gap = offset - prev_end
                if gap >= datalen:
                    return prev_end
            prev_end = max(prev_end, offset + sz)

        return prev_end

    def allocate(self, datalen: int) -> Optional[int]:
        offset = self.find_free_slot(datalen)
        if offset is None:
            return None

        bisect.insort(self.entries, (offset, datalen))
        self.used += datalen
        return offset

    def free(self, offset: int, size: int):
        for i, (o, s) in enumerate(self.entries):
            if o == offset and s == size:
                self.entries.pop(i)
                self.used -= size
                return

class FileStore:
    def __init__(self, path: str, max_container_size: int, logger):
        logger.debug(f"Creating FileStore at {path}")
        self.root_dir = Path(path)
        self.max_container_size = max_container_size
        # self.key_index = HashIndex(path)
        self.key_index: dict[str, Location] = {}
        self.container_pool: dict[int, Container] = {}
        self.open_fds = 0
        self._next_container_id = 0
        self.logger = logger
        self._lock = threading.Lock()

    def _create_new_container(self) -> Container:
        self.container_pool[self._next_container_id] = Container(id=self._next_container_id, used=0, entries=[], fd=None)

        fd = os.open(self.root_dir / f"{self._next_container_id}.bin", os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
        os.posix_fallocate(fd, 0, self.max_container_size)

        current_container = self.container_pool[self._next_container_id]
        self._next_container_id += 1
        return current_container

    def _find_free_container(self, datalen: int) -> Optional[Container]:
        for container in self.container_pool.values():
            if container.used + datalen <= self.max_container_size:
                return container
        return None

    def _ensure_container_for_write(self, datalen: int) -> Container:
        container = self._find_free_container(datalen)

        if container is None:
            return self._create_new_container()
        return container

    def _get_fd(self, container_id: int) -> int:
        if self.container_pool[container_id].fd is not None:
            return self.container_pool[container_id].fd

        if self.open_fds == 512:
            i = 0
            for c in self.container_pool.values():
                if i == 64:
                    break
                if c.fd is not None:
                    os.close(c.fd)
                    c.fd = None
                    self.open_fds -= 1
                    i += 1

        fd = os.open(self.root_dir / f"{container_id}.bin", os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
        self.open_fds += 1
        self.container_pool[container_id].fd = fd
        return fd

    def put(self, key: str, data: bytes):
        with self._lock:
            datalen = len(data)
            container = self._ensure_container_for_write(datalen)
            offset = container.allocate(datalen)
            # self.key_index.put(key, Location(self._current_container.id, datalen, offset))
            self.key_index[key] = Location(container.id, datalen, offset)
            os.pwrite(self._get_fd(container.id), data, offset)

    def get(self, key: str, buffer: bytes) -> Optional[bytes]:
        with self._lock:
            # loc = self.key_index.get(key)
            loc = self.key_index[key]
            if loc is None:
                return None

            os.preadv(self._get_fd(loc.file_id), [buffer], loc.offset)

    def delete(self, key: str):
        with self._lock:
            # loc = self.key_index.get(key)
            loc = self.key_index[key]
            if loc is None:
                return

            self.key_index.delete(key)
            self.container_pool[loc.file_id].free(loc.offset, loc.file_size)
