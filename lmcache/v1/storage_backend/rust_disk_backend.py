import time
from typing import List, Optional, Sequence, Any
import torch
import traceback
import sys

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg

from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from _rust_storage_backend import StorageBackend

from lmcache.v1.storage_backend.cache_policy import get_cache_policy

logger = init_logger(__name__)

class RustStorageBackend(StorageBackendInterface, StorageBackend):
    def __init__(self, config, local_cpu_backend, dst_device="cuda", lmcache_worker=None):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.local_cpu_backend = local_cpu_backend
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()
        self.put_tasks: List[CacheEngineKey] = []
        self.keys_in_request: List[CacheEngineKey] = []
        self.current_cache_size = 0.0
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        logger.debug("Actually working")

    def __str__(self):
        return "RustStorageBackend"

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if key not in self.dict:
            return False
        if pin:
            self.dict[key].pin()
        self.keys_in_request.append(key)
        return True

    def touch_cache(self):
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return key in self.put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        if key in self.dict:
            self.dict[key].pin()
            return True
        else:
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        if key in self.dict:
            self.dict[key].unpin()
            return True
        else:
            return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if not (meta := self.dict.pop(key, None)):
            return False
        self.stats_monitor.update_local_storage_usage(meta.size)
        self.remove_key(key.to_string())
        if force:
            self.cache_policy.update_on_force_evict(key)
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )
        return True

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        keys_str = list(map(self.set_keys_str_on_put, zip(keys, memory_objs)))
        memory_buffers = list(map(self.update_memory_obj_on_put, memory_objs))
        start_time = time.time()
        self.batched_put(keys_str, memory_buffers)
        disk_write_time = time.time() - start_time
        size = sum(ptr[1] for ptr in memory_buffers)
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )

        for key, obj in zip(keys, memory_objs):
            size = obj.get_physical_size()
            shape = obj.metadata.shape
            dtype = obj.metadata.dtype
            fmt = obj.metadata.fmt
            cached_positions = obj.metadata.cached_positions
            obj.ref_count_down()
            has_stored = False
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(key.to_string(), size, shape, dtype, cached_positions, fmt, 0)
            self.put_tasks.remove(key)
            if self.lmcache_worker is not None and not has_stored:
                self.lmcache_worker.put_msg(
                    KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
                )

    def set_keys_str_on_put(self, data: tuple[CacheEngineKey, MemoryObj]) -> str:
        key, memory_obj = data
        self.put_tasks.append(key)

        required_size = memory_obj.get_physical_size()
        all_evict_keys = []
        evict_success = True
        while self.current_cache_size + required_size > self.max_cache_size:
            evict_keys = self.cache_policy.get_evict_candidates(
                self.dict, num_candidates=1
            )
            if not evict_keys:
                logger.warning(
                    "No eviction candidates found. Disk space under pressure."
                )
                evict_success = False
                break

            for evict_key in evict_keys:
                self.current_cache_size -= self.dict[evict_key].size

            self.batched_remove(evict_keys, force=False)

            all_evict_keys.extend(evict_keys)
        if evict_success:
            self.current_cache_size += required_size
        self.cache_policy.update_on_put(key)
        return key.to_string()

    def update_memory_obj_on_put(self, memory_obj: MemoryObj) -> bytearray:
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        # size = len(memory_obj.byte_array)
        size = memory_obj.get_physical_size()
        memory_obj.ref_count_up()
        self.stats_monitor.update_local_storage_usage(size)
        return memory_obj.data_ptr, size

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # traceback.print_stack(limit=30, file=sys.stderr)
        meta = self.dict[key]
        if meta is None:
            return None

        self.cache_policy.update_on_hit(key, self.dict)
        memory_obj = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)

        start_time = time.time()
        # size = len(memory_obj.byte_array)
        size = memory_obj.get_physical_size()
        self.get(key.to_string(), memory_obj.data_ptr, size)
        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions
        return memory_obj

    async def get_non_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        meta = self.dict[key]
        if meta is None:
            return None

        self.cache_policy.update_on_hit(key, self.dict)
        memory_obj = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)

        start_time = time.time()
        # size = len(memory_obj.byte_array)
        size = memory_obj.get_physical_size()
        self.get(key.to_string(), memory_obj.data_ptr, size)
        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions
        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        mem_ptrs = []
        mem_objs = []
        keys_str = [key.to_string() for key in keys]
        for key in keys:
            meta = self.dict[key]
            assert meta is not None, f"Key {key} not found in disk cache after pinning"
            memory_obj = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)
            self.dict[key].pin()
            self.cache_policy.update_on_hit(key, self.dict)
            mem_ptrs.append((memory_obj.data_ptr, memory_obj.get_physical_size()))
            mem_objs.append(memory_obj)

        start_time = time.time()
        self.batched_get(keys_str, mem_ptrs)
        disk_read_time = time.time() - start_time
        size = sum(ptr[1] for ptr in mem_ptrs)
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )
        for key, mem_obj in zip(keys, mem_objs):
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions
            self.dict[key].unpin()

        return mem_objs

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        print("batched_get_non_blocking")
        mem_ptrs = []
        mem_objs = []
        keys_str = [key.to_string() for key in keys]
        for key in keys:
            meta = self.dict[key]
            assert meta is not None, f"Key {key} not found in disk cache after pinning"
            memory_obj = self.local_cpu_backend.allocate(meta.shape, meta.dtype, meta.fmt)
            self.dict[key].pin()
            self.cache_policy.update_on_hit(key, self.dict)
            mem_ptrs.append((memory_obj.data_ptr, memory_obj.get_physical_size()))
            mem_objs.append(memory_obj)

        start_time = time.time()
        self.batched_get(keys_str, mem_ptrs)
        disk_read_time = time.time() - start_time
        size = sum(ptr[1] for ptr in mem_ptrs)
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )
        for key, mem_obj in zip(keys, mem_objs):
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions
            self.dict[key].unpin()

        return mem_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            if key not in self.dict:
                return num_hit_counts
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            num_hit_counts += 1
        return num_hit_counts
    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        self.disk_worker.close()
