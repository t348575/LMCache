# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class HybridDiskWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        # TODO(Jiayi): make executor and its parameters configurable
        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=64)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
            # self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


class HybridDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()
        self.files: Dict[str, tuple[int, int, int]] = {}
        self.fd_pool: Dict[int, tuple[int, int]] = {}
        self.last_file_idx = -1

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        self.loop = loop
        self.chunks_per_file = config.chunks_per_file
        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = HybridDiskWorker(loop)

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

    def __str__(self):
        return "HybridDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                logger.info(f"Key {key} not found in disk cache.")
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        os.remove(path)

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

        logger.info(
            f"Removed {key} from local disk cache. "
            f"Current size: {self.usage / 1024**3:.2f} GB"
        )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
    ) -> None:
        path = self._key_to_path(key)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
                )

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        total_size = sum(mo.get_physical_size() for mo in memory_objs)

        if self.chunks_per_file is None:
            N = len(keys)
        else:
            N = self.chunks_per_file

        keys: List[List[CacheEngineKey]] = [keys[i:i+N] for i in range(0, len(keys), N)]
        memory_objs: List[List[MemoryObj]] = [memory_objs[i:i+N] for i in range(0, len(memory_objs), N)]
        files: List[tuple[int, bool, int]] = []
        for i in range(len(memory_objs)):
            self.last_file_idx += 1
            not_identical = self.is_batch_not_identical(memory_objs[i])
            if not_identical:
                fd = os.open(os.path.join(self.path, f"{self.last_file_idx}.bin"), os.O_CREAT | os.O_RDWR, 0o644)
            else:
                fd = os.open(os.path.join(self.path, f"{self.last_file_idx}.bin"), os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
            files.append((fd, not_identical, self.last_file_idx))
            self.fd_pool[self.last_file_idx] = (fd, not_identical)

        runtime = 0
        for k, mo, file in zip(keys, memory_objs, files):
            fd, not_identical, file_idx = file
            offset = 0
            i = 0
            for item in zip(k, mo, strict=False):
                self.disk_worker.insert_put_task(item[0])
                self.cache_policy.update_on_put(item[0])
                item[1].ref_count_up()
                size = len(item[1].byte_array)
                offset += size
                self.usage += size
                self.stats_monitor.update_local_storage_usage(self.usage)
                self.files[item[0].to_string()] = (file_idx, offset if not_identical else i, size)
                i += 1

            logger.debug(f"Writing not_identical={not_identical}; file={file_idx}")
            start = time.perf_counter()
            if not_identical:
                self.save_batched_bytes_to_disk_one_by_one(zip(k, mo, strict=False), fd)
            else:
                self.save_batched_bytes_to_disk(zip(k, mo, strict=False), fd)
            end = time.perf_counter()
            runtime += end - start

            for item in zip(k, mo, strict=False):
                item[1].ref_count_down()
                key, memory_obj = item
                self.insert_key(key, memory_obj.get_physical_size(), memory_obj.metadata.shape, memory_obj.metadata.dtype, memory_obj.metadata.fmt, cached_positions=memory_obj.metadata.cached_positions)
                self.disk_worker.remove_put_task(item[0])

        end = time.perf_counter()
        runtime = end - start
        mb = total_size / 1e6
        logger.info(f"Took {runtime:.2f} s for {mb} MB, Write bandwidth: {mb/runtime} MB/s for {len(keys)} keys")

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        return self.disk_worker.submit_task("put", self.batched_submit_put_task, keys, objs)

    def prepare_get(self, keys: List[CacheEngineKey]) -> tuple[
        List[List[tuple[CacheEngineKey, MemoryObj]]],
        List[tuple[CacheEngineKey, MemoryObj]],
        List[Optional[MemoryObj]]
    ]:
        self.disk_lock.acquire()
        memory_objs = []
        entries = []
        for key in keys:
            if key not in self.dict:
                memory_objs.append(None)
                continue
            self.cache_policy.update_on_hit(key, self.dict)
            disk_meta = self.dict[key]
            self.dict[key].pin()
            memory_obj = self.local_cpu_backend.allocate(disk_meta.shape, disk_meta.dtype, disk_meta.fmt)
            assert memory_obj is not None, "Memory allocation failed during disk load."
            memory_objs.append(memory_obj)
            entries.append((key, memory_obj))

        sequenced, not_sequenced = self.group_entries_by_sequence(entries)
        logger.debug(f"Sequenced: {len(sequenced)}; Not sequenced: {len(not_sequenced)}")
        self.disk_lock.release()

        def prepare_mem_sequenced(item: list[tuple[CacheEngineKey, MemoryObj]]):
            memory_obj = item[0][1]
            dtype = memory_obj.get_dtype()
            fmt = memory_obj.get_memory_format()
            new_shape = torch.Size((len(item), *memory_obj.get_shape()))
            file_idx, offset, _ = self.files[item[0][0].to_string()]
            _, not_identical = self.fd_pool[file_idx]
            logger.debug(f"mem_allocate in_sequence file_idx={file_idx}; entries={len(item)}; offset={offset}; not_identical={not_identical} shape={new_shape}")
            big = self.local_cpu_backend.allocate(new_shape, dtype, fmt)
            return big

        for i in range(len(sequenced)):
            big = prepare_mem_sequenced(sequenced[i])
            sequenced[i] = (sequenced[i], big)
        return sequenced, not_sequenced, memory_objs

    def cleanup_mem_sequenced(self, item: tuple[list[tuple[CacheEngineKey, MemoryObj]], MemoryObj]):
        entries, big = item
        self.split_memory_obj(big, [mo[1] for mo in entries])
        big.ref_count_down()

    def io_sequenced(self, item: tuple[list[tuple[CacheEngineKey, MemoryObj]], MemoryObj]):
        entries, big = item
        file_idx, offset, size = self.files[entries[0][0].to_string()]
        fd, not_identical = self.fd_pool[file_idx]
        logger.debug(f"in_sequence file_idx={file_idx}; entries={len(entries)}; offset={offset}; not_identical={not_identical}; shape={big.get_shape()}; size={len(big.byte_array)}")
        self.read_file_at_offset(big.byte_array, fd, offset if not_identical else offset * size)

    def load_one(self, item: tuple[CacheEngineKey, MemoryObj]):
        key, obj = item
        file_idx, offset, _ = self.files[key.to_string()]
        fd, not_identical = self.fd_pool[file_idx]
        logger.debug(f"not_sequenced file_idx={file_idx}; offset={offset}; not_identical={not_identical}")
        self.read_file_at_offset(obj.byte_array, fd, offset if not_identical else offset * len(obj.byte_array))

    def cleanup_get(
        self,
        sequenced: List[List[tuple[CacheEngineKey, MemoryObj]]],
        keys: List[CacheEngineKey],
    ):
        for i in range(len(sequenced)):
            self.cleanup_mem_sequenced(sequenced[i])

        with self.disk_lock:
            for key in keys:
                self.dict[key].unpin()

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        return self.batched_get_blocking([key])[0]

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        sequenced, not_sequenced, memory_objs = self.prepare_get(keys)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=64) as executor:
            futs = []
            futs += [executor.submit(self.io_sequenced, item) for item in sequenced]
            futs += [executor.submit(self.load_one, item) for item in not_sequenced]
            for f in futs:
                f.result() 
        end = time.perf_counter()

        self.cleanup_get(sequenced, keys)

        total_size = sum(len(mo.byte_array) for mo in memory_objs if mo is not None)
        runtime = end - start
        mb = total_size / 1e6
        logger.info(f"Took {runtime:.2f} s for {mb} MB, Read bandwidth: {mb/runtime} MB/s")

        return memory_objs

    def group_entries_by_sequence(
        self,
        entries: List[tuple[CacheEngineKey, MemoryObj]],
    ):
        sequenced_groups: List[List[tuple[CacheEngineKey, MemoryObj]]] = []
        non_sequenced: List[tuple[CacheEngineKey, MemoryObj]] = []

        file_entries: List[tuple[int, int, CacheEngineKey, MemoryObj]] = []
        for key, mem_obj in entries:
            key_str = key.to_string()
            if key_str not in self.files:
                non_sequenced.append((key, mem_obj))
                continue
            file_idx, offset, _ = self.files[key_str]
            file_entries.append((file_idx, offset, key, mem_obj))

        from collections import defaultdict
        by_file: dict[int, List[tuple[int, CacheEngineKey, MemoryObj]]] = defaultdict(list)
        for file_idx, offset, key, mem_obj in file_entries:
            by_file[file_idx].append((offset, key, mem_obj))

        for file_idx, file_list in by_file.items():
            file_list.sort(key=lambda x: x[0])
            current_group: List[tuple[CacheEngineKey, MemoryObj]] = []
            last_offset: Optional[int] = None
            for offset, key, mem_obj in file_list:
                if last_offset is None or offset == last_offset + 1:
                    current_group.append((key, mem_obj))
                else:
                    if len(current_group) > 1:
                        sequenced_groups.append(current_group)
                    else:
                        non_sequenced.extend(current_group)
                    current_group = [(key, mem_obj)]
                last_offset = offset
            if len(current_group) > 1:
                sequenced_groups.append(current_group)
            else:
                non_sequenced.extend(current_group)
        return sequenced_groups, non_sequenced

    def reassemble(
        self,
        sequenced_groups: List[List[tuple[CacheEngineKey, MemoryObj]]],
        non_sequenced: List[tuple[CacheEngineKey, MemoryObj]],
        original_keys: List[CacheEngineKey],
    ) -> List[tuple[CacheEngineKey, MemoryObj]]:
        entries_by_key: Dict[CacheEngineKey, tuple[CacheEngineKey, MemoryObj]] = {}

        for group in sequenced_groups:
            for key, mem_obj in group:
                entries_by_key[key] = (key, mem_obj)

        for key, mem_obj in non_sequenced:
            entries_by_key[key] = (key, mem_obj)

        reassembled: List[tuple[CacheEngineKey, MemoryObj]] = []
        for key in original_keys:
            if key not in entries_by_key:
                raise KeyError(f"Key {key} not found in grouped entries during reassembly.")
            reassembled.append(entries_by_key[key])

        return reassembled

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        sequenced, not_sequenced, memory_objs = self.prepare_get(keys)

        start = time.perf_counter()
        tasks: list[asyncio.Future] = []
        for item in sequenced:
            tasks.append(self.disk_worker.submit_task("prefetch", self.io_sequenced, item))
        for item in not_sequenced:
            tasks.append(self.disk_worker.submit_task("prefetch", self.load_one, item))

        await asyncio.gather(*tasks)
        end = time.perf_counter()

        self.cleanup_get(sequenced, keys)

        total_size = sum(len(mo.byte_array) for mo in memory_objs if mo is not None)
        runtime = end - start
        mb = total_size / 1e6
        logger.info(f"Took {runtime:.2f} s for {total_size} MB, Read bandwidth: {mb/runtime} MB/s")

        return memory_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    def is_batch_not_identical(self, memory_objs: Sequence[MemoryObj]) -> bool:
        ref_shape = memory_objs[0].get_shape()
        mismatch_found = False
        for mo in memory_objs:
            shape = mo.get_shape()
            if shape != ref_shape:
                mismatch_found = True

        return mismatch_found

    def save_batched_bytes_to_disk_one_by_one(self, data: Sequence[tuple[CacheEngineKey, MemoryObj]], fd: int):
        offset = 0
        for _, memory_obj in data:
            buf = memory_obj.byte_array
            size = len(buf)
            self.write_file_at_offset(buf, fd, offset)
            offset += size

    def save_batched_bytes_to_disk(self, data: Sequence[tuple[CacheEngineKey, MemoryObj]], fd: int):
        memory_objs = [mo for _, mo in data]
        combined_obj = self.combine_memory_objs(memory_objs)
        logger.debug(f"Writing shape={combined_obj.get_shape()} len={len(memory_objs)}")
        self.write_file_at_offset(combined_obj.byte_array, fd, 0)
        combined_obj.ref_count_down()

    def combine_memory_objs(
        self,
        memory_objs: Sequence[MemoryObj],
    ) -> MemoryObj:
        assert memory_objs, "memory_objs must be non-empty"

        dtype = memory_objs[0].get_dtype()
        fmt = memory_objs[0].get_memory_format()
        orig_shape = memory_objs[0].get_shape()
        new_shape = torch.Size((len(memory_objs), *orig_shape))

        big = self.local_cpu_backend.allocate(new_shape, dtype, fmt)
        assert big is not None
        big_tensor = big.raw_data

        big_tensor = big_tensor.view(len(memory_objs), -1)

        for i, mo in enumerate(memory_objs):
            t = mo.raw_data
            t_flat = t.view(-1)
            assert t_flat.numel() == big_tensor.shape[1]
            big_tensor[i].copy_(t_flat)

        return big

    def split_memory_obj(
        self,
        big: MemoryObj,
        small_objs: Sequence[MemoryObj],
    ) -> None:
        big_tensor = big.raw_data
        assert big_tensor is not None, "Big MemoryObj has no raw_data"

        num_chunks = len(small_objs)
        assert num_chunks > 0, "split_memory_obj: small_objs must be non-empty"

        first_small = small_objs[0]
        small_tensor = first_small.raw_data
        assert small_tensor is not None, "First small MemoryObj has no raw_data"

        chunk_elems = small_tensor.numel()
        total_elems = big_tensor.numel()

        assert total_elems == num_chunks * chunk_elems, (
            f"Mismatch between big tensor size ({total_elems}) and "
            f"num_chunks * chunk_elems ({num_chunks} * {chunk_elems} = {num_chunks * chunk_elems})."
        )

        big_tensor_2d = big_tensor.view(num_chunks, chunk_elems)

        for i, mo in enumerate(small_objs):
            t = mo.raw_data
            assert t is not None, f"Small MemoryObj at index {i} has no raw_data"
            assert t.numel() == chunk_elems, (
                f"Small chunk {i} numel mismatch: "
                f"{t.numel()} vs {chunk_elems}"
            )

            dst_flat = t.view(-1)
            src_flat = big_tensor_2d[i]
            dst_flat.copy_(src_flat)

    def write_file_at_offset(self, buffer, fd: int, offset: int) -> None:
        start_time = time.time()
        size = len(buffer)

        size_aligned = (size % self.os_disk_bs == 0)
        offset_aligned = (offset % self.os_disk_bs == 0)
        use_odirect_now = self.use_odirect and size_aligned and offset_aligned

        if self.use_odirect and not use_odirect_now:
            logger.warning(
                "Cannot use O_DIRECT for this write, "
                "size and/or offset not aligned to disk block size "
                f"(size={size}, offset={offset}, block_size={self.os_disk_bs})."
            )

        os.pwrite(fd, buffer, offset)

        disk_write_time = time.time() - start_time
        logger.debug(
            f"Disk write size: {size} bytes at offset {offset}, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )

    def read_file_at_offset(self, buffer, fd: int, offset: int) -> None:
        start_time = time.perf_counter()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to disk block size."
            )

        ret = os.preadv(fd, [buffer], offset)
        if ret == -1:
            raise RuntimeError("preadv failed")
        if ret != size:
            raise RuntimeError(
                f"preadv returned {ret} bytes, expected {size} bytes at offset {offset}"
            )

        disk_read_time = time.perf_counter() - start_time
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        self.disk_worker.close()