import os
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import List, Literal, Optional

from ..import_utils import (
    is_amdsmi_available,
    is_pynvml_available,
    is_pyrsmi_available,
    is_torch_available,
    is_torch_distributed_available,
)
from ..system_utils import is_nvidia_system, is_rocm_system

if is_rocm_system() and is_pyrsmi_available():
    from pyrsmi import rocml

if is_torch_distributed_available():
    import torch.distributed

if is_nvidia_system() and is_pynvml_available():
    import pynvml

if is_rocm_system() and is_amdsmi_available():
    import amdsmi  # type: ignore

if is_torch_available():
    import torch

import psutil

LOGGER = getLogger("memory")

MEMORY_UNIT = "MB"
Memory_Unit_Literal = Literal["MB"]


@dataclass
class Memory:
    unit: Memory_Unit_Literal

    max_ram: float
    max_global_vram: Optional[float] = None
    max_process_vram: Optional[float] = None
    max_reserved: Optional[float] = None
    max_allocated: Optional[float] = None

    @staticmethod
    def aggregate(memories: List["Memory"]) -> "Memory":
        if len(memories) == 0:
            raise ValueError("No memory measurements to aggregate")
        elif any(memory is None for memory in memories):
            raise ValueError("Some memory measurements are missing")

        unit = memories[0].unit

        max_ram = sum(memory.max_ram for memory in memories)

        max_process_vram = (
            sum(memory.max_process_vram for memory in memories) if memories[0].max_process_vram is not None else None
        )

        max_reserved = sum(memory.max_reserved for memory in memories) if memories[0].max_reserved is not None else None
        max_allocated = (
            sum(memory.max_allocated for memory in memories) if memories[0].max_allocated is not None else None
        )

        max_global_vram = (
            max(memory.max_global_vram for memory in memories) if memories[0].max_global_vram is not None else None
        )

        return Memory(
            unit=unit,
            max_ram=max_ram,
            max_global_vram=max_global_vram,
            max_process_vram=max_process_vram,
            max_reserved=max_reserved,
            max_allocated=max_allocated,
        )

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} memory:")
        if self.max_ram is not None:
            LOGGER.info(f"\t\t\t- max RAM: {self.max_ram:f} ({self.unit})")
        if self.max_global_vram is not None:
            LOGGER.info(f"\t\t\t- max global VRAM: {self.max_global_vram:f} ({self.unit})")
        if self.max_process_vram is not None:
            LOGGER.info(f"\t\t\t- max process VRAM: {self.max_process_vram:f} ({self.unit})")
        if self.max_reserved is not None:
            LOGGER.info(f"\t\t\t- max reserved memory: {self.max_reserved:f} ({self.unit})")
        if self.max_allocated is not None:
            LOGGER.info(f"\t\t\t- max allocated memory: {self.max_allocated:f} ({self.unit})")


class MemoryTracker:
    def __init__(self, device: str, backend: str, device_ids: Optional[str] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids
        self.monitored_pid = os.getpid()
        self.uses_cuda_pytorch_allocator = self.device == "cuda" and self.backend == "pytorch"
        self.is_distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        LOGGER.info(f"\t+ Tracking RAM memory of process [{self.monitored_pid}]")

        if self.device == "cuda":
            if self.device_ids is None:
                raise ValueError("The CUDA device IDs must be provided when tracking VRAM memory.")

            LOGGER.info(f"\t+ Tracking VRAM memory of CUDA devices [{self.device_ids}]")
            self.device_ids = list(map(int, self.device_ids.split(",")))

        if self.uses_cuda_pytorch_allocator:
            self.num_pytorch_devices = torch.cuda.device_count()
            if len(self.device_ids) != self.num_pytorch_devices:
                raise ValueError(
                    "The number of target CUDA devices and Pytorch's CUDA device count do not match. "
                    f"Got {len(self.device_ids)} and {self.num_pytorch_devices} respectively."
                )
            LOGGER.info(f"\t+ Tracking Allocated/Reserved memory of {self.num_pytorch_devices} Pytorch CUDA devices")

        self.max_ram_memory = None
        self.max_global_vram_memory = None
        self.max_process_vram_memory = None
        self.max_reserved_memory = None
        self.max_allocated_memory = None

    def reset(self):
        self.max_ram_memory = None
        self.max_global_vram_memory = None
        self.max_process_vram_memory = None
        self.max_reserved_memory = None
        self.max_allocated_memory = None

    @contextmanager
    def track(self):
        if self.is_distributed:
            torch.distributed.barrier()

        if self.uses_cuda_pytorch_allocator:
            yield from self._cuda_pytorch_memory()
        elif self.device == "cuda":
            yield from self._cuda_memory()
        else:
            yield from self._cpu_memory()

        if self.is_distributed:
            torch.distributed.barrier()

    def _cuda_pytorch_memory(self):
        self.max_allocated_memory = 0
        self.max_reserved_memory = 0

        torch.cuda.synchronize()

        for device in range(self.num_pytorch_devices):
            try:
                torch.cuda.reset_peak_memory_stats(device=device)
            except Exception as e:
                LOGGER.warning(f"\t\t+ Could not reset max memory stats for device {device}: {e}")

        yield from self._cuda_memory()

        torch.cuda.synchronize()

        for device in range(self.num_pytorch_devices):
            try:
                self.max_allocated_memory += torch.cuda.max_memory_allocated(device=device) / 1e6
                self.max_reserved_memory += torch.cuda.max_memory_reserved(device=device) / 1e6
            except Exception as e:
                LOGGER.warning(f"\t\t+ Could not get max memory stats for device {device}: {e}")

    def _cuda_memory(self):
        child_connection, parent_connection = Pipe()

        memory_process = Process(
            target=monitor_gpu_vram_memory, args=(self.monitored_pid, self.device_ids, child_connection), daemon=True
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield from self._cpu_memory()

        parent_connection.send(True)
        self.max_global_vram_memory = parent_connection.recv()
        self.max_process_vram_memory = parent_connection.recv()

    def _cpu_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_cpu_ram_memory, args=(self.monitored_pid, child_connection), daemon=True
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield

        parent_connection.send(True)
        self.max_ram_memory = parent_connection.recv()

    def get_max_memory(self):
        return Memory(
            unit=MEMORY_UNIT,
            max_ram=self.max_ram_memory,
            max_global_vram=self.max_global_vram_memory,
            max_process_vram=self.max_process_vram_memory,
            max_reserved=self.max_reserved_memory,
            max_allocated=self.max_allocated_memory,
        )


def monitor_cpu_ram_memory(monitored_pid: int, connection: Connection, interval: float = 0.001):
    stop = False
    max_used_memory = 0
    process = psutil.Process(monitored_pid)
    connection.send(0)

    while not stop:
        meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        used_memory = getattr(process, meminfo_attr)()[0]
        max_used_memory = max(max_used_memory, used_memory)
        stop = connection.poll(interval)

    try:
        connection.send(max_used_memory / 1e6)  # convert to MB
    except Exception as e:
        LOGGER.warning(f"Could not send memory stats: {e}")
    finally:
        connection.close()


def monitor_gpu_vram_memory(monitored_pid: int, device_ids: List[int], connection: Connection, interval: float = 0.01):
    stop = False
    max_used_global_memory = 0
    max_used_process_memory = 0
    monitored_process = psutil.Process(monitored_pid)
    connection.send(0)

    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )

        pynvml.nvmlInit()
        devices_handles = [pynvml.nvmlDeviceGetHandleByIndex(device_id) for device_id in device_ids]

        while not stop:
            used_global_memory = 0
            used_process_memory = 0

            monitored_pids = [monitored_pid] + [child.pid for child in monitored_process.children(recursive=True)]

            for device_id, device_handle in zip(device_ids, devices_handles):
                try:
                    device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get process list for device {device_id}: {e}.")
                    continue

                for device_process in device_processes:
                    if device_process.pid in monitored_pids:
                        used_process_memory += device_process.usedGpuMemory

                try:
                    device_memory = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get memory info for device {device_id}: {e}.")
                    continue

                used_global_memory += device_memory.used

            max_used_global_memory = max(max_used_global_memory, used_global_memory)
            max_used_process_memory = max(max_used_process_memory, used_process_memory)
            stop = connection.poll(interval)

        pynvml.nvmlShutdown()

    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library AMD SMI is required to track process-specific memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained AMD SMI library from https://github.com/ROCm/amdsmi."
            )
        if not is_pyrsmi_available():
            raise ValueError(
                "The library PyRSMI is required to track global-device memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained PyRSMI library from https://github.com/ROCm/pyrsmi."
            )

        amdsmi.amdsmi_init()
        rocml.smi_initialize()
        devices_handles = amdsmi.amdsmi_get_processor_handles()

        while not stop:
            used_global_memory = 0
            used_process_memory = 0

            monitored_pids = [monitored_pid] + [child.pid for child in monitored_process.children(recursive=True)]

            for device_id in device_ids:
                device_handle = devices_handles[device_id]
                try:
                    processes_handles = amdsmi.amdsmi_get_gpu_process_list(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get process list for device {device_id}: {e}")
                    continue

                for process_handle in processes_handles:
                    try:
                        gpu_process_info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get process info for process {process_handle}: {e}")
                        continue

                    if gpu_process_info["pid"] in monitored_pids:
                        max_used_process_memory += gpu_process_info["memory_usage"]["vram_mem"]

                try:
                    used_global_memory += rocml.smi_get_device_memory_used(device_id)
                except Exception as e:
                    LOGGER.warning(f"Could not get memory usage for device {device_id}: {e}")

            max_used_global_memory = max(max_used_global_memory, used_global_memory)
            max_used_process_memory = max(max_used_process_memory, used_process_memory)
            stop = connection.poll(interval)

        amdsmi.amdsmi_shut_down()
        rocml.smi_shutdown()

    else:
        raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for VRAM tracking.")

    try:
        connection.send(max_used_global_memory / 1e6)  # convert to MB
        connection.send(max_used_process_memory / 1e6)  # convert to MB
    except Exception as e:
        LOGGER.warning(f"Could not send memory stats: {e}")
    finally:
        connection.close()
