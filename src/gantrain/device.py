"""Which accelerator to train on, what precision it allows, and process tuning."""

from __future__ import annotations

import contextlib
import functools
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

Backend = Literal["xpu", "cuda", "mps", "cpu"]


def detect_backend() -> Backend:
    """The best available accelerator. `GANTRAIN_DEVICE` overrides."""
    forced = os.environ.get("GANTRAIN_DEVICE", "").strip().lower()
    if forced:
        if forced not in ("xpu", "cuda", "mps", "cpu"):
            raise ValueError(f"GANTRAIN_DEVICE={forced!r} is not one of xpu/cuda/mps/cpu")
        return forced  # type: ignore[return-value]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _mod():
    """The torch submodule for the active accelerator, or None on the CPU."""
    backend = detect_backend()
    return getattr(torch, backend, None) if backend != "cpu" else None


@dataclass(frozen=True)
class DeviceCaps:
    backend: Backend
    name: str
    total_memory_gb: float
    #: What training autocasts to: bf16 where the GPU has it, else None for fp32. Never fp16,
    #: which would need a gradient scaler.
    autocast_dtype: torch.dtype | None

    @property
    def device(self) -> torch.device:
        return torch.device(self.backend)

    @property
    def training_precision(self) -> str:
        return "bf16 mixed precision" if self.autocast_dtype is torch.bfloat16 else "fp32"

    def summary(self) -> str:
        return "\n".join([
            f"backend   : {self.backend}",
            f"device    : {self.name}",
            f"memory    : {self.total_memory_gb:.1f} GB",
            f"training  : {self.training_precision}",
        ])


@functools.lru_cache(maxsize=1)
def get_caps() -> DeviceCaps:
    """Probe the active accelerator once."""
    backend = detect_backend()
    name, memory, bf16 = platform.processor() or "cpu", 0.0, False
    if backend in ("xpu", "cuda"):
        mod = getattr(torch, backend)
        props = mod.get_device_properties(0)
        name, memory = props.name, props.total_memory / 2**30
        bf16 = bool(mod.is_bf16_supported())
    elif backend == "mps":
        name = "Apple MPS"
    return DeviceCaps(backend, name, memory, torch.bfloat16 if bf16 else None)


def autocast(enabled: bool = True):
    """Mixed precision on the active device where it has bf16, else nothing."""
    caps = get_caps()
    if not enabled or caps.autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=caps.backend, dtype=caps.autocast_dtype)


def empty_cache() -> None:
    if (m := _mod()) is not None:
        m.empty_cache()


def reset_peak_memory() -> None:
    if (m := _mod()) is not None and hasattr(m, "reset_peak_memory_stats"):
        m.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    m = _mod()
    if m is None or not hasattr(m, "max_memory_allocated"):
        return 0.0
    return m.max_memory_allocated() / 1e9


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (m := _mod()) is not None and hasattr(m, "manual_seed_all"):
        m.manual_seed_all(seed)


def cpu_generator(seed: int) -> torch.Generator:
    """A seeded CPU generator, so latents drawn from it are the same on every device."""
    return torch.Generator(device="cpu").manual_seed(seed)


_HIGH_PRIORITY_CLASS = 0x00000080


def _raise_priority() -> str | None:
    """Put this process above the desktop's other work, in whatever the platform calls it."""
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_int
        if k32.SetPriorityClass(k32.GetCurrentProcess(), _HIGH_PRIORITY_CLASS):
            return "HIGH_PRIORITY_CLASS"
        raise OSError(f"SetPriorityClass failed (GetLastError={ctypes.get_last_error()})")
    # POSIX: a lower nice number is higher priority, and going below 0 needs privileges.
    os.nice(-5)
    return f"nice {os.nice(0)}"


def prioritise_gpu_feeder() -> None:
    """Raise this process's priority and, on a hybrid CPU, keep it on the performance cores,
    so the thread feeding the GPU is never parked on an efficiency core.

    Best-effort: whatever the platform refuses is left alone, and one line says what was done."""
    priority, note, cores = None, None, None
    try:
        priority = _raise_priority()
    except Exception as exc:  # noqa: BLE001 - tuning is best-effort everywhere
        note = f"priority unchanged: {exc}"

    try:
        import psutil

        # P-cores without asking the OS for a core's kind: with `p` hyperthreaded performance
        # cores and `e` efficiency cores, `logical = 2p + e` and `physical = p + e`, so
        # `p = logical - physical`, and both Windows and Linux enumerate P-core threads first.
        logical, physical = os.cpu_count() or 0, psutil.cpu_count(logical=False) or 0
        p_cores = (logical - physical) if logical > physical else 0
        fast = list(range(2 * p_cores))
        if not hasattr(psutil.Process(), "cpu_affinity"):
            note = f"no affinity control on {sys.platform}"
        elif 0 < len(fast) < logical:
            psutil.Process().cpu_affinity(fast)
            cores = fast
        else:
            note = "not a hybrid CPU; affinity left alone"
    except Exception as exc:  # noqa: BLE001 - psutil's own errors are not enumerable
        note = f"affinity unchanged: {exc}"

    said = (f"pinned to {len(cores)} P-core threads (0-{cores[-1]}) of {os.cpu_count()}"
            if cores else "left where the scheduler puts it")
    print(f"cpu: {said}" + (f", priority {priority}" if priority else "")
          + (f"  ({note})" if note else ""), flush=True)


def prepare_compile(cache: str = "data/cache/inductor", threads: int = 8) -> None:
    """Set up `torch.compile`: a C compiler on the path, a kernel cache that survives between
    sessions, and parallel kernel builds where the platform allows them."""
    if sys.platform == "win32":
        _activate_msvc()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath(cache))
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    if sys.platform != "win32":
        from torch._inductor import config as inductor_config

        inductor_config.compile_threads = int(
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(threads)))


#: The variables `vcvars64.bat` sets that the compiler needs.
_MSVC_VARS = {"PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "VCTOOLSINSTALLDIR",
              "VCTOOLSVERSION", "WINDOWSSDKDIR", "WINDOWSSDKVERSION", "WINDOWSSDKVERBINPATH",
              "UCRTVERSION", "UNIVERSALCRTSDKDIR", "VSINSTALLDIR"}


def _activate_msvc() -> bool:
    """Put Visual Studio's C++ compiler on this process's path, if it is installed.

    Inductor builds kernels with `cl.exe`, which is only on the path inside a Visual Studio
    developer shell; this imports that shell's environment instead."""
    if shutil.which("cl"):
        return True
    vswhere = (Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.exists():
        return False
    found = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True).stdout.strip().splitlines()
    vcvars = Path(found[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if found else None
    if vcvars is None or not vcvars.exists():
        return False
    out = subprocess.run(f'call "{vcvars}" && set', shell=True, capture_output=True,
                         text=True, errors="replace").stdout
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.upper() in _MSVC_VARS:
            if key.upper() == "PATH":
                value = value + os.pathsep + os.environ.get("PATH", "")
            os.environ[key] = value
    return shutil.which("cl") is not None
