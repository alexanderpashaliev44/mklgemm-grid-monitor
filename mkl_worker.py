"""
mklgemm-grid-monitor — Intel(R) oneAPI Math Kernel Library (oneMKL) cblas_sgemm Engine
Pins dedicated worker processes to individual logical cores via Win32 API.
Executes direct single-precision GEMM via cblas_sgemm from mkl_rt.2.dll
(sequential threading layer per pinned core) with NumPy BLAS fallback.
Tracks exact 1-second floating-point operation counts (FLOPs) via shared counters.
"""

from __future__ import annotations
import os
import sys
import time
import ctypes
from pathlib import Path
import multiprocessing as mp
import numpy as np

# Enforce sequential BLAS execution inside each pinned core worker
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111
MKL_THREADING_SEQUENTIAL = 1


def _load_onemkl_sgemm():
    """
    Locate and load Intel(R) oneAPI Math Kernel Library (mkl_rt.2.dll)
    and bind cblas_sgemm with MKL_THREADING_SEQUENTIAL.
    Returns (cblas_sgemm_fn, library_label) or (None, fallback_label).
    """
    candidate_dirs = []
    if hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        candidate_dirs.extend([meipass, meipass / "mkl_bin"])

    base_dir = Path(__file__).resolve().parent
    candidate_dirs.extend([base_dir, base_dir / "mkl_bin"])

    mkl_root = os.environ.get("MKLROOT")
    if mkl_root:
        candidate_dirs.append(Path(mkl_root) / "bin")

    candidate_dirs.append(Path(r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin"))

    for d in candidate_dirs:
        dll_path = d / "mkl_rt.2.dll"
        if dll_path.exists():
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(d))
                mkl = ctypes.CDLL(str(dll_path))
                if hasattr(mkl, "MKL_Set_Threading_Layer"):
                    mkl.MKL_Set_Threading_Layer.argtypes = [ctypes.c_int]
                    mkl.MKL_Set_Threading_Layer.restype = ctypes.c_int
                    mkl.MKL_Set_Threading_Layer(MKL_THREADING_SEQUENTIAL)

                sgemm = mkl.cblas_sgemm
                sgemm.argtypes = [
                    ctypes.c_int,     # Layout
                    ctypes.c_int,     # TransA
                    ctypes.c_int,     # TransB
                    ctypes.c_int,     # M
                    ctypes.c_int,     # N
                    ctypes.c_int,     # K
                    ctypes.c_float,   # alpha
                    ctypes.c_void_p,  # A
                    ctypes.c_int,     # lda
                    ctypes.c_void_p,  # B
                    ctypes.c_int,     # ldb
                    ctypes.c_float,   # beta
                    ctypes.c_void_p,  # C
                    ctypes.c_int,     # ldc
                ]
                sgemm.restype = None

                ver = "Intel(R) oneAPI Math Kernel Library cblas_sgemm"
                if hasattr(mkl, "MKL_Get_Version_String"):
                    buf = ctypes.create_string_buffer(256)
                    mkl.MKL_Get_Version_String(buf, 256)
                    raw_ver = buf.value.decode(errors="ignore").strip()
                    if raw_ver:
                        ver = f"{raw_ver} | cblas_sgemm"
                return sgemm, ver
            except Exception:
                continue

    return None, "NumPy BLAS Fallback (mkl_rt.2.dll not found)"


def _core_worker_loop(
    core_id: int,
    elements_per_core: int,
    shared_state_arr,
    stop_event,
    active_mask_arr,
    ram_mb_val,
    cycle_counter_arr,
):
    """
    Dedicated worker process pinned directly to core_id.
    Executes Intel oneMKL cblas_sgemm when active_mask_arr[core_id] == 1.
    Dynamically scales resident RAM buffer based on ram_mb_val.value.
    Increments cycle_counter_arr[core_id] on each completed cblas_sgemm step.
    """
    # 1. Pin process strictly to this logical core using Win32 API
    try:
        mask = 1 << (core_id % 64)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
    except Exception:
        pass

    # 2. Load Intel(R) oneAPI Math Kernel Library cblas_sgemm inside worker
    cblas_sgemm, _ = _load_onemkl_sgemm()

    # 3. Allocate initial resident RAM arena
    current_ram_mb = max(16, int(ram_mb_val.value))
    arena_elements = (current_ram_mb * 1024 * 1024) // 4
    persistent_ram_arena = np.ones(arena_elements, dtype=np.float32)

    # 4. Contiguous float32 buffers for cblas_sgemm (512 elements per core)
    n = elements_per_core
    rng = np.random.default_rng(1337 + core_id)
    w_mat = np.ascontiguousarray(
        rng.normal(0.0, np.sqrt(2.0 / n), (n, n)).astype(np.float32)
    )
    w_trans = np.zeros((n, n), dtype=np.float32)
    state = np.ascontiguousarray(rng.uniform(0.0, 1.0, (n,)).astype(np.float32))
    next_s = np.zeros((n,), dtype=np.float32)

    w_mat_ptr = w_mat.ctypes.data
    w_trans_ptr = w_trans.ctypes.data
    state_ptr = state.ctypes.data
    next_s_ptr = next_s.ctypes.data

    shared_view = np.frombuffer(shared_state_arr, dtype=np.float32)
    start_idx = core_id * n
    end_idx = start_idx + n

    while not stop_event.is_set():
        if not bool(active_mask_arr[core_id]):
            time.sleep(0.05)
            continue

        # Dynamic RAM buffer resizing if user updated dropdown
        target_mb = max(16, int(ram_mb_val.value))
        if target_mb != current_ram_mb:
            current_ram_mb = target_mb
            arena_elements = (current_ram_mb * 1024 * 1024) // 4
            persistent_ram_arena = np.ones(arena_elements, dtype=np.float32)

        # Heavy Intel oneMKL cblas_sgemm:
        # 1. [512 x 512] @ [512 x 512] -> 2 * 512^3 = 268,435,456 FLOPs
        # 2. [1 x 512]   @ [512 x 512] -> 2 * 512^2 =     524,288 FLOPs
        # Total exact per step = 268,959,744 FLOPs
        if cblas_sgemm is not None:
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                n, n, n, 1.0, w_mat_ptr, n, w_mat_ptr, n, 0.0, w_trans_ptr, n
            )
            cblas_sgemm(
                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                1, n, n, 1.0, state_ptr, n, w_trans_ptr, n, 0.0, next_s_ptr, n
            )
        else:
            np.matmul(w_mat, w_mat, out=w_trans)
            np.matmul(state, w_trans, out=next_s)

        # In-place leaky activation keeps state_ptr stable
        np.clip(0.85 * state + 0.15 * next_s, 0.0, 1.0, out=state)

        # Publish element state to shared UI buffer
        shared_view[start_idx:end_idx] = state

        # Strided memory touch to exercise physical RAM bandwidth
        stride_idx = (int(state[0] * 1_000_000)) % max(1, arena_elements - n)
        persistent_ram_arena[stride_idx:stride_idx + n] += state * np.float32(0.001)

        cycle_counter_arr[core_id] += 1
        time.sleep(0.005)


class MklGemmGridEngine:
    """
    mklgemm-grid-monitor Multi-Core Engine:
    Detects logical cores N dynamically (32 on Xeon, 4 on i5, 8 on i7).
    Allocates exactly 512 elements per logical core (N * 512 total).
    """
    FLOPS_PER_STEP = 2 * (512 ** 3) + 2 * (512 ** 2)

    def __init__(self, elements_per_core: int = 512, ram_per_core_mb: int = 128):
        self.num_cores = os.cpu_count() or 4
        self.elements_per_core = elements_per_core
        self.total_elements = self.num_cores * elements_per_core
        self.active_cores_count = self.num_cores
        self.ram_per_core_mb = ram_per_core_mb

        _, self.backend_label = _load_onemkl_sgemm()

        self.shared_state = mp.RawArray(ctypes.c_float, self.total_elements)
        self.active_mask = mp.RawArray(ctypes.c_byte, [1] * self.num_cores)
        self.ram_val = mp.Value(ctypes.c_int, ram_per_core_mb)
        self.cycle_counters = mp.RawArray(ctypes.c_ulonglong, self.num_cores)

        self.last_sample_time = time.perf_counter()
        self.last_step_totals = 0
        self.current_gflops = 0.0

        self.stop_event = mp.Event()
        self.workers: list[mp.Process] = []

    def start(self):
        print(
            f"[mklgemm-grid-monitor] Cores detected: {self.num_cores} | "
            f"Backend: {self.backend_label} | Allocating {self.total_elements} Elements ({self.num_cores} x {self.elements_per_core})"
        )
        for c in range(self.num_cores):
            p = mp.Process(
                target=_core_worker_loop,
                args=(c, self.elements_per_core, self.shared_state, self.stop_event, self.active_mask, self.ram_val, self.cycle_counters),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        print(f"[mklgemm-grid-monitor] Spawned {len(self.workers)} pinned cblas_sgemm worker processes.")

    def reconfigure(self, active_cores: int, ram_mb: int | None = None):
        """Dynamically enable first `active_cores` and set RAM buffer size."""
        active_cores = max(1, min(active_cores, self.num_cores))
        self.active_cores_count = active_cores
        for c in range(self.num_cores):
            self.active_mask[c] = 1 if c < active_cores else 0

        if ram_mb is not None:
            self.ram_per_core_mb = ram_mb
            self.ram_val.value = ram_mb

    def is_core_active(self, core_id: int) -> bool:
        if 0 <= core_id < self.num_cores:
            return bool(self.active_mask[core_id])
        return False

    def sample_gflops(self) -> float:
        """
        Samples completed cblas_sgemm steps across all workers over a 1-second window.
        """
        now = time.perf_counter()
        dt = now - self.last_sample_time
        if dt >= 0.95:
            current_total_steps = sum(self.cycle_counters[i] for i in range(self.num_cores))
            steps_delta = current_total_steps - self.last_step_totals
            total_flops = steps_delta * self.FLOPS_PER_STEP
            self.current_gflops = (total_flops / dt) / 1e9
            self.last_step_totals = current_total_steps
            self.last_sample_time = now
        return self.current_gflops

    def get_visual_grid(self) -> np.ndarray:
        """Returns 2D matrix formatted for the left panel visualization."""
        arr = np.frombuffer(self.shared_state, dtype=np.float32)
        dim = int(np.ceil(np.sqrt(self.total_elements)))
        if dim * dim == self.total_elements:
            return arr.reshape((dim, dim))
        padded = np.pad(arr, (0, dim * dim - self.total_elements))
        return padded.reshape((dim, dim))

    def stop(self):
        self.stop_event.set()
        for p in self.workers:
            p.join(timeout=0.5)
            if p.is_alive():
                p.terminate()
