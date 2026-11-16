"""
mklgemm-grid-monitor (main.py)
Intel(R) oneAPI Math Kernel Library (oneMKL) cblas_sgemm Multi-Core Monitor
Multi-process architecture allocating 512 elements per logical core.
Dynamic scaling for all Intel CPUs, unified borderless top ribbon,
high-to-low controls, and 1-second sampled hardware GFLOPS telemetry.
"""

from __future__ import annotations
import os
import sys
import time
import ctypes
from pathlib import Path
import tkinter as tk
from tkinter import ttk

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import psutil
import matplotlib

# Remove lower Matplotlib navigation toolbar
matplotlib.rcParams["toolbar"] = "None"

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mkl_worker import MklGemmGridEngine

# ==============================================================================
# PALETTES (tf-living-pixels styling)
# ==============================================================================
CPU_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "cpu_heatmap_scale",
    ["#63BE7B", "#FFEB84", "#F8696B"],  # Green -> Yellow -> Red
)

ELEMENT_GRID_CMAP = LinearSegmentedColormap.from_list(
    "element_grid_scale",
    ["#040814", "#0A2540", "#00B4D8", "#90E0EF", "#FFF3B0"],
)


def excel_col_name(n: int) -> str:
    res = ""
    while n >= 0:
        res = chr(n % 26 + 65) + res
        n = n // 26 - 1
    return res


def _resolve_asset(rel_path: str) -> Path:
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / rel_path)
    candidates.append(Path(__file__).resolve().parent / rel_path)
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


class MklGemmMonitorViewer:
    def __init__(self, engine: MklGemmGridEngine):
        self.engine = engine
        self.num_cores = engine.num_cores
        self.total_elements = engine.total_elements
        self.window_closed = False

        # Performance tracking metrics
        self.last_fps_time = time.perf_counter()
        self.frame_count = 0
        self.fps = 0.0

        # Heatmap grid layout
        if self.num_cores >= 32:
            self.hm_rows, self.hm_cols = 4, 8
        elif self.num_cores >= 16:
            self.hm_rows, self.hm_cols = 4, 4
        elif self.num_cores >= 8:
            self.hm_rows, self.hm_cols = 2, 4
        else:
            self.hm_rows, self.hm_cols = 2, 2

        plt.ion()
        self.fig = plt.figure(figsize=(14.2, 7.6), dpi=100, facecolor="#FFFFFF")
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        # Plot layout giving clean space for top ribbon & formula bar
        gs = self.fig.add_gridspec(
            1, 2, width_ratios=[1.1, 1.1], left=0.04, right=0.98, top=0.74, bottom=0.06, wspace=0.15
        )
        self.ax_left = self.fig.add_subplot(gs[0, 0])
        self.ax_right = self.fig.add_subplot(gs[0, 1])

        # Excel-Style Formula Bar using full Intel oneAPI MKL cblas_sgemm routine name
        init_ram_mb = self.engine.ram_per_core_mb * self.num_cores
        init_ram_str = f"{init_ram_mb / 1024:.1f} GB" if init_ram_mb >= 1024 else f"{init_ram_mb} MB"
        self.formula_text = self.fig.text(
            0.015, 0.835,
            f"  fx  |  =INTEL.ONEMKL.CBLAS_SGEMM(CORES={self.num_cores}/{self.num_cores}, ELEMENTS_PER_CORE=512, RESIDENT_RAM={init_ram_str})",
            fontsize=9.0, color="#27272A", family="monospace",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="#FFFFFF", edgecolor="#D1D5DB", linewidth=1.0),
        )

        # --- LEFT PANEL: 2D cblas_sgemm State Grid ---
        self.ax_left.set_facecolor("#FFFFFF")
        init_grid = self.engine.get_visual_grid()
        self.im_left = self.ax_left.imshow(init_grid, cmap=ELEMENT_GRID_CMAP, interpolation="nearest", vmin=0, vmax=1)

        self.ax_left.set_title(
            f"2D cblas_sgemm State Grid ({self.total_elements:,} Total Elements)",
            fontsize=10.0, fontweight="bold", color="#27272A", pad=12,
        )
        for s in self.ax_left.spines.values():
            s.set_edgecolor("#9CA3AF")
            s.set_linewidth(1.5)

        grid_dim = init_grid.shape[0]
        step = max(1, grid_dim // 8)
        self.ax_left.set_xticks(range(0, grid_dim, step))
        self.ax_left.set_xticklabels([excel_col_name(i // step) for i in range(0, grid_dim, step)], fontsize=7.5, color="#27272A", fontweight="bold")
        self.ax_left.xaxis.tick_top()
        self.ax_left.set_yticks(range(0, grid_dim, step))
        self.ax_left.set_yticklabels([str(i + 1) for i in range(0, grid_dim, step)], fontsize=7.5, color="#27272A", fontweight="bold")

        # --- RIGHT PANEL: Universal CPU Logical Core Heatmap ---
        self.ax_right.set_facecolor("#FFFFFF")
        self.cpu_grid_data = np.zeros((self.hm_rows, self.hm_cols), dtype=np.float32)
        self.im_right = self.ax_right.imshow(
            self.cpu_grid_data, cmap=CPU_HEATMAP_CMAP, vmin=0, vmax=100, aspect="auto"
        )

        self.ax_right.set_xticks(np.arange(-0.5, self.hm_cols, 1), minor=True)
        self.ax_right.set_yticks(np.arange(-0.5, self.hm_rows, 1), minor=True)
        self.ax_right.grid(which="minor", color="#9CA3AF", linestyle="-", linewidth=1.2)
        self.ax_right.tick_params(which="minor", bottom=False, left=False)

        self.ax_right.set_xticks(range(self.hm_cols))
        self.ax_right.set_xticklabels([excel_col_name(c) for c in range(self.hm_cols)], fontsize=8.5, color="#27272A", fontweight="bold")
        self.ax_right.xaxis.tick_top()
        self.ax_right.set_yticks(range(self.hm_rows))
        self.ax_right.set_yticklabels([f"Bank {r + 1}" for r in range(self.hm_rows)], fontsize=8.5, color="#27272A", fontweight="bold")

        for s in self.ax_right.spines.values():
            s.set_edgecolor("#9CA3AF")
            s.set_linewidth(1.5)

        self.ax_right.set_title(f"Real-Time {self.num_cores}-Core CPU & RAM Load Map", fontsize=10.0, fontweight="bold", color="#27272A", pad=12)

        # Dynamic Core Labels
        self.cell_texts = []
        for r in range(self.hm_rows):
            row_t = []
            for c in range(self.hm_cols):
                tid = r * self.hm_cols + c
                lbl = f"CPU {tid:02d}\n0%" if tid < self.num_cores else "OFF"
                t = self.ax_right.text(
                    c, r, lbl, ha="center", va="center", color="#27272A", fontsize=8.0, fontweight="bold"
                )
                row_t.append(t)
            self.cell_texts.append(row_t)

        self._setup_tk_ribbon()
        self._apply_window_settings()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _setup_tk_ribbon(self):
        """Unified, borderless, 100% horizontally and vertically aligned top ribbon."""
        try:
            mgr = plt.get_current_fig_manager()
            if not hasattr(mgr, "window"):
                return

            win = mgr.window
            self.ribbon_frame = tk.Frame(win, bg="#FFFFFF", height=42, bd=0)
            self.ribbon_frame.place(relx=0.015, rely=0.012, relwidth=0.97, height=42)
            self.ribbon_frame.pack_propagate(False)

            # Left section: Hardware Detection + Official Intel Kit GEMM Routine
            left_sub = tk.Frame(self.ribbon_frame, bg="#FFFFFF")
            left_sub.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))

            tk.Label(
                left_sub,
                text=f"Cores detected: {self.num_cores}  |  512 elements per core  |  Intel oneAPI oneMKL cblas_sgemm",
                font=("Segoe UI", 8.5, "bold"),
                bg="#FFFFFF",
                fg="#1E293B",
            ).pack(side=tk.LEFT, anchor="center")

            # Middle section: High-to-Low Controls
            mid_sub = tk.Frame(self.ribbon_frame, bg="#FFFFFF")
            mid_sub.pack(side=tk.LEFT, fill=tk.Y, padx=(18, 0))

            tk.Label(mid_sub, text="Active Cores:", font=("Segoe UI", 8.5, "bold"), bg="#FFFFFF", fg="#1E293B").pack(side=tk.LEFT, anchor="center", padx=(0, 4))

            core_options = [f"All ({self.num_cores})"]
            for step in [24, 16, 8, 4, 2, 1]:
                if step < self.num_cores:
                    core_options.append(f"{step} Cores")

            self.core_var = tk.StringVar(value=core_options[0])
            self.core_combo = ttk.Combobox(mid_sub, textvariable=self.core_var, values=core_options, width=9, state="readonly")
            self.core_combo.pack(side=tk.LEFT, anchor="center", padx=(0, 14))
            self.core_combo.bind("<<ComboboxSelected>>", self._on_core_selected)

            tk.Label(mid_sub, text="RAM Load:", font=("Segoe UI", 8.5, "bold"), bg="#FFFFFF", fg="#1E293B").pack(side=tk.LEFT, anchor="center", padx=(0, 4))
            ram_options = ["512 MB/Core", "256 MB/Core", "128 MB/Core", "64 MB/Core", "16 MB/Core"]
            self.ram_var = tk.StringVar(value="128 MB/Core")
            self.ram_combo = ttk.Combobox(mid_sub, textvariable=self.ram_var, values=ram_options, width=12, state="readonly")
            self.ram_combo.pack(side=tk.LEFT, anchor="center", padx=(0, 8))
            self.ram_combo.bind("<<ComboboxSelected>>", self._on_ram_selected)

            # Right section: Borderless Status Telemetry
            right_sub = tk.Frame(self.ribbon_frame, bg="#FFFFFF")
            right_sub.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6))

            self.lbl_fps = tk.Label(
                right_sub, text="FPS: 0.0",
                font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#4338CA",
            )
            self.lbl_fps.pack(side=tk.RIGHT, anchor="center", padx=(10, 0))

            self.lbl_compute = tk.Label(
                right_sub, text="Compute: 0.0 GFLOPS",
                font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#0F766E",
            )
            self.lbl_compute.pack(side=tk.RIGHT, anchor="center", padx=(0, 6))

        except Exception as e:
            print(f"[UI Warning] Failed to attach Tk controls: {e}")

    def _on_core_selected(self, event=None):
        val = self.core_var.get()
        target_cores = self.num_cores if "All" in val else int(val.split()[0])
        self.engine.reconfigure(active_cores=target_cores)
        self._update_formula()

    def _on_ram_selected(self, event=None):
        val = self.ram_var.get()
        target_mb = int(val.split()[0])
        self.engine.reconfigure(active_cores=self.engine.active_cores_count, ram_mb=target_mb)
        self._update_formula()

    def _update_formula(self):
        active = self.engine.active_cores_count
        act_elements = active * self.engine.elements_per_core
        ram_mb = self.engine.ram_per_core_mb * active
        ram_str = f"{ram_mb / 1024:.1f} GB" if ram_mb >= 1024 else f"{ram_mb} MB"

        self.formula_text.set_text(
            f"  fx  |  =INTEL.ONEMKL.CBLAS_SGEMM(CORES={active}/{self.num_cores}, ELEMENTS_PER_CORE={self.engine.elements_per_core}, ACTIVE_ELEMENTS={act_elements}, RESIDENT_RAM={ram_str})"
        )

    def _apply_window_settings(self):
        """Set mklgemm-grid-monitor window title and apply Matrix Favicon."""
        try:
            mgr = plt.get_current_fig_manager()
            if hasattr(mgr, "toolbar") and mgr.toolbar is not None:
                mgr.toolbar.pack_forget()

            win = mgr.window
            win.title("mklgemm-grid-monitor — Intel(R) oneAPI Math Kernel Library cblas_sgemm")

            ico_path = _resolve_asset("assets/matrix_favicon.ico")
            if ico_path.exists():
                win.iconbitmap(str(ico_path))
                hwnd = ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id()
                hicon = ctypes.windll.user32.LoadImageW(0, str(ico_path), 1, 32, 32, 0x00000010)
                if hicon:
                    ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)
                    ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)
        except Exception:
            pass

    def _on_close(self, event):
        self.window_closed = True

    def update_frame(self):
        if self.window_closed or not plt.fignum_exists(self.fig.number):
            return False

        # 1. Update left 2D matrix state view
        grid = self.engine.get_visual_grid()
        self.im_left.set_data(grid)

        # 2. Measure per-core CPU utilization across all logical cores
        cpu_percentages = psutil.cpu_percent(interval=None, percpu=True)
        for r in range(self.hm_rows):
            for c in range(self.hm_cols):
                tid = r * self.hm_cols + c
                if tid < self.num_cores:
                    if self.engine.is_core_active(tid):
                        val = cpu_percentages[tid] if tid < len(cpu_percentages) else 0.0
                        self.cpu_grid_data[r, c] = val
                        self.cell_texts[r][c].set_text(f"CPU {tid:02d}\n{int(val)}%")
                        self.cell_texts[r][c].set_color("#27272A")
                    else:
                        self.cpu_grid_data[r, c] = 0.0
                        self.cell_texts[r][c].set_text(f"CPU {tid:02d}\nIDLE")
                        self.cell_texts[r][c].set_color("#94A3B8")
                else:
                    self.cpu_grid_data[r, c] = 0.0
                    self.cell_texts[r][c].set_text("OFF")
                    self.cell_texts[r][c].set_color("#CBD5E1")

        self.im_right.set_data(self.cpu_grid_data)

        # 3. Read live memory usage
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        proc_mb = proc.memory_info().rss / (1024 * 1024)
        active_c = self.engine.active_cores_count
        est_arena_gb = (active_c * self.engine.ram_per_core_mb) / 1024.0

        self.ax_right.set_xlabel(
            f"Sys RAM: {vm.percent:.1f}% ({vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f} GB) | Dedicated: ~{est_arena_gb:.1f} GB | UI RSS: {proc_mb:.0f} MB",
            fontsize=8.6, fontweight="bold", color="#1E293B", labelpad=8,
        )

        # 4. Read 1-second sampled GFLOPS + UI FPS
        gflops = self.engine.sample_gflops()
        now = time.perf_counter()
        dt = now - self.last_fps_time
        self.frame_count += 1
        if dt >= 1.0:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.last_fps_time = now

            if hasattr(self, "lbl_compute") and hasattr(self, "lbl_fps"):
                try:
                    self.lbl_compute.config(text=f"Compute: {gflops:.1f} GFLOPS")
                    self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")
                except Exception:
                    pass

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return True


def main():
    import multiprocessing
    multiprocessing.freeze_support()

    print("================================================================================")
    print("  mklgemm-grid-monitor | Intel(R) oneAPI Math Kernel Library (oneMKL) cblas_sgemm")
    print("================================================================================")

    engine = MklGemmGridEngine(elements_per_core=512, ram_per_core_mb=128)
    engine.start()

    viewer = MklGemmMonitorViewer(engine)

    try:
        while not viewer.window_closed:
            if not viewer.update_frame():
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        plt.close("all")


if __name__ == "__main__":
    main()
