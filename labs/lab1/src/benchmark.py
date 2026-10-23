import time
import numpy as np
import matplotlib.pyplot as plt

from median_native import median_filter_native
from median_lib import median_filter_lib

COLOR_NATIVE = "#D55E00"
COLOR_LIB = "#0072B2"


def time_it(func, img, k, repeats=3):
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(img, k)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    return best


def run_benchmark(img, kernel_sizes):
    native_times = []
    lib_times = []
    for k in kernel_sizes:
        native_times.append(time_it(median_filter_native, img, k))
        lib_times.append(time_it(median_filter_lib, img, k, repeats=20))
    return native_times, lib_times


def check_match(img, k):
    native = median_filter_native(img, k).astype(int)
    lib = median_filter_lib(img, k).astype(int)
    diff = native - lib
    return int(np.abs(diff).max()), float(np.mean(diff != 0))


def plot_speed(kernel_sizes, native_times, lib_times, path):
    plt.figure(figsize=(6, 4))
    plt.plot(kernel_sizes, native_times, marker="o", color=COLOR_NATIVE, label="нативная (Python)")
    plt.plot(kernel_sizes, lib_times, marker="o", color=COLOR_LIB, label="scipy.ndimage")
    plt.yscale("log")
    plt.xlabel("размер ядра")
    plt.ylabel("время, с (лог. шкала)")
    plt.title("Быстродействие медианного фильтра")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
