import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from image_gen import make_test_image, add_salt_pepper
from median_native import median_filter_native
from median_lib import median_filter_lib
from benchmark import run_benchmark, check_match, plot_speed

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "..", "results", "images")
PLOT_DIR = os.path.join(BASE, "..", "results", "plots")

KERNELS_FOR_BENCH = [3, 5, 7, 9, 11]


def save_png(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def main():
    img = make_test_image()
    noisy = add_salt_pepper(img, amount=0.05)

    native_k3 = median_filter_native(noisy, 3)
    lib_k3 = median_filter_lib(noisy, 3)
    native_k9 = median_filter_native(noisy, 9)
    lib_k9 = median_filter_lib(noisy, 9)

    save_png(img, os.path.join(IMG_DIR, "original.png"))
    save_png(noisy, os.path.join(IMG_DIR, "noisy.png"))
    save_png(native_k3, os.path.join(IMG_DIR, "native_k3.png"))
    save_png(lib_k3, os.path.join(IMG_DIR, "lib_k3.png"))
    save_png(native_k9, os.path.join(IMG_DIR, "native_k9.png"))
    save_png(lib_k9, os.path.join(IMG_DIR, "lib_k9.png"))

    panels = [
        ("исходное", img),
        ("с шумом (5%)", noisy),
        ("нативный, k=3", native_k3),
        ("scipy, k=3", lib_k3),
        ("нативный, k=9", native_k9),
        ("scipy, k=9", lib_k9),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, (title, im) in zip(axes.flat, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "comparison.png"), dpi=130)
    plt.close()

    native_times, lib_times = run_benchmark(noisy, KERNELS_FOR_BENCH)
    plot_speed(KERNELS_FOR_BENCH, native_times, lib_times, os.path.join(PLOT_DIR, "speed.png"))

    max_diff, frac_diff = check_match(noisy, 5)

    print("размеры ядра:", KERNELS_FOR_BENCH)
    print("время нативной реализации, с:", [round(t, 4) for t in native_times])
    print("время scipy, с:", [round(t, 5) for t in lib_times])
    print("k=5, макс. отличие пикселя:", max_diff)
    print("k=5, доля несовпадающих пикселей:", round(frac_diff, 4))


if __name__ == "__main__":
    main()
