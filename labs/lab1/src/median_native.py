import numpy as np


def median_filter_native(img, k):
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    h, w = img.shape
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + k, j:j + k].flatten()
            window.sort()
            out[i, j] = window[len(window) // 2]

    return out
