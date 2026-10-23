import numpy as np


def make_test_image(size=128):
    x = np.linspace(0, 255, size)
    img = np.tile(x, (size, 1)).astype(np.uint8)

    yy, xx = np.mgrid[0:size, 0:size]
    circle = (xx - size * 0.3) ** 2 + (yy - size * 0.5) ** 2 < (size * 0.15) ** 2
    img[circle] = 40

    r0, r1 = int(size * 0.6), int(size * 0.85)
    c0, c1 = int(size * 0.55), int(size * 0.9)
    img[r0:r1, c0:c1] = 220

    return img


def add_salt_pepper(img, amount=0.05, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    mask = rng.random(img.shape) < amount
    salt = rng.random(img.shape) < 0.5
    noisy[mask & salt] = 255
    noisy[mask & ~salt] = 0
    return noisy
