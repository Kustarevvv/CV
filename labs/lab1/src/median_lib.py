from scipy.ndimage import median_filter


def median_filter_lib(img, k):
    return median_filter(img, size=k, mode="reflect")
