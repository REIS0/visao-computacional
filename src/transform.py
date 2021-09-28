import math

import numpy as np
from numba import njit

from src.utils import transform_255

def eq_histogram(img: np.ndarray) -> np.ndarray:
    hist = np.zeros(256)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            value = img[y][x]
            hist[value] += 1

    area = img.shape[0] * img.shape[1]
    eq_hist = np.zeros(256)
    tmp = 0
    for i in range(256):
        tmp += 255 * (hist[i] / area)
        eq_hist[i] = math.floor(tmp)

    im_eq = np.empty(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            im_eq[y][x] = eq_hist[img[y][x]]
    return im_eq.astype(np.uint8)


def gama(img: np.ndarray, c: float = 1, g: float = 0.5) -> np.ndarray:
    im_gama = c * np.power(img, g)
    im_gama = transform_255(im_gama)
    return im_gama.astype(np.uint8)


# @njit
def binarizar(
    img: np.ndarray, thresh_max: int, thresh_min=None, invert=False
) -> np.ndarray:
    im_bin = np.empty(img.shape, dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] > thresh_max:
                im_bin[y][x] = 0
            else:
                if thresh_min is not None:
                    if img[y][x] < thresh_min:
                        im_bin[y][x] = 0
                    else:
                        im_bin[y][x] = 1
                else:
                    im_bin[y][x] = 1
    if invert:
        im_bin = 1 ^ im_bin
    return im_bin
