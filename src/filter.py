import math

import numpy as np
from numba import njit

from src.utils import transform_255


@njit
def max_filter(img: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    im_restored = np.empty(img.shape)
    h, w = img.shape[:2]
    wy = math.floor(kernel_size[0] / 2)
    wx = math.floor(kernel_size[1] / 2)
    for y in range(h):
        for x in range(w):
            tmp = 0
            for i in range(-wy, wy + 1):
                vy = y + i
                for j in range(-wx, wx + 1):
                    vx = x + j
                    if 0 <= vy < h and 0 <= vx < w:
                        if img[vy][vx] > tmp:
                            tmp = img[vy][vx]
            im_restored[y][x] = tmp
    return im_restored.astype(np.uint8)


@njit
def min_filter(img: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    im_restored = np.empty(img.shape)
    h, w = img.shape[:2]
    wy = math.floor(kernel_size[0] / 2)
    wx = math.floor(kernel_size[1] / 2)
    for y in range(h):
        for x in range(w):
            tmp = 0
            for i in range(-wy, wy + 1):
                vy = y + i
                for j in range(-wx, wx + 1):
                    vx = x + j
                    if 0 <= vy < h and 0 <= vx < w:
                        if img[vy][vx] < tmp:
                            tmp = img[vy][vx]
            im_restored[y][x] = tmp
    return im_restored.astype(np.uint8)


@njit
def arith_mean(img: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    im_restored = np.empty(img.shape)
    area = kernel_size[0] * kernel_size[1]
    h, w = img.shape[:2]
    wy = math.floor(kernel_size[0] / 2)
    wx = math.floor(kernel_size[1] / 2)
    for y in range(h):
        for x in range(w):
            tmp = 0
            for i in range(-wy, wy + 1):
                vy = y + i
                for j in range(-wx, wx + 1):
                    vx = x + j
                    tmp += img[vy][vx] if 0 <= vy < h and 0 <= vx < w else 0
            im_restored[y][x] = (1 / area) * tmp
    im_restored = transform_255(im_restored)
    return im_restored.astype(np.uint8)


def median(img: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    im_restored = np.empty(img.shape)
    h, w = img.shape[:2]
    wy = math.floor(kernel_size[0] / 2)
    wx = math.floor(kernel_size[1] / 2)
    for y in range(h):
        for x in range(w):
            tmp = []
            for i in range(-wy, wy + 1):
                vy = y + i
                for j in range(-wx, wx + 1):
                    vx = x + j
                    tmp.append(
                        img[vy][vx] if 0 <= vy < h and 0 <= vx < w else img[y][x]
                    )
            im_restored[y][x] = np.median(tmp)
    im_restored = transform_255(im_restored)
    return im_restored.astype(np.uint8)


@njit
def suavizar_box(img: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    img_new = np.empty(img.shape, dtype=np.uint8)
    # constante
    c = 1 / (kernel_size[0] * kernel_size[1])
    # normaliza para o centro do kernel
    wy = math.floor(kernel_size[0] / 2)
    wx = math.floor(kernel_size[1] / 2)
    for y in range(h):
        for x in range(w):
            pixel = 0
            for i in range(-wy, wy + 1):
                for j in range(-wx, wx + 1):
                    # zero padding
                    vy = y + i
                    vx = x + j
                    pixel += img[vy][vx] if 0 <= vy < h and 0 <= vx < w else 0
            img_new[y][x] = pixel * c
    return img_new


def contra_harm_mean(
    im: np.ndarray, kernel_size: tuple[int, int], q: int = 1
) -> np.ndarray:
    im_new = np.empty(im.shape, dtype=np.uint8)
    h, w = im.shape[:2]
    ky = math.floor(kernel_size[0]/2)
    kx = math.floor(kernel_size[1]/2)
    for y in range(h):
        for x in range(w):
            tmp1 = 0
            tmp2 = 0
            for i in range(-ky, ky + 1):
                vy = y + i
                for j in range(-kx, kx + 1):
                    vx = x + j
                    tmp1 += math.pow(im[vy][vx]+1 if 0 <= vy < h and 0 <= vx < w else 1, q+1)
                    tmp2 += math.pow(im[vy][vx]+1 if 0 <= vy < h and 0 <= vx < w else 1, q)
            im_new[y][x] = tmp1 / tmp2
    return im_new
