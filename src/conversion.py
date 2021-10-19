import math

import numpy as np
from numba import njit


@njit
def rgb2hsi(im: np.ndarray) -> np.ndarray:
    im_hsi = np.empty(im.shape)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            r_norm = im[y][x][0] / 255
            g_norm = im[y][x][1] / 255
            b_norm = im[y][x][2] / 255

            if r_norm == g_norm == b_norm:
                im_hsi[y][x][0] = 0
                im_hsi[y][x][1] = 0
                im_hsi[y][x][2] = np.uint8(1 / 3 * (r_norm + g_norm + b_norm))
            else:
                teta = math.acos(
                    (0.5 * ((r_norm - g_norm) + (r_norm - b_norm)))
                    / math.sqrt(
                        (r_norm - g_norm) ** 2 + (r_norm - b_norm) * (g_norm - b_norm)
                    )
                )
                h = teta if b_norm <= g_norm else math.pi * 2 - teta
                s = 1 - (3 / (r_norm + g_norm + b_norm)) * min([r_norm, g_norm, b_norm])
                i = 1 / 3 * (r_norm + g_norm + b_norm)
                im_hsi[y][x][0] = math.degrees(h)
                im_hsi[y][x][1] = s
                im_hsi[y][x][2] = np.uint8(i * 255)
    return im_hsi


@njit
def hsi2rgb(im: np.ndarray) -> np.ndarray:
    im_rgb = np.empty(im.shape)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            h = im[y][x][0]
            s = im[y][x][1]
            i = im[y][x][2]
            if 0 <= h < 120:  # 120
                r = i * (
                    1
                    + ((s * math.cos(math.radians(h))) / math.cos(math.radians(60 - h)))
                )
                b = i * (1 - s)
                g = 3 * i - (r + b)
                im_rgb[y][x][0] = r
                im_rgb[y][x][1] = g
                im_rgb[y][x][2] = b
            elif 120 <= h < 240:
                h -= 120
                r = i * (1 - s)
                g = i * (
                    1
                    + ((s * math.cos(math.radians(h))) / math.cos(math.radians(60 - h)))
                )
                b = 3 * i - (r + g)
                im_rgb[y][x][0] = r
                im_rgb[y][x][1] = g
                im_rgb[y][x][2] = b
            else:
                h -= 240
                g = i * (1 - s)
                b = i * (
                    1
                    + ((s * math.cos(math.radians(h))) / math.cos(math.radians(60 - h)))
                )
                r = 3 * i - (g + b)
                im_rgb[y][x][0] = r
                im_rgb[y][x][1] = g
                im_rgb[y][x][2] = b
    return im_rgb.astype(np.uint8)
