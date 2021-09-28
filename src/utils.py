import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from PIL import Image, ImageOps


@njit  # necessario para rodar com outras funcoes
def transform_255(img: np.ndarray) -> np.ndarray:
    fm = img - np.min(img)
    fs = 255 * (fm / np.amax(fm))
    return fs


def plot_histogram(im: np.ndarray, title=None) -> None:
    hist = np.zeros(256)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            value = im[y][x]
            hist[value] += 1
    plt.figure(figsize=(18, 6))
    plt.bar(np.arange(256), hist)
    if title:
        plt.title(title, fontsize=20)
    plt.show()


def open_gray(path: str) -> tuple[Image.Image, np.ndarray]:
    img = ImageOps.grayscale(Image.open(path))
    im_array = np.array(img)
    return (img, im_array)
