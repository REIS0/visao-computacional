import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image, ImageOps
from IPython.display import display


@njit  # necessario para rodar com outras funcoes
def transform_255(img: np.ndarray) -> np.ndarray:
    fm = img - np.min(img)
    fs = 255 * (fm / np.amax(fm))
    return fs


def plot_image(im: np.ndarray) -> None:
    display(Image.fromarray(im))


def plot_histogram(im: np.ndarray, title=None) -> None:
    hist = np.zeros(256)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            value = im[y][x]
            hist[value] += 1
    plt.figure(figsize=(18, 6))
    plt.xticks(np.arange(0, 256, 10))
    plt.bar(np.arange(256), hist)
    if title:
        plt.title(title, fontsize=20)
    plt.show()


@njit  # necessario para rodar com outras funcoes
def histogram_rgb(im: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_hist = np.zeros(256)
    g_hist = np.zeros(256)
    b_hist = np.zeros(256)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            value = im[y][x]
            r_hist[value[0]] += 1
            g_hist[value[1]] += 1
            b_hist[value[2]] += 1
    return (r_hist, g_hist, b_hist)


def histogram_hsi(im: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_hist = np.zeros(256)
    s_hist = np.zeros(256)
    i_hist = np.zeros(256)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            value = im[y][x]
            h_hist[value[0]] += 1
            s_hist[value[1]] += 1
            i_hist[value[2]] += 1
    return (h_hist, s_hist, i_hist)


def plot_histogram_rgb(im: np.ndarray, title=None) -> None:
    r_hist, g_hist, b_hist = histogram_rgb(im)
    plt.figure(figsize=(18, 6))
    x_axis = np.arange(256)
    plt.xticks(np.arange(0, 256, 10))
    plt.plot(x_axis, r_hist, color="red", alpha=0.6)
    plt.plot(x_axis, g_hist, color="green", alpha=0.6)
    plt.plot(x_axis, b_hist, color="blue", alpha=0.6)
    plt.legend(["Red", "Green", "Blue"])
    if title:
        plt.title(title, fontsize=20)
    plt.show()


def open_gray(path: str) -> tuple[Image.Image, np.ndarray]:
    img = ImageOps.grayscale(Image.open(path))
    im_array = np.array(img)
    return (img, im_array)


def open_img(path: str) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(path)
    im_array = np.array(img)
    return (img, im_array)
