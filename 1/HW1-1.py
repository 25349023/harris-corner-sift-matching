import functools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def normalize(data, lo, hi):
    return cv2.normalize(data, None, lo, hi, norm_type=cv2.NORM_MINMAX)


def gaussian_kernel(k, std):
    half = (k - 1) / 2
    raw_filter = sp.stats.norm.pdf(np.arange(-half, half + 1), scale=std)
    normalized = raw_filter / raw_filter.sum()
    return normalized


def apply_kernel(kernel, image):
    convolved = [sp.signal.convolve2d(image[..., ch], kernel, mode='same', boundary='symm')
                 for ch in range(image.shape[-1])]
    stacked = np.stack(convolved, -1)
    return stacked


def gaussian_blurrer(k_size, sigma):
    f = gaussian_kernel(k_size, sigma)
    kernel = np.outer(f, f)

    return functools.partial(apply_kernel, kernel)


def norm_for_showing(image):
    image = normalize(image, 0, 255)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)


def imshow(image, title=''):
    plt.imshow(norm_for_showing(image))
    plt.title(title)
    plt.show()


def show_multiple_image(images, titles):
    fig, ax = plt.subplots(1, len(images), figsize=(16, 10))
    for i, (image, t) in enumerate(zip(images, titles)):
        ax[i].imshow(norm_for_showing(image))
        ax[i].set_title(t)
    fig.show()


def show_then_write(images, titles):
    show_multiple_image(images, titles)
    for image, title in zip(images, titles):
        cv2.imwrite(f'output/{title}.jpg', normalize(image, 0, 255))


if __name__ == '__main__':
    img = cv2.imread('chessboard-hw1.jpg')
    img2 = cv2.imread('1a_notredame.jpg')
    show_multiple_image([img, img2], ['original chessboard', 'original_notredame'])

    blurrer5x5 = gaussian_blurrer(5, 5)
    blurrer10x10 = gaussian_blurrer(10, 5)

    chessboard_5x5 = blurrer5x5(img)
    chessboard_10x10 = blurrer10x10(img)
    show_then_write([chessboard_5x5, chessboard_10x10],
                    ['chessboard_blur5x5', 'chessboard_blur10x10'])

    notredame_5x5 = blurrer5x5(img2)
    notredame_10x10 = blurrer10x10(img2)
    show_then_write([notredame_5x5, notredame_10x10],
                    ['notredame_blur5x5', 'notredame_blur10x10'])

    plt.show()
