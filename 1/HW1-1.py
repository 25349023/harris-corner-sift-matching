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


def gradient_field(image, threshold_ratio=0.0):
    sobel = (np.array([[1., 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8,
             np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8)  # (hy, hx)
    image = normalize(image, 0, 1)

    gradient = [apply_kernel(s, image) for s in sobel]
    magnitude = np.hypot(*gradient)
    direction = np.arctan2(*gradient)

    threshold = magnitude.max() * threshold_ratio
    mask = magnitude <= threshold
    gradient[0][mask] = 0.0
    gradient[1][mask] = 0.0
    direction[mask] = 0.0
    magnitude[mask] = 0.0
    return gradient, magnitude, direction


def direction_in_hue(magnitude, direction):
    hsv = np.zeros(direction.shape, dtype=np.uint8)

    hsv[..., 0] = direction.mean(axis=-1) * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = normalize(magnitude.mean(axis=-1), 0, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def show_gradient(magnitude, direction, name):
    dire_rgb = direction_in_hue(magnitude, direction)
    show_then_write([magnitude, dire_rgb],
                    [f'{name}_grad_magnitude', f'{name}_grad_direction'])


def summation_kernel(size):
    k = np.ones((size, size))
    return functools.partial(apply_kernel, k)


def structure_tensor(gradient):
    Ixx = gradient[1] ** 2
    Ixy = gradient[0] * gradient[1]
    Iyy = gradient[0] ** 2

    return Ixx, Ixy, Iyy


def calc_eigen_minus(window, Ixx, Ixy, Iyy):
    Ixx = window(Ixx)
    Ixy = window(Ixy)
    Iyy = window(Iyy)

    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy + 1e-5
    eigen_minus = det / trace
    response = det - 0.04 * (trace ** 2)
    return eigen_minus, response


def windowed_slices(idx, wr):
    x, y, ch = idx
    s = [(np.s_[max(u - wr, 0):u + wr + 1], np.s_[max(v - wr, 0):v + wr + 1], c)
         for u, v, c in zip(x, y, ch)]

    return s


def non_maximal_suppression(raw_corner: np.ndarray, w_radius=5, threshold=1e-2):
    raw_corner = raw_corner.copy()
    raw_corner[raw_corner < threshold] = 0
    result = np.zeros_like(raw_corner)

    flatten_cmap = raw_corner.reshape([-1, raw_corner.shape[-1]])

    while np.count_nonzero(raw_corner) > 0:
        pos = flatten_cmap.argmax(axis=0)
        unravel_pos = *np.unravel_index(pos, raw_corner.shape[:-1]), np.array([0, 1, 2], dtype=np.int64)
        result[unravel_pos] = 1
        neighbor_slices = windowed_slices(unravel_pos, w_radius)
        for neighbor in neighbor_slices:
            raw_corner[neighbor] = 0

    return result


def show_corner(image, corner, size, name):
    plt.figure(figsize=(10, 10))
    imshow(image, f'corner_detection_{size}')
    plt.scatter(corner.nonzero()[1], corner.nonzero()[0], c='r')
    plt.savefig(f'output/{name}_corner_detection_{size}.jpg')
    plt.show()


if __name__ == '__main__':
    chessboard = cv2.imread('chessboard-hw1.jpg')
    notredame = cv2.imread('1a_notredame.jpg')
    show_multiple_image([chessboard, notredame], ['original chessboard', 'original_notredame'])

    # Step 1
    blurrer5x5 = gaussian_blurrer(5, 5)
    blurrer10x10 = gaussian_blurrer(10, 5)

    chessboard_5x5 = blurrer5x5(chessboard)
    chessboard_10x10 = blurrer10x10(chessboard)
    show_then_write([chessboard_5x5, chessboard_10x10],
                    ['chessboard_blur5x5', 'chessboard_blur10x10'])

    notredame_5x5 = blurrer5x5(notredame)
    notredame_10x10 = blurrer10x10(notredame)
    show_then_write([notredame_5x5, notredame_10x10],
                    ['notredame_blur5x5', 'notredame_blur10x10'])

    plt.show()

    # Step 2
    images = {'chessboard_5x5': chessboard_5x5,
              'chessboard_10x10': chessboard_10x10,
              'notredame_5x5': notredame_5x5,
              'notredame_10x10': notredame_10x10}

    for name, img in images.items():
        grad, mag, dire = gradient_field(img, 0.1)
        show_gradient(mag, dire, name)

    images_10x10 = {'chessboard': (chessboard_10x10, chessboard),
                    'notredame': (notredame_10x10, notredame)}
    window3x3 = summation_kernel(3)
    window5x5 = summation_kernel(5)

    for name, (img, oimg) in images_10x10.items():
        # Step 3
        grad, mag, dire = gradient_field(img, 0.1)

        Hs = structure_tensor(grad)

        eigen_minus3x3, resp3x3 = calc_eigen_minus(window3x3, *Hs)
        eigen_minus3x3 = normalize(eigen_minus3x3, 0, 1)

        eigen_minus5x5, resp5x5 = calc_eigen_minus(window5x5, *Hs)
        eigen_minus5x5 = normalize(eigen_minus5x5, 0, 1)

        show_then_write([eigen_minus3x3, eigen_minus5x5], [f'{name}_eigen_3x3', f'{name}_eigen_5x5'])

        # Step 4
        corners3x3 = non_maximal_suppression(eigen_minus3x3, 25, 0.15)
        corners5x5 = non_maximal_suppression(eigen_minus5x5, 25, 0.15)
        show_corner(oimg, corners3x3, '3x3', name)
        show_corner(oimg, corners5x5, '5x5', name)


