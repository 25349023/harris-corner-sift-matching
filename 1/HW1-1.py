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
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    imshow(image, f'{name}_corner_detection_{size}')
    ax.scatter(corner.nonzero()[1], corner.nonzero()[0], c='r')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[..., ::-1]
    fig.show()

    return data


def corner_detection(original_image, name):
    img_to_save = {}

    # Step 1: gaussian smooth
    blurrer5x5 = gaussian_blurrer(5, 5)
    blurrer10x10 = gaussian_blurrer(10, 5)

    blurred_5x5 = blurrer5x5(original_image)
    blurred_10x10 = blurrer10x10(original_image)
    show_multiple_image([blurred_5x5, blurred_10x10],
                        [f'{name}_blur5x5', f'{name}_blur10x10'])
    img_to_save.update({f'{name}_blur5x5': blurred_5x5,
                        f'{name}_blur10x10': blurred_10x10})

    blurred_images = {f'5x5': blurred_5x5,
                      f'10x10': blurred_10x10}

    for suffix, bimg in blurred_images.items():
        # Step 2
        grad, mag, dire = gradient_field(bimg, 0.1)
        dire_rgb = direction_in_hue(mag, dire)
        show_multiple_image([mag, dire_rgb],
                            [f'{name}_{suffix}_grad_magnitude', f'{name}_{suffix}_grad_direction'])
        img_to_save.update({f'{name}_{suffix}_grad_magnitude': mag,
                            f'{name}_{suffix}_grad_direction': dire_rgb})

        if suffix == '10x10':
            # Step 3
            Hs = structure_tensor(grad)

            windows = {'3x3': summation_kernel(3), '5x5': summation_kernel(5)}

            for suffix, window in windows.items():
                eigen_minus, resp = calc_eigen_minus(window, *Hs)
                eigen_minus = normalize(eigen_minus, 0, 1)

                imshow(eigen_minus, f'{name}_eigen_{suffix}')
                img_to_save.update({f'{name}_eigen_{suffix}': eigen_minus})

                # Step 4
                corners = non_maximal_suppression(eigen_minus, 25, 0.15)
                result = show_corner(original_image, corners, suffix, name)
                img_to_save.update({f'{name}_corner_detection_{suffix}': result})

    return img_to_save


if __name__ == '__main__':
    images_to_save = {}

    chessboard = cv2.imread('chessboard-hw1.jpg')
    notredame = cv2.imread('1a_notredame.jpg')
    show_multiple_image([chessboard, notredame], ['original chessboard', 'original_notredame'])

    images_to_save.update(corner_detection(chessboard, 'chessboard'))
    images_to_save.update(corner_detection(notredame, 'notredame'))

    for filename, img in images_to_save.items():
        cv2.imwrite(f'output/normal/{filename}.jpg', normalize(img, 0, 255))
