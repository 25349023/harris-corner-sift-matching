import functools
import pathlib

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


def gaussian_blur(image, k_size, sigma):
    f = gaussian_kernel(k_size, sigma)
    kernel = np.outer(f, f)
    return apply_kernel(kernel, image)


def sobel_gradient_field(image, threshold_ratio=0.0):
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


def direction_in_hsv(magnitude, direction):
    hsv = np.zeros(direction.shape, dtype=np.uint8)

    hsv[..., 0] = direction.mean(axis=-1) * 180 / np.pi / 2 + 90
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
    return eigen_minus


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


def draw_corner(image, corner):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    image = normalize(image, 0, 255)
    ax.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.scatter(corner.nonzero()[1], corner.nonzero()[0], c='r')
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[..., ::-1]

    return data


def harris_corner_detection(original_image, name):
    print(f'Harris corner detection for {name}')

    img_to_save = {}

    # Step 1: gaussian blur
    blurred_images = {'5x5': gaussian_blur(original_image, 5, 5),
                      '10x10': gaussian_blur(original_image, 10, 5)}
    img_to_save.update({f'{name}_blur_5x5': blurred_images['5x5'],
                        f'{name}_blur_10x10': blurred_images['10x10']})

    for b_size, bimg in blurred_images.items():
        # Step 2: Sobel edge detection
        grad, mag, dire = sobel_gradient_field(bimg, 0.1)
        dire_hsv = direction_in_hsv(mag, dire)
        img_to_save.update({f'{name}_{b_size}_grad_magnitude': mag,
                            f'{name}_{b_size}_grad_direction': dire_hsv})

        if b_size == '10x10':
            # Step 3: Compute the structure tensor and its smaller eigenvalue
            Hs = structure_tensor(grad)
            windows = {'3x3': summation_kernel(3), '5x5': summation_kernel(5)}

            for w_size, window in windows.items():
                eigen_minus = calc_eigen_minus(window, *Hs)
                # if we normalize to 0 ~ 10, we will get a brighter image
                eigen_minus = normalize(eigen_minus, 0, 1)

                img_to_save.update({f'{name}_eigen_{w_size}': eigen_minus})

                # Step 4: Non-maximal Suppression
                corners = non_maximal_suppression(eigen_minus, 15, 0.1)
                result = draw_corner(original_image, corners)
                img_to_save.update({f'{name}_corner_detection_{w_size}': result})

    return img_to_save


def similarity_transform(image, angle=0.0, scale=1.0):
    trans_mat = cv2.getRotationMatrix2D([image.shape[1] / 2, image.shape[0] / 2],
                                        angle, scale)
    transformed_image = cv2.warpAffine(image, trans_mat, None, flags=cv2.INTER_LINEAR)
    return transformed_image


if __name__ == '__main__':
    images_to_save = {}

    chessboard = cv2.imread('chessboard-hw1.jpg')
    notredame = cv2.imread('1a_notredame.jpg')

    images_to_save.update(harris_corner_detection(chessboard, 'chessboard'))
    images_to_save.update(harris_corner_detection(notredame, 'notredame'))

    normal_dir = pathlib.Path('output', 'normal')
    if not normal_dir.exists():
        normal_dir.mkdir(parents=True, exist_ok=True)
    for filename, img in images_to_save.items():
        cv2.imwrite(str(normal_dir / f'{filename}.jpg'), normalize(img, 0, 255))

    images_to_save.clear()

    chessboard_r30 = similarity_transform(chessboard, angle=30)
    chessboard_s05 = similarity_transform(chessboard, scale=0.5)
    notredame_r30 = similarity_transform(notredame, angle=30)
    notredame_s05 = similarity_transform(notredame, scale=0.5)

    images_to_save.update(harris_corner_detection(chessboard_r30, 'chessboard_rotate_30'))
    images_to_save.update(harris_corner_detection(chessboard_s05, 'chessboard_scale_05'))
    images_to_save.update(harris_corner_detection(notredame_r30, 'notredame_rotate_30'))
    images_to_save.update(harris_corner_detection(notredame_s05, 'notredame_scale_05'))

    transformed_dir = pathlib.Path('output', 'transformed')
    if not transformed_dir.exists():
        transformed_dir.mkdir(parents=True, exist_ok=True)
    for filename, img in images_to_save.items():
        cv2.imwrite(str(transformed_dir / f'{filename}.jpg'), normalize(img, 0, 255))
