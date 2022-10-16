import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize(data, lo, hi):
    return cv2.normalize(data, None, lo, hi, norm_type=cv2.NORM_MINMAX)


def imshow(image):
    image = normalize(image, 0, 255)
    plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))


def sift_detect(image, n_keypoints):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(n_keypoints, 5, sigma=5)
    kp, descriptor = sift.detectAndCompute(gray, None)
    return kp, descriptor


if __name__ == '__main__':
    img_left = cv2.imread('1a_notredame.jpg')
    img_right = cv2.imread('1b_notredame.jpg')

    output_dir = pathlib.Path('output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    n_feat = 300
    left_kp, left_desc = sift_detect(img_left, n_feat)
    img_left_kp = cv2.drawKeypoints(img_left, left_kp, None)
    cv2.imwrite(str(output_dir / '1a_sift_keypoints.jpg'), img_left_kp)

    right_kp, right_desc = sift_detect(img_right, n_feat)
    img_right_kp = cv2.drawKeypoints(img_right, right_kp, None)
    cv2.imwrite(str(output_dir / '1b_sift_keypoints.jpg'), img_right_kp)

    idx = []
    for q, key in enumerate(left_desc):
        distance = np.linalg.norm(key - right_desc, axis=-1)
        sorted_idx = np.argsort(distance)
        for t0, t1 in zip(sorted_idx, sorted_idx[1:]):
            if distance[t0] < 400 and distance[t0] / distance[t1] < 0.7:
                idx.append([cv2.DMatch(q, t0, distance[t0])])
                print('accept: ', distance[t0], distance[t1])
                break

    print(f'totally {len(idx)} matches.')
    img = cv2.drawMatchesKnn(img_left, left_kp, img_right, right_kp, idx, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite(str(output_dir / 'keypoints_matching.jpg'), img)
