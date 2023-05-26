import numpy as np
import cv2

import os
import math


def slide_win_cut_imgs(img: np.ndarray, win_size: tuple = (10, 10), count: tuple = (10, 10)) -> np.ndarray or None:
    if len(img.shape) != 3 or len(win_size) != 2 or len(count) != 2:
        print("Wrong param shapes")
        print(len(img.shape))
        print(len(win_size))
        print(len(count))
        return None

    W, H, _ = img.shape
    win_x, win_y = win_size
    x_step = math.floor((W - win_x) / (count[0] - 1))
    y_step = math.floor((H - win_y) / (count[1] - 1))

    img_list = []
    for x_ in range(0, W, x_step):
        if (x_ + win_x) > W:
            continue
        for y_ in range(0, H, y_step):
            if (y_ + win_y) > H:
                continue
            img_list.append(img[x_:x_ + win_x, y_:y_ + win_y, :])
    return np.asarray(img_list)


def read_imgs(dir_path: str) -> np.ndarray:
    img_list = []
    for _, img_path in enumerate(os.listdir(dir_path)):
        image = cv2.imread(dir_path + img_path)
        image = image[:2048, :, :]
        if image.shape != (2048, 2048, 3):
            continue
        img_list.append(image)
    return np.asarray(img_list)


def save_imgs(imgs: np.ndarray, dir_path: str, prefix: str = 'img_0') -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(imgs.shape) == 4:
        for i, img in enumerate(imgs):  # imgs
            cv2.imwrite(dir_path + prefix + '_clip_' + str(i) + '.jpg', img)
    elif len(imgs.shape) == 3:  # single img
        cv2.imwrite(dir_path + prefix + '_clip_0.jpg', imgs)
    else:
        print("Wrong imgs shape")
