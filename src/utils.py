import numpy as np
import cv2

import os
import os.path as path
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


def read_imgs(dir_path: str, shape: tuple) -> np.ndarray | list:
    img_list = []
    name_list = []
    for _, img_path in enumerate(os.listdir(dir_path)):
        if path.isfile(path.join(dir_path, img_path)):
            image = cv2.imread(path.join(dir_path, img_path))
            image = image[:2048, :, :]
            if image.shape != shape:
                continue
            img_list.append(image)
            name_list.append(img_path)
    return np.asarray(img_list), name_list


def save_imgs(imgs: np.ndarray, dir_path: str, prefix: str = 'img_0') -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(imgs.shape) == 4:
        for i, img in enumerate(imgs):  # imgs
            cv2.imwrite(path.join(dir_path, prefix + '_clip_' + str(i) + '.jpg'), img)
    elif len(imgs.shape) == 3:  # single img
        cv2.imwrite(path.join(dir_path, prefix + '_clip_0.jpg'), imgs)
    else:
        print("Wrong imgs shape")

def save_imgs_png(imgs: np.ndarray, dir_path: str, prefix: str = 'img_0') -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if len(imgs.shape) == 4:
        for i, img in enumerate(imgs):  # imgs
            cv2.imwrite(path.join(dir_path, prefix + '_clip_' + str(i) + '.png'), img)
    elif len(imgs.shape) == 3:  # single img
        cv2.imwrite(path.join(dir_path, prefix + '_clip_0.png'), imgs)
    else:
        print("Wrong imgs shape")

def list_dirs(dir_path: str) -> list:
    dirs = [dir_path]
    idx = 0
    for dir in dirs:
        if path.isdir(dir):
            for _, sec_dir in enumerate(os.listdir(dir)):
                if path.isdir(path.join(dir, sec_dir)):
                    dirs.append(path.join(dir, sec_dir))
        idx += 1
    return dirs