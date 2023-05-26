from utils import *
import os.path as path
from tqdm import tqdm

img_root = '/home/mpyg/Documents/Codes/MIP_SAM/img/SEM Images by Samples'
dirs = list_dirs(img_root)
print("dirs found:")
print(dirs)
print("========== list dir finished. ==========")

save_root = '/home/mpyg/Documents/Codes/MIP_SAM/split_img'
for dir in tqdm(dirs):
    images, names = read_imgs(dir, (2048, 2048, 3))
    for i in range(len(images)):
        imgs = slide_win_cut_imgs(images[i], (224, 224), (9, 9))
        save_imgs(imgs, path.join(save_root, names[i]), names[i])

print("========== spliting finished. ==========")

from backend.pipline import Pipeline

file_path = path.abspath(__file__)
project_path = path.dirname(path.dirname(file_path))

print("project path:")
print(project_path)
pipeline = Pipeline(path.join(project_path, "model/sam_vit_h_4b8939.pth"), "vit_h")

sam_root = '/home/mpyg/Documents/Codes/MIP_SAM/samed_img'
dirs = list_dirs(save_root)
for dir in tqdm(dirs):
    images, names = read_imgs(dir, (224, 224, 3))
    for i in range(len(images)):
        masks = pipeline.make_masks(images[i])
        res = []
        for mask in masks:
            cov_img = np.zeros(images[i].shape, dtype=int)
            cov_img[np.where(mask['segmentation'])] = images[i][np.where(mask['segmentation'])]
            b_channel, g_channel, r_channel = cv2.split(cov_img)
            alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
            alpha_channel[np.where(mask['segmentation'])] = 255
            img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            res.append(img_BGRA)
        save_imgs_png(np.asarray(res), path.join(sam_root, names[i]), names[i])
print("========== sam finished. ==========")