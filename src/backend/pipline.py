from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

from typing import Any, Dict, List

from classifier import VGG16, Dataset
from utils import *

COLOR_LIST = [np.array([1., 0., 0., 0.35]), np.array([0., 1., 0., 0.35]), np.array([0., 0., 1., 0.35])]

class Pipeline:
    def __init__(self, checkpoint:str, model_type:str, classifier:str) -> None:
        Pipeline.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        Pipeline.sam.to('cuda')
        Pipeline.mask_generator = SamAutomaticMaskGenerator(
            Pipeline.sam,
            pred_iou_thresh=0.94,
            stability_score_thresh=0.,
            crop_n_layers=1,
            crop_n_points_downscale_factor=8,
            min_mask_region_area=1024)
        # Pipeline.classifier = VGG16(num_classes=3).cuda()
        Pipeline.classifier = torch.load(classifier)
    
    def make_anns(self, anns:List[Dict[str, Any]]) -> np.ndarray:
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return img
    
    def make_class_anns(self, anns:List[Dict[str, Any]], classes:np.ndarray) -> np.ndarray:
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0

        print(classes)
        print(len(sorted_anns))
        for i, ann in enumerate(sorted_anns):
            m = ann['segmentation']
            color_mask = COLOR_LIST[classes[i]]
            img[m] = color_mask
        return img
    
    def split(self, image:np.ndarray, anns:List[Dict[str, Any]]) -> List[torch.Tensor]:
        res = []
        for mask in anns:
            # if (np.sum(np.where(mask['segmentation'])) < 128):
            #     continue
            cov_img = np.zeros(image.shape, dtype=int)
            cov_img[np.where(mask['segmentation'])] = image[np.where(mask['segmentation'])]
            b_channel, g_channel, r_channel = cv2.split(cov_img)
            img_RGB = cv2.merge((r_channel, g_channel, b_channel))
            img = torch.from_numpy(img_RGB).permute(2, 0, 1)/255
            res.append(img)
        return res
    
    def classify(self, loader:DataLoader) -> np.ndarray:
        res = []
        confidence = []
        for inputs, targets in loader:
            with torch.no_grad():
                outputs = Pipeline.classifier(inputs.to('cuda'))
            for output in outputs:
                res.append(torch.argmax(output).item())
                confidence.append(torch.max(torch.softmax(output, dim=0)).item())
        return np.asarray(res), np.asarray(confidence)
    
    def make_masks(self, image:np.ndarray) -> List[Dict[str, Any]]:
        return Pipeline.mask_generator.generate(image)
    
    def pipeline(self, image:np.ndarray) -> np.ndarray:
        H, W = image.shape[0]//224, image.shape[1]//224
        image = image[:H * 224, :W * 224]
        imgs = slide_win_cut_imgs(image, (224, 224), (H, W))
        masks = []
        class1 = 0
        class2 = 0
        class3 = 0
        confi1 = np.zeros((10), dtype=int)
        confi2 = np.zeros((10), dtype=int)
        confi3 = np.zeros((10), dtype=int)
        for i, img in enumerate(imgs):
            print("no." + str(i))
            anns = self.make_masks(img)
            if len(anns) > 0:
                splited_imgs = self.split(img, anns)
                loader = DataLoader(Dataset(splited_imgs, torch.zeros(len(splited_imgs))))
                classes, confidences = self.classify(loader)
                class1 += np.sum(np.where(classes == 0))
                class2 += np.sum(np.where(classes == 1))
                class3 += np.sum(np.where(classes == 2))
                for i in range(len(classes)):
                    if classes[i] == 0:
                        confi1[int(confidences[i]//0.1)] += 1
                    elif classes[i] == 1:
                        confi2[int(confidences[i]//0.1)] += 1
                    elif classes[i] == 2:
                        confi3[int(confidences[i]//0.1)] += 1
                anns = self.make_class_anns(anns, classes)
            else:
                anns = np.zeros((img.shape[0], img.shape[1], 4))
            masks.append(anns)
        masks = np.asarray(masks)
        lines = []
        for row in range(H):
            lines.append(np.concatenate(masks[row * W:(row+1) * W], axis=1))
        lines = np.asarray(lines)
        masks = np.concatenate(lines, axis=0)

        # sam integral
        px = 1/plt.rcParams['figure.dpi']
        fig = Figure(figsize=(image.shape[0] * px, image.shape[1] * px))
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.imshow(image)
        ax.imshow(masks)
        ax.axis('off')
        ax.margins(0)
        fig.tight_layout(pad=0)
        canvas.draw()
        buf = canvas.buffer_rgba()
        integral = np.asarray(buf)

        # confidence bar
        x = np.arange(len(confi1))
        width = 0.2
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.bar(x - width, confi1, width=width, label='class1')
        ax.bar(x, confi2, width=width, label='class2')
        ax.bar(x + width, confi3, width=width, label='class3')
        ax.set_xlabel('confidence range')
        ax.set_ylabel('count')
        ax.set_title('Classifier Confidences Distribution')
        ax.legend()
        canvas.draw()
        buf = canvas.buffer_rgba()
        analysis1 = np.asarray(buf)

        # masks
        mask1 = np.zeros(masks.shape, dtype=int)
        mask2 = np.zeros(masks.shape, dtype=int)
        mask3 = np.zeros(masks.shape, dtype=int)
        area1 = 0
        area2 = 0
        area3 = 0
        for x in range(masks.shape[0]):
            for y in range(masks.shape[1]):
                if (masks[x, y] == COLOR_LIST[0]).all():
                    mask1[x, y] = COLOR_LIST[0] * 255
                    area1 += 1
                elif (masks[x, y] == COLOR_LIST[1]).all():
                    mask2[x, y] = COLOR_LIST[1] * 255
                    area2 += 1
                elif (masks[x, y] == COLOR_LIST[2]).all():
                    mask3[x, y] = COLOR_LIST[2] * 255
                    area3 += 1

        # area pie
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.pie([area1, area2, area3], labels=['class1', 'class2', 'class3'])
        ax.set_title('Areas of each Class')
        ax.legend()
        canvas.draw()
        buf = canvas.buffer_rgba()
        analysis2 = np.asarray(buf)
        return integral, mask1, mask2, mask3, analysis1, analysis2

