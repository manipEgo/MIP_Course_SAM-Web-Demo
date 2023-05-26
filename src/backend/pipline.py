from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from typing import Any, Dict, List

class Pipeline:
    def __init__(self, checkpoint, model_type) -> None:
        Pipeline.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        Pipeline.sam.to('cuda')
        Pipeline.mask_generator = SamAutomaticMaskGenerator(
            Pipeline.sam,
            pred_iou_thresh=0.94,
            stability_score_thresh=0.,
            crop_n_layers=1,
            crop_n_points_downscale_factor=8,
            min_mask_region_area=1024)
    
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
    
    def make_masks(self, image:np.ndarray) -> List[Dict[str, Any]]:
        return Pipeline.mask_generator.generate(image)
    
    def pipeline(self, image:np.ndarray) -> np.ndarray:
        anns = self.make_anns(self.make_masks(image))
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.imshow(image)
        ax.imshow(anns)
        ax.axis('off')
        ax.margins(0)
        fig.tight_layout(pad=0)
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)

