from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np

from typing import Any, Dict, List

class Pipeline:
    def __init__(self, checkpoint, model_type) -> None:
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to('cuda')
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
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
        return self.mask_generator.generate(image)
    
    def pipeline(self, image:np.ndarray) -> np.ndarray:
        return self.make_anns(self.make_masks(image))

