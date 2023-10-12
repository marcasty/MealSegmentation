import torch
import numpy as np
from typing import Union, Tuple, List
from collections import Counter

from utils import assert_input

global DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_sam_conditioned(image: Union[np.ndarray, torch.Tensor], detections: dict, **kwargs) -> Tuple(List, List, int):
    """given an image and boxes, generate masks"""

    print("Please ensure input image is in RGB format!")
    image = assert_input(image)

    if 'sam_model' in kwargs:
        model = kwargs['sam_model']
    else:
        raise AssertionError("No SAM Model Provided")
    if 'device' in kwargs:
        device = kwargs['device']
    else:
        print(f"No device found, using baseline computing device: {DEVICE}")
        device = DEVICE

    image.to(device=device)
    model.set_image(image)
    bounding_boxes = detections['bbox']
    detected_classes = detections['class_id']
    masks_list = []
    mask_confidence_list = []
    dino_success = 1

    # creating a list with the keys
    items = len(Counter(detected_classes).keys())
    if len(items) == 0: 
        print('MEGA WARNING: no objects detected :(')
        dino_success = 0
        return (masks_list, mask_confidence_list, dino_success)

    else:
        for i in len(bounding_boxes):
            DINO_box = bounding_boxes[i]
            masks, scores, _ = model.predict(box=DINO_box, multimask_output=True)
            best_mask_idx = np.argmax(scores)
            high_conf_mask = masks[best_mask_idx]
            masks_list.append(high_conf_mask)
            mask_confidence_list.append(scores[best_mask_idx])

    return (masks_list, mask_confidence_list, dino_success)
