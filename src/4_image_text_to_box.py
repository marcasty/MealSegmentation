import torch
import numpy as np
from typing import Union, List, Tuple

from utils import assert_input

global DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_dino(image: Union[np.ndarray, torch.Tensor], classes: List[str], **kwargs) -> Tuple(dict, int):
    """given BGR image, produce boxes"""

    print("Please ensure input image is in BGR format!")
    image = assert_input(image)

    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [f"all {class_name}s" for class_name in class_names]

    if 'dino_model' in kwargs:
        model = kwargs["dino_model"]
    else:
        raise AssertionError("No Dino Model Provided")
    if 'box_thresh' in kwargs:
        box_thresh = kwargs["box_thresh"]
    else:
        box_thresh = 0.35
    if 'text_thresh' in kwargs:
        text_thresh = kwargs["text_thresh"]
    else:
        text_thresh = 0.25

    detections = model.predict_with_classes(image=image,
                                            classes=enhance_class_name(class_names=classes),
                                            box_threshold=box_thresh,
                                            text_threshold=text_thresh)
    outside_class = 0

    # catch scenarios where DINO detects object out of classes
    class_ids = []
    for id in detections.class_ids:
        if id is None:
            classes.append('object_outside_class')
            class_ids.append(len(classes) - 1)
            print('WARNING: DINO detected object(s) outside the class list')
            outside_class = 1
        else:
            class_ids.append(int(id))

    detect = {
       "bbox": detections.xyxy,
       "confidence": detections.confidence,
       "class_id": class_ids,
            }

    return (detect, outside_class)
