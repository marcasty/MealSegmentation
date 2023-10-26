import sys
import os
import torch
import matplotlib.pyplot as plt
import time

# Add the parent directory of your project to the sys.path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)
from FoodMetadataCOCO import FoodMetadata


def check_missing_masks(coco: FoodMetadata) -> None:
    count = 0
    fail = 0
    for ann_id, ann in coco.anns.items():
        if (
            "mask" not in ann
            and ann_id not in coco.dataset["info"]["detection_issues"]["failures"]
        ):
            print(f"Annotation {ann_id} is missing a mask!")
            fail = 1
            count += 1
    if fail == 1:
        raise AssertionError(
            f"Your pipeline didn't produce masks for {count} boxed images"
        )
    else:
        print(f"Test Passed! You produced a mask for every image with a box")
