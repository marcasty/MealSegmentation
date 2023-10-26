import sys
import os
import torch
import matplotlib.pyplot as plt
import time

# Add the parent directory of your project to the sys.path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)
from FoodMetadataCOCO import FoodMetadata


def check_missing_boxes(coco: FoodMetadata) -> None:
    count = 0
    fail = 0
    for ann_id, ann in coco.anns.items():
        if "bbox" not in ann:
            if ann_id in coco.dataset["info"]["detection_issues"]["failures"]:
                print(f"Annotation {ann_id} is missing a bbox, but it is accounted for")
            else:
                print(
                    f"Annotation {ann_id} is missing a bbox, but it is *unaccounted* for"
                )
                fail = 1
                count += 1

    if fail == 1:
        raise AssertionError(
            f"Your pipeline didn't box {count} images without telling you"
        )
    else:
        print(
            f"Test Passed! You either boxed all images or know of the ones without boxes"
        )
