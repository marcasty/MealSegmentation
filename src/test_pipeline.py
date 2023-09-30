import matplotlib.pyplot as plt
import cv2
import numpy as np
from FoodMetadatametadata.coco import FoodMetadata


# Let us make a function that adds masks and boxes and classes to the existing dataset
def add_masks_and_labels(img_dir, data_file, box_thresh=0.35, text_thresh=0.25):
    # read in metadata
    metadata = FoodMetadata(data_file)

    for id, annot in metadata['images'].items():
        SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"

        image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        id = metadata.coco["annotations"][id]['id']

        # IMAGE ID
        blip2 = metadata.coco["annotations"][id]['blip2']
        spacy = metadata.coco["annotations"][id]['spacy']

        # DINO STUFF
        num_objects = metadata.coco["annotations"][id]["num_objects"]
        classes = metadata.coco["annotations"][id]["classes"]
        class_ids = metadata.coco["annotations"][id]["class_ids"]
        boxes = metadata.coco["annotations"][id]["xyxy_boxes"]
        box_confidence = metadata.coco["annotations"][id]["box_confidence"]

        # SAM STUFF
        mask = metadata.metadata.coco["annotations"][id]["masks"]
        scores = metadata.metadata.coco["annotations"][id]["mask_confidence"]

        plt.figure()
        plt.imshow(image_rgb)
        plt.annotate(blip2, xy=(2, 1), xytext=(3, 1.5))
        plt.annotate(spacy, xy=(2, 1), xytext=(3, 1.5))
