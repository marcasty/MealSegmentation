import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from FoodMetadataCOCO import FoodMetadata
import supervision as sv

# Let us make a function that adds masks and boxes and classes to the existing dataset
def add_masks_and_labels(img_dir, data_file, box_thresh=0.35, text_thresh=0.25):
    # read in metadata
    metadata = FoodMetadata(data_file)

    for id, annot in metadata['images'].items():
        SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"

        image = cv2.imread(SOURCE_IMAGE_PATH)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        ax.annotate(blip2, xy=(2, 1), xytext=(3, 1.5))
        ax.annotate(spacy, xy=(2, 1), xytext=(3, 1.5))
        
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [f"{classes[class_ids[i]]} {box_confidence[i]:0.2f}" for i in range(num_objects)]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        
        sv.plot_image(annotated_frame, (16, 16))


        # Create a Rectangle patch
        rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()