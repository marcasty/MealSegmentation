import torch
import numpy as np
from typing import Union, List
from FoodMetadataCOCO import FoodMetadata
import cv2

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dino_setup(config_path, checkpoint_path):
    from groundingdino.util.inference import Model as DINOModel
    grounding_dino_model = DINOModel(model_config_path=config_path, model_checkpoint_path=checkpoint_path)
    return grounding_dino_model

def format_bbox(bboxes: np.ndarray, height: int, width: int) -> list:
    bboxes = bboxes.tolist()
    for i in range(len(bboxes)):
        bboxes[i][0] = max(0, bboxes[i][0])  # x1 floor is 0
        bboxes[i][1] = max(0, bboxes[i][1])  # y1 floor is 0
        bboxes[i][2] = min(width - bboxes[i][0], bboxes[i][2])  # x2 ceiling is width - x1
        bboxes[i][3] = min(height - bboxes[i][1], bboxes[i][3])  # y2 ceiling is height - y1
        bboxes[i] = [int(num) for num in bboxes[i]]
    return bboxes

def run_dino(image: Union[np.ndarray, torch.Tensor], classes: List[str], **kwargs) -> dict:
    """given BGR image, produce boxes"""

    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [f"all {class_name}s" for class_name in class_names]

    if "dino_model" in kwargs:
        model = kwargs["dino_model"]
    else:
        raise AssertionError("No Dino Model Provided")
    if "box_thresh" in kwargs:
        box_thresh = kwargs["box_thresh"]
    else:
        box_thresh = 0.35
    if "text_thresh" in kwargs:
        text_thresh = kwargs["text_thresh"]
    else:
        text_thresh = 0.25

    detections = model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=classes),
        box_threshold=box_thresh,
        text_threshold=text_thresh,
    )
    outside_class, dino_success = 0, 1
    class_ids = []

    if len(detections.class_id) == 0:
        print("Warning: No Ojects Detected")
        dino_success = 0
        return {"outside_class": outside_class, "dino_success": dino_success}

    # catch scenarios where DINO detects object out of classes
    else:
        for id in detections.class_id:
            if id is None:
                classes.append("*OTHER*")
                class_ids.append(len(classes) - 1)
                #print("WARNING: DINO detected object(s) outside the class list")
                outside_class = 1
            else:
                class_ids.append(int(id))

        if "image_annot" in kwargs:
            img = kwargs["image_annot"]
            h,w = img["height"],img["width"]
            bboxes = format_bbox(detections.xyxy, h, w)

        DINO_results = {
            "bbox": bboxes,
            "box_confidence": detections.confidence.tolist(),
            "class_id": class_ids,
            "classes": classes,
            "outside_class": outside_class,
            "dino_success": dino_success,
        }

        return DINO_results


def get_boxes(metadata: FoodMetadata, **kwargs) -> FoodMetadata:
    if "model" in kwargs:
        model = kwargs["model"]
    else:
        raise AssertionError("Must specify a model to predict bounding boxes")

    if "image_dir" in kwargs:
        image_dir = kwargs["image_dir"]
    else:
        raise AssertionError("No Llava 1.5 Image Processor Provided")

    if "testing" in kwargs:
        testing = kwargs["testing"]
    else:
        testing = False

    if "class_type" in kwargs:
        class_type = kwargs["class_type"]
    else:
        raise AssertionError("Must specify which classes to send to model f(image+text) = box")

    if model == "dino":
        if "model_chkpt" in kwargs:
            dino_model = dino_setup(kwargs["model_config"], kwargs["model_chkpt"])
        else:
            raise AssertionError("Must specify DINO model checkpoint")

    count = 0
    for cat_id, cat in metadata.cats.items():
        count += 1
        if count > 3 and testing is True:
            return metadata
        print(f'category {count} / 323: {cat["name_readable"]}')

        image_ids = metadata.getImgIds(catIds=cat_id)

        if not image_ids:
            continue
        else:
            imgs = metadata.loadImgs(image_ids)
            for img, img_id in zip(imgs, image_ids):
                image_bgr = cv2.imread(f'{image_dir}/{img["file_name"]}')
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                for ann in metadata.imgToAnns[img_id]:
                    if ann["category_id"] == cat_id:
                        ann_id = ann["id"]

                if model == "dino":
                    classes = metadata.anns[ann_id][class_type]
                    detections = run_dino(image_rgb, classes, image_annot = img, dino_model=dino_model)
                    if detections["dino_success"] == 0:
                        metadata.dataset["info"]["detection_issues"]["failures"].append(ann_id)
                        continue
                    if detections["outside_class"] == 1:
                        metadata.dataset["info"]["detection_issues"]["detect_nonclass"].append(ann_id)
                    metadata.add_dino_annot(ann_id, img_id, detections)
                    #print(metadata.anns[ann_id])
    return metadata
