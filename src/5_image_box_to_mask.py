import torch
import numpy as np
from typing import Union, Tuple, List
from collections import Counter

from utils import assert_input

global DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mobilesam_import():
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def sam_setup(sam_encoder_version, sam_checkpoint_path):
    sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path).to(device=DEVICE)
    sam.eval()
    mask_predictor = SamPredictor(sam)
    return mask_predictor


def run_sam_conditioned(image: Union[np.ndarray, torch.Tensor], annotation: dict, **kwargs) -> Tuple(List, List):
    """given an image and boxes, generate masks"""

    if 'sam_model' in kwargs:
        model = kwargs['sam_model']
    else:
        raise AssertionError("No SAM Model Provided")
    if 'device' in kwargs:
        device = kwargs['device']
    else:
        print(f"No device found, using baseline computing device: {DEVICE}")
        device = DEVICE

    bounding_box = annotation['bbox']

    image.to(device=device)
    model.set_image(image)
    
    masks, scores, _ = model.predict(box=bounding_box, multimask_output=True)
    best_mask_idx = np.argmax(scores)
    high_conf_mask = masks[best_mask_idx]
    mask_confidence = scores[best_mask_idx]

    return (high_conf_mask, mask_confidence)


def get_masks(metadata, **kwargs):
    if "model" in kwargs:
        model = kwargs["model"]
    else:
        raise AssertionError("Must specify a model to create segmentation masks")
    
    if model == "sam":
        mobilesam_import()
        if "encoder" in kwargs:
            sam_encoder_version = kwargs["encoder"]
        else:
            raise AssertionError("Must specify SAM encoder")
        
        if "model_chkpt" in kwargs:
            sam_checkpoint_path = kwargs["model_chkpt"]
        else:
            raise AssertionError("Must specify model checkpoint")
        mask_predictor = sam_setup(sam_encoder_version, sam_checkpoint_path)
    
    if "testing" in kwargs:
        testing = kwargs["testing"]
    else:
        testing = False
    
    if "mask_dir" in kwargs:
        mask_dir = kwargs["mask_dir"]
    else:
        mask_dir = None

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
                if ann["id"] in metadata.dataset["info"]["detection_issues"]["failures"]:
                    continue

                if model == "sam":
                    mask, mask_confidence = run_sam_box(image_rgb, annotation = ann, sam_model = mask_predictor)
                    metadata.add_sam_annot(ann_id, img_id, mask, mask_confidence, mask_dir)
    return metadata