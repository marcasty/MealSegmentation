from typing import List
from string import punctuation
import cv2
import numpy as np
import torch
from FoodMetadataCOCO import FoodMetadata
from embedding_translation import assign_classes
import time

global DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_hotwords(spacy_nlp, text):
    """given text, produce hot words"""
    result = []
    pos_tag = ['PROPN', 'NOUN']
    doc = spacy_nlp(text.lower())
    for token in doc:
        if (token.text in spacy_nlp.Defaults.stop_words or token.text in punctuation):
            continue
        elif (token.pos_ in pos_tag):
            result.append(token.text)
    return result


def run_blip(blip_processor, blip2_model, path):
    """given image path, produce text"""
    image_bgr = cv2.imread(path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prompt = "the food or foods in this image include"
    inputs = blip_processor(image, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
    generated_ids = blip2_model.generate(**inputs, max_new_tokens=20)
    generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = generated_text[0].strip()
    return generated_text


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]


def run_dino(image_bgr, CLASSES, grounding_dino_model, box_thresh=0.35, text_thresh=0.25):
    detections = grounding_dino_model.predict_with_classes(
                image=image_bgr,
                classes=enhance_class_name(class_names=CLASSES),
                box_threshold=box_thresh,
                text_threshold=text_thresh)
    return detections


def run_sam(image_rgb, CLASSES, detections, mask_predictor):
    mask_predictor.set_image(image_rgb)
    bounding_boxes = detections.xyxy
    detected_classes = detections.class_id
    masks_list = []
    mask_confidence_list = []
    for i, ann_id in enumerate(detections):
        print(f'Detected Classes are : {CLASSES[detected_classes[i]]}')
        DINO_box = bounding_boxes[i]
        masks, scores, _ = mask_predictor.predict(box=DINO_box, multimask_output=True)
        best_mask_idx = np.argmax(scores)
        high_conf_mask = masks[best_mask_idx]
        masks_list.append(high_conf_mask)
        mask_confidence_list.append(scores[best_mask_idx])


def get_keywords(img_dir, data_file, spacy_nlp, blip_processor, blip2_model, embedding_vars=None, testing=False):
    # read in metadata
    metadata = FoodMetadata(data_file)
    category_ids = metadata.loadCats(metadata.getCatIds())
    category_names = [_["name_readable"] for _ in category_ids]
    count = 0
    for cat_name in category_names:
        count += 1
        if count < 3 and testing is True:
            start = time.time()

        catIds = metadata.getCatIds([cat_name])
        if len(catIds) == 0: continue
        imgIds = metadata.getImgIds(catIds=catIds)
        if len(imgIds) == 0: continue
        imgs = metadata.loadImgs(imgIds)

        for img in imgs:
            blip2_text = run_blip(blip_processor, blip2_model, f'{img_dir}/{img["file_name"]}')
            spacy_words = set(get_hotwords(spacy_nlp, blip2_text))
            metadata.add_blip2_annot(img["id"], blip2_text)
            metadata.add_spacy_annot(img["id"], spacy_words)

        print(f'Time Taken: {time.time() - start}')

    if embedding_vars is not None:
        assign_classes(metadata, category_names, embedding_vars)

    return metadata


def get_boxes_and_mask(img_dir, mask_dir, data_file, grounding_dino_model, mask_predictor, use_searchwords=False, testing=False):

    # read in metadata
    metadata = FoodMetadata(data_file)
    count = 0
    for img_id, annot in metadata.imgs.items():

        # get annotation id
        # ann_id = metadata.getAnnsIds(imgIds=[img_id])[0]
        ann_id = metadata.id_to_idx(img_id)
        count += 1
        if count < 3 and testing is True:
            start = time.time()
            SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"

            # get classes from filename
            # search_words = set(annot['filename'].split('/')[1].split('_')[0].split('-'))
            search_words = metadata.cats[metadata.imgs[img_id]["category_id"]]["name_readable"]

            forbidden_classes = ['and', 'or', 'bowl', 'bowls', 'plate', 'plates', 'of', 'with',
                                 'glass', 'glasses', 'fork', 'forks', 'knife', 'knives',
                                 'spoon', 'spoons', 'cup', 'cups']

            image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            blip2_words = set(metadata.anns[ann_id]["spacy"])

            if use_searchwords:
                joint_words = blip2_words.union(search_words)
            else:
                joint_words = blip2_words

            CLASSES = []
            for word in joint_words:
                if word not in forbidden_classes:
                    CLASSES.append(word)

            # add DINO
            detections = run_dino(image_bgr, CLASSES, grounding_dino_model)
            dino_ann_ids = metadata.add_dino_annot(img_id, ann_id, CLASSES, detections.class_id, detections.xyxy, detections.confidence)
            print(f'DINO Time Taken: {time.time() - start}')

            # add SAM
            masks_list, mask_confidence_list = run_sam(image_rgb, CLASSES, detections, mask_predictor)
            metadata.add_sam_annot(dino_ann_ids, masks_list, mask_confidence_list, mask_dir)
            print(f'SAM Total Time Taken: {time.time() - start}')

    return metadata
