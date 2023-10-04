from typing import List
from string import punctuation
import cv2
import numpy as np
import torch
from FoodMetadataCOCO import FoodMetadata
import time

global DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_keywords(img_dir, data_file, blip_processor, blip2_model, spacy_nlp):
    # read in metadata
    metadata = FoodMetadata(data_file)

    def get_hotwords(text):
        result = []
        pos_tag = ['PROPN', 'NOUN']
        doc = spacy_nlp(text.lower())
        for token in doc:
            if (token.text in spacy_nlp.Defaults.stop_words or token.text in punctuation):
                continue
            elif (token.pos_ in pos_tag):
                result.append(token.text)
        return result

    count = 0
    for image_id, annot in metadata.imgs.items():
        count += 1
        if count < 5:
            start = time.time()
            SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"

            # BLIP2 + SPACY
            blip2_prompt = "the food here is"
            image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inputs = blip_processor(image_rgb, text=blip2_prompt, return_tensors="pt").to(DEVICE, torch.float16)
            generated_ids = blip2_model.generate(**inputs, max_new_tokens=20)
            generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = generated_text[0].strip()
            blip2_words = set(get_hotwords(generated_text))
            print(f'BLIP2 + spacy items are : {blip2_words}')

            # add annotation
            metadata.add_blip2_spacy_annot(image_id, generated_text, blip2_words)
            print(f'Time Taken: {time.time() - start}')

    return metadata


def get_boxes_and_mask(img_dir, mask_dir, data_file, grounding_dino_model, mask_predictor,
                       box_thresh=0.35, text_thresh=0.25, use_searchwords=False):

    # read in metadata
    metadata = FoodMetadata(data_file)

    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [f"all {class_name}s" for class_name in class_names]

    count = 0
    for img_id, annot in metadata.imgs.items():

        # get annotation id 
        ann_id = metadata.getAnnsIds(imgIds=[img_id])[0]

        count += 1
        if count < 5:
            start = time.time()
            SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"
            
            # get classes from filename
            #search_words = set(annot['filename'].split('/')[1].split('_')[0].split('-'))
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

            # GROUNDING DINO
            CLASSES = []
            for word in joint_words:
                if word not in forbidden_classes:
                    CLASSES.append(word)

            detections = grounding_dino_model.predict_with_classes(
                            image=image_bgr,
                            classes=enhance_class_name(class_names=CLASSES),
                            box_threshold=box_thresh,
                            text_threshold=text_thresh)

            # add dino
            dino_ann_ids = metadata.add_dino_annot(img_id, ann_id, CLASSES, detections.class_id, detections.xyxy, detections.confidence)
            print(f'DINO Time Taken: {time.time() - start}')

            # SAM
            mask_predictor.set_image(image_rgb)
            bounding_boxes = detections.xyxy
            detected_classes = detections.class_id
            masks_list = []
            mask_confidence_list = []
            for i, ann_id in enumerate(dino_ann_ids):
                print(f'Detected Classes are : {CLASSES[detected_classes[i]]}')
                DINO_box = bounding_boxes[i]
                masks, scores, _ = mask_predictor.predict(box=DINO_box, multimask_output=True)
                best_mask_idx = np.argmax(scores)
                high_conf_mask = masks[best_mask_idx]
                masks_list.append(high_conf_mask)
                mask_confidence_list.append(scores[best_mask_idx])
            metadata.add_sam_annot(dino_ann_ids, masks_list, mask_confidence_list, mask_dir)
            print(f'SAM Total Time Taken: {time.time() - start}')
    return metadata


# Let us make a function that adds masks and boxes and classes to the existing dataset
def get_kewords_boxes_and_mask(img_dir, mask_dir, data_file, blip_processor, blip2_model,
                               grounding_dino_model, mask_predictor,
                               spacy_nlp, box_thresh=0.35, text_thresh=0.25, use_searchwords=False):
    # read in metadata
    metadata = FoodMetadata(data_file)

    def get_hotwords(text):
        result = []
        pos_tag = ['PROPN', 'NOUN']
        doc = spacy_nlp(text.lower())
        for token in doc:
            if (token.text in spacy_nlp.Defaults.stop_words or token.text in punctuation):
                continue
            elif (token.pos_ in pos_tag):
                result.append(token.text)
        return result

    def enhance_class_name(class_names: List[str]) -> List[str]:
        return [f"all {class_name}s" for class_name in class_names]

    count = 0
    for id, annot in metadata.coco['images'].items():
        count += 1
        if count < 2:
            SOURCE_IMAGE_PATH = f"{img_dir}{annot['filename']}"
            # get classes from filename

            search_words = set(annot['filename'].split('/')[1].split('_')[0].split('-'))

            forbidden_classes = ['and', 'or', 'bowl', 'bowls', 'plate', 'plates', 'of', 'with',
                                 'glass', 'glasses', 'fork', 'forks', 'knife', 'knives',
                                 'spoon', 'spoons', 'cup', 'cups']

            # BLIP2 + SPACY
            blip2_prompt = "the food here is"
            image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inputs = blip_processor(image_rgb, text=blip2_prompt, return_tensors="pt").to(DEVICE, torch.float16)
            generated_ids = blip2_model.generate(**inputs, max_new_tokens=20)
            generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = generated_text[0].strip()
            blip2_words = set(get_hotwords(generated_text))

            # add annotation
            metadata.add_blip2_spacy_annot(id, generated_text, blip2_words)

            if use_searchwords:
                joint_words = blip2_words.union(search_words)
            else:
                joint_words = blip2_words

            # GROUNDING DINO
            CLASSES = []
            for word in joint_words:
                if word not in forbidden_classes:
                    CLASSES.append(word)

            print(f'BLIP2 + spacy items are : {CLASSES}')
            detections = grounding_dino_model.predict_with_classes(
                            image=image_bgr,
                            classes=enhance_class_name(class_names=CLASSES),
                            box_threshold=box_thresh,
                            text_threshold=text_thresh)

            # add dino
            metadata.add_dino_annot(id, CLASSES, detections.class_id, detections.xyxy, detections.confidence)

            # SAM
            mask_predictor.set_image(image_rgb)
            bounding_boxes = detections.xyxy
            detected_classes = detections.class_id
            masks_list = []
            mask_confidence_list = []
            for i in range(len(bounding_boxes)):
                print(f'Detected Classes are : {CLASSES[detected_classes[i]]}')
                DINO_box = bounding_boxes[i]
                masks, scores, _ = mask_predictor.predict(box=DINO_box, multimask_output=True)
                best_mask_idx = np.argmax(scores)
                high_conf_mask = masks[best_mask_idx]
                masks_list.append(high_conf_mask)
                mask_confidence_list.append(scores[best_mask_idx])
            metadata.add_sam_annot(id, masks_list, mask_confidence_list, mask_dir)

    return metadata
