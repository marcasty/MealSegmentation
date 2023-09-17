import os
from typing import List
from string import punctuation
import spacy
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from groundingdino.util.inference import Model as DINOModel
from segment_anything import sam_model_registry, SamPredictor


HOME = os.path.expanduser("~")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
blip2_processer = AutoProcessor.from_pretrained(BLIP2_MODEL)
blip2_model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL, torch_dtype=torch.float16).to(DEVICE)
SPACY_MODEL = "en_core_web_sm"
spacy_nlp = spacy.load(SPACY_MODEL)

GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
grounding_dino_model = DINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)


# Let us make a function that adds masks and boxes and classes to the existing dataset
def add_masks_and_labels(img_dir, data_file, box_thresh=0.35, text_thresh=0.25):
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

    for item in data_file['images']:
        if len(item['masks']) > 0:
            pass
        else:
            SOURCE_IMAGE_PATH = f"{img_dir}{item['filename']}"
            # get classes from filename
            search_words = set(item['filename'].split('_')[0].split('-'))
            forbidden_classes = ['and', 'or', 'bowl', 'bowls', 'plate', 'plates', 'of', 'with',
                                 'glass', 'glasses', 'fork', 'forks', 'knife', 'knives',
                                 'spoon', 'spoons', 'cup', 'cups']

            # BLIP2 + SPACY
            blip2_prompt = "the food here is"
            image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inputs = blip2_processer(image_rgb, text=blip2_prompt, return_tensors="pt").to(DEVICE, torch.float16)
            generated_ids = blip2_model.generate(**inputs, max_new_tokens=20)
            generated_text = blip2_processer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = generated_text[0].strip()
            blip2_words = set(get_hotwords(generated_text))

            # GROUNDING DINO
            joint_words = blip2_words.union(search_words)
            CLASSES = []
            for word in joint_words:
                if word not in forbidden_classes:
                    CLASSES.append(word)
            detections = grounding_dino_model.predict_with_classes(
                            image=image_bgr,
                            classes=enhance_class_name(class_names=CLASSES),
                            box_threshold=box_thresh,
                            text_threshold=text_thresh)

            # SAM
            mask_predictor.set_image(image_rgb)
            for obj in detections:
                if obj[3] is not None:
                    DINO_box = obj[0]
                    DINO_confidence = obj[2]
                    class_label = CLASSES[obj[3]]
                    masks, scores, logits = mask_predictor.predict(box=DINO_box, multimask_output=True)
                    best_mask_idx = np.argmax(scores)
                    high_conf_mask = masks[best_mask_idx]
                    item['masks'].append(high_conf_mask)
                    item['bounding_box'].append(DINO_box)
                    item['mask_confidence'].append(scores[best_mask_idx])
                    item['dino_confidence'].append(DINO_confidence)
                    item['mask_labels'].append(class_label)
    return data_file
