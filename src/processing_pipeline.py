from typing import List
from string import punctuation
import cv2
import numpy as np
import torch
from FoodMetadataCOCO import FoodMetadata
from text_to_embedding import assign_classes
import time
import matplotlib.pyplot as plt


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
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prompt = "the food or foods in this image include: "
    inputs = blip_processor(image_rgb, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
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
    outside_class = 0

    # catch scenarios where DINO detects object out of classes
    class_ids = []
    for id in detections.class_ids:
        if id is None:
          CLASSES.append('object_outside_class')
          class_ids.append(len(CLASSES) - 1)
          print('WARNING: DINO detected object(s) outside the class list')
          outside_class = 1
        else: class_ids.append(int(id))

    detect = {
       "bbox" : detections.xyxy,
       "confidence" : detections.confidence,
       "class_id" : class_ids,
            }
    return detect, outside_class


def run_sam_full(image_rgb, mask_generator):
    """given an image, generate masks automatically"""
    masks = mask_generator.generate(image_rgb)
    return masks


def run_classifier(image_rgb, blip2_model, blip_processor):
    """given rgb image, produce text"""
    prompt = "the food or foods in this image include"
    inputs = blip_processor(image_rgb, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
    generated_ids = blip2_model.generate(**inputs, max_new_tokens=20)
    generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = generated_text[0].strip()
    return generated_text


def run_sam_box(image_rgb, CLASSES, detections, mask_predictor):
    mask_predictor.set_image(image_rgb)
    bounding_boxes = detections['bbox']
    detected_classes = detections['class_id']
    masks_list = []
    mask_confidence_list = []
    dino_success = 1

    class_list = [CLASSES[i] for i in detected_classes]
    #print(f'Detected Classes are : {class_list}')
    
    if len(class_list) == 0: 
      print('MEGA WARNING: no objects detected :(')
      dino_success = 0
      return masks_list, mask_confidence_list, dino_success
    
    else:
      for i in len(bounding_boxes):
          DINO_box = bounding_boxes[i]
          masks, scores, _ = mask_predictor.predict(box=DINO_box, multimask_output=True)
          best_mask_idx = np.argmax(scores)
          high_conf_mask = masks[best_mask_idx]
          masks_list.append(high_conf_mask)
          mask_confidence_list.append(scores[best_mask_idx])
      return masks_list, mask_confidence_list, dino_success


def get_keywords(img_dir, metadata, spacy_nlp, blip_processor, blip2_model, testing=False):
    count = 0
    start = time.time()
    for cat_id, cat in metadata.cats.items():
        
        count += 1
        if count > 3 and testing is True: return metadata
        
        print(f'category {count} / 323: {cat["name_readable"]}')

        imgIds = metadata.getImgIds(catIds=cat_id)
        if len(imgIds) == 0: 
           continue
        else:
            imgs = metadata.loadImgs(imgIds)

        for img in imgs:
            # sometimes blip2 does not output anything useful
            spacy_words = {}
            attempt = 0
            while len(spacy_words) == 0:
                blip2_text = run_blip(blip_processor, blip2_model, f'{img_dir}/{img["file_name"]}')
                spacy_words = set(get_hotwords(spacy_nlp, blip2_text))
                attempt += 1
                if attempt > 5:
                  blip2_text = 'FAILURE' 
                  spacy_words = set(cat["name_readable"].split('_'))
            
            ann_id = metadata.add_annotation(img["id"], cat_id)
            metadata.add_blip2_annot(ann_id, img["id"], blip2_text)
            metadata.add_spacy_annot(ann_id, img["id"], spacy_words)

    print(f'Time Taken: {time.time() - start}')

    return metadata


def get_boxes_and_mask(img_dir, mask_dir, metadata, word_type, 
                       grounding_dino_model, mask_predictor, use_search_words = False, testing = False, timing = False):
  """
  get DINO boxes and SAM masks
  Args:
  img_dir = image directory
  mask_dir = SAM mask.pt directory
  metadata = FoodMetadata object
  word_type = blip2/spacy or modified class names derived with embeddings
  
  Returns:
  metadata object
  dino annotation ids
  """
  status_report = {'outside_class': [], 'no_detect': []}
  count = 0
  for cat_id, _ in metadata.cats.items():
    count += 1
    if count > 2 and testing is True:
      return metadata, status_report
    else:
      imgIds = metadata.getImgIds(catIds=cat_id)

      for img_id in imgIds:

        # Get Image
        SOURCE_IMAGE_PATH = f'{img_dir}/{metadata.imgs[img_id]["file_name"]}'
        image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # get correct annotation
        for ann in metadata.imgToAnns[img_id]:
           if ann['category_id'] == cat_id: ann_id = ann['id']

        # get words to pass to DINO
        if word_type == 'mod_class':
          try:
            classes = set(metadata.anns[ann_id]['mod_class_from_embd'])
          except:
            print(f'Warning: Annotation {ann_id} does not contain words')
            continue
        elif word_type == 'blip2':
          classes = set(metadata.anns[ann_id]['spacy'])
        
        if use_search_words is not False:
          classes.extend(metadata.imgs[img_id]['Google Search Query'])

        forbidden_classes = ['and', 'or', 'bowl', 'bowls', 'plate', 'plates', 'of', 'with',
                             'glass', 'glasses', 'fork', 'forks', 'knife', 'knives',
                             'spoon', 'spoons', 'cup', 'cups']

        CLASSES = [word for word in classes if word not in forbidden_classes]
        # Run DINO
        start = time.time()
        detections, outside_class = run_dino(image_bgr, CLASSES, grounding_dino_model)
        if outside_class == 1: 
          status_report['outside_class'].append(ann_id)
        dino_ann_ids = metadata.add_dino_annot(ann_id, img_id, CLASSES, detections['class_id'], detections['bbox'], detections['confidence'])
        if timing: print(f'DINO Time Taken: {time.time() - start}')

        # Run SAM
        start = time.time()
        masks_list, mask_confidence_list, outside_class, dino_success = run_sam_box(image_rgb, CLASSES, detections, mask_predictor)
        if dino_success == 0: 
          status_report['no_detect'].append(ann_id)
          continue
        metadata.add_sam_annot(dino_ann_ids, masks_list, mask_confidence_list, mask_dir)
        if timing: print(f'SAM Time Taken: {time.time() - start}')
  return metadata, status_report



def get_mask_and_keywords(img_dir, mask_generator, blip2_model, blip_processor):
    SOURCE_IMAGE_PATH = f'{img_dir}/{"miso-soup/miso-soup_00064.jpg"}'
    image_bgr = cv2.imread(SOURCE_IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    anns = run_sam_full(image_rgb, mask_generator)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img_full = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img_full[:, :, 3] = 0
    for ann in sorted_anns:
        img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        m = ann['segmentation']
        img[m] = image_rgb[m]
        text = run_classifier(img, blip2_model, blip_processor)
        print(f"Food is {text}")
        color_mask = np.concatenate([np.random.random(3), [0.75]])
        img_full[m] = color_mask
    plt.imsave('test_v3.png', img_full)
