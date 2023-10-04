import os
import spacy
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from groundingdino.util.inference import Model as DINOModel
# from segment_anything import sam_model_registry, SamPredictor
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

from processing_pipeline import get_keywords, get_boxes_and_mask, get_kewords_boxes_and_mask
import sys
sys.path.append('../')

HOME = os.path.expanduser("~")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = '/tmp'
os.environ['BLIP2_MODEL'] = "Salesforce/blip2-opt-2.7b"
os.environ['SPACY_MODEL'] = "en_core_web_sm"
os.environ['GROUNDING_DINO_CHECKPOINT_PATH'] = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
os.environ['GROUNDING_DINO_CONFIG_PATH'] = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# os.environ['SAM_ENCODER_VERSION'] = "vit_h"
os.environ['SAM_ENCODER_VERSION'] = "vit_t"
# os.environ['SAM_CHECKPOINT_PATH'] = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
os.environ['SAM_CHECKPOINT_PATH'] = os.path.join(HOME, "MobileSAM/weights/mobile_sam.pt")

GROUNDING_DINO_CONFIG_PATH = os.environ['GROUNDING_DINO_CONFIG_PATH']
GROUNDING_DINO_CHECKPOINT_PATH = os.environ['GROUNDING_DINO_CHECKPOINT_PATH']
grounding_dino_model = DINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = os.environ['SAM_ENCODER_VERSION']
SAM_CHECKPOINT_PATH = os.environ['SAM_CHECKPOINT_PATH']
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam.eval()
mask_predictor = SamPredictor(sam)

# BLIP2_MODEL = os.environ['BLIP2_MODEL']
# blip_processor = AutoProcessor.from_pretrained(BLIP2_MODEL)
# blip2_model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL, torch_dtype=torch.float16).to(DEVICE)
# blip2_model.eval()
# SPACY_MODEL = os.environ['SPACY_MODEL']
# spacy_nlp = spacy.load(SPACY_MODEL)

file = '/me/google_food101_10k_dedup_keywords.json'
img_dir = '/me/images'
mask_dir = '/me/masks'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# new_metadata = get_keywords(img_dir, file, blip_processor, blip2_model, spacy_nlp)

new_metadata = get_boxes_and_mask(img_dir, mask_dir, file, grounding_dino_model, mask_predictor,
                                  box_thresh=0.35, text_thresh=0.25, use_searchwords=False)

new_metadata.export_coco(new_file_name='../google_food101_10k_dedup_keywords_masks.json', replace=False)
