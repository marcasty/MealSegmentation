import os
import spacy
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from groundingdino.util.inference import Model as DINOModel
from segment_anything import sam_model_registry, SamPredictor
import torch

from processing_pipeline import add_masks_and_labels
import sys
sys.path.append('../')

HOME = os.path.expanduser("~")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = '/tmp'
os.environ['BLIP2_MODEL'] = "Salesforce/blip2-opt-2.7b"
os.environ['SPACY_MODEL'] = "en_core_web_sm"
os.environ['GROUNDING_DINO_CHECKPOINT_PATH'] = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
os.environ['GROUNDING_DINO_CONFIG_PATH'] = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
os.environ['SAM_ENCODER_VERSION'] = "vit_h"
os.environ['SAM_CHECKPOINT_PATH'] = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

GROUNDING_DINO_CONFIG_PATH = os.environ['GROUNDING_DINO_CONFIG_PATH']
GROUNDING_DINO_CHECKPOINT_PATH = os.environ['GROUNDING_DINO_CHECKPOINT_PATH']
grounding_dino_model = DINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = os.environ['SAM_ENCODER_VERSION']
SAM_CHECKPOINT_PATH = os.environ['SAM_CHECKPOINT_PATH']
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

BLIP2_MODEL = os.environ['BLIP2_MODEL']
blip_processor = AutoProcessor.from_pretrained(BLIP2_MODEL)
blip2_model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL, torch_dtype=torch.float16).to(DEVICE)
SPACY_MODEL = os.environ['SPACY_MODEL']
spacy_nlp = spacy.load(SPACY_MODEL)

file = '/me/google_food101_10k_dedup.json'
img_dir = '/me/images'

edit_file = add_masks_and_labels(img_dir, file, blip_processor, blip2_model,
                                 grounding_dino_model, mask_predictor, spacy_nlp)

print(edit_file)
