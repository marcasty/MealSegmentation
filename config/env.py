import os

HOME = '/tmp'
os.environ['BLIP2_MODEL'] = "Salesforce/blip2-opt-2.7b"
os.environ['SPACY_MODEL'] = "en_core_web_sm"
os.environ['GROUNDING_DINO_CHECKPOINT_PATH'] = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
os.environ['GROUNDING_DINO_CONFIG_PATH'] = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
os.environ['SAM_ENCODER_VERSION'] = "vit_h"
os.environ['SAM_CHECKPOINT_PATH'] = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
