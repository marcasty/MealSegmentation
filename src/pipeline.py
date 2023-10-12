import hydra
from omegaconf import DictConfig, OmegaConf
from FoodMetadataCOCO import FoodMetadata
from image_to_caption import run_blip2
import torch
import os
import sys
sys.path.append('../')
HOME = os.path.expanduser("~")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = '/tmp'


#@hydra.main(version_base=None, config_path="../conf", config_name="config")
def get_captions(cfg : DictConfig):
    for cat_id, cat in metadata.cats.items():
        
        count += 1
        if count > 3 and cfg.var.testing is True: return metadata
        print(f'category {count} / 323: {cat["name_readable"]}')

        imgIds = metadata.getImgIds(catIds=cat_id)
        
        if len(imgIds) == 0: continue
        else:
            imgs = metadata.loadImgs(imgIds)
            for img in imgs:
                image_bgr = cv2.imread(f'{cfg.path.images}/{img["file_name"]}')
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                caption = run_blip2(f'{img_dir}/{img["file_name"]}', blip_processor, blip2_model)            
                ann_id = metadata.add_annotation(img["id"], cat_id)
                metadata.add_text_annot(ann_id, img["id"], "blip2", caption)
    return metadata

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    if not os.path.exists(cfg.path.save_dir):
        os.makedirs(cfg.path.save_dir)

    metadata = FoodMetadata(cfg.file.metadata, pred = True)

    if cfg.stage.image_to_caption.is_component == True:

        if cfg.stage.image_to_caption.model == 'blip2':
            blip2_model, blip2_processor = blip2_setup(cfg)
        elif cfg.stage.image_to_caption.model == 'llava1.5':
            print('not implemented')
            return
        
        metadata = get_captions()
        print(metadata.anns[1])

    if cfg.stage.image_to_caption.is_component == True:
        metadata = get_captions()
        # do a check