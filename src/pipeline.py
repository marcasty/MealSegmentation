import hydra
from omegaconf import DictConfig
from FoodMetadataCOCO import FoodMetadata
from image_to_caption import get_captions
import torch
import os
import sys
sys.path.append('../')
HOME = os.path.expanduser("~")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = '/tmp'


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.path.save_dir):
        os.makedirs(cfg.path.save_dir)

    metadata = FoodMetadata(cfg.file.metadata, pred=True)

    if cfg.stage.image_to_caption.is_component:
        metadata = get_captions(metadata, model=cfg.stage.image_to_caption.model,image_dir=cfg.path.images,
                                testing=cfg.var.testing,specific_model=cfg.stage.image_to_caption.specific_model)
        print(metadata.anns[1])
        # do a check

    if cfg.stage.caption_to_keyword.is_component:
        metadata = get_captions(metadata, model=cfg.stage.caption_to_keyword.model, 
                                testing=cfg.var.testing, specific_model=cfg.stage.caption_to_keyword.specific_model)
        print(metadata.anns[1])

if __name__ == "__main__":
    main()
