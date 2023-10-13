import hydra
from omegaconf import DictConfig, OmegaConf
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
def main(cfg : DictConfig) -> None:
    if not os.path.exists(cfg.path.save_dir):
        os.makedirs(cfg.path.save_dir)

    metadata = FoodMetadata(cfg.file.metadata, pred = True)

    if cfg.stage.image_to_caption.is_component == True:
        metadata = get_captions(metadata, model= cfg.stage.image_to_caption.model,
                                image_dir = cfg.path.images,
                                testing = cfg.var.testing,
                                model_variation = cfg.stage.image_to_caption.model_variation
                                )
        print(metadata.anns[1])        
        
        # do a check

if __name__ == "__main__":
    main()
