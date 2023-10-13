import hydra
from omegaconf import DictConfig, OmegaConf
from FoodMetadataCOCO import FoodMetadata
from image_to_caption import run_blip2, get_captions
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
        metadata = get_captions(metadata, cfg, model= cfg.stage.image_to_caption.model)
        print(metadata.anns[1])        
        
        # do a check

if __name__ == "__main__":
    main()
