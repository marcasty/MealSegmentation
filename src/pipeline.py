import hydra
from omegaconf import DictConfig
from FoodMetadataCOCO import FoodMetadata
from image_to_caption import get_captions
from text_to_embedding import get_embd_dicts
from embedding_to_category import get_categories
import torch
import os
import sys

sys.path.append("../")
sys.path.append('/me/unit_tests')
from unit_tests.embedding_to_category_check import check_metadata_categories
HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME = "/tmp"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.path.save_dir):
        os.makedirs(cfg.path.save_dir)

    metadata = FoodMetadata(cfg.file.metadata)

    if cfg.stage.image_to_caption.is_component:
        metadata = get_captions(
            metadata,
            model=cfg.stage.image_to_caption.model,
            image_dir=cfg.path.images,
            testing=cfg.var.testing,
            specific_model=cfg.stage.image_to_caption.specific_model,
        )
        print(metadata.anns[1])

    if cfg.stage.caption_to_keyword.is_component:
        metadata = get_captions(
            metadata,
            model=cfg.stage.caption_to_keyword.model,
            testing=cfg.var.testing,
            specific_model=cfg.stage.caption_to_keyword.specific_model,
        )
        print(metadata.anns[1])

    # when we add support for llama and mistral, we should refactor what gets passed to "get_embd_dicts"
    if cfg.stage.text_to_embed.is_component:
        keyword_to_embed, mod_cat_to_embed = get_embd_dicts(
            metadata,
            model=cfg.stage.text_to_embed.model,
            model_dir=cfg.path.glove_dir,
            mod_cat_file=cfg.file.mod_cats,
        )

    if cfg.stage.embed_to_cat.is_component:
        metadata = get_categories(
            metadata,
            keyword_to_embed=keyword_to_embed,
            mod_cat_to_embed=mod_cat_to_embed,
            mod_cat_file=cfg.file.mod_cats,
        )
        print(metadata.anns[1])
        print(len(metadata.anns))
    check_metadata_categories(metadata)


if __name__ == "__main__":
    main()
