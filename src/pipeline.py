import hydra
from omegaconf import DictConfig
from FoodMetadataCOCO import FoodMetadata
from image_to_caption import get_captions
from text_to_embedding import get_embd_dicts
from embedding_to_category import get_categories
from image_text_to_box import get_boxes
from image_box_to_mask import get_masks
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

    metadata = FoodMetadata(cfg.file.metadata)
    #print(metadata.dataset["info"])

    if cfg.stage.image_to_caption.is_component:
        metadata = get_captions(
            metadata,
            model=cfg.stage.image_to_caption.model,
            image_dir=cfg.path.images,
            testing=cfg.var.testing,
            model_chkpt=cfg.stage.image_to_caption.model_chkpt,
        )

    if cfg.stage.caption_to_keyword.is_component:
        metadata = get_captions(
            metadata,
            model=cfg.stage.caption_to_keyword.model,
            testing=cfg.var.testing,
            model_chkpt=cfg.stage.caption_to_keyword.model_chkpt,
        )

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

    if cfg.stage.image_text_to_box.is_component:
        metadata = get_boxes(
            metadata,
            model=cfg.stage.image_text_to_box.model,
            image_dir=cfg.path.images,
            testing=cfg.var.testing,
            model_chkpt=cfg.stage.image_text_to_box.model_chkpt,
            model_config=cfg.stage.image_text_to_box.model_config,
            class_type=cfg.stage.image_text_to_box.class_type,
        )

        metadata.export_coco(new_file_name=cfg.file.metadata_save)
        #print(sorted(metadata.anns.keys()))

    if cfg.stage.image_box_to_mask.is_component:
        metadata = get_masks(
            metadata,
            model=cfg.stage.image_box_to_mask.model,
            model_chkpt=cfg.stage.image_box_to_mask.model_chkpt,
            encoder=cfg.stage.image_box_to_mask.model_encoder,
            image_dir=cfg.path.images,
            mask_dir=cfg.path.mask_dir,
            testing=cfg.var.testing,
        )
        #metadata.export_coco(new_file_name=cfg.file.metadata_save)


if __name__ == "__main__":
    main()
