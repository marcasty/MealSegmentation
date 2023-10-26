from string import punctuation
from FoodMetadataCOCO import FoodMetadata
from omegaconf import DictConfig


def spacy_setup(specific_model):
    import spacy

    model = spacy.load(specific_model)
    return model


def run_spacy(text: str, **kwargs) -> list:
    """given text, produce hot words"""

    if "spacy" in kwargs:
        spacy = kwargs["spacy"]
    else:
        raise AssertionError("No Spacy Model Provided")

    result = []
    pos_tag = ["PROPN", "NOUN"]
    doc = spacy(text.lower())
    for token in doc:
        if token.text in spacy.Defaults.stop_words or token.text in punctuation:
            continue
        elif token.pos_ in pos_tag:
            result.append(token.text)
    return list(set(result))


def get_keywords(metadata: FoodMetadata,  cfg: DictConfig) -> FoodMetadata:

    model_name = cfg.stage.image_to_caption.model
    testing = cfg.var.testing
    model_chkpt = cfg.stage.image_to_caption.model_chkpt
    device = cfg.var.device

    if model_name == "spacy":
        spacy = spacy_setup(model_chkpt)
    else:
        raise AssertionError(f"Model '{model_name}' not supported for keyword extraction")

    count = 0
    for cat_id, cat in metadata.cats.items():
        count += 1
        if count > 3 and testing is True:
            return metadata
        print(f'category {count} / 323: {cat["name_readable"]}')

        img_ids = metadata.getImgIds(catIds=cat_id)

        for img_id in img_ids:
            for ann in metadata.imgToAnns[img_id]:
                if ann["category_id"] == cat_id:
                    if model_name == "spacy":
                        keywords = run_spacy(ann["caption"], spacy)
                    metadata.add_text_annot(ann["id"], img_id, "keywords", keywords)
    return metadata
