from string import punctuation
from FoodMetadataCOCO import FoodMetadata


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


def get_keywords(metadata: FoodMetadata, **kwargs) -> FoodMetadata:
    if "model" not in kwargs:
        model = kwargs["model"]
        raise AssertionError("Must Give a Model to Extract Keywords")
    
    if model == "spacy":
        if "model_chkpt" in kwargs:
            spacy = spacy_setup(kwargs["specific_model"])
        else:
            raise AssertionError("Must Specify Model Details to Extract Keywords")
    else:
        raise AssertionError(f"Model '{model}' not supported for keyword extraction")

    if "testing" in kwargs:
        testing = kwargs["testing"]
    else:
        testing = False

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
                    if model == "spacy":
                        keywords = run_spacy(ann["caption"], spacy)
                    metadata.add_text_annot(ann["id"], img_id, "keywords", keywords)
    return metadata
