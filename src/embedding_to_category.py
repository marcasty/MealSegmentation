from scipy.spatial.distance import euclidean


def run_nn_lookup(text_embed: dict, cat_embed: dict) -> dict:
    """
    translates spacy words to modified categories
    Args:
    spacy_dict -> spacy word : embd
    cat_dict -> modified category word : embd
    Returns:
    spacy_to_cat -> spacy word : modified category word
    """
    text_to_cat = {}
    for word, embedding in text_embed.items():
        nearest = sorted(
            cat_embed.keys(), key=lambda word: euclidean(cat_embed[word], embedding)
        )
        text_to_cat[word] = nearest[0]
    return text_to_cat


def get_categories(metadata, keyword_to_embed: dict, mod_cat_to_embed: dict, **kwargs):
    if "mod_cat_file" in kwargs:
        with open(kwargs["mod_cat_file"], "r") as f:
            cats = f.readlines()
            mod_cat_names = [cat.strip() for cat in cats]

    cat_ids = metadata.loadCats(metadata.getCatIds())
    cat_names = [_["name_readable"] for _ in cat_ids]

    # modified category : category
    mod_to_cat = {}
    for cat, mod in zip(cat_names, mod_cat_names):
        mod_to_cat[mod] = cat

    # keyword : modified category
    keyword_to_mod_cat = run_nn_lookup(keyword_to_embed, mod_cat_to_embed)

    # annotate the dataset
    for cat_id, cat in metadata.cats.items():
        img_ids = metadata.getImgIds(catIds=cat_id)
        for img_id in img_ids:
            for ann in metadata.imgToAnns[img_id]:
                if ann["category_id"] == cat_id:
                    mod_cat = []
                    cat = []
                    for keyword in ann["spacy"]:  # change to keywords
                        if keyword in keyword_to_mod_cat:
                            mod_cat.append(keyword_to_mod_cat[keyword])
                            cat.append(mod_to_cat[keyword_to_mod_cat[keyword]])
                    metadata.add_annot(ann["id"], img_id, "mod_class_from_embd", mod_cat)
                    metadata.add_annot(ann["id"], img_id, "class_from_embd", cat)
    return metadata
