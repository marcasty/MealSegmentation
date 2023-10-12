from scipy.spatial.distance import euclidean


def run_nn_lookup(text_embeddings: dict, category_embeddings: dict) -> dict:
    """
    translates spacy words to modified categories
    Args:
    spacy_dict -> spacy word : embd
    cat_dict -> modified category word : embd
    Returns:
    spacy_to_cat -> spacy word : modified category word
    """
    nn_categories = []
    for word, embedding in text_embeddings.items():
        nearest = sorted(category_embeddings.keys(), key=lambda word: euclidean(category_embeddings[word], embedding))
        nn_categories.append(nearest[0])
    return nn_categories
