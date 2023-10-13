import numpy as np
from typing import Tuple
import os


def download_glove(model_dir):
    """
    Get a dictionary of word embeddings from GloVe
    https://nlp.stanford.edu/projects/glove/

    Args:
        a path to save the model
    Return:
        embed_dict -> word : 200 dimension embedding
    """
    import urllib.request
    import zipfile

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(f"{model_dir}/glove.6B.zip"):
        print("Downloading GloVe")
        urllib.request.urlretrieve(
            "https://nlp.stanford.edu/data/glove.6B.zip", f"{model_dir}/glove.6B.zip"
        )
    if not os.path.exists(f"{model_dir}/glove.6B.200d.txt"):
        print("Unzipping GloVe")
        with zipfile.ZipFile(f"{model_dir}/glove.6B.zip", "r") as zip_ref:
            zip_ref.extractall(f"{model_dir}")

    print("Creating Dictionary of GloVe Embeddings")
    embed_dict = {}
    with open(f"{model_dir}/glove.6B.200d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embed_dict[word] = vector
    return embed_dict


def run_glove(text: str, **kwargs) -> np.ndarray:
    """given text, produce embedding"""
    words = text.split()

    if "model" in kwargs:
        model = kwargs["model"]
    else:
        raise AssertionError("No GloVe Embedding Dictionary Provided")

    vectors = []
    for word in words:
        try:
            vectors.append(model[word.lower()])
        except KeyError:
            print(f"L1 Warning: Word '{word}' has no Embedding")
            continue

    if len(vectors) < 1:
        if len(words) > 1:
            print(f"L2 Warning: Entire String '{text}' has no Embeddings")
            return []
        else:
            return np.array([])
    elif len(vectors) == 1:
        return vectors[0]
    else:
        return np.mean(vectors, axis=0)


def run_mistral(text: str, **kwargs) -> np.ndarray:
    """given text, produce embedding"""

    if "mistral" in kwargs:
        model = kwargs["mistral"]
    else:
        raise AssertionError("No Mistral Model Provided")

    embedding = model(text)
    return embedding


def run_llama2(text: str, **kwargs) -> np.ndarray:
    """given text, produce embedding"""

    if "llama2" in kwargs:
        model = kwargs["llama2"]
    else:
        raise AssertionError("No LLama2 Model Provided")

    embedding = model(text)
    return embedding


def get_embd_dicts(metadata, **kwargs) -> Tuple[dict, dict]:
    if "model" in kwargs:
        if kwargs["model"] == "glove":
            embed_dict = download_glove(kwargs["model_dir"])
        elif kwargs["model"] == "mistral":
            raise AssertionError("Setup Mistral")
    else:
        raise AssertionError("Must specify an embedding model")

    if "mod_cat_file" in kwargs:
        with open(kwargs["mod_cat_file"], "r") as f:
            cats = f.readlines()
            mod_cat_names = [cat.strip() for cat in cats]
    else:
        raise AssertionError("Must Supply File of Modified Category Names to Embed")

    unique_keywords = set()
    for ann in metadata.anns.values():
        unique_keywords.update(ann["spacy"])  # change to keywords
    unique_keywords_list = list(unique_keywords)

    keyword_to_embed = {}
    for word in unique_keywords_list:
        keyword_to_embed[word] = run_glove(word, model=embed_dict)
        if len(keyword_to_embed[word]) == 0:
            print(f"Notice: Removing {word} from Keyword Dictionary")
            del keyword_to_embed[word]

    mod_cat_to_embed = {}
    for word in mod_cat_names:
        mod_cat_to_embed[word] = run_glove(word, model=embed_dict)
        if len(mod_cat_to_embed[word]) == 0:
            print(f"Notice: Removing {word} from Modified Category Dictionary")
            del mod_cat_to_embed[word]

    return keyword_to_embed, mod_cat_to_embed
