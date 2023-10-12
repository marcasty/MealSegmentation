import numpy as np


def run_glove(text: str, **kwargs) -> np.ndarray:
    """given text, produce embedding"""
    words = text.split()

    if 'glove' in kwargs:
        model = kwargs['glove']
    else:
        raise AssertionError("No Glove Model Provided")

    # skip categories that don't appear in validation set
    try:
        if words[0][0] == '$':
            return []
    except Exception as e:
        print(f'Exception is {e} and unknown word is {text}')
    vectors = []
    for word in words:
        try:
            vectors.append(model[word.lower()])
        except Exception as e:
            print(f"Warning: Word '{word}' not found in embeddings.")
            print(f'Exception is {e}')
        continue
    if len(vectors) < 1:
        print(f"Warning: No valid embeddings found in sentence: {text}")
        return 0

    return np.mean(vectors, axis=0)


def run_mistral(text: str, **kwargs) -> np.ndarray:
    """given text, produce embedding"""

    if 'mistral' in kwargs:
        model = kwargs['mistral']
    else:
        raise AssertionError("No Mistral Model Provided")

    embedding = model(text)
    return embedding


def run_llama2(text: str, **kwargs) -> np.ndarray:
    """given text, produce embedding"""

    if 'llama2' in kwargs:
        model = kwargs['llama2']
    else:
        raise AssertionError("No LLama2 Model Provided")

    embedding = model(text)
    return embedding
