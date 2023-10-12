from string import punctuation


def run_spacy(text: str, **kwargs) -> set:
    """given text, produce hot words"""

    if 'spacy' in kwargs:
        spacy = kwargs['spacy']
    else:
        raise AssertionError("No Spacy Model Provided")

    result = []
    pos_tag = ['PROPN', 'NOUN']
    doc = spacy(text.lower())
    for token in doc:
        if (token.text in spacy.Defaults.stop_words or token.text in punctuation):
            continue
        elif (token.pos_ in pos_tag):
            result.append(token.text)
    return set(result)
