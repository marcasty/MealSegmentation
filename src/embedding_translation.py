import numpy as np
from FoodMetadataCOCO import FoodMetadata
from scipy import spatial
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

    # Download 
    if not os.path.exists(f'{model_dir}/glove.6B.zip'):
        print('Downloading GloVe')
        urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip',f'{model_dir}/glove.6B.zip')
    if not os.path.exists('f{model_dir}/glove.6B.200d.txt'):
        print('Unzipping GloVe')
        with zipfile.ZipFile(f'{model_dir}/glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{model_dir}')

    print('Creating Dictionary of GloVe Embeddings')
    embed_dict = {}
    with open(f'{model_dir}/glove.6B.200d.txt','r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:],'float32')
            embed_dict[word]=vector
    return embed_dict

def get_average_embedding(embed_dict, sentence):
  """returns average embedding of all words in fragment"""
  words = sentence.split()

  # skip categories that don't appear in validation set
  if words[0][0] == '$': return []
  
  vectors = []
  for word in words:
    try:
      vectors.append(embed_dict[word.lower()])
    except:
      print(f"Warning: Word '{word}' not found in embeddings.")
      continue
  if not vectors:
    print(f"Warning: No valid embeddings found in sentence: {sentence}")     
  return np.mean(vectors, axis=0)

def find_similar_word(spacy_dict, cat_dict):
    """
    translates spacy words to modified categories
    Args:
    spacy_dict -> spacy word : embd
    cat_dict -> modified category word : embd
    Returns:
    spacy_to_cat -> spacy word : modified category word
    """
    categories = []
    for word, embedding in spacy_dict.items():
        nearest = sorted(cat_dict.keys(), key=lambda word: spatial.distance.euclidean(cat_dict[word], embedding))
        categories.append(nearest[0])
    return categories

def assign_classes(metadata, category_names, embedding_vars):
    
    if embedding_vars[0] == "GloVe":
        embed_dict = download_glove(embedding_vars[1])

    with open(embedding_vars[2], "r" ) as f:
        cats = f.readlines()
        mod_category_names = [cat.strip() for cat in cats]


    # create a dictionary category name : modded category name
    cat2mod = {}
    for cat, mod in zip(category_names, mod_category_names):
        cat2mod[cat] = mod

    # modded category name : embedding   
    embedded_cats_dict = {}
    for cat in mod_category_names:
        if cat[0] == '$': continue
        embedded_cats_dict[cat] = get_average_embedding(embed_dict, cat) 

    # for each annotation, embed the spacy words and find the nearest modified category
    for ann_id, ann in metadata.anns.items():
        words = ann["spacy"]
        spacy_dict = {}
        for word in words:
            spacy_dict[word] = get_average_embedding(embed_dict, word)
        # add the nearest modified classes and the associated classes to json
        mod_classes = find_similar_word(spacy_dict, embedded_cats_dict)
        classes = [cat for cat, mod in cat2mod.items() if mod in mod_classes]
        metadata.add_class_from_embd(ann_id, mod_classes, classes)

if __name__ == '__main__':
    HOME = 'C:/Users/marka/fun'

    model_dir = f'{HOME}/MealSegmentation/tmp/embd_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # download glove and save to some directory; this also happens in "assign_class()"
    #download_glove(model_dir)

    # the model
    embd_model_type = "GloVe"

    # the path to the hand-modified category names
    modded_cat_path = f'{HOME}/MealSegmentation/round2_categories_modified.txt'
    embedding_vars = [embd_model_type, model_dir, modded_cat_path]

    # import a json file that has blip2/spacy annotations
    metadata = FoodMetadata(f'{HOME}/MealSegmentation/public_validation_set_2.1_blip_spacy.json')
    category_ids = metadata.loadCats(metadata.getCatIds())
    category_names = [_["name_readable"] for _ in category_ids]

    # find the nearest class to each blip2/spacy word
    assign_classes(metadata, category_names, embedding_vars)

    # look at one example
    print(metadata.dataset['annotations'][0])

    # save it so you can check it out :)
    metadata.export_coco(new_file_name='embedding_test.json')
