import urllib.request
import zipfile
import os

model_dir = '/tmp/glove'

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
