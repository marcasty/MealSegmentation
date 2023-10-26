import sys
import os

# Add the parent directory of your project to the sys.path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

import torch
import matplotlib.pyplot as plt
import time

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from src.image_to_caption import run_blip2, run_llava15

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME = "/tmp"
os.environ["BLIP2_MODEL"] = "Salesforce/blip2-opt-2.7b"
os.environ["LLAVA_MODEL"] = "liuhaotian/llava-v1.5-13b"

random_image = False
if random_image:
    n = 128
    m = 160
    input_tensor = torch.clip(torch.randn(n, m, 3), 0, 1)
else:
    input_tensor = plt.imread("../images/miso-soup/miso-soup_00011.jpg")
    input_tensor = plt.imread("../images/panna_cotta.jpg")
    input_tensor = plt.imread("../images/pork_bun.jpg")

# BLIP2_MODEL = os.environ['BLIP2_MODEL']
# blip2_processor_ = AutoProcessor.from_pretrained(BLIP2_MODEL)
# blip2_model_ = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL, torch_dtype=torch.float16).to(DEVICE)
# blip2_model_.eval()
# generated_text = run_blip2(input_tensor, blip2_model=blip2_model_, blip2_processor=blip2_processor_)
# print(generated_text)

LLAVA_MODEL = os.environ["LLAVA_MODEL"]
model_name = get_model_name_from_path(LLAVA_MODEL)
llava_tokenizer_, llava_model_, llava_image_processor_, _ = load_pretrained_model(
    LLAVA_MODEL,
    None,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device=DEVICE,
)
for i in range(20):
    start = time.time()
    generated_text = run_llava15(
        input_tensor,
        llava_tokenizer=llava_tokenizer_,
        llava_model=llava_model_,
        llava_image_processor=llava_image_processor_,
        temperature=0.5,
    )
    print(f"Time Taken: {time.time() - start}")
    print(generated_text)
