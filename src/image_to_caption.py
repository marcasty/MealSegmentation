import torch
import numpy as np
from typing import Union
from FoodMetadataCOCO import FoodMetadata
from omegaconf import DictConfig
import cv2
from utils import assert_input


def blip2_setup(blip2_model, device):
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(blip2_model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        blip2_model, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    return model, processor


def run_blip2(image: Union[np.ndarray, torch.Tensor], model, processor, device) -> str:
    """given RGB image, produce text"""

    print("Please ensure input image is in RGB format!")
    image = assert_input(image)

    generated_text, attempts = "", 0
    while not generated_text:
        if attempts > 3:
            print("Warning: BLIP-2 created an empty caption")
            return "FAILURE"

        prompt = "the food or foods in this image include: "
        inputs = processor(image, text=prompt, return_tensors="pt").to(
            device, torch.float16
        )
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        generated_text = generated_text[0].strip()
        attempts += 1
    return generated_text


def llava15_setup(llava_model, device):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(llava_model)
    tokenizer, model, processor, _ = load_pretrained_model(
        llava_model,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device=device,
    )
    return tokenizer, model, processor


def run_llava15(image: Union[np.ndarray, torch.Tensor], model, processor, tokenizer,
                device, temperature=0.2, top_p=0.1, num_beams=1) -> str:
    """given RGB image, produce text"""

    # Import helper functions to run_llava
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token

    print("Please ensure input image is in RGB format!")
    image = assert_input(image)

    qs = "What are all the food items in this image? Please be as specific and detailed as possible."
    if model.config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device=device)
    )

    image_tensor = processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(device=device),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    generated_text = outputs.strip()
    return generated_text


def get_captions(metadata: FoodMetadata, cfg: DictConfig) -> FoodMetadata:

    model_name = cfg.stage.image_to_caption.model
    image_dir = cfg.path.images
    testing = cfg.var.testing
    model_chkpt = cfg.stage.image_to_caption.model_chkpt
    device = cfg.var.device

    if model_name == "blip2":
        model, processor = blip2_setup(model_chkpt, device)
    elif model_name == "llava15":
        tokenizer, model, processor = llava15_setup(model_chkpt, device)
        temperature = cfg.stage.image_to_caption.temperature
        top_p = cfg.stage.image_to_caption.top_p
        num_beams = cfg.stage.image_to_caption.num_beams
    else:
        raise AssertionError(f"Model '{model_name}' not supported for image captioning")

    count = 0
    for cat_id, cat in metadata.cats.items():
        count += 1
        if count > 3 and testing is True:
            return metadata
        print(f'category {count} / 323: {cat["name_readable"]}')

        imgIds = metadata.getImgIds(catIds=cat_id)

        if not imgIds:
            continue
        else:
            imgs = metadata.loadImgs(imgIds)
            for img in imgs:
                image_bgr = cv2.imread(f'{image_dir}/{img["file_name"]}')
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                if model_name == "blip2":
                    caption = run_blip2(image_rgb, model, processor, device)
                elif model_name == 'llava15':
                    caption = run_llava15(image_rgb, model, processor, tokenizer, device, temperature, top_p, num_beams)
                ann_id = metadata.create_annot(img["id"], cat_id)
                metadata.add_annot(ann_id, img["id"], model_name, caption)
    return metadata
