import torch
import numpy as np
from typing import Union

from utils import assert_input

global DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_blip2(image: Union[np.ndarray, torch.Tensor], **kwargs) -> str:
    """given RGB image, produce text"""

    print("Please ensure input image is in RGB format!")
    image = assert_input(image)

    if 'blip2_model' in kwargs:
        blip2_model = kwargs['blip2_model']
    else:
        raise AssertionError("No BLIP2 Model Provided")
    if 'blip2_processor' in kwargs:
        blip2_processor = kwargs['blip2_processor']
    else:
        raise AssertionError("No BLIP2 Processor Provided")
    if 'device' in kwargs:
        device = kwargs['device']
    else:
        print(f"No device found, using baseline computing device: {DEVICE}")
        device = DEVICE

    prompt = "the food or foods in this image include: "
    inputs = blip2_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip2_model.generate(**inputs, max_new_tokens=20)
    generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = generated_text[0].strip()
    return generated_text


def run_llava15(image: Union[np.ndarray, torch.Tensor], **kwargs) -> str:
    """given RGB image, produce text"""

    # Import helper functions to run_llava
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token

    print("Please ensure input image is in RGB format!")
    image = assert_input(image)

    if 'llava_model' in kwargs:
        model = kwargs["llava_model"]
    else:
        raise AssertionError("No Llava 1.5 Model Provided")
    if 'llava_tokenizer' in kwargs:
        tokenizer = kwargs["llava_tokenizer"]
    else:
        raise AssertionError("No Llava 1.5 Text Tokenizer Provided")
    if 'llava_image_processor' in kwargs:
        image_processor = kwargs["llava_image_processor"]
    else:
        raise AssertionError("No Llava 1.5 Image Processor Provided")
    if 'device' in kwargs:
        device = kwargs["device"]
    else:
        print(f"No device found, using baseline computing device: {DEVICE}")
        device = DEVICE
    if 'temperature' in kwargs:
        temperature = kwargs["temperature"]
    else:
        temperature = 0.2
    if 'top_p' in kwargs:
        top_p = kwargs["top_p"]
    else:
        top_p = 0.1
    if 'num_beams' in kwargs:
        num_beams = kwargs["num_beams"]
    else:
        num_beams = 1

    qs = "What are all the food items in this image? Please be as specific and detailed as possible."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=device)

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

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
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    generated_text = outputs.strip()
    return generated_text
