import torch
import numpy as np
from typing import Union, List

from utils import assert_input

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_sam(image: Union[np.ndarray, torch.Tensor], **kwargs) -> List:
    """given an image, generate masks automatically"""

    print("Please ensure input image is in RGB format!")
    image = assert_input(image)

    if "sam_model" in kwargs:
        model = kwargs["sam_model"]
    else:
        raise AssertionError("No SAM Model Provided")
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        print(f"No device found, using baseline computing device: {DEVICE}")
        device = DEVICE

    image.to(device=device)
    masks = model.generate(image)
    return masks
