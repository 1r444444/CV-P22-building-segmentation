import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def load_and_preprocess(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = np.array(image, dtype=np.float32)
    mask = np.array(mask, dtype=np.uint8)

    # Scale RGB values from [0, 255] to [0, 1].
    image = image / 255.0

    # Normalize for the pretrained ResNet18 encoder.
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    # Convert the segmentation mask from {0, 255} to {0, 1}.
    mask = (mask > 0).astype(np.float32)

    # PyTorch expects channels first: C x H x W.
    image = torch.from_numpy(image).permute(2, 0, 1)

    # Add the single mask channel: 1 x H x W.
    mask = torch.from_numpy(mask).unsqueeze(0)

    return image, mask