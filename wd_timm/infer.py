from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(
            f"selected_tags.csv failed to download from {repo_id}"
        ) from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs_ = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs_[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs_[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(gen_labels.items(), key=lambda item: item[1], reverse=True)
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs_[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(char_labels.items(), key=lambda item: item[1], reverse=True)
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


def get_infer_batch(
    model_name: str,
    gen_threshold: float,
    char_threshold: float,
) -> Callable[[List[Path | str]], None]:
    repo_id = MODEL_REPO_MAP.get(model_name) or ""

    print(f"Loading model '{model_name}' from '{repo_id}'...")
    model: nn.Module = timm.create_model("hf-hub:" + repo_id).eval()
    state_dict = timm.models.load_state_dict_from_hf(repo_id)
    model.load_state_dict(state_dict)

    print("Loading tag list...")
    labels = load_labels_hf(repo_id=repo_id)

    print("Creating data transform...")
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    print("Setting up inference function...")

    def infer_batch(batch: List[Path | str]):
        imgs = []
        for image_path in batch:
            img_input = Image.open(image_path)
            img_input = pil_ensure_rgb(img_input)
            img_input = pil_pad_square(img_input)
            inputs = transform(img_input).unsqueeze(0)  # type: ignore
            inputs = inputs[:, [2, 1, 0]]
            imgs.append(inputs)

        inputs = torch.cat(imgs)

        with torch.inference_mode():
            if torch.cuda.is_available():
                model.to("cuda")
                inputs = inputs.to("cuda")

            outputs = model(inputs)
            outputs = F.sigmoid(outputs)

            if torch.cuda.is_available():
                inputs = inputs.to("cpu")
                outputs = outputs.to("cpu")
                model.to("cpu")

        for i, image_path in enumerate(batch):
            output = outputs[i]
            caption, taglist, ratings, character, general = get_tags(
                probs=output,
                labels=labels,
                gen_threshold=gen_threshold,
                char_threshold=char_threshold,
            )
            print(f"Image: {image_path}")
            print("--------")
            print(f"Caption: {caption}")
            print("--------")
            print(f"Tags: {taglist}")

    return infer_batch
