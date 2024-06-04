import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import jax
import numpy as np
import pandas as pd
from tqdm import tqdm
from flax import linen, serialization
from flax import struct
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image

from wd.models import model_registry

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2_v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "swinv2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


@struct.dataclass
class PredModel:
    apply_fun: Callable = struct.field(pytree_node=False)
    params: Any = struct.field(pytree_node=True)

    def jit_predict(self, x):
        # Not actually JITed since this is a single shot script,
        # but this is the function you would decorate with @jax.jit
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        return preds


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


def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    # Resize
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize(
            (target_size, target_size),
            Image.Resampling.BICUBIC,
        )
    return image


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


def load_model_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Tuple[PredModel, int]:
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.msgpack",
        revision=revision,
        token=token,
    )

    model_config = hf_hub_download(
        repo_id=repo_id,
        filename="sw_jax_cv_config.json",
        revision=revision,
        token=token,
    )

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}  # type: ignore

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = model_registry[model_name]()
    model = model_builder.build(
        config=model_builder,
        **model_config["model_args"],
    )
    model = PredModel(model.apply, params=variables)

    return model, model_config["image_size"]


def get_tags(
    probs: Any,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(
        sorted(
            gen_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(
        sorted(
            char_labels.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


def get_infer_batch(
    model_name: str,
    batch_size=32,
    gen_threshold=0.35,
    char_threshold=0.85,
) -> Callable[[List[Path]], List[Any]]:
    print("Using jax...")
    repo_id = MODEL_REPO_MAP.get(model_name) or ""

    print(f"Loading model '{model_name}' from '{repo_id}'...")
    model, target_size = load_model_hf(repo_id=repo_id)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    def infer_files(batch: List[Path]):
        def infer_batch(batch: List[Path]):
            imgs = []
            for image_path in batch:
                img_input = Image.open(image_path)
                img_input = pil_ensure_rgb(img_input)
                img_input = pil_pad_square(img_input)
                img_input = pil_resize(img_input, target_size)
                input_ = np.array(img_input)
                input_ = np.expand_dims(input_, axis=0)
                input_ = input_[..., ::-1]

                imgs.append(input_)

            inputs = jax.numpy.concat(imgs)
            outputs = model.predict(inputs)

            results = []
            for i, image_path in enumerate(batch):
                output = outputs[i]

                caption, taglist, ratings, character, general = get_tags(
                    probs=output,
                    labels=labels,
                    gen_threshold=gen_threshold,
                    char_threshold=char_threshold,
                )

                results.append(
                    (image_path, caption, taglist, ratings, character, general)
                )

            return results

        # Process in batches
        results = []
        for i in tqdm(range(0, len(batch), batch_size)):
            sub_batch = batch[i : i + batch_size]
            batch_results = infer_batch(sub_batch)
            results.extend(batch_results)

        # Print the results
        for image_path, caption, taglist, ratings, character, general in results:
            print(f"Image: {image_path}")
            print("--------")
            print(f"Caption: {caption}")
            print("--------")
            print(f"Taglist: {taglist}")
            print("--------")
            print(f"Ratings: {ratings}")
            print("--------")
            print(f"Character: {character}")
            print("--------")
            print(f"General: {general}")
            print("--------")

        return results

    return infer_files
