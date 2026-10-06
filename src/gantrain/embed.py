"""Image embeddings, for semantic deduplication and the evaluation metrics."""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from . import device as dev
from .imageio import open_drafted

DINO = "dinov2-base"
CLIP = "clip-vit-l14"
BACKBONES = {DINO: "facebook/dinov2-base", CLIP: "openai/clip-vit-large-patch14"}


@functools.lru_cache(maxsize=2)
def _load(name: str):
    """A backbone and its preprocessor, in fp32 on the active device."""
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    if name == CLIP:
        proc = CLIPImageProcessor.from_pretrained(BACKBONES[name])
        model = CLIPVisionModelWithProjection.from_pretrained(BACKBONES[name])
    else:
        proc = AutoImageProcessor.from_pretrained(BACKBONES[name])
        model = AutoModel.from_pretrained(BACKBONES[name])
    return proc, model.to(dev.get_caps().device).eval().requires_grad_(False)


def _decode(item: str | Path | Image.Image) -> Image.Image:
    """RGB, decoded at the smallest scale the backbones' 224-pixel input allows."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    with open_drafted(item, min_edge=256, short_edge=True) as im:
        return im.convert("RGB")


@torch.no_grad()
def embed(images: Sequence[str | Path | Image.Image], backbones: Sequence[str] = (DINO,),
          batch_size: int = 16, progress: bool = True) -> list[np.ndarray]:
    """One `(n, d)` float32 array per backbone. Each image is decoded once for all of them."""
    out: list[list[np.ndarray]] = [[] for _ in backbones]
    device = dev.get_caps().device
    starts = range(0, len(images), batch_size)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        for i in tqdm(starts, desc="embed", unit="batch",
                      disable=not progress or len(images) <= batch_size):
            batch = list(pool.map(_decode, images[i:i + batch_size]))
            for feats, name in zip(out, backbones, strict=True):
                proc, model = _load(name)
                px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
                result = model(pixel_values=px)
                feat = result.image_embeds if name == CLIP else result.last_hidden_state[:, 0]
                feats.append(feat.float().cpu().numpy())
    return [np.concatenate(f) if f else np.zeros((0, 1), np.float32) for f in out]


def unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
