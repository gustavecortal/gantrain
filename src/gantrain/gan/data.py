"""Training images as uint8 tensors of the run's shape, in an endless shuffled stream."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from ..imageio import CropToCanvas, is_landscape, load_oriented
from ..manifest import images_in
from .config import DataConfig


def build_transform(cfg: DataConfig, shape: tuple[int, int]) -> transforms.Compose:
    """A loaded photograph -> one uint8 CHW tensor of `shape` = (height, width)."""
    if cfg.view == "full":
        ops: list = [CropToCanvas(shape)]
    else:
        aspect = shape[1] / shape[0]
        ops = [transforms.RandomResizedCrop(
            shape, scale=cfg.crop_scale, ratio=(aspect, aspect),
            interpolation=transforms.InterpolationMode.BICUBIC)]
    if cfg.hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.PILToTensor())
    return transforms.Compose(ops)


def training_images(cfg: DataConfig, shape: tuple[int, int]) -> list[Path]:
    """The images to train on. A non-square canvas keeps only images of its own orientation,
    judged from the manifest's sizes or, for a plain folder, from each file's header."""
    records = images_in(cfg.path, cfg.split)
    height, width = shape
    if height != width:
        landscape = width > height

        def matches(rec) -> bool:
            size = rec.oriented_size
            found = size[0] >= size[1] if size else is_landscape(rec.path)
            return found is None or found == landscape

        records = [r for r in records if matches(r)]
    if not records:
        raise FileNotFoundError(f"no {cfg.split!r} images in {cfg.path} for a "
                                f"{height}x{width} canvas")
    return [Path(r.path) for r in records]


class ImageDataset(Dataset):
    def __init__(self, cfg: DataConfig, shape: tuple[int, int]) -> None:
        self.files = training_images(cfg, shape)
        self.transform = build_transform(cfg, shape)
        self.min_edge = min(shape)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(load_oriented(self.files[i], self.min_edge, short_edge=True))


class InfiniteSampler(Sampler[int]):
    """An endless stream of indices, reshuffled each pass from a seeded generator."""

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = n
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.n, generator=g).tolist()

    def __len__(self) -> int:
        return 2**31  # the loader asks for a length; this stream has none


def build_loader(cfg: DataConfig, batch_size: int, shape: tuple[int, int], seed: int,
                 pin_memory: bool) -> tuple[Iterator[torch.Tensor], int]:
    """An endless batch iterator, and how many images it draws from."""
    ds = ImageDataset(cfg, shape)
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=InfiniteSampler(len(ds), seed),
        num_workers=cfg.workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=cfg.workers > 0, prefetch_factor=2 if cfg.workers > 0 else None)
    return iter(loader), len(ds)
