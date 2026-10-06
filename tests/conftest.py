"""Force the CPU so the suite never touches the GPU."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def _force_cpu():
    from gantrain import device as dev

    previous = os.environ.get("GANTRAIN_DEVICE")
    os.environ["GANTRAIN_DEVICE"] = "cpu"
    dev.get_caps.cache_clear()
    yield
    if previous is None:
        os.environ.pop("GANTRAIN_DEVICE", None)
    else:
        os.environ["GANTRAIN_DEVICE"] = previous
    dev.get_caps.cache_clear()


def write_photos(folder, n: int = 6, size=(96, 64), seed: int = 0) -> list:
    """`n` distinct noise images, landscape by default."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        paths.append(folder / f"{i}.png")
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(paths[-1])
    return paths


def tiny_config(tmp_path, **overrides):
    """A run small enough to train for a few steps on the CPU."""
    from gantrain.gan.config import DataConfig, TrainConfig

    data = overrides.pop("data", None) or DataConfig(path=str(tmp_path / "photos"), workers=0)
    return TrainConfig(**{"data": data, "out_dir": str(tmp_path / "run"), "im_size": 256,
                          "im_width": 384, "batch_size": 2, "ngf": 8, "ndf": 8,
                          "perceptual_net": "squeeze", "iterations": 4, "log_every": 2,
                          "sample_every": 2, "save_every": 2, "prioritise": False,
                          **overrides})


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A four-step run, shared by the tests that read what training writes."""
    from gantrain.gan.train import train

    tmp_path = tmp_path_factory.mktemp("trained")
    write_photos(tmp_path / "photos")
    cfg = tiny_config(tmp_path)
    train(cfg)
    return cfg
