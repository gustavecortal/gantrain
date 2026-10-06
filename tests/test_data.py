"""Preparing photographs, reading manifests, the device's precision, and the metrics."""

from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gantrain.curate import curate
from gantrain.gan.config import DataConfig
from gantrain.gan.data import training_images
from gantrain.imageio import ORIENTATION_TAG, is_landscape, load_oriented
from gantrain.manifest import Record, images_in, read_manifest, write_manifest

from .conftest import write_photos


def test_curate_drops_duplicates_and_splits_without_overlap(tmp_path):
    photos = write_photos(tmp_path / "photos", n=8)
    shutil.copy(photos[0], tmp_path / "photos" / "exact-copy.png")
    with Image.open(photos[1]) as im:
        im.resize((192, 128)).save(tmp_path / "photos" / "bigger-copy.png")

    report = curate([tmp_path / "photos"], tmp_path / "curated", semantic=False,
                    test_fraction=0.25, min_test=1, workers=1, progress=False)
    assert report["hash_duplicates"] == 2 and report["kept"] == 8

    train = read_manifest(tmp_path / "curated" / "manifest_train.jsonl")
    test = read_manifest(tmp_path / "curated" / "manifest_test.jsonl")
    assert len(train) == 6 and len(test) == 2
    assert not {r.path for r in train} & {r.path for r in test}
    kept = {r.path for r in train + test}
    assert str((tmp_path / "photos" / "bigger-copy.png").resolve()) in kept, "the larger copy wins"


def test_relative_manifest_paths_resolve_against_the_manifest(tmp_path, monkeypatch):
    """Training and `evaluate` both read through `images_in`, from any working directory."""
    write_photos(tmp_path / "set" / "img", n=3)
    write_manifest(tmp_path / "set" / "manifest.jsonl",
                   [Record(id=str(i), path=f"img/{i}.png", width=96, height=64, split="train")
                    for i in range(3)])
    monkeypatch.chdir(tmp_path.parent)
    want = sorted(str((tmp_path / "set" / "img" / f"{i}.png").resolve()) for i in range(3))

    found = sorted(str(p.resolve()) for p in training_images(
        DataConfig(path=str(tmp_path / "set")), (256, 384)))
    assert found == want
    from_manifest = images_in(tmp_path / "set" / "manifest.jsonl")
    assert sorted(str(Path(r.path).resolve()) for r in from_manifest) == want


def test_an_older_manifest_with_dropped_fields_still_loads(tmp_path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"id": "a", "source": "local", "path": "a.jpg", "caption": "x",
                             "bucket": [512, 512], "split": "train"}) + "\n", encoding="utf-8")
    (rec,) = read_manifest(p)
    assert rec.path == "a.jpg" and rec.split == "train"
    assert rec.extra == {"source": "local", "caption": "x", "bucket": [512, 512]}


def test_exif_orientation_is_honoured(tmp_path):
    """A landscape file tagged to turn a quarter loads, and is filtered, as portrait."""
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = 6
    Image.new("RGB", (96, 64)).save(tmp_path / "rotated.jpg", exif=exif)
    Image.new("RGB", (96, 64)).save(tmp_path / "plain.jpg")

    assert is_landscape(tmp_path / "plain.jpg") is True
    assert is_landscape(tmp_path / "rotated.jpg") is False
    assert load_oriented(tmp_path / "rotated.jpg").size == (64, 96)
    landscape = training_images(DataConfig(path=str(tmp_path)), (256, 384))
    assert [p.name for p in landscape] == ["plain.jpg"]


@pytest.mark.parametrize("bf16", [True, False])
def test_training_precision_is_bf16_or_fp32_never_fp16(monkeypatch, bf16):
    from gantrain import device as dev

    monkeypatch.setenv("GANTRAIN_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(name="card", total_memory=8 << 30))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    dev.get_caps.cache_clear()
    try:
        caps = dev.get_caps()
        assert caps.autocast_dtype is (torch.bfloat16 if bf16 else None)
        assert caps.training_precision == ("bf16 mixed precision" if bf16 else "fp32")
    finally:
        monkeypatch.undo()
        dev.get_caps.cache_clear()
    assert dev.get_caps().autocast_dtype is None, "the CPU trains in fp32"


def test_the_metrics_on_tiny_inputs():
    from gantrain.evaluate import cmmd, frechet_distance, kid, precision_recall, vendi_score

    rng = np.random.default_rng(0)
    real = rng.normal(size=(60, 8))
    assert vendi_score(np.eye(5)) == pytest.approx(5.0)
    assert vendi_score(np.ones((5, 3))) == pytest.approx(1.0)
    assert cmmd(real, real) == pytest.approx(0.0, abs=1e-9)
    assert frechet_distance(real, real) == pytest.approx(0.0, abs=1e-6)
    assert frechet_distance(real, real + 3) > frechet_distance(real, real + 1)
    assert kid(real, real + 3)[0] > kid(real, real)[0]

    collapsed = rng.normal(scale=0.01, size=(60, 8))
    precision, recall = precision_recall(real, collapsed)
    assert precision == 1.0 and recall < 0.2
