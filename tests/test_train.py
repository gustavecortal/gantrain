"""Training, checkpoints, resuming, and what is made from a checkpoint."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gantrain.gan.config import TrainConfig
from gantrain.gan.models import Generator
from gantrain.gan.sample import generate, interpolate_video, load_generator, training_images_of
from gantrain.gan.train import FastGANTrainer, train

from .conftest import tiny_config, write_photos


def _checkpoint(cfg: TrainConfig) -> Path:
    return Path(cfg.out_dir) / "checkpoints" / f"{cfg.iterations:07d}.pt"


def test_a_short_run_writes_samples_logs_and_checkpoints(trained_run):
    out = Path(trained_run.out_dir)
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["0000002.pt", "0000004.pt"]
    assert len(list((out / "samples").glob("*.jpg"))) == 2
    assert (out / "validation" / "progress.jpg").exists()
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    losses = [row for row in log if "loss_g" in row]
    assert {"loss_d_real", "loss_d_fake", "loss_rec", "d_sep"} <= set(losses[0])
    assert json.loads((out / "config.json").read_text())["precision"] == "fp32"


def test_ganlive_can_load_the_checkpoint(trained_run):
    """Open it the way ganlive does: `weights_only`, then `g_ema` and the architecture keys."""
    ckpt = torch.load(_checkpoint(trained_run), map_location="cpu", weights_only=True, mmap=True)
    assert {"g", "d", "g_ema", "opt_g", "opt_d", "step", "config"} <= set(ckpt)
    assert ckpt["step"] == trained_run.iterations
    architecture = {k: ckpt["config"][k] for k in ("nz", "ngf", "im_size", "im_width")}
    net = Generator(**architecture)
    net.load_state_dict(ckpt["g_ema"])

    ours, _ = load_generator(_checkpoint(trained_run), "cpu")
    z = torch.randn(2, architecture["nz"])
    torch.manual_seed(0)
    a = net.eval()(z)[0]
    torch.manual_seed(0)
    assert torch.equal(a, ours(z)[0])


def test_resume_continues_the_run_and_refuses_a_changed_setting(trained_run, tmp_path):
    ckpt = str(_checkpoint(trained_run))
    moved = tmp_path / "moved"
    shutil.copytree(trained_run.data.path, moved)
    cfg = replace(trained_run, out_dir=str(tmp_path / "resumed"), iterations=6, resume=ckpt,
                  data=replace(trained_run.data, path=str(moved)))
    train(cfg)
    assert torch.load(_checkpoint(cfg), weights_only=True)["step"] == 6

    with pytest.raises(SystemExit, match="lr"):
        FastGANTrainer(replace(cfg, lr=1e-3)).load(ckpt)
    trainer = FastGANTrainer(replace(cfg, lr=1e-3, accept_config_change=True))
    trainer.load(ckpt)
    assert trainer.step_count == 4
    assert trainer.optG.param_groups[0]["lr"] == 1e-3


def test_an_older_checkpoint_still_loads(trained_run, tmp_path):
    """Older configs carry options this version dropped, and compiled runs prefix every key."""
    ckpt = torch.load(_checkpoint(trained_run), weights_only=True)
    ckpt["config"] |= {"loss": "hinge", "lecam_weight": 0.0, "pixelshuffle_from": 0,
                       "mapping_depth": 0, "channels_last": True}
    ckpt["g_ema"] = {f"_orig_mod.{k}": v for k, v in ckpt["g_ema"].items()}
    old = tmp_path / "old.pt"
    torch.save(ckpt, old)
    _, cfg = load_generator(old, "cpu")
    assert cfg.im_width == trained_run.im_width
    FastGANTrainer(trained_run).load(old)

    ckpt["config"]["mapping_depth"] = 4
    torch.save(ckpt, old)
    with pytest.raises(ValueError, match="mapping_depth"):
        load_generator(old, "cpu")


def test_the_jacobian_clamp_acts_only_outside_its_band(tmp_path):
    from gantrain.gan.losses import JACOBIAN_BAND, jacobian_clamp

    lo, hi = JACOBIAN_BAND
    assert jacobian_clamp(torch.tensor([lo * 2, hi / 2])) == 0
    assert jacobian_clamp(torch.tensor([hi * 3])) > 0
    assert jacobian_clamp(torch.tensor([lo / 3])) > 0

    trainer = FastGANTrainer(tiny_config(tmp_path, jacobian_weight=1.0))
    stats = trainer.step(torch.randint(0, 256, (2, 3, 256, 384), dtype=torch.uint8))
    assert {"jac", "jac_q"} <= set(stats)


def test_sample_and_video_write_files(trained_run, tmp_path):
    ckpt = _checkpoint(trained_run)
    first = generate(ckpt, tmp_path / "a", n=3, batch_size=2, seed=7)
    again = generate(ckpt, tmp_path / "b", n=3, batch_size=2, seed=7)
    assert len(first) == 3
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in again]

    clip = interpolate_video(ckpt, tmp_path / "clip.mp4", seconds=0.2, fps=10, batch_size=2)
    assert clip.stat().st_size > 0


def test_sample_points_evaluate_at_the_runs_training_data(tmp_path):
    curated, plain = tmp_path / "curated", tmp_path / "plain"
    write_photos(plain, n=1)
    curated.mkdir()
    (curated / "manifest_train.jsonl").write_text("", encoding="utf-8")
    for folder, want in ((curated, curated / "manifest_train.jsonl"), (plain, plain)):
        torch.save({"config": {"data": {"path": str(folder)}}}, tmp_path / "c.pt")
        assert training_images_of(tmp_path / "c.pt") == want


def test_the_train_command_passes_its_options_through(monkeypatch):
    """Its defaults are the config's, and `--accept-config-change` reaches the trainer."""
    from typer.testing import CliRunner

    from gantrain.cli import app

    seen = {}

    def fake_train(cfg):
        seen["cfg"] = cfg
        return {"images_per_second": 0.0, "peak_vram_gb": 0.0}

    monkeypatch.setattr("gantrain.gan.train.train", fake_train)
    result = CliRunner().invoke(app, ["train", "--accept-config-change"])
    assert result.exit_code == 0, result.output
    assert seen["cfg"] == replace(TrainConfig(), accept_config_change=True)
