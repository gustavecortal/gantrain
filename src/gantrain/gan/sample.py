"""Images and interpolation videos from a trained checkpoint."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torchvision.utils import save_image

from .. import device as dev
from .config import TrainConfig
from .models import Generator, denormalise, freeze_noise

_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(state: dict) -> dict:
    """Drop the `_orig_mod.` that `torch.compile` puts on every key, so either kind loads."""
    if not any(k.startswith(_COMPILE_PREFIX) for k in state):
        return state
    return {k.removeprefix(_COMPILE_PREFIX): v for k, v in state.items()}


def _open(checkpoint: str | Path) -> dict:
    """A checkpoint with its tensors left on disk until touched. Never runs code from it."""
    return torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)


def load_generator(checkpoint: str | Path, device: torch.device | str | None = None,
                   ) -> tuple[Generator, TrainConfig]:
    """The EMA generator and the config it was trained with."""
    ckpt = _open(checkpoint)
    cfg = TrainConfig.from_saved(ckpt.get("config", {}))
    net = Generator(**cfg.generator_kwargs)
    net.load_state_dict(strip_compile_prefix(ckpt["g_ema"]))
    target = torch.device(device) if device is not None else dev.get_caps().device
    return net.to(target).eval().requires_grad_(False), cfg


def training_images_of(checkpoint: str | Path) -> Path | None:
    """What a checkpoint trained on: its data folder's train manifest, else the folder."""
    path = (_open(checkpoint).get("config", {}).get("data") or {}).get("path")
    if not path:
        return None
    manifest = Path(path) / "manifest_train.jsonl"
    return manifest if manifest.exists() else Path(path)


@torch.no_grad()
def generate(checkpoint: str | Path, out_dir: str | Path, n: int = 64, batch_size: int = 16,
             seed: int = 0) -> list[Path]:
    """Write `n` PNGs. The same seed gives the same latents on every device."""
    net, cfg = load_generator(checkpoint)
    device = next(net.parameters()).device
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z_all = torch.randn(n, cfg.nz, generator=dev.cpu_generator(seed))
    torch.manual_seed(seed)
    written: list[Path] = []
    for start in range(0, n, batch_size):
        with dev.autocast(cfg.amp):
            big = net(z_all[start:start + batch_size].to(device))[0]
        for img in denormalise(big):
            written.append(out / f"{len(written):05d}.png")
            save_image(img, written[-1])
    dev.empty_cache()
    return written


def slerp_path(z0: torch.Tensor, z1: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """Latents along the great circle from `z0` to `z1`, one row per `t` in `ts`."""
    omega = torch.clamp((z0 / z0.norm() * z1 / z1.norm()).sum(), -1.0, 1.0).acos()
    ts = ts.to(z0.device, z0.dtype).view(-1, 1)
    if omega.abs() < 1e-6:
        return z0.view(1, -1) + ts * (z1 - z0).view(1, -1)
    a = ((1.0 - ts) * omega).sin() / omega.sin()
    b = (ts * omega).sin() / omega.sin()
    return a * z0.view(1, -1) + b * z1.view(1, -1)


def interpolation_schedule(frames: int, loop: bool) -> torch.Tensor:
    """Each frame's position `t` between the endpoints: a raised cosine there and back when
    looping, so the clip repeats seamlessly, else a straight sweep."""
    i = torch.arange(frames, dtype=torch.float32)
    if not loop:
        return i / max(frames - 1, 1)
    return 0.5 - 0.5 * torch.cos(2 * torch.pi * i / frames)


@torch.no_grad()
def interpolate_video(checkpoint: str | Path, out_path: str | Path, seconds: float = 30.0,
                      fps: int = 30, batch_size: int = 8, seed: int = 0, loop: bool = True,
                      noise_seed: int = 0, fourcc: str = "mp4v") -> Path:
    """Render a clip that travels between two latents chosen by `seed`."""
    net, cfg = load_generator(checkpoint)
    freeze_noise(net, seed=noise_seed)
    device = next(net.parameters()).device
    frames = max(int(round(seconds * fps)), 1)
    gen = dev.cpu_generator(seed)
    z0, z1 = torch.randn(cfg.nz, generator=gen), torch.randn(cfg.nz, generator=gen)
    path = slerp_path(z0, z1, interpolation_schedule(frames, loop)).to(device)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = cfg.ladder.shape
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps),
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a {fourcc!r} writer for {out}")
    try:
        for start in range(0, frames, batch_size):
            with dev.autocast(cfg.amp):
                big = net(path[start:start + batch_size])[0]
            rgb = denormalise(big).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
            for frame in rgb.cpu().numpy():
                writer.write(frame[:, :, ::-1])
    finally:
        writer.release()
    dev.empty_cache()
    return out
