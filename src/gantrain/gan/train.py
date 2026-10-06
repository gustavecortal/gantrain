"""The FastGAN trainer: one step, checkpoints, and the training loop."""

from __future__ import annotations

import copy
import random
import shutil
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

from .. import device as dev
from ..imageio import SAMPLE_GRID_COLUMNS, SAMPLE_GRID_ROWS, contact_sheet, label_image, open_drafted
from ..manifest import append_jsonl, write_json
from .config import TrainConfig, check_resume
from .data import build_loader
from .diffaug import diff_augment
from .losses import (
    JACOBIAN_BAND,
    JACOBIAN_EPSILON,
    JACOBIAN_INTERVAL,
    g_adversarial,
    get_perceptual,
    hinge_d_fake,
    hinge_d_real,
    jacobian_clamp,
    jacobian_ratio,
)
from .models import Discriminator, Generator, denormalise, freeze_noise, init_gan_weights
from .sample import strip_compile_prefix


class LrWarmup:
    """A linear ramp from zero on every parameter group of the given optimisers."""

    def __init__(self, opts, steps: int) -> None:
        self.opts = tuple(opts)
        self.steps = steps
        self.capture()

    def capture(self) -> None:
        """Record the rates the ramp leads to. Call again after changing them."""
        self.targets = [[g["lr"] for g in o.param_groups] for o in self.opts]

    def at(self, step: int) -> None:
        if not self.steps:
            return
        factor = min(1.0, (step + 1) / self.steps)
        for opt, targets in zip(self.opts, self.targets, strict=True):
            for group, target in zip(opt.param_groups, targets, strict=True):
                group["lr"] = target * factor


class Ema:
    """Exponential moving average of a model's weights, held as a second module."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        src, dst = model.state_dict(), self.model.state_dict()
        floats = [k for k, v in src.items() if v.dtype.is_floating_point]
        self._src = [src[k] for k in floats]
        self._dst = [dst[k] for k in floats]
        # Integer buffers (batch-norm step counts) are copied, not averaged.
        self._counters = [(dst[k], v) for k, v in src.items() if not v.dtype.is_floating_point]

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self._dst, self._src, 1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        for dst, src in self._counters:
            dst.copy_(src)
        return self.model.state_dict()


def make_optimizer(params, cfg: TrainConfig) -> torch.optim.Adam:
    kw: dict = {}
    if dev.get_caps().backend == "cuda":
        kw["fused"] = True
    elif dev.get_caps().backend == "xpu":
        # Intel Arc has no fp64, which the fused kernel needs; take the plain per-tensor path.
        kw["foreach"] = False
    return torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), **kw)


def clip_grads(module: nn.Module, max_norm: float) -> None:
    """Clip gradients to a total norm. Norms are taken one tensor at a time (`foreach=False`):
    the multi-tensor `torch._foreach_norm` returns wrong values on some Intel Arc drivers."""
    grads = [p for p in module.parameters() if p.grad is not None]
    if grads and max_norm:
        total = torch.nn.utils.get_total_norm([p.grad for p in grads], foreach=False)
        torch.nn.utils.clip_grads_with_norm_(grads, max_norm, total)


class FastGANTrainer:
    """The two networks, their optimisers and EMA, and one training step."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.caps = dev.get_caps()
        self.device = self.caps.device
        dev.seed_everything(cfg.seed)

        self.netG = Generator(**cfg.generator_kwargs).to(self.device)
        self.netG.apply(init_gan_weights)
        self.netD = Discriminator(**cfg.discriminator_kwargs).to(self.device)
        self.netD.apply(init_gan_weights)
        self.netG = self.netG.to(memory_format=torch.channels_last)
        self.netD = self.netD.to(memory_format=torch.channels_last)

        self.percept = get_perceptual(cfg.perceptual_net, str(self.device))
        self.optG = make_optimizer(self.netG.parameters(), cfg)
        self.optD = make_optimizer(self.netD.parameters(), cfg)
        self.warmup = LrWarmup((self.optG, self.optD), cfg.lr_warmup)
        self.ema = Ema(self.netG, cfg.ema_decay)
        self._quadrants = torch.arange(4, device=self.device)

        self.fwdG, self.fwdD = self.netG, self.netD
        if cfg.compile_model:
            dev.prepare_compile()
            self.fwdG = torch.compile(self.netG, dynamic=False)
            self.fwdD = torch.compile(self.netD, dynamic=False)
        self.step_count = 0

    def sample_z(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.cfg.nz, device=self.device)

    def _jacobian_step(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Odena et al.'s Jacobian clamp: render a pair of nearby latents with the same noise,
        and penalise a pixel change per unit of latent outside `JACOBIAN_BAND`."""
        base = z[: z.shape[0] // 2]
        perturbed = base + JACOBIAN_EPSILON * F.normalize(torch.randn_like(base), dim=1)
        with dev.autocast(self.cfg.amp):
            freeze_noise(self.netG, True)
            try:
                both = self.netG(torch.cat([base, perturbed]))[0]
            finally:
                freeze_noise(self.netG, False)
            q = jacobian_ratio(both[: len(base)], both[len(base):], base, perturbed)
            penalty = jacobian_clamp(q, *JACOBIAN_BAND)
        weighted = self.cfg.jacobian_weight * JACOBIAN_INTERVAL * penalty
        if weighted.detach():
            weighted.backward()
        return {"jac": penalty.detach(), "jac_q": q.detach().median()}

    def step(self, real: torch.Tensor) -> dict[str, torch.Tensor]:
        """One iteration of D then G. Returns detached 0-d tensors."""
        cfg = self.cfg
        real = real.to(self.device, non_blocking=True).to(
            torch.float32, memory_format=torch.channels_last).div_(127.5).sub_(1.0)
        self.warmup.at(self.step_count)

        z = self.sample_z(real.shape[0])
        with dev.autocast(cfg.amp):
            fakes = self.fwdG(z)
        real_in = diff_augment(real)
        fakes_in = [diff_augment(f) for f in fakes]

        # D: hinge on real images plus reconstructing them, then hinge on generated ones.
        self.optD.zero_grad(set_to_none=True)
        with dev.autocast(cfg.amp):
            part = random.randint(0, 3)
            logits_real, recons = self.fwdD(real_in, reconstruct=True,
                                            part=self._quadrants[part])
            loss_rec = self.percept(recons, real_in, part)
            loss_d_real = hinge_d_real(logits_real)
        (loss_d_real + loss_rec).backward()
        with dev.autocast(cfg.amp):
            logits_fake = self.fwdD([f.detach() for f in fakes_in])
            loss_d_fake = hinge_d_fake(logits_fake)
        loss_d_fake.backward()
        clip_grads(self.netD, cfg.max_grad_norm)
        self.optD.step()

        # G: fool the critic, with D's weights held still.
        self.netD.requires_grad_(False)
        self.optG.zero_grad(set_to_none=True)
        with dev.autocast(cfg.amp):
            loss_g = g_adversarial(self.fwdD(fakes_in))
        loss_g.backward()
        stats: dict[str, torch.Tensor] = {}
        if cfg.jacobian_weight and self.step_count % JACOBIAN_INTERVAL == 0:
            stats = self._jacobian_step(z)
        # After the Jacobian term, which adds to the same gradients.
        clip_grads(self.netG, cfg.max_grad_norm)
        self.optG.step()
        self.netD.requires_grad_(True)
        self.ema.update()
        self.step_count += 1

        with torch.no_grad():
            d_sep = torch.stack([r.mean() - f.mean()
                                 for r, f in zip(logits_real, logits_fake, strict=True)]).mean()
        return {"loss_d_real": loss_d_real.detach(), "loss_d_fake": loss_d_fake.detach(),
                "loss_rec": loss_rec.detach(), "loss_g": loss_g.detach(), "d_sep": d_sep,
                **stats}

    @torch.no_grad()
    def sample_grid(self, z: torch.Tensor, noise_seed: int = 0) -> torch.Tensor:
        """EMA samples in [0, 1], with the noise pinned so grids compare across steps."""
        freeze_noise(self.ema.model, True, seed=noise_seed)
        try:
            with dev.autocast(self.cfg.amp):
                return denormalise(self.ema.model(z)[0])
        finally:
            freeze_noise(self.ema.model, False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"g": self.netG.state_dict(), "d": self.netD.state_dict(),
                    "g_ema": self.ema.state_dict(), "opt_g": self.optG.state_dict(),
                    "opt_d": self.optD.state_dict(), "step": self.step_count,
                    "config": asdict(self.cfg)}, path)

    def load(self, path: str | Path, generator_only: bool = False) -> None:
        """Continue a run, or with `generator_only` start a new one from its generator."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.netG.load_state_dict(strip_compile_prefix(ckpt["g"]))
        self.ema.model.load_state_dict(strip_compile_prefix(ckpt["g_ema"]))
        if generator_only:
            return
        saved = ckpt.get("config", {})
        TrainConfig.from_saved(saved)  # refuses an architecture this version cannot build
        check_resume(path, saved, self.cfg)
        self.netD.load_state_dict(strip_compile_prefix(ckpt["d"]))
        self.optG.load_state_dict(ckpt["opt_g"])
        self.optD.load_state_dict(ckpt["opt_d"])
        for opt in (self.optG, self.optD):
            for group in opt.param_groups:
                group["lr"] = self.cfg.lr
        self.warmup.capture()
        self.step_count = int(ckpt.get("step", 0))


class StepTimer:
    """Wall-clock time per step over a rolling window, for the log line."""

    def __init__(self, batch_size: int, total_steps: int) -> None:
        self.batch_size = batch_size
        self.remaining = total_steps
        self.times: deque[float] = deque(maxlen=50)
        self.ticks = 0
        self._last = 0.0

    def tick(self) -> None:
        """Call at the start of every step. The first step, which holds any compilation, is
        left out of the average."""
        now = time.perf_counter()
        self.ticks += 1
        if self.ticks > 2:
            self.times.append(now - self._last)
        self._last = now
        self.remaining -= 1

    def summary(self) -> dict:
        dt = sum(self.times) / len(self.times) if self.times else 0.0
        return {"ms_per_step": round(dt * 1e3, 1),
                "images_per_second": round(self.batch_size / dt, 2) if dt else 0.0,
                "eta_minutes": round(self.remaining * dt / 60, 1),
                "peak_vram_gb": round(dev.peak_memory_gb(), 2)}


@torch.no_grad()
def fixed_noise_spread(images: torch.Tensor, size: int = 16) -> float:
    """Mean pairwise cosine distance between the fixed-noise samples at 16 pixels: a cheap
    collapse alarm, since it falls toward zero as the samples converge on one image."""
    unit = F.normalize(F.interpolate(images, size=size, mode="area").flatten(1), dim=1, eps=1e-6)
    n = unit.shape[0]
    return float((1.0 - unit @ unit.T).sum() / (n * (n - 1))) if n > 1 else 0.0


def write_progress_sheet(out: Path, columns: int = 8, thumb: int = 220) -> None:
    """`validation/latest.jpg` is the newest sample grid; `progress.jpg` lines up a few."""
    grids = sorted((out / "samples").glob("*.jpg"))
    val = out / "validation"
    val.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(grids[-1], val / "latest.jpg")
    picks = grids if len(grids) <= columns else [
        grids[round(i * (len(grids) - 1) / (columns - 1))] for i in range(columns)]
    cells = []
    for path in picks:
        with open_drafted(path, thumb) as im:
            cell = im.convert("RGB")
        cell.thumbnail((thumb, thumb))
        cells.append(label_image(cell, f"step {int(path.stem)}"))
    contact_sheet(cells).save(val / "progress.jpg", quality=92)


def train(cfg: TrainConfig) -> dict:
    """Run a training job. Returns the final timing summary."""
    out = Path(cfg.out_dir)
    trainer = FastGANTrainer(cfg)
    fixed_z = trainer.sample_z(SAMPLE_GRID_ROWS * SAMPLE_GRID_COLUMNS)

    start = 0
    if cfg.resume_generator:
        trainer.load(cfg.resume_generator, generator_only=True)
        print(f"starting from the generator in {cfg.resume_generator}, with a fresh critic")
    elif cfg.resume:
        trainer.load(cfg.resume)
        start = trainer.step_count
        if start >= cfg.iterations:
            raise ValueError(f"{cfg.resume} is already at step {start}; --iterations is the "
                             f"total to reach, so it must be above that")
        print(f"resumed from {cfg.resume} at step {start}")

    loader, n_images = build_loader(cfg.data, cfg.batch_size, cfg.ladder.shape, cfg.seed + start,
                                    pin_memory=trainer.caps.backend in ("cuda", "xpu"))
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    write_json(out / "config.json", {"config": asdict(cfg), "images": n_images,
                                     "resumed_from": cfg.resume or cfg.resume_generator,
                                     "start_step": start,
                                     "precision": trainer.caps.training_precision})
    height, width = cfg.ladder.shape
    print(f"{n_images} images  {height}x{width}  batch {cfg.batch_size}  "
          f"steps {start + 1}..{cfg.iterations} "
          f"({(cfg.iterations - start) * cfg.batch_size / n_images:.1f} epochs)")
    print(trainer.caps.summary())
    if cfg.prioritise:
        dev.prioritise_gpu_feeder()

    timer = StepTimer(cfg.batch_size, cfg.iterations - start)
    dev.reset_peak_memory()
    since_log: dict[str, list[torch.Tensor]] = {}
    for it in range(start + 1, cfg.iterations + 1):
        timer.tick()
        for key, value in trainer.step(next(loader)).items():
            since_log.setdefault(key, []).append(value)

        if it % cfg.log_every == 0:
            keys = list(since_log)
            means = torch.stack([torch.stack(since_log[k]).float().mean() for k in keys]).tolist()
            scalars = {k: round(v, 4) for k, v in zip(keys, means, strict=True)}
            since_log.clear()
            timing = timer.summary()
            append_jsonl(log_path, {"step": it, **scalars, **timing})
            shown = "  ".join(f"{k.removeprefix('loss_')} {v:.3f}" for k, v in scalars.items())
            print(f"{it:>7}  {shown}  {timing['images_per_second']:.2f} img/s  "
                  f"{timing['ms_per_step']:.0f} ms/step  eta {timing['eta_minutes']:.0f} min")

        if it % cfg.sample_every == 0:
            images = trainer.sample_grid(fixed_z)
            save_image(images, out / "samples" / f"{it:07d}.jpg", nrow=SAMPLE_GRID_COLUMNS)
            write_progress_sheet(out)
            spread = fixed_noise_spread(images)
            append_jsonl(log_path, {"step": it, "fixed_noise_spread": round(spread, 5)})
            print(f"{it:>7}  fixed-noise spread {spread:.4f} (a steady fall toward zero "
                  f"means collapse; `gantrain diversity` measures it properly)")

        if it % cfg.save_every == 0 or it == cfg.iterations:
            trainer.save(out / "checkpoints" / f"{it:07d}.pt")

    summary = timer.summary()
    write_json(out / "throughput.json", summary)
    return summary
