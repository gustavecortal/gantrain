"""The `gantrain` commands."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import typer

from .gan.config import PERCEPTUAL_NETS, SUPPORTED_SIZES, DataConfig, TrainConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Train a FastGAN on your own photographs, on one GPU.")

#: The single source of defaults: every option below reads its default from here.
_T, _D = TrainConfig(), DataConfig()


@app.command()
def device() -> None:
    """Show the accelerator and the precision training will use on it."""
    from .device import get_caps

    print(get_caps().summary())


@app.command()
def curate(
    photos: list[Path] = typer.Argument(..., help="One or more folders of photographs"),
    out: Path = typer.Option(Path(_D.path), help="Where to write the manifests"),
    semantic_dedup: bool = typer.Option(
        True, help="Also drop near-identical scenes, judged by DINOv2 embeddings"),
    semantic_threshold: float = typer.Option(0.92, help="Cosine similarity that counts as the "
                                                        "same scene"),
    test_fraction: float = typer.Option(0.15, help="Share of images held out for `evaluate`"),
) -> None:
    """Scan photographs, drop duplicates, and split them into train and test manifests."""
    from .curate import curate as run

    run(photos, out, semantic=semantic_dedup, semantic_threshold=semantic_threshold,
        test_fraction=test_fraction)


@app.command()
def train(
    data: str = typer.Option(_D.path, help="A `curate` output folder, or any folder of images"),
    out: str = typer.Option(_T.out_dir, help="Run folder: samples, checkpoints, logs"),
    resolution: int = typer.Option(
        _T.im_size, help=f"Output height: {' | '.join(map(str, SUPPORTED_SIZES))}"),
    width: int | None = typer.Option(
        None, help="Output width, a multiple of resolution/4; square when omitted. Images "
                   "of the other orientation are left out."),
    view: str = typer.Option(
        _D.view, help="full: each photograph whole, cropped to the canvas's aspect. "
                      "random: random crops of it, more variety but less composition."),
    batch_size: int = typer.Option(_T.batch_size),
    iterations: int = typer.Option(_T.iterations, help="Total steps to reach"),
    lr: float = typer.Option(_T.lr, help="Adam learning rate for both networks"),
    lr_warmup: int = typer.Option(
        _T.lr_warmup, help="Ramp both learning rates up from zero over this many steps, "
                           "which steadies the start of a run (0 = off)"),
    ngf: int = typer.Option(_T.ngf, help="Generator width"),
    ndf: int = typer.Option(_T.ndf, help="Discriminator width"),
    nz: int = typer.Option(_T.nz, help="Latent dimension"),
    perceptual: str = typer.Option(
        _T.perceptual_net, help=f"LPIPS backbone for the discriminator's reconstruction "
                                f"loss: {' | '.join(PERCEPTUAL_NETS)}"),
    jacobian_weight: float = typer.Option(
        _T.jacobian_weight, help="Jacobian clamp (Odena et al., 2018): keeps how fast the "
                                 "image changes with z inside a band, for smoother "
                                 "interpolations (0 = off)"),
    amp: bool = typer.Option(_T.amp, "--amp/--no-amp",
                             help="bf16 mixed precision where the GPU supports it"),
    compile_model: bool = typer.Option(
        _T.compile_model, "--compile/--no-compile",
        help="Compile G and D with torch.compile: faster steps after a one-off build; needs "
             "a C compiler"),
    workers: int = typer.Option(_D.workers, help="Data-loading processes"),
    resume: Path | None = typer.Option(
        None, help="Continue a run from its checkpoint: weights, optimisers and step count"),
    resume_generator: Path | None = typer.Option(
        None, help="Start a new run from a checkpoint's generator, with a fresh critic"),
    accept_config_change: bool = typer.Option(
        False, "--accept-config-change",
        help="Let --resume continue with settings that differ from the checkpoint's. The "
             "data path, output folder, iteration count and logging may always differ."),
    prioritise: bool = typer.Option(
        _T.prioritise, "--prioritise/--no-prioritise",
        help="Raise process priority and keep it on performance cores"),
    log_every: int = typer.Option(_T.log_every),
    sample_every: int = typer.Option(_T.sample_every),
    save_every: int = typer.Option(_T.save_every),
    seed: int = typer.Option(_T.seed),
) -> None:
    """Train FastGAN from scratch."""
    from .gan.train import train as run

    cfg = TrainConfig(
        data=DataConfig(path=data, workers=workers, view=view), out_dir=out,
        im_size=resolution, im_width=width, batch_size=batch_size, iterations=iterations,
        lr=lr, lr_warmup=lr_warmup, ngf=ngf, ndf=ndf, nz=nz, perceptual_net=perceptual,
        jacobian_weight=jacobian_weight, amp=amp, compile_model=compile_model,
        resume=str(resume) if resume else None,
        resume_generator=str(resume_generator) if resume_generator else None,
        accept_config_change=accept_config_change, prioritise=prioritise,
        log_every=log_every, sample_every=sample_every, save_every=save_every, seed=seed)
    summary = run(cfg)
    print(f"done: {summary['images_per_second']:.2f} img/s, "
          f"{summary['peak_vram_gb']:.2f} GB peak")


@app.command()
def sample(
    checkpoint: Path = typer.Argument(..., help="A .pt from a run's checkpoints folder"),
    out: Path = typer.Option(Path("outputs/samples")),
    n: int = typer.Option(64, help="Images to write"),
    batch_size: int = typer.Option(16),
    seed: int = typer.Option(0, help="Picks the latents; the same on every device"),
) -> None:
    """Write individual images from a checkpoint."""
    from .gan.sample import generate, training_images_of

    paths = generate(checkpoint, out, n=n, batch_size=batch_size, seed=seed)
    print(f"wrote {len(paths)} images to {out}")
    if (real := training_images_of(checkpoint)) is not None:
        print(f"score them with:  gantrain evaluate {real} {out}")


@app.command()
def video(
    checkpoint: Path = typer.Argument(..., help="A .pt from a run's checkpoints folder"),
    out: Path = typer.Option(Path("outputs/interpolation.mp4")),
    seconds: float = typer.Option(30.0),
    fps: int = typer.Option(30),
    batch_size: int = typer.Option(8, help="Frames generated per forward pass"),
    seed: int = typer.Option(0, help="Picks the two latents the clip travels between"),
    loop: bool = typer.Option(True, "--loop/--no-loop",
                              help="Go there and back, so the clip repeats seamlessly"),
    noise_seed: int = typer.Option(0, help="Picks the generator's fixed noise pattern"),
    fourcc: str = typer.Option("mp4v", help="OpenCV codec tag"),
) -> None:
    """Render a latent-interpolation video."""
    from .gan.sample import interpolate_video

    path = interpolate_video(checkpoint, out, seconds=seconds, fps=fps, batch_size=batch_size,
                             seed=seed, loop=loop, noise_seed=noise_seed, fourcc=fourcc)
    print(f"wrote {path}")


@app.command()
def diversity(
    run: Path = typer.Argument(..., help="A run folder with samples/*.jpg"),
    out: Path | None = typer.Option(None, help="Also write the trajectory as JSON here"),
) -> None:
    """Track how varied the fixed-noise samples are across a run (Vendi score)."""
    from .evaluate import fixed_noise_diversity
    from .manifest import write_json

    rows = fixed_noise_diversity(run)
    if not rows:
        raise typer.BadParameter(f"no sample grids under {run / 'samples'}")
    print(f"Vendi score of {rows[0]['cells']} fixed-noise samples (at most {rows[0]['cells']})")
    for r in rows:
        print(f"  step {r['step']:>7}   {r['vendi']:.3f}")
    v = [r["vendi"] for r in rows]
    lo, hi = statistics.median(v[:len(v) // 2] or v), statistics.median(v[len(v) // 2:])
    change = (hi - lo) / lo if lo else 0.0
    trend = "rising" if change > 0.02 else ("FALLING" if change < -0.02 else "flat")
    print(f"first half {lo:.3f} -> second half {hi:.3f} ({trend} {change * 100:+.1f}%)")
    if out:
        write_json(out, {"run": str(run), "trajectory": rows})


@app.command()
def evaluate(
    real: Path = typer.Argument(..., help="Real images: a folder or a manifest"),
    fake: Path = typer.Argument(..., help="Generated images: a folder"),
    out: Path | None = typer.Option(None, help="Also write the report as JSON here"),
    fd: bool = typer.Option(True, help="Include FD-DINOv2, which needs many samples"),
) -> None:
    """Score generated images against real ones: quality, coverage, variety, copies."""
    from .evaluate import evaluate as run
    from .evaluate import render
    from .manifest import images_in, write_json

    real_paths = [r.path for r in images_in(real)]
    fake_paths = [r.path for r in images_in(fake)]
    if not real_paths or not fake_paths:
        raise typer.BadParameter(f"found {len(real_paths)} real and {len(fake_paths)} "
                                 f"generated images")
    report = run(real_paths, fake_paths, compute_fd=fd)
    print(render(report))
    if out:
        write_json(out, report)


if __name__ == "__main__":
    app()
