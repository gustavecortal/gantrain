"""What defines a training run, checked once, and what a resume may change."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

SUPPORTED_SIZES = (256, 512, 1024, 2048)
#: Height of the bottom rung, the generator's first feature map.
BASE = 4
PERCEPTUAL_NETS = ("vgg", "alex", "squeeze")
VIEWS = ("full", "random")

#: Settings that describe the invocation rather than the training, so a resume may change them.
RESUME_MAY_DIFFER = frozenset({
    "out_dir", "iterations", "seed", "resume", "resume_generator", "accept_config_change",
    "compile_model", "prioritise", "log_every", "sample_every", "save_every",
    "data.path", "data.workers",
})

#: Architecture options older checkpoints may carry, at the value this version builds.
_ONLY_BUILDS = {"pixelshuffle_from": 0, "mapping_depth": 0, "patch_crop": 0,
                "mbstd_group": 0, "full_res_head": 0}


@dataclass(frozen=True)
class Ladder:
    """The (height, width) both networks climb, square or not."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height not in SUPPORTED_SIZES:
            raise ValueError(f"height must be one of {SUPPORTED_SIZES}, got {self.height}")
        step = self.height // BASE
        if self.width <= 0 or self.width % step:
            raise ValueError(
                f"width must be a positive multiple of {step} for a {self.height}-tall "
                f"canvas, so that every rung down to the {BASE}-high base is whole pixels. "
                f"Got {self.width}; the nearest legal values are {self.width // step * step} "
                f"and {(self.width // step + 1) * step}.")

    @classmethod
    def of(cls, height: int, width: int | None = None) -> Ladder:
        """`width=None` means square."""
        return cls(height, height if width is None else width)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def at(self, rung: int) -> tuple[int, int]:
        """The (height, width) of the rung `rung` pixels tall."""
        return rung, rung * self.width // self.height


@dataclass(frozen=True)
class DataConfig:
    """Where the images are and how each becomes a training sample."""

    path: str = "data/curated"
    split: str = "train"
    #: "full" trains on whole frames; "random" on random crops of them.
    view: str = "full"
    crop_scale: tuple[float, float] = (0.5, 1.0)
    hflip: bool = True
    workers: int = 4

    def __post_init__(self) -> None:
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {self.view!r}")


@dataclass
class TrainConfig:
    """Everything that defines a run. Saved in every checkpoint."""

    data: DataConfig = field(default_factory=DataConfig)
    out_dir: str = "runs/fastgan"
    im_size: int = 512
    im_width: int | None = None
    batch_size: int = 16
    iterations: int = 50_000
    ngf: int = 64
    ndf: int = 64
    nz: int = 256
    lr: float = 2e-4
    #: Steps over which both learning rates ramp up linearly from zero. 0 disables it.
    lr_warmup: int = 0
    beta1: float = 0.5
    beta2: float = 0.999
    ema_decay: float = 0.999
    #: Clip both networks' gradients to this total norm. 0 disables it.
    max_grad_norm: float = 1.0
    #: Weight of the Jacobian clamp, which keeps how fast the image changes with `z` in a band.
    jacobian_weight: float = 0.0
    perceptual_net: str = "alex"
    amp: bool = True
    compile_model: bool = False
    resume: str | None = None
    resume_generator: str | None = None
    #: Let a full resume continue with settings the checkpoint was not trained with.
    accept_config_change: bool = False
    prioritise: bool = True
    seed: int = 0
    log_every: int = 50
    sample_every: int = 250
    save_every: int = 2000

    def __post_init__(self) -> None:
        if self.perceptual_net not in PERCEPTUAL_NETS:
            raise ValueError(f"perceptual_net must be one of {PERCEPTUAL_NETS}, "
                             f"got {self.perceptual_net!r}")
        if self.lr_warmup < 0:
            raise ValueError(f"lr_warmup must be >= 0, got {self.lr_warmup}")
        if self.jacobian_weight < 0:
            raise ValueError(f"jacobian_weight must be >= 0, got {self.jacobian_weight}")
        if self.jacobian_weight and self.batch_size < 2:
            raise ValueError("the Jacobian clamp needs a batch of at least 2")
        if self.resume and self.resume_generator:
            raise ValueError("pass --resume (continue a run) or --resume-generator (start a "
                             "new run from an old generator), not both")
        self.ladder  # noqa: B018 - validates the canvas before any device work

    @property
    def ladder(self) -> Ladder:
        return Ladder.of(self.im_size, self.im_width)

    @property
    def generator_kwargs(self) -> dict:
        return {"ngf": self.ngf, "nz": self.nz, "im_size": self.im_size,
                "im_width": self.im_width}

    @property
    def discriminator_kwargs(self) -> dict:
        return {"ndf": self.ndf, "im_size": self.im_size, "im_width": self.im_width}

    @classmethod
    def from_saved(cls, saved: dict) -> TrainConfig:
        """A checkpoint's config, ignoring keys this version no longer has."""
        for key, neutral in _ONLY_BUILDS.items():
            if saved.get(key, neutral) != neutral:
                raise ValueError(f"this checkpoint was trained with {key}={saved[key]}, "
                                 f"which gantrain no longer builds")
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in saved.items() if k in known and k != "data"}
        data_known = {f.name for f in fields(DataConfig)}
        data = {k: v for k, v in (saved.get("data") or {}).items() if k in data_known}
        if "crop_scale" in data:
            data["crop_scale"] = tuple(data["crop_scale"])
        return cls(data=DataConfig(**data), **kwargs)


def drift(said: dict, now: dict, at: str = "") -> list[str]:
    """Settings in `now` that the checkpoint's recorded `said` does not share."""
    out = []
    for key, mine in now.items():
        path = f"{at}{key}"
        if path in RESUME_MAY_DIFFER or key not in said:
            continue
        was = said[key]
        if isinstance(mine, dict) and isinstance(was, dict):
            out += drift(was, mine, f"{path}.")
        elif (list(was) if isinstance(was, (list, tuple)) else was) != (
                list(mine) if isinstance(mine, (list, tuple)) else mine):
            out.append(f"{path}: checkpoint has {was!r}, this run {mine!r}")
    return out


def check_resume(path, said: dict, cfg: TrainConfig) -> None:
    """Stop a resume whose settings differ from the checkpoint's, naming each difference.

    A resume restores weights and optimiser state, never settings, so a flag left off the
    command line falls back to its default."""
    differences = drift(said, asdict(cfg))
    if not differences:
        return
    if cfg.accept_config_change:
        print("resume: continuing with settings the checkpoint was not trained with:\n  "
              + "\n  ".join(differences), flush=True)
        return
    raise SystemExit(
        f"resuming {path} would change these settings:\n  " + "\n  ".join(differences)
        + "\nRestate them on the command line, or pass --accept-config-change.")
