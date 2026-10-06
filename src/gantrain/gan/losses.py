"""FastGAN's losses: a hinge with a randomised margin, plus perceptual reconstruction."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

#: The Jacobian clamp keeps `||dG|| / ||dz||` (RMS pixel change per unit of latent) in this
#: band, measures it over a step of this length in `z`, and runs on every Nth step with its
#: weight scaled by N so the cadence does not change its strength.
JACOBIAN_BAND = (0.0124, 0.20)
JACOBIAN_EPSILON = 0.1
JACOBIAN_INTERVAL = 4


def _margin(logits: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(logits) * 0.2 + 0.8


def hinge_d_real(heads: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.relu(_margin(h) - h).mean() for h in heads) / len(heads)


def hinge_d_fake(heads: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.relu(_margin(h) + h).mean() for h in heads) / len(heads)


def g_adversarial(heads: list[torch.Tensor]) -> torch.Tensor:
    return sum(-h.mean() for h in heads) / len(heads)


def crop_quadrant(image: torch.Tensor, part: int) -> torch.Tensor:
    """Quadrant `part` (0-3, row-major) of an image or feature map."""
    half_h, half_w = image.shape[-2] // 2, image.shape[-1] // 2
    top, left = divmod(part, 2)
    return image[:, :, top * half_h : (top + 1) * half_h,
                 left * half_w : (left + 1) * half_w]


def jacobian_ratio(fake: torch.Tensor, fake_perturbed: torch.Tensor,
                   z: torch.Tensor, z_perturbed: torch.Tensor) -> torch.Tensor:
    """Per sample, the RMS pixel change over the latent step that caused it."""
    dz = torch.linalg.vector_norm(z - z_perturbed, dim=1)
    dg = (torch.linalg.vector_norm(fake - fake_perturbed, dim=tuple(range(1, fake.ndim)))
          / fake[0].numel() ** 0.5)
    return dg / dz.clamp_min(1e-8)


def jacobian_clamp(q: torch.Tensor, lo: float = JACOBIAN_BAND[0],
                   hi: float = JACOBIAN_BAND[1]) -> torch.Tensor:
    """Odena et al. 2018: zero inside `[lo, hi]`, quadratic outside it."""
    return (F.relu(q / hi - 1.0).square() + F.relu(lo / q.clamp_min(1e-8) - 1.0).square()).mean()


class PerceptualReconstruction(nn.Module):
    """LPIPS distance between the discriminator's reconstructions and the real image."""

    def __init__(self, net: str, device: torch.device | str) -> None:
        super().__init__()
        import lpips

        self.metric = lpips.LPIPS(net=net, verbose=False).eval().requires_grad_(False)
        self.metric.to(device)

    def forward(self, recons: list[torch.Tensor], real: torch.Tensor, part: int) -> torch.Tensor:
        """The three reconstruction terms (whole, small, one quadrant) in one LPIPS pass."""
        size = recons[0].shape[-2:]
        full = F.interpolate(real, size=size, mode="bilinear", align_corners=False)
        quad = F.interpolate(crop_quadrant(real, part), size=size, mode="bilinear",
                             align_corners=False)
        return self.metric(torch.cat(recons), torch.cat([full, full, quad])).sum()


@functools.lru_cache(maxsize=4)
def get_perceptual(net: str, device: str) -> PerceptualReconstruction:
    return PerceptualReconstruction(net, device)
