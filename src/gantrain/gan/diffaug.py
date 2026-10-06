"""Differentiable augmentation for the discriminator (Zhao et al., NeurIPS 2020).

Colour then translation, applied identically to real and generated images."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_sample(x: torch.Tensor) -> torch.Tensor:
    """One U(0, 1) value per image, broadcast over C, H and W."""
    return torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (_per_sample(x) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    return torch.lerp(x.mean(dim=1, keepdim=True), x, _per_sample(x) * 2)


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    return torch.lerp(x.mean(dim=[1, 2, 3], keepdim=True), x, _per_sample(x) + 0.5)


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Shift each image by a whole number of pixels, up to `ratio` of its size, zero-padded."""
    b, _, h, w = x.shape
    shift_h, shift_w = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ty = torch.randint(-shift_h, shift_h + 1, size=[b, 1, 1], device=x.device)
    tx = torch.randint(-shift_w, shift_w + 1, size=[b, 1, 1], device=x.device)
    idx_b = torch.arange(b, device=x.device).view(b, 1, 1)
    idx_y = torch.clamp(torch.arange(h, device=x.device).view(1, h, 1) + ty + 1, 0, h + 1)
    idx_x = torch.clamp(torch.arange(w, device=x.device).view(1, 1, w) + tx + 1, 0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1])
    return padded.permute(0, 2, 3, 1)[idx_b, idx_y, idx_x].permute(0, 3, 1, 2)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Augment a batch, keeping its memory format."""
    fmt = (torch.channels_last if x.is_contiguous(memory_format=torch.channels_last)
           else torch.contiguous_format)
    for fn in (rand_brightness, rand_saturation, rand_contrast, rand_translation):
        x = fn(x)
    return x.contiguous(memory_format=fmt)
