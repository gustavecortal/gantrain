"""FastGAN's generator and discriminator (Liu et al., ICLR 2021), in plain PyTorch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from .config import BASE, Ladder
from .losses import crop_quadrant

#: Channel width per stage height, as a multiple of `ngf`.
G_WIDTH = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125,
           2048: 0.125}
#: The discriminator's table differs from the generator's in three entries.
D_WIDTH = {**G_WIDTH, 8: 16, 16: 8, 32: 4}


def _widths(table: dict[int, float], base: int) -> dict[int, int]:
    return {k: int(v * base) for k, v in table.items()}


def conv(*args, **kwargs) -> nn.Module:
    """Spectrally normalised convolution."""
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def conv_transpose(*args, **kwargs) -> nn.Module:
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


class NoiseInjection(nn.Module):
    """Per-pixel Gaussian noise with a learned scalar gain, StyleGAN-style."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.freeze = False
        self.frozen: torch.Tensor | None = None
        self.generator: torch.Generator | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if self.freeze:
            if self.frozen is None or self.frozen.shape[-2:] != (h, w):
                if self.generator is None:
                    self.frozen = torch.randn(1, 1, h, w, device=x.device, dtype=x.dtype)
                else:
                    self.frozen = torch.randn(
                        1, 1, h, w, generator=self.generator).to(x.device, x.dtype)
            return x + self.weight * self.frozen.to(dtype=x.dtype)
        noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


def freeze_noise(net: nn.Module, freeze: bool = True, *, seed: int | None = None) -> nn.Module:
    """Pin every noise layer to one pattern, so the same `z` gives the same pixels.

    With a `seed` the pattern depends on nothing else; without one it is drawn from the
    device's random stream at the next forward."""
    generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    for m in net.modules():
        if isinstance(m, NoiseInjection):
            m.freeze = freeze
            m.generator = generator
            m.frozen = None
    return net


class SkipLayerExcitation(nn.Module):
    """Channel-wise gate: a low-resolution feature map modulates a high-resolution one."""

    def __init__(self, ch_low: int, ch_high: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv(ch_low, ch_high, 4, 1, 0, bias=False),
            nn.SiLU(),
            conv(ch_high, ch_high, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high * self.gate(low)


class InitLayer(nn.Module):
    """`z` -> the base feature map, via a transposed convolution from a 1x1 "image"."""

    def __init__(self, nz: int, ch: int, shape: tuple[int, int]) -> None:
        super().__init__()
        self.main = nn.Sequential(
            conv_transpose(nz, ch * 2, shape, 1, 0, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.GLU(dim=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z.view(z.shape[0], -1, 1, 1))


def up_block(ch_in: int, ch_out: int) -> nn.Sequential:
    """2x upsample: nearest resample, conv, batch norm, GLU."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv(ch_in, ch_out * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(ch_out * 2),
        nn.GLU(dim=1),
    )


def up_block_comp(ch_in: int, ch_out: int) -> nn.Sequential:
    """The heavier 2x upsample: two convolutions, each followed by noise injection."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv(ch_in, ch_out * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(ch_out * 2),
        nn.GLU(dim=1),
        conv(ch_out, ch_out * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        nn.BatchNorm2d(ch_out * 2),
        nn.GLU(dim=1),
    )


class Generator(nn.Module):
    """`z` -> [the full-resolution image, a 128-high version of it]."""

    def __init__(self, ngf: int = 64, nz: int = 256, nc: int = 3, im_size: int = 512,
                 im_width: int | None = None) -> None:
        super().__init__()
        self.ladder = Ladder.of(im_size, im_width)
        w = _widths(G_WIDTH, ngf)

        self.init = InitLayer(nz, w[4], self.ladder.at(BASE))
        self.feat_8 = up_block_comp(w[4], w[8])
        self.feat_16 = up_block(w[8], w[16])
        self.feat_32 = up_block_comp(w[16], w[32])
        self.feat_64 = up_block(w[32], w[64])
        self.feat_128 = up_block_comp(w[64], w[128])
        self.feat_256 = up_block(w[128], w[256])

        self.se_64 = SkipLayerExcitation(w[4], w[64])
        self.se_128 = SkipLayerExcitation(w[8], w[128])
        self.se_256 = SkipLayerExcitation(w[16], w[256])

        self.to_small = conv(w[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv(w[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = up_block_comp(w[256], w[512])
            self.se_512 = SkipLayerExcitation(w[32], w[512])
        if im_size > 512:
            self.feat_1024 = up_block(w[512], w[1024])
        if im_size > 1024:
            self.feat_2048 = up_block_comp(w[1024], w[2048])

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        f4 = self.init(z)
        f8 = self.feat_8(f4)
        f16 = self.feat_16(f8)
        f32 = self.feat_32(f16)
        f64 = self.se_64(f4, self.feat_64(f32))
        f128 = self.se_128(f8, self.feat_128(f64))
        feat = self.se_256(f16, self.feat_256(f128))

        if self.ladder.height > 256:
            feat = self.se_512(f32, self.feat_512(feat))
        if self.ladder.height > 512:
            feat = self.feat_1024(feat)
        if self.ladder.height > 1024:
            feat = self.feat_2048(feat)

        return [torch.tanh(self.to_big(feat)), torch.tanh(self.to_small(f128))]


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """Generator output in [-1, 1] -> [0, 1], ready to save as an image."""
    return x.float().add(1).mul(0.5).clamp(0, 1)


class DownBlock(nn.Sequential):
    """Halve the resolution: strided conv, batch norm, leaky ReLU."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__(
            conv(ch_in, ch_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        )


class DownBlockComp(nn.Module):
    """Two-path downsample, averaged: a strided conv pair and a pooled 1x1."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            conv(ch_in, ch_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ch_out, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv(ch_in, ch_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(self.main(x), self.direct(x)).mul_(0.5)


class SimpleDecoder(nn.Module):
    """Feature map -> a 128-high image, for the discriminator's self-supervision."""

    def __init__(self, ch_in: int, nc: int, pool: tuple[int, int]) -> None:
        super().__init__()
        w = _widths(G_WIDTH, 32)
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool),
            up_block(ch_in, w[16]),
            up_block(w[16], w[32]),
            up_block(w[32], w[64]),
            up_block(w[64], w[128]),
            conv(w[128], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def _stem(nc: int, w: dict[int, int], im_size: int) -> nn.Sequential:
    """Bring the full-resolution image down to the 256-high trunk, whatever its height."""
    if im_size == 256:
        return nn.Sequential(conv(nc, w[512], 3, 1, 1, bias=False),
                             nn.LeakyReLU(0.2, inplace=True))
    sizes = [s for s in (2048, 1024, 512) if s <= im_size]
    layers: list[nn.Module] = [conv(nc, w[sizes[0]], 4, 2, 1, bias=False),
                               nn.LeakyReLU(0.2, inplace=True)]
    for hi, lo in zip(sizes, sizes[1:], strict=False):
        layers += [conv(w[hi], w[lo], 4, 2, 1, bias=False), nn.BatchNorm2d(w[lo]),
                   nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """Two-scale critic that also reconstructs its input from its own features."""

    def __init__(self, ndf: int = 64, nc: int = 3, im_size: int = 512,
                 im_width: int | None = None) -> None:
        super().__init__()
        self.ladder = Ladder.of(im_size, im_width)
        w = _widths(D_WIDTH, ndf)

        self.from_big = _stem(nc, w, im_size)
        self.down_4 = DownBlockComp(w[512], w[256])
        self.down_8 = DownBlockComp(w[256], w[128])
        self.down_16 = DownBlockComp(w[128], w[64])
        self.down_32 = DownBlockComp(w[64], w[32])
        self.down_64 = DownBlockComp(w[32], w[16])

        self.se_16 = SkipLayerExcitation(w[512], w[64])
        self.se_32 = SkipLayerExcitation(w[256], w[32])
        self.se_64 = SkipLayerExcitation(w[128], w[16])

        self.rf_big = nn.Sequential(
            conv(w[16], w[8], 1, 1, 0, bias=False),
            nn.BatchNorm2d(w[8]),
            nn.LeakyReLU(0.2, inplace=True),
            conv(w[8], 1, 4, 1, 0, bias=False),
        )

        self.from_small = nn.Sequential(
            conv(nc, w[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(w[256], w[128]),
            DownBlock(w[128], w[64]),
            DownBlock(w[64], w[32]),
        )
        self.rf_small = conv(w[32], 1, 4, 1, 0, bias=False)

        pool = self.ladder.at(8)
        self.decoder_big = SimpleDecoder(w[16], nc, pool)
        self.decoder_small = SimpleDecoder(w[32], nc, pool)
        self.decoder_part = SimpleDecoder(w[32], nc, pool)

    def forward(self, images: list[torch.Tensor] | torch.Tensor, *,
                reconstruct: bool = False, part: torch.Tensor | None = None):
        """Logits per head; with `reconstruct`, also three reconstructions of the input.

        `images` is the generator's two-scale output, or one real image the small scale is
        resized from. `part` picks the quadrant the third reconstruction is of; it is a
        tensor so that a compiled discriminator traces one graph for all four quadrants."""
        if isinstance(images, list):
            big, small = images
        else:
            big = images
            small = F.interpolate(images, size=self.ladder.at(128), mode="bilinear",
                                  align_corners=False)
        f2 = self.from_big(big)
        f4 = self.down_4(f2)
        f8 = self.down_8(f4)
        f16 = self.se_16(f2, self.down_16(f8))
        f32 = self.se_32(f4, self.down_32(f16))
        f_last = self.se_64(f8, self.down_64(f32))

        f_small = self.from_small(small)
        heads = [self.rf_big(f_last).view(-1), self.rf_small(f_small).view(-1)]
        if not reconstruct:
            return heads

        quadrants = torch.stack([crop_quadrant(f32, i) for i in range(4)])
        recons = [self.decoder_big(f_last), self.decoder_small(f_small),
                  self.decoder_part(quadrants[part])]
        return heads, recons


def init_gan_weights(m: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) on conv weights, N(1, 0.02) on batch-norm scales."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(getattr(m, "weight_orig", m.weight), 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
