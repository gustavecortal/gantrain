"""Opening photographs cheaply and the right way up, and the sample-grid layout."""

from __future__ import annotations

import functools
import io
from pathlib import Path

from PIL import ExifTags, Image, ImageCms, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000

#: EXIF orientations that turn the image a quarter turn, swapping its width and height.
ROTATED_ORIENTATIONS = (5, 6, 7, 8)
ORIENTATION_TAG = ExifTags.Base.Orientation

#: Training saves its fixed-noise samples as a grid of this shape; `diversity` reads it back.
SAMPLE_GRID_COLUMNS = 4
SAMPLE_GRID_ROWS = 2
SAMPLE_GRID_PAD = 2


def _register_heif() -> bool:
    """Teach PIL to open HEIC/HEIF, and report whether it worked."""
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        return True
    except Exception:  # noqa: BLE001 - an optional format, present or broken
        return False


HEIF_AVAILABLE = _register_heif()
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
if HEIF_AVAILABLE:
    IMAGE_EXT |= {".heic", ".heif"}


def iter_images(root: str | Path) -> list[Path]:
    """Every image file under `root`, sorted."""
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXT)


def open_drafted(path: str | Path, min_edge: int = 1024, short_edge: bool = False) -> Image.Image:
    """Open an image, letting libjpeg decode at a reduced scale no smaller than `min_edge`."""
    im = Image.open(path)
    w, h = im.size
    try:
        scale = min_edge / (min(w, h) if short_edge else max(w, h))
        im.draft("RGB", (max(1, round(w * scale)), max(1, round(h * scale))))
    except (AttributeError, ValueError, OSError, ZeroDivisionError):
        pass
    return im


_SRGB = ImageCms.createProfile("sRGB")


@functools.lru_cache(maxsize=8)
def _to_srgb(icc: bytes, mode: str):
    """A colour transform from an embedded profile to sRGB, built once per profile."""
    return ImageCms.buildTransform(ImageCms.ImageCmsProfile(io.BytesIO(icc)), _SRGB, mode, "RGB")


def load_oriented(path: str | Path, min_edge: int | None = None,
                  short_edge: bool = False) -> Image.Image:
    """Open an image with its EXIF rotation applied and its colours in sRGB."""
    im = open_drafted(path, min_edge, short_edge) if min_edge else Image.open(path)
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    if icc:
        try:
            return ImageCms.applyTransform(im, _to_srgb(icc, im.mode))
        except Exception:  # noqa: BLE001 - a broken profile is not fatal
            pass
    return im.convert("RGB")


def is_landscape(path: str | Path) -> bool | None:
    """Orientation from the header alone, EXIF rotation included. `None` if unreadable."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(ORIENTATION_TAG) in ROTATED_ORIENTATIONS:
                w, h = h, w
        return w >= h
    except Exception:  # noqa: BLE001 - an unreadable header leaves the image unfiltered
        return None


def crop_box_for_aspect(width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
    """The largest centred box of `aspect` (width / height) that fits."""
    if width / height > aspect:
        cw, ch = int(round(height * aspect)), height
    else:
        cw, ch = width, int(round(width / aspect))
    cw, ch = min(cw, width), min(ch, height)
    left = int(round(min(max(width / 2 - cw / 2, 0), width - cw)))
    top = int(round(min(max(height / 2 - ch / 2, 0), height - ch)))
    return left, top, left + cw, top + ch


class CropToCanvas:
    """Crop to the canvas's aspect ratio and resize to it, in one resample."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.height, self.width = shape

    def __call__(self, image: Image.Image) -> Image.Image:
        box = crop_box_for_aspect(image.width, image.height, self.width / self.height)
        return image.resize((self.width, self.height), Image.BICUBIC, box=box)


def grid_cells(path: str | Path) -> list[Image.Image]:
    """Split a saved sample grid back into its cells."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    pad, cols, rows = SAMPLE_GRID_PAD, SAMPLE_GRID_COLUMNS, SAMPLE_GRID_ROWS
    ch = (rgb.height - pad * (rows + 1)) // rows
    cw = (rgb.width - pad * (cols + 1)) // cols
    return [rgb.crop((pad + c * (cw + pad), pad + r * (ch + pad),
                      pad + c * (cw + pad) + cw, pad + r * (ch + pad) + ch))
            for r in range(rows) for c in range(cols)]


BACKGROUND = (16, 16, 16)


def label_image(im: Image.Image, text: str, height: int = 15) -> Image.Image:
    """A copy of `im` with a line of text above it."""
    out = Image.new("RGB", (im.width, im.height + height), BACKGROUND)
    out.paste(im, (0, height))
    ImageDraw.Draw(out).text((3, 2), text, fill=(235, 235, 235))
    return out


def contact_sheet(cells: list[Image.Image], pad: int = 4) -> Image.Image:
    """Tile `cells` in one row."""
    cw, ch = max(c.width for c in cells), max(c.height for c in cells)
    sheet = Image.new("RGB", (len(cells) * (cw + pad) + pad, ch + 2 * pad), BACKGROUND)
    for i, c in enumerate(cells):
        sheet.paste(c, (pad + i * (cw + pad), pad))
    return sheet
