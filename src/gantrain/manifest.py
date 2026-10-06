"""The manifest: one JSON line per image, written by `curate` and read by every command."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .imageio import IMAGE_EXT, ROTATED_ORIENTATIONS, iter_images

SCHEMA_VERSION = 1


@dataclass
class Record:
    """One image. Fields this class does not name are read into `extra`, not refused."""

    id: str
    path: str | None = None          # absolute, or relative to the manifest's own folder
    sha256: str | None = None
    width: int | None = None         # as stored, before EXIF rotation
    height: int | None = None
    split: str | None = None         # train | test
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def oriented_size(self) -> tuple[int, int] | None:
        """`(width, height)` as the image loads, after EXIF rotation."""
        if not self.width or not self.height:
            return None
        if (self.extra or {}).get("orientation") in ROTATED_ORIENTATIONS:
            return self.height, self.width
        return self.width, self.height

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        known = set(cls.__dataclass_fields__)
        rec = cls(**{k: v for k, v in d.items() if k in known})
        rec.extra.update({k: v for k, v in d.items() if k not in known})
        return rec


def write_manifest(path: str | Path, records: Iterable[Record]) -> int:
    """Write records as JSONL. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_schema": SCHEMA_VERSION}) + "\n")
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_manifest(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if not ("_schema" in d and len(d) == 1):
                    yield Record.from_dict(d)


def read_manifest(path: str | Path) -> list[Record]:
    return list(iter_manifest(path))


def resolve_path(manifest: str | Path, rec: Record) -> Path:
    """A record's image file. Relative paths are relative to the manifest's own folder."""
    p = Path(rec.path or "")
    return p if p.is_absolute() else Path(manifest).parent / p


def images_in(path: str | Path, split: str | None = None) -> list[Record]:
    """The images a manifest, a curated folder or a plain folder holds, paths resolved.

    A folder with `manifest_<split>.jsonl` or `manifest.jsonl` is read through it; any other
    folder is listed. Only existing image files are returned."""
    path = Path(path)
    if path.is_dir():
        per_split = path / f"manifest_{split}.jsonl"
        if split and per_split.exists():
            return images_in(per_split)
        if not (path / "manifest.jsonl").exists():
            return [Record(id=str(p), path=str(p)) for p in iter_images(path)]
        path = path / "manifest.jsonl"
    out = []
    for rec in iter_manifest(path):
        if split and rec.split and rec.split != split:
            continue
        p = resolve_path(path, rec)
        if p.suffix.lower() in IMAGE_EXT and p.exists():
            rec.path = str(p)
            out.append(rec)
    return out


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
