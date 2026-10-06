"""From folders of photographs to a deduplicated, split training set."""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

from .imageio import ORIENTATION_TAG, iter_images, open_drafted
from .manifest import Record, sha256_file, write_json, write_manifest

#: Below this many items a process pool costs more than it saves.
_POOL_MIN_ITEMS = 64


def default_workers() -> int:
    """Leave two cores for the OS and whatever else is running."""
    return max(1, (os.cpu_count() or 2) - 2)


def _call(fn, x):
    try:
        return fn(x)
    except Exception as e:  # noqa: BLE001 - handed back for the caller to skip
        return e


class _Safe:
    """A picklable `fn` that returns its exception instead of raising it."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return _call(self.fn, x)


def pmap(fn: Callable, items: Sequence, desc: str, workers: int | None = None,
         progress: bool = True) -> list:
    """`fn` over `items` across processes, in order; a failure comes back as its exception."""
    workers = default_workers() if workers is None else max(1, workers)
    if workers == 1 or len(items) < _POOL_MIN_ITEMS:
        results = map(_Safe(fn), items)
        return list(tqdm(results, desc=desc, total=len(items), disable=not progress))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_Safe(fn), items, chunksize=8)
        return list(tqdm(results, desc=desc, total=len(items), disable=not progress))


def read_photo(path: Path) -> Record:
    """A record for one photograph: its size, rotation, camera and hash, from the header."""
    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
    return Record(id=hashlib.sha256(str(path).encode()).hexdigest()[:16], path=str(path),
                  sha256=sha256_file(path), width=width, height=height,
                  extra={"orientation": exif.get(ORIENTATION_TAG),
                         "camera_model": exif.get(ExifTags.Base.Model)})


def scan(dirs: Sequence[str | Path], workers: int | None = None,
         progress: bool = True) -> list[Record]:
    """Every readable photograph under `dirs`, with absolute paths."""
    paths = [p.resolve() for d in dirs for p in iter_images(d)]
    return [r for r in pmap(read_photo, paths, "scan", workers, progress)
            if isinstance(r, Record)]


def _linked_pairs(rows: Callable[[int, int], np.ndarray], n: int,
                  chunk: int = 1024) -> list[tuple[int, int]]:
    """Index pairs `(i, j)`, `i < j`, that `rows(i0, i1)` marks, a block of rows at a time so
    no `n x n` matrix is ever held."""
    pairs: list[tuple[int, int]] = []
    for i0 in range(0, n, chunk):
        a, b = np.nonzero(rows(i0, min(n, i0 + chunk)))
        a += i0
        keep = b > a
        pairs += zip(a[keep].tolist(), b[keep].tolist(), strict=True)
    return pairs


def _resolve_clusters(records: list[Record],
                      pairs: list[tuple[int, int]]) -> tuple[list[Record], list[Record]]:
    """Union linked records into clusters and keep the largest image of each."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(b)] = find(a)
    clusters: dict[int, list[Record]] = {}
    for i, r in enumerate(records):
        clusters.setdefault(find(i), []).append(r)
    keep, dropped = [], []
    for group in clusters.values():
        group.sort(key=lambda r: (r.width or 0) * (r.height or 0), reverse=True)
        keep.append(group[0])
        dropped += group[1:]
    return keep, dropped


def _phash(path: str):
    import imagehash

    with open_drafted(path, min_edge=512) as im:
        return imagehash.phash(im.convert("RGB"))


def dedup_hash(records: list[Record], max_hamming: int = 6, workers: int | None = None,
               progress: bool = True) -> tuple[list[Record], list[Record]]:
    """Drop exact copies (same SHA-256), then near-copies (perceptual hashes within
    `max_hamming` bits: the same photo resized, recompressed or lightly edited)."""
    by_sha: dict[str, list[Record]] = {}
    for r in records:
        by_sha.setdefault(r.sha256 or r.id, []).append(r)
    unique, dropped = [], []
    for group in by_sha.values():
        kept, gone = _resolve_clusters(group, [(0, j) for j in range(1, len(group))])
        unique += kept
        dropped += gone

    hashes = pmap(_phash, [r.path for r in unique], "phash", workers, progress)
    hashed = [i for i, h in enumerate(hashes) if not isinstance(h, BaseException)]
    bits = np.array([hashes[i].hash.reshape(-1) for i in hashed], dtype=np.float32)
    inv = 1.0 - bits
    pairs = _linked_pairs(
        lambda i0, i1: bits[i0:i1] @ inv.T + inv[i0:i1] @ bits.T <= max_hamming, len(hashed))
    kept, gone = _resolve_clusters([unique[i] for i in hashed], pairs)
    unhashed = [r for i, r in enumerate(unique) if isinstance(hashes[i], BaseException)]
    return kept + unhashed, dropped + gone


def dedup_semantic(records: list[Record], threshold: float = 0.92,
                   progress: bool = True) -> tuple[list[Record], list[Record]]:
    """Drop records whose DINOv2 embeddings are this similar: the same scene, re-shot."""
    from .embed import embed, unit

    feats = unit(embed([r.path for r in records], progress=progress)[0])
    pairs = _linked_pairs(lambda i0, i1: feats[i0:i1] @ feats.T >= threshold, len(records))
    return _resolve_clusters(records, pairs)


def split_records(records: list[Record], test_fraction: float = 0.15, min_test: int = 20,
                  seed: int = 0) -> tuple[list[Record], list[Record]]:
    """A seeded train/test split: `test_fraction` of the images, at least `min_test` so the
    metrics have something to work with, and never more than half."""
    n_test = min(max(min_test, round(len(records) * test_fraction)), len(records) // 2)
    order = random.Random(seed).sample(sorted(records, key=lambda r: r.id), len(records))
    test, train = order[:n_test], order[n_test:]
    for r in train:
        r.split = "train"
    for r in test:
        r.split = "test"
    return train, test


def curate(dirs: Sequence[str | Path], out: str | Path, semantic: bool = True,
           semantic_threshold: float = 0.92, test_fraction: float = 0.15, min_test: int = 20,
           seed: int = 0, workers: int | None = None, progress: bool = True) -> dict:
    """Scan, deduplicate and split; write the manifests to `out`. Returns the report."""
    out = Path(out)
    records = scan(dirs, workers, progress)
    if not records:
        raise ValueError(f"no images found under {', '.join(map(str, dirs))}")
    print(f"{len(records)} images")
    cameras = Counter(r.extra.get("camera_model") or "unknown" for r in records)
    print("cameras: " + ", ".join(f"{c} ({n})" for c, n in cameras.most_common()))
    shapes = Counter(r.oriented_size for r in records)
    for (w, h), n in shapes.most_common(4):
        print(f"  {w}x{h}  aspect {w / h:.3f}  {n} images")

    kept, hash_dropped = dedup_hash(records, workers=workers, progress=progress)
    print(f"hash dedup: dropped {len(hash_dropped)}")
    semantic_dropped: list[Record] = []
    if semantic and len(kept) > 2:
        kept, semantic_dropped = dedup_semantic(kept, semantic_threshold, progress)
        print(f"semantic dedup (DINOv2 cosine >= {semantic_threshold}): "
              f"dropped {len(semantic_dropped)}")

    train, test = split_records(kept, test_fraction, min_test, seed)
    write_manifest(out / "manifest.jsonl", train + test)
    write_manifest(out / "manifest_train.jsonl", train)
    write_manifest(out / "manifest_test.jsonl", test)
    report = {"input": len(records), "hash_duplicates": len(hash_dropped),
              "semantic_duplicates": len(semantic_dropped), "kept": len(kept),
              "train": len(train), "test": len(test)}
    write_json(out / "curate_report.json", report)
    print(f"kept {len(kept)}: train {len(train)}, test {len(test)}, written to {out}")
    return report
