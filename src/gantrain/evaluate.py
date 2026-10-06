"""How good and how varied generated images are, measured against real ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

from .embed import CLIP, DINO, embed, unit
from .imageio import grid_cells

CMMD_SIGMA = 10.0
CMMD_SCALE = 1000.0
KID_SUBSETS = 100
KID_DEGREE = 3
NEIGHBOURS = 3
COPY_SIMILARITY = 0.95


def cmmd(real: np.ndarray, fake: np.ndarray) -> float:
    """CLIP maximum mean discrepancy (Jayasumana et al., 2024), on raw CLIP embeddings."""
    gamma = 1.0 / (2.0 * CMMD_SIGMA**2)

    def k(a, b):
        return np.exp(-gamma * cdist(a, b, "sqeuclidean")).mean()

    x, y = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    return float(CMMD_SCALE * (k(x, x) + k(y, y) - 2 * k(x, y)))


def frechet_distance(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between Gaussians fitted to two feature sets."""
    x, y = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    diff = x.mean(0) - y.mean(0)
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def kid(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance with a cubic polynomial kernel: (mean, std) over subsets."""
    x, y = np.asarray(real, np.float64), np.asarray(fake, np.float64)
    n = min(len(x), len(y))
    m = min(1000, n // 2)
    if m < 4:
        return float("nan"), float("nan")

    def once(a, b):
        d = x.shape[1]
        kxx, kyy, kxy = ((p @ q.T / d + 1.0) ** KID_DEGREE for p, q in ((a, a), (b, b), (a, b)))
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        return kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()

    rng = np.random.default_rng(seed)
    vals = [once(x[rng.choice(len(x), m, replace=False)], y[rng.choice(len(y), m, replace=False)])
            for _ in range(KID_SUBSETS)]
    return float(np.mean(vals)), float(np.std(vals))


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = NEIGHBOURS) -> tuple[float, float]:
    """Improved precision (fidelity) and recall (coverage), Kynkäänniemi et al. 2019."""
    def radii(a):
        d = cdist(a, a)
        np.fill_diagonal(d, np.inf)
        kk = min(k, max(1, len(a) - 1))
        return np.partition(d, kk - 1, axis=1)[:, kk - 1]

    d_fr = cdist(fake, real)
    precision = float((d_fr <= radii(real)[None, :]).any(axis=1).mean())
    recall = float((d_fr.T <= radii(fake)[None, :]).any(axis=1).mean())
    return precision, recall


def vendi_score(feats: np.ndarray) -> float:
    """The effective number of distinct items in a set (Friedman and Dieng, 2023)."""
    x = unit(np.asarray(feats, np.float64))
    if len(x) < 2:
        return float(len(x))
    w = np.linalg.eigvalsh(x @ x.T / len(x))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum())) if len(w) else 1.0


def memorisation(fake: np.ndarray, real: np.ndarray, fake_paths: list, real_paths: list,
                 top_n: int = 10) -> dict:
    """For each generated image, its most similar real one; the closest pairs are listed."""
    sim = unit(fake) @ unit(real).T
    top1, nearest = sim.max(axis=1), sim.argmax(axis=1)
    worst = [{"generated": str(fake_paths[i]), "nearest_real": str(real_paths[nearest[i]]),
              "similarity": round(float(top1[i]), 4)} for i in np.argsort(-top1)[:top_n]]
    return {"max_similarity": round(float(top1.max()), 4),
            "copies": int((top1 >= COPY_SIMILARITY).sum()), "threshold": COPY_SIMILARITY,
            "worst": worst}


def evaluate(real_paths: list, fake_paths: list, compute_fd: bool = True) -> dict:
    """Every metric, from one embedding pass over each set."""
    warnings = []
    n = min(len(real_paths), len(fake_paths))
    if n < 50:
        warnings.append(f"only {n} samples per side; every metric here is noisy")
    if len(real_paths) != len(fake_paths):
        warnings.append(f"subsampled both sides to {n} so the set metrics compare like with like")
        real_paths = list(real_paths)[::max(1, len(real_paths) // n)][:n]
        fake_paths = list(fake_paths)[::max(1, len(fake_paths) // n)][:n]
    if compute_fd and n < 2000:
        warnings.append(f"FD-DINOv2 fits a 768-dimensional Gaussian from {n} samples; "
                        "treat it as indicative and prefer CMMD")

    dino_r, clip_r = embed(real_paths, (DINO, CLIP))
    dino_f, clip_f = embed(fake_paths, (DINO, CLIP))
    dino_r, dino_f = unit(dino_r), unit(dino_f)
    kid_mean, kid_std = kid(dino_r, dino_f)
    precision, recall = precision_recall(dino_r, dino_f)
    mem = memorisation(dino_f, dino_r, fake_paths, real_paths)
    if mem["copies"]:
        warnings.append(f"{mem['copies']} generated image(s) are near-copies of real ones")
    if recall < 0.3 and precision > 0.7 and n >= 100:
        warnings.append("high precision with low recall: the model covers one corner of the data")

    return {"real": len(real_paths), "generated": len(fake_paths),
            "cmmd": round(cmmd(clip_r, clip_f), 4),
            "fd_dinov2": round(frechet_distance(dino_r, dino_f), 3) if compute_fd else None,
            "kid_mean": kid_mean, "kid_std": kid_std,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "vendi_generated": round(vendi_score(dino_f), 3),
            "vendi_real": round(vendi_score(dino_r), 3),
            "memorisation": mem, "warnings": warnings}


def render(report: dict) -> str:
    rows = [("samples", f"real={report['real']}  generated={report['generated']}"),
            ("CMMD (lower is better)", f"{report['cmmd']:.4f}")]
    if report["fd_dinov2"] is not None:
        rows.append(("FD-DINOv2 (lower is better)", f"{report['fd_dinov2']:.2f}"))
    rows += [("KID (lower is better)", f"{report['kid_mean']:.5f} +/- {report['kid_std']:.5f}"),
             ("precision (fidelity)", f"{report['precision']:.4f}"),
             ("recall (coverage)", f"{report['recall']:.4f}"),
             ("Vendi generated", f"{report['vendi_generated']:.2f}"),
             ("Vendi real", f"{report['vendi_real']:.2f}"),
             ("closest copy", f"{report['memorisation']['max_similarity']:.3f}")]
    return "\n".join([f"{k:<30}{v}" for k, v in rows] + [f"  ! {w}" for w in report["warnings"]])


def fixed_noise_diversity(run: str | Path) -> list[dict]:
    """Vendi score of each saved fixed-noise grid's cells, per training step."""
    grids = sorted((Path(run) / "samples").glob("*.jpg"))
    cells = [grid_cells(p) for p in grids]
    if not cells:
        return []
    feats = embed([c for group in cells for c in group], progress=False)[0]
    per = len(cells[0])
    return [{"step": int(p.stem), "cells": per,
             "vendi": round(vendi_score(feats[i * per:(i + 1) * per]), 4)}
            for i, p in enumerate(grids)]
