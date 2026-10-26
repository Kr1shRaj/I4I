"""
Rebuild object instances from fragmented source polygons without fusing touching objects
=========================================================================================
Replaces the morphological-closing "merge" that build_clean_9class_dataset.py used for
fire_extinguisher and edge_banding_machine. That closing joined every polygon within a few pixels,
so extinguishers standing side by side became one instance (e.g. 3 cylinders -> 1 blob).

Rule (per image, per class, in pixels):
1. Speckle consolidation: polygons smaller than `small_ratio` of the image are rasterised together and
   closed with a small kernel, so a cylinder annotated only as hundreds of specks becomes one region.
   Large polygons are never closed together.
2. Candidates = large polygons + consolidated speckle regions, each >= `min_area_ratio` of the image.
3. Largest first: a candidate is absorbed into an existing instance if it is small relative to it
   (< `absorb_ratio` of its area) and touches it (>= `touch_frac` of the candidate lies within a
   dilation of `dilate_frac` x image diagonal), e.g. a hose or handle; or if >= 50% of it lies inside it
   (duplicate). Otherwise it becomes a new instance. Similar-sized neighbours therefore stay separate.
4. Each instance -> one polygon (fragments < 10% dropped, rest joined by zero-width bridges).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel_cctv import mask_to_polygon  # noqa: E402

PARAMS = {
    # class name: thresholds (ratios of image area / diagonal)
    "fire_extinguisher": dict(small_ratio=0.002, min_area_ratio=0.0005, close_frac=0.006,
                              absorb_ratio=0.35, touch_frac=0.3, dilate_frac=0.01),
    "edge_banding_machine": dict(small_ratio=0.01, min_area_ratio=0.005, close_frac=0.012,
                                 absorb_ratio=0.35, touch_frac=0.3, dilate_frac=0.015),
}


def rebuild_instances(polys_px: list[np.ndarray], shape: tuple[int, int], small_ratio: float,
                      min_area_ratio: float, close_frac: float, absorb_ratio: float, touch_frac: float,
                      dilate_frac: float, simplify: float = 0.002) -> list[np.ndarray]:
    h, w = shape
    img_area, diag = h * w, float(np.hypot(h, w))
    large, small = [], np.zeros((h, w), np.uint8)
    for p in polys_px:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [p.astype(np.int32)], 1)
        if m.sum() >= small_ratio * img_area:
            large.append(m.astype(bool))
        else:
            small |= m
    cands = list(large)
    if small.any():
        k = max(3, int(close_frac * diag)) | 1
        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab = cv2.connectedComponents(closed)
        cands += [lab == i for i in range(1, n)]
    cands = [m for m in cands if m.sum() >= min_area_ratio * img_area]
    cands.sort(key=lambda m: -int(m.sum()))

    r = max(1, int(dilate_frac * diag))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inst: list[np.ndarray] = []
    for m in cands:
        a = int(m.sum())
        target = None
        for i, im in enumerate(inst):
            if (im & m).sum() >= 0.5 * a:
                target = i
                break
            if a < absorb_ratio * im.sum():
                near = cv2.dilate(im.astype(np.uint8), kern).astype(bool)
                if (near & m).sum() >= touch_frac * a:
                    target = i
                    break
        if target is None:
            inst.append(m.copy())
        else:
            inst[target] |= m
    out = []
    for m in inst:
        p = mask_to_polygon(m, simplify)
        if p is not None and len(p) >= 3:
            out.append(p)
    return out


def rebuild_from_norm(coord_lists: list[list[float]], shape: tuple[int, int], class_name: str) -> list[np.ndarray]:
    """Normalised YOLO coordinate lists -> normalised instance polygons (N x 2 arrays)."""
    h, w = shape
    polys = [(np.array(c, float).reshape(-1, 2) * [w, h]) for c in coord_lists]
    out = rebuild_instances(polys, (h, w), **PARAMS[class_name])
    return [np.clip(p.astype(np.float64) / [w, h], 0, 1) for p in out]
