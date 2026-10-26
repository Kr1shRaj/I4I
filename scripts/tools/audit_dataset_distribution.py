"""
Audit datasets/9class/ (or any YOLO-seg list-based dataset): split balance, area distribution, geometry integrity
================================================================================================================
Reads the image lists named in a data YAML (default datasets/9class/data_9class.yaml), resolves every entry the
way Ultralytics does ('./' -> list folder; /images/ -> /labels/), and reports per split:

- images, positives, negatives (empty label), missing images / labels, repeated list entries (oversampling)
- instances per class, images per class
- bounding-box area per class: tiny < 5%, medium 5-50%, large 50-80%, oversized > 80% of the image
- geometry: malformed lines, class id out of range, coordinates outside [0, 1], < 3 points,
  self-intersecting polygons (excluding zero-width bridges), zero-area polygons, duplicate labels
  (same class, polygon IoU > 0.95 within one image)

Usage
-----
    python scripts/tools/audit_dataset_distribution.py
    python scripts/tools/audit_dataset_distribution.py --data datasets/9class/data_9class.yaml --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINS = [("tiny<5%", 0.0, 0.05), ("med5-50%", 0.05, 0.50), ("large50-80%", 0.50, 0.80), ("over>80%", 0.80, 1.01)]


def read_list(list_path: Path) -> list[Path]:
    out = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        x = line.strip()
        if not x:
            continue
        out.append(Path(str(list_path.parent) + os.sep + x[2:]) if x.startswith("./") else Path(x))
    return out


def label_for(img: Path) -> Path:
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(s.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def self_intersects(pts: np.ndarray) -> bool:
    """Proper crossings between non-adjacent edges; collinear/touching edges (zero-width bridges) are ignored."""
    n = len(pts)
    if n < 4:
        return False
    a = pts
    b = np.roll(pts, -1, axis=0)

    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if not len(j):
            continue
        o1 = orient(a[i], b[i], a[j]); o2 = orient(a[i], b[i], b[j])
        o3 = orient(a[j], b[j], a[i][None]); o4 = orient(a[j], b[j], b[i][None])
        if np.any((o1 * o2 < 0) & (o3 * o4 < 0)):
            return True
    return False


def poly_iou(p1: np.ndarray, p2: np.ndarray, size: int = 256) -> float:
    m1 = np.zeros((size, size), np.uint8); m2 = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m1, [(p1 * size).astype(np.int32)], 1); cv2.fillPoly(m2, [(p2 * size).astype(np.int32)], 1)
    u = (m1 | m2).sum()
    return float((m1 & m2).sum() / u) if u else 0.0


def audit_split(name: str, list_path: Path, names: dict[int, str]) -> dict:
    entries = read_list(list_path)
    uniq = list(dict.fromkeys(entries))
    nc = len(names)
    inst = Counter(); imgs_per_cls = Counter(); area_bins = defaultdict(Counter)
    geo = Counter(); examples = defaultdict(list)
    pos = neg = miss_img = miss_lbl = 0
    for img in uniq:
        if not img.exists():
            miss_img += 1; examples["missing_image"].append(str(img)); continue
        lp = label_for(img)
        if not lp.exists():
            miss_lbl += 1; examples["missing_label"].append(str(lp)); continue
        polys = []
        for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
            p = line.split()
            if not p:
                continue
            try:
                c = int(p[0]); v = np.array(p[1:], float)
            except ValueError:
                geo["malformed"] += 1; examples["malformed"].append(f"{lp}:{ln}"); continue
            if not 0 <= c < nc:
                geo["class_out_of_range"] += 1; examples["class_out_of_range"].append(f"{lp}:{ln}"); continue
            if len(v) < 6 or len(v) % 2:
                geo["lt3_points"] += 1; examples["lt3_points"].append(f"{lp}:{ln}"); continue
            if v.min() < 0 or v.max() > 1:
                geo["out_of_bounds"] += 1; examples["out_of_bounds"].append(f"{lp}:{ln}")
            xy = np.clip(v.reshape(-1, 2), 0, 1)
            x, y = xy[:, 0], xy[:, 1]
            parea = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            if parea < 1e-6:
                geo["zero_area"] += 1; examples["zero_area"].append(f"{lp}:{ln}")
            if self_intersects(xy):
                geo["self_intersecting"] += 1; examples["self_intersecting"].append(f"{lp}:{ln}")
            box = float(np.ptp(x) * np.ptp(y))
            for b, lo, hi in BINS:
                if lo <= box < hi:
                    area_bins[c][b] += 1
            inst[c] += 1
            polys.append((c, xy))
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i][0] == polys[j][0] and poly_iou(polys[i][1], polys[j][1]) > 0.95:
                    geo["duplicate_label"] += 1; examples["duplicate_label"].append(str(lp))
        if polys:
            pos += 1
            for c in {c for c, _ in polys}:
                imgs_per_cls[c] += 1
        else:
            neg += 1
    return {"list": list_path.name, "entries": len(entries), "unique_images": len(uniq), "positives": pos,
            "negatives": neg, "missing_images": miss_img, "missing_labels": miss_lbl,
            "instances": {names[c]: inst[c] for c in range(nc)},
            "images_per_class": {names[c]: imgs_per_cls[c] for c in range(nc)},
            "area_bins": {names[c]: {b: area_bins[c][b] for b, _, _ in BINS} for c in range(nc)},
            "geometry": dict(geo), "examples": {k: v[:5] for k, v in examples.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "datasets" / "9class" / "data_9class.yaml")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    res = {}
    for split in ("train", "val", "test"):
        if cfg.get(split):
            res[split] = audit_split(split, args.data.parent / cfg[split], names)

    print(f"\n=== Split balance ({args.data.name})")
    print(f"{'split':<6} {'entries':>8} {'unique':>7} {'pos':>6} {'neg':>5} {'neg%':>5} {'miss img':>8} {'miss lbl':>8}")
    for s, r in res.items():
        print(f"{s:<6} {r['entries']:>8} {r['unique_images']:>7} {r['positives']:>6} {r['negatives']:>5} "
              f"{100 * r['negatives'] / max(1, r['unique_images']):>4.0f}% {r['missing_images']:>8} {r['missing_labels']:>8}")
    print("\n=== Instances (images) per class")
    print(f"{'class':<22}" + "".join(f"{s:>16}" for s in res))
    for c in names.values():
        print(f"{c:<22}" + "".join(f"{res[s]['instances'][c]:>9} ({res[s]['images_per_class'][c]:>4})" for s in res))
    print("\n=== Box-area distribution per class (% of instances): tiny<5 / med5-50 / large50-80 / over>80")
    for c in names.values():
        row = f"{c:<22}"
        for s in res:
            b = res[s]["area_bins"][c]; t = max(1, sum(b.values()))
            row += "  " + "/".join(f"{100 * b[k] / t:>3.0f}" for k, _, _ in BINS)
        print(row)
    print("\n=== Geometry integrity")
    for s, r in res.items():
        print(f"{s:<6} {r['geometry'] or 'clean'}")
        for k, v in r["examples"].items():
            print(f"         e.g. {k}: {v[0]}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    bad = sum(r["missing_images"] + r["missing_labels"] + sum(r["geometry"].get(k, 0) for k in
              ("malformed", "class_out_of_range", "lt3_points", "out_of_bounds")) for r in res.values())
    print(f"\nRESULT: {'PASS' if not bad else 'FAIL'} ({bad} blocking issues: missing files, malformed, "
          f"out-of-range class, <3 points, out-of-bounds coords)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
