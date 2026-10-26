"""
scripts/tools/compare_sam_models.py
===================================
Compare two SAM checkpoints (default sam2.1_b.pt vs sam2.1_l.pt) as box-prompted annotators on the
9-class val set, per class, against the existing hand/Roboflow polygons.

For each sampled instance the ground-truth box (+2% margin) is the prompt. Metrics:
- mask IoU against the ground-truth polygon
- boundary F1 at a tolerance of 1% of the box diagonal (how well the outline follows the real edge)
- latency per prompt and peak GPU memory
The ground-truth polygons are coarse (few vertices), so absolute IoU is capped below 1; the useful
signal is the b-vs-l difference on the same instances. A crop sheet of the hardest geometries
(rollers, blades, fences) is written for visual comparison.

Usage
-----
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\compare_sam_models.py
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\compare_sam_models.py --per-class 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
OUT_DIR = DATASET_DIR / "label_fixes" / "sam_compare"
CLASSES = ["panel_saw", "hydraulic_press", "table_saw", "planer", "jointer",
           "band_saw", "miter_saw", "edge_banding_machine", "fire_extinguisher"]


def sample(per_class, seed):
    rng = np.random.default_rng(seed)
    by_cls = {c: [] for c in range(len(CLASSES))}
    for line in (DATASET_DIR / "valid_9class.txt").read_text().split():
        ip = (DATASET_DIR / line).resolve()
        lp = DATASET_DIR / "valid_9class" / "labels" / (ip.stem + ".txt")
        rows = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        for k, r in enumerate(rows):
            xy = np.array(r[1:], float).reshape(-1, 2)
            bw, bh = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
            if 0.02 < bw * bh < 0.8 and len(xy) >= 6:     # skip specks and frame-filling slabs
                by_cls[int(r[0])].append((str(ip), k, xy))
    out = []
    for c, lst in by_cls.items():
        pick = rng.choice(len(lst), size=min(per_class, len(lst)), replace=False) if lst else []
        out += [(c,) + lst[i] for i in pick]
    return out


def boundary_f1(pred, gt, tol):
    def edge(m):
        return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ep, eg = edge(pred), edge(gt)
    if not ep.any() or not eg.any():
        return 0.0
    dg = cv2.distanceTransform((~eg).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 3)
    precision = (dg[ep] <= tol).mean()
    recall = (dp[eg] <= tol).mean()
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs=2, default=["sam2.1_b.pt", "sam2.1_l.pt"])
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from ultralytics import SAM
    items = sample(args.per_class, args.seed)
    print(f"{len(items)} instances")
    res = {m: [] for m in args.models}
    masks_keep = {m: {} for m in args.models}
    for m in args.models:
        sam = SAM(str(PROJECT_ROOT / "models" / "pretrained" / m))
        torch.cuda.reset_peak_memory_stats()
        sam(np.zeros((64, 64, 3), np.uint8), bboxes=[[8, 8, 40, 40]], verbose=False)  # warm-up
        for j, (c, path, k, xy) in enumerate(items):
            im = cv2.imread(path)
            h, w = im.shape[:2]
            gt = np.zeros((h, w), np.uint8)
            cv2.fillPoly(gt, [(xy * [w, h]).astype(np.int32)], 1)
            gt = gt.astype(bool)
            x1, y1 = xy.min(0) * [w, h]
            x2, y2 = xy.max(0) * [w, h]
            mx, my = 0.02 * w, 0.02 * h
            box = [max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)]
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = sam(im, bboxes=[box], verbose=False)[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            pm = r.masks.data[0].cpu().numpy() > 0.5 if r.masks is not None and len(r.masks) else np.zeros_like(gt)
            if pm.shape != gt.shape:
                pm = cv2.resize(pm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            iou = float((pm & gt).sum() / max((pm | gt).sum(), 1))
            tol = 0.01 * np.hypot(x2 - x1, y2 - y1)
            res[m].append({"cls": c, "iou": iou, "bf1": boundary_f1(pm, gt, tol), "ms": dt * 1000})
            masks_keep[m][j] = pm
        res[m + "_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        del sam
        torch.cuda.empty_cache()

    a, b = args.models
    table = []
    print(f"\n{'class':<22}{'n':>4}{'IoU ' + a[:9]:>16}{'IoU ' + b[:9]:>16}{'BF1 ' + a[:9]:>16}{'BF1 ' + b[:9]:>16}{'l>b':>6}")
    for c in list(range(len(CLASSES))) + [None]:
        ra = [r for r in res[a] if c is None or r["cls"] == c]
        rb = [r for r in res[b] if c is None or r["cls"] == c]
        if not ra:
            continue
        row = {"class": "ALL" if c is None else CLASSES[c], "n": len(ra),
               "iou_a": np.mean([r["iou"] for r in ra]), "iou_b": np.mean([r["iou"] for r in rb]),
               "bf1_a": np.mean([r["bf1"] for r in ra]), "bf1_b": np.mean([r["bf1"] for r in rb]),
               "b_better": float(np.mean([y["iou"] > x["iou"] + 0.01 for x, y in zip(ra, rb)])),
               "a_better": float(np.mean([x["iou"] > y["iou"] + 0.01 for x, y in zip(ra, rb)]))}
        table.append(row)
        print(f"{row['class']:<22}{row['n']:>4}{row['iou_a']:>16.3f}{row['iou_b']:>16.3f}"
              f"{row['bf1_a']:>16.3f}{row['bf1_b']:>16.3f}{row['b_better']:>6.0%}")
    for m in args.models:
        print(f"{m}: {np.median([r['ms'] for r in res[m]]):.0f} ms/prompt (median), peak {res[m + '_peak_mb']:.0f} MB")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps({"table": table, "raw": res}, indent=1, default=float))

    # crop sheet: instances where the two models disagree most, from the thin-geometry classes
    thin = [j for j, it in enumerate(items) if CLASSES[it[0]] in
            ("panel_saw", "table_saw", "band_saw", "jointer", "planer", "edge_banding_machine")]
    thin.sort(key=lambda j: -abs(res[a][j]["iou"] - res[b][j]["iou"]))
    tiles = []
    for j in thin[:8]:
        c, path, k, xy = items[j]
        im = cv2.imread(path)
        h, w = im.shape[:2]
        x1, y1 = (xy.min(0) * [w, h]).astype(int)
        x2, y2 = (xy.max(0) * [w, h]).astype(int)
        row = []
        for m, col in ((a, (0, 0, 255)), (b, (0, 200, 0))):
            ov = im.copy()
            ov[masks_keep[m][j]] = (0.5 * ov[masks_keep[m][j]] + 0.5 * np.array(col)).astype(np.uint8)
            cv2.polylines(ov, [(xy * [w, h]).astype(np.int32)], True, (255, 255, 0), 2)
            crop = ov[max(0, y1):y2, max(0, x1):x2]
            s = 420 / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
            t = np.full((440, 420, 3), 255, np.uint8)
            t[:crop.shape[0], :crop.shape[1]] = crop
            cv2.putText(t, f"{m[:9]} {CLASSES[c][:10]} IoU {res[m][j]['iou']:.2f}", (3, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            row.append(t)
        tiles.append(np.hstack(row))
    if tiles:
        pairs = [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.full_like(tiles[i], 255)])
                 for i in range(0, len(tiles), 2)]
        cv2.imwrite(str(OUT_DIR / "disagreements.jpg"), np.vstack(pairs), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"results + disagreements.jpg in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
