"""
Propose tightened edge_banding_machine polygons with SAM 2.1 (Task 2 of the run8 prep)
=====================================================================================
Policy (docs/label_policy.md): one instance per physical machine; include chassis, top cover, gluing/trimming
units, infeed/outfeed roller conveyors and the edge-tape coil holder; exclude panels being fed, floor, walls,
text banners and inset photos.

For every source edge-bander polygon (38-class id 26) in the 9-class build whose bounding box covers > --flag-box
of the image, SAM 2.1 is prompted with that box (plus optional manual boxes from the overrides YAML) and
the mask is converted to one polygon. Proposals are written to a JSON file for review; nothing in the dataset
is changed here. build_clean_9class_dataset.py applies accepted proposals.

Usage
-----
    python scripts/tools/fix_edge_bander_labels.py --out datasets/9class/label_fixes/edge_bander_proposals.json --review-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel_cctv import mask_to_polygon  # noqa: E402

SRC = PROJECT_ROOT / "datasets" / "source_38class"
DST = PROJECT_ROOT / "datasets" / "9class"
EB_SRC = 26


def foreground(m: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """SAM given a near-full-image box often returns the plain catalog BACKGROUND. If the mask covers most of the
    prompt box's border (inset 4 px), invert it inside the box; then keep the largest component plus components
    >= 15% of it (drops text/icons)."""
    x1, y1, x2, y2 = box
    i = 4
    sub = m[y1 + i:y2 - i, x1 + i:x2 - i]
    if sub.size and np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]]).mean() > 0.5:
        inv = np.zeros_like(m)
        inv[y1:y2, x1:x2] = ~m[y1:y2, x1:x2]
        m = inv
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * areas.max()]
    return np.isin(lab, keep)


def parea(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flag-box", type=float, default=0.80)
    ap.add_argument("--sam", type=Path, default=PROJECT_ROOT / "models" / "pretrained" / "sam2.1_l.pt")
    ap.add_argument("--review-dir", type=Path, default=None)
    args = ap.parse_args()

    from ultralytics import SAM
    sam = SAM(str(args.sam))
    proposals, tiles = [], []
    for split in ("train", "valid", "test"):
        for lp in sorted((SRC / split / "labels").glob("*.txt")):
            if not (DST / f"{split}_9class" / "labels" / lp.name).exists():
                continue
            polys = [np.array(l.split()[1:], float).reshape(-1, 2) for l in lp.read_text().splitlines()
                     if l.split() and int(l.split()[0]) == EB_SRC]
            flagged = [p for p in polys if np.ptp(p[:, 0]) * np.ptp(p[:, 1]) > args.flag_box]
            if not flagged:
                continue
            im = cv2.imread(str(SRC / split / "images" / (lp.stem + ".jpg")))
            h, w = im.shape[:2]
            for k, p in enumerate(flagged):
                x1, y1 = p.min(0) * [w, h]
                x2, y2 = p.max(0) * [w, h]
                r = sam(im, bboxes=[[x1, y1, x2, y2]], verbose=False)[0]
                m = (r.masks.data.cpu().numpy()[0] > 0.5)
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                m = foreground(m, (int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))))
                poly = mask_to_polygon(m, 0.002)
                new = (np.clip(poly / [w, h], 0, 1)).round(6).tolist() if poly is not None else None
                rec = {"split": split, "stem": lp.stem, "k": k, "old_box": round(float(np.ptp(p[:, 0]) * np.ptp(p[:, 1])), 3),
                       "old_poly_area": round(float(parea(p)), 3), "new_area": round(float(m.mean()), 3), "polygon": new}
                proposals.append(rec)
                if args.review_dir is not None:
                    vis = im.copy()
                    cv2.polylines(vis, [(p * [w, h]).astype(np.int32)], True, (0, 0, 255), max(2, w // 150))
                    if poly is not None:
                        ov = vis.copy(); ov[m] = (0, 255, 0)
                        vis = cv2.addWeighted(vis, 0.6, ov, 0.4, 0)
                    t = cv2.resize(vis, (360, 360))
                    cv2.putText(t, f"{len(proposals) - 1} {split[:2]} old{rec['old_poly_area']:.2f} new{rec['new_area']:.2f}",
                                (4, 22), 0, 0.55, (0, 255, 255), 2)
                    tiles.append(t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proposals, indent=1), encoding="utf-8")
    if args.review_dir is not None:
        args.review_dir.mkdir(parents=True, exist_ok=True)
        for s in range(0, len(tiles), 20):
            chunk = tiles[s:s + 20]
            while len(chunk) % 5:
                chunk.append(np.zeros_like(chunk[0]))
            cv2.imwrite(str(args.review_dir / f"eb_{s // 20:02d}.jpg"),
                        np.vstack([np.hstack(chunk[i:i + 5]) for i in range(0, len(chunk), 5)]))
    print(f"{len(proposals)} flagged edge-bander polygons -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
