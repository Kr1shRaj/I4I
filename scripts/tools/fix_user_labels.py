"""
scripts/tools/fix_user_labels.py
================================
Re-annotate the images the user listed under "TO FIX ANNOTATIONS" (datasetchange ->
parse_user_curation.py -> datasets/9class/label_fixes/user_curation.json) with SAM 2.1 Large box prompts,
following docs/label_policy.md.

Steps (outputs in datasets/9class/label_fixes/review_user_fixes/):
    review   current 9-class label (white outline) + run5 detections (numbered, coloured) on a 10% grid
             -> review/sheet_XX.jpg, and seed_boxes.yaml (run5 boxes >= --conf, else the current label boxes)
             as a starting point for the reviewed boxes
    propose  read label_decisions.yaml -> user_fixes (per stem: a list of [class, x1, y1, x2, y2] boxes in
             normalised coordinates), run SAM 2.1 L, clip each mask to its box (no room, floor, stock),
             write ../user_fix_polygons.json and before|after sheets (sheet_XX.jpg)

The build (build_clean_9class_dataset.py) writes the polygons from user_fix_polygons.json for these stems
instead of the source labels.

Usage
-----
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\fix_user_labels.py review
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\fix_user_labels.py propose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_highangle import (CLASSES, COLORS, EPS, SHORT, foreground, grid_tile,  # noqa: E402
                                keep_main_parts, write_sheets)
from prelabel_cctv import mask_to_polygon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "datasets" / "source_38class"
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
FIX_DIR = DATASET_DIR / "label_fixes"
OUT_DIR = FIX_DIR / "review_user_fixes"
SAM_WEIGHTS = PROJECT_ROOT / "models" / "pretrained" / "sam2.1_l.pt"
HINT_WEIGHTS = PROJECT_ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt"


def fix_items():
    cur = json.loads((FIX_DIR / "user_curation.json").read_text(encoding="utf-8"))
    return cur["fix"]["images"]


def current_label(it):
    """Polygons the current build writes for this image (what training sees now)."""
    lp = DATASET_DIR / f"{it['split']}_9class" / "labels" / (it["stem"] + ".txt")
    out = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) >= 7:
                out.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2)))
    return out


def studio_foreground(im, white=232, close_frac=0.01):
    """Object mask on a plain white background: non-white pixels, gaps closed, holes filled."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] < white) | (hsv[..., 1] > 30)).astype(np.uint8)
    k = max(3, int(close_frac * np.hypot(*im.shape[:2]))) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def draw_polys(tile, polys, s, w, h, color=None, fill=True, thick=2):
    for c, poly in polys:
        pts = (poly * [w * s, h * s]).astype(np.int32)
        col = COLORS[c] if color is None else color
        if fill:
            ov = tile.copy()
            cv2.fillPoly(ov, [pts], col)
            tile[:] = cv2.addWeighted(ov, 0.35, tile, 0.65, 0)
        cv2.polylines(tile, [pts], True, col, thick)
        cv2.putText(tile, SHORT[c], tuple(int(v) for v in pts.min(0) + [3, 14]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def cmd_review(args):
    from ultralytics import YOLO
    model = YOLO(str(HINT_WEIGHTS))
    tiles, seeds = [], {}
    for n, it in enumerate(fix_items()):
        ip = SRC_DIR / it["split"] / "images" / (it["stem"] + ".jpg")
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        r = model.predict(str(ip), imgsz=960, conf=0.10, verbose=False)[0]
        tile, s = grid_tile(im, args.size, f"#{n} {it['split']} {'/'.join(SHORT[c] for c in it['classes'])} "
                                           f"{it['stem'][4:30]}")
        old = current_label(it)
        draw_polys(tile, old, s, w, h, color=(255, 255, 255), fill=False, thick=2)
        dets = []
        for k, b in enumerate(r.boxes):
            c, conf = int(b.cls), float(b.conf)
            box = [round(v, 3) for v in b.xyxyn[0].tolist()]
            dets.append((c, conf, box))
            x1, y1, x2, y2 = (np.array(box) * [w * s, h * s, w * s, h * s]).astype(int)
            cv2.rectangle(tile, (x1, y1), (x2, y2), COLORS[c], 2)
            cv2.putText(tile, f"d{k} {SHORT[c]}{conf:.2f}", (x1 + 3, max(12, y2 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[c], 2)
        keep = [[CLASSES[c], *box] for c, conf, box in dets if conf >= args.conf and c in it["classes"]]
        if not keep:   # fall back to the current label boxes
            keep = [[CLASSES[c], *[round(float(v), 3) for v in (*p.min(0), *p.max(0))]] for c, p in old]
        seeds[it["stem"]] = keep
        tiles.append(tile)
    write_sheets(tiles, OUT_DIR / "review")
    with open(OUT_DIR / "seed_boxes.yaml", "w", encoding="utf-8") as f:
        for n, (stem, boxes) in enumerate(seeds.items()):
            f.write(f"  {stem}:  # #{n}\n")
            for b in boxes:
                f.write(f"    - [{b[0]}, {', '.join(f'{v:.2f}' for v in b[1:])}]\n")
    print(f"{len(tiles)} images, sheets in {OUT_DIR / 'review'}, seeds in seed_boxes.yaml")


def cmd_propose(args):
    from ultralytics import SAM
    dec = yaml.safe_load((FIX_DIR / "label_decisions.yaml").read_text(encoding="utf-8"))
    fixes = dec.get("user_fixes") or {}
    items = {it["stem"]: it for it in fix_items()}
    sam = SAM(str(args.sam))
    polys_out, tiles = {}, []
    for n, (stem, it) in enumerate(items.items()):
        if stem not in fixes:
            print(f"#{n} {stem}: no reviewed boxes, skipped")
            continue
        if fixes[stem] == "exclude":   # no target machine: the build drops the image
            polys_out[stem] = {"split": it["split"], "exclude": True, "instances": []}
            continue
        ip = SRC_DIR / it["split"] / "images" / (stem + ".jpg")
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        boxes_n = fixes[stem]
        boxes = [[b[1] * w, b[2] * h, b[3] * w, b[4] * h] for b in boxes_n]
        res = sam(im, bboxes=boxes, verbose=False)[0].masks
        masks = list(res.data.cpu().numpy() > 0.5) if res is not None else []
        masks += [np.zeros((h, w), bool)] * (len(boxes) - len(masks))   # SAM may return none for a box
        parts = []   # [class name, mask, boxes] per instance
        for b, bx, m in zip(boxes_n, boxes, masks):
            x1, y1, x2, y2 = [int(round(v)) for v in bx]
            clip = np.zeros((h, w), bool)  # policy: nothing outside the reviewed box
            clip[max(0, y1):y2, max(0, x1):x2] = True
            if len(b) > 5 and b[5] == "rect":   # genuine close-up: the box is the label
                m = clip
            else:
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                if len(b) > 5 and b[5] == "fg":   # studio shot where SAM fills the box: non-white pixels
                    m = studio_foreground(im)
                m = keep_main_parts(foreground(m, bx) & clip)
            if b[0].startswith("+") and parts:   # second box of the previous instance
                parts[-1][1] |= m
                parts[-1][2].append(b[1:5])
            else:
                parts.append([b[0].lstrip("+"), m, [b[1:5]]])
        inst = []
        for k, (cname, m, bxs) in enumerate(parts):
            src = [b for b in boxes_n if not b[0].startswith("+")][k]
            if len(src) > 5 and src[5] == "rect" and len(bxs) == 1:
                x1, y1, x2, y2 = bxs[0]
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
                inst.append({"cls": CLASSES.index(cname), "box": bxs, "polygon": poly.tolist(),
                             "area": round((x2 - x1) * (y2 - y1), 4)})
                continue
            p = mask_to_polygon(m, EPS[cname])
            if p is None or len(p) < 3:
                print(f"#{n} {stem}: empty mask for {cname} {bxs}")
                continue
            poly = np.clip(p.astype(np.float64) / [w, h], 0, 1)
            inst.append({"cls": CLASSES.index(cname), "box": bxs, "polygon": poly.round(5).tolist(),
                         "area": round(float(m.mean()), 4)})
        polys_out[stem] = {"split": it["split"], "instances": inst}

        before, s = grid_tile(im, args.size, f"#{n} BEFORE {it['split']} {stem[4:34]}")
        draw_polys(before, current_label(it), s, w, h)
        after, _ = grid_tile(im, args.size, f"#{n} AFTER  SAM2.1-L, {len(inst)} inst")
        draw_polys(after, [(i["cls"], np.array(i["polygon"])) for i in inst], s, w, h)
        for b in boxes_n:
            x1, y1, x2, y2 = (np.array(b[1:5]) * [w * s, h * s, w * s, h * s]).astype(int)
            cv2.rectangle(after, (x1, y1), (x2, y2), (255, 255, 255), 1)
        tiles.append(np.hstack([before, np.full((before.shape[0], 6, 3), 0, np.uint8), after]))
    (FIX_DIR / "user_fix_polygons.json").write_text(json.dumps(polys_out, indent=1), encoding="utf-8")
    write_sheets(tiles, OUT_DIR, cols=2, rows=2)
    print(f"{len(polys_out)} images, {sum(len(v['instances']) for v in polys_out.values())} instances "
          f"-> user_fix_polygons.json; before|after sheets in {OUT_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["review", "propose"])
    ap.add_argument("--sam", type=Path, default=SAM_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--size", type=int, default=600)
    args = ap.parse_args()
    {"review": cmd_review, "propose": cmd_propose}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
