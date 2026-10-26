"""
scripts/tools/annotate_highangle.py
===================================
Annotate the gate-accepted high-angle scrape (pipeline/dataset/train/<class>/img_*_highangle_*.jpg)
for the 9-class segmentation set. Labels follow what is IN the image, not the search keyword.

Steps (each one writes into datasets/highangle/):
    review   index the images, run run5/run6 at low confidence as hints, write review sheets with a
             10% grid (review/sheet_XX.jpg) so instances can be boxed by eye
    propose  read decisions.yaml (per image: reject, or a list of [class, x1, y1, x2, y2] boxes in
             normalised coordinates), run SAM 2.1 box prompts, apply the extent policy
             (docs/label_policy.md) and write proposals.json + overlay sheets (overlay/sheet_XX.jpg)
    export   copy approved images to datasets/highangle/images, write YOLO-seg labels to
             datasets/highangle/labels and the list datasets/9class/train_highangle_9class.txt

Usage
-----
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\annotate_highangle.py review
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\annotate_highangle.py propose
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\annotate_highangle.py export
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel_cctv import mask_to_polygon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "pipeline" / "dataset" / "train"
OUT_DIR = PROJECT_ROOT / "datasets" / "highangle"
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
SAM_WEIGHTS = PROJECT_ROOT / "models" / "pretrained" / "sam2.1_l.pt"
HINT_WEIGHTS = [PROJECT_ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt",
                PROJECT_ROOT / "runs" / "run6_9class_cctv_robust" / "weights" / "best.pt"]
CLASSES = ["panel_saw", "hydraulic_press", "table_saw", "planer", "jointer",
           "band_saw", "miter_saw", "edge_banding_machine", "fire_extinguisher"]
SHORT = ["PS", "HP", "TS", "PL", "JO", "BS", "MS", "EB", "FE"]
COLORS = [(0, 0, 255), (0, 140, 255), (0, 220, 220), (0, 200, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 0, 140)]
# Polygon simplification and fragment threshold per class (fraction of image diagonal / of largest part)
EPS = {c: 0.002 for c in CLASSES}
EPS["fire_extinguisher"] = 0.001


def index():
    p = OUT_DIR / "index.json"
    if p.exists():
        return json.loads(p.read_text())
    files = sorted(f for f in SRC_DIR.glob("*/img_*_highangle_*.jpg"))
    items = []
    for i, f in enumerate(files):
        h, w = cv2.imread(str(f)).shape[:2]
        items.append({"idx": i, "query_class": f.parent.name, "path": str(f.relative_to(PROJECT_ROOT)),
                      "stem": f.stem, "w": w, "h": h})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=1))
    return items


def grid_tile(im, size, label):
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    th, tw = im.shape[:2]
    for k in range(1, 10):
        x, y = int(tw * k / 10), int(th * k / 10)
        cv2.line(im, (x, 0), (x, th), (255, 255, 255), 1)
        cv2.line(im, (0, y), (tw, y), (255, 255, 255), 1)
        cv2.putText(im, str(k), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(im, str(k), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    tile = np.full((size + 22, size, 3), 255, np.uint8)
    tile[:th, :tw] = im
    cv2.putText(tile, label, (4, size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return tile, s


def write_sheets(tiles, folder, cols=3, rows=2):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("sheet_*.jpg"):
        old.unlink()
    per = cols * rows
    for k in range(0, len(tiles), per):
        group = tiles[k:k + per]
        while len(group) < per:
            group.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(group[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(folder / f"sheet_{k // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def cmd_review(args):
    from ultralytics import YOLO
    items = index()
    hints = {it["idx"]: [] for it in items}
    for w in HINT_WEIGHTS:
        model, tag = YOLO(str(w)), w.parents[1].name.split("_")[0]
        for it in items:
            r = model.predict(str(PROJECT_ROOT / it["path"]), imgsz=960, conf=args.conf, verbose=False)[0]
            for b in r.boxes:
                hints[it["idx"]].append({"model": tag, "cls": int(b.cls), "conf": round(float(b.conf), 3),
                                         "box": [round(v, 3) for v in b.xyxyn[0].tolist()]})
    (OUT_DIR / "hints.json").write_text(json.dumps(hints, indent=1))
    tiles = []
    for it in items:
        im = cv2.imread(str(PROJECT_ROOT / it["path"]))
        tile, s = grid_tile(im, args.size, f"#{it['idx']} q={it['query_class']} {it['w']}x{it['h']}")
        th, tw = int(it["h"] * s), int(it["w"] * s)
        for d in hints[it["idx"]]:
            x1, y1, x2, y2 = (np.array(d["box"]) * [tw, th, tw, th]).astype(int)
            cv2.rectangle(tile, (x1, y1), (x2, y2), COLORS[d["cls"]], 1)
            cv2.putText(tile, f"{d['model'][-1]}{SHORT[d['cls']]}{d['conf']:.2f}", (x1 + 2, y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS[d["cls"]], 1)
        tiles.append(tile)
    write_sheets(tiles, OUT_DIR / "review")
    print(f"{len(items)} images, review sheets in {OUT_DIR / 'review'}")


def foreground(m, box):
    """SAM sometimes returns the background of a near-full-image box; keep the object side."""
    x1, y1, x2, y2 = [int(v) for v in box]
    if (x2 - x1) * (y2 - y1) < 0.8 * m.size:   # a machine filling its own (smaller) box is not background
        return m
    i = 4
    border = np.concatenate([m[y1 + i, x1:x2], m[y2 - i - 1, x1:x2], m[y1:y2, x1 + i], m[y1:y2, x2 - i - 1]])
    if border.size and border.mean() > 0.6:
        inv = np.zeros_like(m)
        inv[y1:y2, x1:x2] = ~m[y1:y2, x1:x2]
        m = inv
    return m


def keep_main_parts(m, min_frac=0.15):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [k + 1 for k, a in enumerate(areas) if a >= min_frac * areas.max()]
    return np.isin(lab, keep)


def cmd_propose(args):
    from ultralytics import SAM
    items = {it["idx"]: it for it in index()}
    dec = yaml.safe_load((OUT_DIR / "decisions.yaml").read_text(encoding="utf-8"))
    sam = SAM(str(args.sam))
    proposals = {}
    tiles = []
    for idx in sorted(dec["images"]):
        d = dec["images"][idx]
        it = items[idx]
        im = cv2.imread(str(PROJECT_ROOT / it["path"]))
        h, w = im.shape[:2]
        tile, s = grid_tile(im, args.size, f"#{idx} {d['status']} q={it['query_class']}")
        inst = []
        if d["status"] == "accept":
            boxes = [[b[1] * w, b[2] * h, b[3] * w, b[4] * h] for b in d["boxes"]]
            masks = sam(im, bboxes=boxes, verbose=False)[0].masks.data.cpu().numpy() > 0.5
            for b, bx, m in zip(d["boxes"], boxes, masks):
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                clip = np.zeros_like(m)  # policy: nothing outside the reviewed box (no room, no stock)
                x1, y1, x2, y2 = [int(round(v)) for v in bx]
                clip[max(0, y1):y2, max(0, x1):x2] = True
                m = keep_main_parts(foreground(m, bx) & clip)
                p = mask_to_polygon(m, EPS[b[0]])
                if p is None or len(p) < 3:
                    print(f"#{idx}: empty mask for {b}")
                    continue
                poly = np.clip(p.astype(np.float64) / [w, h], 0, 1)
                inst.append({"cls": CLASSES.index(b[0]), "box": b[1:], "polygon": poly.round(5).tolist(),
                             "area": float(m.mean())})
                pts = (poly * [w * s, h * s]).astype(np.int32)
                ov = tile.copy()
                cv2.fillPoly(ov, [pts], COLORS[CLASSES.index(b[0])])
                tile = cv2.addWeighted(ov, 0.35, tile, 0.65, 0)
                cv2.polylines(tile, [pts], True, COLORS[CLASSES.index(b[0])], 2)
                cv2.putText(tile, SHORT[CLASSES.index(b[0])], tuple(pts.min(0) + [3, 14]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[CLASSES.index(b[0])], 2)
        proposals[idx] = {"status": d["status"], "note": d.get("note", ""), "instances": inst}
        if d["status"] == "accept":   # overlay sheets are for verifying the labels, rejects have none
            tiles.append(tile)
    (OUT_DIR / "proposals.json").write_text(json.dumps(proposals, indent=1))
    write_sheets(tiles, OUT_DIR / "overlay")
    n_acc = sum(p["status"] == "accept" for p in proposals.values())
    print(f"{len(proposals)} decided, {n_acc} accepted, {sum(len(p['instances']) for p in proposals.values())} instances")


def cmd_export(args):
    items = {it["idx"]: it for it in index()}
    proposals = json.loads((OUT_DIR / "proposals.json").read_text())
    for sub in ("images", "labels"):
        d = OUT_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    lst, counts = [], [0] * len(CLASSES)
    for idx, p in sorted(proposals.items(), key=lambda kv: int(kv[0])):
        if p["status"] != "accept" or not p["instances"]:
            continue
        it = items[int(idx)]
        dst = OUT_DIR / "images" / (it["stem"] + ".jpg")
        shutil.copy2(PROJECT_ROOT / it["path"], dst)
        lines = []
        for ins in p["instances"]:
            counts[ins["cls"]] += 1
            lines.append(f"{ins['cls']} " + " ".join(f"{v:.6f}" for pt in ins["polygon"] for v in pt))
        (OUT_DIR / "labels" / (it["stem"] + ".txt")).write_text("\n".join(lines) + "\n")
        lst.append("./" + Path(Path("..") / "highangle" / "images" / dst.name).as_posix())
    (DATASET_DIR / "train_highangle_9class.txt").write_text("\n".join(lst) + "\n")
    print(f"exported {len(lst)} images -> datasets/highangle, list datasets/9class/train_highangle_9class.txt")
    print("instances per class:", dict(zip(CLASSES, counts)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["review", "propose", "export"])
    ap.add_argument("--sam", type=Path, default=SAM_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--size", type=int, default=600)
    args = ap.parse_args()
    {"review": cmd_review, "propose": cmd_propose, "export": cmd_export}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
