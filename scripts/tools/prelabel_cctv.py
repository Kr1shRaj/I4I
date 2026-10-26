"""
Pre-label sampled CCTV frames with the current best checkpoint
==============================================================
Runs the model at a low confidence threshold and writes YOLO-seg polygon labels into a STAGING
folder laid out as an Ultralytics YOLO export, which CVAT ("Ultralytics YOLO Segmentation 1.0")
and Roboflow both import. Annotators then correct boundaries instead of drawing from scratch.

Pre-labels are never written next to the frames (datasets/cctv/frames/.../labels/): that folder is
for REVIEWED labels only, because build_cctv_hybrid_dataset.py trains on whatever is there.

Output
------
    <out>/data.yaml                 class names (same IDs as datasets/9class/data_9class.yaml)
    <out>/train.txt                 image list
    <out>/images/train/<frame>.jpg  hard links to the frames (copies if linking fails)
    <out>/labels/train/<frame>.txt  "<cls> x1 y1 x2 y2 ..." normalised to [0, 1], one instance per line
    <out>/prelabel_summary.csv      per frame: instance count, classes, min/max confidence, source path
                                    (sort by min_conf to review the least certain frames first)

Frames with no detection get an empty .txt; the summary marks them so they are checked for
missed objects, not skipped.

Usage
-----
    python scripts/tools/prelabel_cctv.py --source datasets/cctv/frames/factoryA/cam01/images
    python scripts/tools/prelabel_cctv.py --source datasets/cctv/frames/factoryA --imgsz 640 --conf 0.15
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "datasets" / "cctv" / "prelabels"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def simplify(poly_px: np.ndarray, eps_frac: float) -> np.ndarray:
    """Reduce vertex count (easier to edit in CVAT) while keeping the shape: epsilon = eps_frac * perimeter."""
    c = poly_px.reshape(-1, 1, 2).astype(np.float32)
    eps = eps_frac * cv2.arcLength(c, True)
    return cv2.approxPolyDP(c, eps, True).reshape(-1, 2) if eps > 0 else poly_px


def mask_to_polygon(mask: np.ndarray, eps_frac: float, min_frag: float = 0.10) -> np.ndarray | None:
    """One YOLO polygon (pixel coords) per instance mask.

    Ultralytics' masks.xy joins ALL fragments with bridges; after simplification those bridges become
    spikes across the image. Here (plan section 2): drop fragments under `min_frag` of the instance area,
    simplify each fragment on its own, then join the rest with zero-width bridges (merge_multi_segment).
    """
    from ultralytics.data.converter import merge_multi_segment

    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None
    areas = np.array([cv2.contourArea(c) for c in contours])
    keep = [c for c, a in zip(contours, areas) if a >= min_frag * areas.sum() and len(c) >= 3]
    if not keep:
        return None
    polys = [simplify(c.reshape(-1, 2).astype(np.float32), eps_frac) if eps_frac > 0 else c.reshape(-1, 2)
             for c in keep]
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return None
    return np.concatenate(merge_multi_segment(polys)) if len(polys) > 1 else polys[0]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="image folder (searched recursively) or a single image")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"staging folder (default {DEFAULT_OUT_ROOT}/<timestamp>)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference size; default 640 for frames <= 960 px wide (sub-stream), else 1280 (plan section 4.1)")
    ap.add_argument("--simplify", type=float, default=0.002, help="polygon simplification, fraction of perimeter (0 = off)")
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.weights.exists():
        ap.error(f"weights not found: {args.weights}")
    if args.source.is_dir():
        frames = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in IMG_EXTS and "_staging" not in p.parts)
    elif args.source.is_file():
        frames = [args.source]
    else:
        ap.error(f"source not found: {args.source}")
    if not frames:
        ap.error(f"no images in {args.source}")
    names_by_stem: dict[str, Path] = {}
    for p in frames:
        if p.stem in names_by_stem:
            ap.error(f"duplicate frame name {p.name} in {names_by_stem[p.stem].parent} and {p.parent}; "
                     "frame names must be unique (use sample_cctv_video.py naming)")
        names_by_stem[p.stem] = p

    out = args.out or DEFAULT_OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    img_out, lbl_out = out / "images" / "train", out / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    if model.task != "segment":
        ap.error(f"{args.weights} is a '{model.task}' model; a segmentation checkpoint is required")
    names = model.names

    summary, n_inst, n_empty = [], 0, 0
    for p in frames:
        im = cv2.imread(str(p))
        if im is None:
            print(f"  [WARN] unreadable image skipped: {p}")
            continue
        h, w = im.shape[:2]
        imgsz = args.imgsz or (640 if w <= 960 else 1280)
        r = model.predict(im, conf=args.conf, imgsz=imgsz, retina_masks=True, verbose=False, device=args.device)[0]

        lines, confs, classes = [], [], []
        if r.masks is not None and len(r.masks):
            masks = r.masks.data.cpu().numpy()  # retina_masks=True -> original image size
            for m, cls, cf in zip(masks, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                poly = mask_to_polygon(m > 0.5, args.simplify)
                if poly is None or len(poly) < args.min_points:
                    continue
                norm = np.clip(np.asarray(poly, dtype=np.float64) / [w, h], 0.0, 1.0)
                lines.append(f"{int(cls)} " + " ".join(f"{v:.6f}" for v in norm.flatten()))
                confs.append(cf); classes.append(names[int(cls)])

        (lbl_out / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        link_or_copy(p, img_out / p.name)
        n_inst += len(lines); n_empty += not lines
        summary.append({"frame": p.name, "instances": len(lines), "classes": ";".join(sorted(set(classes))),
                        "min_conf": round(min(confs), 3) if confs else "", "max_conf": round(max(confs), 3) if confs else "",
                        "imgsz": imgsz, "width": w, "height": h, "source": str(p)})

    (out / "data.yaml").write_text(
        "path: .\ntrain: train.txt\n\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items())),
        encoding="utf-8")
    (out / "train.txt").write_text("".join(f"images/train/{s['frame']}\n" for s in summary), encoding="utf-8")
    with (out / "prelabel_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader(); wr.writerows(summary)

    print(f"Pre-labelled {len(summary)} frames with {args.weights.parent.parent.name} at conf {args.conf}: "
          f"{n_inst} instances, {n_empty} frames with no detection")
    print(f"Staging folder (import into CVAT/Roboflow, then put REVIEWED labels in "
          f"datasets/cctv/frames/<site>/<cam>/labels/): {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
