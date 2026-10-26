"""
scripts/tools/audit_negatives.py
================================
Screen the empty-label (negative) images of the 9-class build for unlabelled target machines.

An empty label file tells YOLO "nothing of the 9 classes is here". If a negative actually shows a
fire extinguisher or a saw in the background, every training step on it pushes that class towards
background. This tool makes those images easy to find:

1. reads datasets/9class/negatives_9class.txt (the negatives the build sampled, all splits)
2. records the 38-class source labels of each image (what the image was collected for)
3. runs the given checkpoints at low confidence and records every target-class detection
4. writes contact sheets (detections drawn, index + source classes printed) for visual review
5. writes negative_audit.json with everything above

Decisions after review go into datasets/9class/label_fixes/label_decisions.yaml (section `negatives`);
build_clean_9class_dataset.py applies them.

Usage
-----
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\audit_negatives.py
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\audit_negatives.py --conf 0.1 --weights runs\\run5_9class_clean_finetune\\weights\\best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
SRC_DIR = PROJECT_ROOT / "datasets" / "source_38class"
OUT_DIR = DATASET_DIR / "label_fixes" / "negative_audit"
DEFAULT_WEIGHTS = [PROJECT_ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt",
                   PROJECT_ROOT / "runs" / "run6_9class_cctv_robust" / "weights" / "best.pt"]
SHORT = ["PS", "HP", "TS", "PL", "JO", "BS", "MS", "EB", "FE"]
COLORS = [(0, 0, 255), (0, 160, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 0, 160)]


def load_negatives():
    out = []
    for line in (DATASET_DIR / "negatives_9class.txt").read_text().split():
        p = (DATASET_DIR / line).resolve()
        split = Path(line).parts[0].replace("_9class", "")
        out.append({"split": split, "stem": p.stem, "path": str(p)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", default=[str(w) for w in DEFAULT_WEIGHTS])
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--per-sheet", type=int, default=20)
    args = ap.parse_args()

    from ultralytics import YOLO
    src_names = yaml.safe_load((SRC_DIR / "data.yaml").read_text())["names"]
    negs = load_negatives()
    print(f"{len(negs)} negatives")
    for n in negs:
        lbl = SRC_DIR / n["split"] / "labels" / (n["stem"] + ".txt")
        ids = sorted({int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}) if lbl.exists() else []
        n["source_classes"] = [src_names[i] for i in ids]
        n["detections"] = []

    for w in args.weights:
        model = YOLO(w)
        tag = Path(w).parents[1].name.split("_")[0]
        for k in range(0, len(negs), 16):
            chunk = negs[k:k + 16]
            for n, r in zip(chunk, model.predict([n["path"] for n in chunk], imgsz=args.imgsz, conf=args.conf,
                                                 verbose=False)):
                for b in r.boxes:
                    n["detections"].append({"model": tag, "cls": int(b.cls), "conf": round(float(b.conf), 3),
                                            "box": [round(float(v), 4) for v in b.xyxyn[0].tolist()]})
        print(f"{tag}: done")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(negs):
        n["idx"] = i
        n["max_conf"] = max([d["conf"] for d in n["detections"]], default=0.0)
    (OUT_DIR / "negative_audit.json").write_text(json.dumps(negs, indent=1))

    T = 360
    cols = 5
    for s in range(0, len(negs), args.per_sheet):
        group = negs[s:s + args.per_sheet]
        rows = (len(group) + cols - 1) // cols
        sheet = np.full((rows * (T + 34), cols * T, 3), 255, np.uint8)
        for j, n in enumerate(group):
            im = cv2.imread(n["path"])
            h, w = im.shape[:2]
            sc = T / max(h, w)
            im = cv2.resize(im, (int(w * sc), int(h * sc)))
            for d in n["detections"]:
                x1, y1, x2, y2 = (np.array(d["box"]) * ([im.shape[1], im.shape[0]] * 2)).astype(int)
                c = COLORS[d["cls"]]
                cv2.rectangle(im, (x1, y1), (x2, y2), c, 2 if d["model"] == "run5" else 1)
                cv2.putText(im, f"{d['model'][-1]}{SHORT[d['cls']]}{d['conf']:.2f}", (x1 + 2, max(10, y1 + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            r, c = divmod(j, cols)
            y0, x0 = r * (T + 34), c * T
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(sheet, f"#{n['idx']} {n['split']} {','.join(n['source_classes'])[:34]}", (x0 + 3, y0 + T + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            cv2.putText(sheet, n["stem"][:46], (x0 + 3, y0 + T + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90, 90, 90), 1)
        cv2.imwrite(str(OUT_DIR / f"sheet_{s // args.per_sheet:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    flagged = [n for n in negs if n["max_conf"] >= 0.25]
    print(f"negatives with any target detection >= {args.conf}: {sum(1 for n in negs if n['detections'])}")
    print(f"negatives with a detection >= 0.25: {len(flagged)}")
    by_cls = {}
    for n in negs:
        for d in n["detections"]:
            if d["conf"] >= 0.25:
                by_cls[SHORT[d["cls"]]] = by_cls.get(SHORT[d["cls"]], 0) + 1
    print("detections >= 0.25 by class:", by_cls)
    print(f"sheets + json in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
