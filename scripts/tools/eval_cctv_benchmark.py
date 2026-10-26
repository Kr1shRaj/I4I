"""Evaluate checkpoints on the frozen factory CCTV benchmark (docs/cctv_benchmark.md).

EVALUATION ONLY. Nothing here trains, tunes thresholds or mines negatives from the benchmark.

Per model it writes, under --out/<tag>/:
  val/            Ultralytics val on all 328 frames (box + mask mAP, per class, speed)
  val_cam03/05/   the same, per camera
  metrics.json    everything below in one file
  fp_crops/       crops of false positives at the operating threshold, for eyeballing

and a matched-prediction analysis at the operating threshold:
  - TP / FP / FN per class and per camera (same class, box IoU >= 0.5)
  - absent-class detections per frame (any of them is a false positive)
  - mask quality on true positives: mask IoU, leakage (share of the predicted mask outside the
    label) and coverage miss (share of the label the mask leaves out)

Usage:
    env\\yolo-env\\Scripts\\python.exe scripts\\tools\\eval_cctv_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "9class" / "data_9class_cctv_benchmark.yaml"
LIST = ROOT / "datasets" / "9class" / "cctv_benchmark.txt"
MODELS = {
    "run8": ROOT / "runs" / "run8_9class_curated_cctv" / "weights" / "best.pt",
    "run5": ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt",
}
NAMES = ["panel_saw", "hydraulic_press", "table_saw", "planer", "jointer",
         "band_saw", "miter_saw", "edge_banding_machine", "fire_extinguisher"]
PRESENT = {0, 1, 7}


def benchmark_images() -> list[Path]:
    base = LIST.parent
    return [(base / line.strip()).resolve() for line in LIST.read_text().splitlines() if line.strip()]


def camera_of(img: Path) -> str:
    return "cam03" if "cam03" in img.name else "cam05"


def label_path(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def load_gt(img: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    lp = label_path(img)
    if not lp.exists():
        return out
    for line in lp.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        pts = (np.array(v[1:], dtype=np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
        out.append((int(v[0]), pts))
    return out


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def write_camera_yaml(out: Path, imgs: list[Path]) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    yamls = {}
    for cam in ("cam03", "cam05"):
        lst = out / f"cctv_benchmark_{cam}.txt"
        lst.write_text("\n".join(str(p) for p in imgs if camera_of(p) == cam) + "\n")
        y = out / f"data_cctv_benchmark_{cam}.yaml"
        y.write_text(
            "# EVALUATION ONLY: per-camera slice of the frozen CCTV benchmark\n"
            f"train: {lst}\nval: {lst}\ntest: {lst}\nnc: 9\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
        yamls[cam] = y
    return yamls


def summarise_val(res) -> dict:
    per_class = {}
    for i, c in enumerate(res.box.ap_class_index):
        per_class[NAMES[int(c)]] = {
            "box_map50": float(res.box.ap50[i]), "box_map50_95": float(res.box.ap[i]),
            "mask_map50": float(res.seg.ap50[i]), "mask_map50_95": float(res.seg.ap[i]),
            "box_p": float(res.box.p[i]), "box_r": float(res.box.r[i]),
            "mask_p": float(res.seg.p[i]), "mask_r": float(res.seg.r[i]),
        }
    return {
        "box_map50": float(res.box.map50), "box_map50_95": float(res.box.map),
        "mask_map50": float(res.seg.map50), "mask_map50_95": float(res.seg.map),
        "per_class": per_class, "speed_ms": {k: float(v) for k, v in res.speed.items()},
    }


def matched_analysis(model: YOLO, imgs: list[Path], conf: float, crops: Path | None) -> dict:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    absent = defaultdict(int)
    frames = defaultdict(int)
    mask_stats = defaultdict(list)
    fp_records = []
    for img in imgs:
        cam = camera_of(img)
        frames[cam] += 1
        r = model.predict(str(img), imgsz=960, conf=conf, device=0, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        gts = load_gt(img, w, h)
        gt_boxes = [(c, (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())) for c, p in gts]
        used = set()
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
        for k in np.argsort(-confs):
            c = int(clss[k])
            best, bj = 0.0, -1
            for j, (gc, gb) in enumerate(gt_boxes):
                if gc == c and j not in used:
                    iou = box_iou(boxes[k], gb)
                    if iou > best:
                        best, bj = iou, j
            if best >= 0.5:
                used.add(bj)
                tp[(cam, c)] += 1
                if masks is not None:
                    gm = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(gm, [gts[bj][1]], 1)
                    gm = gm.astype(bool)
                    pm = masks[k]
                    inter = (pm & gm).sum(); pa = pm.sum(); ga = gm.sum()
                    mask_stats[(cam, c)].append((inter / max(pa + ga - inter, 1),
                                                 1 - inter / max(pa, 1), 1 - inter / max(ga, 1)))
            else:
                fp[(cam, c)] += 1
                if c not in PRESENT:
                    absent[(cam, c)] += 1
                fp_records.append((float(confs[k]), cam, c, best, img, boxes[k].tolist()))
        for j, (gc, _) in enumerate(gt_boxes):
            if j not in used:
                fn[(cam, gc)] += 1

    if crops is not None:
        crops.mkdir(parents=True, exist_ok=True)
        for old in crops.glob("*.jpg"):
            old.unlink()
        for n, (cf, cam, c, iou, img, b) in enumerate(sorted(fp_records, key=lambda t: -t[0])[:24]):
            im = cv2.imread(str(img))
            x1, y1, x2, y2 = map(int, b)
            pad = 60
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
            crop = im[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            cv2.imwrite(str(crops / f"{n:02d}_{cam}_{NAMES[c]}_{cf:.2f}_iou{iou:.2f}_{img.stem}.jpg"), crop)

    cams = sorted(frames)
    out = {"conf": conf, "frames": dict(frames), "per_camera": {}}
    for cam in cams:
        rows = {}
        for c in range(9):
            t, f, n = tp[(cam, c)], fp[(cam, c)], fn[(cam, c)]
            if t + f + n == 0:
                continue
            ms = mask_stats[(cam, c)]
            rows[NAMES[c]] = {
                "tp": t, "fp": f, "fn": n,
                "recall": t / (t + n) if t + n else None,
                "precision": t / (t + f) if t + f else None,
                "fp_per_frame": f / frames[cam],
                "mask_iou": float(np.mean([m[0] for m in ms])) if ms else None,
                "leakage": float(np.mean([m[1] for m in ms])) if ms else None,
                "coverage_miss": float(np.mean([m[2] for m in ms])) if ms else None,
            }
        out["per_camera"][cam] = rows
        out.setdefault("absent_fp_per_frame", {})[cam] = sum(
            v for (cm, _), v in absent.items() if cm == cam) / frames[cam]
    out["fp_examples"] = [
        {"conf": cf, "camera": cam, "class": NAMES[c], "best_iou_same_class": iou, "image": img.name, "box": b}
        for cf, cam, c, iou, img, b in sorted(fp_records, key=lambda t: -t[0])[:40]]
    return out


def latency(model: YOLO, imgs: list[Path], n: int = 100) -> dict:
    frames = [cv2.imread(str(p)) for p in imgs[:n]]
    for f in frames[:10]:
        model.predict(f, imgsz=960, conf=0.25, device=0, verbose=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    stages = defaultdict(float)
    for f in frames:
        r = model.predict(f, imgsz=960, conf=0.25, device=0, verbose=False)[0]
        for k, v in r.speed.items():
            stages[k] += v
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000 / len(frames)
    return {"batch1_wall_ms": wall, "batch1_fps": 1000 / wall,
            "batch1_stage_ms": {k: v / len(frames) for k, v in stages.items()}, "frames_timed": len(frames)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "eval_run8_cctv_benchmark")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--conf", type=float, nargs="+", default=[0.40, 0.25],
                    help="Thresholds for the matched analysis; the first one also gets FP crops")
    args = ap.parse_args()

    imgs = benchmark_images()
    assert len(imgs) == 328, f"expected 328 benchmark frames, found {len(imgs)}"
    cam_yaml = write_camera_yaml(args.out / "_lists", imgs)
    summary = {"gpu": torch.cuda.get_device_name(0), "frames": len(imgs)}

    for tag in args.models:
        model = YOLO(str(MODELS[tag]))
        res = {"weights": str(MODELS[tag]),
               "end2end": bool(getattr(model.model, "end2end", False))}
        common = dict(imgsz=960, batch=8, device=0, split="val", project=str(args.out / tag), exist_ok=True)
        res["all"] = summarise_val(model.val(data=str(DATA), name="val", plots=True, **common))
        for cam, y in cam_yaml.items():
            res[cam] = summarise_val(model.val(data=str(y), name=f"val_{cam}", plots=False, **common))
        res["matched"] = {str(c): matched_analysis(model, imgs, c, args.out / tag / "fp_crops" if i == 0 else None)
                          for i, c in enumerate(args.conf)}
        res["latency"] = latency(model, imgs)
        (args.out / tag / "metrics.json").write_text(json.dumps(res, indent=2))
        summary[tag] = res
        print(f"{tag}: box {res['all']['box_map50']:.3f}/{res['all']['box_map50_95']:.3f}  "
              f"mask {res['all']['mask_map50']:.3f}/{res['all']['mask_map50_95']:.3f}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
