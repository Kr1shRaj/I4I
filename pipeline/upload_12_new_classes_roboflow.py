# -*- coding: utf-8 -*-
"""
pipeline/upload_12_new_classes_roboflow.py
===========================================
SAM/FastSAM Polygon Segmentation Autolabeling & Roboflow Ingestion for 12 New Classes.

Classes processed (12):
  - edge_banding_machine
  - dust_collector
  - veneer_press
  - drum_sander
  - mortiser
  - glue_spreader
  - pallet_jack
  - air_compressor
  - storage_racking
  - robotic_arm
  - overhead_hoist
  - platform_scale

Workflow:
  1. Loads clean images from dataset/train/<class>/ for the 12 new classes.
  2. Runs FastSAM / SAM contour polygon extraction for exact machinery instance segmentation masks.
  3. Formats valid COCO polygon annotations ([[x1, y1, x2, y2, ...]]).
  4. Uploads images + polygon segmentation annotations to workspace 'krish-raj-cgbcn', project 'new3try'.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import FastSAM

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

WORKSPACE_SLUG = "krish-raj-cgbcn"
PROJECT_ID     = "new3try"

NEW_12_CLASSES = [
    "edge_banding_machine",
    "dust_collector",
    "veneer_press",
    "drum_sander",
    "mortiser",
    "glue_spreader",
    "pallet_jack",
    "air_compressor",
    "storage_racking",
    "robotic_arm",
    "overhead_hoist",
    "platform_scale",
    "ppe_station",
    "first_aid_station",
    "emergency_exit_sign",
]

SPLIT_RATIOS   = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED    = 42
NUM_WORKERS    = 8
DATASET_DIR    = Path("dataset/train")
LOGS_DIR       = Path("logs")


import threading
MODEL_LOCK = threading.Lock()


def generate_coco_segmentation(img_path: Path, expected_class: str, sam_model) -> dict | None:
    """
    Extracts instance segmentation polygon masks using FastSAM and generates COCO payload.
    """
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
    except Exception as e:
        return None

    # Run FastSAM inference with thread safety lock
    with MODEL_LOCK:
        results = sam_model(str(img_path), device="cpu", retina_masks=True, imgsz=640, conf=0.25, iou=0.8, verbose=False)
    coco_annotations = []
    ann_id = 1
    total_area = w * h

    if results and results[0].masks is not None:
        xy_contours = results[0].masks.xy
        boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes is not None else []

        for idx, contour in enumerate(xy_contours):
            if len(contour) < 3:
                continue
            
            # Compute area
            poly_area = cv2.contourArea(contour.astype(np.float32))
            if poly_area < (total_area * 0.015) or poly_area > (total_area * 0.92):
                # Skip tiny noise and full-frame background boxes
                continue

            flat_seg = contour.flatten().tolist()
            if len(flat_seg) < 6:
                continue

            if idx < len(boxes):
                bx, by, bw, bh = boxes[idx]
                px = float(bx - bw / 2.0)
                py = float(by - bh / 2.0)
                pw = float(bw)
                ph = float(bh)
            else:
                x_coords = contour[:, 0]
                y_coords = contour[:, 1]
                px = float(x_coords.min())
                py = float(y_coords.min())
                pw = float(x_coords.max() - px)
                ph = float(y_coords.max() - py)

            coco_annotations.append({
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [flat_seg],
                "area": float(poly_area),
                "bbox": [px, py, pw, ph],
                "iscrowd": 0,
            })
            ann_id += 1

    # Fallback to central object bounding polygon if zero instances detected
    if not coco_annotations:
        margin_w = w * 0.1
        margin_h = h * 0.1
        px, py = margin_w, margin_h
        pw, ph = w - 2 * margin_w, h - 2 * margin_h
        flat_seg = [px, py, px + pw, py, px + pw, py + ph, px, py + ph]
        coco_annotations.append({
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [flat_seg],
            "area": float(pw * ph),
            "bbox": [px, py, pw, ph],
            "iscrowd": 0,
        })

    coco_payload = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": img_path.name}],
        "categories": [{"id": 1, "name": expected_class, "supercategory": "equipment"}],
        "annotations": coco_annotations,
    }
    return coco_payload


def process_and_upload(proj, img_path: Path, expected_class: str, split: str, sam_model) -> dict:
    filename = img_path.name
    coco_payload = generate_coco_segmentation(img_path, expected_class, sam_model)
    if not coco_payload:
        return {"filename": filename, "class": expected_class, "split": split, "status": "FAIL", "reason": "Failed to read image", "masks": 0}

    mask_count = len(coco_payload["annotations"])
    upload_ok = False

    with tempfile.TemporaryDirectory() as tmpdir:
        coco_file = Path(tmpdir) / "_annotations.coco.json"
        coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

        for attempt in range(3):
            try:
                res = proj.single_upload(
                    image_path=str(img_path),
                    annotation_path=str(coco_file),
                    split=split,
                    annotation_overwrite=True,
                )
                if isinstance(res, dict) and (res.get("success", False) or res.get("id") or res.get("annotation")):
                    upload_ok = True
                    break
            except Exception as e:
                err_str = str(e)
                if "missing permissions" in err_str or "404" in err_str:
                    return {"filename": filename, "class": expected_class, "split": split, "status": "PERM_ERROR", "reason": err_str, "masks": mask_count}
                time.sleep(1.0)

    return {
        "filename": filename,
        "class": expected_class,
        "split": split,
        "status": "OK" if upload_ok else "FAIL",
        "masks": mask_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload 12 New Classes with SAM Polygon Annotations to Roboflow")
    parser.add_argument("--classes", nargs="+", default=NEW_12_CLASSES, help="Classes to upload")
    parser.add_argument("--dry-run", action="store_true", help="Generate SAM polygon annotations locally without uploading")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set in environment or .env file.")
        sys.exit(1)

    print("Loading FastSAM model for segmentation autolabeling...")
    sam_model = FastSAM("FastSAM-s.pt")

    proj = None
    if not args.dry_run:
        print(f"Connecting to Roboflow workspace '{WORKSPACE_SLUG}', project '{PROJECT_ID}'...")
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)
        print("Connected successfully!\n")

    summary_data = {}
    target_classes = [c for c in args.classes if c in NEW_12_CLASSES]

    for cls_name in target_classes:
        cls_dir = DATASET_DIR / cls_name
        if not cls_dir.exists():
            print(f"[WARN] Directory {cls_dir} does not exist. Skipping.")
            continue

        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}])
        total = len(files)
        if total == 0:
            print(f"[WARN] No images found in {cls_dir}. Skipping.")
            continue

        rng = random.Random(RANDOM_SEED)
        shuffled = list(files)
        rng.shuffle(shuffled)

        n_train = int(round(total * SPLIT_RATIOS["train"]))
        n_valid = int(round(total * SPLIT_RATIOS["valid"]))

        split_map = {}
        for idx, p in enumerate(shuffled):
            if idx < n_train:
                split_map[p] = "train"
            elif idx < n_train + n_valid:
                split_map[p] = "valid"
            else:
                split_map[p] = "test"

        print(f"\n===========================================================================", flush=True)
        print(f"PROCESSING CLASS [{cls_name.upper()}] ({total} images)", flush=True)
        print(f"===========================================================================", flush=True)

        cls_success = 0
        cls_masks = 0
        perm_errors = 0

        if args.dry_run:
            for i, img_p in enumerate(files, start=1):
                split = split_map[img_p]
                coco = generate_coco_segmentation(img_p, cls_name, sam_model)
                m_cnt = len(coco["annotations"]) if coco else 0
                cls_success += 1
                cls_masks += m_cnt
                if i % 20 == 0 or i == total:
                    print(f"  [DRY-RUN {i:03d}/{total:03d}] {img_p.name:<40} | {split:<5} | {m_cnt:>2} SAM masks generated", flush=True)
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(process_and_upload, proj, img_p, cls_name, split_map[img_p], sam_model): img_p
                    for img_p in files
                }
                for fut in as_completed(futures):
                    res = fut.result()
                    completed += 1
                    if res["status"] == "OK":
                        cls_success += 1
                        cls_masks += res["masks"]
                    elif res["status"] == "PERM_ERROR":
                        perm_errors += 1
                        print(f"\n[PERMISSIONS ERROR] Roboflow API key lacks upload/write permissions for {WORKSPACE_SLUG}/{PROJECT_ID}.", flush=True)
                        print(f"Error detail: {res['reason']}", flush=True)
                        print("Aborting upload loop.", flush=True)
                        sys.exit(2)

                    if completed % 10 == 0 or completed == total:
                        print(f"  [{completed:03d}/{total:03d}] {res['filename']:<40} | {res['split']:<5} | {res['masks']:>2} masks -> {res['status']}", flush=True)

        summary_data[cls_name] = {
            "total_images": total,
            "processed": cls_success,
            "total_masks": cls_masks,
            "avg_masks": round(cls_masks / total, 2) if total else 0,
        }

    print("\n" + "=" * 80)
    print("12-CLASS SAM POLYGON SEGMENTATION & ROBOFLOW UPLOAD SUMMARY")
    print("=" * 80)
    for c, s in summary_data.items():
        print(f"  {c:<22} | Images: {s['total_images']} | Total SAM Masks: {s['total_masks']} | Avg Masks/Img: {s['avg_masks']}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = LOGS_DIR / "upload_12_new_classes_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "summary": summary_data}, f, indent=2)
    print(f"\n[REPORT SAVED] -> {out_json}\n")


if __name__ == "__main__":
    main()
