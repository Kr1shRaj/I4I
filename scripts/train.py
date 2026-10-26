"""
9-Class Instance Segmentation: run8 curated fine-tune
=====================================================
- Base: run5_9class_clean_finetune (the generalist champion; run6 and run7 are not bases).
- Data: datasets/9class/data_9class.yaml (cleaned edge-bander / extinguisher labels, 12% negatives incl.
  107 targeted look-alike hard negatives). The factory CCTV frames are an EVALUATION-ONLY benchmark
  (docs/cctv_benchmark.md); never pass data_9class_cctv_benchmark.yaml here.
- Explicit AdamW (optimizer=auto silently replaces lr0), lr0 5e-4, lrf 0.01, cosine, imgsz 960 (run5's size).
- cctv_aug: milder codec/sensor degradation than run7 + random lens distortion (mostly barrel).
  Ultralytics silently skips Albumentations when the package is missing, so this script refuses to start
  without it; check the training log for the 'albumentations:' line.

Usage
-----
    env\\yolo-env\\Scripts\\python.exe scripts\\train.py
    # smoke test: 1 epoch on 2% of the list, separate output folder
    env\\yolo-env\\Scripts\\python.exe scripts\\train.py --epochs 1 --fraction 0.02 --name smoke_run8 --project <tmp>
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"


def cctv_aug():
    """Codec / sensor degradation + lens distortion (Albumentations 2.x argument names).

    Albumentations 2.x silently IGNORES 1.x names (quality_lower, scale_min, var_limit, shift_limit) and falls
    back to defaults, so only 2.x names are used here:
      ImageCompression(quality_lower=30, quality_upper=75) -> quality_range=(30, 75)
      Downscale(scale_min=0.4, scale_max=0.75)             -> scale_range=(0.4, 0.75)
      GaussNoise(var_limit=(10, 50)) on 0-255 pixels       -> std_range=(0.012, 0.028)  (sqrt(var)/255)
    OpticalDistortion: in 2.x NEGATIVE distort_limit is PINCUSHION and POSITIVE is BARREL (verified on a grid).
    CCTV wide-angle lenses are barrel, so (-0.05, 0.25) = mostly barrel. Polygons are warped with the image
    (Ultralytics 8.4 passes segment vertices as keypoints; verified on train batches).
    """
    try:
        import albumentations as A
        import cv2
    except ImportError:
        raise SystemExit("albumentations is not installed; Ultralytics would silently skip cctv_aug. Run:\n"
                         "  env\\yolo-env\\Scripts\\python.exe -m pip install --no-deps albumentations albucore\n"
                         "  env\\yolo-env\\Scripts\\python.exe -m pip install pydantic simsimd stringzilla")
    return [
        A.OpticalDistortion(distort_limit=(-0.05, 0.25), interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_REFLECT_101, p=0.35),   # wide-angle CCTV lens curvature
        A.ImageCompression(quality_range=(30, 75), p=0.4),                  # H.264-like blocking
        A.Downscale(scale_range=(0.4, 0.75), p=0.3),                        # low-res sensor / sub-stream
        A.GaussNoise(std_range=(0.012, 0.028), p=0.3),                      # sensor noise
        A.OneOf([A.MotionBlur(blur_limit=(3, 7)), A.GaussianBlur(blur_limit=(3, 5))], p=0.15),
        A.ToGray(p=0.05),                                                   # keep colour cues (red cylinders vs drums)
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
    ]


def main():
    ap = argparse.ArgumentParser(description="CCTV pilot fine-tune (plan Option A)")
    ap.add_argument("--weights", type=Path,
                    default=PROJECT_ROOT / "runs" / "run5_9class_clean_finetune" / "weights" / "best.pt")
    ap.add_argument("--data", type=Path, default=DATASET_DIR / "data_9class.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=960, help="run5's training size; best on catalog val")
    ap.add_argument("--batch", type=int, default=6, help="fits 8 GB at 960")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction of the list (smoke tests)")
    ap.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    ap.add_argument("--name", default="run8_9class_curated_cctv")
    ap.add_argument("--resume", action="store_true", help="resume training from runs/<name>/weights/last.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        last_pt = args.project / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            raise SystemExit(f"Cannot resume: {last_pt} not found")
        augs = cctv_aug()
        print(f"Resuming training from {last_pt} with cctv_aug (workers=0, cache=ram)...")
        model = YOLO(str(last_pt))
        model.train(resume=True, workers=0, augmentations=augs)
        return

    for p in (args.weights, args.data):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    if "benchmark" in args.data.name or "cctv_factory_intake" in args.data.read_text(encoding="utf-8"):
        raise SystemExit(f"{args.data} uses the frozen CCTV benchmark (docs/cctv_benchmark.md); never train on it")
    if (args.project / args.name).exists():
        raise SystemExit(f"{args.project / args.name} already exists; pick a new --name (runs are never overwritten)")

    augs = cctv_aug()

    print(f"Loading base model: {args.weights}")
    model = YOLO(str(args.weights))
    print(f"Training on {args.data} -> {args.project / args.name}")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=0,
        workers=0,
        cache="ram",
        # Optimizer: explicit, or 'auto' overrides lr0
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2,
        # Geometry: turned down (plan 3.2/3.3); roll and strong keystone don't model CCTV pitch
        mosaic=0.5,
        close_mosaic=10,
        scale=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0002,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.5,
        copy_paste=0.3,
        copy_paste_mode="flip",
        mixup=0.0,
        augmentations=augs,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
