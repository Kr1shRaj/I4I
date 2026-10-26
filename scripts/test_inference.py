"""Test and inference harness for the 9-class YOLO segmentation model.

Runs a trained Ultralytics YOLO-seg checkpoint on a single image, a folder of images, or a video file.
It writes annotated outputs (tight boxes and translucent instance masks), prints a per-frame detection
summary, and saves CSV/JSON/Markdown logs of every detection.

Examples:
    python scripts\\test_inference.py --source "..\\trial_videos\\Screen Recording 2026-09-28 124839.mp4"
    python scripts\\test_inference.py --source path\\to\\images --conf 0.40 --device 0
    python scripts\\test_inference.py --source img.jpg --class-conf edge_banding_machine=0.20,fire_extinguisher=0.30
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "run6_9class_cctv_robust" / "weights" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}

# Fixed BGR colour per class id so the same class always gets the same colour in every output.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
]


# --------------------------------------------------------------------------------------------- data records
@dataclass
class Detection:
    source: str
    frame: int
    timestamp_s: float | None
    det_id: int
    class_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    box_w: float
    box_h: float
    box_area_pct: float
    mask_area_px: int | None
    mask_area_pct: float | None


@dataclass
class FrameStat:
    source: str
    frame: int
    timestamp_s: float | None
    n_detections: int
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    classes: str


@dataclass
class RunLog:
    detections: list[Detection] = field(default_factory=list)
    frames: list[FrameStat] = field(default_factory=list)


# --------------------------------------------------------------------------------------------- arguments
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a YOLO-seg checkpoint on images, folders, or video.")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to the .pt checkpoint")
    p.add_argument("--source", type=Path, required=True, help="Image file, folder of images, or video file")
    p.add_argument("--conf", type=float, default=0.40, help="Global confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (ignored by NMS-free end2end heads)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (long side)")
    p.add_argument("--device", default=None, help="'0' for first GPU or 'cpu' (default: GPU if available)")
    p.add_argument("--save-dir", type=Path, default=None, help="Output directory (default: runs/inference_tests/<time>)")
    p.add_argument(
        "--class-conf",
        default="",
        help="Per-class thresholds, e.g. 'edge_banding_machine=0.20,fire_extinguisher=0.30' (names or ids)",
    )
    p.add_argument("--max-det", type=int, default=50, help="Maximum detections per frame")
    p.add_argument("--max-frames", type=int, default=0, help="Stop video after N processed frames (0 = all)")
    p.add_argument("--vid-stride", type=int, default=1, help="Process every Nth video frame")
    p.add_argument("--mask-alpha", type=float, default=0.40, help="Mask fill opacity in the annotated output")
    p.add_argument("--no-retina", action="store_true", help="Disable full-resolution masks (faster, blockier edges)")
    p.add_argument("--half", action="store_true", help="FP16 inference (CUDA only)")
    p.add_argument(
        "--nms",
        action="store_true",
        help="For end2end (YOLO26) checkpoints: use the one-to-many head + NMS so --iou takes effect",
    )
    p.add_argument("--quiet", action="store_true", help="Do not print per-frame lines")
    return p.parse_args()


def resolve_device(device: str | None) -> str:
    if device is None:
        return "0" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        print(f"[warn] CUDA not available, falling back to CPU (requested device={device})")
        return "cpu"
    return device


def parse_class_conf(spec: str, names: dict[int, str]) -> dict[int, float]:
    """Parse 'name=0.2,3=0.3' into {class_id: threshold}."""
    out: dict[int, float] = {}
    if not spec:
        return out
    by_name = {v: k for k, v in names.items()}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        if key.isdigit() and int(key) in names:
            cid = int(key)
        elif key in by_name:
            cid = by_name[key]
        else:
            raise SystemExit(f"--class-conf: unknown class '{key}'. Valid: {', '.join(names.values())}")
        out[cid] = float(val)
    return out


# --------------------------------------------------------------------------------------------- drawing
def colour(cid: int) -> tuple[int, int, int]:
    return PALETTE[cid % len(PALETTE)]


def draw_label(img: np.ndarray, text: str, x1: int, y1: int, col: tuple[int, int, int]) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, max(0.5, img.shape[0] / 1400), max(1, img.shape[0] // 700)
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y_top = y1 - th - base - 4
    if y_top < 0:  # no room above the box: draw inside it
        y_top = y1
    x1 = min(max(0, x1), max(0, img.shape[1] - tw - 4))
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y_top + th + base + 4), col, -1)
    cv2.putText(img, text, (x1 + 2, y_top + th + 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def annotate(frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray, clss: np.ndarray,
             masks: np.ndarray | None, names: dict[int, str], alpha: float) -> np.ndarray:
    """Draw translucent masks, crisp mask contours, tight boxes, and labels."""
    out = frame.copy()
    thick = max(2, frame.shape[0] // 400)
    if masks is not None:
        for m, c in zip(masks, clss):
            col = np.array(colour(int(c)), dtype=np.float32)
            sel = m.astype(bool)
            out[sel] = (out[sel].astype(np.float32) * (1 - alpha) + col * alpha).astype(np.uint8)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, colour(int(c)), thick, cv2.LINE_AA)
    for (x1, y1, x2, y2), cf, c in zip(boxes.astype(int), confs, clss):
        cv2.rectangle(out, (x1, y1), (x2, y2), colour(int(c)), thick, cv2.LINE_AA)
        draw_label(out, f"{names[int(c)]} {cf:.2f}", x1, y1, colour(int(c)))
    return out


def draw_hud(img: np.ndarray, text: str) -> None:
    """Frame status in the bottom-left corner, clear of the box labels."""
    org = (10, img.shape[0] - 12)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------------------------- core inference
class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(args.device)
        if not args.weights.exists():
            raise SystemExit(f"Weights not found: {args.weights}")
        self.model = YOLO(str(args.weights))
        self.names: dict[int, str] = self.model.names
        self.class_conf = parse_class_conf(args.class_conf, self.names)
        # Predict at the lowest threshold in play, then filter per class.
        self.pred_conf = min([args.conf, *self.class_conf.values()])
        self.end2end = bool(getattr(self.model.model, "end2end", False))
        self.log = RunLog()

    def describe(self) -> None:
        print("=" * 78)
        print(f"weights   : {self.args.weights}")
        print(f"task      : {self.model.task}   classes: {len(self.names)}   end2end(NMS-free): {self.end2end}")
        print(f"device    : {self.device}   imgsz: {self.args.imgsz}   conf: {self.args.conf}   iou: {self.args.iou}")
        if self.class_conf:
            print("class conf: " + ", ".join(f"{self.names[k]}={v}" for k, v in self.class_conf.items()))
        if self.end2end and not self.args.nms:
            print("note      : end2end head - NMS-free one-to-one output, --iou has no effect (use --nms to enable)")
        elif self.end2end:
            print("note      : end2end head overridden (--nms) - one-to-many head + NMS at --iou")
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
              + (f"  gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
        print("=" * 78)

    def warmup(self, frame: np.ndarray, n: int = 2) -> None:
        """Run untimed passes on a real frame so CUDA init and cudnn autotuning stay out of the latency stats."""
        for _ in range(n):
            self.predict(frame)

    def predict(self, frame: np.ndarray):
        return self.model.predict(
            frame,
            imgsz=self.args.imgsz,
            conf=self.pred_conf,
            iou=self.args.iou,
            device=self.device,
            max_det=self.args.max_det,
            retina_masks=not self.args.no_retina,
            quantize=16 if self.args.half else None,
            verbose=False,
            **({"end2end": False} if self.args.nms else {}),
        )[0]

    def process(self, frame: np.ndarray, source: str, frame_idx: int, ts: float | None) -> np.ndarray:
        t0 = time.perf_counter()
        r = self.predict(frame)
        total_ms = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            masks = None
            if r.masks is not None:
                m = r.masks.data
                if tuple(m.shape[1:]) != (h, w):  # --no-retina: masks come at letterboxed size, strip padding
                    m = scale_masks(m[None].float(), (h, w))[0]
                masks = (m > 0.5).cpu().numpy()
            keep = np.array([cf >= self.class_conf.get(int(c), self.args.conf) for cf, c in zip(confs, clss)])
            boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
            if masks is not None:
                masks = masks[keep]
        else:
            boxes, confs, clss, masks = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None

        for i, ((x1, y1, x2, y2), cf, c) in enumerate(zip(boxes, confs, clss)):
            m_px = int(masks[i].sum()) if masks is not None else None
            self.log.detections.append(Detection(
                source=source, frame=frame_idx, timestamp_s=None if ts is None else round(ts, 3), det_id=i,
                class_id=int(c), class_name=self.names[int(c)], conf=round(float(cf), 4),
                x1=round(float(x1), 1), y1=round(float(y1), 1), x2=round(float(x2), 1), y2=round(float(y2), 1),
                box_w=round(float(x2 - x1), 1), box_h=round(float(y2 - y1), 1),
                box_area_pct=round(100 * float((x2 - x1) * (y2 - y1)) / (w * h), 3),
                mask_area_px=m_px, mask_area_pct=None if m_px is None else round(100 * m_px / (w * h), 3),
            ))

        sp = r.speed
        summary = ", ".join(f"{self.names[int(c)]} {cf:.2f}" for c, cf in zip(clss, confs)) or "-"
        self.log.frames.append(FrameStat(
            source=source, frame=frame_idx, timestamp_s=None if ts is None else round(ts, 3),
            n_detections=len(clss), preprocess_ms=round(sp.get("preprocess", 0.0), 2),
            inference_ms=round(sp.get("inference", 0.0), 2), postprocess_ms=round(sp.get("postprocess", 0.0), 2),
            total_ms=round(total_ms, 2), classes=summary,
        ))
        if not self.args.quiet:
            tag = f"{source} f{frame_idx:05d}" + (f" t={ts:7.2f}s" if ts is not None else "")
            print(f"{tag} | {len(clss):2d} det | inf {sp.get('inference', 0):6.1f} ms | "
                  f"total {total_ms:6.1f} ms | {summary}")

        annotated = annotate(frame, boxes, confs, clss, masks, self.names, self.args.mask_alpha)
        draw_hud(annotated, f"{len(clss)} det | {sp.get('inference', 0):.0f} ms")
        return annotated

    # ------------------------------------------------------------------ sources
    def run_images(self, paths: list[Path], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        warmed = False
        for i, p in enumerate(paths):
            frame = cv2.imread(str(p))
            if frame is None:
                print(f"[warn] could not read {p}")
                continue
            if not warmed:
                self.warmup(frame)
                warmed = True
            annotated = self.process(frame, p.name, i, None)
            cv2.imwrite(str(out_dir / f"{p.stem}_pred.jpg"), annotated)

    def run_video(self, path: Path, out_dir: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, self.args.vid_stride)
        out_path = out_dir / f"{path.stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
        print(f"video     : {path.name}  {w}x{h} @ {fps:.2f} fps, {n_total} frames, stride {stride}")

        idx = processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx == 0:
                self.warmup(frame)
            if idx % stride == 0:
                writer.write(self.process(frame, path.name, idx, idx / fps))
                processed += 1
                if self.args.max_frames and processed >= self.args.max_frames:
                    break
            idx += 1
        cap.release()
        writer.release()
        print(f"saved     : {out_path}  ({processed} frames)")


# --------------------------------------------------------------------------------------------- reporting
def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def write_reports(runner: Runner, save_dir: Path, wall_s: float) -> None:
    log, names = runner.log, runner.names
    det_rows = [asdict(d) for d in log.detections]
    frame_rows = [asdict(f) for f in log.frames]

    for fname, rows, cls in (("detections.csv", det_rows, Detection), ("frames.csv", frame_rows, FrameStat)):
        with open(save_dir / fname, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(cls.__dataclass_fields__))
            wr.writeheader()
            wr.writerows(rows)

    per_class: dict[str, list[float]] = defaultdict(list)
    frames_with: dict[str, set] = defaultdict(set)
    for d in log.detections:
        per_class[d.class_name].append(d.conf)
        frames_with[d.class_name].add((d.source, d.frame))

    inf = [f.inference_ms for f in log.frames]
    tot = [f.total_ms for f in log.frames]
    n_frames = len(log.frames)
    summary = {
        "weights": str(runner.args.weights),
        "source": str(runner.args.source),
        "device": runner.device,
        "imgsz": runner.args.imgsz,
        "conf": runner.args.conf,
        "iou": runner.args.iou,
        "class_conf": {names[k]: v for k, v in runner.class_conf.items()},
        "end2end": runner.end2end,
        "frames_processed": n_frames,
        "frames_with_detections": sum(1 for f in log.frames if f.n_detections),
        "total_detections": len(log.detections),
        "wall_time_s": round(wall_s, 2),
        "latency_ms": {
            "inference_mean": round(statistics.fmean(inf), 2) if inf else 0,
            "inference_p50": round(pct(inf, 50), 2),
            "inference_p95": round(pct(inf, 95), 2),
            "end_to_end_mean": round(statistics.fmean(tot), 2) if tot else 0,
            "end_to_end_p95": round(pct(tot, 95), 2),
            "throughput_fps": round(1000 / statistics.fmean(tot), 2) if tot else 0,
        },
        "per_class": {
            name: {
                "detections": len(per_class[name]),
                "frames": len(frames_with[name]),
                "frame_pct": round(100 * len(frames_with[name]) / n_frames, 1) if n_frames else 0,
                "conf_mean": round(statistics.fmean(per_class[name]), 3),
                "conf_min": round(min(per_class[name]), 3),
                "conf_max": round(max(per_class[name]), 3),
            }
            for name in names.values()
            if per_class[name]
        },
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Inference summary — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Weights: `{summary['weights']}`",
        f"- Source: `{summary['source']}`",
        f"- Device {summary['device']}, imgsz {summary['imgsz']}, conf {summary['conf']}, iou {summary['iou']}"
        + (f", per-class {summary['class_conf']}" if summary["class_conf"] else "")
        + (" (end2end head: NMS-free, iou unused)" if summary["end2end"] else ""),
        f"- Frames: {n_frames} processed, {summary['frames_with_detections']} with detections, "
        f"{summary['total_detections']} detections total",
        f"- Latency: inference mean {summary['latency_ms']['inference_mean']} ms "
        f"(p95 {summary['latency_ms']['inference_p95']} ms), end-to-end mean "
        f"{summary['latency_ms']['end_to_end_mean']} ms → {summary['latency_ms']['throughput_fps']} FPS",
        "",
        "| Class | Detections | Frames | % frames | Conf mean | Conf min | Conf max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in sorted(summary["per_class"].items(), key=lambda kv: -kv[1]["detections"]):
        lines.append(f"| {name} | {s['detections']} | {s['frames']} | {s['frame_pct']} | "
                     f"{s['conf_mean']:.3f} | {s['conf_min']:.3f} | {s['conf_max']:.3f} |")
    if not summary["per_class"]:
        lines.append("| (no detections) | 0 | 0 | 0 | - | - | - |")
    (save_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines[5:]))
    print(f"\nlogs      : {save_dir / 'detections.csv'}, frames.csv, summary.json, summary.md")


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    src: Path = args.source
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    save_dir = args.save_dir or PROJECT_ROOT / "runs" / "inference_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = Runner(args)
    runner.describe()

    t0 = time.perf_counter()
    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise SystemExit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) in {src}")
        print(f"images    : {len(paths)} in {src}")
        runner.run_images(paths, save_dir / "images")
    elif src.suffix.lower() in IMAGE_EXTS:
        runner.run_images([src], save_dir / "images")
    elif src.suffix.lower() in VIDEO_EXTS:
        runner.run_video(src, save_dir)
    else:
        raise SystemExit(f"Unsupported source type: {src.suffix}")

    write_reports(runner, save_dir, time.perf_counter() - t0)


if __name__ == "__main__":
    sys.exit(main())
