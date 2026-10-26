"""
Sample diverse, de-duplicated frames from CCTV recordings or a live RTSP stream
==============================================================================
Implements docs/cctv_domain_adaptation_plan.md section 1.4:

1. Coarse time stride   -- decode 1 frame every --stride seconds (default 3 s).
2. Change-based dedup   -- 64-bit dHash; a frame is kept only if its Hamming distance to EVERY frame
                           already kept for this camera (this run and earlier runs) is >= --min-hamming.
3. Per-camera cap       -- at most --cap frames per run, spread evenly over the recording time
                           (not just the first N), so a 24 h file covers all shifts.

Output (one folder per camera; Ultralytics finds labels by swapping /images/ -> /labels/):

    datasets/cctv/frames/<site>/<cam_id>/images/cctv_<cam_id>_f<frame_idx>_<YYYYmmdd-HHMMSS-mmm>.jpg
    datasets/cctv/frames/<site>/<cam_id>/frames.csv      (one row per kept frame; appended per run)

The timestamp is wall-clock time: --start-time, else parsed from the file name (ffmpeg segment names
like cam01_20261015_143000.mp4, or "Screen Recording 2026-09-28 124839.mp4"), else the file's
modification time minus its duration. Live streams use the system clock. Existing files are never
overwritten.

Usage
-----
    # one recording (or a folder of segments) from camera cam01 at site factoryA
    python scripts/tools/sample_cctv_video.py --source D:/cctv_raw/factoryA/cam01 --site factoryA --cam-id cam01

    # live RTSP, 20 minutes, 1 frame / 2 s
    python scripts/tools/sample_cctv_video.py --source "rtsp://user:pass@10.0.0.5:554/Streaming/Channels/101" \
        --site factoryA --cam-id cam03 --stride 2 --duration 1200
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "cctv" / "frames"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".dav", ".h264", ".h265", ".m4v"}
ID_RE = re.compile(r"^[A-Za-z0-9-]+$")  # no '_' so the filename fields stay parseable
CSV_FIELDS = ["file", "site", "cam_id", "source", "frame_idx", "pts_sec", "timestamp", "dhash"]


def dhash(frame: np.ndarray, n: int = 8) -> int:
    """64-bit difference hash of a BGR frame (plan section 1.4)."""
    g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def ssim_thumb(frame: np.ndarray) -> np.ndarray:
    """160 px greyscale thumbnail for SSIM (plan section 1.4)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (160, max(1, round(160 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.float64)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (Gaussian window 7, sigma 1.5)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    s_a, s_b, s_ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))).mean())


def is_stream(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def redact(source: str) -> str:
    """Strip credentials from stream URLs before they are logged or written to CSV."""
    return re.sub(r"//[^/@]+@", "//***@", source)


def parse_start_time(path: Path) -> datetime | None:
    """Recording start time from common NVR / ffmpeg-segment / screen-recorder file names."""
    for pat, fmt in ((r"(\d{8}_\d{6})", "%Y%m%d_%H%M%S"),
                     (r"(\d{4}-\d{2}-\d{2} \d{6})", "%Y-%m-%d %H%M%S"),
                     (r"(\d{8}T\d{6})", "%Y%m%dT%H%M%S"),
                     (r"(\d{14})", "%Y%m%d%H%M%S")):
        m = re.search(pat, path.stem)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                pass
    return None


NAME_FORMAT = "timestamp"  # set from --name-format in main()


def frame_name(cam_id: str, frame_idx: int, ts: datetime) -> str:
    if NAME_FORMAT == "plain":
        return f"cctv_{cam_id}_f{frame_idx:07d}.jpg"
    return f"cctv_{cam_id}_f{frame_idx:07d}_{ts.strftime('%Y%m%d-%H%M%S')}-{ts.microsecond // 1000:03d}.jpg"


def load_existing(cam_dir: Path) -> tuple[list[int], int]:
    """dHashes of frames already kept for this camera, so repeat runs don't re-add the same view."""
    hashes, n = [], 0
    csv_path = cam_dir / "frames.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (cam_dir / "images" / row["file"]).exists():
                    hashes.append(int(row["dhash"], 16)); n += 1
    else:
        for p in sorted((cam_dir / "images").glob("*.jpg")):
            im = cv2.imread(str(p))
            if im is not None:
                hashes.append(dhash(im)); n += 1
    return hashes, n


def scan(source: str, start: datetime | None, stride: float, min_ham: int, kept_hashes: list[int],
         staging: Path, cam_id: str, duration: float | None, jpeg_q: int,
         resize: tuple[int, int] | None = None, method: str = "dhash", ssim_th: float = 0.92) -> list[dict]:
    """Decode `source`, apply stride + dedup, write candidates to `staging`. Returns candidate rows.

    method 'dhash': keep if Hamming >= min_ham to every kept frame of the camera.
    method 'ssim':  keep if SSIM to the last kept frame < ssim_th AND Hamming >= min_ham to every kept frame
                    (use a small min_ham, e.g. 3, as an exact-repeat guard). Better for fixed cameras, where a
                    whole-frame 64-bit dHash hardly changes when workers or stock move.
    """
    last_thumb = None
    live = is_stream(source)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"  [ERROR] cannot open {redact(source)}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not live and (not fps or fps <= 0 or fps > 240):
        print(f"  [WARN] bad FPS ({fps}) reported for {source}; assuming 25")
        fps = 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not live and start is None:
        p = Path(source)
        start = parse_start_time(p) or (datetime.fromtimestamp(p.stat().st_mtime)
                                       - timedelta(seconds=(n_frames / fps) if n_frames > 0 else 0))
    print(f"  source={redact(source)} fps={fps:.2f} frames={int(n_frames) if n_frames > 0 else '?'} "
          f"start={start.isoformat(timespec='seconds') if start else 'live clock'}")

    rows, idx, next_t, t0, dropped_dup, fails = [], -1, 0.0, time.monotonic(), 0, 0
    while True:
        if not cap.grab():
            if live and fails < 50:          # transient RTSP hiccup: retry briefly
                fails += 1; time.sleep(0.2); continue
            break
        fails = 0
        idx += 1
        # Position in seconds: frame count / fps for files (timestamps from some NVR files are unreliable),
        # wall clock for live streams (their FPS property is often wrong).
        t = (time.monotonic() - t0) if live else idx / fps
        if duration is not None and t > duration:
            break
        if t < next_t:
            continue
        next_t = t + stride
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        if resize and (frame.shape[1], frame.shape[0]) != resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
        h = dhash(frame)
        if method == "ssim":
            th = ssim_thumb(frame)
            if last_thumb is not None and ssim(last_thumb, th) >= ssim_th:
                dropped_dup += 1
                continue
        if any(hamming(h, k) < min_ham for k in kept_hashes):
            dropped_dup += 1
            continue
        if method == "ssim":
            last_thumb = th
        kept_hashes.append(h)
        ts = datetime.now() if live else start + timedelta(seconds=t)
        name = frame_name(cam_id, idx, ts)
        cv2.imwrite(str(staging / name), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        rows.append({"file": name, "source": redact(source), "frame_idx": idx, "pts_sec": round(t, 3),
                     "timestamp": ts.isoformat(timespec="milliseconds"), "dhash": f"{h:016x}",
                     "_h": h})
    cap.release()
    print(f"    -> {len(rows)} new distinct frames, {dropped_dup} near-duplicates dropped")
    return rows


def spread_select(rows: list[dict], k: int) -> list[dict]:
    """Pick k rows evenly spaced in recording order (keeps coverage across shifts / lighting)."""
    if len(rows) <= k:
        return rows
    pick = np.linspace(0, len(rows) - 1, k).round().astype(int)
    return [rows[i] for i in sorted(set(pick.tolist()))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, folder of videos, or rtsp:// URL")
    ap.add_argument("--site", required=True, help="site ID, e.g. factoryA (letters, digits, '-')")
    ap.add_argument("--cam-id", required=True, help="camera ID, e.g. cam01 (letters, digits, '-')")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"frames root (default {DEFAULT_OUT})")
    ap.add_argument("--stride", type=float, default=3.0, help="seconds between decoded frames (default 3)")
    ap.add_argument("--min-hamming", type=int, default=7, help="min dHash distance to every kept frame (default 7)")
    ap.add_argument("--cap", type=int, default=100, help="max frames kept per run for this camera (default 100)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to read per source (required for streams)")
    ap.add_argument("--start-time", default=None, help="recording start 'YYYY-mm-dd HH:MM:SS' (single file only)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--dest", type=Path, default=None,
                    help="explicit output folder (frames go to <dest>/images); overrides --out/<site>/<cam-id>")
    ap.add_argument("--name-format", choices=["timestamp", "plain"], default="timestamp",
                    help="'plain' = cctv_<cam-id>_f<idx>.jpg (make --cam-id unique per recording)")
    ap.add_argument("--dedup-with", type=Path, nargs="*", default=[],
                    help="other output folders of the SAME camera to de-duplicate against")
    ap.add_argument("--method", choices=["dhash", "ssim"], default="dhash",
                    help="'ssim' = SSIM vs last kept frame < --ssim-threshold, plus --min-hamming as a repeat guard "
                         "(recommended for fixed cameras, with --min-hamming 3)")
    ap.add_argument("--ssim-threshold", type=float, default=0.92)
    ap.add_argument("--resize", default=None,
                    help="WxH applied before hashing/saving, e.g. 1920x1080 to un-squeeze 960x1080 '1080P Lite' NVR video")
    args = ap.parse_args()
    global NAME_FORMAT
    NAME_FORMAT = args.name_format
    resize = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None

    for label, v in (("--site", args.site), ("--cam-id", args.cam_id)):
        if not ID_RE.match(v):
            ap.error(f"{label} must be letters, digits or '-' (got {v!r}); '_' is reserved as the filename separator")
    if is_stream(args.source) and args.duration is None:
        ap.error("--duration is required for live streams")

    if is_stream(args.source):
        sources = [args.source]
    else:
        src = Path(args.source)
        if src.is_dir():
            sources = [str(p) for p in sorted(src.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
        elif src.is_file():
            sources = [str(src)]
        else:
            ap.error(f"source not found: {src}")
        if not sources:
            ap.error(f"no video files ({', '.join(sorted(VIDEO_EXTS))}) in {src}")
    start = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S") if args.start_time else None
    if start and len(sources) > 1:
        ap.error("--start-time only applies to a single file")

    cam_dir = args.dest or args.out / args.site / args.cam_id
    img_dir = cam_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    staging = cam_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)  # leftovers from an interrupted run of this script only
    staging.mkdir()

    kept_hashes, n_existing = load_existing(cam_dir)
    for other in args.dedup_with:
        h_other, n_other = load_existing(other)
        kept_hashes += h_other
        print(f"  de-duplicating against {n_other} frames in {other}")
    print(f"Camera {args.site}/{args.cam_id}: {n_existing} frames already kept; "
          f"stride {args.stride}s, min Hamming {args.min_hamming}, cap {args.cap}")

    candidates: list[dict] = []
    for s in sources:
        candidates += scan(s, start, args.stride, args.min_hamming, kept_hashes, staging,
                           args.cam_id, args.duration, args.jpeg_quality, resize, args.method, args.ssim_threshold)

    selected = spread_select(candidates, args.cap)
    new_rows, collisions = [], 0
    for r in selected:
        dst = img_dir / r["file"]
        if dst.exists():  # same camera, same frame index, same millisecond: never overwrite
            collisions += 1
            continue
        shutil.move(str(staging / r["file"]), dst)
        new_rows.append(r)
    shutil.rmtree(staging)

    csv_path = cam_dir / "frames.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in new_rows:
            w.writerow({**r, "site": args.site, "cam_id": args.cam_id})

    print(f"\nKept {len(new_rows)} of {len(candidates)} distinct candidates "
          f"({collisions} name collisions skipped) -> {img_dir}")
    if len(candidates) > args.cap:
        print(f"  cap {args.cap} reached; frames were spread evenly over the recording")
    return 0


if __name__ == "__main__":
    sys.exit(main())
