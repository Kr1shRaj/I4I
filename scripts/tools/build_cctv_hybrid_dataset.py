"""
Build the hybrid catalog + CCTV training set, split by camera
=============================================================
Implements docs/cctv_domain_adaptation_plan.md sections 1.2 and 1.4 on top of the existing
catalog build (build_clean_9class_dataset.py). It never modifies the catalog build.

Inputs
------
- Catalog block: datasets/9class/train_oversampled_9class.txt (positives + other-machinery negatives).
- CCTV frames:   <cctv-root>/frames/<site>/<cam>/images/*.jpg (or <cctv-root>/<recording>/images/*.jpg, one
                 folder per recording) with REVIEWED labels in .../labels/*.txt.
                 Frames without a label file are not labelled yet and are skipped.
                 An EMPTY label file marks a reviewed hard negative from that camera.
                 Folders starting with '_' (e.g. _prelabels) are ignored.
- Camera split:  <cctv-root>/camera_splits.yaml, e.g.
                     factoryA/cam01: train
                     factoryA/cam03: val
                     ch05_20260928130326: {camera: cam05, split: val}   # several recordings, one camera
                 Every source with labels must be assigned. A camera belongs to exactly one split.
- --preview DIR: before QA, count frames using the pre-labels under DIR. Writes only the frame lists
                 (cctv_train_frames / cctv_val / cctv_test) and a *_preview.json; never a trainable list.

Outputs (in datasets/9class/, all paths './...' relative to that folder, so the project can move)
-------
- train_hybrid_9class<sfx>.txt   epoch shares catalog / CCTV / hard negatives (default 50 / 40 / 10)
- cctv_train_frames<sfx>.txt     the unique CCTV train frames (before repeats)
- cctv_val<sfx>.txt, cctv_test<sfx>.txt   held-out cameras only, each frame once
- data_9class_cctv<sfx>.yaml     train = hybrid list, val = CCTV-val (best.pt is selected on CCTV)
- build_cctv_hybrid_stats<sfx>.json

Share logic: CCTV frames are repeated up to --max-repeat times (plan: never > ~5x). If that still
can't reach the CCTV share, the catalog block is subsampled down to --min-catalog entries; below
that the achieved share is reported as lower than the target instead of forcing it.

Usage
-----
    python scripts/tools/build_cctv_hybrid_dataset.py
    python scripts/tools/build_cctv_hybrid_dataset.py --shares 0.55 0.35 0.10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"
SRC_DIR = PROJECT_ROOT / "datasets" / "source_38class"
CATALOG_LIST = DATASET_DIR / "train_oversampled_9class.txt"
DEFAULT_CCTV_ROOT = PROJECT_ROOT / "datasets" / "cctv"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES_9 = ["panel_saw", "hydraulic_press", "table_saw", "planer", "jointer",
                 "band_saw", "miter_saw", "edge_banding_machine", "fire_extinguisher"]
SEED = 42


def rel(p: Path) -> str:
    """'./...' path relative to DATASET_DIR. Ultralytics replaces only the leading './' with the list's
    folder (data/base.py get_img_files), so './../cctv/...' resolves correctly."""
    return "./" + Path(os.path.relpath(p, DATASET_DIR)).as_posix()


def resolve(entry: str) -> Path:
    """Resolve a list entry exactly as Ultralytics does."""
    return Path(str(DATASET_DIR) + os.sep + entry[2:]) if entry.startswith("./") else Path(entry)


def label_for(img: Path) -> Path:
    """Ultralytics img2label_paths: last /images/ -> /labels/, extension -> .txt."""
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(s.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def check_junctions(errors: list[str]) -> dict:
    out = {}
    for split in ("train", "valid", "test"):
        link, target = DATASET_DIR / f"{split}_9class" / "images", SRC_DIR / split / "images"
        try:
            is_junction = bool(os.readlink(link))
        except OSError:
            is_junction = False
        ok = link.exists() and target.exists() and os.path.samefile(link, target)
        n = sum(1 for _ in link.iterdir()) if ok else 0
        out[split] = {"junction": is_junction, "resolves_to_source": ok, "images": n}
        if not ok:
            errors.append(f"junction {link} does not resolve to {target} (re-link per CHANGELOG_REORGANIZATION.md)")
    return out


def check_label(lp: Path, errors: list[str]) -> list[int]:
    """Validate a YOLO-seg label file; return its class IDs (empty list = negative)."""
    classes = []
    for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{lp}:{ln}: non-numeric value"); continue
        if not 0 <= cls < len(CLASS_NAMES_9):
            errors.append(f"{lp}:{ln}: class {cls} outside 0-{len(CLASS_NAMES_9) - 1}")
        if len(vals) < 6 or len(vals) % 2:
            errors.append(f"{lp}:{ln}: polygon needs >= 3 x,y pairs (got {len(vals)} values)")
        if any(v < 0 or v > 1 for v in vals):
            errors.append(f"{lp}:{ln}: coordinate outside [0, 1]")
        classes.append(cls)
    return classes


def dhash_file(p: Path) -> int:
    import cv2
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def repeat_to(items: list[str], n_target: int) -> list[str]:
    """Repeat `items` to n_target entries: whole passes first, then an evenly spaced partial pass."""
    if not items or n_target <= 0:
        return []
    full, extra = divmod(n_target, len(items))
    out = items * full
    if extra:
        idx = np.linspace(0, len(items) - 1, extra).round().astype(int)
        out += [items[i] for i in idx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cctv-root", type=Path, default=DEFAULT_CCTV_ROOT)
    ap.add_argument("--shares", type=float, nargs=3, default=(0.50, 0.40, 0.10),
                    metavar=("CATALOG", "CCTV", "HARDNEG"))
    ap.add_argument("--max-repeat", type=float, default=5.0, help="max times a CCTV frame appears per epoch")
    ap.add_argument("--min-catalog", type=int, default=1500, help="never subsample the catalog block below this")
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _smoke")
    ap.add_argument("--allow-empty-val", action="store_true", help="for smoke tests only")
    ap.add_argument("--preview", type=Path, default=None, metavar="PRELABEL_DIR",
                    help="dry run before QA: treat pre-labels under this folder as labels; writes only "
                         "cctv_train_frames/cctv_val/cctv_test lists and stats, never the training list or YAML")
    args = ap.parse_args()

    s_cat, s_cctv, s_neg = args.shares
    if abs(s_cat + s_cctv + s_neg - 1) > 1e-6:
        ap.error("--shares must sum to 1")
    errors: list[str] = []
    warnings: list[str] = []
    rng = np.random.default_rng(SEED)

    # 1. Catalog side: junctions and every list entry
    junctions = check_junctions(errors)
    catalog = [l.strip() for l in CATALOG_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing_img = [e for e in catalog if not resolve(e).exists()]
    missing_lbl = [e for e in catalog if resolve(e).exists() and not label_for(resolve(e)).exists()]
    if missing_img:
        errors.append(f"{len(missing_img)} catalog entries have no image, e.g. {missing_img[0]}")
    if missing_lbl:
        errors.append(f"{len(missing_lbl)} catalog entries have no label file, e.g. {missing_lbl[0]}")
    print(f"Catalog list: {len(catalog)} entries ({len(set(catalog))} unique), "
          f"{len(missing_img)} missing images, {len(missing_lbl)} missing labels")

    # 2. CCTV side: camera split and reviewed labels.
    # A "source" is any folder holding images/ (datasets/cctv/frames/<site>/<cam>/ or one folder per recording,
    # e.g. datasets/cctv_factory_intake/<video_stem>/). Folders whose name starts with '_' are ignored.
    frames_root = args.cctv_root / "frames" if (args.cctv_root / "frames").is_dir() else args.cctv_root
    split_file = args.cctv_root / "camera_splits.yaml"
    sources = sorted(d.relative_to(frames_root).as_posix()
                     for pat in ("*", "*/*") for d in frames_root.glob(pat)
                     if (d / "images").is_dir() and not any(part.startswith("_") for part in d.relative_to(frames_root).parts))
    if not split_file.exists():
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text("# One entry per source folder: '<folder>: <split>' or '<folder>: {camera: <id>, split: <split>}'.\n"
                              "# Recordings of the same physical camera must share one camera id, and a camera has ONE split.\n"
                              "# Held-out cameras (val/test) must never appear in train (plan section 1.4).\n"
                              + "".join(f"# {c}: train\n" for c in sources), encoding="utf-8")
        print(f"[ERROR] wrote template {split_file}; assign the cameras and re-run")
        return 1
    raw_assign = yaml.safe_load(split_file.read_text(encoding="utf-8")) or {}
    assign, camera_of = {}, {}
    for k, v in raw_assign.items():
        cam_id, split = (v.get("camera", k), v.get("split")) if isinstance(v, dict) else (k, v)
        if split not in SPLITS:
            errors.append(f"camera_splits.yaml: {k}: invalid split {split!r} (use train/val/test)")
        assign[k], camera_of[k] = split, cam_id
    splits_per_camera = defaultdict(set)
    for k, cam_id in camera_of.items():
        splits_per_camera[cam_id].add(assign[k])
    for cam_id, s in splits_per_camera.items():
        if len(s) > 1:
            errors.append(f"camera {cam_id} is assigned to several splits {sorted(s)}: camera leakage")

    # --preview: count frames as if the pre-labels were reviewed, to check balance and paths before QA.
    # Writes only the plain frame lists, never the training list or YAML.
    prelabel_index = {}
    if args.preview:
        prelabel_index = {p.stem: p for p in args.preview.rglob("*.txt") if "labels" in p.parts}
        print(f"PREVIEW: using {len(prelabel_index)} unreviewed pre-label files from {args.preview}")
    get_label = (lambda p: prelabel_index.get(p.stem)) if args.preview else \
                (lambda p: label_for(p) if label_for(p).exists() else None)

    per_split = {s: {"pos": [], "neg": [], "cams": []} for s in SPLITS}
    class_counts = {s: Counter() for s in SPLITS}
    unlabelled_total, hashes = 0, {s: [] for s in SPLITS}
    for cam in sources:
        img_dir = frames_root / cam / "images"
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        labelled = [p for p in imgs if get_label(p) is not None]
        unlabelled_total += len(imgs) - len(labelled)
        if not labelled:
            print(f"  {cam}: {len(imgs)} frames, none labelled yet - skipped")
            continue
        split = assign.get(cam)
        if split not in SPLITS:
            errors.append(f"source {cam} has {len(labelled)} labelled frames but no split in {split_file.name}")
            continue
        per_split[split]["cams"].append(f"{cam} [{camera_of[cam]}]")
        for p in labelled:
            cls = check_label(get_label(p), errors)
            # An empty PRE-label only means the model found nothing; only a reviewed empty label is a hard negative
            per_split[split]["pos" if cls or args.preview else "neg"].append(p)
            class_counts[split].update(CLASS_NAMES_9[c] for c in cls if 0 <= c < len(CLASS_NAMES_9))
            hashes[split].append((dhash_file(p), p))
        print(f"  {cam} [{camera_of[cam]}] -> {split}: {len(labelled)} labelled "
              f"({sum(1 for p in labelled if get_label(p).stat().st_size)} with objects), "
              f"{len(imgs) - len(labelled)} awaiting labels")

    # 3. Leakage: near-identical frames across splits (same scene from a mis-assigned or copied camera)
    leaks = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for ha, pa in hashes[a]:
            for hb, pb in hashes[b]:
                if bin(ha ^ hb).count("1") <= 3:
                    leaks.append((str(pa), str(pb)))
    if leaks:
        errors.append(f"{len(leaks)} near-identical frame pairs across splits, e.g. {leaks[0]}")

    if not per_split["val"]["pos"] and not args.allow_empty_val:
        errors.append("no labelled CCTV-val frames: assign at least one held-out camera to 'val'")
    if not per_split["train"]["pos"]:
        errors.append("no labelled CCTV train frames")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    # 4. Train list at the requested epoch shares
    P = [rel(p) for p in per_split["train"]["pos"]]
    N = [rel(p) for p in per_split["train"]["neg"]]
    C = len(catalog)
    n_cctv = min(round(s_cctv / s_cat * C), math.floor(len(P) * args.max_repeat))
    n_cat = C
    if n_cctv < s_cctv / s_cat * C:  # repeat cap bound: shrink the catalog block towards the ratio
        n_cat = max(args.min_catalog, min(C, round(n_cctv * s_cat / s_cctv)))
        if n_cat > round(n_cctv * s_cat / s_cctv):
            warnings.append(f"CCTV share below target: only {len(P)} labelled train frames x{args.max_repeat:g} "
                            f"max repeat; catalog kept at the --min-catalog floor ({args.min_catalog})")
    total_wo_neg = n_cat + n_cctv
    n_neg = min(round(s_neg / (1 - s_neg) * total_wo_neg), math.floor(len(N) * args.max_repeat))
    if n_neg < round(s_neg / (1 - s_neg) * total_wo_neg):
        warnings.append(f"hard-negative share below target: {len(N)} reviewed empty frames from train cameras")

    cat_block = catalog if n_cat == C else [catalog[i] for i in sorted(rng.choice(C, n_cat, replace=False))]
    train = cat_block + repeat_to(P, n_cctv) + repeat_to(N, n_neg)
    rng.shuffle(train)
    total = len(train)
    achieved = {"catalog": round(n_cat / total, 3), "cctv": round(n_cctv / total, 3), "hardneg": round(n_neg / total, 3)}

    sfx = args.suffix
    # Plain CCTV frame lists (each frame once), written in both modes
    lists = {}
    frame_lists = {"train": per_split["train"]["pos"] + per_split["train"]["neg"],
                   "val": per_split["val"]["pos"] + per_split["val"]["neg"],
                   "test": per_split["test"]["pos"] + per_split["test"]["neg"]}
    for s, name in (("train", "cctv_train_frames"), ("val", "cctv_val"), ("test", "cctv_test")):
        entries = sorted(rel(p) for p in frame_lists[s])
        f = DATASET_DIR / f"{name}{sfx}.txt"
        f.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
        lists[s] = f.name
        if not entries:
            warnings.append(f"{f.name} is empty")

    yaml_path = None
    if args.preview:
        # Paths only: pre-labels are not in labels/, so Ultralytics would read these frames as backgrounds
        unresolved = [e for e in set(train) if not resolve(e).exists()]
    else:
        out_train = DATASET_DIR / f"train_hybrid_9class{sfx}.txt"
        out_train.write_text("\n".join(train) + "\n", encoding="utf-8")
        yaml_path = DATASET_DIR / f"data_9class_cctv{sfx}.yaml"
        yaml_path.write_text(
            f"# Generated by scripts/tools/build_cctv_hybrid_dataset.py; val/test are held-out CCTV cameras.\n"
            f"train: {out_train.name}\nval: {lists['val'] if per_split['val']['pos'] else 'valid_9class.txt'}\n"
            f"test: {lists['test']}\n\nnc: 9\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES_9)),
            encoding="utf-8")
        if not per_split["val"]["pos"]:
            warnings.append("val falls back to catalog valid_9class.txt (smoke test only)")
        # 5. Re-check every written entry the way Ultralytics will read it
        unresolved = [e for e in set(train) if not resolve(e).exists() or not label_for(resolve(e)).exists()]
    if unresolved:
        print(f"[ERROR] {len(unresolved)} train entries do not resolve, e.g. {unresolved[0]}")
        return 1

    stats = {
        "mode": f"PREVIEW from unreviewed pre-labels in {args.preview}" if args.preview else "reviewed labels",
        "lists": lists,
        "shares_target": {"catalog": s_cat, "cctv": s_cctv, "hardneg": s_neg},
        "shares_achieved": achieved,
        "train_entries": total,
        "catalog_entries": n_cat,
        "cctv_unique_train_frames": len(P), "cctv_train_entries": n_cctv,
        "cctv_repeat": round(n_cctv / len(P), 2) if P else 0,
        "hardneg_unique": len(N), "hardneg_entries": n_neg,
        "cameras": {s: per_split[s]["cams"] for s in SPLITS},
        "frames": {s: {"positive": len(per_split[s]["pos"]), "negative": len(per_split[s]["neg"])} for s in SPLITS},
        "instances_per_class": {s: dict(class_counts[s]) for s in SPLITS},
        "unlabelled_frames_skipped": unlabelled_total,
        "junctions": junctions,
        "warnings": warnings,
    }
    stats_name = f"build_cctv_hybrid_stats{sfx}{'_preview' if args.preview else ''}.json"
    (DATASET_DIR / stats_name).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n{'PREVIEW (nothing trainable written)' if args.preview else 'Train list train_hybrid_9class' + sfx + '.txt'}: "
          f"{total} entries | shares catalog {achieved['catalog']:.0%} / "
          f"CCTV {achieved['cctv']:.0%} (x{stats['cctv_repeat']}) / hard-neg {achieved['hardneg']:.0%}")
    print(f"Cameras: {stats['cameras']}")
    print(f"Frame lists: {lists} | stats: {stats_name}")
    if yaml_path:
        print(f"YAML: {yaml_path}")
    for w in warnings:
        print(f"  [WARN] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
