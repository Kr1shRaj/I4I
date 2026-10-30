"""
pipeline/stage2c_accept_gate.py
===============================
Acceptance gate for a tagged scrape batch (e.g. --tag highangle) before anything is labelled.

Gates, in order (each image gets the FIRST reason it fails):
1. cross_class   : a near-duplicate (dHash distance <= --max-dist) exists in ANOTHER class of the batch ->
                   every copy is rejected (we can't know which label is right).
2. in_batch_dup  : near-duplicate of an earlier image of the same class in the batch (first one kept).
3. catalog_dup   : near-duplicate of an image already in the training data (datasets/source_38class/*/images)
                   or of an untagged/other-tag intake image (dataset/train/<class>/).
4. clip_scene    : OpenCLIP zero-shot: the best "wanted" scene prompt (high-angle factory / production line)
                   must beat every "unwanted" prompt (product shot on plain background, ad/infographic,
                   security-camera hardware, close-up of parts, diagram).
5. clip_class    : the class name must rank in the top-3 of the 9 class prompts for that image.

Rejected images are MOVED (never deleted) to dataset/train/<class>/_gate_rejected/<reason>/.
A CSV with every decision is written to logs/stage2c_gate_<tag>.csv.

Usage
-----
    python -m pipeline.stage2c_accept_gate --tag highangle
    python -m pipeline.stage2c_accept_gate --tag highangle --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import CLASSES, CLIP_MODEL_NAME, CLIP_PRETRAINED, DATASET_TRAIN_DIR, LOGS_DIR, TARGET_9_CLASS_NAMES
from pipeline import progress

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # Krish/
CATALOG_GLOBS = [PROJECT_ROOT / "datasets" / "source_38class" / s / "images" for s in ("train", "valid", "test")]

WANTED = ["a high-angle photo of a factory floor with industrial machines",
          "an overhead view of a woodworking production line",
          "a wide photo of a busy furniture factory interior seen from above"]
UNWANTED = ["a product photo of a single machine on a plain white background",
            "an advertisement or infographic with a lot of text",
            "a photo of a security camera or CCTV camera product",
            "a close-up photo of machine parts",
            "a technical drawing or diagram",
            "a satellite or drone aerial photo of a factory complex from very high above",
            "a software screenshot or 3D CAD rendering",
            "a collage of several small photos"]


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def hamming_min(h: int, arr: np.ndarray) -> int:
    if not len(arr):
        return 64
    x = np.bitwise_xor(arr, np.uint64(h))
    return int(np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1).min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--classes", nargs="*", default=TARGET_9_CLASS_NAMES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    display = {c["name"]: c["display"] for c in CLASSES}

    batch = {c: sorted((DATASET_TRAIN_DIR / c).glob(f"img_{c}_{args.tag}_*.jpg")) for c in args.classes}
    n_total = sum(map(len, batch.values()))
    print(f"batch '{args.tag}': {n_total} images in {len(args.classes)} classes", flush=True)

    hashes = {}
    for c, files in batch.items():
        for p in files:
            hashes[p] = dhash(Image.open(p))
    # reference hashes: catalog + other intake images of every class
    ref = []
    for d in CATALOG_GLOBS:
        for p in d.iterdir():
            try:
                ref.append(dhash(Image.open(p)))
            except Exception:
                pass
    for c in args.classes:
        for p in (DATASET_TRAIN_DIR / c).glob("*.jpg"):
            if p not in hashes:
                ref.append(dhash(Image.open(p)))
    ref = np.array(ref, dtype=np.uint64)
    print(f"reference hashes (catalog + other intake): {len(ref)}", flush=True)

    reason = {}
    all_items = [(c, p, hashes[p]) for c, fs in batch.items() for p in fs]
    arr = np.array([h for _, _, h in all_items], dtype=np.uint64)
    cls_arr = np.array([c for c, _, _ in all_items])
    for i, (c, p, h) in enumerate(all_items):
        x = np.bitwise_xor(arr, np.uint64(h))
        d = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)
        if np.any((d <= args.max_dist) & (cls_arr != c)):
            reason[p] = "cross_class"
    kept_per_class = defaultdict(list)
    for c, p, h in all_items:
        if p in reason:
            continue
        if hamming_min(h, np.array(kept_per_class[c], dtype=np.uint64)) <= args.max_dist:
            reason[p] = "in_batch_dup"; continue
        if hamming_min(h, ref) <= args.max_dist:
            reason[p] = "catalog_dup"; continue
        kept_per_class[c].append(h)

    survivors = [(c, p) for c, p, _ in all_items if p not in reason]
    scores = {}
    if survivors:
        import open_clip
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=dev)
        tok = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        model.eval()
        with torch.no_grad():
            T = model.encode_text(tok(WANTED + UNWANTED).to(dev)).float()
            T = T / T.norm(dim=-1, keepdim=True)
            C = model.encode_text(tok([f"a photo of a {display[c]}" for c in TARGET_9_CLASS_NAMES]).to(dev)).float()
            C = C / C.norm(dim=-1, keepdim=True)
            for k in range(0, len(survivors), 64):
                chunk = survivors[k:k + 64]
                ims = torch.stack([pre(Image.open(p).convert("RGB")) for _, p in chunk]).to(dev)
                E = model.encode_image(ims).float(); E = E / E.norm(dim=-1, keepdim=True)
                s = (E @ T.T).cpu().numpy(); sc = (E @ C.T).cpu().numpy()
                for (c, p), row, crow in zip(chunk, s, sc):
                    best_w, best_u = row[:len(WANTED)].max(), row[len(WANTED):].max()
                    rank = int((crow > crow[TARGET_9_CLASS_NAMES.index(c)]).sum()) + 1
                    scores[p] = (float(best_w), float(best_u), rank)
                    if best_w <= best_u:
                        reason[p] = "clip_scene"
                    elif rank > 3:
                        reason[p] = "clip_class"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = LOGS_DIR / f"stage2c_gate_{args.tag}.csv"
    with log.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "decision", "wanted_sim", "unwanted_sim", "class_rank"])
        for c, p, _ in all_items:
            sw, su, rk = scores.get(p, ("", "", ""))
            w.writerow([c, p.name, reason.get(p, "accepted"), sw, su, rk])
    counts = defaultdict(lambda: defaultdict(int))
    for c, p, _ in all_items:
        counts[c][reason.get(p, "accepted")] += 1
        if p in reason and not args.dry_run:
            dst = DATASET_TRAIN_DIR / c / "_gate_rejected" / reason[p]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dst / p.name)
    reasons = ["accepted", "cross_class", "in_batch_dup", "catalog_dup", "clip_scene", "clip_class"]
    print(f"\n{'class':<22}" + "".join(f"{r:>13}" for r in reasons))
    for c in args.classes:
        print(f"{c:<22}" + "".join(f"{counts[c][r]:>13}" for r in reasons))
    print(f"{'TOTAL':<22}" + "".join(f"{sum(counts[c][r] for c in args.classes):>13}" for r in reasons))
    print(f"\n{'DRY RUN - nothing moved' if args.dry_run else 'rejects moved to <class>/_gate_rejected/'}; log: {log}")
    return 0


if __name__ == "__main__":
    with progress.run("stage2c_accept_gate"):
        sys.exit(main())
