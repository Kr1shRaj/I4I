"""
pipeline/stage4_clip_filter.py
===============================
Stage 4 -- Hybrid CLIP Filtering (PRD Section 3.2).

Two-level filtering pipeline:
1. Text-based coarse filter: scores every candidate image against its class
   text prompt ("a photo of a {display_name}"). Discards images scoring below
   the coarse text threshold.
2. Image-based fine filter: applied ONLY to classes in a known confusable
   cluster (Section 2.1). Scores candidate images against hand-picked reference
   images in reference_images/<class>/ via mean cosine similarity.

Usage
-----
    python -m pipeline.stage4_clip_filter
    python -m pipeline.stage4_clip_filter --classes lathe table_saw
    python -m pipeline.stage4_clip_filter --coarse-thresh 0.18 --fine-thresh 0.60

Output
------
- Removes rejected images from dataset/train/<class>/
- Saves score statistics and acceptance logs to logs/stage4_clip_filter_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    CLIP_BATCH_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
    REFERENCE_IMAGES_DIR,
    class_by_name,
)

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip not installed. Run: pip install open-clip-torch")
    sys.exit(1)


def load_model():
    device = "cpu"
    if torch.cuda.is_available():
        try:
            _dummy = torch.zeros(1, device="cuda") + 1
            device = "cuda"
        except Exception:
            device = "cpu"
    print(f"Loading OpenCLIP model ({CLIP_MODEL_NAME} / {CLIP_PRETRAINED}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer, device


def get_image_embeddings(
    image_paths: list[Path], model, preprocess, device, batch_size: int = CLIP_BATCH_SIZE
) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        valid_indices = []
        for idx, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                t = preprocess(img)
                tensors.append(t)
                valid_indices.append(idx)
            except Exception as exc:
                print(f"  [WARN] Skipping bad image {p.name}: {exc}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch_tensor)
            embs /= embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs)

    if not all_embs:
        return torch.empty((0, 512), device=device)
    return torch.cat(all_embs, dim=0)


def filter_class(
    cls_name: str,
    model,
    preprocess,
    tokenizer,
    device,
    coarse_thresh: float,
    fine_thresh: float,
) -> dict:
    cls = class_by_name(cls_name)
    display_name = cls["display"]
    is_confusable = cls["confusable"]
    class_dir = DATASET_TRAIN_DIR / cls_name

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if class_dir.exists() else []

    if not files:
        return {
            "class": cls_name,
            "confusable": is_confusable,
            "input_count": 0,
            "passed_coarse": 0,
            "passed_fine": 0,
            "final_accepted": 0,
            "rejected_coarse": 0,
            "rejected_fine": 0,
        }

    # Step 1: Text prompt embeddings for coarse filtering (Positive + Negative non-photo prompts)
    prompt = f"a photo of a {display_name}"
    neg_prompts = [
        "diagram", "cad drawing", "schematic", "technical specification sheet",
        "blueprint", "user manual illustration", "rendering", "vector graphics",
        "sketch", "poster art", "line art", "safety infographic", "signage template",
        "clipart icon", "vector graphic poster", "digital illustration", "vector logo",
        "graphic design poster", "stock vector icon", "safety poster design"
    ]

    all_prompts = [prompt] + neg_prompts
    tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        all_text_embs = model.encode_text(tokens)
        all_text_embs /= all_text_embs.norm(dim=-1, keepdim=True)
        text_emb = all_text_embs[0:1]
        neg_embs = all_text_embs[1:]

    # Compute image embeddings for candidate images
    img_embs = get_image_embeddings(files, model, preprocess, device)
    if len(img_embs) == 0:
        return {
            "class": cls_name,
            "confusable": is_confusable,
            "input_count": len(files),
            "passed_coarse": 0,
            "passed_fine": 0,
            "final_accepted": 0,
            "rejected_coarse": len(files),
            "rejected_fine": 0,
        }

    # Coarse scoring (Text-Image Cosine Similarity)
    coarse_scores = (img_embs @ text_emb.T).squeeze(-1).cpu().numpy()
    neg_scores = (img_embs @ neg_embs.T).max(dim=-1).values.cpu().numpy()

    # Pass coarse text threshold AND must be more photo-like than non-photo diagram prompt
    passed_coarse_mask = (coarse_scores >= coarse_thresh) & (coarse_scores > neg_scores - 0.05)
    rejected_coarse_indices = np.where(~passed_coarse_mask)[0]

    passed_coarse_files = [files[i] for i in range(len(files)) if passed_coarse_mask[i]]
    passed_coarse_embs = img_embs[passed_coarse_mask]

    # Delete coarse-rejected files
    for idx in rejected_coarse_indices:
        files[idx].unlink(missing_ok=True)

    passed_fine_files = passed_coarse_files
    rejected_fine_count = 0

    # Step 2: Fine Image-to-Image filtering for confusable classes + Cross-class filtering
    if is_confusable or cls_name in ["drum_sander", "edge_banding_machine", "sanding_machines", "overhead_hoist", "crane", "ppe_station", "first_aid_station", "emergency_exit_sign"]:
        ref_dir = REFERENCE_IMAGES_DIR / cls_name
        ref_files = sorted([p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if ref_dir.exists() else []

        if ref_files and len(passed_coarse_embs) > 0:
            ref_embs = get_image_embeddings(ref_files, model, preprocess, device)
            if len(ref_embs) > 0:
                # Mean cosine similarity against target reference set
                fine_scores = (passed_coarse_embs @ ref_embs.T).mean(dim=-1).cpu().numpy()
                passed_fine_mask = fine_scores >= fine_thresh

                # DUAL-SET CROSS-CONTAMINATION CHECKS
                # 1. drum_sander vs sanding_machines
                if cls_name == "drum_sander":
                    sander_ref_dir = REFERENCE_IMAGES_DIR / "sanding_machines"
                    sander_refs = sorted([p for p in sander_ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if sander_ref_dir.exists() else []
                    if sander_refs:
                        sander_ref_embs = get_image_embeddings(sander_refs, model, preprocess, device)
                        if len(sander_ref_embs) > 0:
                            sander_sim = (passed_coarse_embs @ sander_ref_embs.T).mean(dim=-1).cpu().numpy()
                            # Discard if image is closer to general sanding_machines than drum_sander
                            passed_fine_mask = passed_fine_mask & (fine_scores >= sander_sim)

                # 2. edge_banding_machine vs cnc_router / panel_saw
                if cls_name == "edge_banding_machine":
                    router_ref_dir = REFERENCE_IMAGES_DIR / "cnc_router"
                    router_refs = sorted([p for p in router_ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if router_ref_dir.exists() else []
                    if router_refs:
                        router_ref_embs = get_image_embeddings(router_refs, model, preprocess, device)
                        if len(router_ref_embs) > 0:
                            router_sim = (passed_coarse_embs @ router_ref_embs.T).mean(dim=-1).cpu().numpy()
                            passed_fine_mask = passed_fine_mask & (fine_scores >= router_sim)

                # 3. overhead_hoist vs crane
                if cls_name == "overhead_hoist":
                    crane_ref_dir = REFERENCE_IMAGES_DIR / "crane"
                    crane_refs = sorted([p for p in crane_ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if crane_ref_dir.exists() else []
                    if crane_refs:
                        crane_ref_embs = get_image_embeddings(crane_refs, model, preprocess, device)
                        if len(crane_ref_embs) > 0:
                            crane_sim = (passed_coarse_embs @ crane_ref_embs.T).mean(dim=-1).cpu().numpy()
                            passed_fine_mask = passed_fine_mask & (fine_scores >= crane_sim)

                if cls_name == "crane":
                    hoist_ref_dir = REFERENCE_IMAGES_DIR / "overhead_hoist"
                    hoist_refs = sorted([p for p in hoist_ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if hoist_ref_dir.exists() else []
                    if hoist_refs:
                        hoist_ref_embs = get_image_embeddings(hoist_refs, model, preprocess, device)
                        if len(hoist_ref_embs) > 0:
                            hoist_sim = (passed_coarse_embs @ hoist_ref_embs.T).mean(dim=-1).cpu().numpy()
                            passed_fine_mask = passed_fine_mask & (fine_scores >= hoist_sim)

                # 4. ppe_station vs storage_racking / control_panel
                if cls_name == "ppe_station":
                    for other_cls in ["storage_racking", "control_panel"]:
                        other_ref_dir = REFERENCE_IMAGES_DIR / other_cls
                        other_refs = sorted([p for p in other_ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if other_ref_dir.exists() else []
                        if other_refs:
                            other_ref_embs = get_image_embeddings(other_refs, model, preprocess, device)
                            if len(other_ref_embs) > 0:
                                other_sim = (passed_coarse_embs @ other_ref_embs.T).mean(dim=-1).cpu().numpy()
                                passed_fine_mask = passed_fine_mask & (fine_scores >= other_sim)

                rejected_fine_indices = np.where(~passed_fine_mask)[0]
                for idx in rejected_fine_indices:
                    passed_coarse_files[idx].unlink(missing_ok=True)
                    rejected_fine_count += 1

                passed_fine_files = [passed_coarse_files[i] for i in range(len(passed_coarse_files)) if passed_fine_mask[i]]
        else:
            if not ref_files:
                print(f"  [WARN] Confusable class '{cls_name}' has 0 reference images! Fine filtering skipped.")

    final_accepted = len(passed_fine_files)
    return {
        "class": cls_name,
        "confusable": is_confusable,
        "input_count": len(files),
        "passed_coarse": len(passed_coarse_files),
        "passed_fine": final_accepted,
        "final_accepted": final_accepted,
        "rejected_coarse": len(rejected_coarse_indices),
        "rejected_fine": rejected_fine_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: Hybrid CLIP Filtering.")
    parser.add_argument("--classes", nargs="+", default=None, help="Filter specific class slugs")
    parser.add_argument("--coarse-thresh", type=float, default=0.18, help="Coarse text similarity threshold (default: 0.18)")
    parser.add_argument("--fine-thresh", type=float, default=0.30, help="Fine reference similarity threshold (default: 0.30). Calibration showed reference self-similarity as low as 0.17 (planer) so 0.55 was rejecting correct images.")
    args = parser.parse_args()

    if args.classes:
        target_names = [c for c in args.classes if c in ALL_CLASS_NAMES]
    else:
        target_names = ALL_CLASS_NAMES

    model, preprocess, tokenizer, device = load_model()

    print("=" * 65)
    print("STAGE 4 -- Hybrid CLIP Filtering")
    print(f"Classes        : {len(target_names)}")
    print(f"Coarse Thresh  : {args.coarse_thresh} (Text-to-Image)")
    print(f"Fine Thresh    : {args.fine_thresh} (Image-to-Image for confusable clusters — reference self-sim range: 0.17–0.94)")
    print("=" * 65)

    results = []
    for cls_name in tqdm(target_names, desc="CLIP Filtering", unit="class"):
        res = filter_class(cls_name, model, preprocess, tokenizer, device, args.coarse_thresh, args.fine_thresh)
        results.append(res)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "stage4_clip_filter_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "coarse_thresh": args.coarse_thresh,
                "fine_thresh": args.fine_thresh,
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("HYBRID CLIP FILTERING SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<22} {'Input':>6} {'PassCoarse':>11} {'PassFine':>9} {'Accepted':>9}")
    print("-" * 65)
    tot_in = tot_coarse = tot_fine = tot_acc = 0
    for r in results:
        print(
            f"  {r['class']:<22} {r['input_count']:>6} {r['passed_coarse']:>11} "
            f"{r['passed_fine']:>9} {r['final_accepted']:>9}"
        )
        tot_in += r["input_count"]
        tot_coarse += r["passed_coarse"]
        tot_fine += r["passed_fine"]
        tot_acc += r["final_accepted"]
    print("-" * 65)
    print(f"  {'TOTAL':<22} {tot_in:>6} {tot_coarse:>11} {tot_fine:>9} {tot_acc:>9}")
    print("=" * 65)
    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
