"""
Build Clean 9-Class Dataset Pipeline
====================================
1. Quarantine 6 fire extinguisher outlier failures
2. Quarantine 54 CAD diagrams across target classes
3. fire_extinguisher / edge_banding_machine instances: rebuilt with scripts/tools/label_instances.py
   (2026-10-21: the old morphological-closing merges are OFF; they fused touching cylinders and ballooned
   edge-bander masks). Reviewed fixes from datasets/9class/label_fixes/label_decisions.yaml are applied:
   SAM-tightened edge-bander polygons, SAM per-cylinder masks, and image-level quarantines.
4. (removed) edge-bander fragment closing
5. Remap target class IDs to 0-8 in separate label folders (zero image copying via junctions)
6. Keep a sample of non-target images as negative (background) samples with empty labels
7. Generate data_9class.yaml, train_oversampled_9class.txt (1.5x positives, negatives 1x),
   valid_9class.txt and test_9class.txt -- all paths relative, so the project can be moved
8. Compute updated instance counts and validation metrics
"""

import os
import json
import _winapi
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "datasets" / "source_38class"   # 38-class Roboflow export (images + labels)
DATASET_DIR = PROJECT_ROOT / "datasets" / "9class"       # 9-class build output (junctions, labels, lists)

# 9 Target Classes & Mapping
TARGET_MAP = {
    14: 0, # panel_saw
    5:  1, # hydraulic_press
    11: 2, # table_saw
    15: 3, # planer
    16: 4, # jointer
    12: 5, # band_saw
    13: 6, # miter_saw
    26: 7, # edge_banding_machine
    21: 8  # fire_extinguisher
}

CLASS_NAMES_9 = [
    "panel_saw",
    "hydraulic_press",
    "table_saw",
    "planer",
    "jointer",
    "band_saw",
    "miter_saw",
    "edge_banding_machine",
    "fire_extinguisher"
]

# Negative (background) sampling: fraction of each split's positive image count
NEG_RATIO = {"train": 0.15, "valid": 0.15, "test": 0.15}
NEG_SEED = 42
# Source classes (38-class IDs) whose images often show unlabeled fire extinguishers nearby:
# emergency_exit_sign, first_aid_station, ppe_station. Never use these as negatives.
NEG_EXCLUDE_SOURCE_CLASSES = {27, 28, 34}

# 1. Quarantined fire extinguisher outlier failures (6 images)
FE_OUTLIER_FAILURES = {
    'img_fire_extinguisher_ddgs_00027_jpg.rf.ccee3fe5bf4ead45fafedf9e856dcddc',
    'img_fire_extinguisher_ddgs_00101_jpg.rf.a8486dc822ddceea53b44aed2b8008e7',
    'img_fire_extinguisher_ddgs_00312_jpg.rf.9429622a88afbffaffe723a5a148ddc9',
    'img_fire_extinguisher_ddgs_00081_jpg.rf.1457c473d15cce5046092ce8dce55096',
    'img_fire_extinguisher_ddgs_00100_jpg.rf.40039a7693e8b9b9c303021ba8457c29',
    'img_fire_extinguisher_ddgs_00481_jpg.rf.651be1b2d0c5fd539803638b55196a13'
}

# 2. Quarantined CAD diagrams (54 images)
diag_json_path = DATASET_DIR / "quarantined_diagrams_54.json"
with open(diag_json_path, "r") as f:
    diag_data = json.load(f)
CAD_DIAGRAM_STEMS = set(diag_data["stems"])

ALL_QUARANTINE_STEMS = FE_OUTLIER_FAILURES.union(CAD_DIAGRAM_STEMS)
print(f"Total Quarantined Images: {len(ALL_QUARANTINE_STEMS)} (6 FE failures + {len(CAD_DIAGRAM_STEMS)} CAD diagrams)")

# 2026-10-21 label fixes (docs/label_policy.md)
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_instances import rebuild_from_norm  # noqa: E402
from prelabel_cctv import mask_to_polygon  # noqa: E402

FIX_DIR = DATASET_DIR / "label_fixes"
DECISIONS = yaml.safe_load((FIX_DIR / "label_decisions.yaml").read_text(encoding="utf-8"))
EB_PROPOSALS = json.loads((FIX_DIR / "edge_bander_proposals.json").read_text(encoding="utf-8"))
EB_FLAG_BOX = 0.80  # must match fix_edge_bander_labels.py --flag-box
EB_ACTION = {}
for action in ("accept", "keep", "quarantine"):
    for i in DECISIONS["edge_bander"][action]:
        EB_ACTION[i] = action
EB_FIX = {(p["split"], p["stem"], p["k"]): (EB_ACTION.get(i, "keep"), p["polygon"]) for i, p in enumerate(EB_PROPOSALS)}
FE_SAM_BOXES = DECISIONS["fire_extinguisher"]["sam_boxes"] or {}
LABEL_QUARANTINE = set(DECISIONS["fire_extinguisher"]["quarantine"] or []) | {
    p["stem"] for i, p in enumerate(EB_PROPOSALS) if EB_ACTION.get(i) == "quarantine"}
print(f"Label-fix quarantine (needs human re-annotation): {len(LABEL_QUARANTINE)} images")
NEG_QUARANTINE = set((DECISIONS.get("negatives") or {}).get("quarantine") or {})
NEG_EXCLUDE_REVIEWED = set((DECISIONS.get("negatives") or {}).get("exclude_source_classes") or [])
# User curation (datasetchange file): bad images out of every split; flawed labels replaced by reviewed
# SAM 2.1 L polygons (scripts/tools/fix_user_labels.py -> user_fix_polygons.json). Fix entries marked
# "exclude" (no target machine) leave the build like the quarantine.
USER_FIXES = DECISIONS.get("user_fixes") or {}
USER_QUARANTINE = set((DECISIONS.get("user_quarantine") or {}).get("stems") or []) | {
    s for s, v in USER_FIXES.items() if v == "exclude"}
_ufp = FIX_DIR / "user_fix_polygons.json"
USER_FIX_POLYS = {s: v["instances"] for s, v in
                  (json.loads(_ufp.read_text(encoding="utf-8")) if _ufp.exists() else {}).items()
                  if s in USER_FIXES and USER_FIXES[s] != "exclude"}
_missing = [s for s, v in USER_FIXES.items() if v != "exclude" and s not in USER_FIX_POLYS]
if _missing:
    raise RuntimeError(f"{len(_missing)} user_fixes stems have no polygons; run fix_user_labels.py propose")
print(f"User quarantine: {len(USER_QUARANTINE)} images | user label fixes: {len(USER_FIX_POLYS)} images")
_SAM = None


def sam_masks(img, boxes_norm):
    global _SAM
    if _SAM is None:
        from ultralytics import SAM
        _SAM = SAM(str(PROJECT_ROOT / "models" / "pretrained" / "sam2.1_l.pt"))
    h, w = img.shape[:2]
    boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes_norm]
    masks = _SAM(img, bboxes=boxes, verbose=False)[0].masks.data.cpu().numpy() > 0.5
    out = []
    for m in masks:
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        p = mask_to_polygon(m, 0.002)
        if p is not None and len(p) >= 3:
            out.append(np.clip(p.astype(np.float64) / [w, h], 0, 1))
    return out

# Merge Functions
def merge_fire_extinguisher(polys, img_shape, ksize_ratio=0.005, min_area_ratio=0.0005):
    h, w = img_shape[:2]
    diag = np.sqrt(h**2 + w**2)
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [p], 255)
    ksize = max(3, int(diag * ksize_ratio))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = w * h
    merged = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area_ratio * total_area:
            epsilon = 0.002 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) >= 3:
                merged.append(approx)
    return merged

def merge_edge_banding_machine(polys, img_shape, ksize_ratio=0.012):
    # Light, single closing: only bridges small gaps between touching fragments of one machine.
    # (Previously 4% rect kernel x2 iterations, which ballooned masks to ~full-frame slabs.)
    h, w = img_shape[:2]
    diag = np.sqrt(h**2 + w**2)
    mask = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [p], 255)
    ksize = max(5, int(diag * ksize_ratio))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = w * h
    merged = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= 0.01 * total_area:
            epsilon = 0.003 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) >= 3:
                merged.append(approx)
    return merged

def ensure_junction(link, target):
    """Create link -> target junction, replacing a stale/broken junction (never a real folder)."""
    if os.path.exists(link) and os.path.samefile(link, target):
        print(f"Junction OK: {link}")
        return
    if os.path.lexists(link):
        try:
            os.readlink(link)  # succeeds only for junctions/symlinks
        except OSError:
            raise RuntimeError(f"{link} is a real directory, not a junction; refusing to replace it")
        os.rmdir(link)  # removes the junction only, never the target's contents
        print(f"Removed stale junction: {link}")
    _winapi.CreateJunction(str(target), str(link))
    print(f"Created junction: {link} -> {target}")

def rel(p):
    """Path relative to DATASET_DIR, in the './...' form Ultralytics resolves against the list file."""
    return "./" + Path(os.path.relpath(p, DATASET_DIR)).as_posix()  # './../hard_negatives/...' also resolves

# Setup Zero-Copy Folder Structure
print("\nSetting up zero-copy folder structure with NTFS junctions...")
for split in ["train", "valid", "test"]:
    split_dir = DATASET_DIR / f"{split}_9class"
    split_dir.mkdir(exist_ok=True)
    ensure_junction(split_dir / "images", SRC_DIR / split / "images")

    # Fresh label directory (backup of the previous build: _backup_9class_pre_cctv)
    lbl_dir = split_dir / "labels"
    lbl_dir.mkdir(exist_ok=True)
    for old in lbl_dir.glob("*.txt"):
        old.unlink()
    cache = split_dir / "labels.cache"
    if cache.exists():
        cache.unlink()

# Process Labels Across Splits
print("\nProcessing and remapping labels (0-8) with merged geometries...")

stats_summary = {
    'fire_extinguisher': {'before_instances': [], 'after_instances': []},
    'edge_banding_machine': {'before_instances': [], 'after_instances': [],
                             'before_heights': [], 'after_heights': [], 'sam_replaced': 0}
}

class_image_files = {name: [] for name in CLASS_NAMES_9}
positive_files = {}
negative_files = {}
split_counts = {}

for split in ["train", "valid", "test"]:
    src_lbl_dir = SRC_DIR / split / "labels"
    src_img_dir = SRC_DIR / split / "images"
    dst_lbl_dir = DATASET_DIR / f"{split}_9class" / "labels"

    raw_files = sorted(src_lbl_dir.glob("*.txt"))
    kept_count = 0
    quarantined_count = 0
    label_quarantined_count = 0
    user_quarantined_count = 0
    user_fixed_count = 0
    neg_candidates = []
    neg_source_ids = {}
    excluded_neg = 0
    positive_files[split] = []

    for lp in raw_files:
        stem = lp.stem
        ip = src_img_dir / (stem + ".jpg")

        # Check quarantine
        if stem in ALL_QUARANTINE_STEMS:
            quarantined_count += 1
            continue
        if stem in LABEL_QUARANTINE:
            label_quarantined_count += 1
            continue
        if stem in USER_QUARANTINE:
            user_quarantined_count += 1
            continue
        if stem in USER_FIX_POLYS:   # reviewed SAM 2.1 L polygons replace all target labels
            new_label_lines = []
            for ins in USER_FIX_POLYS[stem]:
                if split == "train":
                    class_image_files[CLASS_NAMES_9[ins["cls"]]].append(ip)
                new_label_lines.append(f"{ins['cls']} " + " ".join(f"{v:.6f}" for pt in ins["polygon"] for v in pt))
            (dst_lbl_dir / (stem + ".txt")).write_text("\n".join(new_label_lines) + "\n")
            positive_files[split].append(DATASET_DIR / f"{split}_9class" / "images" / ip.name)
            kept_count += 1
            user_fixed_count += 1
            continue

        with open(lp, 'r') as f:
            lines = [l.strip().split() for l in f if l.strip()]

        # Group polygons by original class ID
        by_class = {}
        for l in lines:
            cid = int(l[0])
            if cid in TARGET_MAP:
                coords = [float(x) for x in l[1:]]
                by_class.setdefault(cid, []).append(coords)

        if not by_class:
            # No target classes: candidate negative (background) sample
            if not ip.exists():
                continue
            if any(int(l[0]) in NEG_EXCLUDE_SOURCE_CLASSES for l in lines):
                excluded_neg += 1
                continue
            neg_candidates.append(ip)
            neg_source_ids[ip] = {int(l[0]) for l in lines}
            continue

        # Needs image size for merging
        im = None
        new_label_lines = []

        for cid, coord_list in by_class.items():
            new_cid = TARGET_MAP[cid]
            cls_name = CLASS_NAMES_9[new_cid]

            if split == "train":
                class_image_files[cls_name].append(ip)

            if cid == 21: # fire_extinguisher: instance rebuild (no fusion) or reviewed SAM per-cylinder masks
                if im is None:
                    im = cv2.imread(str(ip))
                h, w = im.shape[:2]
                stats_summary['fire_extinguisher']['before_instances'].append(len(coord_list))
                if stem in FE_SAM_BOXES:
                    merged = sam_masks(im, FE_SAM_BOXES[stem])
                else:
                    merged = rebuild_from_norm(coord_list, (h, w), "fire_extinguisher")
                stats_summary['fire_extinguisher']['after_instances'].append(len(merged))
                for mp in merged:
                    new_label_lines.append(f"{new_cid} " + " ".join(f"{x:.6f}" for x in mp.flatten()))

            elif cid == 26: # edge_banding_machine: apply reviewed SAM tightening, then instance rebuild
                if im is None:
                    im = cv2.imread(str(ip))
                h, w = im.shape[:2]
                stats_summary['edge_banding_machine']['before_instances'].append(len(coord_list))
                stats_summary['edge_banding_machine']['before_heights'] += [
                    float(np.ptp(np.array(c)[1::2])) for c in coord_list]
                fixed, k = [], 0
                for c in coord_list:
                    xy = np.array(c).reshape(-1, 2)
                    if np.ptp(xy[:, 0]) * np.ptp(xy[:, 1]) > EB_FLAG_BOX:
                        action, poly = EB_FIX.get((split, stem, k), ("keep", None))
                        k += 1
                        if action == "accept" and poly:
                            fixed.append([v for pt in poly for v in pt])
                            stats_summary['edge_banding_machine']['sam_replaced'] += 1
                            continue
                    fixed.append(c)
                merged = rebuild_from_norm(fixed, (h, w), "edge_banding_machine")
                stats_summary['edge_banding_machine']['after_instances'].append(len(merged))
                for mp in merged:
                    stats_summary['edge_banding_machine']['after_heights'].append(float(np.ptp(mp[:, 1])))
                    new_label_lines.append(f"{new_cid} " + " ".join(f"{x:.6f}" for x in mp.flatten()))

            else: # Other 7 classes (single clean objects)
                for coords in coord_list:
                    new_label_lines.append(f"{new_cid} " + " ".join(f"{x:.6f}" for x in coords))

        # Write clean 9-class label
        dst_lp = dst_lbl_dir / (stem + ".txt")
        with open(dst_lp, 'w') as f:
            for l in new_label_lines:
                f.write(l + "\n")
        positive_files[split].append(DATASET_DIR / f"{split}_9class" / "images" / ip.name)
        kept_count += 1

    # Sample negatives (deterministic) and write empty label files
    rng = np.random.default_rng(NEG_SEED)
    # Sized from the positives before the user quarantine, so the reviewed negative sample is unchanged
    n_neg = min(len(neg_candidates), int(round((kept_count + user_quarantined_count) * NEG_RATIO[split])))
    picked = sorted(rng.choice(len(neg_candidates), size=n_neg, replace=False).tolist())
    negative_files[split] = []
    neg_dropped = 0
    for i in picked:
        ip = neg_candidates[i]
        # Reviewed after sampling (audit_negatives.py), so the rest of the sample stays the same
        if ip.stem in NEG_QUARANTINE or ip.stem in USER_QUARANTINE or neg_source_ids[ip] & NEG_EXCLUDE_REVIEWED:
            neg_dropped += 1
            continue
        (dst_lbl_dir / (ip.stem + ".txt")).write_text("")
        negative_files[split].append(DATASET_DIR / f"{split}_9class" / "images" / ip.name)

    n_neg = len(negative_files[split])
    split_counts[split] = {'positive': kept_count, 'negative': n_neg, 'negative_review_dropped': neg_dropped,
                           'negative_candidates': len(neg_candidates),
                           'excluded_fire_safety_sources': excluded_neg,
                           'quarantined': quarantined_count,
                           'label_fix_quarantined': label_quarantined_count,
                           'user_quarantined': user_quarantined_count, 'user_label_fixed': user_fixed_count}
    print(f"[{split.upper()}_9CLASS] Positives: {kept_count} | Negatives: {n_neg} "
          f"(from {len(neg_candidates)} candidates, {excluded_neg} fire-safety sources excluded) "
          f"| Quarantined: {quarantined_count} | Label-fix quarantined: {label_quarantined_count} "
          f"| User quarantined: {user_quarantined_count} | User label fixes: {user_fixed_count}")

# Generate data_9class.yaml (no 'path:' -> Ultralytics uses this YAML's folder as dataset root)
data_yaml_path = DATASET_DIR / "data_9class.yaml"
data_yaml_content = """train: train_oversampled_9class.txt
val: valid_9class.txt
test: test_9class.txt

nc: 9
names:
  0: panel_saw
  1: hydraulic_press
  2: table_saw
  3: planer
  4: jointer
  5: band_saw
  6: miter_saw
  7: edge_banding_machine
  8: fire_extinguisher
"""
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)
print(f"\nGenerated: {data_yaml_path.name}")

# Generate valid_9class.txt / test_9class.txt (positives + negatives)
for split in ["valid", "test"]:
    lst = sorted(rel(p) for p in positive_files[split] + negative_files[split])
    (DATASET_DIR / f"{split}_9class.txt").write_text("\n".join(lst) + "\n")
    print(f"Generated {split}_9class.txt with {len(lst)} images")

# Generate train_oversampled_9class.txt (1.5x across all 9 classes, negatives added once)
train_oversampled_path = DATASET_DIR / "train_oversampled_9class.txt"
final_train_list = []

# Deduplicate images per class, then apply 1.5x
for name, img_list in class_image_files.items():
    unique_imgs = sorted(list(set(img_list)))
    n_base = len(unique_imgs)
    # 1.5x oversampling
    n_oversampled = int(round(n_base * 1.5))
    repeated = unique_imgs.copy()
    extra_needed = n_oversampled - n_base
    if extra_needed > 0:
        step = max(1, n_base // extra_needed)
        extra_imgs = unique_imgs[::step][:extra_needed]
        repeated.extend(extra_imgs)
    # Re-route path through train_9class/images/
    for p in repeated:
        final_train_list.append(rel(DATASET_DIR / "train_9class" / "images" / p.name))
    print(f"[{name.upper()}] Unique Images: {n_base} -> Oversampled (1.5x): {len(repeated)}")

# Reviewed high-angle web images (scripts/tools/annotate_highangle.py export). Train only, added once.
HIGHANGLE_LIST = DATASET_DIR / "train_highangle_9class.txt"
highangle = HIGHANGLE_LIST.read_text().split() if HIGHANGLE_LIST.exists() else []
final_train_list += highangle
print(f"[HIGH-ANGLE] Added once: {len(highangle)} (from {HIGHANGLE_LIST.name})")

final_train_list += [rel(p) for p in negative_files["train"]]
print(f"[NEGATIVES] Added once (not oversampled): {len(negative_files['train'])}")

# Targeted look-alike hard negatives (datasets/hard_negatives, reviewed; empty labels). Train only.
# Total negatives (other-machinery + targeted) are capped at HARDNEG_MAX_SHARE of the train list.
HARDNEG_DIR = PROJECT_ROOT / "datasets" / "hard_negatives"
HARDNEG_MAX_SHARE = 0.15
hard_negs = sorted(p for p in (HARDNEG_DIR / "images").glob("*.jpg") if (HARDNEG_DIR / "labels" / (p.stem + ".txt")).exists())
n_pos_entries = len(final_train_list) - len(negative_files["train"])
max_neg = int(HARDNEG_MAX_SHARE / (1 - HARDNEG_MAX_SHARE) * n_pos_entries)
room = max(0, max_neg - len(negative_files["train"]))
if len(hard_negs) > room:
    hard_negs = hard_negs[:room]
final_train_list += [rel(p) for p in hard_negs]
n_neg_total = len(negative_files["train"]) + len(hard_negs)
print(f"[HARD NEGATIVES] Targeted look-alikes added: {len(hard_negs)} | all negatives "
      f"{n_neg_total}/{len(final_train_list)} = {n_neg_total / len(final_train_list):.1%} of the train list")

# Shuffle deterministically
np.random.seed(42)
np.random.shuffle(final_train_list)

with open(train_oversampled_path, 'w') as f:
    for line in final_train_list:
        f.write(line + "\n")
print(f"\nGenerated {train_oversampled_path.name} with {len(final_train_list)} total training samples.")

# Print and Save Instance Count Comparison
print("\n" + "=" * 70)
print("UPDATED INSTANCE-COUNT-PER-IMAGE AVERAGES")
print("=" * 70)
for cname in ['edge_banding_machine', 'fire_extinguisher']:
    b_arr = stats_summary[cname]['before_instances']
    a_arr = stats_summary[cname]['after_instances']
    print(f"\n[{cname.upper()}] (Total images evaluated: {len(b_arr)}):")
    print(f"  BEFORE MERGE: Mean = {np.mean(b_arr):.2f}, Median = {np.median(b_arr):.1f}, Max = {np.max(b_arr)}")
    print(f"  AFTER MERGE:  Mean = {np.mean(a_arr):.2f}, Median = {np.median(a_arr):.1f}, Max = {np.max(a_arr)}")
eb = stats_summary['edge_banding_machine']
print(f"\n[EDGE_BANDING_MACHINE] Instance height (fraction of image): "
      f"before median = {np.median(eb['before_heights']):.3f}, after median = {np.median(eb['after_heights']):.3f}")

stats_out = {
    cname: {
        'total_images': len(stats_summary[cname]['before_instances']),
        'before_mean': float(np.mean(stats_summary[cname]['before_instances'])),
        'before_median': float(np.median(stats_summary[cname]['before_instances'])),
        'before_max': int(np.max(stats_summary[cname]['before_instances'])),
        'after_mean': float(np.mean(stats_summary[cname]['after_instances'])),
        'after_median': float(np.median(stats_summary[cname]['after_instances'])),
        'after_max': int(np.max(stats_summary[cname]['after_instances']))
    }
    for cname in ['edge_banding_machine', 'fire_extinguisher']
}
stats_out['edge_banding_machine']['before_median_height'] = float(np.median(eb['before_heights']))
stats_out['edge_banding_machine']['after_median_height'] = float(np.median(eb['after_heights']))
stats_out['splits'] = split_counts

with open(DATASET_DIR / "build_9class_stats.json", "w") as f:
    json.dump(stats_out, f, indent=2)

# Negative lists for manual spot-checking (unlabeled target objects would poison training)
with open(DATASET_DIR / "negatives_9class.txt", "w") as f:
    for split in ["train", "valid", "test"]:
        for p in negative_files[split]:
            f.write(rel(p) + "\n")

print("\nPIPELINE COMPLETED SUCCESSFULLY!")
