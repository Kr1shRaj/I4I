# Run Summary — `9_class_cctv_robust` (YOLO-seg, 9 classes)

**Report date:** 2026-10-07 (revised 2026-10-08: trial-video re-test, end2end/NMS finding, training-negative audit; see §8 and `run_analysis_10q.md`)  
**Run directory:** `Krish\runs\segment\9_class_cctv_robust\`  
**Weights evaluated:** `weights/best.pt` (best epoch = 50 = final epoch, so `best.pt` and `last.pt` are the same checkpoint)

> **Sources.** Every figure here comes from the run artifacts: `results.csv`, `args.yaml`, the PR/F1/P/R curve plots, both confusion matrices, `val_batch{0,1,2}_{labels,pred}.jpg`, and `valid_9class.txt`.
> Per-class precision and recall are **read off the P/R-confidence curves** at the F1-optimal threshold, so they are approximate (±0.02). The column means match the `results.csv` totals to within 0.003.
> Per-class mAP50-95 is not exported by this run. You can get it with `model.val(...)` (see §6).

---

## 0. Corrections to the task brief

Several facts in the brief disagree with the artifacts. This report uses the artifact values.

| Brief states | Artifacts show | Source |
|---|---|---|
| `perspective=0.001`, `scale=0.7` | `perspective=0.0005`, `scale=0.9` (also `degrees=10`, `translate=0.2`, `erasing=0.4`, `hsv_s/v=0.5`) | `args.yaml` |
| Box precision 88.1%, Mask precision 86.5% | Best/final epoch 50: **Box P 87.0%, Mask P 86.3%**. 88.0% was epoch 49, and no epoch logged 86.5% mask P. | `results.csv` |
| "Top performers (>85% mAP50)" includes Panel Saw, Planer, Band Saw | Box mAP50 is **84.8% / 84.6% / 84.0%**, just under 85%. Mask mAP50 is 82.0% / 84.6% / 83.0%. | `BoxPR_curve.png`, `MaskPR_curve.png` |
| F1-optimal `conf ≈ 0.50` | F1 peaks at **conf = 0.388 (box)** and **0.366 (mask)**, both at F1 = 0.78 | `BoxF1_curve.png`, `MaskF1_curve.png` |
| 63 negatives are "empty workshop floors, office desks, ceilings, overhead bars" | The 63 negatives are **other industrial equipment** from the 38-class set: pallet jacks (8), injection moulders (5), forklifts, glue spreaders, robotic arms (4 each), CNC mills/routers, control panels, grinders, mortisers, hoists, spindle moulders, wood lathes (3 each), and others. **None are desks, ceilings, floors, or CCTV frames.** | `valid_9class.txt` lines 420–482 + filename prefixes |
| 95% CI describes "real-world unseen factory deployment" | The SE and CI arithmetic is correct, but it measures **spread across the 9 classes**, not sampling uncertainty of the overall score or transfer to CCTV. See §3. | computed from per-class AP |
| *(added 2026-10-08)* Training added desk/ceiling negatives | The **training** list has no desk, ceiling, office or CCTV images either. Its 301 negatives (9.1% of entries) are all other machinery. | `train_oversampled_9class.txt` filename scan |
| *(added 2026-10-08)* `iou=0.45` controls NMS at inference | This checkpoint is a **YOLO26 end2end (NMS-free)** model, so `iou` has **no effect**. Use `end2end=False` to get the one-to-many head + NMS. | `test_inference.py` model report; `ultralytics/utils/nms.py:66` |

---

## 1. Executive Summary

**Verdict (revised 2026-10-08): do not hand off as a replacement for the previous checkpoint yet.**
- On validation images it is a reasonable 7-class machine detector.
- **The trial-video re-test shows a regression on real footage.** On identical frames of the workshop clip, the previous checkpoint scored the machine as `panel_saw` at 0.56–0.63. This one scores ≤0.18 and detects **nothing at conf 0.40 in all 1,398 frames** (§8).
- The two stated objectives remain **unverified**. Neither train nor val contains desk, ceiling or CCTV images, so this run could not have targeted them directly.
- *(Original 2026-10-07 verdict, based on validation artifacts only: conditional pass for handoff as a 7-class machine detector.)*

| Metric (epoch 50, val, 482 imgs / 490 instances) | Box | Mask |
|---|---|---|
| mAP50 | **81.0%** | **79.5%** |
| mAP50-95 | 64.9% | 65.4% |
| Precision (at F1-optimal conf) | 87.0% | 86.3% |
| Recall (at F1-optimal conf) | 71.8% | 72.1% |
| Peak F1 (conf) | 0.78 (0.388) | 0.78 (0.366) |

- **Training:**
  - 50/50 epochs, 10,764 s (**2 h 59 min**).
  - `imgsz=1280`, `batch=4`, AMP.
  - Fine-tuned from `9_class_clean_finetune/best.pt`.
  - Early stopping (`patience=15`) never triggered.
- **Improvement over starting checkpoint (epoch 1 → 50):**
  - Box mAP50: 72.3 → 81.0 (+8.7 pt)
  - Mask mAP50: 70.9 → 79.5 (+8.6 pt)
  - Mask mAP50-95: 57.1 → 65.4 (+8.3 pt)
- **Strong:** hydraulic press, table saw, jointer (Box AP50 87–96%).
- **Weak:**
  - edge_banding_machine: AP50 58.4%, and 53% of real instances missed.
  - fire_extinguisher: AP50 68.0%, and 46% missed.
- **Deployment suitability:**
  - ~~Fine for supervised pilot use on the seven machine classes.~~ Withdrawn after the §8 trial-video regression. Pilot only alongside the previous checkpoint, as an A/B comparison.
  - **Not ready as a sole safety signal for fire-extinguisher presence or absence.**
  - Not ready to be called "CCTV-robust" until it is tested on labelled CCTV frames.

---

## 2. Quantitative Performance Breakdown

### 2.1 Per-class metrics (epoch 50, `best.pt`)

Sorted by Box mAP50. Precision and recall are read from the curves at conf 0.388 (box) and 0.366 (mask).

| Class | GT inst. | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hydraulic_press | 58 | ~0.93 | ~0.88 | **0.961** | n/e | ~0.89 | ~0.85 | **0.916** | n/e |
| table_saw | 41 | ~0.94 | ~0.83 | **0.931** | n/e | ~0.91 | ~0.81 | **0.904** | n/e |
| jointer | 39 | ~0.80 | ~0.80 | **0.869** | n/e | ~0.80 | ~0.81 | **0.869** | n/e |
| panel_saw | 53 | ~0.95 | ~0.75 | 0.848 | n/e | ~0.94 | ~0.77 | 0.820 | n/e |
| planer | 35 | ~0.87 | ~0.80 | 0.846 | n/e | ~0.86 | ~0.80 | 0.846 | n/e |
| band_saw | 64 | ~0.83 | ~0.78 | 0.840 | n/e | ~0.82 | ~0.78 | 0.830 | n/e |
| miter_saw | 46 | ~0.81 | ~0.70 | 0.733 | n/e | ~0.82 | ~0.72 | 0.724 | n/e |
| fire_extinguisher | 105 | ~0.88 | ~0.52 | 0.680 | n/e | ~0.92 | ~0.55 | 0.693 | n/e |
| edge_banding_machine | 49 | ~0.81 | ~0.41 | **0.584** | n/e | ~0.81 | ~0.42 | **0.558** | n/e |
| **all (mean)** | **490** | **0.870** | **0.718** | **0.810** | **0.649** | **0.863** | **0.721** | **0.795** | **0.654** |

- GT instance counts are the column sums of `confusion_matrix.png`.
- n/e = not exported by the training run (needs `model.val()`).
- Overall P and R come from `results.csv`.

### 2.2 Tier ranking (Box mAP50; corrected thresholds)

| Tier | Classes |
|---|---|
| **Top (>85%)** | hydraulic_press 96.1, table_saw 93.1, jointer 86.9 |
| **Solid (80–85%)** | panel_saw 84.8, planer 84.6, band_saw 84.0 |
| **Moderate (70–80%)** | miter_saw 73.3 |
| **Borderline (<70%)** | fire_extinguisher 68.0, edge_banding_machine 58.4 |

### 2.3 Root causes for the bottom performers

**edge_banding_machine (AP50 58.4%, recall ≈ 41%)**
- The confusion matrix shows **26 of 49 instances missed entirely** (predicted as background), and **zero confusions with other classes**. The model doesn't mistake these machines for something else; it fails to fire at all.
- The P-confidence curve is unstable above conf 0.6, dipping to 0.50 at 0.90. The class has a few confidently wrong boxes, consistent with **label/extent noise**.
- ~~Likely cause: edge banders are long, low, enclosed boxes with very different extents across images.~~ **Measured cause (2026-10-08): the labelling convention.**
  - **45% of val edge-banding labels are frame-filling boxes** (>95% of the image), against 0–7% for every other class.
  - Recall is **0/15 for edge banders under 25% of the image** and **0/18 when two or more share an image**.
  - The model learned "edge bander = the whole photo". Re-annotation comes before adding data. See `run_analysis_10q.md` Q3.
- The F1 peak is at **conf 0.16** and falls off fast, so any global threshold above about 0.3 suppresses most of this class.

**fire_extinguisher (AP50 68.0%, recall ≈ 52%)**
- **48 of 105 instances missed**, the largest absolute miss count of any class.
- The class has 105 instances in only 45 images, about 2.3 per image.
- *Measured 2026-10-08:*
  - Crowding is the main driver. Recall is 86% when an extinguisher is alone, but **39% in images with 3 or more**, which hold 67 of the 105 instances.
  - Relative size also matters: **32% recall below 5% of the image**.
  - Absolute size is not the problem on val: the median long side is 550 px at 1280.
- Precision is high (about 0.88–0.92), so detections are trustworthy. The failure is recall on small or atypical views.
- This matches the earlier trial-video finding: handheld or close-up extinguishers are detected, while small ceiling-CCTV extinguishers are missed. That earlier finding came from the session's `analysis/extinguisher_degradation_sweep.png`.

**miter_saw (AP50 73.3%)**
- 13 of 46 missed (28%).
- The only cross-class leak is 1 miter_saw predicted as band_saw.
- Compact benchtop saws are visually similar to the small benchtop table and band saws in the training data.

### 2.4 Cross-class confusion

Confusion between classes is negligible: only **5 of 361 matched predictions** went to the wrong class.

| Predicted → | as | count |
|---|---|---:|
| planer | jointer | 2 |
| hydraulic_press | planer | 1 |
| band_saw | planer | 1 |
| miter_saw | band_saw | 1 |
| panel_saw ↔ table_saw | — | **0** |

The jointer/planer pair is the only systematic confusion, at 6% of planer instances. The main error mode is **missed detections (FN to background), not class swaps**.

### 2.5 Convergence audit

| Loss | Epoch 1 | Epoch 50 | Trend |
|---|---:|---:|---|
| train/box | 0.957 | 0.547 | ↓ steady |
| train/seg | 1.714 | 0.777 | ↓ (step down at ep 41 when mosaic closed) |
| train/cls | 1.447 | 0.398 | ↓ |
| val/box | 1.091 | 0.824 | ↓ (min 0.824 at ep 50) |
| val/cls | 1.225 | 0.940 | ↓ |
| **val/seg** | **1.345** | **1.438** | **flat / slightly ↑** (range 1.34–1.65) |

- **Box and cls heads converged without overfitting.** Validation loss was still falling at epoch 50.
- **Mask head:** train seg loss halved while validation seg loss did not improve. This shows the mask head memorising training polygon detail that doesn't transfer, a mild overfit or label-noise ceiling. Mask mAP still rose (70.9 → 79.5), so instance-level mask quality improved even though pixel-level mask loss did not.
- **Plateau:** mAP50 gained only +0.5 pt over epochs 40–50. More epochs at this learning rate would add little. More, cleaner data would help more.

---

## 3. Statistical & Operational Confidence Analysis

### 3.1 Statistical 95% interval

The figures below are the mean of the 9 per-class APs, ± 1.96 × (SD across classes / √9).

| Metric | Mean | SD across classes | SE | 95% CI (z) | 95% CI (t₈ = 2.306) |
|---|---:|---:|---:|---|---|
| Box mAP50 | 81.0% | 12.2 pt | 4.1 pt | 73.1% – 89.0% | 71.7% – 90.4% |
| Mask mAP50 | 79.5% | 11.6 pt | 3.9 pt | 71.9% – 87.1% | 70.6% – 88.5% |

**What this does and doesn't mean:**
- The brief's numbers are arithmetically correct. With only 9 classes, the t-interval (right column) is the more appropriate one.
- This interval describes **how much performance varies between machine types**. It does not describe how much the overall score would move on a new set of factory images.
- Plainly: a typical class in this model scores around 81%. Individual classes range from about 58% to 96%, and the ±8 pt band summarises that spread.
- **It says nothing about CCTV.** All three validation sheets are product, catalogue, and web photos: white backgrounds, eye-level, close framing. In ceiling-CCTV footage, objects are smaller, viewed from above, and compressed. Expect real-site AP to fall **below** this interval until a labelled CCTV test set shows otherwise.

### 3.2 Operating threshold

The aggregate F1 curve is **flat at 0.77–0.78 from conf 0.20 to 0.50**. It drops to about 0.75 at 0.60, 0.70 at 0.70, and 0.65 at 0.80.

| Setting | Value | Rationale |
|---|---|---|
| F1-optimal | **0.39** (box) / **0.37** (mask) | Measured peak |
| **Recommended default** | **`conf = 0.40`** | On the F1 plateau; overall P ≈ 0.87, R ≈ 0.72 |
| Operating window | **0.30 – 0.55** | F1 stays ≥ 0.76 throughout |
| Clean, well-lit, eye-level cameras | 0.45 – 0.55 | Precision ≈ 0.89–0.91, small recall cost |
| Cluttered, poorly lit, or overhead CCTV | 0.25 – 0.35, **plus temporal confirmation** (§6) | Recall matters more; let the tracker filter one-frame false positives |

**Trade-offs:**
- **Below 0.30:** precision falls from 0.84 to about 0.70 at 0.10. Phantom detections rise, mostly on band_saw, jointer, and miter_saw, which already have 9–13 background false positives each at conf 0.25.
- **Above 0.55–0.65:** the classes that matter most collapse.
  - edge_banding_machine recall falls from 0.38 at conf 0.45 to 0.24 at 0.60 and 0.18 at 0.70.
  - fire_extinguisher recall falls to about 0.46 at 0.60 and 0.40 at 0.70.
  - Box precision barely improves (0.93 at 0.60).
- **So the brief's `conf=0.50` is acceptable but already costs edge-banding recall. A 0.65 ceiling is too high for this model.**

**Per-class override (recommended if the pipeline allows it):**

| Class | Threshold |
|---|---|
| edge_banding_machine | 0.20 (F1 peak 0.16) |
| fire_extinguisher | 0.30 (F1 plateau 0.15–0.45) |
| All other classes | 0.40 |

---

## 4. Background & False-Positive Suppression

### 4.1 What the artifacts can and can't show

The confusion matrix (conf 0.25, IoU 0.45) has **67 predictions with no matching ground truth** ("background" column).

| Class | FP to background |
|---|---:|
| band_saw | 13 |
| fire_extinguisher | 12 |
| jointer | 9 |
| miter_saw | 9 |
| hydraulic_press | 7 |
| edge_banding_machine | 6 |
| panel_saw | 5 |
| table_saw | 3 |
| planer | 3 |

- These 67 are pooled over **all 482 images**. They include duplicate boxes and unlabelled real machines in positive images (for example, the six-saw collage in `val_batch0` where only one saw is labelled). The training plots can't separate how many fell on the 63 negative images.
- Precision at the confusion-matrix threshold: 356 TP / (356 + 5 + 67) = **83.2%**.
- At the F1-optimal threshold: **87.0% (box) / 86.3% (mask)**. Both clear the >85% target, but they measure **precision over all predictions**, not suppression on negative images specifically.

### 4.2 Visual evidence

- `val_batch0_pred.jpg` contains 3 negatives: 2 veneer/cold presses and 1 wood-lathe workshop interior.
  - **No detections on any of them.** The veneer presses are hydraulic-press-like, so this is meaningful hard-negative suppression.
  - The workbench full of tools and furniture in `table_saw_00340` produced **only** the correct small table-saw insert box (conf 0.3), with no box on the surrounding workbench.
- **No office desk, ceiling, overhead bar, monitor, or floor image exists in the validation set** (§0). The specific false positives this run was meant to remove (desk → edge_banding_machine, overhead bar → panel_saw) are **untested**.

### 4.3 Verdict

| Check | Result |
|---|---|
| Hard-negative machinery (presses, lathes, forklifts, CNC, etc.) | ✅ Suppressed well (precision 87%; no class leakage) |
| Desk / ceiling / overhead-bar / screen false positives | ⚠️ **Not measured.** Needs a dedicated negative test set |

---

## 5. Segmentation Mask Quality

### 5.1 Coverage (half-machine / SAM tabletop truncation)

The `val_batch*_pred.jpg` sheets were compared with their `_labels.jpg` counterparts.

- **Full-body coverage achieved on validation images.** Masks on freestanding saws include the cabinet, base, legs, and extension wings, not just the table surface:
  - Harvey, King (both views), JET, Grizzly G0651, Ingco folding stand, NS-Machinery stand saw, Artisan site saw, red LH-CNC saw.
- **Not reproduced in validation:** the "top surface only" truncation seen in the workshop trial video (`Screen Recording 124839`, frame 1303). The labels in these sheets are full-body polygons, and the model reproduces them.
- **Trial-video machine (re-tested 2026-10-08):** the truncation question is moot. This checkpoint no longer detects the machine at all at conf 0.40 (§8).

### 5.2 Edge smoothness and failure patterns

| Observation | Example | Severity |
|---|---|---|
| Clean, tight contours on product-style photos | most of `val_batch1/2` | ✅ good |
| **Mask bleed** onto adjacent same-coloured machinery and floor | `table_saw_ddgs_00350` (grey sliding saw: mask leaks left onto red/blue machines behind) | moderate |
| **Speckled holes / islands** inside large uniform regions | `table_saw_00338` (white cabinet saw: brown flecks where doors open) | moderate |
| Ragged right edge on partly occluded machine | `table_saw_ddgs_00411` (conf 0.3) | minor |

At `imgsz=1280` and `mask_ratio=4`, the prototype mask is 320 px across. Edges are visibly sharper than in the earlier trial-video renders. Remaining jaggedness on video is mostly a display-pipeline issue (polygon simplification and crop padding), as diagnosed in the earlier session's `smooth_mask_inference.py`.

### 5.3 Mask vs box disparity

| | mAP50 | mAP50-95 |
|---|---:|---:|
| Box | 81.0 | 64.9 |
| Mask | 79.5 | **65.4** |

- The mask/box gap at IoU 0.5 is only 1.5 pt, and at mAP50-95 **masks score higher than boxes**. When the model finds an object, its mask fits tightly.
- Most of the gap is on panel_saw (−2.8 pt), edge_banding (−2.6), and hydraulic_press (−4.5). These are long or open-frame machines, where thin rails, fences, and frame members are hard to segment.
- fire_extinguisher masks score **higher** than boxes (+1.3 pt). Compact convex shapes suit mask IoU.

---

## 6. Deployment & Next-Step Recommendations

### 6.1 Inference configuration

```python
from ultralytics import YOLO

model = YOLO(r"...\9_class_cctv_robust\weights\best.pt")
results = model.predict(
    source=source,
    imgsz=1280,          # trained resolution; for 480-line CCTV, 960 gives the same accuracy at ~half the latency
    conf=0.40,           # F1 plateau (0.30–0.55 safe window); brief's 0.50 is acceptable but costs edge-banding recall
    iou=0.45,            # NO EFFECT on this end2end (NMS-free) checkpoint; only applies with end2end=False
    # end2end=False,     # optional: one-to-many head + NMS at `iou`; test on crowded extinguisher scenes
    retina_masks=True,   # full-resolution masks -> smooth contours on video
    max_det=50,
)
```

Per-class thresholds can be applied after prediction:

```python
CLASS_CONF = {"edge_banding_machine": 0.20, "fire_extinguisher": 0.30}   # others 0.40
keep = [b for b in r.boxes if float(b.conf) >= CLASS_CONF.get(r.names[int(b.cls)], 0.40)]
```

To run predictions at `conf=0.20` and then filter, set `conf=0.20` in `predict()`.

### 6.2 Live-CCTV risk register

| # | Risk | Evidence | Mitigation |
|---|---|---|---|
| 1 | **Domain gap.** Validation is catalogue/web photos; deployment is ceiling CCTV. | val sheets; §3.1 | Build a 200–400-frame **labelled CCTV test set** from your own cameras and report mAP on it before calling the model CCTV-robust |
| 2 | Desk / ceiling-bar false positives untested | §4 | Add office/ceiling/floor/monitor negatives to **val and test**, not just train. Track FP-per-hour on 1 h of empty-scene footage |
| 3 | Fire-extinguisher misses (46% FN), worse when small or overhead | confusion matrix; prior trial video | Min-size-aware handling; add overhead/small extinguisher labels; never use absence of a detection as a safety alarm |
| 4 | Edge-banding misses (53% FN) | confusion matrix | Per-class conf 0.20; relabel with consistent extents; add CCTV views |
| 5 | Single-frame flicker | low-conf band 0.3–0.5 | **Tracker + temporal vote**: ByteTrack/BoT-SORT, accept a track after 3 of 5 frames, keep it for N frames after loss |
| 6 | Tiny spurious boxes | FP column | **Min box area filter**: drop boxes < 0.3% of frame area (about 32×32 px at 640×480) for machine classes; keep extinguishers ≥ 16×16 px |
| 7 | Mask bleed / holes on uniform surfaces | §5.2 | Post-process: largest connected component + hole fill + light morphological close; `retina_masks=True` |
| 8 | Jointer↔planer swap (6%) | confusion matrix | Accept, or merge to "jointer/planer" if the downstream use doesn't need the distinction |

### 6.3 Before the next training run

1. Run full validation once to export per-class mAP50-95 (missing from this report):
   `yolo segment val model=...\best.pt data=...\data_9class.yaml imgsz=1280 conf=0.001 iou=0.7 plots=True`
2. ~~Re-run the trial videos with this checkpoint.~~ Done 2026-10-08. See §8: regression on the machine clip.
3. Fix the validation split so it tests the stated objectives: add desk/ceiling/overhead-bar negatives and CCTV-view positives.
4. Clean edge_banding_machine and fire_extinguisher labels (consistent extents; label every extinguisher in multi-object images).

---

## 7. Handoff Notes for Downstream Team

**Strengths**
- **Model:** `best.pt` (YOLO-seg, 9 classes, epoch 50, 2 h 59 min fine-tune at 1280 px).
- **Validation (web/catalogue imagery):**
  - Box mAP50 **81.0%**, Mask mAP50 **79.5%**, Mask mAP50-95 **65.4%**.
  - Precision **87%**, recall **72%** at conf 0.40.
- **Reliable classes:** hydraulic_press, table_saw, jointer, panel_saw, planer, band_saw (AP50 84–96%).
- Almost no cross-class confusion (5 of 361). Errors are almost all misses, not mislabels.
- Masks cover the whole machine (base, legs, wings) on validation images. At mAP50-95, masks fit as tightly as boxes or tighter.
- Hard-negative industrial equipment (presses, lathes, forklifts, CNC) does **not** trigger detections.

**Boundaries**
- **Not validated on CCTV.** Expect lower accuracy on overhead or low-resolution cameras.
- **Desk / ceiling-bar false positives:** not measured in this run. Monitor during pilot.
- **fire_extinguisher:** detections are trustworthy (P ≈ 0.9), but about half are missed. **Do not use "no extinguisher detected" as a compliance alert.**
- **edge_banding_machine:** misses about half. Use conf 0.20 for this class.
- **miter_saw:** moderate (AP50 73%).
- Masks can bleed onto adjacent same-coloured equipment or show small holes on large flat panels. Apply largest-component + hole-fill if area measurements matter.

**Load with:** `imgsz=1280, conf=0.40, retina_masks=True`, plus a tracker with 3-of-5-frame confirmation.
- `iou` is ignored: the model is end2end and NMS-free.
- **Hold the handoff** until the §8 regression is resolved.

---

## 8. Trial-Video Re-test (added 2026-10-08)

Run with `Krish\test_inference.py`. Outputs are in `trial_videos\analysis\cctv_robust_inference\`:
- annotated MP4s
- `detections.csv`, `frames.csv`, `summary.md`
- `old_vs_new_checkpoint.json`

**Workshop machine: `Screen Recording 124839`** (1,398 frames). Input: the camera view cropped and restored to the 480×360 stream, as in the earlier session.

| Frame | Old `9_class_clean_finetune` (imgsz 960) | New `9_class_cctv_robust` (imgsz 960) | New (imgsz 1280) |
|---|---|---|---|
| 186 | **panel_saw 0.56** | fire_extinguisher 0.20 (spurious), table_saw 0.15, panel_saw 0.11 | fire_extinguisher 0.30, panel_saw 0.18 |
| 1210 | **panel_saw 0.61** | table_saw 0.16, jointer 0.13 | panel_saw 0.07 |
| 1303 | **panel_saw 0.63** | fire_extinguisher 0.22, table_saw 0.14 | fire_extinguisher 0.31, panel_saw 0.13 |

- Full-clip run at conf 0.40: **0 detections in 1,398 frames**.
- The machine is a sliding-table panel saw in a cluttered workshop. The new model's suppression of it is consistent with the hard-negative mix. Training negatives include spindle moulders, veneer presses, CNC routers and racking, which are long, flat, table-and-beam machinery. See `run_analysis_10q.md` Q9–Q10.

**Fire extinguisher: `Screen Recording 125921`** (649 frames).
- At conf 0.40 there were 65 detections in 44 frames, all within frames 395–524.
- **41 of the 65 are frame-filling boxes**, fired on the near-black fade-in where only the CCTV tool's own drawn polygon outline is visible.
- **The old checkpoint does the same on those frames** (0.58–0.66 frame-filling boxes). This is a recording artifact: the tool's overlay is burned into the video. It is not a regression.
- On the clear close-up frames 530–649, neither checkpoint reaches 0.40.
- **This clip is not a valid test input.** Record trial clips with the overlay turned off.

**Latency:** RTX 3070 Ti, FP32, imgsz 1280, `retina_masks=True`.
- Inference 12–16 ms per frame.
- End-to-end 17–22 ms per frame, including mask rendering: 45–60 FPS.
