# TODO — Krish 9-class YOLO-seg: CCTV robustness

*Last updated 2026-10-14. This file is the handoff for the next session.*

## 1. Where things stand

- **Goal:** make the 9-class segmentation model (`panel_saw`, `hydraulic_press`, `table_saw`, `planer`, `jointer`, `band_saw`, `miter_saw`, `edge_banding_machine`, `fire_extinguisher`) work on real factory CCTV.
- **Handoff is on hold.** `run6_9class_cctv_robust` (Box mAP50 0.810 / Mask mAP50 0.795 on val) **regressed** on real footage against `run5_9class_clean_finetune`:
  - On the workshop clip, run5 scores `panel_saw` 0.56–0.63. run6 scores ≤0.18 and detects **nothing at conf 0.40** in all 1,398 frames.
  - **Fine-tune from run5, not run6.**
- **No new training has happened since run6.** Everything since 10-08 was analysis, planning, reorganization and data collection.

## 2. Key findings

| Finding | Detail |
|---|---|
| **Regression on real footage** | See §1. |
| **The workshop saw is not small** | It is 266×208 px on a 480×360 frame, about 55% of the width. The gap is resolution, codec, viewpoint, tint and clutter, not object size. This is why SAHI is deprioritised. |
| **imgsz 1280 hurts on the 480×360 sub-stream** | run5 scores 0.56 at imgsz 960 and 0.27 at 1280 on the same frame. Use imgsz 640 for the sub-stream, or record the main stream. |
| **`albumentations` was missing** | Ultralytics silently skipped its blur, greyscale and compression augmentations in every past run. `erasing` does nothing for segmentation. It is installed in the venv only if you ran the pip command in §6. |
| **Optimizer was AdamW, not SGD** | `optimizer=auto` chose AdamW at lr0 0.000769 and ignored `lr0=0.01`. Always set `optimizer` explicitly. |
| **`iou` is ignored** | run6 is YOLO26 end2end (NMS-free), so `iou=` has no effect unless `end2end=False`. |
| **No desk, ceiling or CCTV data** | All 301 training negatives are other machinery. |
| **Edge banding: labelling problem** | 45% of val labels are frame-filling boxes. Recall is 0% on edge banders under 25% of the image or with 2+ per image. |
| **Fire extinguisher: crowding** | Recall is 86% alone vs 39% in images with 3+; those images hold 67 of 105 instances. |
| **Trial recordings are not clean inputs** | The CCTV tool's overlay is burned into the screen recordings, and both models fire on it. |
| **The "upfront" clips are not evidence about production** | They are framed like catalog photos. Their 96% / 99% / 76% "frames with a detection" figures count true and false positives together. |

## 3. What we did (chronological)

### 2026-10-07 → 10-08: audit and tooling
1. Post-training audit of `9_class_cctv_robust`, with corrections to the task brief.
2. Answered the 10 technical questions in `docs\run_analysis_10q.md`, grounded in the Ultralytics 8.4.123 source and new measurements.
3. Wrote and tested `scripts\test_inference.py`: image, folder or video input, per-class thresholds, `--nms` switch, CSV/JSON/Markdown logs.
4. Re-ran both trial videos with the new checkpoint (`trial_videos\analysis\cctv_robust_inference\`).

### 2026-10-10: CCTV domain-adaptation plan
- Wrote `docs\cctv_domain_adaptation_plan.md`: dataset ratio, labelling rules, why the augmentations failed, SAHI vs static ROI crops, tracking, training recipe and roadmap.
- Recorded the corrections to the original brief (the findings in §2).

### 2026-10-10: workspace reorganized
- The nested `Krish\Krish\` project and the two `runs\` trees were merged into one root. **Nothing was deleted.**
- Full details, the run-name mapping and the rollback log: `CHANGELOG_REORGANIZATION.md` (project root) and `reorg_move_log.csv`.
- Pre-existing breakage fixed along the way:
  - Venv launchers and activate scripts pointed at another machine's path.
  - The 38-class manifest had 9,571 dead absolute paths.
- Verified: val of run6 through the new layout reproduced Box mAP50 0.810 / Mask mAP50 0.795.

### 2026-10-10 → 10-14: CCTV-perspective web scrape (finished)
- **Ran:** `pipeline.stage2_scrape --templates cctv --cap 300` for all 9 classes, as a detached process.
- **Result:** 2,677 raw images, saved as `pipeline\dataset\train\<class>\img_<class>_cctv_*.jpg`:

  | Class | Images |
  |---|---:|
  | `panel_saw` | 278 |
  | `band_saw` | 299 |
  | other 7 classes | 300 each |

- **Not yet cleaned or labelled.**
- **2026-10-15 review: the batch is not CCTV data.** 27% are byte-duplicates, 347 files are duplicated across classes, about 10% are camera hardware, and only about 5 images look like CCTV. See `docs\cctv_scrape_postmortem.md`.
- Log: `pipeline\logs\scrape_cctv_diversity.log`. Summary: `pipeline\logs\stage2_scrape_log_cctv.json`.
- `ddgs` was installed into `yolo-env`.
- **Code changes in `pipeline\pipeline\`:**
  - `config.py`: added `CCTV_QUERY_TEMPLATES` (10 queries) and `TARGET_9_CLASS_NAMES`.
  - `stage2_scrape.py`:
    - New `--templates {default,cctv}` and `--tag` options.
    - CCTV mode relaxes the "cctv"/"surveillance" title filter and blocks camera-product listings instead.
    - Tagged filenames and cap counting, so batches stay separate.
    - Log file written per tag.

## 4. Current layout

```text
Krish\
├── datasets\9class\        data_9class.yaml, manifests, {train,valid,test}_9class (images = junctions → source_38class)
├── datasets\source_38class\
├── runs\                   run1…run6, evals\, inference_tests\
├── scripts\                train.py, test_inference.py, tools\
├── pipeline\               I4I scraping repo (own .git); dataset\train\<class>\ is the scrape intake
├── models\pretrained\
├── docs\                   this file, runsummary.md, run_analysis_10q.md, cctv_domain_adaptation_plan.md
├── archive\                old scripts, aborted runs, unnamed val runs, backups
└── env\yolo-env\           Python 3.10, torch 2.13+cu130, ultralytics 8.4.123, RTX 3070 Ti
```

| Run | Meaning |
|---|---|
| `run1_23class_oversampled`, `run2_23class_finetune_clean` | 23-class era |
| `run3_38class_habitat_full`, `run4_38class_habitat_unified_finetune` | 38-class era |
| `run5_9class_clean_finetune` | 9-class champion; **base for the next fine-tune** |
| `run6_9class_cctv_robust` | Regressed on real footage; do not ship |

## 5. Gotchas

- **Python:** use `env\yolo-env\Scripts\python.exe`. `Krish\instancesegmentationtraining\yolo-env` is only a compatibility junction.
- **Never `Remove-Item -Recurse` a junction.** Use `cmd /c rmdir`. This applies to the three dataset-image junctions and the venv junction.
- **Dataset junctions store absolute targets.** If the project folder moves again, recreate them. The command is in `CHANGELOG_REORGANIZATION.md`.
- **`scripts\train.py` writes to `runs\run7_9class_cctv_pilot`** and uses the plan's Option A recipe on `data_9class_cctv.yaml`. Use `--name` for later runs; the script refuses to overwrite an existing run.
- **Old files keep their original paths.** `args.yaml` files and the older reports are historical records; don't "fix" them.
- **Tool noise:** `test_inference.py --help` returns -1 when piped through `Select -First`. That is the pipe closing, not an error.

## 6. Immediate next steps

- [ ] **Plan Step 0: stop the regression.** Re-test run5 on the workshop clip at imgsz 480 / 640 / 960:
  ```
  cd C:\Users\Tanushree_i4i\Desktop\ckaude\Krish
  env\yolo-env\Scripts\python.exe scripts\test_inference.py --weights runs\run5_9class_clean_finetune\weights\best.pt --source "..\trial_videos\Screen Recording 2026-09-28 124839.mp4" --imgsz 640
  ```
- [ ] **Clean the scraped batch** (2,677 images). Run stage 3 (dedup) and stage 4 (CLIP filter) from `pipeline\`, then inspect a contact sheet (stage 6).
  - **Expect noise.** The "surveillance camera" queries pulled in CCTV-hardware ads on the test query.
  - **Treat this batch as supplementary variety.** It does not replace footage from your own cameras.
- [x] **Install the augmentation package** (2026-10-15): `albumentations` 2.0.8 installed with `--no-deps` to keep the GUI build of `cv2`.
- [x] **run8 preparation done (2026-10-21). Training not started.**
  - **run7:** archived to `archive\run7_cctv_pilot_2026-10\`.
  - **Factory frames:** frozen as a test-only benchmark (`docs\cctv_benchmark.md`).
  - **Labels:** cleaned (`docs\label_policy.md`).
  - **Hard negatives:** 107 targeted ones added (12% negatives in total).
  - **Training config:** `train.py` uses the run8 recipe.
  - **Re-baselined on the CLEANED val:** run5 box 0.789 / mask 0.741. Compare run8 with these numbers, not with 0.82.
  - **CCTV benchmark:** run5 mask50 0.054 (640) / 0.010 (960).
  - **Follow-up (same day, `docs\label_policy.md`):**
    - 17 negatives dropped: 5 with unlabelled target machines, plus all 12 veneer-press images.
    - SAM 2.1 Large is now the annotation default.
    - 30 high-angle web images (52 instances) added to train via `train_highangle_9class.txt`.
    - New train list: 3,373 entries, 11.6% negatives.
    - Audit: PASS.
    - run5 on the current val: unchanged at 0.789 / 0.741 (box/mask mAP50), 0.665 / 0.632 (mAP50-95).
  - **Open decisions:**
    - relabel the 12 veneer-press images as `hydraulic_press`;
    - hand-draw high-angle #5 and #94 (SAM failed on both).
  - **Next:** human re-annotation of the 64 quarantined images (optional), then `env\yolo-env\Scripts\python.exe scripts\train.py`.
- [x] **run7 pilot trained and analysed** (2026-10-19, `docs\run7_analysis_report.md`). **Not a champion; keep run5.**
  - **Training camera (cam05):** fits it at 100% / 0.97.
  - **Unseen cameras:** Camera 03 and the workshop clip get 0 detections.
  - **Catalog:** box mAP50 0.821 → 0.79 at imgsz 960.
  - **False alarms:** `fire_extinguisher` false alarms on Camera 03 went from 0 to 51–74.
  - **Cause:** one viewpoint per class, repeated 5×, so the model memorised the scene.
  - **Next:** get ≥ 5 viewpoints per class, use 1–2× repeats, add cylinder hard negatives, fix the panel_saw/table_saw label boundary.
- [x] **Factory labels finalised; hybrid dataset built** (2026-10-19, report §4.5).
  - **Split:** swapped, cam05 → train and cam03 → val.
  - **Labels:** SAM 2.1 auto-labels with workers removed, spot-checked but not hand-drawn.
  - **Dataset:** `data_9class_cctv.yaml` with shares 56/44/0; 0 corrupt.
  - **Next:** optional human QA in CVAT, hard-negative crops, then `scripts\train.py`.
- [x] **Factory footage analysed and staged** (2026-10-19). See `docs\factory_footage_analysis.md`.
  - **Footage:** the 5 files come from **2 cameras** (cam03, cam05) and are stored squeezed 2:1 (960×1080 "1080P Lite").
  - **Frames:** 328 in `datasets\cctv_factory_intake\`, with run5 pre-labels in `_prelabels\`.
  - **Split and balance:** cam03 → train, cam05 → val gives a projected 81/19/0.
  - **Next:** decide the split direction (§4.2), label in CVAT, then run the builder without `--preview`.
- [x] **CCTV ingestion pipeline ready** (2026-10-15). The sampler, pre-labeller and camera-split hybrid builder are new, and `train.py` now uses the run7 pilot recipe. All were smoke-tested. See `docs\cctv_ingestion_runbook.md`. Live RTSP is still untested.
- [ ] **Audit training negatives and labels for the panel-saw regression.** Run the model over all training images at low confidence and review every hit:
  ```
  env\yolo-env\Scripts\python.exe scripts\test_inference.py --source datasets\9class\train_9class\images --conf 0.20 --save-dir runs\inference_tests\audit_train
  ```
  For each hit in `detections.csv`, decide:
  - **Missing label** (a real target machine) → add the label.
  - **Look-alike** (spindle moulder, veneer press, CNC router, racking) → keep as a negative or drop, per the class policy.
  - **Real false positive** → keep as a hard negative.

## 7. Then (follows `cctv_domain_adaptation_plan.md`)

**Data**
- [ ] **Step 1:** record **raw RTSP main stream** (no overlay) for ≥ 24 h per camera, across shifts and IR night. Take a camera inventory, and shoot pseudo-CCTV phone video (3–4 m, pitched down) of machines no camera covers.
- [ ] **Step 2:** build CCTV-val, CCTV-test and a hard-negative set (desks, ceilings, bars, empty floors) **before any training**. Split by camera, never by frame.
- [ ] Re-record the trial clips with the CCTV tool's overlay turned **off**.

**Labels**
- [ ] Write a class-boundary policy for look-alikes, with example images (e.g. sliding-table saw = `panel_saw`; spindle moulder = negative).
- [ ] Re-annotate all 312 edge-banding instances: one extent policy, no frame-filling polygons, separate instances.
- [ ] Label every extinguisher in crowded images.
- [ ] Label CCTV frames per the plan's §2 rules: visible pixels only, ≥ 40% visible, ≥ 16 px shorter side, 10% double-labelled for QA.

**Training**
- [ ] **Pilot (Option A):** fine-tune run5 on a hybrid list (~55% catalog / 35% CCTV / 10% hard negatives).
  - **Settings:** `optimizer=AdamW`, `lr0=2e-4`, `cos_lr`, 30 epochs, `scale=0.5`, `degrees=0`, no frozen layers, plus the plan's `cctv_aug` list.
  - **Selection:** validate on CCTV-val and select the checkpoint on mask mAP50-95.
  - **Learning curve:** train on 25% / 50% / 100% of the CCTV frames to see whether more data is still helping.
- [ ] **Release (Option B):** retrain from COCO weights after the labelling fixes land.
- [ ] Compare against run5 on the trial clips before any handoff.

**Inference and deployment**
- [ ] Test `--nms` (one-to-many head + NMS) on crowded extinguisher scenes.
- [ ] Per-camera ROI crops from the main stream. Skip SAHI unless targets are under ~40 px at imgsz 1280.
- [ ] 5 fps decode with ByteTrack at conf 0.10 and an 8-of-15 confirmation window. Per-class thresholds come from CCTV-val.

**Reporting**
- [ ] Export per-class mAP50-95: `yolo segment val model=runs\run6_9class_cctv_robust\weights\best.pt data=datasets\9class\data_9class.yaml imgsz=1280`.
- [ ] Release gates are in the plan's §5.3:
  - CCTV-test mask mAP50 ≥ 0.60 and recall ≥ 0.80.
  - ≤ 1 false positive per camera-hour.
  - Catalog val within 2 points of run5.

## 8. Files to read first

| File | What it holds |
|---|---|
| `docs\cctv_domain_adaptation_plan.md` | The full CCTV plan: ratios, labelling rules, augmentations, recipe, roadmap, release gates |
| `docs\run_analysis_10q.md` | Answers to the 10 technical questions, with source citations |
| `docs\runsummary.md` | Post-training audit of run6, including the trial-video re-test (§8) |
| `CHANGELOG_REORGANIZATION.md` | What moved where, the run-name mapping, rollback and re-link instructions |
| `scripts\test_inference.py` | Inference and test harness |
| `trial_videos\analysis\` | Trial-video diagnostics and the old-vs-new checkpoint comparison |
