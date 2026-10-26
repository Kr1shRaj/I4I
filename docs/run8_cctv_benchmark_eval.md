# run8 on the frozen factory CCTV benchmark (2026-10-26)

**Bottom line:** run8 does **not** transfer to the factory cameras any better than run5.

- **Raw scores:** box mAP50 drops by 2.8 points. Mask mAP50 rises from 0.010 to 0.025, but both are close to zero.
- **Recall:** at the operating threshold (conf 0.40), both models detect none of the 913 labelled machines.
- **Real gain:** false positives. run5 draws whole-frame `edge_banding_machine` boxes over the hall, and run8 has stopped doing that. Detections of classes that aren't there fall from 34 to 8 at conf 0.25.

The benchmark was used only for evaluation. Nothing was trained, tuned or mined on it.

## Setup

| | |
|---|---|
| Models | `runs\run8_9class_curated_cctv\weights\best.pt` (50 epochs, finished); baseline `runs\run5_9class_clean_finetune\weights\best.pt`. Both are end2end (NMS-free) |
| Data | `datasets\9class\data_9class_cctv_benchmark.yaml`: 328 frames, 913 instances (`panel_saw` 257, `hydraulic_press` 328, `edge_banding_machine` 328). Camera 03 has 71 frames, Camera 05 has 257 |
| Settings | imgsz 960, batch 8, device 0 (RTX 3070 Ti), split `val`. mAP uses the Ultralytics default conf 0.001 |
| Script | `scripts\tools\eval_cctv_benchmark.py` (rerun with `env\yolo-env\Scripts\python.exe scripts\tools\eval_cctv_benchmark.py`) |
| Outputs | `runs\eval_run8_cctv_benchmark\{run8,run5}\`: `val\` (plots, confusion matrix), `val_cam03\`, `val_cam05\`, `metrics.json`, `fp_crops\`; `summary.json` |

A check on the method: run5's mask mAP50 comes out at 0.0098, which matches the 0.010 recorded earlier in `docs\todo.md`.

## 1. Overall results

| Metric | run5 | run8 | Δ (run8 − run5) |
|---|---:|---:|---:|
| Box mAP50 | 0.241 | 0.213 | **−0.028** |
| Box mAP50-95 | 0.148 | 0.125 | **−0.023** |
| Mask mAP50 | 0.010 | 0.025 | +0.015 |
| Mask mAP50-95 | 0.001 | 0.004 | +0.003 |

Ultralytics averages these over the 3 classes that appear in the benchmark, not all 9. The 6 absent classes have no ground truth and therefore no AP. They are covered by the false-positive counts in §4.

## 2. Per class

### All 328 frames

| Class | GT | run5 box 50 / 50-95 | run8 box 50 / 50-95 | run5 mask 50 / 50-95 | run8 mask 50 / 50-95 |
|---|---:|---|---|---|---|
| panel_saw | 257 | **0.723** / 0.444 | 0.640 / 0.375 | 0.030 / 0.004 | **0.074** / 0.012 |
| hydraulic_press | 328 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| edge_banding_machine | 328 | 0.0003 / 0.0001 | 0.00001 / 0.00001 | 0 / 0 | 0 / 0 |
| table_saw, planer, jointer, band_saw, miter_saw, fire_extinguisher | 0 (absent) | n/a, see §4 | n/a, see §4 | | |

### Per camera (box mAP50 / mask mAP50)

| Camera | Class | run5 | run8 |
|---|---|---|---|
| cam03 (71 frames) | hydraulic_press | 0 / 0 | 0 / 0 |
| | edge_banding_machine | 0.005 / 0.0001 | 0.0002 / 0 |
| | **mean** | 0.002 / 0.0001 | 0.0001 / 0 |
| cam05 (257 frames) | panel_saw | **0.770** / 0.030 | 0.660 / **0.077** |
| | hydraulic_press | 0 / 0 | 0 / 0 |
| | edge_banding_machine | 0 / 0 | 0 / 0 |
| | **mean** | 0.257 / 0.010 | 0.220 / 0.026 |

### Recall at the operating threshold

The operating threshold is conf 0.40, the default in `scripts\test_inference.py`. A detection counts as a match when it has the same class and box IoU ≥ 0.5.

| Class | run5 @0.40 | run8 @0.40 | run5 @0.25 | run8 @0.25 |
|---|---|---|---|---|
| panel_saw (257) | 0 | 0 | 1 (0.4%) | 5 (1.9%) |
| hydraulic_press (328) | 0 | 0 | 0 | 0 |
| edge_banding_machine (328) | 0 | 0 | 0 | 0 |

The panel saw's box mAP50 of 0.64–0.72 does not mean the saw is found in practice. The `panel_saw` boxes rank well but score below about 0.05. At conf 0.40 the model finds nothing.

## 3. Latency (RTX 3070 Ti, imgsz 960)

| | run5 | run8 |
|---|---:|---:|
| Batch 1, end-to-end wall time (100 frames, after 10 warm-up) | 14.4 ms → **69 FPS** | 14.3 ms → **70 FPS** |
| of which pre / inference / post | 2.3 / 11.7 / 0.2 ms | 2.3 / 11.6 / 0.2 ms |
| Batch 8 val, per image (pre + inference + post) | 1.5 + 6.0 + 0.5 = 8.0 ms (~125 FPS) | 1.0 + 7.4 + 0.5 = 8.9 ms (~112 FPS) |

Both models share the architecture, so the speed difference is noise. During val, Ultralytics warned about slow image access; the benchmark images are reached through `datasets\9class`. Batch-1 wall time does not include decoding the JPEG from disk.

## 4. False positives and hard negatives

The benchmark contains no `table_saw`, `planer`, `jointer`, `band_saw`, `miter_saw` or `fire_extinguisher`, so any detection of those classes is wrong. Unmatched detections of the three classes that are present are also counted.

| Wrong detections | run5 @0.40 | run8 @0.40 | run5 @0.25 | run8 @0.25 |
|---|---:|---:|---:|---:|
| `edge_banding_machine`, unmatched | 3 | 0 | 29 | 0 |
| `table_saw` (absent class) | 2 | 1 | 5 | 7 |
| `fire_extinguisher` (absent class) | 0 | 0 | 0 | 1 |
| **Total** | **5** (0.015/frame) | **1** (0.003/frame) | **34** (0.104/frame) | **8** (0.024/frame) |

What the wrong detections land on (crops in `runs\eval_run8_cctv_benchmark\<model>\fp_crops\`):

- **run5's whole-scene `edge_banding_machine` boxes:**
  - they span the frame, for example `38,0 → 1910,717` on Camera 05 and `76,0 → 1882,1080` on Camera 03;
  - this is run5's worst habit on CCTV;
  - run8 produces none, even at conf 0.05 on the frames checked.
- **`table_saw` on Camera 05 (both models): the panel saw under the wrong name:**
  - the box covers the sliding-table panel saw, at roughly IoU 0.75 with its label;
  - from this high angle, a panel saw's flat table and fence look like a table saw;
  - this is class confusion, not the model firing on background;
  - it is also why `panel_saw` recall stays near zero.
- **`fire_extinguisher` on Camera 03 (run8, 1 frame, conf 0.38):**
  - it hit a thin upright object by the doorway, under the "Camera 03" overlay;
  - the white/yellow drum that earlier models called an extinguisher was **not** detected by run8 (checked at conf 0.05), and run5 doesn't fire on it either.
- **Lockers, workbenches, metal cabinets and red drums:**
  - neither model fired on any of them at conf 0.25 or above;
  - this covers Camera 05's red drum next to the edge bander and Camera 03's workbench, plywood stacks, cabinets and drums;
  - caveat: the benchmark has no labels for look-alikes, so this is "no wrong detection anywhere in 328 frames", not a per-object hard-negative score.

## 5. Missed machines and mask boundaries

- **hydraulic_press (both cameras) and edge_banding_machine (both cameras) are never found**, by either model, at any threshold that counts.
  - **On Camera 03:** the press is small and far from the camera.
  - **On Camera 05:** both machines are mid-distance, cream-coloured boxes against cream walls. The edge bander on Camera 03 is large and close, but seen from above and partly hidden by workers.
  - The catalog images don't show these machines from a high corner angle, and the CCTV augmentations (blur, compression, lens) don't change the viewpoint.
- **Panel saw masks are too small, not leaking.**
  - **Matches at conf 0.25 (run8, n = 5):**
    - mean mask IoU with the label is 0.40;
    - only 12% of the predicted mask falls outside the label (leakage);
    - the mask misses 58% of the labelled area (coverage miss).
  - **Why:** the model segments the saw body and leaves out the long sliding carriage and the plywood on the table, which the SAM-based label includes (see "Known label limitations" in `docs\cctv_benchmark.md`).
  - **Effect:** this is why mask mAP is roughly 10× lower than box mAP (0.074 vs 0.640).
- **Camera perspective:** Camera 05 is wide-angle and squeezed 2:1 at source; frames are stretched back before labelling.
  - The saw sits at the bottom edge of the frame and is cut off by it, with the "Camera 05" overlay on top.
  - The `table_saw` boxes on it stretch upward over the workers standing behind the saw, so they overshoot the machine's top edge by about 100–150 px.

## 6. What this means for real-world transfer

1. **The curated data and `cctv_aug` did not add machine recall on real CCTV.** Box mAP fell slightly. The mask mAP gain (+0.015) is real but too small to matter in practice.
2. **What did improve is how clean the output is.** On factory footage, run8 is quieter than run5 by a factor of 4–5 in false positives. For an alerting system this matters, but it doesn't help until recall improves.
3. **The gap is viewpoint, not image quality.** The augmentations model sensor effects; what's missing is high, angled views of panel saws, presses and edge banders. The `panel_saw` → `table_saw` confusion points the same way.
4. **Next steps worth considering:**
   - **Viewpoint data:** collect high-angle images of these three machines, from other CCTV sources or catalog shots taken from above, never from Cameras 03 or 05.
   - **Measure confusion:** check how often `table_saw` boxes overlap the panel saw across the whole benchmark.
   - **Merged-class view:** report the panel-saw family both separately and as one class.
   - **Forgetting gate:** compare run8 with run5 on catalog val at imgsz 960. That gate (within 2 points of run5 mask mAP50-95) was not part of this evaluation.

## Limitations

- Each camera is in effect one viewpoint. 328 frames means 2 scenes, not 328 independent samples.
- The labels come from SAM 2.1 consensus outlines, not hand drawing. They include plywood on the saw table, which pushes mask scores down.
- The "operating threshold" of 0.40 is the project default. It was not tuned on this set, and must not be.
