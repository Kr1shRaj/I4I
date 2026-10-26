# run7 post-training analysis (`run7_9class_cctv_pilot`)

*Written 2026-10-19. Evaluation scripts and raw outputs: `runs\evals\run7_analysis\` (`eval_run7.py`, `results_*.json`, `video\`, `video_lowconf\`).*

## Verdict

**run7 is not a new champion. Keep run5 as the baseline and the fallback.**
- **Training camera (cam05):** run7 detects the panel saw, press and edge bander in **100% of frames at about 0.97 confidence**. But it saw that exact scene, five times per epoch, in training.
- **Any camera it hasn't seen:** run7 detects **nothing**.
  - Camera 03 val: box mAP50 0.01. **0 of 142** labelled machines are found, even at conf 0.01.
  - Workshop clip (the original panel saw regression): no detection at conf 0.25; max 0.17.
- **False alarms:** `fire_extinguisher` false alarms on Camera 03 rose from **0 (run5)** to **51–74 (run7)**.
- **Forgetting:** catalog box mAP50 fell from **0.821 (run5) to 0.786–0.792** (−3 points) at imgsz 960, and box mAP50-95 fell by 9–11 points. That fails the plan's "within 2 points of run5" gate.

**Root cause: memorisation, not a transfer bug.** The CCTV training data is **one viewpoint per class**: 257 near-identical frames of a fixed camera, repeated 5×, making up 44% of every epoch. Camera 03's machines are **different machine models** seen from a different angle. The model learned *this scene* (and where things sit in it), not the machines.

## 1. Training curve and early stopping

**Setup:** run5 weights, AdamW, lr0 2e-4, cosine schedule, imgsz 1280, batch 4. Train = 2,891 entries (1,606 catalog + 257 cam05 frames × 5). Val = 71 Camera 03 frames (71 presses, 71 edge banders).

| Epoch | train box / seg / cls loss | val box / seg / cls loss | val box mAP50 | val mask mAP50-95 |
|---:|---|---|---:|---:|
| 1 | 1.08 / 1.90 / 1.20 | 2.14 / 3.30 / 4.34 | 0.0027 | 0.0002 |
| 7 (**best.pt**) | 0.66 / 1.23 / 0.61 | 1.56 / 3.09 / 4.74 | 0.0090 | 0.0056 |
| 17 (stopped) | 0.58 / 1.03 / 0.51 | 1.33 / 2.38 / 4.81 | 0.0023 | 0.0004 |

**Train vs val**
- **Training losses** fall steadily: box halves, cls falls 57%.
- **Val classification loss** never improves (4.2–5.6, noisy, no trend).
- **Val box and seg losses** improve somewhat. The model gets better at "something is here" but no better at "which class it is", on Camera 03.

**Early stopping was triggered by noise.**
- Val fitness never rises above about 0.012. Its peaks at epochs 2, 7 and 14 (0.0112–0.0117) are statistically identical.
- `best.pt` is epoch 7 only because of that noise. Patience 10 then stopped training at epoch 17.
- **`best.pt` was therefore chosen essentially at random.** `last.pt` (epoch 17) is better on catalog val (§3).

### What the model predicts on Camera 03

**Confusion matrix** (`confusion_matrix.png`, conf 0.25):
- `edge_banding_machine`: 71 of 71 missed.
- `hydraulic_press`: 59 missed, and 12 predicted as `edge_banding_machine`.
- **112 `fire_extinguisher` predictions on background**, with no extinguisher in the frames.

**Val predictions** (`val_batch*_pred.jpg`): the `fire_extinguisher` boxes land on the white/yellow plastic drum, the large cardboard tubes, water bottles and **standing workers**. There are no boxes on the edge bander and none on the press.

### Below threshold, or no transfer?

I measured it directly. For each labelled Camera 03 machine, I took the highest-confidence prediction of the **correct class** overlapping it (box IoU ≥ 0.5), at conf ≥ 0.01. Script: `eval_run7.py`.

| Model | imgsz | Presses found (of 71) at ≥0.25 / ≥0.10 / ≥0.01 | Edge banders found (of 71) | Same box, wrong class (≥0.25) |
|---|---:|---|---|---|
| run7 best | 640 | 0 / 0 / 0 | 0 / 0 / 0 | press called `edge_banding_machine` **64×** |
| run7 best | 1280 | 0 / 0 / 3 (max 0.02) | 0 / 0 / 0 | 1× |
| run7 last | 640 | 0 / 0 / 0 | 0 / 0 / 0 | press called `edge_banding_machine` 44× |
| run5 | 640 / 1280 | 0 / 0 / 0 | 0 / 0 / 0 | – |
| run6 | 1280 | 0 / 0 / 0 | 0 / 0 / 9 (max 0.03) | – |

**This is a failure to transfer, not a threshold problem.** Lowering the threshold to 0.01 recovers 3 of 142 instances.

**Why it fails:**
1. **One viewpoint per class.** All 257 train frames come from one fixed camera. The press, edge bander and saw occupy the **same pixels in every frame**, with near-identical SAM outlines. The plan's minimum for a pilot signal is ≥ 5 viewpoints per class (§1.3); this run had 1.
2. **The val machines are different machines.** Camera 03's edge bander is a compact manual unit with a tape-coil plate; Camera 05's is a long automatic line. Camera 03's press is a hot press; Camera 05's is a large white frame press. One training example per class can't cover a different model of the machine.
3. **The model learned location, not appearance.** At 640, run7 labels the Camera 03 **press** as `edge_banding_machine` in 64 of 71 frames. In Camera 05, the edge bander sits in the upper centre of the image as a long white shape; in Camera 03, that is exactly where the press is.
4. **The scene is memorised.** run7 scores 0.96–0.97 on Camera 05 **even on the raw squeezed 960×1080 video**, although it trained only on de-squeezed 1920×1080 frames. Confidence that holds up under a 2:1 geometry change suggests it matches scene texture and layout rather than machine shape.

## 2. Real-world test: panel saw

All tests at conf 0.25 unless noted. Detection rate = frames with ≥ 1 detection of that class.

### Camera 05 `working_panel_saw.mp4` (your command: imgsz 640, first 200 frames)

| Model | Panel saw | Other |
|---|---|---|
| run5 | 0/200 | nothing |
| run6 | 0/200 | nothing |
| **run7** | **200/200, mean 0.97** | press 200/200 (0.96); edge bander 200/200 (0.82) |

**The same recording, de-squeezed to 1920×1080, 200 frames spread over all 72 minutes:**

| Model | Panel saw |
|---|---|
| run5 | 36/200 (18%), mean 0.36, max 0.57; plus `table_saw` 27/200 |
| run6 | 65/200 (32%), mean 0.37, max 0.70 |
| **run7** | **200/200, mean 0.96** (press and edge bander also 100%) |

**This is in-sample.** `working_panel_saw.mp4` is one of the three training recordings: 120 of its frames, spread across the whole video, are in `cctv_train_frames.txt`, and every other frame is a near-duplicate of them. The result shows run7 fits its training camera. It does not show that panel-saw detection generalises. There is no held-out Camera 05 footage.

### Workshop clip (`Screen Recording 2026-09-28 124839.mp4`): the held-out panel saw

| Model | imgsz 640, conf 0.25 (your command) | conf 0.05, imgsz 640 | conf 0.05, imgsz 960 |
|---|---|---|---|
| run5 | `table_saw` 100/200, mean 0.28 | `table_saw` max 0.36 | **`table_saw` 200/200, max 0.70** |
| run6 | nothing | `panel_saw` max 0.05 | `fire_extinguisher` 110/200 ≥ 0.25 (false positives) |
| **run7** | **nothing** | `panel_saw`/`table_saw`/press max 0.07 | **`panel_saw` max 0.16, `table_saw` max 0.17** |

**run7 did not restore panel-saw detection on unseen footage; it made it worse.** run5 still finds the saw strongly (0.70 at imgsz 960), but calls it `table_saw`. That points at the class boundary between sliding-table saws and table saws in the catalog labels (todo §7), not at the CCTV domain.

Caveat: the workshop clip is a screen recording with the CCTV tool's overlay burned in (todo §2). It is the only held-out panel-saw footage there is.

## 3. Catastrophic forgetting (catalog val, `data_9class.yaml`)

Each model was validated at imgsz 960 (run5's training size) and at 1280 (run6's and run7's training size). The table shows box mAP50 / box mAP50-95 / mask mAP50 / mask mAP50-95.

| Model | imgsz 960 | imgsz 1280 |
|---|---|---|
| run5 | **0.821** / 0.706 / 0.798 / 0.679 | 0.745 / 0.359 / 0.714 / 0.466 |
| run6 | **0.825** / 0.698 / 0.813 / 0.690 | 0.810 / 0.649 / 0.795 / 0.653 |
| run7 best (ep 7) | 0.786 / 0.597 / 0.767 / 0.633 | 0.728 / 0.509 / 0.710 / 0.553 |
| run7 last (ep 17) | 0.792 / 0.614 / 0.771 / 0.645 | 0.770 / 0.538 / 0.744 / 0.588 |

**Reading the table**
- **Same size, run5 vs run7 at 960:** run7 loses **2.9–3.5 points of box mAP50** and **9–11 points of box mAP50-95**. Mask mAP50-95 drops less (−3.4 to −4.6). Fails the plan's gate.
- **Why run5 looks bad at 1280:** run5 was trained at 960, and 1280 is outside its training scale (the same effect as in todo §2). The ~0.81 you quoted is run5 at 960 (logged 0.83 on its original label build; 0.821 on today's).
- **Per class:** at 1280, run7's biggest drops compared with run6 are `table_saw` (0.93 → 0.74 best / 0.85 last) and `jointer` (0.87 → 0.74 / 0.83). `panel_saw` holds (0.85 → 0.78 / 0.85). `fire_extinguisher` rises slightly (0.68 → 0.71 / 0.73).

**Did the 56% replay prevent forgetting?** Partly. mAP50 held to within about 3 points. The larger mAP50-95 loss means box precision degraded: 44% of every epoch was spent on one camera's static outlines, under heavy codec and blur augmentation. Replay limited the damage; it didn't prevent it.

## 4. Scorecard

| Metric / test | run5 (baseline) | run6 (robust run) | run7 (pilot hybrid) |
|---|---|---|---|
| Catalog box mAP50, imgsz 960 | **0.821** | **0.825** | 0.786 (best) / 0.792 (last) |
| Catalog box mAP50, imgsz 1280 | 0.745 | 0.810 | 0.728 / 0.770 |
| Cam05 panel saw, de-squeezed spread (in-sample for run7) | 18%, max 0.57 | 32%, max 0.70 | 100%, mean 0.96 ⚠ training camera |
| Workshop panel saw (held out), best setting | 200/200 **as `table_saw`**, max 0.70 | none (false positives only) | none; max 0.17 |
| Cam03 generalisation: box mAP50 (640 / 1280) | 0.000 / 0.001 | 0.000 / 0.008 | 0.010 / 0.009 |
| Cam03 machines found at ≥ 0.25 (of 142) | 0 | 0 | 0 |
| Cam03 `fire_extinguisher` false alarms, conf ≥ 0.25, 71 frames (640 / 1280) | **0 / 0** | 32 / 2 | 51 / 74 |

## 5. What to change for the production run

1. **Get viewpoint diversity before training again.** This is the constraint that decided the result. Per plan §1.3 the minimum is **≥ 5 distinct viewpoints per class**:
   - pseudo-CCTV phone video of the panel saw, press and edge bander from 3–4 m at several angles;
   - more cameras;
   - a second site.

   A third camera is also needed for a test split.
2. **Weight CCTV by viewpoints, not frames.** 257 frames of one fixed view is about one view of information, and repeating it 5× to fill 44% of each epoch rewarded memorisation. Until there are ≥ 5 viewpoints:
   - repeat CCTV frames at most 1–2× (a share of about 15–25%);
   - subsample each fixed camera to the frames that actually differ (occlusion, workers, panel positions).
3. **Fix model selection.**
   - Camera 03 val (different machine models, mAP about 0.01) is pure noise as a selection signal, so `best.pt` and early stopping were driven by noise.
   - Until CCTV val has the same classes and machine types as train, either run fixed epochs with `patience=0` and use `last.pt`, or select on catalog val and report CCTV val separately.
   - Make sure val contains a held-out panel saw.
4. **Add CCTV hard negatives for cylinders.** `fire_extinguisher` false alarms grew in run7, on drums, cardboard tubes, water bottles and workers. Add empty-labelled crops from the training cameras (bins, buckets, tubes, people) for the 10% hard-negative share. That share was 0% in this run. Camera 03 can't supply them while it is the val camera.
5. **Fix the panel-saw / table-saw class boundary in the catalog labels** (todo §7). run5 recognises the workshop saw strongly but as `table_saw`. Re-labelling sliding-table saws as `panel_saw` may recover more on real footage than further CCTV fine-tuning.
6. **Train at imgsz 960 and consider a lighter augmentation strength.** run5 and run6 both score best at 960 on catalog val, and run7's box mAP50-95 loss points at localisation damage. Sweep `cctv_aug` probabilities: Downscale 0.4 and JPEG 20–60 at p=0.5 is aggressive.
7. **Keep run5 as the deployment fallback.** Don't promote run6 either: its catalog score is the best, but it has the Camera 03 extinguisher false alarm and fails on the workshop clip.

## Appendix: commands and artifacts
- **CCTV-val and catalog val, all models:** `env\yolo-env\Scripts\python.exe runs\evals\run7_analysis\eval_run7.py cctv` (or `catalog`). Results go to `results_cctv.json`, `results_catalog.json` and `results_catalog_960.json`.
- **Video tests:** `scripts\test_inference.py` outputs in `runs\evals\run7_analysis\video\<model>_{cam05_raw, workshop, cam05_desq_spread}\` and `video_lowconf\`.
- **Summaries:** `summarize_video.py` and `print_results.py` in the same folder.
- **Catalog val method:** run through the Ultralytics Python API (`YOLO(...).val(data=..., imgsz=...)`) in `eval_run7.py`. This is equivalent to the requested `yolo segment val` command, and adds imgsz 960 for a fair run5 comparison.
