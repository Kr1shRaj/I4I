# Factory footage analysis (2026-09 NVR exports)

*Written 2026-10-19. Source: `factory_dat\` (5 files, 2.03 GB). Baseline outputs: `runs\inference_tests\factory_baseline\`.*

## Summary

1. **There are 2 cameras, not 5 sources.** `panel_saw_hydraulic_lift.mp4` and `working_panel_saw.mp4` are not close-up or handheld clips. Both carry the burned-in **"Camera 05"** label and show exactly the same fixed view as `ch05_…`. The lift clip runs 12:20:00–13:03:25 on 09-28 and ends one second before `ch05_20260928130326` starts. The footage is **3 recordings of Camera 05 and 2 of Camera 03**.
2. **The video is stored squeezed 2:1 horizontally.** Every file is 960×1080, which is Hikvision's "1080P Lite" mode: a 16:9 picture encoded at half width. The stream doesn't declare a pixel aspect ratio, so OpenCV, Ultralytics and CVAT all show it squeezed. A round wall clock measures about 16×28 px; resized to 1920×1080, it is round and people have normal proportions. **The intake frames are saved at 1920×1080.** Inference must do the same resize (see §5).
3. **Both checkpoints are close to blind on this footage.**
   - The obvious Camera 05 panel saw is found in at most 10 of 24 frames (run6, imgsz 640, max conf 0.70). run5 finds it in 3 of 24 or fewer.
   - Neither model detects either press or either edge bander.
   - run6 has a **static false positive**: it calls a white/yellow plastic drum next to the Camera 03 edge bander a `fire_extinguisher` at conf 0.58–0.84, in nearly every frame.
4. **No real hard negatives exist on these cameras.** Every Camera 03 frame shows the press and the edge bander, and every Camera 05 frame shows the panel saw. The 10% hard-negative share has to come from crops of machine-free regions, or from other cameras.

## 1. Technical metadata

All five files are identical in format:

| Property | Value |
|---|---|
| Container | Hikvision proprietary MPEG-PS (`IMKH` header), named `.mp4`; no `ftyp`/`moov` boxes |
| Video codec | H.264, High-profile SPS with scaling matrices |
| Coded size | **960 × 1080** (macroblock height 1088, cropped). Stored aspect 8:9; intended display 16:9 (1920×1080) |
| Pixel aspect | Not declared in the SPS. Visually 2:1 (see Summary 2) |
| Frame rate | 12 fps |
| Bitrate | 1.32 Mbps average (constant across files, so CBR) |
| Burned-in overlay | Date/day/time at top-left (`09-28-2026 Mon 13:10:44`); `Camera 03` / `Camera 05` at bottom-right. No other HUD |

| File | Camera | Recorded (from the burned-in clock) | Frames | Duration | Size |
|---|---|---|---:|---:|---:|
| `ch03_20260907165321.mp4` | 03 | 09-07 16:53:21 → 17:30:45 | 26,930 | 37 min 24 s | 370 MB |
| `ch03_20260924144142.mp4` | 03 | 09-24 14:41:42 → 15:16:47 | 25,261 | 35 min 05 s | 347 MB |
| `ch05_20260928130326.mp4` | 05 | 09-28 13:03:26 → 13:20:27 | 12,263 | 17 min 02 s | 168 MB |
| `panel_saw_hydraulic_lift.mp4` | **05** | 09-28 12:20:00 → 13:03:25 | 31,274 | 43 min 26 s | 430 MB |
| `working_panel_saw.mp4` | **05** | 09-25 16:18:14 → 17:30:33 | 52,085 | 72 min 20 s | 715 MB |

Total: 147,813 frames (3 h 25 min). Camera 05 has 2 h 13 min; Camera 03 has 1 h 12 min. All footage is daytime (12:20–17:30). **There is no night/IR footage.**

## 2. Visual content and environment

### Camera 03: cluttered machine room
- **Mounting:** high in a corner, about 3 m, pitched down about 35–45°.
- **Lens:** wide-angle with strong barrel distortion. The left wall and the window frame on the right bow visibly.
- **The view is identical in the 09-07 and 09-24 recordings**, so the camera did not move.

**Target machines**

| Class | Where | Notes |
|---|---|---|
| `edge_banding_machine` | Centre-right, foreground | Compact edge bander (cream enclosure, touch panel, black edge-tape coil plate, glue unit). Operated in most of the 09-24 recording. About 15% of the frame. Partly hidden by the operator and by plastic containers in front. |
| `hydraulic_press` | Back wall, centre | White hot/cold press frame with yellow-black hazard platens. About 10% of the frame. Partly hidden by stacked boards. |
| `panel_saw` (partial) | Far left edge | The sliding table of a saw is cut off by the frame edge (09-24). Only label it if at least 40% of the machine is visible (plan §2). |

**Not present:** fire extinguishers (none visible), table saw, jointer, planer, band saw, mitre saw.

**Look-alike / false-positive risk:** a white/yellow plastic drum beside the edge bander (run6 fires on it), large-diameter cardboard tubes, a workbench with vices and small tools at bottom-left.

**Scene conditions**
- Even artificial and daylight lighting. Mean luma 110, blown pixels 0.2%.
- A bright vertical window/door strip on the right edge.
- Heavy clutter: boards, offcuts, furniture parts, chairs.
- 1–4 workers often stand in front of the edge bander. They are motion-blurred at 12 fps.

### Camera 05: main workshop floor
- **Mounting:** under the roof trusses, about 4–5 m, pitched down about 30–40°. The truss is in the top third of the view.
- **Lens:** wide-angle with barrel distortion.
- **The view is identical in all three recordings (09-25 and 09-28).**

**Target machines**

| Class | Where | Notes |
|---|---|---|
| `panel_saw` | Right foreground | Large sliding-table panel saw: carriage, outrigger, scoring and riving guard, rip fence. The bottom frame edge cuts it off, and the `Camera 05` label sits on top of it. Visible area about 20–30% of the frame. Seen from above at a steep oblique angle; catalog images have nothing like this view. |
| `edge_banding_machine` | Back wall, centre | Long automatic edge bander with infeed table. About 6–8% of the frame, often hidden by workers. |
| `hydraulic_press` | Left, mid-ground | White press frame with hazard-striped platens. About 8% of the frame. Hidden by board stacks. |

**The "hydraulic lift":** no lift table is identifiable in the sampled frames. The press on the left may be what the file name refers to. A lift table is **not** the `hydraulic_press` class, so add it to the class-boundary policy as a negative if one turns up.

**Not present:** fire extinguishers (none clearly visible; there are red buckets and bins), table saw, jointer, planer, band saw, mitre saw.

**Scene conditions**
- **Severe glare.** The open roller door at the back right is completely blown out (3.5–4.7% of all pixels ≥ 250). Fluorescent tubes flare.
- Cobwebs on the truss near the lens.
- 2–4 workers feed panels through the saw for most of the time. Large panels and workers' bodies hide 30–60% of the saw in many frames.
- Board stacks hide the press.

### What this means for labelling and sampling
- **Viewpoints:** 2 cameras = **2 distinct viewpoints in total**. Per-class viewpoint coverage is at most 2 (press, edge bander), and 1 for panel saw. That is the "pilot signal" tier of plan §1.3 (≥ 5 viewpoints per class) **at best**. More cameras, or pseudo-CCTV phone video, are needed.
- **Classes covered:** only 3 of the 9 appear: `panel_saw`, `edge_banding_machine` and `hydraulic_press`. The other 6, including `fire_extinguisher`, get no CCTV data from this drop.
- **Missing conditions:** no night/IR, and no empty scenes with the lights off.

## 3. Baseline model check

### Method
- **Frames:** 40, 8 evenly spaced per file, from `runs\inference_tests\factory_baseline\frames_native\`.
- **Variants:** each frame was tested natively (960×1080) and de-squeezed (1920×1080).
- **Settings:** both checkpoints at imgsz 640, 960 and 1280, conf 0.15, through `scripts\test_inference.py`.
- **Ground truth:** every Camera 05 frame contains the panel saw (24 frames). Camera 03 frames contain the press and the edge bander (16 frames). No frame contains a fire extinguisher.

### Panel saw on Camera 05: frames detected out of 24, at conf ≥ 0.15

| Model | Input | imgsz 640 | imgsz 960 | imgsz 1280 | Max conf |
|---|---|---:|---:|---:|---:|
| run5 | native | 2 | 1 | 1 | 0.35 |
| run5 | de-squeezed | 2 | 3 | 2 | 0.24 |
| run6 | native | **9** | 6 | 2 | **0.70** |
| run6 | de-squeezed | **10** | 0 | 1 | 0.46 |

- Masks, where they exist, cover the carriage and table area but miss the parts hidden by workers.
- **At conf 0.40, no combination detects the saw in more than 3 of 24 frames** (run6 de-squeezed 640: 3; run6 native 640: 2; every run5 setting: 0).

### Everything else
- **Edge banders and presses:** not detected by any configuration.
- **Frame-filling false positives:** run5 sometimes predicts a frame-filling `edge_banding_machine` (box covering 64–68% of the image, conf 0.16–0.49) on both cameras. This matches the known frame-filling edge-banding labels (todo §2).
- **Fire extinguisher false positive (run6, Camera 03):**

  | Input | imgsz | False positives | Max conf |
  |---|---:|---:|---:|
  | native | 640 | 19 | 0.84 |
  | native | 1280 | 29 | 0.69 |

  All are on the static white/yellow drum. In deployment it would fire continuously, which breaks the ≤ 1 false positive per camera-hour gate.
- **Other false positives:** occasional `table_saw` (0.50–0.65), `jointer` (≤ 0.20) and `band_saw` (≤ 0.19) on Camera 03 clutter and the workbench.
- **Empty floor and walls:** no detections at all. The false positives are on objects (drum, clutter), not on empty areas.

### Interpretation
- **Image size:** imgsz 640 is the best size for both models here, consistent with the earlier workshop-clip finding (todo §2: 1280 hurts).
- **Squeezed vs de-squeezed:** de-squeezing didn't help these checkpoints. It only changes geometry, and the main gaps are viewpoint (steep top-down), occlusion and codec. Keep de-squeezed frames anyway: they show true geometry to annotators and match the square-pixel catalog set. YOLO polygon coordinates are normalised, so labels stay valid in either geometry.
- **Which model:** run6 beats run5 here on the panel saw at 640, unlike on the earlier screen-recorded workshop clip. But its fire-extinguisher false positive makes it unusable. Neither model can be deployed on these cameras without CCTV fine-tuning, which is the purpose of Phase 2.

## 4. Phase 2: frame extraction, pre-labelling, manifests

### 4.1 Sampling and de-duplication (`scripts\tools\sample_cctv_video.py`)

**Settings:** 1 frame every 3 s, de-squeezed to 1920×1080, cap 120 per recording. Names follow `cctv_<camera>-<date>-<time>_f<frame_idx>.jpg` (e.g. `cctv_cam05-20260928-1220_f0006912.jpg`), so they are unique per recording and reproducible. The camera ID is part of the name, and the frame index is the index in the source video.

**Dedup method.** I first ran the plan's default: a 64-bit dHash, Hamming distance ≥ 7 to every frame already kept. It kept only **73 frames** (13 / 8 / 35 / 16 / 1), because a whole-frame 8×8 hash hardly changes when workers or panels move in front of a fixed camera. I switched to the SSIM option you named:
- keep a frame if SSIM to the last kept frame is **< 0.92** (160 px greyscale thumbnail);
- plus a repeat guard: Hamming ≥ 3 to every kept frame of the same camera, **including the camera's other recordings** (`--dedup-with`), so a scene that comes back unchanged later is not kept twice.

| Recording | Camera | Split | Sampled (every 3 s) | Distinct | Kept |
|---|---|---|---:|---:|---:|
| `ch03_20260907165321` | cam03 | train | 748 | 37 | 37 |
| `ch03_20260924144142` | cam03 | train | 702 | 34 | 34 |
| `working_panel_saw` | cam05 | val | 1,447 | 285 | 120 (cap, spread over 72 min) |
| `panel_saw_hydraulic_lift` | cam05 | val | 869 | 121 | 120 (cap) |
| `ch05_20260928130326` | cam05 | val | 341 | 17 | 17 (the rest repeat the lift clip from the same day) |
| **Total** | | | 4,107 | 494 | **328** |

**Output:** `datasets\cctv_factory_intake\<video_stem>\images\` plus `frames.csv` (source frame index, video time, wall-clock time, dHash). Log: `datasets\cctv_factory_intake\sampling.log`.

**Camera 03 yields few frames** (71 from 72 min): the scene is static apart from 1–2 workers.

### 4.2 Camera-level split (`datasets\cctv_factory_intake\camera_splits.yaml`)

| Camera | Recordings | Split |
|---|---|---|
| cam03 | both `ch03_*` | **train** (71 frames) |
| cam05 | `ch05_*`, `panel_saw_hydraulic_lift`, `working_panel_saw` | **val** (257 frames) |

- **Clip assignment:** `panel_saw_hydraulic_lift` and `working_panel_saw` are Camera 05 recordings, so they **must** be in val with `ch05_*`. Putting them in train would be camera leakage.
- **Leak check:** the builder now maps each recording to a physical camera and fails if one camera appears in two splits. Tested: moving `working_panel_saw` to train gives `camera cam05 is assigned to several splits ['train', 'val']: camera leakage`, and nothing is written.
- **Test split:** none. There is no third camera.

**Consequences of a two-camera split (decide before training):**
- **Val covers only the panel saw (and partly the press and edge bander).** Train covers only the press and edge bander. The CCTV fine-tune will see **no CCTV panel saw at all**, which is the class that regressed in deployment.
- **Val is 3.6× larger than train** (257 vs 71 frames).
- **The alternative is to swap:** cam05 → train, cam03 → val. That gives 257 train frames including the panel saw, and validates on the press and edge bander. Either way, val measures transfer to a *new* camera, which is what the plan's split rule is for.
- **The real fix is more cameras:** a third camera for test, and pseudo-CCTV phone video of the panel saw from other angles (plan §1.3).

### 4.3 Pre-labels (`scripts\tools\prelabel_cctv.py`)

**Primary set: run5 at conf 0.15, imgsz 640** (640 was run5's best size in §3). Location: `datasets\cctv_factory_intake\_prelabels\run5_conf015\<video_stem>\`. Each package holds `images\train`, `labels\train`, `data.yaml`, `train.txt` and `prelabel_summary.csv`; import into CVAT as "Ultralytics YOLO Segmentation 1.0".

| | Instances | Frames | Verdict (overlay review) |
|---|---:|---:|---|
| `panel_saw` (cam05) | 58 | 58 | Correct object. Polygons often include workers or panels resting on the saw; tighten them. |
| `table_saw` (cam05) | 42 | 40 | **The same panel saw, wrong class.** Relabel as `panel_saw`. 13 frames have both; delete the duplicate. |
| `edge_banding_machine` (cam03) | 2 | 2 | Check these. |
| `band_saw` (cam03) | 5 | 5 | False positive (no band saw on that camera). Delete. |
| **Total** | **107** | | **The saw is pre-labelled in 85 of 257 Camera 05 frames** (33%). About 60 instances are usable as they stand or after a class fix. |

**Not pre-labelled anywhere:** both presses, both edge banders (apart from the 2 to check) and the panel saw in the other 172 Camera 05 frames. Annotators will draw these from scratch.

**Alternate set: run6 at conf 0.15, imgsz 640.** Location: `_prelabels\run6_conf015_alt\`.
- `panel_saw`: 44 instances in 41 frames, cleaner class.
- 43 false-positive `fire_extinguisher` on the Camera 03 drum.
- 23 `edge_banding_machine` on Camera 05 board stacks and one ceiling-sized polygon; all false positives.
- Use it only if its panel-saw polygons save time; delete all of its `fire_extinguisher` and Camera 05 `edge_banding_machine` instances.

**Polygon fix during this work.** Ultralytics' `masks.xy` joins all fragments of a mask with bridges. After simplification those bridges became spikes across the image (seen in the first pass). The pre-labeller now:
- drops fragments under 10% of the instance;
- simplifies each fragment on its own;
- joins the remaining fragments with zero-width bridges (plan §2).

The re-run polygons are clean.

**Pre-labels are not in `labels\`.** The training builder only reads `<video_stem>\labels\` (reviewed labels), so an unreviewed pre-label can't reach training.

**Labelling rules for these cameras:**
- Label the press and the edge bander in **every** Camera 03 frame, and the panel saw, edge bander and press in **every** Camera 05 frame where at least 40% is visible.
- The drum next to the Camera 03 edge bander is **not** a fire extinguisher.
- Leave the burned-in OSD text unlabelled. Don't extend polygons over the `Camera 05` label.

### 4.4 Manifests and balance (`scripts\tools\build_cctv_hybrid_dataset.py --preview`)

**Lists written** (in `datasets\9class\`):
- `cctv_train_frames.txt` (71 frames)
- `cctv_val.txt` (257 frames)
- `cctv_test.txt` (empty)
- `build_cctv_hybrid_stats_preview.json`

**Not written: `train_hybrid_9class.txt` and `data_9class_cctv.yaml`.** They are only produced from reviewed labels, because Ultralytics would read the unlabelled frames as backgrounds.

**Path checks:**
- All 71 + 257 CCTV paths and all 3,312 catalog paths resolve through Ultralytics' own `get_img_files`, with 0 missing.
- The three catalog junctions resolve to `datasets\source_38class\{train,valid,test}\images`.
- Entries use the form `./../cctv_factory_intake/<stem>/images/<frame>.jpg`.

**Projected balance: 81% catalog / 19% CCTV / 0% hard negatives. The 50/40/10 target is not reachable with this drop as split:**
- **CCTV share:** 71 train frames at the 5× repeat cap give 355 entries. The catalog block stays at its 1,500-entry floor.
- **Hard negatives:** none, because every frame from these cameras contains a target machine (Summary 4). An empty pre-label is not a reviewed negative.

**Ways to reach the target** (with the 3,312-entry catalog list):

| Option | Projected shares | Cost |
|---|---|---|
| As split (cam03 train) | 81 / 19 / 0 | No CCTV panel saw in training |
| Swap (cam05 train, cam03 val) | about 56 / 44 / 0 | Validates only on press and edge bander |
| `--min-catalog 444` with cam03 train | about 56 / 44 / 0 | Only 444 catalog images as replay; forgetting risk |
| Add hard-negative **crops** (machine-free regions: Camera 05 ceiling/truss and door, Camera 03 left storage area), label them empty | +10% hard negatives | About 1 hour of cropping |
| More cameras, or pseudo-CCTV phone video | target reachable | Field work |

## 4.5 Update 2026-10-19: split swapped, labels finalised

**Split (approved):** cam05 → train (257 frames), cam03 → val (71 frames). The builder reports no camera leakage.

**How the labels were made** (`scripts\tools\annotate_static_cameras.py`, prompts in `datasets\cctv_factory_intake\machine_prompts.yaml`). They are machine-generated and spot-checked on about 25 overlays; they are **not hand-drawn**.
1. **Segmentation:** SAM 2.1 (`models\pretrained\sam2.1_b.pt`, official Ultralytics release) segments each fixed machine in every frame from a box prompt.
2. **Spill guard:** each mask is clipped to that camera's consensus outline (pixels present in at least 50% of frames, dilated 15 px).
3. **Workers and text removed:** worker masks from the COCO model (`yolo26s-seg.pt`, dilated 9 px) and the burned-in text boxes are subtracted.
4. **Visibility rule:** an instance is kept only if at least 40% of the machine is visible. If SAM misses a machine that workers don't hide, the consensus outline minus workers is used instead. That happened in 1 frame (`cam05-20260928-1220_f0012…`).
5. **Polygon export:** fragments under 10% of the instance are dropped, and the rest are joined with zero-width bridges.

**What happened to the run5 pre-labels:** they were replaced, not edited.
- **Panel saw:** the 42 `table_saw` → `panel_saw` class fixes and the 13 duplicate frames no longer apply. Every Camera 05 frame now has exactly one SAM-traced `panel_saw` with workers removed.
- **Camera 03:** the 5 false `band_saw` instances are gone. The final label files contain only classes 0, 1 and 7, so there are no `fire_extinguisher`, `band_saw` or `miter_saw` labels, and the drum is unlabelled background.
- **Run used:** the first SAM pass also had a carriage prompt, which bled onto floor and board stacks. It was removed before the labels were written.

**Final instance counts**

| Class | Train (cam05, 257 frames) | Val (cam03, 71 frames) |
|---|---:|---:|
| `panel_saw` | 257 | 0 |
| `hydraulic_press` | 257 | 71 |
| `edge_banding_machine` | 257 | 71 |
| Other 6 classes | 0 | 0 |

**Build result**
- `train_hybrid_9class.txt`: 2,891 entries = 1,606 catalog + 1,285 CCTV (257 frames × 5), giving **55.6% / 44.4% / 0%**.
- `cctv_val.txt`: 71 frames.
- `data_9class_cctv.yaml`: generated.
- Ultralytics loads train and val with **0 corrupt** labels.

**Known label limitations (check before trusting val numbers):**
- **Plywood on the saw table** is inside the `panel_saw` outline (SAM can't separate it).
- **The sliding carriage, when pulled far left** beyond x ≈ 930 px, is not labelled.
- **Board stacks partly in front of both presses:** SAM follows the visible press, but the edges are approximate.
- **Each machine has one viewpoint,** and outlines in neighbouring frames are highly correlated. The 257 train frames are close to 1 viewpoint per class (plan §1.3).
- **Val measures only the press and the edge bander.** There is no panel saw on Camera 03.
- **A human QA pass is still recommended** (overlay review in CVAT). Per-frame visibility and fallback flags are in `datasets\cctv_factory_intake\annotation_report.csv`.

## 5. Recommendations
1. **Choose the split direction** (§4.2). Without a third camera, swapping is the stronger pilot for the panel-saw regression.
2. **Fix inference geometry for these cameras:** resize every NVR frame to 1920×1080 before inference. Or configure the NVR channels to full 1920×1080 instead of "1080P Lite". Use imgsz 640 until CCTV-val says otherwise.
3. **Label in CVAT** from the run5 packages, following §4.3. Then export to `datasets\cctv_factory_intake\<stem>\labels\` and run the builder without `--preview`.
4. **Add hard negatives:** machine-free crops from both cameras, plus the white/yellow drum as an explicit hard negative.
5. **Fill the coverage gaps:** night/IR recordings, a third camera, and phone video of the panel saw. The other 6 classes, including `fire_extinguisher`, get no CCTV data from this drop.
