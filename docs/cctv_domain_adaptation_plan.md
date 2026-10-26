# CCTV Domain Adaptation Plan — 9-class YOLO26-seg

*Written 2026-10-10. Follows `runsummary.md` and `run_analysis_10q.md`, and answers the CCTV brief.*

---

## 0. Corrections to the brief, checked before planning

Before answering the five questions, I re-measured the evidence. Five points in the brief change the plan.

| # | Brief says | Measured | Why it matters |
|---|---|---|---|
| 1 | The workshop machine is "small, distant", 50–150 px | **No.** In `machine_f1303_stream_480x360_polygons.json` the saw box is **266 × 208 px on a 480 × 360 frame**: 55% of the width and ~32% of the area. The upfront panel-saw clip boxes cover ~67% of the frame (`panel_Saw\detections.csv`, `box_area_pct`). | Object size differs by about **2×, not 10×**. The gap is **native resolution, codec, viewpoint, tint and clutter**, not small objects. This deprioritises SAHI (§4). |
| 2 | "Complete failure" is a domain gap | **Partly a regression.** On the same frames the previous checkpoint `9_class_clean_finetune` scored `panel_saw` **0.56 / 0.61 / 0.63** at imgsz 960. `cctv_robust` scores **≤ 0.18** (`old_vs_new_checkpoint.json`). | The domain gap existed before, but the old model got over 0.5. This run made it worse, through the look-alike hard negatives and the augmentation mix. **Step 0 is to stop using `cctv_robust` as the base.** |
| 3 | imgsz 1280 is the right inference size | **It hurts on this source.** The CCTV source is a **480 × 360 sub-stream**. The old model scores 0.56 at imgsz 960 and **0.27 at imgsz 1280** on the same frame. | Upsampling 480 px by 2.7× creates huge, blurry objects the model never trained on. Choose the inference size per stream (§4.1). |
| 4 | Augmentations "were applied" | **`albumentations` is not installed** in `yolo-env`. Ultralytics silently skips its Albumentations block: blur, median blur, ToGray, CLAHE, compression. `erasing=0.4` is classification-only and does nothing for `segment`. | The run had **no photometric degradation** beyond HSV jitter. That is part of the answer to Q3. |
| 5 | `lr0=0.01` | `optimizer=auto` chose **AdamW at lr0 = 0.000769** and ignored `lr0` (see `run_analysis_10q.md`). | Every recipe below sets `optimizer=AdamW` explicitly. |

Two more caveats about the "success" clips:

- **"% frames with a detection" is not accuracy.** With no ground truth, it counts true and false positives together. In the panel-saw clip, 44 of 173 frames have **two** `panel_saw` boxes, even with the NMS-free head.
- **The upfront clips are in-distribution.** They are framed like catalog photos, so they confirm what val mAP already said. They are not evidence about production. The extinguisher clip also has the CCTV tool's overlay burned in.

---

## 1. Dataset strategy and composition

### 1.1 Hybrid: yes, and never drop the catalog set

- **Train on catalog + CCTV together.** Fine-tuning on CCTV alone causes **catastrophic forgetting** of the close-up domain. That is what happened in a milder form here: a fine-tune with a shifted data mix lost a capability the parent had.
- **The catalog images keep doing a job.** They are the only source of clean, high-resolution part-level detail (blades, fences, rails). CCTV frames are low-detail, and the model needs both.

### 1.2 Ratio: control the sampling share, not the file count

The ratio that matters is the **share of each training epoch**. Set it by repeating list entries in `train_*.txt`, as `train_oversampled_9class.txt` already does.

| Phase | Unique CCTV frames | Epoch sampling share (catalog / CCTV / hard negatives) | CCTV repeat factor |
|---|---|---|---|
| Pilot (weeks 1–2) | ~1–3 k | **55 / 35 / 10** | ×3–5 |
| Production | 5 k+ | **40 / 50 / 10** | ×1–2 |

- **Keep catalog at ≥ 35–40%** as replay.
- **Keep hard negatives at 10–15%.** Mine them from **your deployment cameras**: empty floors, desks, ceilings, racking, look-alike machines.
- **Don't repeat any frame more than ~5×.** Beyond that the model memorises it, and more unique frames are worth more than more repeats.

### 1.3 How many labelled CCTV frames per class

No paper gives a universal number. These tiers are practitioner estimates for fine-tuning a COCO-pretrained seg model. **Verify them with a learning curve** (Step 5).

The unit that matters is **distinct viewpoints**, meaning (camera × machine) pairs, not frames. In a fixed camera, a static machine looks almost identical in 300 frames, so those 300 frames are roughly one viewpoint of information.

| Tier | Instances per class | Distinct viewpoints per class | What you get |
|---|---|---|---|
| Pilot signal | 150–200 | ≥ 5 | Enough to show whether CCTV data moves CCTV-val mAP |
| Working | ~500 | ≥ 10–15 | Usable on cameras similar to the training ones |
| Production | 1,000–1,500 | ≥ 20, from ≥ 3 sites, day + night | Generalises to new cameras |

**The real constraint is viewpoint diversity.** A factory usually has one or two of each machine, and each fixed camera sees it from one angle. Ways to get more viewpoints:

1. **Several sites and cameras.** Every new camera is worth more than a week of footage from an existing one.
2. **"Pseudo-CCTV" phone video.** Hold the phone at **3–4 m height** (ladder, mezzanine, pole mount), pitched down 25–50°, and walk an arc around each machine. Then degrade it to CCTV quality (§3.3). This is the cheapest viewpoint coverage for rare classes: edge bander, jointer, planer.
3. **Synthetic renders** from 3D/CAD models, viewed from elevated angles (§3.3 item 5).

### 1.4 Sampling CCTV video without autocorrelation

Record **raw RTSP**, not screen recordings: no overlay, no tint, no double compression. Pull the **main stream**, not the sub-stream. The stream URL depends on the vendor. On Hikvision, channel `101` is the main stream and `102` the sub-stream. On Dahua, use `subtype=0`.

```powershell
ffmpeg -rtsp_transport tcp -i "rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/101" `
  -c copy -f segment -segment_time 600 -reset_timestamps 1 -strftime 1 cam01_%Y%m%d_%H%M%S.mp4
```

Then sample in four stages:

1. **Coarse time stride.** Decode **1 frame every 2–5 s**.
2. **Change-based dedup.** Keep a frame only if it differs from the last kept frame **of that camera**. Use either a difference hash (Hamming distance ≥ 6–8 on a 64-bit dHash) or SSIM < 0.92 on a 160 px greyscale thumbnail.
3. **Stratified caps per camera.** Keep at most about **50–100 frames per camera per day**, balanced across the strata below. Reject strata already full.

   | Stratum | Values to cover |
   |---|---|
   | Shift / light | day, evening artificial, night IR (greyscale), lights off |
   | Occlusion | none, worker in front, cart/stock in front |
   | Machine state | idle, running (blade, workpiece), covered |
   | Weather / sun | sun patches and blown-out windows (visible in f1303), overcast |

4. **Hard-negative mining.** Run the current model at `conf=0.15` on empty-scene footage. Every hit goes to the hard-negative pool.

```python
# sketch: 1 frame / 3 s, dHash dedup, cap per camera-day
import cv2, numpy as np
def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n))
    return (g[:, 1:] > g[:, :-1]).flatten()
cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); step = int(fps * 3)
last, kept, i = None, 0, 0
while kept < 100:
    ok = cap.grab()
    if not ok: break
    if i % step == 0:
        _, f = cap.retrieve(); h = dhash(f)
        if last is None or np.count_nonzero(h != last) >= 7:
            cv2.imwrite(f"{out}/{cam}_{i:07d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95]); last, kept = h, kept + 1
    i += 1
```

**Split by camera, never by frame.** At least 3 cameras, and ideally one whole site, are **only** in CCTV-val/test. If frames from one camera appear in both train and val, the val mAP will look excellent and mean nothing.

---

## 2. Labelling guidelines for the CCTV perspective

Write these into a one-page labeller sheet, with 2–3 example images per rule.

### 2.1 Extent: what belongs to the instance

| Rule | Decision |
|---|---|
| **Mask type** | **Modal**: label the **visible** pixels only. YOLO-seg learns what it can see. Amodal (hallucinated) outlines teach it to paint masks onto occluders. |
| **Attached parts** | Part of the machine **if rigidly attached and visible**: sliding carriage, outrigger/crosscut table, fences, extension tables, rails, guards, the hydraulic press frame and its fixed bed. |
| **Not part of the machine** | Loose workpieces and boards on the table, dust-extraction hoses beyond the machine's outlet flange, stacked lumber, tools, carts, workers. |
| **Foreshortened rails / long tables** | Include the full visible run, even when perspective makes it a thin sliver. If a segment narrows to **< 3 px** at native resolution, stop the polygon there; don't trace sub-pixel slivers. |
| **Occluder splits the machine** | **One instance.** Label each visible part. Ultralytics accepts one polygon per line, so join the parts with a zero-width bridge (CVAT/Roboflow exports do this), or drop a fragment under 10% of the instance's area. |
| **Truncated by the frame edge** | Label the visible part. |
| **Class policy for look-alikes** | Follow the class-boundary sheet from `todo.md`, e.g. sliding-table saw = `panel_saw`, spindle moulder = negative. Settle it **before** labelling CCTV. |

### 2.2 Visibility and size thresholds

All thresholds are at **native stream resolution**.

- **Label** an instance if a person can name its class **from this frame alone**, it is **≥ 40% visible**, and its shorter side is **≥ 16 px**.
- **Below that threshold**, or when it can't be identified, **neutralise it.** Fill the region with grey (114,114,114) in the training image. Ultralytics has no "ignore" label. Leaving it unlabelled teaches "this is background", and labelling it teaches noise.
- **Label every target instance in every frame.** A missed label is a hard negative against your own class. The extinguisher-crowding problem came from this.

### 2.3 Polygon quality

- Place vertices **every 5–10 px**, within **1–2 px** of the edge, at native resolution. Don't trace compression blocks or sensor noise.
- **Follow the machine silhouette, not shadows.** A high camera puts the floor shadow next to the base, and it is a common mistake.
- **Label only raw frames.** Never label screen recordings with the tool overlay or tint.

### 2.4 Throughput and QA

- **Pre-label, then correct.** Run `9_class_clean_finetune` at `conf=0.1` to pre-label, then correct with SAM 2 point/box prompts (CVAT and Roboflow both support it).
- For a static machine in a fixed camera, **label once and propagate** with SAM 2 video propagation. Re-check the frames that have occlusion.
- A second person re-labels **10%** of frames. The target is **mask IoU ≥ 0.80** per instance. Below that, the rule is ambiguous, so fix the guideline, not the labeller.

---

## 3. Why the augmentations failed, and what transfers

### 3.1 A 2D homography cannot create a 3D viewpoint

- **Raising a camera is an out-of-plane rotation of a 3D object.** At 30–50° pitch you see the **table top and carriage from above**, the front face shrinks or disappears, near parts hide far parts, and parts at different depths shift against each other. Those pixels **are not in a frontal catalog photo**, so no 2D warp can produce them.
- **`perspective` applies one homography to the whole image.** That is exactly correct for a flat plane, such as tilting a **printed photo** of the machine. A real 3D machine viewed from above is something else.
- **The magnitude was also small.** Ultralytics centres the 2560 px mosaic canvas, then crops the middle 1280 px (`augment.py:1129-1135`). At the crop edge (|x| ≤ 640), `perspective=0.0005` changes the projective scale by at most **±32%**, and about ±16% typically. That is a mild keystone, roughly a few degrees of virtual tilt.
- **`degrees=10` is in-plane roll.** CCTV cameras are almost never rolled; elevation is **pitch**. Roll augmentation has nothing to do with the CCTV gap.

### 3.2 The photometric and context gap was barely touched

| Gap | Real CCTV (visible in `machine_f1303_*`) | What the run did |
|---|---|---|
| **Resolution** | A 480 × 360 sub-stream, upsampled ×2.7 at inference | `scale=0.9` shrinks **sharp** catalog pixels. Downscaling a 12 MP photo gives a small but **clean** object, which is not the same as a low-resolution sensor. |
| **Codec** | H.264/H.265 at low bitrate: 8×8 blocking, smeared chroma, ringing, keyframe pulsing | Nothing (`albumentations` missing) |
| **Sensor** | Noise, WDR halos, blown-out windows, motion blur on workers and blades | Nothing |
| **Colour** | Colour cast and tint; **IR night is monochrome**. The extinguisher sweep scored **≈ 0 in greyscale at every size**. | HSV jitter only. `ToGray` never ran. |
| **Lens** | Wide-angle barrel distortion | Nothing |
| **Context** | Leaning lumber, clutter, workers, a machine embedded in the scene | Clean catalog backgrounds, and hard negatives that are **look-alike machinery**. That pushed the class boundary away from the real target. |

**Root cause:** augmentation can only **widen** the training distribution around the images you have. It cannot **move** it to a viewpoint, sensor or context that isn't in the data. Real CCTV pixels are the fix, and augmentation makes the most of them.

### 3.3 Augmentations that do transfer, ranked by value per effort

1. **Codec/sensor degradation pipeline** (online, Albumentations). First run `env\yolo-env\Scripts\python.exe -m pip install albumentations`. Ultralytics 8.4.123 accepts a custom list through `model.train(augmentations=[...])` (`augment.py:2801,2842`).

   ```python
   import albumentations as A
   cctv_aug = [
       A.Downscale(scale_range=(0.25, 0.6), p=0.4),            # low-res sensor / sub-stream
       A.ImageCompression(quality_range=(20, 60), p=0.5),       # codec blocking
       A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7)), A.Defocus(radius=(1, 3))], p=0.3),
       A.OneOf([A.GaussNoise(std_range=(0.02, 0.08)), A.ISONoise()], p=0.3),
       A.ToGray(p=0.15),                                        # IR night
       A.RandomBrightnessContrast(0.3, 0.3, p=0.4),
       A.RandomGamma(gamma_limit=(60, 150), p=0.2),
       A.CLAHE(p=0.1),
       A.RandomSunFlare(p=0.03),                                # window glare
   ]
   ```

   It runs on every training image, CCTV included. At these probabilities, the extra degradation on CCTV frames does no harm.

2. **Offline "real codec" copies of the catalog set** (one-time). Resize each catalog image to 640×360 or 480×360, encode with `ffmpeg -c:v libx264 -b:v 400k`, decode, and add as a **duplicate** entry with the same labels. This reproduces true H.264 artefacts, which JPEG only approximates. Use it on 30–50% of catalog images.
3. **Copy-paste** (`copy_paste=0.3`, seg only). Once CCTV instances exist, this pastes them into other CCTV frames, multiplying contexts and occlusions.
4. **Compositing catalog cutouts onto empty CCTV backgrounds.** You already have the masks. Scale each cutout to a plausible floor-plane size and degrade it as in item 1. This helps **context and clutter**, but **not viewpoint**, because the cutout is still frontal.
5. **Elevated-view synthetic renders.** Use manufacturer CAD or 3D models in BlenderProc, with camera height 2.5–5 m, pitch 20–55°, randomised lighting and textures, and degradation applied. This is the only augmentation-style route that actually **creates the top-down appearance**. Medium effort; do it for classes with fewer than 10 real viewpoints.

**Turn down** the geometric augmentations that don't help: `degrees=0–3`, `perspective≤0.0002`, `scale=0.5`, `shear=0`.

---

## 4. Inference and architecture

### 4.1 Resolution and SAHI

**Fix the input stream first.**

| Source | Recommended inference |
|---|---|
| **Sub-stream 480×360 / 640×360** (today) | `imgsz=640`. Don't upsample to 1280. Sweep imgsz {480, 640, 960} on CCTV-val and keep the best. |
| **Main stream 1080p / 4 MP** (target) | `imgsz=1280` letterboxed. Machines that are 25–55% of the frame width will be **270–700 px**, inside the training scale range. |

**SAHI is not the right tool for this site.** The machines aren't small (§0 #1), and on a 480×360 stream tiling only re-samples the same pixels, adding cost without information. Consider it only if **both** of these hold:

- The camera is a wide-angle main stream (≥ 2560 px wide).
- Target objects are under ~3% of the frame width (under ~40 px at imgsz 1280). Extinguishers on far walls are the likely case.

Its cost is real: 4K with 1280 tiles at 20% overlap is about **6–9 forward passes per frame**. Run it at `vid_stride` ≥ 10 and pass `end2end` predictions straight through. `sahi` isn't installed.

**Better fit for fixed CCTV: static ROI crops.** Cameras and machines don't move. Define 1–3 ROIs per camera once, in a per-camera YAML. Crop each ROI from the **main stream**, run it at its native size, and map the masks back. This gives SAHI's gain with no tiling overhead and no edge-merge artefacts. Keep a low-rate full-frame pass, e.g. every 30th frame, to catch relocated items.

### 4.2 Temporal consistency

**All 9 classes are stationary objects**, extinguishers included. That changes the design: you need **presence confirmation per location**, not motion tracking.

**Recommended pipeline per camera:**

1. Decode at **5 fps** (`vid_stride` = fps/5). Stationary objects don't need 30 fps, and 5 fps lets one RTX 3070 Ti serve about **8–10 cameras** at imgsz 1280 (measured 17–22 ms/frame end-to-end).
2. Run detection at a **low** threshold: `conf=0.10`.
3. **Association.** Run `model.track(..., tracker="bytetrack.yaml", persist=True, conf=0.10)` per camera. ByteTrack's second matching stage uses **low-confidence detections** (between `track_low_thresh` and `track_high_thresh`) to extend existing tracks, which is exactly the problem of single-frame confidence dipping. Suggested `bytetrack.yaml` overrides:

   | Parameter | Value |
   |---|---|
   | `track_high_thresh` | 0.35 |
   | `track_low_thresh` | 0.10 |
   | `new_track_thresh` | 0.45 |
   | `track_buffer` | 50 (10 s at 5 fps) |
   | `match_thresh` | 0.8 |

   Use BoT-SORT only if the camera moves: its camera-motion compensation is wasted on fixed mounts. Ultralytics' default here is `tracktrack.yaml`, so set the tracker explicitly.
4. **Confirmation state machine per track or ROI**, with hysteresis:
   - **Confirm** when **≥ 8 of the last 15** frames (3 s) have a matched detection with mean conf ≥ a per-class threshold.
   - **Drop** when **≤ 2 of the last 15** frames match, held for `track_buffer`.
   - **Report the class** as the confidence-weighted vote over the track's lifetime. That stops single-frame class flips like the table_saw/panel_saw flips seen at f186.
5. **Per-class thresholds** come from the PR curves on **CCTV-val**, never catalog val. `test_inference.py` already supports per-class thresholds.
6. **Log for active learning.** Save frames where a confirmed track's per-frame conf falls under 0.3, or where the class vote is split. Those are next month's labelling queue.

---

## 5. Training recipe and roadmap

### 5.1 Recommendation: A now as a pilot, B for the release

| | **Option A: fine-tune** | **Option B: retrain on a unified dataset** |
|---|---|---|
| Base | **`9_class_clean_finetune\weights\best.pt`**, *not* `cctv_robust` (it regressed, §0 #2) | COCO-pretrained `yolo26?-seg.pt`, same model size as now |
| When | As soon as ~1–3 k CCTV frames and the CCTV-val set exist | After the class-boundary policy, edge-banding re-annotation and extinguisher re-labelling are done |
| Why | Fast (hours). Measures the learning curve and proves the data works. | Policy changes rewrite class boundaries. A checkpoint trained on the old boundaries (frame-filling edge banders, look-alikes) carries those biases. Retraining removes them and produces a clean, reproducible release. |

**Freezing: don't freeze the backbone for this gap.** The CCTV gap is mostly **low-level**: codec blocking, noise, blur, greyscale and low resolution. Those are encoded in the **early** layers, so freezing the backbone (`freeze=10`) locks exactly the features that need to adapt. Use a **low learning rate with replay** instead.

One exception: if the pilot has under ~500 CCTV frames, do a 2-stage run. Stage 1 is `freeze=10` for 5 epochs to settle the head. Stage 2 unfreezes everything at the Option A learning rate.

### 5.2 Hyperparameters (Ultralytics 8.4.123, RTX 3070 Ti 8 GB)

| Parameter | Option A (pilot fine-tune) | Option B (release retrain) | Note |
|---|---|---|---|
| `model` | `9_class_clean_finetune\...\best.pt` | `yolo26?-seg.pt` (COCO) | |
| `optimizer` | `AdamW` | `AdamW` | **Must be explicit**, or `auto` overrides `lr0` |
| `lr0` | **2e-4** | **1e-3** | |
| `lrf` / `cos_lr` | 0.05 / True | 0.01 / True | |
| `warmup_epochs` | 1 | 3 | |
| `epochs` / `patience` | 30 / 10 | 150 / 30 | |
| `imgsz` | 1280 if main stream, else 640 | same | Match the deployment stream (§4.1) |
| `batch` | 4 at 1280 (16 at 640) | same | `nbs=64` accumulates gradients |
| `freeze` | none (see §5.1 exception) | none | |
| `mosaic` / `close_mosaic` | 0.5 / 10 | 1.0 / 15 | |
| `scale` | 0.5 | 0.5 | |
| `degrees` / `shear` / `perspective` | 0 / 0 / 0.0002 | 3 / 0 / 0.0002 | |
| `translate` / `fliplr` | 0.1 / 0.5 | 0.1 / 0.5 | |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.6 / 0.5 | same | |
| `copy_paste` / `copy_paste_mode` | 0.3 / `flip` | 0.3 / `flip` | seg only |
| `mixup` | 0 | 0 | |
| `augmentations` | `cctv_aug` (§3.3) | `cctv_aug` | Needs `albumentations` installed |
| `box` / `cls` / `dfl` | 7.5 / 0.5 / 1.5 (default) | default | **Don't tune loss weights.** The bottleneck is data, and tuning them without a CCTV val set is noise. |
| `data` → `train` | hybrid list (§1.2 ratios) | hybrid list | |
| `data` → `val` | **CCTV-val** (held-out cameras) | **CCTV-val** | `best.pt` is chosen on the val split, so it must be the target domain |

Example pilot call:

```python
from ultralytics import YOLO
m = YOLO(r"runs\run5_9class_clean_finetune\weights\best.pt")
m.train(data=r"datasets\9class\data_9class_cctv.yaml", project="runs", name="run7_9class_cctv_pilot",
        optimizer="AdamW", lr0=2e-4, lrf=0.05, cos_lr=True, warmup_epochs=1, epochs=30, patience=10,
        imgsz=1280, batch=4, mosaic=0.5, close_mosaic=10, scale=0.5, degrees=0, shear=0,
        perspective=0.0002, translate=0.1, copy_paste=0.3, mixup=0, augmentations=cctv_aug, workers=2)
```

### 5.3 Release gates (all must pass before handoff)

1. **CCTV-test** (held-out cameras/site): per-class **mask mAP50 ≥ 0.60** and **recall ≥ 0.80 at the operating threshold**. Classes with fewer than 50 test instances are reported but not gated.
2. **Hard-negative set:** **≤ 1 false positive per camera-hour** after the temporal confirmation in §4.2, and ≤ 1% image-level FP rate before it. That needs ≥ 300 negative images (rule of three).
3. **Catalog val:** mask mAP50-95 drops by **≤ 2 points** against `9_class_clean_finetune`, showing no forgetting.
4. **Trial clips**, re-recorded from RTSP with no overlay: the new model must beat `9_class_clean_finetune` on the workshop saw.

### 5.4 Roadmap

| Step | When | Work | Output |
|---|---|---|---|
| **0. Stop the regression** | Day 0–1 | Make `9_class_clean_finetune` the base and fallback again. Re-test the workshop clip at imgsz 480/640/960. | Baseline numbers per imgsz |
| **1. Capture** | Week 1 | Inventory cameras × machines × sites. Record **raw RTSP main stream** for ≥ 24 h per camera, covering all shifts and IR night. Shoot pseudo-CCTV phone video (3–4 m, pitched down) of machines no camera covers. | `cctv_raw\<site>\<cam>\*.mp4`, camera inventory sheet |
| **2. Sample and split** | Week 1 | Run the §1.4 sampler. Assign whole cameras to **train / val / test**. Build the **CCTV-val, CCTV-test and hard-negative sets first.** | `cctv_val.txt`, `cctv_test.txt`, `hardneg_cctv.txt` |
| **3. Labelling policy and labels** | Week 1–2 | Finalise the class-boundary sheet and the §2 rules. Pre-label with the model plus SAM 2, correct, 10% double-label QA. | Labelled CCTV set, QA report |
| **4. Augmentation infrastructure** | Week 2 | `pip install albumentations`. Add `cctv_aug`. Make the offline H.264 catalog copies. Build `train_hybrid_9class.txt` at the §1.2 ratios. | `data_9class_cctv.yaml` |
| **5. Pilot (Option A) and learning curve** | Week 2 | Train on 25% / 50% / 100% of the CCTV frames and plot CCTV-val mAP. If it is still rising at 100%, collect more before Step 6. | Learning curve, pilot checkpoint |
| **6. Label fixes and Option B** | Week 3–4 | Re-annotate edge banding, re-label crowded extinguishers, apply the class policy, then retrain from COCO weights. | Release candidate |
| **7. Deployment pipeline** | Week 3–4 | Per-camera ROI YAML, 5 fps decode, ByteTrack and the confirmation state machine, per-class thresholds from CCTV-val. | Pipeline plus a gate report (§5.3) |
| **8. Active-learning loop** | Ongoing | Each week, label the logged low-confidence and split-vote frames. Fine-tune each month at the Option A settings with replay, and re-run the gates. | Monthly model updates |
