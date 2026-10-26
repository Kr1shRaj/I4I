# CCTV ingestion runbook

*Written 2026-10-15. All four steps were smoke-tested end to end on the trial videos (see "Smoke test" below).*

Run everything from the project root with `env\yolo-env\Scripts\python.exe`.

## 1. Sample frames — `scripts\tools\sample_cctv_video.py`

```powershell
# a recording, or a folder of NVR / ffmpeg segments, from one camera
env\yolo-env\Scripts\python.exe scripts\tools\sample_cctv_video.py --source D:\cctv_raw\factoryA\cam01 --site factoryA --cam-id cam01
# live RTSP (main stream), 20 min
env\yolo-env\Scripts\python.exe scripts\tools\sample_cctv_video.py --source "rtsp://user:pass@IP:554/Streaming/Channels/101" --site factoryA --cam-id cam03 --duration 1200
```

**What it does**
- Takes 1 frame every `--stride` seconds (default 3).
- Drops a frame unless its 64-bit dHash is at least `--min-hamming` (default 7) bits from **every** frame already kept for that camera, in this run or earlier ones.
- Keeps at most `--cap` frames per run (default 100), spread evenly over the recording.

**Output**
- Frames: `datasets\cctv\frames\<site>\<cam>\images\cctv_<cam>_f<frame_idx>_<YYYYmmdd-HHMMSS-mmm>.jpg`
- Log: `frames.csv` in the camera folder (source, frame index, timestamp, dHash).

**Options added 2026-10-19 (used for the factory drop)**
- `--method ssim --min-hamming 3`: keep a frame if SSIM to the last kept frame is < 0.92. Use this for fixed cameras: the default whole-frame dHash ≥ 7 kept only 73 of 328 useful frames.
- `--resize 1920x1080`: un-squeeze 960×1080 "1080P Lite" NVR video.
- `--dest <folder>`: one folder per recording.
- `--name-format plain`: `cctv_<id>_f<idx>.jpg`.
- `--dedup-with <folders>`: other recordings of the same camera.

**Notes**
- Site and camera IDs may use letters, digits and `-` only.
- The timestamp comes from `--start-time`, else the file name (`cam01_20261015_143000.mp4`), else the file's modification time.
- Existing frames are never overwritten. Credentials are removed from RTSP URLs before they're logged.

## 2. Pre-label — `scripts\tools\prelabel_cctv.py`

```powershell
env\yolo-env\Scripts\python.exe scripts\tools\prelabel_cctv.py --source datasets\cctv\frames\factoryA
```

- **Model and settings:** run5 `best.pt` at conf 0.15.
- **Inference size:** 640 for frames up to 960 px wide (sub-stream), else 1280. Override with `--imgsz`.
- **Output:** an Ultralytics-YOLO staging folder at `datasets\cctv\prelabels\<timestamp>\`, containing `data.yaml`, `train.txt`, `images\train`, `labels\train` and `prelabel_summary.csv`. Import it into CVAT as **"Ultralytics YOLO Segmentation 1.0"**, or into Roboflow.
- **Review order:** sort `prelabel_summary.csv` by `min_conf`. Check frames with no detection for missed objects.
- **After review:** export from CVAT and put the `.txt` files in `datasets\cctv\frames\<site>\<cam>\labels\`. An **empty** `.txt` marks a reviewed hard negative.
- **Never copy unreviewed pre-labels there.** The builder trains on whatever is in that folder.

## 3. Assign cameras and build — `scripts\tools\build_cctv_hybrid_dataset.py`

1. On the first run, the script writes `datasets\cctv\camera_splits.yaml` as a template. Edit it so each camera is in exactly one split:
   ```yaml
   factoryA/cam01: train
   factoryA/cam02: train
   factoryA/cam03: val
   factoryB/cam01: test
   ```
2. Build:
   ```powershell
   env\yolo-env\Scripts\python.exe scripts\tools\build_cctv_hybrid_dataset.py            # default shares 50/40/10
   env\yolo-env\Scripts\python.exe scripts\tools\build_cctv_hybrid_dataset.py --shares 0.55 0.35 0.10
   ```

**Also supported**
- Per-recording folders, with `<folder>: {camera: <id>, split: <split>}` entries. The build fails if one camera spans two splits.
- `--preview <prelabel dir>`: check balance and paths before QA. It writes only the frame lists, never a trainable list or YAML.

**Output** (in `datasets\9class\`):
- `train_hybrid_9class.txt`, `cctv_train_frames.txt`, `cctv_val.txt`, `cctv_test.txt`
- `data_9class_cctv.yaml` (val = CCTV-val)
- `build_cctv_hybrid_stats.json`

**The build fails, and writes no YAML, if:**
- a dataset junction doesn't resolve, or any catalog entry is missing its image or label;
- a labelled camera has no split;
- a label has a class outside 0–8, fewer than 3 points, or a coordinate outside [0, 1];
- a frame in one split is a near-copy of a frame in another (dHash ≤ 3, e.g. a copied file or two cameras sharing a view);
- there are no CCTV-val frames.

**Shares:**
- CCTV frames are repeated at most 5×.
- If that isn't enough, the catalog block is subsampled, but never below 1,500 entries. Past that floor, the script reports a lower achieved share instead of forcing it.
- With the 3,312-entry catalog, about 300 labelled train frames reach 50/40/10 exactly. Fewer than that falls short.

## 4. Train — `scripts\train.py`

```powershell
env\yolo-env\Scripts\python.exe scripts\train.py                          # main stream: imgsz 1280, batch 4
env\yolo-env\Scripts\python.exe scripts\train.py --imgsz 640 --batch 16   # sub-stream cameras (plan 4.1)
```

- **Base and output:** fine-tunes from run5 and writes to `runs\run7_9class_cctv_pilot\`. It refuses to overwrite an existing run.
- **Recipe:** AdamW with lr0 2e-4, lrf 0.05, `cos_lr`, warmup 1, 30 epochs, patience 10, mosaic 0.5, close_mosaic 10, scale 0.5, degrees 0, shear 0, perspective 0.0002, copy_paste 0.3, mixup 0.
- **Augmentation check:** the script stops if `albumentations` is missing. Ultralytics also swallows Albumentations errors silently, so **confirm that the training log has an `albumentations: Downscale(...)` line**. `args.yaml` shows `augmentations:` empty because the list can't be serialised; that is normal.
- **`best.pt` is selected on CCTV-val.** Check catalog regression separately:
  ```powershell
  yolo segment val model=runs\run7_9class_cctv_pilot\weights\best.pt data=datasets\9class\data_9class.yaml
  ```

## Environment notes
- `albumentations` 2.0.8 and `albucore` were installed with `--no-deps`, so they don't pull in `opencv-python-headless`, which would replace the GUI build of `cv2`.
- Expect `pip check` to report "requires opencv-python-headless". That warning is harmless here, because `cv2` comes from `opencv-python` 5.0.0.93.
- Ultralytics writes each list's `labels.cache` next to the labels of the first image in that list, e.g. `datasets\cctv\frames\<site>\<cam>\labels.cache`. The catalog cache is not touched.

## Smoke test (2026-10-15)

| Step | Result |
|---|---|
| Sample | Both trial videos used as `smoke/cam01` and `smoke/cam02`: 16 frames, 30 near-duplicates dropped. A re-run added 0. A cap of 5 gave 5 frames spread from 0 s to 46 s. A bad ID was rejected. |
| Pre-label | 16 label files and 6 instances, 0 invalid. Coordinates within [0.008, 0.998]. Overlays line up with the image. |
| Build | 3,312/3,312 catalog entries resolve, and the junctions are OK. Camera split cam01 → train, cam02 → val. A planted cross-split copy, class 12 and a coordinate of 1.3 were all caught. |
| Load | Ultralytics `YOLODataset` loaded all 1,545 hybrid-train entries (1,500 catalog via the junctions + 45 CCTV) and 7 val frames, with 0 corrupt. |
| Train | `train.py` ran 1 epoch (fraction 0.03, imgsz 640): exit 0, AdamW at lr 0.0002, `albumentations:` line present, val ran on the CCTV camera. |

**Not tested:** a live RTSP stream (there was no camera or RTSP server here). The code path uses wall-clock stride and retries 50 times on dropped frames. Test it once against a real camera with `--duration 60`.

**Smoke artefacts** can be deleted once you've looked at them:
- `datasets\_smoketest\`
- `datasets\9class\*_smoke.*`, `train_hybrid_9class_smoke.txt`, `cctv_val_smoke.txt`, `cctv_test_smoke.txt`
