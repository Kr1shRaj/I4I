# Factory CCTV benchmark (frozen 2026-10-21)

**This is a permanent, strictly held-out evaluation set. It must never be used for training, validation-based checkpoint selection, threshold tuning or hard-negative mining.** Any training data from Camera 03 or Camera 05, including other frames from `factory_dat\`, also counts as leakage.

## Contents

| | |
|---|---|
| Frames | 328, in `datasets\cctv_factory_intake\<recording>\images\` (1920×1080, de-squeezed from 960×1080 "1080P Lite") |
| Labels | `…\labels\`: `panel_saw` 257, `hydraulic_press` 328, `edge_banding_machine` 328 instances |
| Cameras | cam03: 71 frames (2 recordings, 09-07 and 09-24); cam05: 257 frames (3 recordings, 09-25 and 09-28) |
| Label method | SAM 2.1 box prompts, clipped to each machine's consensus outline, with workers and OSD text subtracted, and at least 40% visibility (`docs\factory_footage_analysis.md` §4.5). Spot-checked; not hand-drawn |
| Classes absent | table saw, planer, jointer, band saw, mitre saw, fire extinguisher. **Any detection of these is a false positive.** Camera 03 has a white/yellow plastic drum that earlier models called `fire_extinguisher` |

## How to evaluate

```powershell
env\yolo-env\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'<weights>').val(data=r'datasets\9class\data_9class_cctv_benchmark.yaml', imgsz=960, plots=False)"
```

**Report:**
- per-class and per-camera mask mAP50;
- recall at the operating threshold;
- false positives per frame for absent classes, especially `fire_extinguisher` on Camera 03.

## Guards
- `datasets\cctv_factory_intake\camera_splits.yaml` marks every recording as `test`, so `build_cctv_hybrid_dataset.py` can't put a frame into a training list.
- The run7 hybrid training lists that used Camera 05 are archived in `archive\run7_cctv_pilot_2026-10\datasets_9class\`.

## Known label limitations
- Plywood lying on the saw table is inside the `panel_saw` outline.
- The far end of the sliding carriage is not labelled.
- Press edges are approximate where board stacks partly hide them.
- Labels are correlated across frames: each camera is effectively one viewpoint.
