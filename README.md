# Machine Type Classifier — Dataset & Instance Segmentation Pipeline

Automated dataset generation, verification, and auto-annotation pipeline for a 23-class industrial/woodworking Machine Type instance segmentation model.

## Quick Start

```powershell
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch with CUDA support (check your GPU / CUDA driver first: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Pipeline Architecture & Workflow

The pipeline follows a 6-stage progression: **Scrape → Clean → Filter → Diversify → Auto-Label → Train**.

| Stage | Script | Description |
|---|---|---|
| 1a | `pipeline/utils/manifest_gen.py` | Scan flat ref-image folder → blank manifest CSV |
| 1b | `pipeline/stage1_ingest_refs.py` | Ingest filled manifest → `reference_images/<class>/` |
| 2  | `pipeline/stage2_scrape.py` | Multi-engine web scraper (5 query templates per class) |
| 3  | `pipeline/stage3_clean.py` | Dedup, corrupt-file removal, resolution filter |
| 4  | `pipeline/stage4_clip_filter.py` | Hybrid OpenCLIP filter (coarse text + fine image similarity) |
| 4b | `pipeline/stage4b_calibrate.py` | Calibrate OpenCLIP thresholds from labeled sample |
| 5  | `pipeline/stage5_diversity_kselect.py` | Diversity dedup + K-selection with stopping rule |
| 6  | `pipeline/upload_polygon_annotations.py` | **Auto-Label & Ingest** — Gemini Vision class verification + SAM2 polygon contour extraction to Roboflow |

---

## Reference Image Ingestion Workflow

```powershell
# Step 1: Generate blank manifest (point at flat folder of reference images)
python -m pipeline.utils.manifest_gen --src C:\path\to\your\ref_images

# Step 2: Open reference_manifest.csv and fill in the 'class' column
#         Use exact slugs from pipeline/config.py (e.g. band_saw, table_saw)

# Step 3: Ingest
python -m pipeline.stage1_ingest_refs
```

---

## Output Layout

```
dataset/
  train/
    <class_name>/        ← Final cleaned images per class
reference_images/
  <class_name>/          ← Hand-picked reference images (confusable clusters, used in Stage 4)
logs/                    ← Per-stage JSON logs and annotation audit outputs
```

---

## Supported Machine Classes (23 Classes)

| Class Slug | Display Name | Confusable Cluster |
|---|---|---|
| `band_saw` | Band Saw | cutting |
| `cnc_milling` | CNC Milling | cnc_group |
| `cnc_router` | CNC Router | cnc_group |
| `control_panel` | Control Panel | — |
| `conveyor` | Conveyor | — |
| `crane` | Industrial Crane | — |
| `drilling` | Drilling | cnc_group |
| `fire_extinguisher` | Fire Extinguisher | — |
| `forklift` | Forklift | — |
| `grinding` | Grinding | — |
| `hydraulic_press` | Hydraulic Press | — |
| `injection_molding` | Injection Molding | — |
| `jointer` | Jointer | surfacing |
| `lathe` | Lathe | — |
| `milling` | Milling | cnc_group |
| `miter_saw` | Miter Saw | cutting |
| `packaging_machine` | Packaging Machine | — |
| `panel_saw` | Panel Saw | cutting |
| `planer` | Planer (Thicknesser) | surfacing |
| `sanding_machines` | Sanding Machines | — |
| `spindle_moulder` | Wood Router / Spindle Moulder | surfacing |
| `table_saw` | Table Saw | cutting |
| `wood_lathe` | Wood Lathe | surfacing |

---

## Model Training & Validation Results

### 1. 9-Class Curated CCTV Fine-Tune (`run8_9class_curated_cctv` — Champion Model)
Trained for 50 epochs at `imgsz=960` with full SAM 2.1 dataset curation (quarantined mislabeled images, corrected masks with SAM 2.1 Large, 107 look-alike hard negatives, and CCTV codec/sensor/lens augmentations).

| Metric Group | Metric | Base (Run 5) | Curated Run 8 | Absolute Gain |
| :--- | :--- | :---: | :---: | :---: |
| **Polygon Mask** | **mAP@50 (M)** | 74.1% | **86.7%** 🏆 | **+12.6%** |
| | **mAP@50-95 (M)** | 61.5% | **76.0%** 🏆 | **+14.5%** |
| | Precision (M) | 77.4% | **89.2%** | +11.8% |
| | Recall (M) | 70.3% | **78.5%** | +8.2% |
| **Bounding Box** | **mAP@50 (B)** | 78.9% | **87.9%** 🏆 | **+9.0%** |
| | **mAP@50-95 (B)** | 60.9% | **77.0%** 🏆 | **+16.1%** |
| | Precision (B) | 79.3% | **89.5%** | +10.2% |
| | Recall (B) | 71.5% | **78.8%** | +7.3% |

#### Per-Class Performance Breakdown (450 Validation Images)
| Class Name | Instances | Mask mAP50 | Mask mAP50-95 | Box mAP50 |
| :--- | :---: | :---: | :---: | :---: |
| `jointer` | 40 | **97.2%** | **93.3%** | **97.2%** |
| `hydraulic_press` | 58 | **95.8%** | **81.8%** | **97.3%** |
| `table_saw` | 41 | **91.8%** | **81.4%** | **93.5%** |
| `band_saw` | 62 | **91.1%** | **85.1%** | **93.6%** |
| `planer` | 36 | **83.5%** | **77.8%** | **83.5%** |
| `panel_saw` | 53 | **83.0%** | **71.0%** | **88.0%** |
| `edge_banding_machine` | 22 | **80.5%** | **68.6%** | **80.5%** |
| `miter_saw` | 47 | **79.3%** | **66.4%** | **81.0%** |
| `fire_extinguisher` | 109 | **78.0%** | **58.9%** | **76.0%** |

---

## Real-World Factory CCTV Benchmark Evaluation

Evaluated against the frozen 328-frame surveillance benchmark (`docs/cctv_benchmark.md`, Camera 03 & Camera 05):

* **~75% Reduction in False Alarms:** False detections dropped from 34 (run5) down to 8 (run8). Hallucinated whole-frame `edge_banding_machine` boxes were completely eliminated.
* **Hard Negative Rejection:** Lockers, workbenches, metal cabinets, and chemical drums are cleanly rejected with zero false positives.
* **Key Perspective Finding:** While pixel degradation (blur, sensor noise, H.264 compression) was resolved by Albumentations augmentations, steep 60° bird's-eye ceiling perspectives cause panel saws to resemble table saws. Future iterations require genuine high-angle training captures.
* Detailed evaluation report: [`docs/run8_cctv_benchmark_eval.md`](docs/run8_cctv_benchmark_eval.md).

---

## Project Structure & Tools

* `scripts/train.py` — Training engine with explicit AdamW, cosine annealing, and custom CCTV lens/compression transforms.
* `scripts/tools/eval_cctv_benchmark.py` — Automated evaluation pipeline for real-world surveillance benchmarks.
* `scripts/tools/parse_user_curation.py` — Curation parser for dataset sanitization.
* `docs/viewer/` — Interactive web-based dataset and annotation review gallery.
* `configs/` — Dataset splits and 9-class YAML configuration files.
