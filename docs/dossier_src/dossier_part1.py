"""Title and chapters 1-3 (v2.6: 15-20 pages, bullet-driven, no table of contents)."""

G, R, A, B, Y = "green", "red", "amber", "blue", "grey"


def front(d):
    d.title_block(
        "Industrial Machine Instance Segmentation & CCTV Domain Adaptation Dossier",
        "Architecture, Data Pipelines, Empirical Failure Modes, and Production Readiness for 9-Class YOLO26-seg",
        [
            "**Project Version 2.6  |  Status: Active Engineering  |  Target: Factory Safety & Inventory**",
            "**Date:** 23 October 2026  |  Written as the Krish team's account; compiled by Claude (Claude Code) from project artifacts.",
        ],
    )


# =====================================================================================================
def ch1(d):
    d.h(1, "1. Executive Overview & the 9-Class Taxonomy", new_page=False)
    d.big_picture(
        "Catalog photos are a solved problem: our models have scored above 80% mAP50 on them since September. Real "
        "factory CCTV is not: our best model scores 0.054 mask mAP50 on the factory benchmark. This document is about that gap.",
    )
    d.h(2, "1.1 Project Goal")
    d.bullets([
        "**Inventory:** know which machines each fixed camera sees, and that they stay put.",
        "**Safety zones:** turn each machine's mask into a hazard perimeter around saws, planers, jointers and presses.",
        "**Fire safety:** confirm extinguishers are on the wall. This is an assistive signal, **never a compliance alarm**: "
        "our best run still misses 46% of them.",
    ])
    d.h(2, "1.2 The Nine Classes")
    d.table(
        ["ID", "Class", "How we recognise it", "Main risk (measured)"],
        [["0", "`panel_saw`", "Long sliding carriage + outrigger table", "Called `table_saw` by run5 (up to 0.70)"],
         ["1", "`hydraulic_press`", "Heavy frame, platens, cylinders", "Veneer / hot presses look the same"],
         ["2", "`table_saw`", "Blade through a flat table, rip fence", "Panel saws; benchtop saws"],
         ["3", "`planer`", "Enclosed box, stock fed underneath", "Jointers (6% swapped), sanders"],
         ["4", "`jointer`", "Long open beds, cutter on top, fence", "Planers"],
         ["5", "`band_saw`", "Two wheel housings, vertical blade", "Drill presses, mitre saws"],
         ["6", "`miter_saw`", "Pivoting arm over a turntable", "28% missed (run6)"],
         ["7", "`edge_banding_machine`", "Long enclosed body, tape coil, conveyors", "Lowest AP; old labels covered whole images"],
         ["8", "`fire_extinguisher`", "Red cylinder, handle, hose", "Drums, tubes, workers (run7 false alarms)"]],
        align="clll", widths=[0.75, 3.25, 5.3, 5.5], size=8)
    d.h(2, "1.3 Executive Scorecard")
    d.table(
        ["Run", "Classes / base / imgsz", "Box mAP50", "Mask mAP50", "Verdict"],
        [["run1", "23 / COCO / 960", "0.777", "0.755", ("badge", "SUPERSEDED", Y)],
         ["run2", "23 / run1 / 960", "0.787", "0.765", ("badge", "SUPERSEDED", Y)],
         ["run3", "38 / COCO / 960", "0.608", "0.544", ("badge", "RUN5'S PARENT", Y)],
         ["run4", "38 / run3 / 960", "0.611", "0.552", ("badge", "SUPERSEDED", Y)],
         ["run5", "9 / run3 / 960", "0.825", "0.802", ("badge", "CHAMPION: FINDS THE REAL SAW", G)],
         ["run6", "9 / run5 / 1280", "0.810", "0.795", ("badge", "FAKE WIN: 0 DETECTIONS ON VIDEO", R)],
         ["run7", "9 / run5 / 1280", "0.786*", "0.767*", ("badge", "MEMORISED ONE CAMERA", A)],
         ["run8", "9 / run5 / 960", "0.830†", "0.813†", ("badge", "TRAINING", B)]],
        align="llrrc", widths=[1.2, 3.6, 2, 2, 6], size=8)
    d.bullets([
        "**Read within an era:** each era validates on a different set. *run7: catalog val at imgsz 960. †run8: epoch 1 only.",
        "**Today's baseline:** run5 = box 0.789 / mask 0.741 on the cleaned val (461 images). Re-measure it on the current "
        "450-image val before comparing with run8.",
    ])


# =====================================================================================================
def ch2(d):
    d.h(1, "2. Core Tech Stack & System Architecture")
    d.big_picture("One Windows workstation with an 8 GB GPU. That shaped our batch sizes, the zero-copy dataset layout and the choice of an NMS-free model.")
    d.h(2, "2.1 Stack")
    d.bullets([
        "**Hardware:** RTX 3070 Ti (8 GB), Windows 11. Batch 6 at imgsz 960, batch 4 at 1280.",
        "**Software:** Python 3.10 venv `env\\yolo-env`, PyTorch 2.13.0+cu130, Ultralytics 8.4.123, Albumentations 2.0.8.",
        "**Model:** YOLO26s-seg with an end-to-end, NMS-free head: 12–16 ms per frame at 1280 (45–60 FPS end-to-end), so "
        "one GPU serves 8–10 cameras at 5 fps.",
        "**The head's catch:** `iou=` does nothing unless you pass `end2end=False` (or `--nms`). That is worth testing on "
        "crowded extinguishers, where recall drops to 39% with three or more in the frame.",
    ])
    d.h(2, "2.2 Workspace Layout")
    d.table(
        ["Path", "What it is"],
        [["`datasets\\9class\\*_9class\\images`", "NTFS junction to `source_38class` images (stored once, never copied)"],
         ["`datasets\\9class\\*_9class\\labels`", "9-class labels, rewritten on every build"],
         ["`datasets\\9class\\*.txt`, `data_9class.yaml`", "The dataset definition: training reads only the listed images"],
         ["`datasets\\cctv_factory_intake\\`", "Frozen factory benchmark: test only"]],
        align="ll", widths=[6.4, 8.4], size=8)
    d.callout("warn", "two rules that protect the data", [
        "- **Never delete inside `*_9class\\images`:** it deletes the originals in `source_38class`.",
        "- **Never `Remove-Item -Recurse` a junction:** use `cmd /c rmdir`. If the project moves, re-link with the loop in `CHANGELOG_REORGANIZATION.md`.",
    ])
    d.h(2, "2.3 Tools You Will Actually Use")
    d.table(
        ["Tool", "Job"],
        [["`scripts/train.py`", "run8 recipe; refuses to run without Albumentations, refuses the benchmark, never overwrites a run"],
         ["`scripts/test_inference.py`", "Image / folder / video inference; per-class thresholds, `--nms`, CSV + Markdown logs"],
         ["`tools/build_clean_9class_dataset.py`", "Builds labels and lists from `source_38class`: quarantines, fixes, negatives"],
         ["`tools/fix_user_labels.py`", "SAM 2.1 box-prompt re-annotation with before/after review sheets"],
         ["`tools/sample_cctv_video.py`", "De-duplicated CCTV frame sampling; `--resize 1920x1080` un-squeezes 1080P Lite"],
         ["`tools/build_dataset_viewer.py`", "Browsable gallery of exactly what training sees (`docs/viewer/`)"]],
        align="ll", widths=[5.4, 9.4], size=8)
    d.h(2, "2.4 Essential Commands")
    d.code(r"""
env\yolo-env\Scripts\python.exe scripts\tools\build_clean_9class_dataset.py   # rebuild labels and lists
env\yolo-env\Scripts\python.exe scripts\tools\audit_dataset_distribution.py   # must say PASS
env\yolo-env\Scripts\python.exe scripts\train.py --name run9_<description>    # run8 recipe, from run5
env\yolo-env\Scripts\python.exe scripts\test_inference.py --weights <best.pt> --source <video> ^
    --imgsz 960 --conf 0.40 --class-conf "edge_banding_machine=0.20,fire_extinguisher=0.30"
""")


# =====================================================================================================
def ch3(d):
    d.h(1, "3. The I4I Web Scraping & Ingestion Engine")
    d.big_picture(
        "Before we had plant access, we had no images of most of these machines. I4I (`pipeline/`) harvests web photos per "
        "class, filters them with CLIP, removes duplicates and auto-labels them with Gemini + SAM 2. It built the data behind "
        "every model from run1 onwards. It could not find CCTV views, and Section 3.8 explains why.",
    )
    d.h(2, "3.1 Why Scrape?")
    d.bullets([
        "**Bootstrap rare classes:** edge banders, jointers and panel saws barely exist in public datasets. A scrape-and-filter loop builds a class in hours.",
        "**Diversity by design:** each class is expanded across query templates (angle, distance, workers, occlusion) instead of hoping for variety.",
        "**Known risk:** the design document warned that web photos are “catalog-style” and CCTV is not. Its "
        "real-world canary set (`dataset/real_validation/`) was never built.",
    ])
    d.table(
        ["Stage", "Script", "Gate"],
        [["1 References", "`stage1_ingest_refs.py`", "≥ 3 hand-picked exemplars per confusable class"],
         ["2 Scrape", "`stage2_scrape.py`, `batch_topup_300.py`", "Keyword blocklist, decode check, ≥ 200 px"],
         ["3 Clean", "`stage3_clean.py`", "Corrupt files, ≥ 150 px, pHash ≤ 8"],
         ["4 / 4b CLIP", "`stage4_clip_filter.py`, `stage4b_calibrate.py`", "Text ≥ 0.18; reference similarity ≥ 0.30"],
         ["5 Diversity", "`stage5_diversity_kselect.py`", "Drop if cosine > 0.95 to a kept image"],
         ["6 Auto-label", "`upload_polygon_annotations.py`", "Gemini class check, then a SAM polygon"]],
        align="lll", widths=[2.6, 6.2, 6], size=8)

    d.h(2, "3.2 Stage 1: Reference Exemplars")
    d.bullets([
        "**What:** a person lists unambiguous photos in `reference_manifest.csv`; `stage1_ingest_refs.py` copies them to "
        "`reference_images/<class>/` (364 images across 36 classes today).",
        "**Why:** CLIP's text sense cannot separate a planer from a jointer. The references become the visual definition of each confusable class.",
        "**Example:** planer has 30 references, the most of any class, because its photos are the least alike (self-similarity as low as 0.17).",
    ])
    d.h(2, "3.3 Stage 2: Multi-Query Scraping")
    d.bullets([
        "**Engine:** DuckDuckGo through `ddgs`, whose image results are Bing-backed. It returns direct image URLs and needs "
        "no API key. It replaced an earlier icrawler/Bing HTML scraper; there is no Google scraper.",
        "**Query expansion:** 15 generic templates, hand-written per-class lists (“sliding panel saw machine”, "
        "“automatic edgebander machine close up detail”), plus `cctv` (10) and `highangle` (8) sets.",
        "**Politeness:** the class cap is split across templates; 1.5 s between queries; exponential back-off (10 s, doubling, 5 tries).",
        "**Integrity:** HTTP 200 only, Pillow `verify()`, both sides ≥ 200 px, re-saved as JPEG at quality 90.",
        "**Blocklist:** diagrams, clipart, logos, renders; in CCTV mode, camera-product terms (dome / bullet / IP camera, NVR, DVR).",
        "**Fixed on 21 Oct:** URLs are now de-duplicated across templates and classes (537 repeat downloads), and each template stops at its quota.",
    ])
    d.h(2, "3.4 Stage 3: Integrity & De-duplication")
    d.bullets([
        "**URL hashing:** files are named `img_<class>_<tag>_<idx>_<md5(url)>.jpg`; together with the seen-URL set, this stops repeat downloads.",
        "**Hygiene:** corrupt files and anything under 150 px are deleted (the download step already enforces 200 px).",
        "**Perceptual de-dup:** pHash, Hamming distance ≤ 8, within each class.",
        "**Gap:** no byte-level or cross-class check. The CCTV batch had 27% byte copies and 347 files saved under two classes.",
    ])
    d.h(2, "3.5 Stage 4 & 4b: OpenCLIP Filtering")
    d.bullets([
        "**Coarse tier (text):** cosine to “a photo of a {class}” ≥ 0.18, and not beaten by 20 non-photo prompts "
        "(diagram, CAD, clipart) by more than 0.05. Removes junk, not look-alikes.",
        "**Fine tier (image):** for confusable classes, mean similarity to the class's references (a dot product with the "
        "reference centroid) must be ≥ 0.30.",
        "**Rival checks:** drop an image if it is closer to a rival's references: drum vs belt sander, edge bander vs CNC "
        "router, hoist vs crane, PPE station vs racking.",
        "**Calibration (4b):** histograms of a hand-sorted good and bad sample. Coarse cut = midpoint of the bad max and "
        "the good 1st percentile; fine cut = the good 1st percentile. It favours recall, and it moved the fine cut from 0.55 to 0.30 after it rejected real planers.",
    ])
    d.h(2, "3.6 Stage 5: Diversity Ceiling")
    d.bullets([
        "**Rule:** rank by CLIP text score, then keep an image only if its cosine similarity to everything already kept is ≤ 0.95.",
        "**Floors:** 80 images for standard classes, 150 for confusable ones; shortfalls are flagged, not padded.",
        "**Limit:** it kills re-uploads, not genre. White-background catalog shots still dominate what survives.",
    ])
    d.h(2, "3.7 Stage 6: Auto-Labelling (Gemini Vision + SAM 2 in Roboflow)")
    d.bullets([
        "**Loop:** a Roboflow workflow, where Gemini verifies the expected class (conflicts go to review), SAM traces the "
        "polygon, and the script uploads it as COCO (70/15/15 per-class split, seed 42).",
        "**Output:** this loop, plus later uploads for the added classes, fed the 38-class Roboflow export (`source_38class`) we still build from.",
    ])
    d.callout("warn", "one mask per image", [
        "The script uploads only the **first** prediction. In an image with two saws or a row of extinguishers, every other "
        "instance silently became background. That likely contributed to the crowded-extinguisher and multi-machine fixes in Chapter 4.",
    ])

    d.h(2, "3.8 The CCTV Reality Check")
    d.p("After run6 failed on video, we pointed the engine at CCTV: 2,677 images from the `cctv` templates (10–14 Oct).")
    d.table(
        ["Finding", "Result"],
        [["Byte-identical copies", "723 (27%), 537 of them a scraper bug"], ["Camera hardware and NVR adverts", "~279 (~10%)"],
         ["Catalog or eye-level photos", "~1,700 (~63%)"], ["Real CCTV views of a target machine", "**~5, none clearly one of our nine**"]],
        align="lr", widths=[8, 6.8], size=8)
    d.figure("pipeline/logs/cctv_scrape_analysis/sheet_cctv_candidates.jpg",
             "What CLIP rated “most CCTV-like”: security-camera stock photos and extinguisher-factory adverts.", 10.5, d._root)
    d.bullets([
        "**Why it failed:** real CCTV of these machines lives on private NVRs, not the indexed web; “surveillance” "
        "queries found pages *about cameras*; and even good hits were front-on, studio-lit photos.",
        "**The deeper problem:** a ceiling camera sees the table top from above. Those pixels are not in any front-on photo, and no filter can create them.",
        "**The fix:** place-based `highangle` queries plus an acceptance gate (30 usable images of 113), and, above all, "
        "footage from our own cameras or pseudo-CCTV video shot from 3–4 m up.",
    ])
