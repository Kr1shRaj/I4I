"""Chapters 4-8 (v2.6: bullet-driven, 15-20 pages)."""

G, R, A, B, Y = "green", "red", "amber", "blue", "grey"


def ch4(d):
    d.h(1, "4. Label Cleanup & Spatial Policy Overhaul")
    d.big_picture(
        "run6's two weakest classes were not short of data; they were badly labelled. 45% of val edge-bander labels covered "
        "the whole image, and touching extinguishers were merged into blobs. On 21 October we rebuilt the labels around a "
        "written policy (`label_decisions.yaml`, applied by `build_clean_9class_dataset.py`).",
    )
    d.h(2, "4.1 Edge Banders: One Machine, One Polygon")
    d.bullets([
        "**The problem:** a morphological “merge” step closed gaps of 0.5–1.2% of the image diagonal, so fragments ballooned "
        "into slabs. The model learned “edge bander = the whole photo” and found 0 of 15 small ones.",
        "**The policy:** chassis, hood, gluing and trimming units, attached conveyors and the tape coil are in. Panels on the "
        "conveyor, the room, text banners and collage tiles are out. Two machines means two polygons.",
        "**The fix:** merge turned off. SAM 2.1 re-proposed all 175 frame-filling polygons: 68 accepted, 47 kept as genuine close-ups, 60 quarantined.",
        "**The result:** boxes over 80% of the image fell from 48% to 20% (train) and from 59% to 18% (val).",
    ])
    d.h(2, "4.2 Fire Extinguishers: Every Cylinder Counts")
    d.bullets([
        "**The problem:** 348 fragments in one image fused into a single blob covering three cylinders. Recall was 86% for a lone cylinder and 39% with three or more.",
        "**The policy:** one polygon per cylinder, including its own hose, handle and strap. Label every one that is at least 40% visible.",
        "**The fix:** `label_instances.py` keeps neighbours separate; SAM drew per-cylinder masks in 15 images; 4 dense, blurred rows were quarantined.",
        "**The result:** instances went from 435 / 105 / 96 to 606 / 129 / 147 (train / val / test), and are 586 / 109 / 140 after the final curation.",
    ])
    d.h(2, "4.3 SAM 2.1 Base vs Large")
    d.table(
        ["Model", "Mean IoU", "Boundary F1", "Jointer", "Mitre saw", "ms / prompt", "GPU MB"],
        [["SAM 2.1 Base", "0.880", "0.794", "0.928", "0.914", "108", "776"],
         ["SAM 2.1 Large", "0.886", "0.802", "0.971", "0.877", "271", "1,423"]],
        align="lrrrrrr", widths=[3, 1.9, 2.1, 1.8, 2, 2, 2], size=8)
    d.bullets([
        "**Verdict:** Large is the default, but the gain is small (+0.006 IoU), it is 2.5× slower, and it loses on mitre saws, so every mask is still reviewed.",
        "**House rules:** clip every mask to its prompt box; invert a mask only when the box covers more than 80% of the "
        "image; on white studio shots, cut the object out of the non-white pixels instead of using SAM.",
    ])
    d.h(2, "4.4 Quarantine, Negatives & Manual Curation")
    d.bullets([
        "**64 quarantined:** 60 edge-bander collages, infographics and multi-machine scenes, plus 4 dense extinguisher rows (38 train / 16 val / 10 test). They wait for hand re-annotation.",
        "**Veneer presses removed from the negatives:** the Camera 03 press is a woodworking hot press, so 12 veneer-press "
        "negatives were teaching it to be background. 5 more negatives turned out to contain real target machines. Negatives went from 295 / 60 / 65 to 283 / 58 / 62.",
        "**107 hard negatives:** drums, cardboard tubes, gas cylinders, water jugs, red bins, hose reels, alarm boxes, "
        "floors, ceilings and workbenches, aimed squarely at extinguisher false alarms.",
        "**`datasetchange` curation:** 80 bad images removed at build time, 110 labels re-drawn with SAM 2.1 L (128 instances), 3 excluded. Audit: PASS.",
        "**High-angle images:** 30 web images (52 instances), train only. They are the first elevated views the model has seen.",
        "**Caveat:** every review was done by Claude, not a human annotator. Spot-check the sheets in `label_fixes/` before a release.",
    ])


def ch5(d, root):
    d.h(1, "5. The Trial of the Models: Runs 5, 6, 7 & 8")
    d.big_picture(
        "run5 set the bar. run6 and run7 were our attempts to make it CCTV-ready, and both made it worse in ways validation "
        "could not see. run8 is the first run built on clean labels and augmentation that actually runs.",
    )
    d.h(2, "5.1 Run 5: The Baseline Champion")
    d.bullets([
        "**Recipe:** 20-epoch fine-tune of the 38-class run3 on 9-class catalog data, imgsz 960, 37 minutes.",
        "**Why it works:** it is the only model that finds the held-out workshop saw (0.56–0.63 at imgsz 960), and it raises zero extinguisher false alarms on Camera 03.",
        "**Its flaw:** on 200 sampled frames it finds that saw every time but calls it `table_saw` (max 0.70). The class boundary in our labels is wrong, not the model.",
        "**Status:** deployment fallback and the parent of every fine-tune. Factory benchmark: still only 0.054 mask mAP50.",
    ])
    d.table(
        ["Box AP50", "panel", "press", "table", "planer", "jointer", "band", "mitre", "edge b.", "extinguisher", "all"],
        [["run5", "0.893", "0.963", "0.937", "0.849", "0.877", "0.883", "0.783", "0.574", "0.718", "**0.831**"],
         ["run6", "0.848", "0.961", "0.931", "0.846", "0.869", "0.840", "0.733", "0.584", "0.680", "**0.810**"]],
        align="lrrrrrrrrrr", widths=[1.5, 1.2, 1.2, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.9, 1.2], size=7.5)
    d.figure("runs/run5_9class_clean_finetune/BoxPR_curve.png",
             "run5 per-class precision-recall: edge bander (0.574) and extinguisher (0.718) trail every machine class.", 11.5, root)
    d.h(2, "5.2 Run 6: The Fake Win")
    d.bullets([
        "**Recipe:** run5 at imgsz 1280, 50 epochs, 301 machinery negatives, heavier geometric augmentation.",
        "**On paper:** box 0.810 / mask 0.795, precision 0.87. Looks like a solid model.",
        "**On video:** the workshop saw fell to ≤ 0.18, with **0 detections at conf 0.40 in 1,398 frames**, plus 29 false extinguishers in 16 of 40 factory frames.",
        "**Why it failed:** spindle moulders, CNC routers and racking, used as negatives, share the sliding-table saw's long, "
        "flat shape, so we taught that shape to be background. The degradation augmentation never ran (Albumentations was "
        "missing). And the val set held no CCTV, so it could not see any of this.",
    ])
    d.h(2, "5.3 Run 7: The Memorization Disaster")
    d.bullets([
        "**Recipe:** run5 plus 257 Camera 05 frames repeated 5× (44% of every epoch), validated on Camera 03.",
        "**Result:** **100% of Camera 05 frames at ~0.97; 0 of 142 machines on Camera 03**, even at conf 0.01. "
        "Extinguisher false alarms rose to 51–74, and catalog scores fell 3 points (mAP50) and 9–11 points (mAP50-95).",
    ])
    d.table(
        ["Test", "run5", "run6", "run7"],
        [["Camera 05 (run7's training camera), 200 raw frames", "0 / 200", "0 / 200", "**200 / 200 at ~0.97**"],
         ["Camera 03 machines found at ≥ 0.25 (of 142)", "0", "0", "**0**"],
         ["Camera 03 extinguisher false alarms (imgsz 640 / 1280)", "0 / 0", "32 / 2", "51 / 74"],
         ["Catalog box mAP50 / mAP50-95 (imgsz 960)", "0.821 / 0.706", "0.825 / 0.698", "0.786 / 0.597"]],
        align="lccc", widths=[6.8, 2.4, 2.4, 3.2], size=8)
    d.p("**The four root causes:**", keep=True)
    d.numbered([
        "**One viewpoint per class:** 257 frames from one fixed camera are roughly one view of information.",
        "**Different machines:** Camera 03 has a manual edge bander and a hot press; Camera 05 has an inline line and a frame press.",
        "**Position, not appearance:** Camera 03's press sits where Camera 05's edge bander does, and run7 called it an edge bander in 64 of 71 frames.",
        "**Scene texture:** it scored 0.97 even on the 2:1-squeezed raw video it never trained on.",
    ])
    d.bullets(["**The lesson:** count viewpoints, not frames. We need ≥ 5 per class before CCTV training, with at most 1–2× repeats."])
    d.h(2, "5.4 Run 8: The Production Recipe")
    d.bullets([
        "**Base:** run5, never run6 or run7.",
        "**Data:** cleaned and curated labels, 30 high-angle images, 107 hard negatives. Negatives are 11.8% of the 3,298-entry list (11.6% before the 21 Oct curation).",
        "**Recipe:** explicit AdamW (lr0 5e-4, cosine), imgsz 960, batch 6, 50 epochs, patience 20, geometry turned down (`degrees=0`, `perspective=0.0002`).",
        "**`cctv_aug` (Albumentations 2.x):** barrel lens distortion, JPEG 30–75, downscale 0.4–0.75, sensor noise, mild blur, rare greyscale.",
        "**Status:** epoch 1 scored 0.830 / 0.813 on the 450-image val; resumed on 23 Oct. Epoch 1 took 3,525 s, against "
        "111 s per epoch for run5, so about two days for the full run unless the cause is found.",
    ])


def ch6(d):
    d.h(1, "6. Critical Engineering Traps & Bug Post-Mortems")
    d.big_picture("Our most expensive bugs never raised an error. Each one made a training recipe mean something different from what it said.")
    d.h(2, "6.1 The Silent Albumentations Bypass")
    d.bullets([
        "**The problem:** Ultralytics wraps Albumentations in an import guard. If the package is missing, blur, compression and greyscale are skipped **without any error**.",
        "**The impact:** run1–run6 trained with no degradation beyond HSV jitter. The CCTV-robust run was never robust.",
        "**The fix:** we installed 2.0.8 with `--no-deps` (keeping the GUI OpenCV build), and `train.py` now refuses to start without it. Check every training log for the `albumentations:` line.",
    ])
    d.h(2, "6.2 Albumentations 2.x Breaking Names")
    d.table(
        ["Transform", "1.x (silently ignored)", "2.x (what we use)"],
        [["ImageCompression", "`quality_lower`, `quality_upper`", "`quality_range=(30, 75)`"],
         ["Downscale", "`scale_min`, `scale_max`", "`scale_range=(0.4, 0.75)`"],
         ["GaussNoise", "`var_limit=(10, 50)`", "`std_range=(0.012, 0.028)`"]],
        align="lll", widths=[3.4, 5.6, 5.8], size=8)
    d.bullets([
        "**Barrel vs pincushion:** in 2.x a *negative* `distort_limit` gives pincushion. CCTV lenses barrel, so we use (-0.05, 0.25). Polygons warp with the image.",
        "**`optimizer=auto` too:** it ignores `lr0` and picks AdamW at 0.000769 for nine classes. run5 and run6 list lr0 0.01 but never used it. Always set the optimizer explicitly.",
    ])
    d.h(2, "6.3 The “1080P Lite” Squeeze & the Upscaling Trap")
    d.bullets([
        "**The squeeze:** every factory recording is 960×1080, a 16:9 picture stored at half width with no aspect flag, so every tool shows machines half as wide as they are.",
        "**The fix:** de-squeeze to 1920×1080 before labelling or inference (`sample_cctv_video.py --resize 1920x1080`). The benchmark is built that way.",
        "**The upscaling trap:** on the 480×360 workshop sub-stream, run5 scored 0.56 at imgsz 960 and 0.27 at 1280. Upsampling 2.7× creates blurry giants the model never saw.",
        "**Rule:** imgsz 640 for sub-streams, 960 for de-squeezed main streams, or native-size ROI crops.",
    ])
    d.h(2, "6.4 The Small Object Myth")
    d.bullets([
        "**The assumption:** the workshop saw was “small and distant”, so SAHI tiling should help.",
        "**The measurement:** it is 266×208 px on a 480×360 frame, 32% of the image. The gap is viewpoint, codec and clutter, not size.",
        "**The decision:** we dropped SAHI (6–9 forward passes per 4K frame) for **static ROI crops**: fixed cameras, fixed machines, 1–3 regions per camera at native resolution.",
        "**Exception:** SAHI is reconsidered only for far-wall extinguishers under ~40 px on a wide main stream.",
    ])
    d.h(2, "6.5 The Ignored iou Parameter")
    d.bullets([
        "**The problem:** our checkpoints run YOLO26's NMS-free head, so `iou=0.45` did nothing for weeks (`ultralytics/utils/nms.py:66`).",
        "**The fix:** `end2end=False` (or `--nms` in the harness) switches to the one-to-many head with NMS, and every inference summary now prints which head ran.",
    ])
    d.h(2, "6.6 What the Traps Taught Us")
    d.table(
        ["Lesson", "Guard we adopted"],
        [["Catalog val mAP does not predict CCTV performance", "Gate on the factory benchmark and the held-out clip"],
         ["Frames are not viewpoints", "Weight CCTV data by camera × machine pairs; ≤ 1–2× repeats"],
         ["Libraries fail silently", "Fail-fast checks; explicit optimizer; read the training log"],
         ["Labels cap accuracy", "Written extent policy; reviewed SAM outlines"],
         ["Noise-level val picks checkpoints at random (run7)", "Select on catalog val or run fixed epochs"]],
        align="ll", widths=[7, 7.8], size=8)


def ch7(d):
    d.h(1, "7. Deployment, Benchmark & Release Gates")
    d.big_picture("run7 proved that a camera used in training can never again measure generalisation. So we froze our factory footage as a test-only benchmark and wrote down the gates a model must pass.")
    d.h(2, "7.1 The Frozen Factory Benchmark (328 Frames)")
    d.table(
        ["Camera", "Recordings", "Frames", "Labels"],
        [["cam03", "2 (07 and 24 Sep)", "71", "press, edge bander"], ["cam05", "3 (25 and 28 Sep)", "257", "panel saw, press, edge bander"],
         ["**Total**", "5, de-squeezed to 1920×1080", "**328**", "panel saw 257, press 328, edge bander 328"]],
        align="llrl", widths=[2, 4.4, 1.8, 6.6], size=8)
    d.bullets([
        "**Absent classes:** the other six never appear, so any detection of them is a false positive (watch Camera 03's plastic drum).",
        "**Zero leakage:** `train.py` refuses the benchmark; it is never used for checkpoint selection, threshold tuning or negative mining; splits are by camera, never by frame.",
        "**Limits:** 3 of 9 classes and 2 viewpoints, so it measures transfer to these cameras, not general readiness.",
    ])
    d.figure("docs/factory_footage_contact_sheet.jpg", "The two factory cameras behind the benchmark.", 12, d._root)
    d.h(2, "7.2 Multi-Frame Temporal Filtering (Design, Not Yet Built)")
    d.bullets([
        "**Detect low, decide slowly:** decode at 5 fps, detect at conf 0.10, and associate with ByteTrack (high 0.35, low 0.10, new track 0.45, buffer 10 s).",
        "**8-of-15 rule:** confirm a track when at least 8 of the last 15 frames match above the class threshold; drop it at 2 or fewer.",
        "**Class by vote:** a confidence-weighted vote over the track's life stops table-saw / panel-saw flicker.",
        "**Thresholds (provisional):** edge bander 0.20, extinguisher 0.30, others 0.40. Re-derive them on CCTV val.",
        "**Extinguisher not confirmed:** mark it *unverified* for human review. Never raise an alarm.",
    ])
    d.h(2, "7.3 Release Gates")
    d.table(
        ["Gate", "Criterion", "run5 today"],
        [["1. CCTV accuracy", "Per-class mask mAP50 ≥ 0.60 and recall ≥ 0.80 on held-out cameras", ("badge", "FAIL (0.054)", R)],
         ["2. False alarms", "≤ 1 per camera-hour after temporal confirmation", ("badge", "NO DATA YET", Y)],
         ["3. Catalog retention", "Mask mAP50-95 within 2.0 points of run5", ("badge", "BASELINE", G)],
         ["+ Trial clips", "Raw RTSP clips without overlay; must beat run5 on the workshop saw", ("badge", "NOT RECORDED", Y)]],
        align="llc", widths=[3, 8.8, 3], size=8)
    d.p("The plan lists the trial-clip check as a fourth gate. run8 has not been scored against any gate yet.", size=9)


def ch8(d):
    d.h(1, "8. Immediate Handoff & Action Items")
    d.big_picture("Let run8 finish, measure it honestly, then spend effort on viewpoints, not epochs. The model is not the bottleneck; real elevated views are.")
    d.bullets([
        "**☐ Monitor run8:** resumed 23 Oct with 50 epochs planned. Do not run other GPU jobs on the 8 GB card meanwhile. "
        "Find out why epoch 1 took 3,525 s (check GPU utilisation and dataloader CPU load).",
        "**☐ Re-baseline, then evaluate:** run5 on the current 450-image val, then run8 `best.pt` / `last.pt` on that val, "
        "the frozen benchmark and the workshop clip (imgsz 640 and 960).",
        "**☐ Human review of `datasetchange`:** check the before/after sheets in `label_fixes/review_user_fixes/` and decide on the 6 self-intersecting polygons.",
        "**☐ Settle class boundaries:** sliding-table saw = `panel_saw`; veneer press → `hydraulic_press`.",
        "**☐ Record raw RTSP:** ≥ 24 h per camera (day, night, IR) with the overlay off, plus pseudo-CCTV phone video from 3–4 m up. Target ≥ 5 viewpoints per class.",
        "**☐ Build CCTV negatives:** ≥ 300 frames from cameras outside the benchmark, for gate 2.",
        "**☐ Pipeline hardening:** cross-class and byte-level de-dup in stage 3; upload every SAM instance, not just the first.",
    ])
    d.h(2, "Document Control")
    d.table(
        ["Version", "Date", "Change"],
        [["2.4", "2026-10-23", "First consolidated dossier"], ["2.5", "2026-10-23", "Narrative voice; scraping pipeline expanded"],
         ["2.6", "2026-10-23", "Condensed to a bullet-driven brief; table of contents and appendix removed"]],
        align="cll", widths=[1.6, 2.4, 10.8], size=8)
    d.table(
        ["Role", "Name", "Signature", "Date"],
        [["Technical owner (ML)", "", "", ""], ["Data and labelling reviewer", "", "", ""], ["Operations / safety stakeholder", "", "", ""]],
        align="llll", widths=[5, 4, 3.5, 2.3], size=8.5)
    d.p("**Sources:** run logs in `runs/`, `docs/runsummary.md`, `docs/run7_analysis_report.md`, `docs/label_policy.md`, "
        "`docs/cctv_benchmark.md`, `docs/cctv_scrape_postmortem.md`, `pipeline/`.", size=8.5, color="64748B")
