"""Front matter and chapters 1-3 of the Krish dossier (v2.5: narrative voice, condensed, pipeline chapter expanded)."""

G, R, A, B, Y = "green", "red", "amber", "blue", "grey"


def front(d):
    d.title_block(
        "Industrial Machine Instance Segmentation & CCTV Domain Adaptation Dossier",
        "Architecture, Data Pipelines, Empirical Failure Modes, and Production Readiness for 9-Class YOLO26-seg",
        [
            "**Project:** Krish: 9-Class Industrial Woodworking Machinery Instance Segmentation & CCTV Domain Adaptation",
            "**Project Version 2.5  |  Status: Active Engineering  |  Target: Factory Safety & Inventory**",
            "**Document date:** 23 October 2026  |  **Workspace:** `Krish\\` project root (Windows 11, RTX 3070 Ti)",
            "**Voice:** written as the Krish engineering team's account of the work. Compiled by Claude (Claude Code) "
            "from the project's own artifacts; sign-off in Section 10.4.",
        ],
    )
    d.callout("info", "how to read this", [
        "This is the story of what we set out to build, where our assumptions failed, why some early metrics fooled us, "
        "and what we changed. The tables carry the numbers; the text explains why they moved.",
        "Every figure comes from a project artifact: run logs, the reports in `docs/`, `scripts/train.py` and the "
        "pipeline source in `pipeline/pipeline/`. Where a brief or an earlier design document disagrees with the code or the "
        "logs, we go with the code and logs, and Appendix A lists every such case.",
    ])
    d.toc_placeholder()


# =====================================================================================================
def ch1(d):
    d.h(1, "1. Executive Summary & System Overview")
    d.big_picture(
        "We are building a model that watches fixed factory CCTV and outlines nine things: seven woodworking machines, an "
        "industrial press, and fire extinguishers. The outlines feed three uses: knowing which machines are where "
        "(inventory), drawing hazard zones around the dangerous ones, and confirming that extinguishers are on the wall.",
        "The hard part has never been the catalog photos. Our models have scored above 80% mAP50 on those since September. "
        "The hard part is that a ceiling camera in a dusty workshop sees these machines from an angle, at a resolution and "
        "through a codec that no web photo shows. Most of this dossier is about that gap.",
    )

    d.h(2, "1.1 Purpose, Operational Scope & Safety Objectives")
    d.bullets([
        "**Inventory and presence:** confirm which machines each camera sees and that they stay put. All nine classes are "
        "stationary, so the deployment design confirms presence per location; it does not track motion.",
        "**Safety perimeter zoning:** the machine mask, dilated by a margin, becomes the hazard zone around saws, planers, "
        "jointers and presses. Detecting people is a separate model (Section 10.3).",
        "**Fire-safety monitoring:** find extinguishers and confirm they are still at their wall positions.",
    ])
    d.callout("warn", "what this system must never be", [
        "- It is an assistive signal, **not a safety interlock**. It never gates machine power.",
        "- A missing extinguisher detection must **never raise a compliance alarm**. Our best validation run still missed "
        "46% of extinguishers, and 61% in crowded scenes.",
    ])

    d.h(2, "1.2 The 9-Class Taxonomy & Visual Differentiation Matrix")
    d.p("Class ids follow `datasets/9class/data_9class.yaml`. Sizes are typical industry ranges for orientation, not "
        "measurements from this project. The last column records confusions we actually measured.")
    d.table(
        ["ID", "Class", "Typical size", "How we tell it apart", "Where it gets confused (evidence)"],
        [
            ["0", "`panel_saw`", "Carriage 2.6–3.8 m", "Long sliding carriage with an outrigger crosscut table", "`table_saw`: run5 calls the workshop sliding-table saw `table_saw` (up to 0.70)"],
            ["1", "`hydraulic_press`", "2–4 m wide frame", "Heavy frame, platens, cylinders", "Veneer / hot presses; Camera 03's hot press vs Camera 05's frame press"],
            ["2", "`table_saw`", "0.6×0.8 m to 1.2×2 m", "Blade through a flat table, rip fence, cabinet base", "`panel_saw`; small benchtop saws"],
            ["3", "`planer`", "0.6–1.2 m body", "Enclosed box; stock fed underneath; depth handwheel", "`jointer` (6% of planers, run6); drum sanders, moulders"],
            ["4", "`jointer`", "1.5–2.5 m beds", "Open low beds with the cutter on top and a side fence", "`planer`"],
            ["5", "`band_saw`", "1.5–2.2 m tall", "Two wheel housings and a continuous blade", "Drill presses; mitre saws"],
            ["6", "`miter_saw`", "0.5–0.9 m", "Pivoting arm over a turntable", "`band_saw`; 28% missed (run6)"],
            ["7", "`edge_banding_machine`", "1.5 m (manual) to 8 m+ (line)", "Enclosed body, tape coil disc, attached conveyors", "Conveyors, extrusion lines; historically frame-filling labels"],
            ["8", "`fire_extinguisher`", "0.4–0.9 m", "Red cylinder with handle, hose and nozzle", "Drums, gas cylinders, tubes, standing workers (run7)"],
        ],
        align="cllll", widths=[0.45, 2.75, 2.0, 3.3, 3.6], caption="Classes, cues and measured confusions", size=8)

    d.h(2, "1.3 Chronological Milestones & Current State (October 2026)")
    d.table(
        ["When", "What happened", "What we learned"],
        [
            ["Sep 15–18", "run1–run4: 23-class and 38-class web-data models", "Breadth costs accuracy; the 38-class model became run5's parent"],
            ["Sep 25", "run5: 9-class fine-tune", "Champion; still the baseline today"],
            ["Oct 5–8", "run6 “CCTV-robust” and its audit", "Higher val mAP, worse on real video"],
            ["Oct 10–15", "CCTV plan, workspace merge, CCTV web scrape and post-mortem", "The web has almost no CCTV views of these machines"],
            ["Oct 19", "Factory footage (2 cameras) and run7", "One camera per class teaches the scene, not the machine"],
            ["Oct 21", "Label overhaul, frozen benchmark, high-angle images, manual curation", "Labels were capping two classes"],
            ["Oct 22–23", "run8 started, then resumed", "In progress; not yet evaluated"],
        ],
        align="lll", widths=[2.2, 6.3, 6.3], caption="Milestones", size=8.5)
    d.p("**Where we stand (23 October 2026):** run5 is the fallback and the parent of every future fine-tune. run6 and run7 "
        "must not be deployed. run8 is training (1 of 50 epochs logged). The dataset is 2,200 / 450 / 473 labelled "
        "train / val / test images, and the train list holds 3,298 entries with 11.8% negatives. The 328-frame factory benchmark is frozen for testing only.")

    d.h(2, "1.4 Executive Scorecard: Run 1 through Run 8 Comparison")
    d.table(
        ["Run", "Classes / base / imgsz", "Val box / mask mAP50", "On real footage", "Verdict"],
        [
            ["run1", "23 / COCO / 960", "0.777 / 0.755", "not tested", ("badge", "SUPERSEDED", Y)],
            ["run2", "23 / run1 / 960", "0.787 / 0.765", "not tested", ("badge", "SUPERSEDED", Y)],
            ["run3", "38 / COCO / 960", "0.608 / 0.544", "not tested", ("badge", "RUN5'S PARENT", Y)],
            ["run4", "38 / run3 / 960", "0.611 / 0.552", "not tested", ("badge", "SUPERSEDED", Y)],
            ["run5", "9 / run3 / 960", "0.825 / 0.802", "Workshop saw found (0.56–0.63); 0 extinguisher false alarms", ("badge", "CHAMPION", G)],
            ["run6", "9 / run5 / 1280", "0.810 / 0.795", "Workshop saw lost (≤ 0.18); 0 detections at 0.40 in 1,398 frames", ("badge", "DO NOT DEPLOY", R)],
            ["run7", "9 / run5 / 1280", "0.786 / 0.767 (catalog, 960)", "100% on its training camera; 0 of 142 on an unseen one", ("badge", "ARCHIVED", A)],
            ["run8", "9 / run5 / 960", "0.830 / 0.813 (epoch 1)", "not yet tested", ("badge", "TRAINING", B)],
        ],
        align="llrlc", widths=[1.2, 3, 3.4, 5.4, 2.2], caption="Every run at a glance (each on its own validation set)", size=8)
    d.p("Validation sets differ between eras, and they shrank again on 2026-10-21, so read the mAP column as a trend within "
        "an era. On the current cleaned validation set, run5 scores box 0.789 / mask 0.741 (measured on 461 images before the "
        "final curation). It must be re-measured on today's 450-image set before anyone compares it with run8. On the factory "
        "benchmark, run5 scores mask mAP50 0.054 (imgsz 640) and 0.010 (imgsz 960). That gap is the real problem.")


# =====================================================================================================
def ch2(d):
    d.h(1, "2. Infrastructure, Tooling & Technical Environment")
    d.big_picture(
        "Everything runs on one Windows workstation with an 8 GB RTX 3070 Ti. That shaped several decisions: batch sizes, "
        "the zero-copy dataset layout, and an NMS-free model for real-time inference. This chapter keeps only what you need "
        "to reproduce a run or hand the project over.",
    )
    d.h(2, "2.1 Compute, GPU & Software Stack Specifications")
    d.table(
        ["Component", "What we use", "Why it matters"],
        [
            ["GPU", "RTX 3070 Ti, 8 GB", "Batch 6 at imgsz 960, batch 4 at 1280"],
            ["Environment", "Python 3.10 venv `env\\yolo-env`; torch 2.13.0+cu130; Ultralytics 8.4.123", "Pinned; the venv launchers were rebuilt after the 2026-10-10 move"],
            ["Augmentation", "Albumentations 2.0.8 (installed `--no-deps`)", "Keeps the GUI OpenCV build. Without it, Ultralytics silently skips degradation (Section 7.1)"],
            ["Models", "`yolo26s-seg` lineage; SAM 2.1 Large for annotation", "SAM L is the annotation default since 2026-10-21"],
            ["Speed", "12–16 ms per frame inference at 1280; 17–22 ms end-to-end", "45–60 FPS, so one GPU can serve 8–10 cameras at 5 fps"],
        ],
        align="lll", widths=[2.4, 6, 6.4], caption="Stack", size=8.5)
    d.callout("note", "run8 is training about 30 times slower than run5", [
        "Epoch 1 took 3,525 s, against 111 s per epoch for run5 at the same imgsz, so 50 epochs could take about two days. "
        "Our suspects are CPU-bound Albumentations with `workers=4` and the RAM cache fill. We have not confirmed either; "
        "the first check is GPU utilisation during epoch 2.",
    ])

    d.h(2, "2.2 Workspace Architecture & NTFS Junction Symlink Strategy")
    d.p("After the 2026-10-10 merge (68 logged moves, nothing deleted), the project has one root. The decision that matters is "
        "the dataset layout: each `datasets\\9class\\<split>_9class\\images` is an NTFS junction to the 38-class image "
        "folder, so the 8,119 train images are stored once. The `.txt` lists, not the folders, define the dataset. "
        "Ultralytics reads only the listed paths and maps each one to `...\\labels\\` by string substitution, so unlisted "
        "images are never seen. The build rewrites the 9-class labels from scratch every time.")
    d.callout("warn", "the one rule that can destroy data", [
        "Deleting “unused” files inside `*_9class\\images` deletes the originals in `source_38class`. Never run "
        "`Remove-Item -Recurse` on a junction either; remove it with `cmd /c rmdir`. Point training at the yaml or `.txt` "
        "lists, never at the images folder.",
    ])
    d.code("""
foreach ($s in 'train','valid','test') {
  cmd /c rmdir "datasets\\9class\\${s}_9class\\images"
  New-Item -ItemType Junction -Path "datasets\\9class\\${s}_9class\\images" -Target "$PWD\\datasets\\source_38class\\$s\\images"
}
""", caption="Re-link the junctions after moving the project (CHANGELOG_REORGANIZATION.md)")
    d.p("A full rollback of the merge means replaying `reorg_move_log.csv` (`src,dst`) in reverse, then re-linking as above.")

    d.h(2, "2.3 The YOLO26 Architecture: End-to-End Head vs Auxiliary NMS Head")
    d.p("YOLO26 trains two heads. The default **one-to-one head** assigns one prediction per object and skips NMS, which "
        "keeps latency flat and the output deterministic. The **one-to-many head** behaves like classic YOLO and needs NMS. "
        "The catch: on our checkpoints `iou=` does nothing unless you pass `end2end=False` (or `--nms` in "
        "`test_inference.py`). We set `iou=0.45` for weeks before we noticed.")
    d.p("**Still to test:** whether the one-to-many head recovers crowded extinguishers. Recall is 86% for a lone cylinder "
        "but 39% when three or more share an image. Run the crowded subset with both heads at the same confidence and compare recall and duplicates.")

    d.h(2, "2.4 Tooling Suite: Training, Evaluation Harness and Annotation Utilities")
    d.table(
        ["Tool", "What it is for"],
        [
            ["`scripts/train.py`", "The run8 recipe. It refuses to start without Albumentations, refuses the frozen benchmark, and never overwrites a run"],
            ["`scripts/test_inference.py`", "Image, folder or video inference with per-class thresholds (`--class-conf`), `--nms`, and CSV/JSON/Markdown logs"],
            ["`tools/build_clean_9class_dataset.py`", "Builds the 9-class labels and lists: merges, quarantines, manual fixes, negatives, junctions"],
            ["`tools/audit_dataset_distribution.py`", "Blocking dataset checks; must report PASS before training"],
            ["`tools/fix_user_labels.py`, `fix_edge_bander_labels.py`, `annotate_highangle.py`", "SAM 2.1 box-prompt re-annotation workflows with review sheets"],
            ["`tools/sample_cctv_video.py`", "De-duplicated frame sampling from CCTV video; `--resize 1920x1080` un-squeezes 1080P Lite"],
            ["`tools/build_dataset_viewer.py`", "A browsable gallery of exactly what training sees (`docs/viewer/index.html`)"],
        ],
        align="ll", widths=[5.6, 9.2], caption="The tools you will actually use", size=8.5)
    d.code(r"""
env\yolo-env\Scripts\python.exe scripts\tools\build_clean_9class_dataset.py
env\yolo-env\Scripts\python.exe scripts\tools\audit_dataset_distribution.py
env\yolo-env\Scripts\python.exe scripts\train.py --name run9_<description>
env\yolo-env\Scripts\python.exe scripts\test_inference.py --weights <best.pt> --source <video> ^
    --imgsz 960 --conf 0.40 --class-conf "edge_banding_machine=0.20,fire_extinguisher=0.30"
""", caption="Rebuild, audit, train, test")


# =====================================================================================================
def ch3(d):
    d.h(1, "3. The I4I Data Harvesting & Curation Architecture")
    d.big_picture(
        "In September we had no factory footage and almost no labelled images of edge banders, jointers or panel saws. Public "
        "datasets barely cover them. So we built an engine, I4I (now `pipeline/`), that harvests web images per class, "
        "filters them with CLIP, removes near-duplicates and sends the survivors through an automatic Gemini + SAM 2 "
        "labelling loop into Roboflow. That engine produced the data behind every model from run1 onwards.",
        "It worked for what it was designed to do: clean, well-lit, recognisable photos of each machine. It could not do the "
        "one thing we later asked of it, which was to find CCTV views. This chapter explains how each stage works, what it "
        "actually does in code (some of it differs from the original design document), and where it hit its limits.",
    )

    d.h(2, "3.1 The Core Philosophy: Why Automated Web Harvesting")
    d.bullets([
        "**Bootstrap rare classes cheaply.** A 38-class detector needs hundreds of images per class. Hand-collecting them for "
        "classes like mortisers, veneer presses or edge banders would have taken weeks. A scrape-and-filter loop produces a class in hours.",
        "**Source diversity on purpose.** The design (`pipeline/machine-type-classifier-prd.md`) expands each class across "
        "query templates for angle, distance, workers and occlusion, rather than hoping filtering would create diversity afterwards.",
        "**Let a person stay in the loop only where it pays.** Humans pick a handful of reference images per confusable class "
        "and review contact sheets. CLIP, Gemini and SAM do the bulk work.",
        "**We knew about the domain gap from day one.** The design document warns that scraped photos are “clean, well-lit, "
        "catalog-style” while deployment is elevated CCTV, and that the gap “needs to be measured, not assumed.” It "
        "planned a small real-world canary set (5–10 images per class, `dataset/real_validation/`). That folder was never "
        "created, and that is one reason the gap surprised us later.",
    ])

    d.h(2, "3.2 Architecture at a Glance")
    d.table(
        ["Stage", "Script", "Input → output", "Gate"],
        [
            ["1", "`utils/manifest_gen.py`, `stage1_ingest_refs.py`", "Hand-picked photos → `reference_images/<class>/`", "≥ 3 references per confusable class (warning)"],
            ["2", "`stage2_scrape.py`, `batch_topup_300.py`", "Class + templates → raw JPEGs in `dataset/train/<class>/`", "Keyword blocklist, decode check, min side 200 px"],
            ["2c", "`stage2c_accept_gate.py`", "Tagged batch → accepted / `_gate_rejected/`", "Cross-class and catalog duplicates, CLIP scene and class rank"],
            ["3", "`stage3_clean.py`", "Raw → clean", "Corrupt files, min side 150 px, pHash ≤ 8"],
            ["4 / 4b", "`stage4_clip_filter.py`, `stage4b_calibrate.py`", "Clean → on-class", "CLIP text ≥ 0.18; reference similarity ≥ 0.30"],
            ["5", "`stage5_diversity_kselect.py`", "On-class → diverse set", "Drop if cosine > 0.95 to a kept image; floors 80 / 150"],
            ["6", "`upload_polygon_annotations.py` + Roboflow workflow", "Images → polygons in Roboflow", "Gemini class check; SAM polygon"],
        ],
        align="clll", widths=[1.2, 4.6, 4.8, 4.2], caption="The pipeline, stage by stage", size=8)
    d.p("Two drivers tie the stages together. `run.py` walks one class through stages 2–5 interactively, asking for "
        "confirmation at each boundary. `batch_topup_300.py` is the unattended version: it scrapes with 8 download threads and "
        "a 4 s timeout, runs stages 3–5, and loops until each class reaches 300 clean images. All parameters live in "
        "`pipeline/pipeline/config.py`.")

    d.h(2, "3.3 Stage 1: Reference Manifest & Exemplar Ingestion")
    d.p("CLIP's text understanding is too coarse to separate a planer from a jointer or a drum sander from a belt sander. So "
        "for every confusable class, a person picks a few unambiguous photos and they become the visual definition of the class.")
    d.numbered([
        "`python -m pipeline.utils.manifest_gen --src <folder>` lists a flat folder of candidate photos into `reference_manifest.csv`.",
        "A person fills in the `class` column with exact slugs from `config.py`.",
        "`python -m pipeline.stage1_ingest_refs` copies them into `reference_images/<class>/` and prints a coverage report "
        "(`--dry-run` available). Confusable classes with fewer than 3 references are flagged LOW.",
    ])
    d.table(
        ["Target class", "References", "Target class", "References"],
        [["planer", "30", "jointer", "11"], ["table_saw", "21", "hydraulic_press", "9"], ["panel_saw", "14", "band_saw", "8"],
         ["miter_saw", "14", "edge_banding_machine", "8"], ["fire_extinguisher", "none (not a confusable class)", "All 36 folders", "364 images"]],
        align="lrlr", widths=[3.6, 3.8, 3.6, 3.8], caption="Reference images held today (pipeline/reference_images/)", size=8.5)
    d.p("Planer has the most references for a reason: its CLIP self-similarity is the lowest of any class (as low as 0.17), "
        "so it needed the widest set of exemplars to avoid rejecting good photos.")

    d.h(2, "3.4 Stage 2: Multi-Template Web Scraping")
    d.h(3, "Search engine")
    d.p("The design document planned Bing and Google Images. In practice we started with an icrawler scraper over Bing's HTML "
        "and replaced it with **DuckDuckGo image search through the `ddgs` package**, which returns direct image URLs (its "
        "image results come from Bing). Every scraper in the repository now uses `ddgs`; there is no Google scraper.")
    d.h(3, "Query templates")
    d.table(
        ["Template set", "Size", "Examples", "Used for"],
        [["`default`", "15", "“{machine}”, “industrial {machine} equipment”, “{machine} in workshop factory”, “professional {machine} machine”", "Original catalog harvest"],
         ["Per-class lists in `batch_topup_300.py`", "~10 per class", "“sliding panel saw machine”, “vertical panel saw machine”, “automatic edgebander machine close up detail”, “woodworking edgebander machine conveyor line”", "Top-ups to 300 per class"],
         ["`cctv`", "10", "“{machine} workshop CCTV view”, “{machine} overhead camera workshop setup”, “security camera {machine} factory”", "The October CCTV attempt (Section 3.10)"],
         ["`highangle`", "8", "“{machine} furniture factory production line”, “{machine} woodworking factory mezzanine view”, “{machine} factory tour”", "Place-based replacement for `cctv`"]],
        align="lrll", widths=[3.4, 1.6, 6.6, 3.2], caption="Query template sets", size=8)
    d.h(3, "Rate limiting, integrity and blocklists")
    d.bullets([
        "**Quota:** the per-class cap (400 by default) is split evenly across the templates. Each query asks for "
        "min(max(3 × quota, 60), 100) results, waits 1.5 s, and retries rate limits with exponential back-off "
        "(10 s, doubling, up to 5 attempts).",
        "**Download integrity:** HTTP 200 only; the bytes must pass Pillow's `verify()` and re-open cleanly; both sides must "
        "be at least 200 px; images are converted to RGB and saved as JPEG at quality 90. If the full-size URL fails, the thumbnail is tried.",
        "**Blocklist on title and URL:** diagrams, schematics, clipart, logos, sketches, blueprints, renderings, 3D models "
        "and, by default, “cctv” and “surveillance”. CCTV mode swaps those two for camera-product words "
        "(“dome camera”, “bullet camera”, “ip camera”, “nvr”, “dvr”, “camera kit”). High-angle "
        "mode adds “camera”, “infographic”, “brochure” and “spare parts”.",
        "**Naming:** `img_<class>[_<tag>]_<index>_<md5(url)[:10]>.jpg`. The tag keeps batches separate, so a CCTV batch never mixes with catalog images.",
    ])
    d.callout("note", "two scraper bugs we fixed on 2026-10-21", [
        "- **Duplicate downloads.** The old existence check compared filenames that included a running index, so it never "
        "matched, and the same URL was saved by every query and class that returned it (537 repeats in the CCTV batch). "
        "The scraper now keeps a set of seen URLs across templates and classes.",
        "- **Template quota ignored.** The first query could fill 35 of 40 slots on its own, so the other templates added no "
        "diversity. Each template now stops at its quota.",
    ])

    d.h(2, "3.5 Stage 2c: The Batch Acceptance Gate")
    d.p("Added in October for the high-angle batch, the gate checks a tagged batch before anyone spends time labelling it. "
        "Each image gets the first reason it fails, and rejects are **moved, never deleted**, to `<class>/_gate_rejected/<reason>/`.")
    d.numbered([
        "**Cross-class duplicate:** a near-copy (dHash distance ≤ 6) exists under another class, so every copy is rejected, because we cannot know which label is right.",
        "**In-batch duplicate:** a near-copy of an earlier image in the same class (the first is kept).",
        "**Catalog duplicate:** a near-copy of an image already in `datasets/source_38class` or the intake folders.",
        "**Scene test (CLIP):** the best “wanted” prompt (high-angle factory floor, overhead production line) must beat "
        "every “unwanted” one (white-background product shot, advert or infographic, CCTV camera product, close-up of parts, "
        "diagram, drone aerial, CAD render, collage).",
        "**Class test (CLIP):** the expected class must rank in the top 3 of the 9 class prompts.",
    ])
    d.p("On the high-angle batch the gate saw 113 images and passed 99; the other 14 failed the scene test. Our review then "
        "accepted 30 of the 99 (52 instances), so the automatic gate caught the easy failures and judgement handled the rest (Section 4.5).")

    d.h(2, "3.6 Stage 3: Integrity, De-duplication & Resolution Hygiene")
    d.bullets([
        "**Corrupt files:** every image must survive `Image.verify()` and convert to RGB; failures are deleted.",
        "**Resolution:** either side under 150 px is deleted. In practice the download step already enforces 200 px, so this stage catches files that arrived another way.",
        "**Perceptual de-duplication:** a pHash per image; anything within Hamming distance 8 of an already-kept image in the same class is deleted.",
    ])
    d.callout("info", "what stage 3 does not do (yet)", [
        "There is no byte-level MD5 check (the MD5 in the filename is of the URL), no cross-class duplicate check, no "
        "aspect-ratio filter, and the size floor is 150–200 px, not 400×300. The post-mortem in Section 3.10 found 27% "
        "byte-identical copies and 347 cross-class copies in one batch, so the cross-class check from stage 2c belongs in "
        "stage 3 too. Raising the floor to about 400×300 is a sensible next step for training data. We list it as a "
        "recommendation because the code does not do it today.",
    ])

    d.h(2, "3.7 Stage 4 & 4b: Hybrid OpenCLIP Semantic Filtering & Calibration")
    d.p("All CLIP stages use OpenCLIP ViT-B-32 with the `laion2b_s34b_b79k` weights, batch 64, with embeddings L2-normalised so that dot products are cosine similarities.")
    d.h(3, "Coarse tier: is this a photo of the machine at all?")
    d.p("Each image is scored against “a photo of a {display name}” and against 20 non-photo prompts (diagram, CAD "
        "drawing, schematic, specification sheet, blueprint, manual illustration, rendering, vector graphic, sketch, poster, "
        "line art, safety infographic, clipart and similar). It passes if the class score is **≥ 0.18** and it is no more than "
        "0.05 below the best non-photo score. This tier aims for recall: it removes wrong objects and drawings, not look-alike machines.")
    d.h(3, "Fine tier: does it look like our reference machines?")
    d.p("For confusable classes, each survivor's embedding is compared with that class's reference embeddings; the **mean cosine "
        "similarity must be ≥ 0.30**. For the riskiest pairs, the code also asks whether the image is closer to a "
        "rival's references than to its own, and drops it if so:")
    d.table(
        ["Class", "Rival reference set", "Rule"],
        [["drum_sander", "sanding_machines", "Keep only if closer to drum-sander references"],
         ["edge_banding_machine", "cnc_router", "Keep only if closer to edge-bander references"],
         ["overhead_hoist / crane", "crane / overhead_hoist", "Each must beat the other"],
         ["ppe_station", "storage_racking, control_panel", "Must beat both"]],
        align="lll", widths=[3.8, 4.4, 6.6], caption="Dual-set cross-contamination checks in stage 4", size=8.5)
    d.p("There is no dual-set rule for band saw vs drill press, or edge bander vs panel saw (the design document lists the "
        "latter). Those confusions surfaced later in model results and were handled with negatives (Section 4.4).")
    d.h(3, "Calibration (stage 4b)")
    d.p("Raw CLIP scores are much lower than intuition suggests, so we calibrate rather than guess. `stage4b_calibrate.py` "
        "embeds a hand-sorted good and bad sample for one class, prints both distributions and saves a histogram to "
        "`logs/calibration_<class>.png`. The suggested cut-offs are:")
    d.bullets([
        "**Coarse:** the midpoint between the bad set's maximum and the good set's 1st percentile. If they overlap, the script "
        "says the coarse tier cannot separate them, recommends a loose floor (about 0.20), and leaves the work to the fine tier.",
        "**Fine:** the good set's 1st percentile.",
        "**This is a recall-first rule, not a precision-first one:** it keeps 99% of known-good images and accepts some "
        "look-alikes, which later stages and human review must catch. Calibration is why the fine threshold dropped from "
        "0.55 to 0.30: planer references are only 0.17 similar to each other, so 0.55 rejected correct planers.",
    ])

    d.h(2, "3.8 Stage 5: Embedding Diversity & K-Selection")
    d.p("Stock photos get re-uploaded, cropped and watermarked across dozens of sites, and pHash misses many of those copies. "
        "Stage 5 works in CLIP space instead:")
    d.numbered([
        "Rank every image by its CLIP score for “a photo of a {class}”, best first.",
        "Walk down the list and keep an image only if its cosine similarity to every already-kept image is **≤ 0.95**. "
        "This is a greedy threshold filter. It is not K-Means and not farthest-point sampling.",
        "Check the floor: 80 images for standard classes and 150 for confusable ones. A class below its floor is **flagged, "
        "not padded**, in `logs/stage5_kselect_log.json`.",
    ])
    d.p("**Its limit:** the 0.95 threshold removes near-identical images, but it does nothing about *genre*. If 80% of the "
        "web shows a machine on a white background, 80% of what survives will too. The October post-mortem measured exactly "
        "that: the scraped images were the same kind of photo as the existing dataset (mean nearest-neighbour similarity "
        "0.78, against 0.835 inside the dataset). Balancing by viewpoint would need clustering plus per-cluster quotas, which the stage does not have.")

    d.h(2, "3.9 Stage 6: Auto-Annotation with Gemini Vision & SAM 2")
    d.p("The last stage turns a folder of verified photos into segmentation labels without a person drawing polygons. "
        "`upload_polygon_annotations.py` sends each image, with its expected class, to a Roboflow serverless workflow "
        "(`gemini-machine-instance-auto-label-2`):")
    d.numbered([
        "**Gemini Vision verifies the class.** If Gemini's answer conflicts with the expected class, the workflow flags it for "
        "manual review, and the image is skipped. A small synonym table accepts known variants (drill press = drilling; sander or belt sander = sanding machines).",
        "**SAM traces the outline** of the verified object and returns polygon points.",
        "**The script uploads a COCO polygon** to the Roboflow project, using 4 worker threads and a deterministic per-class "
        "70 / 15 / 15 train / val / test split (seed 42). Each class gets a JSON log of OK, skipped and failed images.",
    ])
    d.p("Labels from this loop, together with later uploads for the added classes (`upload_12_new_classes_roboflow.py`), "
        "fed the Roboflow project whose 38-class export is `datasets/source_38class` (8,119 / 1,718 / 1,747 images). The "
        "9-class build reads that export today.")
    d.callout("warn", "three limits we only understood months later", [
        "- **One polygon per image.** The script uploads only the first prediction. In any image it labelled with two "
        "saws or a row of extinguishers, every other instance silently became background. Not every label in the export "
        "came through this script (some images carry several extinguisher instances), but the pattern matches the "
        "missing-instance fixes in Chapter 4, and we think it contributed.",
        "- **SAM does not know the label policy.** On white product shots SAM often selects the background, as we saw "
        "again when re-annotating in October. Unreviewed outlines of that kind are a likely source of the over-large "
        "edge-bander labels that the later merge step turned into frame-filling slabs.",
        "- **Split by image, not by source.** The random split runs after de-duplication within each class only, so a "
        "near-copy can land in both train and val. We have not measured how often it happens.",
    ])

    d.h(2, "3.10 The CCTV Web-Scrape Reality Check")
    d.p("After run6 failed on real video, we pointed the engine at the problem: the `cctv` template set, 300 images per "
        "class, 2,677 images in total, scraped from 10 to 14 October. The review on 15 October (`docs/cctv_scrape_postmortem.md`) was blunt.")
    d.table(
        ["What we measured", "Result"],
        [["Byte-identical copies within the batch", "723 of 2,677 (27%)"],
         ["… of which the same URL saved twice (the scraper bug)", "537"],
         ["Identical files saved under two different classes", "347 files in 116 groups"],
         ["Near-copies of images already in the dataset", "311 (12%)"],
         ["Camera hardware, NVR adverts and security graphics", "~279 (~10%)"],
         ["Catalog or eye-level photos", "~1,700 (~63%)"],
         ["Images that actually look like CCTV of a target machine", "**~5, and none clearly shows one of the nine machines**"]],
        align="lr", widths=[10, 4.8], caption="CCTV scrape audit", size=8.5)
    d.figure("pipeline/logs/cctv_scrape_analysis/sheet_cctv_candidates.jpg",
             "The 49 images CLIP rated closest to “CCTV footage”: repeated stock photos of security cameras, extinguisher-factory adverts, and only a handful of real overhead frames.", 15.5, d._root)
    d.h(3, "Why it failed")
    d.bullets([
        "**The images do not exist on the open web.** Real CCTV of a panel saw sits on private NVRs or inside videos. Image "
        "search ranks by page text, and no page says “CCTV still of a panel saw”.",
        "**Our own query words attracted the wrong pages.** “Surveillance camera” found pages about cameras. "
        "“High angle” and “from above” are photographers' words that captions rarely use.",
        "**One page of results.** DDGS returned about 35 results per query, so popular stock photos dominated.",
        "**Almost nothing was rejected.** CCTV mode removed “cctv” from the blocklist, product titles like “Dahua WiFi "
        "4MP” matched no blocked phrase, and stage 4 had not run yet.",
        "**Viewpoint is the real gap, and text cannot fix it.** Even the good results were front-on photos. A ceiling camera "
        "sees the table top and carriage from above, with the front face foreshortened; those pixels are not in a front-on "
        "photo, and no filter or augmentation can create them.",
    ])
    d.h(3, "What we changed")
    d.bullets([
        "Fixed the URL de-duplication and the template quota (Section 3.4).",
        "Replaced camera words with **place words** (the `highangle` set) and added the stage 2c gate. The result was 30 "
        "usable high-angle images out of 113: helpful, but a supplement, not a solution.",
        "Accepted the lesson: **only footage from our own cameras, or pseudo-CCTV video shot from 3–4 m up, matches the "
        "deployment domain**. The web engine stays the tool for catalog breadth and new classes.",
    ])

    d.h(2, "3.11 Lessons From Building the Engine")
    d.table(
        ["Lesson", "What we would do differently"],
        [["Filtering removes junk; it cannot add a missing viewpoint", "Measure the domain gap with a real canary set before training, as the design asked"],
         ["Duplicates hide everywhere (URLs, mirrors, crops, classes)", "De-duplicate by URL, bytes and perceptual hash, across all classes, before saving"],
         ["Auto-labels inherit the labeller's blind spots", "Upload every instance, not just the first; review crowded images by hand"],
         ["Calibrated thresholds beat intuition", "Keep the 4b histograms with the dataset so thresholds can be audited"],
         ["Diversity by threshold is not diversity by viewpoint", "Cluster embeddings and set per-cluster quotas"]],
        align="ll", widths=[7, 7.8], caption="Engine lessons", size=8.5)
