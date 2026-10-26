"""Chapters 4-10 and Appendix A of the Krish dossier (v2.5)."""

G, R, A, B, Y = "green", "red", "amber", "blue", "grey"


def ch4(d):
    d.h(1, "4. Label Engineering, Spatial Policies & Dataset Sanitization")
    d.big_picture(
        "When run6's audit came back, the two weakest classes were not short of data. They were badly labelled. 45% of "
        "validation edge-bander labels covered the whole image, so the model learned “edge bander = the whole photo” "
        "and scored 0 of 15 on small ones. Touching extinguishers had been merged into blobs. On 21 October we rebuilt the "
        "labels around a written policy and fixed what we could with SAM 2.1. Everything below is recorded in "
        "`datasets/9class/label_fixes/label_decisions.yaml` and applied by `build_clean_9class_dataset.py`.",
    )
    d.callout("warn", "who reviewed this", [
        "Every SAM proposal, negative drop and re-annotation was checked visually on overlay sheets **by Claude, not by a "
        "human annotator**. The sheets are in `datasets/9class/label_fixes/` and deserve a human spot check before a release.",
    ])

    d.h(2, "4.1 The 2026-10-21 Label Overhaul")
    d.table(
        ["Change", "What it fixed", "Result"],
        [["Morphological merge turned off", "Fused cylinders; ballooned edge-bander fragments", "Instance clustering instead (`label_instances.py`)"],
         ["Edge-bander outlines tightened with SAM", "Frame-filling polygons", "Boxes > 80% of the image: train 48% → 20%, val 59% → 18%"],
         ["Per-cylinder extinguishers", "One blob per row", "Instances 435 / 105 / 96 → 606 / 129 / 147"],
         ["64 images quarantined", "Unlabelable collages and dense rows", "38 / 16 / 10 out of the build"],
         ["Negative audit; 107 hard negatives", "Hidden targets; look-alike false positives", "Section 4.4"],
         ["30 high-angle images", "No elevated views at all", "Section 4.5"],
         ["Manual curation (`datasetchange`)", "Specific bad images and bad labels", "80 removed, 110 re-drawn, 3 excluded"]],
        align="lll", widths=[4.4, 4.8, 5.6], caption="What changed on 21 October", size=8.5)
    d.p("Backups of the labels before each step are in `archive/dataset_backups/9class_labels_pre_run8_2026-10-21/` and "
        "`archive/dataset_backups/9class_labels_pre_user_curation_2026-10-21/`.")

    d.h(2, "4.2 Class Extent Policies: Edge Banders vs Fire Extinguishers")
    d.table(
        ["Class", "Include", "Exclude"],
        [["Edge banding machine: one polygon per machine", "Chassis and base, hood, gluing / trimming / buffing units, attached infeed and "
          "outfeed conveyors, tape coil disc, control arm", "Panels on the conveyors, floor and walls, text and spec tables, inset or collage tiles, people, hoses that leave the machine"],
         ["Fire extinguisher: one polygon per cylinder", "The cylinder's own hose, nozzle, handle and wall strap", "Gas cylinders, drums, water jugs, red bins, hose reels, alarm points (these are hard negatives)"],
         ["Saws, jointers, planers", "Bed, fences, frame, attached extension tables and guards", "Loose stock, people, the room"]],
        align="lll", widths=[3.8, 5.6, 5.4], caption="Extent policy", size=8.5)
    d.bullets([
        "Two machines are two instances; touching extinguishers are separate instances. Never merge them.",
        "Label a partly hidden object only if at least 40% is visible, and trace only what is visible.",
        "A polygon covering more than 80% of the image is allowed only for a genuine close-up. If wall, floor or white background is inside, it is wrong.",
    ])

    d.h(2, "4.3 SAM 2.1 Large and How We Used It")
    d.p("We compared SAM 2.1 Base and Large by prompting both with the ground-truth boxes of 222 validation instances.")
    d.table(
        ["", "Mean IoU", "Boundary F1", "Jointer", "Band saw", "Mitre saw", "Edge bander", "ms / prompt", "GPU MB"],
        [["SAM 2.1 Base", "0.880", "0.794", "0.928", "0.890", "0.914", "0.776", "108", "776"],
         ["SAM 2.1 Large", "0.886", "0.802", "0.971", "0.920", "0.877", "0.774", "271", "1,423"]],
        align="lrrrrrrrr", widths=[2.6, 1.4, 1.6, 1.4, 1.5, 1.5, 1.7, 1.6, 1.5], caption="SAM 2.1 Base vs Large (box prompts, 222 instances)", size=8)
    d.p("Large is our default, but the gain is small (+0.006 IoU), it is 2.5× slower, and it loses on mitre saws. "
        "So we still review every mask. Three habits made SAM usable:")
    d.bullets([
        "**Clip every mask to its prompt box.** That removes room, floor and stock in one step.",
        "**Invert a mask only for boxes covering more than 80% of the image.** Inverting smaller ones flipped correct masks.",
        "**Skip SAM on white studio shots.** It often picks the background there, so the `fg` mode cuts the object out of the non-white pixels instead.",
    ])

    d.h(2, "4.4 Quarantine, Manual Curation & Hard Negatives")
    d.h(3, "The 64 quarantined images")
    d.p("These are not all diagrams. 60 are edge-bander images the policy cannot label (collages, infographics, inset photos, "
        "rooms with several machines), and 4 are dense, blurred rows of extinguishers. They stay out of the build until someone re-draws them by hand.")
    d.h(3, "Manual curation through `datasetchange`")
    d.p("A manual review produced two lists in `datasetchange`: images whose labels were wrong, and images that should not be "
        "in the dataset at all. `parse_user_curation.py` cleaned them up (113 unique fix stems; 87 removal lines, 80 unique).")
    d.bullets([
        "**80 bad images** went into `user_quarantine`: dropped from every list at build time, never deleted from `source_38class`.",
        "**110 labels re-drawn** with SAM 2.1 L from reviewed boxes (`fix_user_labels.py`), giving 128 instances, 62 of them "
        "edge banders. Some fixes added machines the old labels had missed. 28 first-pass proposals were wrong (mostly white "
        "backgrounds); 27 were redone and 1 excluded.",
        "**3 excluded** because nothing labelable was there: an extrusion-line infographic, a roller-conveyor collage, and a table saw under 40% visible.",
        "**Checks passed:** audit PASS; 0 quarantined images still listed; 110 of 110 fixes carry valid polygons; 6 re-drawn polygons self-intersect (warning only).",
    ])
    d.h(3, "Negatives: auditing the old ones, adding targeted ones")
    d.p("We ran all 420 sampled negatives through run5 and run6 at conf 0.10 and reviewed every hit. Two kinds had to go: "
        "**5 images that actually contained a target machine** (a benchtop planer, a panel saw at the edge, background band "
        "saws), and **all 12 veneer-press images**, because the Camera 03 press is a woodworking hot press and we would have "
        "been teaching the model that it is background. Negatives went from 295 / 60 / 65 to 283 / 58 / 62.")
    d.p("We then collected **107 targeted hard negatives** (`datasets/hard_negatives/`): about 10 each of red bins, yellow "
        "and blue drums, gas cylinders, water jugs, cardboard tubes, hose reels, alarm call boxes, empty floors and ceilings, plus "
        "7 workbenches. They aim squarely at run7's extinguisher false alarms. Together the negatives make up 11.8% of the 3,298-entry train list.")

    d.h(2, "4.5 Integration of High-Angle Overhead Industrial Images")
    d.p("From the 99 high-angle images that passed the stage 2c gate, we accepted **30 images with 52 instances**: mitre saw 12, "
        "press 9, panel saw 7, table saw 6, edge bander 6, band saw 5, extinguisher 4, planer 2, jointer 1. The 69 rejects "
        "were mostly unlabelable scenes (14), ambiguous classes (13), part close-ups (12) and images with no target (10). "
        "SAM failed on two (#5, #94), which still need hand drawing. The images are train-only and added once "
        "(`train_highangle_9class.txt`). They are the first elevated views the model has ever seen, and still far short of the "
        "≥ 5 viewpoints per class that real CCTV needs.")


def ch5(d, root):
    d.h(1, "5. Model Training History & Quantitative Benchmarks")
    d.big_picture(
        "Eight runs, three eras. The 23-class and 38-class runs taught us that breadth costs accuracy but builds useful "
        "features. run5 narrowed to nine classes and became the model to beat. run6 and run7 were our attempts to make it "
        "CCTV-ready, and both made it worse in ways the validation numbers did not show. run8 is the first run built on clean "
        "labels and working augmentation. The tables below carry the numbers; the text explains why they moved.",
    )

    d.h(2, "5.1 Run 1–4 Historical Context (23-Class to 38-Class Habitat Experiments)")
    d.table(
        ["Run", "Data", "Base", "Epochs", "Box / mask mAP50", "Train time"],
        [["run1", "23-class web dataset (oversampled)", "COCO yolo26s-seg", "79 of 100", "0.777 / 0.755", "10.6 h"],
         ["run2", "23-class, cleaned labels", "run1", "20", "0.787 / 0.765", "5.2 h"],
         ["run3", "38-class “habitat” (machines + infrastructure + safety)", "COCO yolo26s-seg", "57 of 100", "0.608 / 0.544", "11.7 h"],
         ["run4", "38-class, unified labels", "run3", "15", "0.611 / 0.552", "2.6 h"]],
        align="lllrrr", widths=[1.2, 5.4, 2.8, 1.6, 2.4, 1.6], caption="The early runs (imgsz 960, batch 8)", size=8.5)
    d.p("Going from 23 to 38 classes dropped mean mAP by about 17 points: more classes, many of them small infrastructure "
        "items. But run3 learned what factory context looks like, which is why it, and not run2, became run5's parent. Every "
        "run in this period used `optimizer=auto`, so the `lr0` in its settings file was never actually used (Section 7.2).")

    d.h(2, "5.2 Run 5: The Generalist Champion")
    d.p("A 20-epoch fine-tune of run3 on the 9-class catalog data at imgsz 960. It reached box mAP50 0.832 at its best epoch "
        "and 0.825 at the end, in 37 minutes. Three things keep it the champion:")
    d.bullets([
        "It is the only model that **finds the held-out workshop saw** (0.56–0.63 on the test frames at imgsz 960).",
        "It produces **zero extinguisher false alarms** on Camera 03.",
        "It has the **best catalog localisation** at its training size (box mAP50-95 0.706).",
    ])
    d.p("**Its known flaw:** on 200 sampled workshop frames it finds the saw every time (max 0.70), but calls it `table_saw`. "
        "The machine is recognised, and the class boundary in our catalog labels is wrong (Section 10.2).")
    d.table(
        ["Class", "run5 box", "run5 mask", "run6 box", "run6 mask", "run6 R", "Δ box"],
        [["panel_saw", "0.893", "0.824", "0.848", "0.820", "~0.75", "−0.045"],
         ["hydraulic_press", "0.963", "0.937", "0.961", "0.916", "~0.88", "−0.002"],
         ["table_saw", "0.937", "0.937", "0.931", "0.904", "~0.83", "−0.006"],
         ["planer", "0.849", "0.822", "0.846", "0.846", "~0.80", "−0.003"],
         ["jointer", "0.877", "0.877", "0.869", "0.869", "~0.80", "−0.008"],
         ["band_saw", "0.883", "0.877", "0.840", "0.830", "~0.78", "−0.043"],
         ["miter_saw", "0.783", "0.757", "0.733", "0.724", "~0.70", "−0.050"],
         ["edge_banding_machine", "0.574", "0.497", "0.584", "0.558", "~0.41", "+0.010"],
         ["fire_extinguisher", "0.718", "0.720", "0.680", "0.693", "~0.52", "−0.038"],
         ["**all**", "**0.831**", "**0.805**", "**0.810**", "**0.795**", "**0.718**", "**−0.021**"]],
        align="lrrrrrr", widths=[3.8, 1.7, 1.7, 1.7, 1.7, 1.5, 1.7],
        caption="Per-class AP50, run5 vs run6 (training-time validation, before the cleanup; run6 recall read off its curves)", size=8)
    d.figure("runs/run5_9class_clean_finetune/BoxPR_curve.png", "run5 box precision-recall per class. Edge bander and extinguisher trail every other class.", 13, root)
    d.table(
        ["run5 measured on", "Box mAP50", "Mask mAP50"],
        [["Its own training-time val (482 images)", "0.825", "0.802"], ["Catalog val, 19 Oct labels, imgsz 960", "0.821", "0.798"],
         ["Cleaned val, 21 Oct (461 images)", "0.789", "0.741"], ["Curated val, after `datasetchange` (450 images)", "re-measure", "re-measure"]],
        align="lrr", widths=[8, 3, 3], caption="One model, four validation sets: why baselines must be re-measured", size=8.5)

    d.h(2, "5.3 Run 6: The “CCTV-Robust” Fine-tune & Uncovered Regressions")
    d.p("We fine-tuned run5 at imgsz 1280 for 50 epochs with 301 machinery negatives and heavier geometric augmentation. "
        "Validation looked fine: box 0.810, mask 0.795, precision 0.87. On real video it failed: the workshop saw dropped "
        "from 0.56–0.63 to ≤ 0.18, with **0 detections at conf 0.40 in all 1,398 frames**, and on factory frames it "
        "started calling things extinguishers (29 false detections in 16 of 40 frames).")
    d.p("**Why the numbers lied:**")
    d.bullets([
        "**The negatives pushed the wrong boundary.** Spindle moulders, veneer presses, CNC routers and racking are long, flat, "
        "table-and-beam machines, just like a sliding-table saw. We taught the model that this shape is background.",
        "**The degradation we thought we added never ran.** Albumentations was not installed, so there was no blur, codec or "
        "greyscale augmentation, and `erasing` does nothing for segmentation (Section 7.1).",
        "**The validation set could not see the problem.** It held catalog photos only: no CCTV, no desks, no ceilings.",
        "**The mask head was memorising.** Train segmentation loss halved (1.71 → 0.78) while validation segmentation loss stayed flat (1.35 → 1.44).",
    ])

    d.h(2, "5.4 Run 7: CCTV Pilot Hybrid Experiment & Catastrophic Overfitting Breakdown")
    d.p("Our first run with real factory frames: run5 plus 257 Camera 05 frames repeated 5×, which was 44% of every epoch "
        "(lr0 2e-4, imgsz 1280), validated on Camera 03. On Camera 05 it scored about 0.97 on 100% of frames. On Camera 03 "
        "it found **0 of 142 machines**, raised 51–74 extinguisher false alarms, and lost 3 points of catalog box mAP50 "
        "and 9–11 points of mAP50-95. Validation fitness never rose above 0.012, so early stopping picked `best.pt` "
        "essentially at random (epoch 7 of 17). Chapter 6 explains why.")

    d.h(2, "5.5 Master Training Metrics Matrix")
    d.table(
        ["Run", "Ep.", "P (box)", "R (box)", "Box mAP50", "Box 50-95", "Mask mAP50", "Mask 50-95", "Val seg loss first → last"],
        [["run1", "79", "0.843", "0.708", "0.777", "0.675", "0.755", "0.645", "2.04 → 1.80"],
         ["run2", "20", "0.842", "0.729", "0.787", "0.687", "0.765", "0.659", "2.21 → 1.92"],
         ["run3", "57", "0.674", "0.572", "0.608", "0.493", "0.544", "0.424", "3.55 → 2.43"],
         ["run4", "15", "0.686", "0.583", "0.611", "0.507", "0.552", "0.433", "2.44 → 2.56"],
         ["run5", "20", "0.847", "0.788", "0.825", "0.711", "0.802", "0.679", "1.46 → 1.39"],
         ["run6", "50", "0.870", "0.718", "0.810", "0.649", "0.795", "0.654", "1.34 → 1.44"],
         ["run7*", "17", "–", "–", "0.002", "–", "–", "0.000", "3.30 → 2.38"],
         ["run8", "1", "0.847", "0.764", "0.830", "0.629", "0.813", "0.671", "1.37 (epoch 1)"]],
        align="lrrrrrrrl", widths=[1.2, 0.9, 1.3, 1.3, 1.5, 1.5, 1.6, 1.6, 3.9],
        caption="Final-epoch metrics from each run's results.csv (own validation sets)", size=8)
    d.p("* run7 validated on Camera 03 only, so its numbers measure transfer to an unseen camera, not catalog quality. "
        "Read the last column as the recurring theme: from run5 onwards, validation segmentation loss barely moves while "
        "training loss halves. Mask quality is limited by label noise, which is why Chapter 4 mattered more than more epochs. "
        "run8's first epoch already matches run6's final validation losses, with augmentation that now actually runs.", size=9)


def ch6(d):
    d.h(1, "6. Deep-Dive Failure Analyses & Empirical Case Studies")
    d.big_picture(
        "Four failures taught us most of what we know about this problem. Each case study gives the symptom, what we "
        "suspected, what the evidence said, and what we changed.",
    )

    d.h(2, "6.1 Case Study 1: The Workshop Clip Panel Saw Regression (Resolution vs Viewpoint)")
    d.p("**Symptom:** run6 lost a saw that run5 found (frames 186 / 1210 / 1303: run5 0.56 / 0.61 / 0.63; run6 ≤ 0.18, "
        "mostly as table saw or a spurious extinguisher).")
    d.table(
        ["What we suspected", "What the evidence said", "Verdict"],
        [["The saw is too small, so try SAHI tiling", "It is 266×208 px on a 480×360 frame: 32% of the area", ("badge", "WRONG", R)],
         ["Inference resolution", "Same frame, run5: 0.56 at imgsz 960, 0.27 at 1280. Upsampling 480 px 2.7× creates blurry giants the model never saw", ("badge", "RIGHT", G)],
         ["run6's data mix", "Look-alike machinery negatives; run6 fails at every size", ("badge", "RIGHT", G)],
         ["Class boundary", "run5 recognises the saw but calls it table saw (max 0.70)", ("badge", "RIGHT", G)]],
        align="llc", widths=[4.4, 8, 2.4], caption="Workshop-clip diagnosis", size=8.5)
    d.p("**What changed:** fine-tunes start from run5; we never upsample a sub-stream; SAHI was dropped for static ROI crops; "
        "the sliding-table-saw boundary goes into the labels. A side lesson: the trial recordings had the CCTV tool's overlay "
        "burned in, and both models fired on it. Test clips must come from raw RTSP.")

    d.h(2, "6.2 Case Study 2: Camera 03 vs Camera 05 Memorization & Spatial Bias")
    d.table(
        ["Test", "run5", "run6", "run7"],
        [["Camera 05 (run7's training camera), raw, 200 frames", "0 / 200", "0 / 200", "**200 / 200 at ~0.97**"],
         ["Camera 03 machines found at ≥ 0.25 (of 142)", "0", "0", "**0**"],
         ["Camera 03: press labelled `edge_banding_machine` (imgsz 640)", "–", "–", "64 of 71 frames"]],
        align="llll", widths=[7, 2.2, 2.2, 3.4], caption="Training camera vs unseen camera", size=8.5)
    d.p("Lowering the threshold to 0.01 recovered 3 of 142 machines, so this was a failure to transfer, not a threshold "
        "problem. We traced it to four root causes:")
    d.numbered([
        "**One viewpoint per class.** 257 frames from one fixed camera amount to roughly one view of information. We repeated it "
        "5×, and the model rewarded us by memorising it.",
        "**Different machines.** Camera 03 has a compact manual edge bander and a hot press. Camera 05 has a long automatic line "
        "and a white frame press. One example per class cannot cover another model of the machine.",
        "**Position, not appearance.** In Camera 05, the edge bander is a long white shape at the upper centre of the frame. "
        "In Camera 03, the press sits exactly there, and run7 called it an edge bander 64 times.",
        "**Scene texture.** run7 scored 0.96–0.97 even on the 2:1-squeezed raw video it never trained on. Confidence that "
        "survives that distortion is matching the scene, not the machine.",
    ])

    d.h(2, "6.3 Case Study 3: The Fire Extinguisher False-Alarm Explosion")
    d.p("On Camera 03, which has no extinguishers, false alarms at conf 0.25 went from **0 (run5) to 32 / 2 (run6) to 51 / 74 "
        "(run7)** at imgsz 640 / 1280. run7 put them on a white and yellow plastic drum, cardboard tubes, water bottles and "
        "standing workers: vertical, cylinder-ish shapes it had never been told were background, after 44% of every epoch "
        "on a single scene with 0% hard negatives. run8 answers with the 107 targeted negatives (drums, tubes, cylinders, "
        "bins) and keeps `ToGray` at p = 0.05 so colour stays a cue. What is still missing is CCTV-domain cylinder negatives, "
        "which Camera 03 cannot supply while it is in the benchmark.")

    d.h(2, "6.4 Case Study 4: Catalog Catastrophic Forgetting During Unbalanced Fine-Tuning")
    d.table(
        ["Model (catalog val, imgsz 960)", "Box mAP50", "Box 50-95", "Mask mAP50", "Mask 50-95"],
        [["run5", "0.821", "0.706", "0.798", "0.679"], ["run6", "0.825", "0.698", "0.813", "0.690"],
         ["run7 best (ep 7)", "0.786", "0.597", "0.767", "0.633"], ["run7 last (ep 17)", "0.792", "0.614", "0.771", "0.645"]],
        align="lrrrr", widths=[5.4, 2.3, 2.3, 2.3, 2.3], caption="Forgetting under a one-camera fine-tune", size=8.5)
    d.p("56% catalog replay held mAP50 within about 3 points, but box mAP50-95 fell 9–11 points: the model still found "
        "machines, just with sloppier boxes. That fails the “within 2 points of run5” gate. Two practical notes: run5 "
        "looks bad at imgsz 1280 only because that is outside its training scale, so always compare checkpoints at their "
        "training size; and replay limits forgetting but does not prevent it when one domain dominates the epoch.")


def ch7(d):
    d.h(1, "7. Critical Bug Fixes, Gotchas & Engineering Lessons")
    d.big_picture(
        "The most expensive bugs in this project never raised an error. A missing package, renamed arguments, an "
        "“auto” setting and an ignored parameter each made a training recipe mean something other than what it "
        "said. Each section below is one silent failure and the guard we put in place.",
    )

    d.h(2, "7.1 The Silent Albumentations Bypass in Ultralytics 8.4")
    d.p("Ultralytics wraps its Albumentations block in an import guard. Without the package, blur, compression, greyscale and "
        "any custom `augmentations=` list are skipped **with no error**. run1–run6 therefore trained with no photometric "
        "degradation beyond HSV jitter. We installed Albumentations 2.0.8 on 15 October (with `--no-deps`, to keep the GUI "
        "OpenCV build), and `train.py` now refuses to start without it:")
    d.code("""
try:
    import albumentations as A
    import cv2
except ImportError:
    raise SystemExit("albumentations is not installed; Ultralytics would silently skip cctv_aug.")
""", caption="scripts/train.py: fail fast (message shortened)")
    d.p("**Habit:** check every training log for the `albumentations:` line.")

    d.h(2, "7.2 Albumentations 2.x API Compatibility & Parameter Traps")
    d.p("Albumentations 2.x silently ignores 1.x argument names and falls back to defaults, so a 1.x recipe runs, just with "
        "different strengths from the ones written down. We rewrote the pipeline with 2.x names:")
    d.table(
        ["Transform", "1.x (ignored)", "2.x (used)"],
        [["ImageCompression", "`quality_lower=30, quality_upper=75`", "`quality_range=(30, 75)`"],
         ["Downscale", "`scale_min=0.4, scale_max=0.75`", "`scale_range=(0.4, 0.75)`"],
         ["GaussNoise", "`var_limit=(10, 50)`", "`std_range=(0.012, 0.028)`  (√10/255, √50/255)"]],
        align="lll", widths=[3.2, 5.6, 6], caption="Argument migration", size=8.5)
    d.code("""
A.OpticalDistortion(distort_limit=(-0.05, 0.25), interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101, p=0.35),  # CCTV lens curvature
A.ImageCompression(quality_range=(30, 75), p=0.4),       # H.264-like blocking
A.Downscale(scale_range=(0.4, 0.75), p=0.3),             # low-res sensor / sub-stream
A.GaussNoise(std_range=(0.012, 0.028), p=0.3),           # sensor noise
A.OneOf([A.MotionBlur(blur_limit=(3, 7)), A.GaussianBlur(blur_limit=(3, 5))], p=0.15),
A.ToGray(p=0.05),                                        # keep colour cues
A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
""", caption="cctv_aug in scripts/train.py (run8)")
    d.p("The degradation is deliberately milder than run7's, because run7's mAP50-95 loss pointed at localisation damage. "
        "run8 also uses explicit AdamW (lr0 5e-4, cosine), imgsz 960, batch 6, and turned-down geometry (`degrees=0`, `perspective=0.0002`).")
    d.h(3, "Why optimizer=auto is dangerous")
    d.p("With `optimizer=auto`, Ultralytics ignores `lr0` and picks AdamW with a learning rate fitted to the class count, "
        "`round(0.002 × 5 / (4 + nc), 6)`. For our nine classes that is **0.000769**. run5 and run6 both list "
        "`lr0=0.01` in their settings and trained 13× slower than that. Any conclusion about learning rate drawn from "
        "those runs is void. **Rule:** always set the optimizer explicitly.")

    d.h(2, "7.3 Wide-Angle CCTV Lens Barrel Distortion Emulation")
    d.p("We checked on a grid image that in Albumentations 2.x a **negative** `distort_limit` gives pincushion and a "
        "**positive** one gives barrel, which is the opposite of what many older recipes assume. Wide-angle CCTV lenses "
        "barrel, so we sample (-0.05, 0.25), mostly barrel, at p = 0.35, with reflected borders so the model cannot learn "
        "black corners as a cue. Ultralytics 8.4 passes polygon vertices to Albumentations as keypoints, so labels warp with "
        "the image. Distortion emulates the lens. It does not create the elevated viewpoint.")

    d.h(2, "7.4 Aspect Ratio Distortions: Managing 960×1080 “1080P Lite” Frames")
    d.p("Every factory recording is 960×1080: Hikvision's “1080P Lite” stores a 16:9 picture at half width, without "
        "declaring the pixel aspect ratio, so every tool shows machines at half their real width. We de-squeeze to "
        "1920×1080 before labelling and evaluation (`sample_cctv_video.py --resize 1920x1080`, cubic interpolation), and "
        "the frozen benchmark is built that way. On 40 factory frames, run5's scores were too low to choose between native and "
        "de-squeezed input (at most 11 of 40 frames with any detection either way), so we follow geometry: de-squeezed frames "
        "match the shapes the model learned from catalog photos.")

    d.h(2, "7.5 NMS Independence in YOLO26 End2End Detection")
    d.p("Our checkpoints run the NMS-free one-to-one head, so `iou=` has **no effect** (`ultralytics/utils/nms.py:66`). "
        "`end2end=False`, or `--nms` in the harness, switches to the one-to-many head with NMS. Every inference summary now "
        "prints which head ran. Any past experiment that tuned `iou` was a no-op and should be rerun if the question still matters.")
    d.table(
        ["Lesson", "Guard we adopted"],
        [["Catalog validation mAP does not predict CCTV performance", "Gate every candidate on the factory benchmark and the held-out clip"],
         ["Frames are not viewpoints", "Weight CCTV data by camera × machine pairs; at most 1–2× repeats"],
         ["Libraries fail silently", "Fail-fast checks; explicit optimizer; read the training log"],
         ["Labels cap accuracy", "A written extent policy and reviewed SAM outlines"],
         ["Noise-level validation picks checkpoints at random", "Select on catalog val or run fixed epochs; report CCTV separately"],
         ["Junctions make deletes destructive", "The lists define the dataset; `cmd /c rmdir` only"]],
        align="ll", widths=[7, 7.8], caption="What the silent failures taught us", size=8.5)


def ch8(d):
    d.h(1, "8. Edge Deployment, Real-Time Inference & Video Processing")
    d.big_picture(
        "Nothing in this chapter is running yet. It is the deployment design we will build once a checkpoint passes the "
        "gates in Chapter 9. Its core idea is simple: every object we detect is stationary, so we confirm presence at known "
        "locations over seconds, rather than chasing every frame.",
    )
    d.h(2, "8.1 Ingestion Architecture: RTSP Main Stream vs Sub-Stream Analysis")
    d.table(
        ["Source", "Inference recipe"],
        [["Sub-stream 480×360 / 640×360", "imgsz 640 (sweep 480 / 640 / 960). Never 1280: upsampling halved run5's confidence"],
         ["Main stream, 1080P Lite (960×1080)", "De-squeeze to 1920×1080, then imgsz 960 or ROI crops"],
         ["Main stream, full 1080p / 4 MP", "imgsz 1280 letterboxed, or ROI crops"]],
        align="ll", widths=[5.4, 9.4], caption="Stream and size", size=8.5)
    d.p("Record raw RTSP, never screen recordings (no overlay, no double compression). On Hikvision the main stream is "
        "channel 101 and the sub-stream 102; on Dahua, use `subtype=0`. Decode at 5 fps: at 17–22 ms per frame, one RTX "
        "3070 Ti serves about 8–10 cameras.")

    d.h(2, "8.2 Static Camera ROI Cropping Strategy (Why SAHI was Deprioritized)")
    d.p("SAHI tiles a frame to find small objects. Our machines are not small (the workshop saw is 32% of the frame), "
        "tiling a 480×360 stream only re-samples the same pixels, and a 4K frame costs 6–9 forward passes. Instead, "
        "because cameras and machines do not move, we define 1–3 regions of interest per camera once, crop them from the "
        "main stream at native resolution, and map the masks back. A full-frame pass every 30th frame catches anything that "
        "moved. SAHI stays an option only for far-wall extinguishers under about 40 px on a wide main stream.")

    d.h(2, "8.3 Multi-Frame Temporal Filtering: ByteTrack Association & 8-of-15 Window Rules")
    d.bullets([
        "**Detect low, decide slowly:** detection at conf 0.10, ByteTrack association (`track_high_thresh` 0.35, "
        "`track_low_thresh` 0.10, `new_track_thresh` 0.45, `track_buffer` 50 = 10 s at 5 fps). Set the tracker explicitly; "
        "the Ultralytics default is `tracktrack.yaml`.",
        "**Confirm** a track when at least 8 of the last 15 frames match above the class threshold. **Drop** it when 2 or fewer match for the whole buffer.",
        "**Class by vote:** a confidence-weighted vote over the track's lifetime, which stops single-frame table-saw / panel-saw flips.",
        "**Learn from doubt:** frames with a confirmed track below 0.3 confidence, or a split vote, go to the labelling queue.",
    ])

    d.h(2, "8.4 Per-Class Operational Confidence Thresholds & Decision Logic")
    d.table(
        ["Class", "Threshold", "Basis (run6 catalog curves; re-derive on CCTV val)"],
        [["edge_banding_machine", "0.20", "F1 peaks at 0.16; recall collapses above 0.3"], ["fire_extinguisher", "0.30", "F1 plateau 0.15–0.45"],
         ["All other classes", "0.40", "Aggregate F1 plateau 0.20–0.50"]],
        align="lcl", widths=[4, 2, 8.8], caption="Provisional thresholds", size=8.5)
    d.table(
        ["Event", "Action"],
        [["Machine confirmed in its ROI", "Update inventory; no alert"], ["No confirmed machine for N minutes in working hours", "Low-priority check request"],
         ["Extinguisher confirmed at its position", "Mark verified"], ["Extinguisher not confirmed", "Mark **unverified** for human review. **Never alarm**"],
         ["Tiny box (machine < 0.3% of frame, extinguisher < 16×16 px)", "Discard before tracking"]],
        align="ll", widths=[7, 7.8], caption="Decision logic (proposed)", size=8.5)


def ch9(d):
    d.h(1, "9. The Frozen Factory CCTV Benchmark & Production Verification Gates")
    d.big_picture(
        "run7 taught us that a camera used in training can never again tell you anything about generalisation. So on 21 "
        "October we froze all the factory frames we have as a test-only benchmark, with guards in code, and wrote down the "
        "gates a model must pass before anyone deploys it.",
    )
    d.h(2, "9.1 Dataset Isolation Protocol: The 328-Frame Frozen Benchmark")
    d.table(
        ["Camera", "Recordings", "Frames", "Labelled instances"],
        [["cam03", "`ch03_20260907165321` (37), `ch03_20260924144142` (34)", "71", "press 71, edge bander 71"],
         ["cam05", "`working_panel_saw` (120), `panel_saw_hydraulic_lift` (120), `ch05_20260928130326` (17)", "257", "panel saw 257, press 257, edge bander 257"],
         ["**Total**", "5 recordings, 1 frame / 3 s, de-squeezed to 1920×1080", "**328**", "panel saw 257, press 328, edge bander 328"]],
        align="llrl", widths=[1.8, 7.6, 1.6, 3.8], caption="Benchmark composition (docs/cctv_benchmark.md)", size=8)
    d.bullets([
        "**Absent classes:** the other six classes do not appear, so any detection of them is a false positive, especially an extinguisher on Camera 03's drum.",
        "**Labels:** SAM 2.1 box prompts clipped to each machine's consensus outline, with workers and on-screen text removed; spot-checked, not hand-drawn.",
        "**Limits:** 3 of 9 classes and 2 viewpoints, so it measures transfer to these cameras, not general CCTV readiness.",
    ])
    d.figure("docs/factory_footage_contact_sheet.jpg", "The two factory cameras behind the benchmark.", 14, d._root)
    d.code("""
from ultralytics import YOLO
YOLO(r"<weights>").val(data=r"datasets\\9class\\data_9class_cctv_benchmark.yaml", imgsz=960, plots=False)
""", caption="Benchmark evaluation")

    d.h(2, "9.2 Validation Rules: Zero-Tolerance Leakage Guardrails")
    d.bullets([
        "Never train on benchmark frames, or on any other footage from Camera 03 or 05. `train.py` exits if the data file "
        "is the benchmark or mentions `cctv_factory_intake`.",
        "Never use the benchmark for checkpoint selection, early stopping, threshold tuning or hard-negative mining.",
        "Split by camera, never by frame. `camera_splits.yaml` marks every recording as test, and the hybrid builder fails if a camera lands in two splits.",
        "run7's contaminated training lists are archived in `archive/run7_cctv_pilot_2026-10/`.",
    ])

    d.h(2, "9.3 Gate Criteria for Champion Model Deployment")
    d.table(
        ["Gate", "Criterion", "run5 today"],
        [["1. CCTV accuracy", "Per-class mask mAP50 ≥ 0.60 and recall ≥ 0.80 on held-out cameras (classes with < 50 instances reported, not gated)", ("badge", "FAIL (0.054 / 0.010)", R)],
         ["2. False alarms", "≤ 1 per camera-hour after confirmation; ≤ 1% image-level on ≥ 300 negatives", ("badge", "NO DATA YET", Y)],
         ["3. No forgetting", "Catalog val mask mAP50-95 within 2.0 points of run5 (plan definition; we also report box mAP50)", ("badge", "BASELINE", G)],
         ["4. Trial clips", "Raw RTSP clips without overlay; beat run5 on the workshop saw", ("badge", "CLIPS NOT RECORDED", Y)]],
        align="llc", widths=[2.8, 9, 3], caption="Release gates (all four must pass)", size=8.5)
    d.p("run8 has not been evaluated against any gate yet. Gate 3 needs run5 re-measured on the current 450-image val first, "
        "and gate 2 needs empty-scene footage and negatives from cameras outside the benchmark.")


def ch10(d):
    d.h(1, "10. Current Handoff Status, Outstanding Curation & Future Roadmap")
    d.big_picture(
        "If you are picking this project up, here is the short version: let run8 finish, measure it honestly against run5 on "
        "the current validation set and the frozen benchmark, then spend your effort on viewpoints, not epochs. The model is "
        "not the bottleneck. Real elevated views of these machines are.",
    )
    d.h(2, "10.1 Immediate Execution Tasks: Training Run 8 & Validating Curation Lists")
    d.numbered([
        "**Let run8 finish** (resumed from `last.pt` on 23 Oct at 17:07 and again at 18:03; 50 epochs planned). Do not run another GPU job alongside it on the 8 GB card.",
        "**Find out why it is slow:** 3,525 s for epoch 1 against run5's 111 s. Check GPU utilisation and dataloader CPU load.",
        "**Re-baseline run5** on the current 450-image val at imgsz 960.",
        "**Evaluate run8** (`best.pt` and `last.pt`) on that val, on the frozen benchmark, and on the workshop clip at imgsz 640 and 960, then apply the gates.",
        "**Curation lists:** `datasetchange` is applied and audited (PASS). What remains is a human pass over "
        "`label_fixes/review_user_fixes/` and a decision on the 6 self-intersecting polygons.",
    ])

    d.h(2, "10.2 Pending Class Boundary Refinements")
    d.table(
        ["Open question", "Why it matters", "Our recommendation"],
        [["Sliding-table saw: `panel_saw` or `table_saw`?", "run5 finds the workshop saw but calls it table saw", "Label sliding-table saws as `panel_saw`, with an example sheet"],
         ["Veneer press", "12 images removed because they resemble the Camera 03 press", "Relabel them as `hydraulic_press`"],
         ["Moulders and drum sanders", "Look like planers and table saws", "Keep them as negatives; add them to the boundary sheet"],
         ["64 quarantined images", "16 are val images", "Hand re-annotation (optional)"],
         ["High-angle #5 and #94; FE_00025 hose fragment", "SAM failed; a detached hose fragment in one label", "Hand-draw, or drop"]],
        align="lll", widths=[4.6, 5, 5.2], caption="Decisions waiting on the team", size=8.5)

    d.h(2, "10.3 Long-Term Roadmap: Multi-Camera Tracking, Night IR Adaptation & Worker Perimeter Zones")
    d.table(
        ["Horizon", "Work", "Why"],
        [["Next", "Record raw RTSP for ≥ 24 h per camera (day, night, IR), plus pseudo-CCTV phone video from 3–4 m up; target ≥ 5 viewpoints per class", "Viewpoint diversity decided run7"],
         ["Next", "Build CCTV val / test and ≥ 300 CCTV negatives from cameras outside the benchmark", "Gates 1 and 2 need them"],
         ["Mid", "Release retrain from COCO weights once labels settle (AdamW lr0 1e-3, 150 epochs)", "Remove biases inherited from old labels"],
         ["Mid", "Multi-camera presence: per-camera ROI state machines feeding one site registry", "Stationary objects need no re-identification"],
         ["Mid", "Night IR: an IR val split and greyscale-heavy training", "Extinguishers scored about 0 in greyscale tests"],
         ["Mid", "Worker perimeter zones: a COCO person detector plus dilated machine masks and dwell rules; privacy review first", "The safety use case"],
         ["Ongoing", "Active learning: label doubtful frames weekly, fine-tune monthly with replay", "Keeps the model current"]],
        align="lll", widths=[1.6, 9, 4.2], caption="Roadmap", size=8.5)

    d.h(2, "10.4 Document Revision History & Technical Ownership Sign-Off")
    d.table(
        ["Version", "Date", "Change"],
        [["2.4", "2026-10-23", "First consolidated dossier, compiled from project reports and run artifacts"],
         ["2.5", "2026-10-23", "Rewritten in the team's narrative voice; condensed; data-harvesting pipeline expanded into Chapter 3"]],
        align="cll", widths=[1.6, 2.4, 10.8], caption="Revision history", size=8.5)
    d.p("**Primary sources:** `docs/runsummary.md`, `docs/run7_analysis_report.md`, `docs/cctv_domain_adaptation_plan.md`, "
        "`docs/label_policy.md`, `docs/cctv_benchmark.md`, `docs/cctv_scrape_postmortem.md`, "
        "`pipeline/machine-type-classifier-prd.md` and the `pipeline/pipeline/` source.", size=9)
    d.p("This dossier records engineering state; it does not approve deployment. That needs the gates in Section 9.3 and the signatures below.", keep=True)
    d.table(
        ["Role", "Name", "Signature", "Date"],
        [["Technical owner (ML)", "", "", ""], ["Data and labelling reviewer", "", "", ""], ["Operations / safety stakeholder", "", "", ""]],
        align="llll", widths=[5, 4, 3.5, 2.3], caption="Sign-off", size=9)


def appendix(d):
    d.h(1, "Appendix A. Where Briefs and Design Documents Differ From the Code")
    d.p("Each row records a claim from an authoring brief or a design document, what the artifacts actually show, and the value this dossier uses.")
    d.table(
        ["#", "Claim", "What the artifacts show", "Used here"],
        [["1", "run5 catalog box mAP50 82.1%", "0.821 is run5 at imgsz 960 on the 19 Oct labels; its own log ends at 0.825 (best 0.832)", "All, labelled by val set"],
         ["2", "run8 not yet trained", "Started 22 Oct; resumed 23 Oct (17:07, 18:03)", "In progress"],
         ["3", "run8 negatives 11.6%", "11.8% after the `datasetchange` curation", "11.8%"],
         ["4", "64 quarantined “diagrams”", "60 edge-bander collages and multi-machine scenes + 4 dense extinguisher rows", "Breakdown"],
         ["5", "Gate 3: within 2.0 points of run5", "Plan defines it as mask mAP50-95; the run7 report used box mAP50", "Plan definition, report both"],
         ["6", "Scraping via Bing / Google", "All scrapers use `ddgs` (DuckDuckGo; Bing-backed results). An icrawler/Bing HTML scraper was replaced; there is no Google scraper", "ddgs"],
         ["7", "MD5 byte-level de-duplication", "MD5 is only of the URL, for filenames; de-duplication is by URL set, pHash (stage 3) and dHash (stage 2c)", "As implemented"],
         ["8", "Minimum resolution 400×300; aspect-ratio checks", "Download floor 200 px, stage 3 floor 150 px; no aspect-ratio check", "As implemented; 400×300 recommended"],
         ["9", "Stage 5 uses K-Means / farthest-point sampling", "Greedy filter: ranked by CLIP text score, drop if cosine > 0.95 to any kept image", "As implemented"],
         ["10", "Calibration uses precision-first cut-offs", "Recall-first: good-set 1st percentile (fine) and the bad-max / good-p1 midpoint (coarse)", "As implemented"],
         ["11", "Fine filter rejects drill presses vs band saws", "No band-saw/drill-press dual check; dual checks exist for 4 other class pairs", "As implemented"],
         ["12", "SAM generates polygons from prompt boxes (stage 6)", "SAM runs inside a Roboflow workflow after Gemini verification; its prompt configuration is not in the repository. Only the first polygon per image is uploaded", "As implemented"],
         ["13", "Pipeline README: 23 classes", "`config.py` lists 38 class slugs", "38"],
         ["14", "Per-class precision and recall for run5", "Not exported by run5; only per-class AP50", "AP50 only"]],
        align="clll", widths=[0.6, 4, 7, 3.2], caption="Claims vs artifacts", size=8)
