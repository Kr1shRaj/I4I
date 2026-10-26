# `9_class_cctv_robust` — Technical Analysis (10 Questions)

**Date:** 2026-10-08  
**Companion to:** `runsummary.md`  
**Inference script:** `Krish\test_inference.py`

### Evidence base

Every claim below rests on one of these:

1. The run artifacts: `results.csv`, `args.yaml`, plots.
2. The installed library source: Ultralytics **8.4.123** in `instancesegmentationtraining\yolo-env`. File and line references point there.
3. Measurements made for this analysis:
   - **Label geometry** of the train/val/test lists.
   - **Per-instance recall slices** on the val split, at conf 0.39 and IoU 0.5.
   - **Trial-video re-runs** of the old checkpoint (`9_class_clean_finetune`) and the new one on identical frames.

Outputs from the trial-video re-runs are in `trial_videos\analysis\cctv_robust_inference\`.

### Facts that change the framing of several answers

| Fact | Evidence |
|---|---|
| **The model is a YOLO26-seg end-to-end (NMS-free) model.** It has a `reg_max=1` box head, so it uses **no DFL**. The third box term is an L1 loss. | `test_inference.py` reports `end2end(NMS-free): True`. `loss.py:357` names the term `l1_loss` when `reg_max == 1`. `tasks.py:661` selects `E2ELoss(..., v8SegmentationLoss)`. |
| **`iou=` has no effect at inference** on this checkpoint. NMS is skipped for end2end models. | `utils/nms.py:66` |
| **The optimizer was AdamW at lr0 = 0.000769, not SGD 0.01.** `optimizer=auto` picked AdamW because the run had only 2,600 iterations. Iterations = ceil(3,312 / nbs 64) × 50 = 2,600, which is below the 10,000 cutoff, so `lr0`/`momentum` in `args.yaml` were ignored. LR decays linearly to 2.3e-5 by epoch 50. | `trainer.py:1113-1121`; `results.csv` lr column (7.38e-4 at ep 3 → 2.29e-5 at ep 50) |
| **The training set has no desk, ceiling, office, or CCTV images.** The 301 negative images (9.1% of 3,312 list entries; 13.0% of 2,308 unique images) are all other machinery: grinders, scales, control panels, hoists, CNC routers, etc. | `train_oversampled_9class.txt` filename scan |
| **The new checkpoint regressed on the real workshop machine.** On the identical 480×360 stream frames from `Screen Recording 124839`, the old model scores `panel_saw` **0.56–0.63**. The new model scores **≤0.18**, and its top output is a spurious `fire_extinguisher` at 0.20–0.31. At conf 0.40 it detects **nothing in all 1,398 frames**. | `cctv_robust_inference\old_vs_new_checkpoint.json`, `124839_machine\summary.md` |

---

## 1. Epoch progression (`best_epoch = last_epoch = 50`): will 70–100 epochs help?

**Short answer: no, not meaningfully.** Best = last is an artifact of how this schedule ends, not a sign the model was still learning.

**The evidence:**

| Epoch window | Mean Box mAP50 | Std across epochs | Phase |
|---|---:|---:|---|
| 21–30 | 78.6% | — | mosaic on, LR 4.8e-4 → 3.3e-4 |
| 31–40 | 78.7% | — | mosaic on, LR 3.1e-4 → 1.8e-4 |
| 41–50 | **80.1%** | 1.0 pt | mosaic off (`close_mosaic=10`), LR 1.6e-4 → 2.3e-5 |

**Why "best = last" doesn't imply more epochs will help:**
- **The plateau came first.** For 20 epochs at a meaningful learning rate (epochs 21–40), validation mAP50 did not move (78.6 → 78.7).
- **The final +1.5 pt has two one-time causes.** Both happen only in the last 10 epochs:
  - **Mosaic switched off.** Training images then match the val distribution: single, large, centred objects.
  - **LR annealed to near zero.** That, plus the EMA weights, smooths the last epochs.
  - Any schedule that ends this way tends to put "best" at or near the last epoch. That says how the schedule ends, not that more training would help.
- **Longer runs repeat the plateau.** A 100-epoch run doesn't add 50 more "final" epochs. It stretches the whole schedule (warmup, 90 mosaic epochs, 10 closing epochs). You'd expect the same plateau for longer, then the same +1–2 pt bump at the end. Epoch-to-epoch noise in the last 10 epochs is already ±1.0 pt.
- **Mask quality already peaked.** Mask mAP50-95 reached its maximum of **66.0% at epoch 38**, not epoch 50 (65.4%).
- **Validation mask loss has been flat or rising since epoch 1** (see Q7).
- **The YOLO26 loss shifts weight away from the shared mask prototypes.** `E2ELoss` moves weight from the one-to-many head (0.8 → 0.1) to the one-to-one head (`loss.py:1303-1325`). The one-to-one branch receives a **detached** prototype (`head.py:428-430`). Longer runs therefore mostly refine the one-to-one classifier, not the masks.

**Risk vs reward:**

| | Box | Mask |
|---|---|---|
| Expected gain from 70–100 epochs | ≤1 pt mAP50, inside the noise band | none |
| Risk | larger domain overfit | larger mask overfit; more confident suppression of in-the-wild machines (see the regression in the framing table) |

**Recommendation:**
- Don't extend epochs.
- If you must, **resume from `last.pt` with a short schedule:** 10–15 epochs, `mosaic=0`, `lr0` around 1e-4, early-stop on mask mAP50-95.
- The data problems in Q2, Q3 and Q9 are worth far more than training longer.

---

## 2. Making the validation set measure what the run was meant to fix

**The problem.**
- The current val split (482 images) is the same kind of web/catalogue photography as the training set.
- Its 63 negatives are other machines.
- It contains **no CCTV frames and no desks, ceilings or overhead bars**.
- So the val split can't measure either goal of this run: CCTV-angle generalisation, and desk/ceiling false-positive suppression.
- The trial-video re-run shows exactly the failure the val split can't see: a real panel saw on a real camera went from detected (0.56–0.63) to undetected (≤0.18).

**Best practice: one evaluation set per objective, never pooled.**

| Set | Contents | What it measures | Minimum size |
|---|---|---|---|
| **A. In-domain regression** (keep current val) | web/catalogue images, current labels | "did we break anything we had" | as now |
| **B. CCTV positives** (new, held out) | frames from **your** cameras, same polygon policy, every target object labelled | mAP / recall per class on the deployment domain | ≥ 30–50 instances per class, ≥ 3 cameras, several days and lighting conditions |
| **C. Hard negatives** (new, held out) | desks, ceilings, overhead bars, monitors, empty floors, racking, non-target machines — **from your sites** | false positives per image and per camera-hour at the operating threshold | ≥ 300 images, or ≥ 1 h of empty-scene video per camera |
| **D. Video clips** (new) | 10–20 short labelled clips | temporal stability: track fragmentation, flicker rate | — |

**Rules:**

1. **Split by group, not by frame.**
   - Frames from one camera-day or clip go wholly to train or wholly to eval.
   - Adjacent frames are near-duplicates. Splitting them across train and eval inflates mAP and is the most common CCTV mistake.
2. **Keep the hard negatives in their own set (C).**
   - mAP ignores images without ground truth, except through false positives.
   - Pooled into the val split, the hard negatives disappear into the "background" column. That is what happened here: 67 background false positives, unattributable.
3. **Report false-positive rate with a confidence bound.**
   - With 63 negatives and zero false positives, the 95% upper bound on the image-level false-positive rate is about 3/63 = **4.8%** (rule of three).
   - Bounding it at 1% needs about 300 clean negatives.
4. **Select thresholds on B and C, then report on a third untouched test split.**
   - `test_9class.txt` should get its own CCTV and negative slices too.
5. **Match the slices to what the goal claims.**
   - Track at least overhead vs eye-level, near vs far (box < 2% of frame), day vs night, colour vs IR/greyscale.
   - The earlier session showed greyscale input collapses extinguisher confidence to about 0, so greyscale/IR needs its own slice.

---

## 3. Edge-banding machine: why 53% are missed, without cross-class confusion

**Root cause: a labelling-convention problem, not a capacity or aspect-ratio problem.**

**Measured on the val split (49 instances, conf 0.39, IoU ≥ 0.5):**

| Slice | Recall |
|---|---:|
| Label box covers **>95% of the image** (full-frame label) | 13/22 = 59% |
| Label box 25–95% of image | 6/12 = 50% |
| Label box **< 25% of image** | **0/15 = 0%** |
| Machine **alone** in image | 19/31 = 61% |
| **2 or more** edge banders in image | **0/18 = 0%** |

**Label geometry:**
- **44.9% of edge-banding val labels are full-frame boxes**, against 0–7% for every other class.
- In training, the median edge-banding box covers 78.5% of the image; the 10th percentile covers 4%.
- The median pixel aspect ratio is 1.0. That is the image's own aspect ratio, which is consistent with "box = whole photo".

**Mechanism:**
1. **The model learned that edge banders fill the frame.** Close-up product photos labelled with a frame-sized polygon are an unusual target. The network learns to predict this class from global context and frame-filling boxes, not from a localised object. When an edge bander is one object among several, or small in the scene, there is no learned pattern to fire on. Recall is exactly 0%.
2. **The assigner pushes confidence down when box targets disagree.**
   - Ultralytics' Task-Aligned Assigner builds the classification target from `score^0.5 × IoU^6` (`loss.py:364-371`: `alpha=0.5, beta=6.0`).
   - When the box target for a class is inconsistent (sometimes the whole image, sometimes a 4% box), predicted-box IoU stays low.
   - IoU^6 then crushes the soft classification target, so the head learns low confidence for the whole class.
   - That is why the F1 peak for this class sits at **conf 0.16**, while every other class peaks at 0.3–0.5. The objects are "found" but scored below any reasonable threshold. They don't show up as another class; they show up as background.
3. **No class confusion, because nothing else looks like the target.** Other classes are labelled as tight objects. The edge-banding signal ("whole-image industrial texture") overlaps with no other class's object-level features. It loses to background instead.
4. **Aspect ratio is a minor factor.**
   - True long edge banders do reach 8–11:1 boxes (max 10.98 in train).
   - The head is anchor-free with `reg_max=1`, and long boxes are representable, so aspect ratio alone doesn't explain 0% recall on small instances.
   - Mosaic truncation of long machines does contribute: see Q5. Instances keep their label with as little as 1% of their area remaining.

**Fix order:**
1. **Re-annotate first.**
   - Write an extent policy (machine body including infeed/outfeed tables, or not) and apply it to all 312 train/val/test edge-banding instances.
   - Remove frame-filling polygons. Split multi-machine images into separate instances.
   - This is the change that moves recall on scenes from 0%.
2. **Then add imagery.** Edge banders as **one object in a wider scene**: workshop wide shots, CCTV views, partial views along long machines.
3. **Don't just add more of the same close-up photos.** They reinforce the frame-filling prior.

---

## 4. Fire extinguisher: why 105 instances in 45 images, and why 46% are missed at high precision

**Why 105 instances in 45 images:**
- Extinguishers are **co-located by design**: stations, racks, product line-ups, catalogue shots of several sizes.
- In val, 15 of 45 images have 3 or more extinguishers (max 8). **67 of 105 instances (64%) are in those crowded images.** Train shows the same pattern: 435 instances in 210 images, max 9 per image.

**Where the misses are (val, conf 0.39, IoU ≥ 0.5; 57/105 recalled):**

| Slice | Recall |
|---|---:|
| Alone in image | 19/22 = 86% |
| 2 per image | 12/16 = 75% |
| **3 or more per image** | **26/67 = 39%** |
| Box > 25% of image | 20/24 = 83% |
| Box 5–25% | 23/37 = 62% |
| **Box < 5% of image** | **14/44 = 32%** |

Only 5 of the 48 misses have a same-class box at IoU 0.1–0.5. **The misses are absences, not loose boxes.**

**Explanation:**
- **"Small at 1280 px" is not the main cause on val.**
  - The median extinguisher's long side is 550 px at 1280. Only 2% are under 64 px.
  - The P3 stride-8 grid sees these comfortably.
  - "Small" here means small **relative to the scene**: 44 instances under 5% of the image, nearly all in crowded shots.
- **Crowding is the dominant factor.**
  - In this NMS-free YOLO26 model, the one-to-one head assigns each ground-truth object to exactly one prediction (`tal_topk2=1`) and must learn to suppress duplicates itself.
  - Several identical, touching, same-colour cylinders are the hardest case for learned duplicate suppression. The head merges or suppresses neighbours.
  - A classic NMS model at IoU 0.45–0.7 is more forgiving here.
  - **This is a hypothesis to test**, not yet measured: run `test_inference.py --nms` (one-to-many head + NMS) on the crowded val images and compare recall.
- **High precision (≈ 0.88–0.92) and low recall is the expected signature of under-confidence.**
  - What fires is right. What doesn't fire scores below threshold: the recall curve falls from 0.82 at conf 0 to 0.52 at 0.39.
  - Background clutter adds to it: red signage, hoses and cylinders compete for the same colour cue.
- **On CCTV, true small scale is the second factor.** The earlier degradation sweep (`analysis\extinguisher_summary.json`) showed:
  - Confidence holds at 0.8–0.95 for an extinguisher 80–160 px tall.
  - It falls to 0.5 at 60 px and to about 0.1 at 40 px or below.
  - In greyscale it is about 0 at every size.
  - The ceiling-CCTV extinguishers measured 41–56 px tall in the screenshot, about 50–72 px in the 480-line stream. That is exactly the cliff.
  - Upscaling the input to 1280 doesn't add detail that a 480-line camera never captured.

**Mitigations:**
1. Label every extinguisher in crowded training images. Audit for unlabelled neighbours: they actively teach suppression (see Q10).
2. Add CCTV-scale examples: 30–80 px tall, top-down, partly hidden, in colour **and** greyscale/IR.
3. Use per-class conf 0.30 for this class.
4. Test `--nms` for crowded scenes.
5. Never treat "no extinguisher detected" as a compliance alarm.

---

## 5. Mosaic augmentation vs large, close-up machinery

**What mosaic actually does here** (`augment.py`):
- Four images, each letterboxed to 1280, go onto a 2560×2560 canvas.
- The canvas is then affine-cropped back to 1280 (`border = -imgsz/2`) with `scale=0.9`, meaning a random scale in **[0.1, 1.9]**, plus `translate=0.2` and `degrees=10`.
- **Net effect:** each object appears at roughly **0.05×–0.95×** its single-image linear size, about 0.5× on average.
- For segmentation, an instance is **kept even if only 1% of its area survives the crop** (`augment.py:1232`: `area_thr=0.01 if len(segments)`).

**Does it hurt big-machine learning?** Partly.

**Where it helps:**
- CCTV is the target domain, and machines there are **smaller than in the training photos**.
- Mosaic plus downscaling is the main thing teaching the model what a machine looks like at 5–30% of the frame.
- Turning it off entirely would make the CCTV domain gap worse.

**Where it hurts:**
1. **Truncated slivers keep full labels.** A long panel saw or edge bander cut by a tile edge keeps its class label with as little as 1% of its body visible. This teaches part-of-machine equals the class. That is a plausible contributor to partial-machine masks ("only the table surface") and to inconsistent extents.
2. **Very small rescales of large machines.** At s ≈ 0.1, a machine that filled the frame becomes about 60 px across, with no context. Those samples are mostly noise.
3. **Train/val mismatch for 40 of 50 epochs.** The validation set is large, centred single objects. **Evidence: val mAP sat flat at 78.6% for epochs 21–40, then rose 1.5 pt the moment mosaic closed.**

**When to reduce mosaic, or close it earlier:**
- The deployment view is close-up and frame-filling (not this project's CCTV case).
- The dataset is dominated by long or large objects whose crops become meaningless slivers.
- Fine-tuning from a converged checkpoint (as here). Use `close_mosaic=15–20` so more of the short budget is spent on whole objects.
- Masks show part-object truncation. Reduce `scale` before touching `mosaic`.

**Recommended for the next run:**
- Keep `mosaic=1.0`; it's needed for CCTV scale.
- Lower `scale` from 0.9 to **0.5**. The range [0.5, 1.5] avoids the extreme 0.05× samples.
- Set `close_mosaic=15`.
- Add real CCTV frames so scale variety comes from data, not just augmentation.
- If truncation shows up in masks, set `copy_paste=0.3` (`copy_paste_mode=flip`). It adds **whole** instances to new backgrounds instead of cutting existing ones.

---

## 6. Why `train/l1_loss` jumped at epoch 40

**Correcting the premise:**
- `close_mosaic` does **not** switch on an L1 loss in Ultralytics. That is YOLOX behaviour.
- Here `l1_loss` is present for the **whole run**: 0.0208 at epoch 1.
- It is the regression term of the `reg_max=1` YOLO26 box head, which replaces DFL. `loss.py:142-153` computes it as the L1 distance between predicted and target left/top/right/bottom distances, **normalised by image size**. It is weighted by `dfl=1.5` (`loss.py:459`).

**When the jump happened:**
- `trainer.py:454` closes mosaic when the 0-indexed epoch equals `epochs − close_mosaic` = 40.
- That is **CSV epoch 41**: `l1_loss` 0.0121 → 0.0212 (+75%).

**Why it jumped, step by step:**
1. **L1 on image-normalised distances is scale-dependent.**
   - With mosaic, objects are about 0.5× size (Q5), so box edges sit a small fraction of the image from the anchor point. A given relative error is a small absolute number.
   - Without mosaic, each sample is one letterboxed photo. A typical machine fills 40–90% of the frame (median edge-banding box is 78% of the image), so distances are 2–5× larger.
   - The same relative error therefore produces a proportionally larger L1.
2. **The scale-invariant box loss barely moved.** CIoU `box_loss` went 0.634 → 0.670 (+6%). If localisation had actually got worse, CIoU would have jumped too.
3. **The other losses moved the same way.**
   - Easier samples: `cls_loss` 0.779 → 0.517, `seg_loss` 1.021 → 0.910, `sem_loss` 0.480 → 0.184. All are single-object images with fewer boundaries.
   - Then `l1_loss` fell steadily again: 0.0212 → 0.0165 by epoch 50, while val box loss reached its run minimum (0.824).

**Verdict:** a measurement artifact of switching input distribution. It is **not a regression** and **not a new loss being enabled**.

---

## 7. Why `val/seg_loss` plateaued while `train/seg_loss` kept falling

**The numbers:**

| | Epoch 1 | Epoch 50 | Notes |
|---|---:|---:|---|
| train seg_loss | 1.71 | 0.78 | −55% |
| val seg_loss | 1.345 | 1.438 | best 1.345 at epoch 1; range 1.34–1.65 |
| val mask mAP50-95 | 57.1% | 65.4% | +8.3 pt (peak 66.0% at epoch 38) |

**Diagnosis.** Ranked by evidence, it is a mix of three effects, and not a failing mask head.

1. **Boundary and label ambiguity, so validation loss has a floor (primary).**
   - The mask loss is per-pixel BCE inside the GT box, normalised by box area (`loss.py:583-585`).
   - Validation polygons have inconsistent extents and SAM-style boundaries: 45% of edge-banding labels are frame-filling; fences and extension tables are included in some labels and not others.
   - Those pixels carry an irreducible loss.
   - The training set is **1.44× oversampled** (3,312 entries for 2,308 unique images), so the model sees the same polygons repeatedly and can memorise their quirks. That pushes train loss down without generalising.
2. **Calibration drift (secondary).**
   - As the model grows more confident, its errors on ambiguous pixels cost more BCE, even while more instances pass the IoU thresholds.
   - Rising loss alongside rising mAP is the textbook signature of this. The metric that matters (mask mAP50-95, +8.3 pt) improved.
3. **The training objective shifts away from the masks (structural, YOLO26-specific).**
   - The logged `seg_loss` is the one-to-one branch's (`loss.py:1315`), and that branch sees a **detached** prototype (`head.py:428-430`).
   - The prototypes are trained only through the one-to-many branch, whose weight decays from 0.8 to 0.1 (`loss.py:1303-1325`).
   - Late in training the prototypes get about 8× less gradient than at the start, so the val mask loss has little reason to improve.

**Resolution limits are not the bottleneck.**
- The prototype grid is 320×320 at `imgsz=1280` (`mask_ratio=4`), so one prototype pixel covers a 4×4 block of input pixels.
- For machines 300–1,200 px across, the boundary band is a small share of the box.
- It matters for thin rails and fences (§5.3 of `runsummary.md`), not for the loss trend.

**Mitigations:**
1. Fix the labels: one extent policy per class; re-draw frame-filling and partial polygons; deduplicate oversampled images or oversample with augmentation.
2. Select checkpoints on **mask mAP50-95**, not loss: epoch 38 had the best masks.
3. Try `copy_paste=0.3` for mask diversity, and `scale=0.5` (Q5).
4. Use `retina_masks=True` at inference; `test_inference.py` does by default.
5. Use `mask_ratio=2` (640² prototypes) only if thin structures matter. It roughly quadruples mask-loss memory at 1280.

---

## 8. Why `val/sem_loss` is always 0

**It is a real auxiliary loss, and it is computed only in training by design.**
- YOLO26-seg's `Proto26` contains a semantic-segmentation branch, `self.semseg` (`block.py:1996`): two 3×3 convs and a 1×1 conv to `nc` channels.
- It returns semantic logits **only when `self.training` is true** (`block.py:2007-2009`).
- In `v8SegmentationLoss`, `sem_loss` is BCE + Dice between those logits and per-class masks built from the instance masks (`loss.py:536-552`). It is weighted by the `box` gain (7.5).

**Why the validation value is 0:**
- During validation the model runs in eval mode, so `Proto26` returns no semantic output.
- `pred_semantic` is then `None` and `loss[4]` keeps its initial 0 (`loss.py:506, 536`).

**Its purpose:**
- It is a dense, class-aware training signal that helps the shared prototype features separate machine pixels from background.
- `fuse()` deletes it for inference and export (`block.py:2012-2014`), so it costs nothing at deployment.

**Reading the training column:** `train/sem_loss` fell 1.07 → 0.13. The step at epoch 41 (0.48 → 0.18) is the mosaic switch-off from Q6: single-object images give far fewer semantic boundaries.

**Action:** none. Ignore the `val/sem_loss` column; it can't be computed in eval mode.

---

## 9. Proposal: 50% cluttered machines / 50% clean negatives

**Verdict:**
- **Clutter on the positive side: yes.**
- **50% negative images: no.** Keep backgrounds at **10–15% of training images**, made of **hard** negatives, and measure false positives on a separate negative set.

**Where the run stands now:**
- 301 negatives = 9.1% of training entries (13.0% of unique images). That is already in the normal band.
- Ultralytics' guidance is 0–10%; COCO is about 1%.
- The problem is **which** negatives: all machinery, no desks, ceilings or bars. It isn't how many.

**Why 50% would hurt recall:**

1. **Negatives are already over 99.9% of the training signal.**
   - At 1280 the head scores about 33,600 anchor points per image (160² + 80² + 40²).
   - A positive image has tens of positive anchors: top-k = 10 per ground-truth object for the one-to-many head, 1 for the one-to-one head. Every other anchor is a negative for all 9 classes.
   - Background images add only more negatives. They supply no hard examples unless they contain look-alikes.
2. **They shift the classification prior down.**
   - The BCE loss is summed over every anchor in the batch and divided by the batch's sum of positive target scores (`loss.py:435-441`).
   - A background image adds negative-only loss to the numerator and nothing to the denominator. As positives per batch fall, the denominator shrinks, so each step's push toward "background" on every class grows.
   - Confidences drop across the board, so recall at a fixed threshold drops: under-detection.
   - This run already shows a milder version: a real panel saw that scored 0.56–0.63 now scores ≤0.18.
3. **They halve positive exposure.**
   - With `batch=4`, an average of 2 positive images per step instead of about 3.6. With mosaic, about half the tiles in each canvas are empty.
   - The model sees half as many object examples per epoch. Rare classes like edge banders (263 training instances) suffer most.
4. **A 50/50 eval split misstates deployment.**
   - mAP is computed only where ground truth exists. Adding negatives to val/test changes only the false-positive count, pooled and unattributable (Q2).
   - Eval mixes should mirror deployment frequency per slice, or be split by purpose (sets B/C in Q2).

**What to do instead:**

| Split | Composition |
|---|---|
| **Train** | ~85–90% positives, as many as possible **in cluttered, in-context, CCTV-like scenes**. ~10–15% hard negatives **mined from your deployment footage**: run the current model on empty-scene video at conf 0.2 and add the frames it fires on (desks, ceilings, bars, monitors). Never add random "clean" backgrounds; they teach nothing. |
| **Val / test** | Keep ~10–15% negatives for regression, **plus** a separate hard-negative set (Q2 set C) that reports false-positive rate per image or hour. |
| **Optional** | If false positives persist, add a small weighted loss on negative images or tune per-class thresholds. Prefer these over ratio changes. |

---

## 10. Unlabelled machines in background or workshop images

There are two very different cases, and the run's data contains both.

**Case A: non-target machines (spindle moulder, CNC router, lathe, hoist…) left unlabelled.**
- **Effect:** intended and useful. Each anchor on them gets a classification target of 0 for all 9 classes (BCE with `target_scores = 0`). The model learns "industrial-looking but not one of ours". These are the hard negatives that build discrimination.
- **Risk:** look-alikes.
  - A spindle moulder or shaper looks like a table saw. A beam saw or sliding-table saw *is* a kind of panel saw. A veneer press looks like a hydraulic press.
  - If a negative image contains something visually inside a target class, its gradient **directly contradicts** the positive examples.
  - The head resolves the conflict by lowering confidence for the whole class.
  - **This is the most likely reason for the panel-saw regression on the workshop video.** The training negatives include 8 spindle moulders, 7 veneer presses, 15 CNC routers and 6 storage racks: long, flat, beam-and-table machinery close to the sliding panel saw in the trial clip.

**Case B: target-class machines present but unlabelled (missing labels).**
- **Effect:** actively harmful, in three ways.
  1. **Training:** a real panel saw with no label is a negative. Each correct prediction on it is **penalised as a false positive**. The model learns to suppress true detections, which lowers recall and confidence.
  2. **Semantic branch:** its pixels become background in `sem_masks` (`loss.py:537-551`), teaching the prototype features the same wrong thing.
  3. **Validation:** a correct detection on an unlabelled target counts as a false positive and deflates precision. Some of the 67 "background" false positives in the confusion matrix are this. Example: the six-saw collage in `val_batch0` has 1 label and 6 saws.
- **Crowded images are where this concentrates.** Unlabelled neighbours in multi-extinguisher images are a direct cause of the 39% recall on crowded scenes (Q4).

**How to audit and fix:**

1. **Run the current model over every negative and training image** at conf 0.2:
   `python test_inference.py --source <images dir> --conf 0.20 --save-dir audit_negatives`
   Then review every hit in `detections.csv`:
   - **A true target, unlabelled** → add the label (Case B).
   - **A look-alike non-target** → keep as a hard negative only if the class definition clearly excludes it. Otherwise drop it or relabel.
   - **A genuine false positive** (desk, bar) → keep. These are the valuable hard negatives.
2. **Write a class-boundary policy** for look-alikes, with images. For example: "sliding-table saw = panel_saw; spindle moulder = negative".
3. **If look-alikes are common in the factory,** add an explicit `other_machine` class rather than relying on unlabelled negatives. A labelled distractor class gives a clean gradient instead of a contradictory one.
