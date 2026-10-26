# Label policy and the 2026-10-21 label cleanup

## Edge banding machine (`edge_banding_machine`, class 7)

**Extent: one polygon per physical machine.**

| Include | Exclude |
|---|---|
| Machine chassis and base cabinet | Panels being fed through or stacked on the conveyors |
| Top cover / hood (open or closed) | Floor, walls, ceiling, the surrounding room |
| Gluing, end-cutting, trimming, scraping and buffing units | Text banners, logos, spec tables, feature strips |
| Infeed and outfeed roller conveyors / support tables attached to the machine | Inset or zoom-circle photos, collage tiles |
| Edge-tape coil holder (disc) and control panel arm | People, dust hoses that leave the machine |

**Other rules:**
- **Two machines in one image are two instances.** Never draw one polygon around both.
- **Partly visible machines:** label only if at least 40% is visible, and trace only the visible pixels.
- **Close-ups:** a polygon covering more than 80% of the image is allowed only when the machine really fills the frame. If walls, floor or a white background are inside the polygon, it is wrong.

## Fire extinguisher (`fire_extinguisher`, class 8)

**Rules:**
- **One polygon per physical cylinder,** including its own hose, nozzle, handle and wall bracket strap.
- **Touching or overlapping extinguishers are separate instances.** Never merge a row into one blob.
- **Label every extinguisher in the image,** including partly hidden ones (≥ 40% visible) and those in the background.
- **Not extinguishers:** gas cylinders, plastic drums, water jugs, red bins, hose reels and alarm call points. These are hard negatives (`datasets\hard_negatives\`).

## What was changed on 2026-10-21

**Build:** `scripts\tools\build_clean_9class_dataset.py`. Review decisions are in `datasets\9class\label_fixes\label_decisions.yaml`. A backup of the previous labels and lists is in `archive\dataset_backups\9class_labels_pre_run8_2026-10-21\`.

**The morphological "merge" step is turned off** for both classes. It closed gaps of 0.5–1.2% of the image diagonal:
- touching extinguishers fused into one instance (e.g. 348 source fragments → 1 blob covering 3 cylinders);
- edge-bander fragments ballooned into slabs.

**Its replacement is `scripts\tools\label_instances.py`:**
- consolidates speckle-sized fragments;
- absorbs small touching parts (hoses, handles) into their cylinder;
- keeps similar-sized neighbours separate.

**Edge banders:**
- **Flagged:** 175 polygons with a box over 80% of the image, across all splits.
- **Proposals:** SAM 2.1 (`scripts\tools\fix_edge_bander_labels.py`) proposed a tighter mask for each. If SAM selected the plain background, the mask is inverted inside the prompt box.
- **Review:** every proposal was checked visually on overlay sheets **by Claude, not by a human annotator**.
  - **68 accepted** (tight machine outline on a studio or plain background).
  - **47 kept** (genuine close-ups).
  - **60 images quarantined** (collages, infographics, inset photos, several machines). These leave the build until someone re-draws them by hand.

**Fire extinguishers:**
- **Instance rebuild:** applied to every image.
- **SAM per-cylinder masks:** used in 15 images where the source fragments were unusable. The masks come from box prompts I set on a grid.
- **Quarantine:** 4 images of dense, blurred cylinder rows.

| | Before | After |
|---|---:|---:|
| Train edge-bander boxes over 80% of the image | 48% | 20% |
| Val edge-bander boxes over 80% of the image | 59% | 18% |
| Fire-extinguisher instances (train / val / test) | 435 / 105 / 96 | 606 / 129 / 147 |
| Images quarantined for re-annotation | 0 | 64 (38 train, 16 val, 10 test) |

## Negatives, SAM 2.1 Large and high-angle images (later on 2026-10-21)

**Negative screening:** `scripts\tools\audit_negatives.py`
- **Method:**
  - All 420 empty-label images were run through run5 and run6 at conf 0.10 (228 had any detection, 133 at ≥0.25).
  - Every contact sheet in `datasets\9class\label_fixes\negative_audit\` was reviewed, plus full-resolution zooms of busy scenes.
  - Most detections are false positives on other machinery (drill presses scored as band saws, moulders as planers).
- **Dropped (`label_decisions.yaml` → `negatives`):**
  - **5 images showing an unlabelled target machine:**
    - a benchtop planer and mitre saw;
    - a panel saw at the frame edge;
    - two band saws in the background;
    - one illustration containing four target classes.
  - **All 12 `veneer_press` negatives:** the Camera 03 press is a woodworking hot/cold press, so these would teach the factory press to be background. *Open decision:* relabel them as `hydraulic_press` instead.
- **Effect:** the drops are applied after sampling, so the rest of the negative sample is unchanged. Negatives went from 295/60/65 to 283/58/62.

**SAM 2.1 Large** (`models\pretrained\sam2.1_l.pt`)
- **Source:** Meta's official `sam2.1_hiera_large.pt`, 856.5 MB, SHA-256 `2647878D…DD318`.
- **Comparison:** `scripts\tools\compare_sam_models.py`, box prompts on 222 val instances. Large beat Base on 25% of instances, and Base beat Large on 14%.

  | | sam2.1_b | sam2.1_l |
  |---|---:|---:|
  | Mean mask IoU vs labels | 0.880 | 0.886 |
  | Boundary F1 | 0.794 | 0.802 |
  | Jointer IoU | 0.928 | 0.971 |
  | Band saw IoU | 0.890 | 0.920 |
  | Mitre saw IoU | 0.914 | 0.877 |
  | Edge-bander IoU | 0.776 | 0.774 |
  | Speed (ms/prompt) | 108 | 271 |
  | Peak GPU memory (MB) | 776 | 1,423 |

- **Default:** all annotation scripts now use Large.
- **Caveat:** the gain is small, and on mitre saws it is negative, so masks still need review.

**High-angle images:** `scripts\tools\annotate_highangle.py`; data in `datasets\highangle\`; decisions in `decisions.yaml`.
- **Review:** all 99 gate-accepted images were reviewed. Classes follow image content: 11 accepted images contain classes other than the one searched for.
- **Accepted: 30 images, 52 instances.**

  | Class | Instances |
  |---|---:|
  | Mitre saw | 12 |
  | Hydraulic press | 9 |
  | Panel saw | 7 |
  | Table saw | 6 |
  | Edge bander | 6 |
  | Band saw | 5 |
  | Fire extinguisher | 4 |
  | Planer | 2 |
  | Jointer | 1 |

- **Rejected: 69 images.**

  | Reason | Images |
  |---|---:|
  | Dense / unlabelable background | 14 |
  | Ambiguous class or extent | 13 |
  | Close-up of a part | 12 |
  | No target machine | 10 |
  | Collage or infographic | 8 |
  | AI render | 6 |
  | Aerial | 4 |
  | SAM failure, needs hand drawing (#5, #94) | 2 |

- **Annotation:** reviewed boxes are used as SAM 2.1 Large prompts. Masks are clipped to the box, which excludes room and stock. Overlays are in `datasets\highangle\overlay\`.
- **Training use:** train only, listed in `datasets\9class\train_highangle_9class.txt`, added once (not oversampled). Rebuilt train list: 3,373 entries, negatives 11.6%.

## User curation from the `datasetchange` list (later on 2026-10-21)

**Source:** the `datasetchange` file at the project root, with two sections: images whose labels are wrong, and images that are bad in themselves.
- **Parsing:** `scripts\tools\parse_user_curation.py` writes `label_fixes\user_curation.json`. It strips whitespace and extensions, lowercases `Img_`, removes duplicates and finds each image's split.
- **Fix list:** 113 lines, all unique.
- **Remove list:** 87 lines, 80 unique (7 duplicates).
- **Matching:** every image was found and was in the current build. No image was on both lists.

**Bad images: `label_decisions.yaml` → `user_quarantine`** (80 images: 51 train, 11 val, 18 test).
- The build drops them from every list, and their 9-class label files are no longer written.
- The image files stay in `source_38class`.
- The negative sample is sized from the positive count before this quarantine, so the reviewed negatives are unchanged (283/58/62).

**Wrong labels: `label_decisions.yaml` → `user_fixes`.** The tool is `scripts\tools\fix_user_labels.py`.
- **`review`:** draws the current label and run5's detections on a 10% grid, in `label_fixes\review_user_fixes\review\`.
- **Reviewed boxes:** I read each sheet and set boxes `[class, x1, y1, x2, y2]` following the extent rules above. The review was **by Claude, not a human annotator**.
- **`propose`:** runs SAM 2.1 L on the boxes and clips each mask to its box, so room, floor and stock are cut out. It writes `label_fixes\user_fix_polygons.json`, which the build reads, and before/after sheets in `label_fixes\review_user_fixes\sheet_XX.jpg`.
- **Box modifiers:**
  - `rect` (12 boxes): a genuine close-up, where the box itself is the polygon.
  - `fg` (15 boxes): a product photo on a white background where SAM inverted or left gaps. The object is cut out from the non-white pixels inside the box.
  - `+class` (8 boxes): a second box merged into the previous instance, such as a control panel on a stand or the outfeed rails.
- **`exclude`** (3 images, which leave the build): an extrusion-line infographic and a roller-conveyor collage, neither showing an edge bander, and a table saw less than 40% visible (`table_saw_00209`).
- **Second pass:** 28 proposals were corrected after the first round of sheets (27 re-drawn and checked again, 1 excluded). The before/after sheets show the final masks.
- **Result:** 110 images now have 128 instances.

  | Class | Instances |
  |---|---:|
  | Edge bander | 62 |
  | Planer | 15 |
  | Table saw | 13 |
  | Jointer | 13 |
  | Mitre saw | 13 |
  | Band saw | 12 |

- **New instances:** some fixes add objects the old label missed, such as more saws in a row, a planer next to a jointer, and mitre saws on a bench.

**Effect** (backup of the previous build: `archive\dataset_backups\9class_labels_pre_user_curation_2026-10-21\`)

| | Before | After |
|---|---:|---:|
| Positives (train / val / test) | 1,969 / 403 / 431 | 1,917 / 392 / 411 |
| Train list entries | 3,373 | 3,298 |
| Edge-bander instances (train / val / test) | 229 / 33 / 55 | 134 / 22 / 25 |
| Fire-extinguisher instances (train / val / test) | 606 / 129 / 147 | 586 / 109 / 140 |

Train counts exclude the 30 high-angle images. Edge bander is now the smallest class, with only 22 val instances, so its val AP will be noisy. Compare future runs with run5 re-evaluated on this val, not the 0.789 / 0.741 measured on the previous one.

**To restore a quarantined image:** re-draw it by hand following this policy, put the corrected source label in `datasets\source_38class\<split>\labels\`, remove its stem or index from `label_decisions.yaml`, then re-run the build.
