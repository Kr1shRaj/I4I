# 9-class dataset layout: why `images/` holds far more files than `labels/`

Investigated 2026-10-21. Ultralytics 8.4.123. All numbers measured on the current build.

**Short answer:** the extra images are inert. The text lists define the dataset, and the junction setup should stay. Below are the evidence and the one real risk.

## 1. The folder discrepancy

| Split | Files in `images/` | `.txt` in `labels/` | of which empty (negatives) | Entries in list |
|---|---:|---:|---:|---:|
| train | 8,119 | 2,200 | 283 | 3,298 (2,337 unique, incl. 107 hard negatives + 30 high-angle outside this folder) |
| valid | 1,718 | 450 | 58 | 450 |
| test | 1,747 | 473 | 62 | 473 |

*Updated after the user curation (`docs/label_policy.md`). The rest of this report was measured on the build before it: 2,252 / 461 / 493 labels and 3,373 train entries, and the 5,867 figure below. The argument is unchanged; 52 more train images are now unlisted (user quarantine).*

**The junction.** `build_clean_9class_dataset.py` (`ensure_junction`) makes each `datasets\9class\<split>_9class\images` an NTFS junction to `datasets\source_38class\<split>\images`.
- A junction is a directory alias, not a copy. Listing `train_9class\images` lists the full 38-class image folder: 8,119 files, 384 MB, stored once on disk.
- `labels\` is a real folder. The build deletes and rewrites it on every run, and writes a label file only for images it selects: 1,969 positives and 283 negatives.

**Where the 5,867 unselected train images come from** (8,119 − 2,252):

| Group | Images |
|---|---:|
| Non-target candidates not drawn into the 15% negative sample | 5,519 |
| Negatives dropped after today's screening | 12 |
| Exit-sign / first-aid / PPE sources, never used as negatives because they often show extinguishers | 251 |
| CAD / FE-failure quarantine | 47 |
| Label-fix quarantine | 38 |

**How Ultralytics loads the data.** It never scans `images\` when the data yaml points at a `.txt` list. In `ultralytics/data/base.py`, `BaseDataset.get_img_files`:
- **Directory:** if the path is a directory, it uses `glob(dir/**/*.*)`. Only this branch scans the disk.
- **File:** if it is a file, it reads the lines and replaces a leading `./` with the list file's folder, by plain string substitution with no `resolve()`.
- **Then:** it keeps image extensions and sorts. Duplicates are kept, which is how oversampling works.
- **Labels:** each label path is derived from the image path *string* by `img2label_paths` (last `\images\` → `\labels\`). Because the path is never resolved through the junction, `9class\train_9class\images\x.jpg` maps to `9class\train_9class\labels\x.txt`, never to the 38-class label in `source_38class\train\labels`.

**Measured** by building `YOLODataset` from `data_9class.yaml`:
- `im_files` = 3,373, of which 2,389 unique: exactly the list.
- 0 label paths point into `source_38class`.
- Label roots: 3,236 `train_9class`, 107 `hard_negatives`, 30 `highangle`.
- The maximum class id is 8, and there are 390 empty-label entries.

**One harmless quirk:** Ultralytics names the cache after the first label file's folder. The list is sorted and `..\hard_negatives` sorts first, so the train cache is written to `datasets\hard_negatives\labels.cache`. The cache holds a hash of every file, so a stale one is rebuilt automatically.

## 2. Effect on training

### The 5,867 unlisted images

They have no effect at all: no VRAM, no loss terms, no gradient steps, no influence on mAP. They are never opened. The dataloader's length, sampler and caches are built only from `im_files`. The only cost is disk space, and that is in `source_38class` anyway.

### The 283 listed negatives (other machinery, empty labels)

**What they do in training:**
- An empty label makes every prediction on the image a negative.
- The classification BCE pushes all 9 class scores down there. Box, DFL and mask losses get no positives from them.
- They teach the model that *industrial machine-looking* is not enough: the specific features of the 9 classes must be present.
- They are 8.4% of the train list (11.6% with the 107 targeted hard negatives). They are added once and not oversampled.

**Empirical evidence:**
- In today's screening, 129 of the 403 kept negatives (all splits) still get a target detection ≥ 0.25 from run5 or run6. The confusion is real, so these images carry signal.
- We have no ablation (training with versus without them), so their exact contribution to mAP is unmeasured.
- What we *have* measured is the failure mode when negatives are wrong. run6's look-alike negatives cut its panel-saw confidence on the workshop clip from 0.56–0.63 to ≤ 0.18.

**Can they suppress true targets?** Only in two cases:
1. **A target machine is in the negative.** Then the model is explicitly trained to call that machine background. *Found today:* 5 images.
2. **The negative's class overlaps a target's definition** (a class boundary, not a look-alike):
   - **Veneer / hot press vs `hydraulic_press`:** the Camera 03 press is this type. *Found today:* 12 images.
   - **`drum_sander` / wide-belt sanders vs `planer`:** boxy feed-through machines (8 kept).
   - **`spindle_moulder` and four-sided moulders vs `planer` / `table_saw`:** 14 kept.

   These are *correct* negatives under our class definitions, a moulder is not a planer. But they sharpen exactly the boundaries that matter, so the labels must stay consistent. Any image whose class is debatable should leave the negative list, as the veneer presses did.

**Dissimilar machinery does not suppress targets:** lathes, forklifts, pallet jacks, cranes, control panels. Suppression only acts where the features match the negative, and a lathe does not share a planer's feed table and cutter-head housing. These images reduce false positives without costing recall.

### Today's screening outcome

**Method:** `scripts\tools\audit_negatives.py`. All 420 negatives were run through run5 and run6 at conf 0.10, then every contact sheet was reviewed, with zooms on busy scenes.

**Dropped (decisions in `label_decisions.yaml` → `negatives`):**
- **5 with unlabelled target machines:**
  - benchtop planer and mitre saw (`dust_collector_00198`);
  - panel saw at the edge (`glue_spreader_00455`);
  - background band saws (`mortiser_00244`, `wood_lathe_00220`);
  - an illustration with four target classes (`dust_collector_00464`).
- **All 12 `veneer_press` negatives** (class-boundary conflict).

**Result:** negatives went from 295/60/65 to 283/58/62. The drops are applied after sampling, so no new, unreviewed negatives came in. run5 on the current val is unchanged: box 0.789, mask 0.741.

## 3. Should the folders hold only the 9-class images?

| | Keep junctions (current) | Copy only active images into standalone folders |
|---|---|---|
| Training speed | Same: only the list is read | Same |
| mAP / loss | Same: unlisted images are never seen | Same |
| Disk | 0 extra | +99 MB per copy of the 2,252 active train images (more with val/test); a second copy that can drift from the source |
| Rebuilds | Labels rewritten, images untouched | The copy must be redone, and files removed, on every rebuild, or stale images build up |
| Human confusion | Real: 8,119 vs 2,252 looks like a bug | Gone: `images` and `labels` match |
| Script risk | A tool pointed at `train_9class/images` as a *directory* would load all 8,119 images, 5,867 of them unlabelled, as backgrounds. Deleting "unused" files in that folder **deletes the originals in `source_38class`** | A directory path would be safe |

**Physically removing the 5,867 files** would change nothing in speed or mAP. Through the junction it would destroy part of the 38-class dataset. Never delete inside `*_9class\images`. Never `Remove-Item -Recurse` the junction itself either; use `cmd /c rmdir`.

### Recommendation: keep the zero-copy junctions, and treat the lists as the definition of the dataset

1. **Only point training and evaluation at the yaml or `.txt` lists,** never at `*_9class/images` as a directory. `data_9class.yaml` already does this.
2. **Treat this document as the folder's README:** the count gap is expected.
3. **To look at the active dataset,** use the gallery. It is built from the lists, so it shows exactly what training sees. Re-run it after every rebuild:
   - build: `env\yolo-env\Scripts\python.exe scripts\tools\build_dataset_viewer.py`
   - serve: `python -m http.server 8000 --directory .`
   - open: `http://localhost:8000/docs/viewer/index.html`
4. **Decouple only if the dataset has to leave this machine** (archive, another PC, cloud training). Junctions do not survive zip/copy cleanly. At that point, export the list-referenced images and labels into a self-contained folder as a one-off step, rather than changing the working layout.
