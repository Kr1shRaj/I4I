# Post-mortem: CCTV-perspective web scrape (2026-10-14)

*Written 2026-10-15. Batch: `pipeline\dataset\train\<class>\img_<class>_cctv_*.jpg` (2,677 files).*

## Verdict

The batch does not contain CCTV footage. Do not label it as CCTV data.

| Measure | Result |
|---|---|
| Files that are byte-identical copies of another file in the batch | **723 / 2,677 (27%)** |
| ...of which the same URL downloaded twice (scraper bug) | 537 |
| Files identical across **different classes** (e.g. same image in `jointer` and `planer`) | 347 files in 116 groups |
| Near-duplicates of images already in `datasets\9class` (dHash ≤ 4 and CLIP cos ≥ 0.93) | 311 (12%) |
| CLIP zero-shot: closest to "CCTV footage still" | 49 (1.8%) |
| ...that are actually CCTV-style frames after visual review | ~5, all warehouses; none clearly shows one of the 9 machines |
| CLIP zero-shot: closest to "camera device" or "camera ad" | ~279 (~10%) |
| CLIP zero-shot: catalog photo or eye-level close-up | ~1,700 (~63%) |

"Similar to the existing dataset" is mostly about **genre**, not exact copies. The mean CLIP nearest-neighbour similarity to the existing set is 0.78. Existing images score 0.835 against each other. The batch is the same kind of photo (eye-level, well lit, catalog, stock, often AI-generated), not literally the same files. `edge_banding_machine` is the exception: it scores 0.849, with 63 near-copies.

## Why it happened

### 1. The web has almost no indexed CCTV stills of these machines (main cause)
- Real CCTV footage of a panel saw sits on private NVRs, or inside videos such as YouTube incident clips. It is almost never published as a captioned image.
- Image search (DDGS → Bing) ranks by **page text**, not pixels. When no page says "CCTV still of a panel saw", the engine falls back to whichever query word matches best.

### 2. The query words pulled in the wrong things
- **"surveillance camera", "security camera", "CCTV"** matched pages *about cameras*. The batch has camera products, NVR monitor ads, security guards, cyber-security shield and padlock graphics, and one stock photo of a camera over a factory floor. That photo was saved 8+ times, across several classes.
- **"fire extinguisher factory" / "...manufacturing plant"** returned extinguisher *manufacturing* plants and "project report" banners.
- **"high angle", "wide angle", "from above"** are photographers' terms. Captions rarely contain them, so they did little. "From above" mostly returned top-down product shots.
- The machine name is the strongest token in every query. The remaining hits are the same popular catalog and stock photos the default scrape already found.

### 3. Only the top ~35 results per query were used
- The scraper asks for up to 90 results, but DDGS returned 35 (one page) for almost every query (`stage2_scrape_log_cctv.json`).
- Ten queries × 35 results is the cap. Top-ranked results are the most popular images: stock and catalog photos.

### 4. Almost nothing was rejected
- About 33 of every 35 results were downloaded. The only gate is a substring blocklist on title and URL.
- CCTV mode **removed** "cctv" and "surveillance" from the blocklist (`stage2_scrape.py:90`). Product titles such as "Dahua WiFi 4MP" match none of the blocked phrases.
- The thumbnail fallback URL (a Bing proxy link) carries no keywords, so the URL check never sees anything there.
- There is no visual (CLIP) check at download time. That check only exists in stage 4, which has not been run yet.

### 5. Deduplication bug in the scraper
- `_url_to_filename()` puts the running index in the filename: `img_<cls>_cctv_<idx>_<md5>.jpg`.
- The `dest_path.exists()` check in `stage2_scrape.py:267` therefore never matches. The same URL returned by two queries is downloaded twice, which accounts for **537** files.
- Mirrors of one image under different URLs add more copies, for **723** byte-identical repeats in total.
- Classes are scraped independently. The same image returned for "woodworking planer..." and "woodworking jointer..." was saved under **both labels** (347 files). Labelling those as-is would teach the model contradictory classes.

## What to do with this batch
1. Treat it as generic variety at best. Run stage 3 (dedup) first, and include a cross-class duplicate check: drop any image that appears under two classes.
2. Run stage 4 with negative prompts for camera hardware, security, diagrams and infographics. Expect roughly 1,000–1,300 usable images, nearly all catalog or eye-level.
3. Expect little CCTV benefit from it. It does not replace step 1 of `cctv_domain_adaptation_plan.md` (raw RTSP recording from your own cameras plus pseudo-CCTV phone video).

## Better sources of CCTV-like data
- **Own cameras** (plan step 1). This is the only source that matches the deployment domain.
- **Video, not image search:** factory-tour and shop-walkthrough videos filmed from a mezzanine or a wall mount. Sample 1 frame every 2–5 s and dedupe. Check the licence before using any frames.
- **Synthetic CCTV look** on existing labelled images: downscale to 480×360, JPEG/H.264 artefacts, colour tint, greyscale/IR, a timestamp overlay, and a high-angle perspective warp. Labels come for free. This needs `albumentations` (todo §6).

## If the scraper is used again
- Dedup on the URL digest (a set of seen URLs per run) **and** on a perceptual hash across all classes, before saving.
- Add a CLIP gate at download: keep an image only if the target class beats the "camera device / security / diagram" prompts.
- Drop the "surveillance camera" and "security camera" queries. Use place-based queries instead, for example "{machine} furniture factory production line" or "{machine} mezzanine view".

*Analysis script, per-image CSV and contact sheets: `pipeline\logs\cctv_scrape_analysis\` (`analyze_cctv.py`, `report.json`, `per_image.csv`, `sheet_<class>.jpg`, `sheet_cctv_candidates.jpg`).*
