"""
Collect targeted hard-negative candidates (look-alike false-positive triggers) from DuckDuckGo images
====================================================================================================
Candidates land in datasets/hard_negatives/candidates/<query_slug>_NNN.jpg. They are NOT training data until
they are reviewed: an image may only be accepted if it contains none of the 9 classes (in particular no
fire extinguisher). Accepted images are moved to datasets/hard_negatives/images/ by hand or with
--accept-list; build_clean_9class_dataset.py writes their empty label files.

Gates applied here:
- min size 320 px, JPEG re-encode (PIL),
- dHash (64-bit) near-duplicate rejection, distance <= 6, against all other candidates AND all catalog
  images in datasets/source_38class/*/images,
- URL de-duplication within the run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "datasets" / "hard_negatives"

QUERIES = {
    "yellow_drum": ["yellow plastic chemical drum", "yellow plastic barrel factory floor", "white plastic drum workshop"],
    "blue_drum": ["blue plastic drum barrel warehouse", "blue HDPE drum industrial"],
    "water_jug": ["20 litre water jug", "water dispenser bottle office", "large plastic water can"],
    "red_bin": ["red plastic dustbin standing", "red waste bin factory", "red bucket on floor workshop"],
    "hose_reel": ["fire hose reel wall mounted red", "hose reel cabinet corridor"],
    "alarm_box": ["red fire alarm call point", "fire alarm bell red wall"],
    "cardboard_tube": ["cardboard tubes stacked", "large cardboard core tubes"],
    "gas_cylinder": ["gas cylinders chained wall workshop", "oxygen cylinder trolley workshop"],
    "ceiling": ["factory ceiling steel rafters", "workshop roof trusses interior"],
    "floor": ["empty concrete factory floor", "workshop floor sawdust"],
    "workbench": ["woodworking workbench with tools", "metal workbench vice garage"],
}


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-query", type=int, default=10)
    ap.add_argument("--max-dist", type=int, default=6)
    args = ap.parse_args()
    from ddgs import DDGS

    cand = OUT / "candidates"
    cand.mkdir(parents=True, exist_ok=True)
    print("hashing catalog images ...", flush=True)
    known = []
    for d in (PROJECT_ROOT / "datasets" / "source_38class").glob("*/images"):
        for p in d.iterdir():
            try:
                known.append(dhash(Image.open(p)))
            except Exception:
                pass
    for p in cand.glob("*.jpg"):
        known.append(dhash(Image.open(p)))
    known_arr = np.array(known, dtype=np.uint64)
    seen_urls, kept, rejected_dup = set(), 0, 0

    def is_dup(h: int) -> bool:
        if not len(known_arr):
            return False
        x = np.bitwise_xor(known_arr, np.uint64(h))
        return bool((np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1) <= args.max_dist).any())

    for slug, queries in QUERIES.items():
        n = 0
        for q in queries:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(q, max_results=40))
            except Exception as e:
                print(f"  [{slug}] search failed for {q!r}: {e}")
                time.sleep(5)
                continue
            for r in results:
                if n >= args.per_query:
                    break
                url = r.get("image") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception:
                    continue
                if min(img.size) < 320:
                    continue
                h = dhash(img)
                if is_dup(h):
                    rejected_dup += 1
                    continue
                known_arr = np.append(known_arr, np.uint64(h))
                name = f"{slug}_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
                img.save(cand / name, "JPEG", quality=92)
                n += 1
                kept += 1
            time.sleep(2)
        print(f"  [{slug}] {n} candidates", flush=True)
    print(f"{kept} candidates saved to {cand} ({rejected_dup} near-duplicates rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
